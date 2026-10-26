//! Whisper in pure Rust with Hugging Face candle (candle-transformers): no C++ build.
//! Multilingual like whisper.cpp; transcribes the whole command at the end.
//!
//! Model directory (any of these layouts):
//! - openai/whisper-tiny style: config.json, tokenizer.json, model.safetensors (f32)
//! - lmz/candle-whisper style: config-tiny.json, tokenizer-tiny.json, model-tiny-q80.gguf

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config, audio};
use tokenizers::Tokenizer;

use crate::speech::words_only;

enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl Model {
    fn encode(&mut self, mel: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Model::Normal(w) => w.encoder.forward(mel, true),
            Model::Quantized(w) => w.encoder.forward(mel, true),
        }
    }

    fn decode(&mut self, tokens: &Tensor, audio: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Model::Normal(w) => w.decoder.forward(tokens, audio, flush),
            Model::Quantized(w) => w.decoder.forward(tokens, audio, flush),
        }
    }

    fn final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Model::Normal(w) => w.decoder.final_linear(x),
            Model::Quantized(w) => w.decoder.final_linear(x),
        }
    }

    fn reset(&mut self) {
        match self {
            Model::Normal(w) => w.reset_kv_cache(),
            Model::Quantized(w) => w.reset_kv_cache(),
        }
    }
}

pub struct CandleWhisper {
    model: Model,
    config: Config,
    tokenizer: Tokenizer,
    filters: Vec<f32>,
    /// <|startofprev|> hint <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>
    prompt: Vec<u32>,
    eot: u32,
    /// 0 or -inf per vocabulary entry.
    suppress: Tensor,
    device: Device,
    short_window: bool,
}

fn find(dir: &Path, pred: impl Fn(&str) -> bool) -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(&pred))
        .collect();
    hits.sort();
    hits.into_iter().next()
}

impl CandleWhisper {
    /// `short_window`: encode only the actual audio instead of the full 30 s window.
    /// `wake_word` is given to Whisper as preceding text so it transcribes it correctly.
    pub fn new(dir: &str, language: &str, short_window: bool, wake_word: &str) -> Result<Self> {
        let d = Path::new(dir);
        let config_path = find(d, |n| n.starts_with("config") && n.ends_with(".json"))
            .ok_or_else(|| anyhow!("no config*.json in {dir}"))?;
        let tokenizer_path = find(d, |n| n.starts_with("tokenizer") && n.ends_with(".json"))
            .ok_or_else(|| anyhow!("no tokenizer*.json in {dir}"))?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
            .with_context(|| format!("parsing {}", config_path.display()))?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{}: {e}", tokenizer_path.display()))?;
        let device = Device::Cpu;

        let model = if let Some(gguf) = find(d, |n| n.ends_with(".gguf")) {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&gguf, &device)?;
            Model::Quantized(m::quantized_model::Whisper::load(&vb, config.clone())?)
        } else if let Some(st) = find(d, |n| n.ends_with(".safetensors")) {
            // SAFETY: the model file is memory-mapped read-only and not modified while in use.
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[st], m::DTYPE, &device)? };
            Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
        } else {
            bail!("no *.gguf or *.safetensors model in {dir}");
        };

        let id = |t: &str| tokenizer.token_to_id(t).ok_or_else(|| anyhow!("token {t} missing in tokenizer"));
        // <|startofprev|> " Computer," <|startoftranscript|> <|lang|> ...
        let mut cap = wake_word.chars();
        let hint: String = cap.next().map(|f| f.to_uppercase().chain(cap).collect()).unwrap_or_default();
        let hint_ids = tokenizer
            .encode(format!(" {hint},"), false)
            .map_err(|e| anyhow!("encoding wake hint: {e}"))?
            .get_ids()
            .to_vec();
        let mut prompt = vec![id("<|startofprev|>")?];
        prompt.extend(hint_ids);
        prompt.extend([
            id(m::SOT_TOKEN)?,
            id(&format!("<|{language}|>"))?,
            id(m::TRANSCRIBE_TOKEN)?,
            id(m::NO_TIMESTAMPS_TOKEN)?,
        ]);
        let eot = id(m::EOT_TOKEN)?;
        let no_ts = id(m::NO_TIMESTAMPS_TOKEN)?;
        let suppress: Vec<f32> = (0..config.vocab_size as u32)
            .map(|i| {
                // also forbid timestamp tokens (all ids after <|notimestamps|>)
                if config.suppress_tokens.contains(&i) || i > no_ts { f32::NEG_INFINITY } else { 0.0 }
            })
            .collect();
        let suppress = Tensor::new(suppress.as_slice(), &device)?;
        let filters = mel_filters(config.num_mel_bins);
        Ok(Self { model, config, tokenizer, filters, prompt, eot, suppress, device, short_window })
    }

    pub fn transcribe(&mut self, samples: &[i16]) -> Result<String> {
        let pcm: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let mel = audio::pcm_to_mel(&self.config, &pcm, &self.filters);
        let n_mels = self.config.num_mel_bins;
        let mel = Tensor::from_vec(mel.clone(), (1, n_mels, mel.len() / n_mels), &self.device)?;
        // Whisper is trained on 30 s windows; the mel is zero-padded to at least that.
        // The short window keeps the real audio plus 1 s of silence (100 frames per second).
        let frames = if self.short_window {
            (samples.len() / m::HOP_LENGTH + 100).min(m::N_FRAMES)
        } else {
            m::N_FRAMES
        };
        let frames = frames.min(mel.dim(2)?);
        let mel = mel.narrow(2, 0, frames)?;

        self.model.reset();
        let audio_features = self.model.encode(&mel)?;
        let mut tokens = self.prompt.clone();
        let max_new = self.config.max_target_positions / 2 - tokens.len();
        for i in 0..max_new {
            let t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self.model.decode(&t, &audio_features, i == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self.model.final_linear(&ys.i((..1, seq_len - 1..))?)?.i(0)?.i(0)?;
            let logits = logits.broadcast_add(&self.suppress)?;
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            if next == self.eot {
                break;
            }
            tokens.push(next);
        }
        let text = self
            .tokenizer
            .decode(&tokens[self.prompt.len()..], true)
            .map_err(|e| anyhow!("decoding tokens: {e}"))?;
        Ok(words_only(&text))
    }
}

/// Whisper's mel filterbank (librosa.filters.mel, Slaney scale and norm), row-major
/// `n_mels x (N_FFT/2 + 1)`, as stored in OpenAI's mel_filters.npz.
fn mel_filters(n_mels: usize) -> Vec<f32> {
    let sr = m::SAMPLE_RATE as f64;
    let n_freqs = m::N_FFT / 2 + 1;
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    let hz_to_mel = |hz: f64| if hz < min_log_hz { hz / f_sp } else { min_log_mel + (hz / min_log_hz).ln() / logstep };
    let mel_to_hz = |mel: f64| if mel < min_log_mel { mel * f_sp } else { min_log_hz * (logstep * (mel - min_log_mel)).exp() };

    let fft_freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * sr / m::N_FFT as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_f: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64)).collect();

    let mut w = vec![0f32; n_mels * n_freqs];
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (j, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - mel_f[i]) / (mel_f[i + 1] - mel_f[i]);
            let upper = (mel_f[i + 2] - f) / (mel_f[i + 2] - mel_f[i + 1]);
            w[i * n_freqs + j] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    #[test]
    fn mel_filters_match_openai() {
        // Values from OpenAI's 80-mel filters (candle's melfilters.bytes); the whole
        // matrix was checked to match within 1e-8.
        let f = super::mel_filters(80);
        assert_eq!(f.len(), 80 * 201);
        for (i, want) in [(1, 0.024862595), (202, 0.0019908219), (203, 0.022871772)] {
            assert!((f[i] - want).abs() < 1e-7, "f[{i}] = {} != {want}", f[i]);
        }
        assert!(f.iter().all(|v| *v >= 0.0));
    }
}
