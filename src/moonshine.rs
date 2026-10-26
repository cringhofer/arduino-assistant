//! Moonshine (Useful Sensors) speech-to-text via sherpa-onnx: small, fast models made for
//! edge devices. Non-streaming: the whole command is transcribed once it has ended.
//! English only (no German model exists).

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use sherpa_onnx::{OfflineMoonshineModelConfig, OfflineRecognizer, OfflineRecognizerConfig};

use crate::audio::SAMPLE_RATE;
use crate::speech::words_only;

pub struct Moonshine {
    rec: OfflineRecognizer,
}

impl Moonshine {
    /// `dir`: an unpacked sherpa-onnx Moonshine model, v1 (preprocess/encode/
    /// uncached_decode/cached_decode) or v2 (encoder/merged decoder), plus tokens.txt.
    pub fn new(dir: &str, threads: i32) -> Result<Self> {
        let find = |names: &[&str]| -> Option<String> {
            names.iter().map(|n| Path::new(dir).join(n)).find(|p| p.exists()).map(|p| p.to_string_lossy().into_owned())
        };
        let tokens = find(&["tokens.txt"]).ok_or_else(|| anyhow!("no tokens.txt in {dir}"))?;
        let moonshine = if let Some(merged) = find(&["merged_decoder.int8.onnx", "merged_decoder.ort", "merged_decoder.onnx", "decoder_model_merged.onnx"]) {
            OfflineMoonshineModelConfig {
                encoder: find(&["encoder.int8.onnx", "encoder.ort", "encoder.onnx", "encoder_model.onnx"]),
                merged_decoder: Some(merged),
                ..Default::default()
            }
        } else {
            OfflineMoonshineModelConfig {
                preprocessor: find(&["preprocess.onnx"]),
                encoder: find(&["encode.int8.onnx", "encode.onnx"]),
                uncached_decoder: find(&["uncached_decode.int8.onnx", "uncached_decode.onnx"]),
                cached_decoder: find(&["cached_decode.int8.onnx", "cached_decode.onnx"]),
                ..Default::default()
            }
        };
        if moonshine.encoder.is_none() {
            bail!("no Moonshine encoder model found in {dir}");
        }
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.moonshine = moonshine;
        config.model_config.tokens = Some(tokens);
        config.model_config.num_threads = threads;
        config.model_config.provider = Some("cpu".into());
        let rec = OfflineRecognizer::create(&config).ok_or_else(|| anyhow!("cannot load Moonshine from {dir}"))?;
        Ok(Self { rec })
    }

    /// Transcribes 16 kHz mono audio to lowercase words without punctuation.
    pub fn transcribe(&self, samples: &[i16]) -> String {
        let pcm: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let stream = self.rec.create_stream();
        stream.accept_waveform(SAMPLE_RATE as i32, &pcm);
        self.rec.decode(&stream);
        let text = stream.get_result().map(|r| r.text).unwrap_or_default();
        words_only(&text)
    }
}
