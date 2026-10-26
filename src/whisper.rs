//! Whisper speech-to-text via whisper-rs (whisper.cpp). Multilingual, so unlike Moonshine
//! it also covers German. Non-streaming: transcribes the whole command at the end.

use anyhow::{Context, Result, anyhow};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::SAMPLE_RATE;
use crate::speech::words_only;

pub struct Whisper {
    ctx: WhisperContext,
    language: &'static str,
    threads: i32,
    short_window: bool,
    /// Text Whisper is told came before, so it spells the wake word right instead of
    /// "konkschöker" (German) and the command can be split off after it.
    hint: String,
}

impl Whisper {
    /// `model`: a whisper.cpp ggml model, e.g. ggml-base-q5_1.bin (multilingual, not ".en").
    /// `short_window`: encode only the actual audio instead of Whisper's fixed 30 s window.
    /// Several times faster for short commands, slightly less accurate.
    pub fn new(model: &str, language: &'static str, threads: i32, short_window: bool, wake_word: &str) -> Result<Self> {
        whisper_rs::install_logging_hooks(); // silence whisper.cpp's stderr chatter
        let ctx = WhisperContext::new_with_params(model, WhisperContextParameters::default())
            .map_err(|e| anyhow!("{e:?}"))
            .with_context(|| format!("loading Whisper model {model}"))?;
        Ok(Self { ctx, language, threads, short_window, hint: wake_hint(wake_word) })
    }

    pub fn transcribe(&self, samples: &[i16]) -> Result<String> {
        let mut pcm: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        // whisper.cpp refuses input shorter than 1 s
        pcm.resize(pcm.len().max(SAMPLE_RATE as usize * 11 / 10), 0.0);
        let mut state = self.ctx.create_state().map_err(|e| anyhow!("{e:?}"))?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(self.language));
        params.set_n_threads(self.threads);
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_no_timestamps(true);
        params.set_suppress_nst(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_initial_prompt(&self.hint);
        if self.short_window {
            // 1500 encoder positions = 30 s, i.e. 50 per second; plus 1 s of slack.
            let secs = pcm.len() as f32 / SAMPLE_RATE as f32;
            params.set_audio_ctx(((secs + 1.0) * 50.0).min(1500.0) as i32);
        }
        state.full(params, &pcm).map_err(|e| anyhow!("{e:?}"))?;
        let text: String = state
            .as_iter()
            .filter_map(|seg| seg.to_str_lossy().ok().map(|s| s.into_owned()))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(words_only(&text))
    }
}

/// "computer" -> "Computer," (how the wake word starts a sentence).
pub fn wake_hint(wake_word: &str) -> String {
    let mut c = wake_word.chars();
    let cap: String = c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default();
    format!("{cap},")
}
