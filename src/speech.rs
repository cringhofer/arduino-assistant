//! Wake word detection and command transcription (Vosk), plus optional Moonshine /
//! Whisper transcription of open questions.

use anyhow::{Context, Result, anyhow, bail};
use vosk::{DecodingState, Model, Recognizer};

use crate::audio::SAMPLE_RATE;
use crate::config::{Config, WakeEngine};

pub trait WakeDetector {
    /// Returns true when the wake word was heard in (or right before) these samples.
    fn feed(&mut self, samples: &[i16]) -> bool;
    fn reset(&mut self);
}

pub fn load_model(path: &str) -> Result<Model> {
    vosk::set_log_level(vosk::LogLevel::Error);
    Model::new(path).ok_or_else(|| anyhow!("cannot load Vosk model from {path}"))
}

pub fn wake_detector(cfg: &Config, model: &mut Model) -> Result<Box<dyn WakeDetector>> {
    match cfg.wake.engine {
        WakeEngine::Vosk => Ok(Box::new(VoskWake::new(model, &cfg.wake.word)?)),
    }
}

/// A Vosk recognizer restricted to the grammar `["<wake word>", "[unk]"]`: everything that
/// isn't the wake word collapses to [unk], so it's cheap and needs no training.
pub struct VoskWake {
    rec: Recognizer,
    word: String,
}

impl VoskWake {
    pub fn new(model: &mut Model, word: &str) -> Result<Self> {
        let word = normalize(word);
        if word.is_empty() {
            bail!("wake word is empty");
        }
        for w in word.split(' ') {
            if model.find_word(w).is_none() {
                bail!("wake word part '{w}' is not in the Vosk model vocabulary");
            }
        }
        let rec = Recognizer::new_with_grammar(model, SAMPLE_RATE as f32, &[word.as_str(), "[unk]"])
            .ok_or_else(|| anyhow!("cannot create wake recognizer"))?;
        Ok(Self { rec, word })
    }
}

impl WakeDetector for VoskWake {
    fn feed(&mut self, samples: &[i16]) -> bool {
        // Trigger on the partial result so the command can follow without a pause.
        let state = self.rec.accept_waveform(samples);
        let hit = match state {
            Ok(DecodingState::Finalized) => {
                let r = self.rec.result();
                r.single().is_some_and(|s| contains_words(s.text, &self.word))
            }
            Ok(_) => contains_words(self.rec.partial_result().partial, &self.word),
            Err(e) => {
                log::warn!("wake recognizer: {e}");
                false
            }
        };
        if hit {
            self.rec.reset();
        }
        hit
    }

    fn reset(&mut self) {
        self.rec.reset();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreeKind {
    Vosk,
    Moonshine,
    #[cfg_attr(not(feature = "whisper"), allow(dead_code))]
    Whisper,
    Candle,
}

/// Second transcriber: the main engine (Moonshine/Whisper/candle, non-streaming, run on
/// the command audio once it ended) or a streaming free Vosk recognizer for LLM questions.
enum Free {
    Vosk(Recognizer),
    #[cfg(feature = "moonshine")]
    Moonshine(crate::moonshine::Moonshine),
    #[cfg(feature = "moonshine-voice")]
    MoonshineVoice(crate::moonshine_voice::MoonshineVoice),
    #[cfg(feature = "whisper")]
    Whisper(crate::whisper::Whisper),
    #[cfg(feature = "candle")]
    Candle(crate::candle_whisper::CandleWhisper),
}

/// A finished command.
#[derive(Debug, Clone)]
pub struct Command {
    /// Transcript of the main recognizer (grammar-restricted unless disabled).
    pub text: String,
    /// The grammar recognizer heard words it doesn't know: probably not a command.
    pub unk: bool,
    /// Mean word confidence (0..1). Speech that isn't a command gets forced onto grammar
    /// words with low confidence.
    pub conf: f32,
    /// Free dictation of the same audio (only with grammar + LLM), for open questions.
    pub free: Option<String>,
    /// Mean word confidence of `free` (Vosk only).
    pub free_conf: f32,
    pub free_kind: FreeKind,
    /// The audio of this command including the pre-roll (16 kHz mono).
    pub audio: Vec<i16>,
    /// Sample in `audio` where the command starts (after the wake word), 0 if unknown.
    pub command_start: usize,
    /// The wake word was part of what was said (continuous mode: it was addressed to us).
    pub addressed: bool,
    /// When the end of the command was detected (before transcription).
    pub ended: std::time::Instant,
    /// Delay from the last speech to detecting the end of the command.
    pub endpoint_ms: f32,
    /// Time the main engine took to transcribe (0 for Vosk).
    pub stt_ms: f32,
}

pub enum Heard {
    Listening,
    Text(Command),
    /// Nothing (intelligible) was said.
    Nothing,
    /// The full recognizer never heard the wake word: the wake detector misfired.
    FalseWake(String),
}

/// Recognizer for the command after the wake word.
///
/// It is primed with the audio just before the wake detection, so the transcript reads
/// "... computer what time is it"; everything up to the wake word is dropped. That also
/// gives a second, more accurate opinion on whether the wake word was really said.
///
/// With a grammar it only knows the words/phrases the skills understand, which is far
/// more robust with a noisy, distant microphone than free dictation.
pub struct CommandListener {
    rec: Recognizer,
    /// Free-dictation recognizer fed the same audio (grammar mode + LLM).
    free: Option<Free>,
    wake: String,
    verify: bool,
    wake_seen: bool,
    /// Seconds of audio processed since the (last) start; audio time, not wall time.
    elapsed: f32,
    heard_speech: bool,
    initial_silence: f32,
    max_secs: f32,
    end_silence: f32,
    /// How long the command part of the partial result hasn't changed.
    stable_secs: f32,
    /// What the recognizer currently thinks is being said (for the status page).
    pub partial: String,
    /// Partial result including "[unk]" (words outside the grammar), for end detection.
    raw_partial: String,
    free_text: Vec<Utterance>,
    /// Pre-roll + command audio.
    audio: Vec<i16>,
    /// Where the wake word ended, in seconds since `start` (Vosk word timestamps).
    wake_end: Option<f32>,
    /// Background level (dBFS) and loudest chunk of this command, for end detection.
    noise_db: f32,
    peak_db: f32,
    cut_wake_word: bool,
    /// Continuous mode: no wake word needed.
    continuous: bool,
    /// Audio time of the last chunk that still looked like speech.
    last_voice: f32,
}

/// Pre-roll chunks (100 ms) given to a streaming engine. On the UNO Q with the small model,
/// 0.8 s: median response 4.3 s instead of 5.6 s (whole pre-roll, batch), no misses;
/// 0.5 s cut off the start of some commands.
const STREAM_PREROLL_CHUNKS: usize = 8;

/// How long the full recognizer may take to confirm the wake word.
const VERIFY_SECS: f32 = 2.5;

impl CommandListener {
    /// `grammar`: phrases to restrict recognition to, None for free dictation.
    /// `free_too`: also run a free-dictation recognizer (for LLM questions).
    pub fn new(model: &mut Model, cfg: &Config, grammar: Option<&[&str]>, free_too: bool) -> Result<Self> {
        let wake = normalize(&cfg.wake.word);
        let mut rec = match grammar {
            Some(phrases) => {
                let mut ok: Vec<&str> = vec![wake.as_str()];
                for p in phrases {
                    match p.split(' ').find(|w| *w != "[unk]" && model.find_word(w).is_none()) {
                        Some(w) => log::debug!("grammar: skipping '{p}' ('{w}' not in model)"),
                        None => ok.push(p),
                    }
                }
                log::info!("command recognizer restricted to {} phrases", ok.len());
                Recognizer::new_with_grammar(model, SAMPLE_RATE as f32, &ok)
            }
            None => Recognizer::new(model, SAMPLE_RATE as f32),
        }
        .ok_or_else(|| anyhow!("cannot create command recognizer"))?;
        rec.set_max_alternatives(0);
        rec.set_words(true);
        let free = match (grammar, free_too) {
            (Some(_), true) => Some(free_transcriber(model, cfg)?),
            _ => None,
        };
        Ok(Self {
            rec,
            free,
            wake,
            verify: cfg.wake.verify,
            wake_seen: false,
            elapsed: 0.0,
            heard_speech: false,
            initial_silence: cfg.stt.initial_silence_secs,
            max_secs: cfg.stt.max_command_secs,
            end_silence: cfg.stt.end_silence_secs,
            stable_secs: 0.0,
            partial: String::new(),
            raw_partial: String::new(),
            free_text: Vec::new(),
            audio: Vec::new(),
            wake_end: None,
            noise_db: -50.0,
            peak_db: -100.0,
            cut_wake_word: cfg.stt.cut_wake_word,
            continuous: false,
            last_voice: 0.0,
        })
    }

    /// Which main speech-to-text engine and model actually run (after any fallback).
    pub fn engine_info(&self, cfg: &Config) -> serde_json::Value {
        use serde_json::json;
        let model = crate::info::model;
        match &self.free {
            None => json!({ "engine": "vosk (grammar only)", "model": model(&cfg.stt.model_path) }),
            Some(Free::Vosk(_)) => json!({ "engine": "vosk", "model": model(&cfg.stt.model_path) }),
            #[cfg(feature = "moonshine")]
            Some(Free::Moonshine(_)) => json!({
                "engine": "moonshine", "runtime": "sherpa-onnx", "model": model(&cfg.stt.moonshine_dir),
            }),
            #[cfg(feature = "moonshine-voice")]
            Some(Free::MoonshineVoice(m)) => json!({
                "engine": "moonshine", "runtime": format!("moonshine-voice (library {})", m.version),
                "arch": m.arch, "model": model(&cfg.stt.moonshine_dir),
                "library": crate::moonshine_voice::library_path(&cfg.stt.moonshine_lib),
            }),
            #[cfg(feature = "whisper")]
            Some(Free::Whisper(_)) => json!({ "engine": "whisper", "runtime": "whisper.cpp", "model": model(&cfg.stt.whisper_model) }),
            #[cfg(feature = "candle")]
            Some(Free::Candle(_)) => json!({ "engine": "whisper", "runtime": "candle", "model": model(&cfg.stt.candle_model_dir) }),
        }
    }

    /// Starts a new command, replaying the audio that led up to the wake detection.
    /// `noise_db`: background level measured while idle. `continuous`: started by voice
    /// activity instead of the wake word, so don't require it.
    pub fn start<'a>(&mut self, preroll: impl IntoIterator<Item = &'a [i16]>, noise_db: f32, continuous: bool) {
        self.rec.reset();
        self.continuous = continuous;
        match &mut self.free {
            Some(Free::Vosk(f)) => f.reset(),
            #[cfg(feature = "moonshine-voice")]
            Some(Free::MoonshineVoice(m)) if m.streaming() => m.begin(),
            _ => {}
        }
        self.wake_seen = false;
        self.wake_end = None;
        self.partial.clear();
        self.raw_partial.clear();
        self.free_text.clear();
        self.audio.clear();
        self.noise_db = noise_db;
        self.peak_db = -100.0;
        let preroll: Vec<&[i16]> = preroll.into_iter().collect();
        // Streaming Moonshine only gets the end of the pre-roll (the wake word was detected
        // just before): less to analyse on a slow CPU, same accuracy in tests.
        let skip = if self.streaming_free() { preroll.len().saturating_sub(STREAM_PREROLL_CHUNKS) } else { 0 };
        for (i, chunk) in preroll.into_iter().enumerate() {
            if i < skip {
                self.audio.extend_from_slice(chunk);
            } else {
                self.feed_free(chunk);
            }
            if let Ok(DecodingState::Finalized) = self.rec.accept_waveform(chunk) {
                let u = utterance(&mut self.rec, false);
                let _ = self.segment(u);
            }
        }
        self.elapsed = 0.0;
        self.last_voice = 0.0;
        self.stable_secs = 0.0;
        self.heard_speech = false;
    }

    fn streaming_free(&self) -> bool {
        match &self.free {
            #[cfg(feature = "moonshine-voice")]
            Some(Free::MoonshineVoice(m)) => m.streaming(),
            _ => false,
        }
    }

    /// The wake word must be confirmed before a command counts.
    fn require_wake(&self) -> bool {
        self.verify && !self.continuous
    }

    fn feed_free(&mut self, samples: &[i16]) {
        self.audio.extend_from_slice(samples);
        match &mut self.free {
            Some(Free::Vosk(f)) => {
                if let Ok(DecodingState::Finalized) = f.accept_waveform(samples) {
                    // Keep the free transcript across its own endpoints.
                    self.free_text.push(utterance(f, false));
                }
            }
            // streaming: analysed while the command is spoken
            #[cfg(feature = "moonshine-voice")]
            Some(Free::MoonshineVoice(m)) if m.streaming() => m.push(samples),
            _ => {}
        }
    }

    /// First sample after the wake word, so the main engine never hears "computer" (it
    /// mishears it as "compute ..." / "voltage" otherwise). 0 if unknown.
    fn command_start(&self) -> usize {
        let Some(end) = self.wake_end.filter(|_| self.cut_wake_word) else { return 0 };
        let start = ((end + 0.05) * SAMPLE_RATE as f32) as usize;
        // implausible (less than 0.3 s left): keep everything
        if start + SAMPLE_RATE as usize * 3 / 10 > self.audio.len() { 0 } else { start }
    }

    pub fn feed(&mut self, samples: &[i16]) -> Heard {
        self.feed_free(samples);
        let heard = self.feed_main(samples);
        let Heard::Text(mut cmd) = heard else { return heard };
        let start = self.command_start();
        let command_audio = &self.audio[start..];
        let secs = command_audio.len() as f32 / SAMPLE_RATE as f32;
        cmd.endpoint_ms = (self.elapsed - self.last_voice).max(0.0) * 1000.0;
        cmd.addressed = self.wake_seen;
        let started = std::time::Instant::now();
        let transcript: Option<(FreeKind, String)> = match &mut self.free {
            Some(Free::Vosk(f)) => {
                self.free_text.push(utterance(f, true));
                let parts = std::mem::take(&mut self.free_text);
                let all = Utterance {
                    text: normalize(&parts.iter().map(|u| u.text.as_str()).collect::<Vec<_>>().join(" ")),
                    words: parts.into_iter().flat_map(|u| u.words).collect(),
                };
                cmd.free_conf = all.conf_after(&self.wake);
                Some((FreeKind::Vosk, all.text))
            }
            #[cfg(feature = "moonshine")]
            Some(Free::Moonshine(m)) => Some((FreeKind::Moonshine, m.transcribe(command_audio))),
            #[cfg(feature = "moonshine-voice")]
            Some(Free::MoonshineVoice(m)) => Some((FreeKind::Moonshine, if m.streaming() { m.finish() } else { m.transcribe(command_audio) }.unwrap_or_else(|e| {
                log::error!("{e:#}");
                String::new()
            }))),
            #[cfg(feature = "whisper")]
            Some(Free::Whisper(w)) => Some((FreeKind::Whisper, w.transcribe(command_audio).unwrap_or_else(|e| {
                log::error!("whisper: {e:#}");
                String::new()
            }))),
            #[cfg(feature = "candle")]
            Some(Free::Candle(w)) => Some((FreeKind::Candle, w.transcribe(command_audio).unwrap_or_else(|e| {
                log::error!("candle whisper: {e:#}");
                String::new()
            }))),
            None => None,
        };
        if let Some((kind, all)) = transcript {
            if kind != FreeKind::Vosk {
                cmd.stt_ms = started.elapsed().as_secs_f32() * 1000.0;
                log::debug!("{kind:?}: {secs:.1} s audio in {:.2} s", started.elapsed().as_secs_f32());
            }
            let text = after_words(&all, &self.wake).unwrap_or_else(|| strip_wake_like(&all, &self.wake));
            cmd.free_kind = kind;
            cmd.free = (!text.is_empty()).then_some(text);
        }
        cmd.command_start = start;
        cmd.audio = std::mem::take(&mut self.audio);
        Heard::Text(cmd)
    }

    fn feed_main(&mut self, samples: &[i16]) -> Heard {
        let secs = samples.len() as f32 / SAMPLE_RATE as f32;
        self.elapsed += secs;
        let elapsed = self.elapsed;
        match self.rec.accept_waveform(samples) {
            Ok(DecodingState::Finalized) => {
                let u = utterance(&mut self.rec, false);
                if let Some(h) = self.segment(u) {
                    return h;
                }
            }
            Ok(_) => {
                let raw = normalize(self.rec.partial_result().partial);
                let partial = clean(&raw);
                // The command part: after the wake word, or everything if not verifying.
                let command = match after_words(&partial, &self.wake) {
                    Some(rest) => {
                        self.wake_seen = true;
                        rest
                    }
                    None if self.wake_seen || !self.require_wake() => partial.clone(),
                    None => String::new(),
                };
                if !command.is_empty() {
                    self.heard_speech = true;
                }
                // Words outside the grammar only show up as "[unk]", so compare the raw
                // partial; and don't end while the mic is still clearly at speech level.
                let db = level_db(samples);
                self.peak_db = self.peak_db.max(db);
                let speaking = db > (self.noise_db + 8.0).max(self.peak_db - 12.0);
                if raw == self.raw_partial && !speaking {
                    self.stable_secs += secs;
                } else {
                    self.stable_secs = 0.0;
                    self.last_voice = elapsed;
                }
                self.raw_partial = raw;
                self.partial = partial;
                // Don't wait for Vosk's own endpoint (needs real silence, which a noisy
                // room rarely has): if the words stopped changing, the command is done.
                if !command.is_empty() && self.stable_secs >= self.end_silence {
                    let u = utterance(&mut self.rec, true);
                    if let Some(h) = self.segment(u) {
                        return h;
                    }
                }
            }
            Err(e) => log::warn!("command recognizer: {e}"),
        }
        let unconfirmed = self.require_wake() && !self.wake_seen;
        if elapsed > self.max_secs || (unconfirmed && elapsed > VERIFY_SECS) {
            let u = utterance(&mut self.rec, true);
            return match self.segment(u) {
                Some(h) => h,
                None if unconfirmed => Heard::FalseWake(String::new()),
                None => Heard::Nothing,
            };
        }
        // Continuous mode starts on any noise: give up on it quickly.
        let initial_silence = if self.continuous { self.initial_silence.min(2.0) } else { self.initial_silence };
        if !self.heard_speech && elapsed > initial_silence {
            return Heard::Nothing;
        }
        Heard::Listening
    }

    /// Handles one finalized utterance. None = keep listening.
    fn segment(&mut self, u: Utterance) -> Option<Heard> {
        let text = u.text.clone();
        if clean(&text).is_empty() {
            return None;
        }
        if let Some(rest) = after_words(&text, &self.wake) {
            self.wake_seen = true;
            if let Some(end) = u.wake_end(&self.wake) {
                self.wake_end = Some(end);
            }
            if clean(&rest).is_empty() {
                // "computer" ... pause ... "what time is it": wait for the command.
                self.elapsed = 0.0;
                self.last_voice = 0.0;
                self.stable_secs = 0.0;
                self.heard_speech = false;
                return None;
            }
            return Some(command(&rest, u.conf_after(&self.wake)));
        }
        if self.wake_seen || !self.require_wake() {
            return Some(command(&text, u.conf_after(&self.wake)));
        }
        // Speech from before the wake word, or a misfire; decided by the VERIFY_SECS timeout.
        log::debug!("ignoring '{text}' (no wake word yet)");
        None
    }
}

fn command(raw: &str, conf: f32) -> Heard {
    Heard::Text(Command {
        text: clean(raw),
        unk: raw.contains("[unk]"),
        conf,
        free: None,
        free_conf: 0.0,
        free_kind: FreeKind::Vosk,
        audio: Vec::new(),
        command_start: 0,
        addressed: false,
        ended: std::time::Instant::now(),
        endpoint_ms: 0.0,
        stt_ms: 0.0,
    })
}

/// A finalized result with per-word confidences.
struct Utterance {
    text: String,
    /// (word, confidence, end time in seconds since the recognizer was reset)
    words: Vec<(String, f32, f32)>,
}

impl Utterance {
    /// Mean confidence of the words after the (last) wake word, ignoring "[unk]".
    fn conf_after(&self, wake: &str) -> f32 {
        let wake: Vec<&str> = wake.split(' ').collect();
        let start = (0..self.words.len().saturating_sub(wake.len() - 1))
            .rev()
            .find(|&i| self.words[i..i + wake.len()].iter().map(|(w, ..)| w.as_str()).eq(wake.iter().copied()))
            .map_or(0, |i| i + wake.len());
        let confs: Vec<f32> =
            self.words[start..].iter().filter(|(w, ..)| w != "[unk]").map(|(_, c, _)| *c).collect();
        if confs.is_empty() { 0.0 } else { confs.iter().sum::<f32>() / confs.len() as f32 }
    }

    /// End time of the last occurrence of the wake word.
    fn wake_end(&self, wake: &str) -> Option<f32> {
        let wake: Vec<&str> = wake.split(' ').collect();
        (0..self.words.len().saturating_sub(wake.len() - 1))
            .rev()
            .find(|&i| self.words[i..i + wake.len()].iter().map(|(w, ..)| w.as_str()).eq(wake.iter().copied()))
            .map(|i| self.words[i + wake.len() - 1].2)
    }
}

fn utterance(rec: &mut Recognizer, final_: bool) -> Utterance {
    let r = if final_ { rec.final_result() } else { rec.result() };
    match r.single() {
        Some(s) => Utterance {
            text: normalize(s.text),
            words: s.result.iter().map(|w| (w.word.to_lowercase(), w.conf, w.end)).collect(),
        },
        None => Utterance { text: String::new(), words: Vec::new() },
    }
}

/// Normalizes and drops the "[unk]" placeholders grammar recognizers emit.
fn clean(s: &str) -> String {
    normalize(s).split(' ').filter(|w| *w != "[unk]").collect::<Vec<_>>().join(" ")
}

/// "Computer, what's the time?" -> "computer what's the time"
pub fn words_only(text: &str) -> String {
    let t: String = text
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c.to_lowercase().next().unwrap_or(c) } else { ' ' })
        .collect();
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Loudness of a chunk in dBFS.
pub fn level_db(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return -100.0;
    }
    let ms = samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64;
    (10.0 * (ms.max(1.0) / (32768.0 * 32768.0)).log10()) as f32
}

/// Drops a misheard wake word at the start ("compute correctly ..." -> "correctly ...").
fn strip_wake_like(text: &str, wake: &str) -> String {
    let mut words: Vec<&str> = text.split_whitespace().collect();
    let first = wake.split(' ').next().unwrap_or(wake);
    if let Some(w) = words.first() {
        let shared = w.chars().zip(first.chars()).take_while(|(a, b)| a == b).count();
        if shared >= 5 || (shared >= 4 && w.len() + 2 >= first.len()) {
            words.remove(0);
        }
    }
    words.join(" ")
}

pub fn normalize(s: &str) -> String {
    s.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

fn contains_words(text: &str, words: &str) -> bool {
    format!(" {} ", normalize(text)).contains(&format!(" {words} "))
}

/// The text after the last occurrence of `words`, None if it doesn't occur.
fn after_words(text: &str, words: &str) -> Option<String> {
    let padded = format!(" {text} ");
    let needle = format!(" {words} ");
    let i = padded.rfind(&needle)?;
    Some(padded[i + needle.len()..].trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_wake_word() {
        assert_eq!(after_words("hey computer what time is it", "computer").as_deref(), Some("what time is it"));
        assert_eq!(after_words("computer", "computer").as_deref(), Some(""));
        assert_eq!(after_words("my commuter train", "computer"), None);
    }

    #[test]
    fn strips_misheard_wake_word() {
        assert_eq!(strip_wake_like("compute correctly nice meat", "computer"), "correctly nice meat");
        assert_eq!(strip_wake_like("computers what time is it", "computer"), "what time is it");
        assert_eq!(strip_wake_like("what time is it", "computer"), "what time is it");
        assert_eq!(strip_wake_like("come here", "computer"), "come here");
    }
}

/// Transcribes WAV files with the configured main engine: "<secs> <audio secs> <file> <text>".
pub fn transcribe_files(model: &mut Model, cfg: &Config, files: &[String]) -> Result<()> {
    let mut l = CommandListener::new(model, cfg, Some(&[]), true)?;
    for f in files {
        let data = std::fs::read(f).with_context(|| format!("reading {f}"))?;
        let pcm = crate::audio::wav_data(&data).ok_or_else(|| anyhow!("{f}: not a WAV"))?;
        let samples: Vec<i16> = pcm.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let started = std::time::Instant::now();
        let text = match &mut l.free {
            Some(Free::Vosk(r)) => {
                let _ = r.accept_waveform(&samples);
                let t = utterance(r, true).text;
                r.reset();
                t
            }
            #[cfg(feature = "moonshine")]
            Some(Free::Moonshine(m)) => m.transcribe(&samples),
            // streaming models: fed in real time like the mic; the time is the wait after the end
            #[cfg(feature = "moonshine-voice")]
            Some(Free::MoonshineVoice(m)) if m.streaming() => {
                m.begin();
                for chunk in samples.chunks(1600) {
                    m.push(chunk);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                let end = std::time::Instant::now();
                let text = m.finish()?;
                println!("{:.2}\t{:.1}\t{f}\t{text}", end.elapsed().as_secs_f32(), samples.len() as f32 / SAMPLE_RATE as f32);
                continue;
            }
            #[cfg(feature = "moonshine-voice")]
            Some(Free::MoonshineVoice(m)) => m.transcribe(&samples)?,
            #[cfg(feature = "whisper")]
            Some(Free::Whisper(w)) => w.transcribe(&samples)?,
            #[cfg(feature = "candle")]
            Some(Free::Candle(w)) => w.transcribe(&samples)?,
            None => String::new(),
        };
        println!(
            "{:.2}\t{:.1}\t{f}\t{text}",
            started.elapsed().as_secs_f32(),
            samples.len() as f32 / SAMPLE_RATE as f32
        );
    }
    Ok(())
}

fn vosk_free(model: &Model) -> Result<Free> {
    let mut r = Recognizer::new(model, SAMPLE_RATE as f32).ok_or_else(|| anyhow!("cannot create free recognizer"))?;
    r.set_max_alternatives(0);
    r.set_words(true);
    Ok(Free::Vosk(r))
}

fn free_transcriber(model: &Model, cfg: &Config) -> Result<Free> {
    use crate::config::{FreeEngine, Lang};
    match cfg.stt.engine {
        FreeEngine::Vosk => vosk_free(model),
        // moonshine-voice model (English or German)
        #[cfg(feature = "moonshine-voice")]
        FreeEngine::Moonshine if crate::moonshine_voice::is_model_dir(&cfg.stt.moonshine_dir) => {
            let mut terms = vec![cfg.wake.word.clone()];
            terms.extend(cfg.stt.moonshine_keyterms.iter().cloned());
            match crate::moonshine_voice::MoonshineVoice::new(&cfg.stt.moonshine_lib, &cfg.stt.moonshine_dir, &terms) {
                Ok(m) => {
                    log::info!("main engine: Moonshine ({})", cfg.stt.moonshine_dir);
                    Ok(Free::MoonshineVoice(m))
                }
                Err(e) => {
                    log::error!("{e:#}, using Vosk for free dictation");
                    vosk_free(model)
                }
            }
        }
        FreeEngine::Moonshine if cfg.language != Lang::En => {
            log::warn!("no Moonshine model for {:?} in {}, using Vosk for free dictation", cfg.language, cfg.stt.moonshine_dir);
            vosk_free(model)
        }
        #[cfg(feature = "moonshine")]
        FreeEngine::Moonshine => match crate::moonshine::Moonshine::new(&cfg.stt.moonshine_dir, cfg.stt.moonshine_threads) {
            Ok(m) => {
                log::info!("free dictation: Moonshine ({})", cfg.stt.moonshine_dir);
                Ok(Free::Moonshine(m))
            }
            Err(e) => {
                log::error!("{e:#}, using Vosk for free dictation");
                vosk_free(model)
            }
        },
        #[cfg(not(feature = "moonshine"))]
        FreeEngine::Moonshine => {
            log::warn!("built without the `moonshine` feature, using Vosk for free dictation");
            vosk_free(model)
        }
        #[cfg(feature = "candle")]
        FreeEngine::Candle => match crate::candle_whisper::CandleWhisper::new(
            &cfg.stt.candle_model_dir,
            lang_code(cfg.language),
            cfg.stt.whisper_short_window,
            &cfg.wake.word,
        ) {
            Ok(w) => {
                log::info!("free dictation: candle Whisper ({})", cfg.stt.candle_model_dir);
                Ok(Free::Candle(w))
            }
            Err(e) => {
                log::error!("{e:#}, using Vosk for free dictation");
                vosk_free(model)
            }
        },
        #[cfg(not(feature = "candle"))]
        FreeEngine::Candle => {
            log::warn!("built without the `candle` feature, using Vosk for free dictation");
            vosk_free(model)
        }
        #[cfg(feature = "whisper")]
        FreeEngine::Whisper => {
            let lang = lang_code(cfg.language);
            match crate::whisper::Whisper::new(
                &cfg.stt.whisper_model,
                lang,
                cfg.stt.whisper_threads,
                cfg.stt.whisper_short_window,
                &cfg.wake.word,
            ) {
                Ok(w) => {
                    log::info!("free dictation: Whisper ({})", cfg.stt.whisper_model);
                    Ok(Free::Whisper(w))
                }
                Err(e) => {
                    log::error!("{e:#}, using Vosk for free dictation");
                    vosk_free(model)
                }
            }
        }
        #[cfg(not(feature = "whisper"))]
        FreeEngine::Whisper => {
            log::warn!("built without the `whisper` feature, using Vosk for free dictation");
            vosk_free(model)
        }
    }
}

#[allow(dead_code)] // unused when built without Whisper engines
fn lang_code(lang: crate::config::Lang) -> &'static str {
    match lang {
        crate::config::Lang::En => "en",
        crate::config::Lang::De => "de",
    }
}
