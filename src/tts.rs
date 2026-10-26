//! Text to speech: Piper (neural, natural) or espeak-ng (tiny, robotic), played via aplay.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::audio::aplay_raw;
use crate::config::{Config, TtsConfig, TtsEngine};

pub struct Tts {
    cfg: Config,
    piper: Mutex<Option<Piper>>,
}

impl Tts {
    pub fn new(cfg: &Config) -> Self {
        // start Piper now, so its voice is loaded before the first reply
        let piper = (cfg.tts.engine == TtsEngine::Piper)
            .then(|| Piper::start(&cfg.tts).map_err(|e| log::error!("{e:#}")).ok())
            .flatten();
        Self { cfg: cfg.clone(), piper: Mutex::new(piper) }
    }

    /// Speaks and blocks until playback is finished. Returns the spoken audio (also when
    /// no speaker works), so the status page can play it back.
    pub fn say(&self, text: &str) -> Option<Speech> {
        if self.cfg.tts.engine != TtsEngine::Print {
            log::info!("> {text}");
        }
        let res = match self.cfg.tts.engine {
            TtsEngine::Piper => self.piper(text).map(Some),
            TtsEngine::Espeak => self.espeak(text).map(Some),
            TtsEngine::Print => {
                println!("{text}");
                Ok(None)
            }
        };
        res.unwrap_or_else(|e| {
            log::error!("tts failed: {e:#}");
            None
        })
    }

    /// Piper, kept running (loading the voice takes ~2.3 s on the UNO Q). The reply goes
    /// in sentence by sentence: the first one plays while the rest is synthesized.
    fn piper(&self, text: &str) -> Result<Speech> {
        let t = &self.cfg.tts;
        let started = Instant::now();
        let mut guard = self.piper.lock().unwrap_or_else(|e| e.into_inner());
        if guard.as_mut().is_none_or(|p| !p.alive()) {
            *guard = Some(Piper::start(t)?);
        }
        let piper = guard.as_mut().unwrap();
        let result = piper.speak(text, &self.cfg.audio.playback_device, t.piper_sample_rate, started);
        if result.is_err() {
            *guard = None; // restarted next time
        }
        result
    }

    fn espeak(&self, text: &str) -> Result<Speech> {
        let started = Instant::now();
        let out = Command::new("espeak-ng")
            .args(["-v", &self.cfg.tts.espeak_voice, "-s", "160", "--stdout", text])
            .stderr(Stdio::null())
            .output()
            .context("running espeak-ng")?;
        let first_ms = started.elapsed().as_secs_f32() * 1000.0;
        if let Ok(mut aplay) = Command::new("aplay")
            .args(["-q", "-D", &self.cfg.audio.playback_device])
            .stdin(Stdio::piped())
            .spawn()
        {
            let _ = aplay.stdin.take().unwrap().write_all(&out.stdout);
            let _ = aplay.wait();
        }
        let pcm = crate::audio::wav_data(&out.stdout).unwrap_or_default();
        let samples = pcm.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        Ok(Speech { samples, rate: 22050, first_ms })
    }
}

/// Synthesized speech: mono 16-bit samples.
pub struct Speech {
    pub samples: Vec<i16>,
    pub rate: u32,
    /// Time until the first audio was ready (synthesis latency).
    pub first_ms: f32,
}

/// A running Piper process. Each input line becomes a WAV file in `dir`; Piper prints its
/// path once it is written.
struct Piper {
    child: Child,
    stdin: ChildStdin,
    out: BufReader<ChildStdout>,
    dir: PathBuf,
}

impl Piper {
    fn start(t: &TtsConfig) -> Result<Self> {
        let dir = std::env::temp_dir().join(format!("uno-piper-{}", std::process::id()));
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let mut child = Command::new(&t.piper_bin)
            .args(["--model", &t.piper_model, "--quiet", "--output_dir"])
            .arg(&dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("starting {}", t.piper_bin))?;
        let stdin = child.stdin.take().unwrap();
        let out = BufReader::new(child.stdout.take().unwrap());
        log::debug!("piper started");
        Ok(Self { child, stdin, out, dir })
    }

    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn speak(&mut self, text: &str, device: &str, rate: u32, started: Instant) -> Result<Speech> {
        let parts = sentences(text);
        for p in &parts {
            writeln!(self.stdin, "{p}")?;
        }
        self.stdin.flush()?;
        let mut aplay = aplay_raw(device, rate).stdin(Stdio::piped()).spawn().map_err(|e| log::warn!("aplay: {e}")).ok();
        let mut sink = aplay.as_mut().and_then(|a| a.stdin.take());
        let mut samples: Vec<i16> = Vec::new();
        let mut first_ms = None;
        for _ in &parts {
            let mut line = String::new();
            if self.out.read_line(&mut line)? == 0 {
                bail!("piper stopped");
            }
            let path = PathBuf::from(line.trim());
            let data = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let _ = std::fs::remove_file(&path);
            let pcm = crate::audio::wav_data(&data).unwrap_or_default();
            first_ms.get_or_insert_with(|| started.elapsed().as_secs_f32() * 1000.0);
            if let Some(s) = &mut sink {
                if s.write_all(pcm).is_err() {
                    sink = None; // no speaker; keep collecting
                }
            }
            samples.extend(pcm.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])));
        }
        drop(sink);
        if let Some(mut a) = aplay {
            let _ = a.wait();
        }
        Ok(Speech { samples, rate, first_ms: first_ms.unwrap_or(0.0) })
    }
}

impl Drop for Piper {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Splits a reply into sentences (one Piper line each); never yields empty parts.
fn sentences(text: &str) -> Vec<String> {
    let text = text.replace('\n', " ");
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?') && chars.peek().is_none_or(|n| n.is_whitespace()) {
            out.push(std::mem::take(&mut cur));
        }
    }
    out.push(cur);
    out.into_iter().map(|s| s.trim().to_string()).filter(|s| s.chars().any(char::is_alphanumeric)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences() {
        assert_eq!(sentences("Yes, I am. How may I help?"), vec!["Yes, I am.", "How may I help?"]);
        assert_eq!(sentences("It is 3.5 degrees.\n"), vec!["It is 3.5 degrees."]);
        assert_eq!(sentences("..."), Vec::<String>::new());
        assert_eq!(sentences("No end"), vec!["No end"]);
    }
}
