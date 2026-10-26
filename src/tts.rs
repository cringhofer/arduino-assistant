//! Text to speech: Piper (neural, natural) or espeak-ng (tiny, robotic), played via aplay.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::audio::aplay_raw;
use crate::config::{Config, TtsEngine};

pub struct Tts {
    cfg: Config,
}

impl Tts {
    pub fn new(cfg: &Config) -> Self {
        Self { cfg: cfg.clone() }
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

    fn piper(&self, text: &str) -> Result<Speech> {
        let t = &self.cfg.tts;
        let started = Instant::now();
        let mut piper = Command::new(&t.piper_bin)
            .args(["--model", &t.piper_model, "--output_raw", "--quiet"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("starting {}", t.piper_bin))?;
        {
            let mut stdin = piper.stdin.take().unwrap();
            writeln!(stdin, "{}", text.replace('\n', " "))?;
        }
        // Piper streams sentence by sentence: pass each piece on to aplay right away and
        // keep a copy.
        let mut pcm = piper.stdout.take().unwrap();
        let mut aplay = aplay_raw(&self.cfg.audio.playback_device, t.piper_sample_rate)
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| log::warn!("aplay: {e}"))
            .ok();
        let mut sink = aplay.as_mut().and_then(|a| a.stdin.take());
        let mut bytes = Vec::new();
        let mut buf = [0u8; 8192];
        let mut first_ms = None;
        loop {
            let n = pcm.read(&mut buf)?;
            if n == 0 {
                break;
            }
            first_ms.get_or_insert_with(|| started.elapsed().as_secs_f32() * 1000.0);
            bytes.extend_from_slice(&buf[..n]);
            if let Some(s) = &mut sink {
                if s.write_all(&buf[..n]).is_err() {
                    sink = None; // no speaker; keep collecting
                }
            }
        }
        drop(sink);
        piper.wait()?;
        if let Some(mut a) = aplay {
            a.wait()?;
        }
        let samples = bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        Ok(Speech { samples, rate: t.piper_sample_rate, first_ms: first_ms.unwrap_or(0.0) })
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
