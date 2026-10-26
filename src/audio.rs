//! Audio I/O via ALSA command line tools. `arecord`/`aplay` do format conversion and
//! resampling (plughw), keep the binary free of native audio deps and make
//! cross-compiling trivial.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::Event;

pub const SAMPLE_RATE: u32 = 16_000;
/// 100 ms chunks.
const CHUNK: usize = 1600;

/// Streams 16 kHz mono i16 audio from the microphone into the event channel.
pub fn spawn_mic(device: &str, tx: Sender<Event>) -> Result<()> {
    let mut child = Command::new("arecord")
        .args(["-q", "-D", device, "-f", "S16_LE", "-c", "1", "-t", "raw"])
        .args(["-r", &SAMPLE_RATE.to_string()])
        .stdout(Stdio::piped())
        .spawn()
        .context("starting arecord (is alsa-utils installed?)")?;
    let mut out = child.stdout.take().unwrap();
    thread::spawn(move || {
        let mut buf = vec![0u8; CHUNK * 2];
        while out.read_exact(&mut buf).is_ok() {
            if tx.send(Event::Audio(Instant::now(), bytes_to_i16(&buf))).is_err() {
                break;
            }
        }
        let _ = child.kill();
        let _ = tx.send(Event::MicStopped);
    });
    Ok(())
}

/// Feeds a 16 kHz mono 16-bit WAV file instead of the microphone (for testing).
pub fn spawn_wav(path: &str, tx: Sender<Event>) -> Result<()> {
    let data = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let pcm = wav_data(&data).context("expected a 16 kHz mono 16-bit PCM WAV")?;
    let mut samples = bytes_to_i16(pcm);
    samples.extend(std::iter::repeat_n(0, SAMPLE_RATE as usize * 2));
    thread::spawn(move || {
        for chunk in samples.chunks(CHUNK) {
            if tx.send(Event::Audio(Instant::now(), chunk.to_vec())).is_err() {
                return;
            }
            thread::sleep(Duration::from_millis(100)); // real time, like a mic
        }
        let _ = tx.send(Event::MicStopped);
    });
    Ok(())
}

pub fn wav_data(data: &[u8]) -> Option<&[u8]> {
    if data.get(0..4)? != b"RIFF" || data.get(8..12)? != b"WAVE" {
        return None;
    }
    let mut pos = 12;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let len = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().ok()?) as usize;
        if id == b"data" {
            return data.get(pos + 8..(pos + 8 + len).min(data.len()));
        }
        pos += 8 + len + (len & 1);
    }
    None
}

fn bytes_to_i16(b: &[u8]) -> Vec<i16> {
    b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Spawns `aplay` reading raw mono S16_LE from stdin.
pub fn aplay_raw(device: &str, rate: u32) -> Command {
    let mut cmd = Command::new("aplay");
    cmd.args(["-q", "-D", device, "-f", "S16_LE", "-c", "1", "-t", "raw"])
        .args(["-r", &rate.to_string()]);
    cmd
}

/// Plays raw samples. `wait` = block until finished.
pub fn play(device: &str, samples: Vec<i16>, wait: bool) {
    let child = aplay_raw(device, SAMPLE_RATE).stdin(Stdio::piped()).spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return log::error!("aplay failed: {e}"),
    };
    let mut stdin = child.stdin.take().unwrap();
    let job = move || {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let _ = stdin.write_all(&bytes);
        drop(stdin);
        let _ = child.wait();
    };
    if wait { job() } else { drop(thread::spawn(job)) }
}

/// Short sine tones, e.g. `tones(&[(880.0, 0.08), (1320.0, 0.1)])`.
pub fn tones(parts: &[(f32, f32)]) -> Vec<i16> {
    let mut out = Vec::new();
    for &(freq, secs) in parts {
        let n = (SAMPLE_RATE as f32 * secs) as usize;
        let fade = (n / 8).max(1);
        for i in 0..n {
            let env = (i.min(n - i) as f32 / fade as f32).min(1.0);
            let v = (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin();
            out.push((v * env * 9000.0) as i16);
        }
    }
    out
}

pub fn wake_chime() -> Vec<i16> {
    tones(&[(880.0, 0.07), (1320.0, 0.09)])
}

pub fn done_chime() -> Vec<i16> {
    tones(&[(1320.0, 0.07), (880.0, 0.09)])
}

pub fn alarm() -> Vec<i16> {
    let mut v = Vec::new();
    for _ in 0..3 {
        v.extend(tones(&[(1000.0, 0.15), (0.0, 0.1), (1000.0, 0.15), (0.0, 0.35)]));
    }
    v
}

/// Mono 16-bit PCM as WAV file bytes.
pub fn wav_bytes(samples: &[i16], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Writes 16 kHz mono 16-bit PCM as a WAV file.
pub fn write_wav(path: &str, samples: &[i16]) -> std::io::Result<()> {
    std::fs::write(path, wav_bytes(samples, SAMPLE_RATE))
}
