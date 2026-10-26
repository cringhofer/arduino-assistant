//! Moonshine speech-to-text via the official moonshine-voice runtime (libmoonshine.so,
//! loaded at run time). Its streaming models come in English and German; the command
//! is transcribed once it has ended.
//!
//! Library: the `moonshine-voice-linux-<arch>.tar.gz` release from
//! github.com/moonshine-ai/moonshine (libmoonshine.so + libonnxruntime.so.1 next to it).
//! Models: scripts/get-moonshine-model.sh.

use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::sync::{Arc, mpsc};
use std::thread;

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;

use crate::audio::SAMPLE_RATE;
use crate::speech::words_only;

const HEADER_VERSION: i32 = 30000;
const ARCH_TINY: u32 = 0;
const ARCH_BASE: u32 = 1;
const ARCH_TINY_STREAMING: u32 = 2;
const ARCH_SMALL_STREAMING: u32 = 4;
const ARCH_MEDIUM_STREAMING: u32 = 5;

#[repr(C)]
struct MoonshineOption {
    name: *const c_char,
    value: *const c_char,
}

/// `struct transcript_line_t` (moonshine-c-api.h, header version 30000).
#[repr(C)]
struct Line {
    text: *const c_char,
    audio_data: *const f32,
    audio_data_count: usize,
    start_time: f32,
    duration: f32,
    id: u64,
    is_complete: i8,
    is_updated: i8,
    is_new: i8,
    has_text_changed: i8,
    have_speakers_changed: i8,
    speaker_spans: *const std::ffi::c_void,
    speaker_span_count: u64,
    last_transcription_latency_ms: u32,
    words: *const std::ffi::c_void,
    word_count: u64,
}

#[repr(C)]
struct Transcript {
    lines: *const Line,
    line_count: u64,
}

type LoadFn = unsafe extern "C" fn(*const c_char, u32, *const MoonshineOption, u64, i32) -> i32;
type TranscribeFn = unsafe extern "C" fn(i32, *mut f32, u64, i32, u32, *mut *mut Transcript) -> i32;
type ErrorFn = unsafe extern "C" fn(i32) -> *const c_char;
type FreeFn = unsafe extern "C" fn(i32);
type VersionFn = unsafe extern "C" fn() -> i32;
type CreateStreamFn = unsafe extern "C" fn(i32, u32) -> i32;
type StreamFn = unsafe extern "C" fn(i32, i32) -> i32;
type AddAudioFn = unsafe extern "C" fn(i32, i32, *const f32, u64, i32, u32) -> i32;
type TranscribeStreamFn = unsafe extern "C" fn(i32, i32, u32, *mut *mut Transcript) -> i32;

/// The library functions used (valid as long as the library stays loaded).
#[derive(Clone, Copy)]
struct Api {
    transcribe: TranscribeFn,
    error: ErrorFn,
    free: FreeFn,
    create_stream: CreateStreamFn,
    free_stream: StreamFn,
    start_stream: StreamFn,
    stop_stream: StreamFn,
    add_audio: AddAudioFn,
    transcribe_stream: TranscribeStreamFn,
}

impl Api {
    fn check(&self, rc: i32) -> Result<()> {
        if rc == 0 {
            return Ok(());
        }
        let msg = unsafe { CStr::from_ptr((self.error)(rc)) }.to_string_lossy().into_owned();
        bail!("Moonshine: {msg}")
    }
}

/// Text of all lines of a transcript.
///
/// SAFETY: `t` must be null or a transcript just returned by the library (it stays valid
/// until the next call on the same transcriber).
unsafe fn text_of(t: *const Transcript) -> String {
    let Some(t) = (unsafe { t.as_ref() }) else { return String::new() };
    let lines: &[Line] =
        if t.lines.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(t.lines, t.line_count as usize) } };
    let text: Vec<String> = lines
        .iter()
        .filter(|l| !l.text.is_null())
        .map(|l| unsafe { CStr::from_ptr(l.text) }.to_string_lossy().into_owned())
        .collect();
    words_only(&text.join(" "))
}

enum Msg {
    Start,
    Audio(Vec<f32>),
    Finish(mpsc::Sender<Result<String>>),
}

/// Streaming: a worker thread analyses the audio while the command is still being
/// spoken, so only the last bit is left when it ends.
struct Streamer {
    tx: Option<mpsc::Sender<Msg>>,
    worker: Option<thread::JoinHandle<()>>,
    stream: i32,
}

fn worker(api: Api, handle: i32, stream: i32, rx: mpsc::Receiver<Msg>) {
    let mut running = false;
    let mut queued: Option<Msg> = None;
    loop {
        let msg = match queued.take() {
            Some(m) => m,
            None => match rx.recv() {
                Ok(m) => m,
                Err(_) => break,
            },
        };
        match msg {
            Msg::Start => {
                if running {
                    unsafe { (api.stop_stream)(handle, stream) };
                }
                running = api.check(unsafe { (api.start_stream)(handle, stream) }).map_err(|e| log::error!("{e:#}")).is_ok();
            }
            Msg::Audio(pcm) if running => {
                unsafe { (api.add_audio)(handle, stream, pcm.as_ptr(), pcm.len() as u64, SAMPLE_RATE as i32, 0) };
                // catch up: add everything that arrived meanwhile, then analyse once
                loop {
                    match rx.try_recv() {
                        Ok(Msg::Audio(pcm)) => unsafe {
                            (api.add_audio)(handle, stream, pcm.as_ptr(), pcm.len() as u64, SAMPLE_RATE as i32, 0);
                        },
                        Ok(other) => {
                            queued = Some(other);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                if queued.is_none() {
                    let mut out = std::ptr::null_mut();
                    unsafe { (api.transcribe_stream)(handle, stream, 0, &mut out) };
                }
            }
            Msg::Audio(_) => {}
            Msg::Finish(reply) => {
                let text = if running {
                    running = false;
                    let mut out = std::ptr::null_mut();
                    api.check(unsafe { (api.stop_stream)(handle, stream) })
                        .and_then(|_| api.check(unsafe { (api.transcribe_stream)(handle, stream, 0, &mut out) }))
                        .map(|_| unsafe { text_of(out) })
                } else {
                    Ok(String::new())
                };
                let _ = reply.send(text);
            }
        }
    }
}

pub struct MoonshineVoice {
    api: Api,
    handle: i32,
    streamer: Option<Streamer>,
    /// Library version, e.g. 30000.
    pub version: i32,
    pub arch: &'static str,
    // keeps the function pointers above valid
    _lib: Arc<Library>,
}

/// True if `dir` holds a moonshine-voice model (as opposed to a sherpa-onnx one).
pub fn is_model_dir(dir: &str) -> bool {
    let d = Path::new(dir);
    d.join("tokenizer.bin").exists() && (d.join("streaming_config.json").exists() || d.join("encoder_model.ort").exists())
}

/// Model architecture from the directory: streaming models have a streaming_config.json,
/// the size is taken from the name (as created by get-moonshine-model.sh).
fn arch(dir: &str) -> Result<u32> {
    let name = Path::new(dir).file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let streaming = Path::new(dir).join("streaming_config.json").exists();
    Ok(match (streaming, ["tiny", "base", "small", "medium"].into_iter().find(|s| name.contains(s))) {
        (true, Some("tiny")) => ARCH_TINY_STREAMING,
        (true, Some("small")) => ARCH_SMALL_STREAMING,
        (true, Some("medium")) => ARCH_MEDIUM_STREAMING,
        (false, Some("tiny")) => ARCH_TINY,
        (false, Some("base")) => ARCH_BASE,
        _ => bail!("cannot tell the Moonshine model size from '{dir}' (name it like moonshine-tiny-streaming-de)"),
    })
}

/// The configured library, or by default lib/moonshine/libmoonshine.so next to the
/// program's bin directory (as deployed), else the library search path.
pub fn library_path(configured: &str) -> String {
    if !configured.is_empty() {
        return configured.to_string();
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.parent()?.join("lib/moonshine/libmoonshine.so")))
        .filter(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "libmoonshine.so".into())
}

impl MoonshineVoice {
    /// `lib`: path of libmoonshine.so (or just the name, to use the library search path).
    /// `keyterms`: words to bias recognition towards (the wake word, command words).
    pub fn new(lib: &str, dir: &str, keyterms: &[String]) -> Result<Self> {
        let arch = arch(dir)?;
        let streaming_arch = matches!(arch, ARCH_TINY_STREAMING | ARCH_SMALL_STREAMING | ARCH_MEDIUM_STREAMING);
        // MOONSHINE_BATCH=1: transcribe the finished command at once instead (benchmarks)
        let streaming = streaming_arch && std::env::var("MOONSHINE_BATCH").is_err();
        let lib = library_path(lib);
        // SAFETY: loading the moonshine library runs no unusual initialisers.
        let lib = Arc::new(unsafe { Library::new(&lib) }.with_context(|| format!("loading {lib}"))?);
        let (load, version, api) = unsafe {
            (
                *lib.get::<LoadFn>(b"moonshine_load_transcriber_from_files\0")?,
                *lib.get::<VersionFn>(b"moonshine_get_version\0")?,
                Api {
                    transcribe: *lib.get(b"moonshine_transcribe_without_streaming\0")?,
                    error: *lib.get(b"moonshine_error_to_string\0")?,
                    free: *lib.get(b"moonshine_free_transcriber\0")?,
                    create_stream: *lib.get(b"moonshine_create_stream\0")?,
                    free_stream: *lib.get(b"moonshine_free_stream\0")?,
                    start_stream: *lib.get(b"moonshine_start_stream\0")?,
                    stop_stream: *lib.get(b"moonshine_stop_stream\0")?,
                    add_audio: *lib.get(b"moonshine_transcribe_add_audio_to_stream\0")?,
                    transcribe_stream: *lib.get(b"moonshine_transcribe_stream\0")?,
                },
            )
        };
        let mut opts: Vec<(CString, CString)> = Vec::new();
        let terms: Vec<&str> = keyterms.iter().map(String::as_str).filter(|t| !t.is_empty()).collect();
        // only the streaming architectures can bias the decoder
        if !terms.is_empty() && streaming_arch {
            opts.push((CString::new("keyterms")?, CString::new(terms.join(","))?));
        }
        // Only the final text is used: encode while audio arrives, but decode a line once
        // (when it is complete) instead of after every chunk.
        if streaming && std::env::var("MOONSHINE_DECODE_PARTIAL").is_err() {
            opts.push((CString::new("decode_incomplete_lines")?, CString::new("false")?));
        }
        let raw: Vec<MoonshineOption> =
            opts.iter().map(|(n, v)| MoonshineOption { name: n.as_ptr(), value: v.as_ptr() }).collect();
        let path = CString::new(dir)?;
        let handle = unsafe { load(path.as_ptr(), arch, raw.as_ptr(), raw.len() as u64, HEADER_VERSION) };
        if handle < 0 {
            let msg = unsafe { CStr::from_ptr((api.error)(handle)) }.to_string_lossy().into_owned();
            return Err(anyhow!("loading Moonshine model {dir}: {msg}"));
        }
        let streamer = if streaming {
            let stream = unsafe { (api.create_stream)(handle, 0) };
            if stream < 0 {
                unsafe { (api.free)(handle) };
                api.check(stream)?;
            }
            let (tx, rx) = mpsc::channel();
            let w = thread::Builder::new().name("moonshine".into()).spawn(move || worker(api, handle, stream, rx))?;
            Some(Streamer { tx: Some(tx), worker: Some(w), stream })
        } else {
            None
        };
        let version = unsafe { version() };
        let arch = match arch {
            ARCH_TINY => "tiny",
            ARCH_BASE => "base",
            ARCH_TINY_STREAMING => "tiny streaming",
            ARCH_SMALL_STREAMING => "small streaming",
            _ => "medium streaming",
        };
        Ok(Self { api, handle, streamer, version, arch, _lib: lib })
    }

    /// Streaming models: analyses audio while it is being spoken (`begin`, `push`, `finish`).
    pub fn streaming(&self) -> bool {
        self.streamer.is_some()
    }

    fn send(&self, msg: Msg) {
        if let Some(tx) = self.streamer.as_ref().and_then(|s| s.tx.as_ref()) {
            let _ = tx.send(msg);
        }
    }

    /// Starts a new utterance (drops anything unfinished).
    pub fn begin(&self) {
        self.send(Msg::Start);
    }

    /// Adds 16 kHz mono audio to the current utterance.
    pub fn push(&self, samples: &[i16]) {
        self.send(Msg::Audio(samples.iter().map(|&s| s as f32 / 32768.0).collect()));
    }

    /// Ends the utterance and returns its text (lowercase words, no punctuation).
    pub fn finish(&self) -> Result<String> {
        let (tx, rx) = mpsc::channel();
        self.send(Msg::Finish(tx));
        rx.recv().map_err(|_| anyhow!("Moonshine worker stopped"))?
    }

    /// Transcribes a whole recording at once to lowercase words without punctuation.
    pub fn transcribe(&mut self, samples: &[i16]) -> Result<String> {
        let mut pcm: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut out: *mut Transcript = std::ptr::null_mut();
        let rc = unsafe {
            (self.api.transcribe)(self.handle, pcm.as_mut_ptr(), pcm.len() as u64, SAMPLE_RATE as i32, 0, &mut out)
        };
        self.api.check(rc)?;
        // SAFETY: the transcript is owned by the transcriber and valid until the next call.
        Ok(unsafe { text_of(out) })
    }
}

impl Drop for MoonshineVoice {
    fn drop(&mut self) {
        if let Some(s) = &mut self.streamer {
            drop(s.tx.take()); // ends the worker
            if let Some(w) = s.worker.take() {
                let _ = w.join();
            }
            unsafe { (self.api.free_stream)(self.handle, s.stream) };
        }
        unsafe { (self.api.free)(self.handle) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_from_dir_name() {
        let dir = std::env::temp_dir().join(format!("moonshine-small-streaming-de-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("streaming_config.json"), "{}").unwrap();
        assert_eq!(arch(dir.to_str().unwrap()).unwrap(), ARCH_SMALL_STREAMING);
        std::fs::remove_dir_all(dir).ok();
        assert!(arch("/nowhere/some-model").is_err());
    }
}
