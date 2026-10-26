//! Moonshine speech-to-text via the official moonshine-voice runtime (libmoonshine.so,
//! loaded at run time). Its streaming models come in English and German; the command
//! is transcribed once it has ended.
//!
//! Library: the `moonshine-voice-linux-<arch>.tar.gz` release from
//! github.com/moonshine-ai/moonshine (libmoonshine.so + libonnxruntime.so.1 next to it).
//! Models: scripts/get-moonshine-model.sh.

use std::ffi::{CStr, CString, c_char};
use std::path::Path;

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

pub struct MoonshineVoice {
    handle: i32,
    transcribe: TranscribeFn,
    error: ErrorFn,
    free: FreeFn,
    /// Library version, e.g. 30000.
    pub version: i32,
    pub arch: &'static str,
    // keeps the function pointers above valid
    _lib: Library,
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

impl MoonshineVoice {
    /// `lib`: path of libmoonshine.so (or just the name, to use the library search path).
    /// `keyterms`: words to bias recognition towards (the wake word, command words).
    pub fn new(lib: &str, dir: &str, keyterms: &[String]) -> Result<Self> {
        let arch = arch(dir)?;
        // SAFETY: loading the moonshine library runs no unusual initialisers.
        let lib = unsafe { Library::new(lib) }.with_context(|| format!("loading {lib}"))?;
        let (load, transcribe, error, free, version) = unsafe {
            (
                *lib.get::<LoadFn>(b"moonshine_load_transcriber_from_files\0")?,
                *lib.get::<TranscribeFn>(b"moonshine_transcribe_without_streaming\0")?,
                *lib.get::<ErrorFn>(b"moonshine_error_to_string\0")?,
                *lib.get::<FreeFn>(b"moonshine_free_transcriber\0")?,
                *lib.get::<VersionFn>(b"moonshine_get_version\0")?,
            )
        };
        let mut opts: Vec<(CString, CString)> = Vec::new();
        let terms: Vec<&str> = keyterms.iter().map(String::as_str).filter(|t| !t.is_empty()).collect();
        // only the streaming architectures can bias the decoder
        if !terms.is_empty() && matches!(arch, ARCH_TINY_STREAMING | ARCH_SMALL_STREAMING | ARCH_MEDIUM_STREAMING) {
            opts.push((CString::new("keyterms")?, CString::new(terms.join(","))?));
        }
        let raw: Vec<MoonshineOption> =
            opts.iter().map(|(n, v)| MoonshineOption { name: n.as_ptr(), value: v.as_ptr() }).collect();
        let path = CString::new(dir)?;
        let handle = unsafe { load(path.as_ptr(), arch, raw.as_ptr(), raw.len() as u64, HEADER_VERSION) };
        if handle < 0 {
            let msg = unsafe { CStr::from_ptr(error(handle)) }.to_string_lossy().into_owned();
            return Err(anyhow!("loading Moonshine model {dir}: {msg}"));
        }
        let version = unsafe { version() };
        let arch = match arch {
            ARCH_TINY => "tiny",
            ARCH_BASE => "base",
            ARCH_TINY_STREAMING => "tiny streaming",
            ARCH_SMALL_STREAMING => "small streaming",
            _ => "medium streaming",
        };
        Ok(Self { handle, transcribe, error, free, version, arch, _lib: lib })
    }

    /// Transcribes 16 kHz mono audio to lowercase words without punctuation.
    pub fn transcribe(&mut self, samples: &[i16]) -> Result<String> {
        let mut pcm: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut out: *mut Transcript = std::ptr::null_mut();
        let rc = unsafe {
            (self.transcribe)(self.handle, pcm.as_mut_ptr(), pcm.len() as u64, SAMPLE_RATE as i32, 0, &mut out)
        };
        if rc != 0 {
            let msg = unsafe { CStr::from_ptr((self.error)(rc)) }.to_string_lossy().into_owned();
            bail!("Moonshine: {msg}");
        }
        if out.is_null() {
            return Ok(String::new());
        }
        // SAFETY: the transcript is owned by the transcriber and valid until the next call.
        let t = unsafe { &*out };
        let lines: &[Line] =
            if t.lines.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(t.lines, t.line_count as usize) } };
        let text: Vec<String> = lines
            .iter()
            .filter(|l| !l.text.is_null())
            .map(|l| unsafe { CStr::from_ptr(l.text) }.to_string_lossy().into_owned())
            .collect();
        Ok(words_only(&text.join(" ")))
    }
}

impl Drop for MoonshineVoice {
    fn drop(&mut self) {
        unsafe { (self.free)(self.handle) }
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
