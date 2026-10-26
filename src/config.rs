use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Language of commands and replies. Must match the Vosk model and the Piper voice.
    pub language: Lang,
    pub wake: WakeConfig,
    pub audio: AudioConfig,
    pub stt: SttConfig,
    pub tts: TtsConfig,
    pub camera: CameraConfig,
    pub llm: LlmConfig,
    pub status: StatusConfig,
    pub history: HistoryConfig,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            log::warn!("config {} not found, using defaults", path.display());
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    #[default]
    En,
    De,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WakeEngine {
    /// Grammar-restricted Vosk recognizer: any word in the model vocabulary, no training.
    Vosk,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WakeConfig {
    /// Wake word (or short phrase). Lowercase, must exist in the Vosk vocabulary.
    pub word: String,
    pub engine: WakeEngine,
    /// Second stage: only accept the wake if the full recognizer also heard the wake word.
    /// Cuts false triggers ("commuter", TV) a lot.
    pub verify: bool,
    /// "word": wait for the wake word. "continuous": no wake word, transcribe everything
    /// that is said (switchable on the status page).
    pub mode: WakeMode,
    /// Continuous mode: what to do with what was heard.
    pub continuous_act: ContinuousAct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WakeMode {
    Word,
    Continuous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ContinuousAct {
    /// Only transcribe.
    None,
    /// Run recognized commands ("what time is it"); everything else is only transcribed,
    /// unless it was addressed with the wake word (then it also goes to the LLM).
    Commands,
    /// Treat every utterance like a command after the wake word (LLM included).
    All,
}

impl Default for WakeConfig {
    fn default() -> Self {
        Self {
            word: "computer".into(),
            engine: WakeEngine::Vosk,
            verify: true,
            mode: WakeMode::Word,
            continuous_act: ContinuousAct::Commands,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// ALSA capture device (see `arecord -L`), e.g. the webcam microphone.
    pub capture_device: String,
    /// ALSA playback device (see `aplay -L`), e.g. the USB speaker.
    pub playback_device: String,
    /// ALSA card + simple mixer control used for volume commands (see `amixer -c N scontrols`).
    pub mixer_card: String,
    pub mixer_control: String,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            capture_device: "default".into(),
            playback_device: "default".into(),
            mixer_card: "0".into(),
            mixer_control: "PCM".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FreeEngine {
    Vosk,
    /// Moonshine: English and German (moonshine-voice) or English (sherpa-onnx).
    Moonshine,
    /// whisper.cpp via whisper-rs; multilingual (German too), slower.
    Whisper,
    /// Whisper in pure Rust (candle-transformers); multilingual.
    Candle,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    pub model_path: String,
    /// Restrict Vosk to the phrases the skills understand: much more accurate with a
    /// far-away or noisy microphone. Vosk always does the wake word and end-of-command
    /// detection; with the grammar it is also the fallback for commands.
    pub grammar: bool,
    /// The command is complete once the transcript hasn't changed for this long.
    pub end_silence_secs: f32,
    /// With grammar + LLM: trust the free transcript over the grammar one from this mean
    /// word confidence on (noise makes the free one garbage with ~0.7).
    pub free_trust_conf: f32,
    /// Main speech-to-text engine. "vosk": the grammar recognizer decides, a free Vosk
    /// recognizer only runs for LLM questions. "moonshine" / "whisper" / "candle": that
    /// engine transcribes every command and decides; the Vosk grammar result is the
    /// fallback when its words agree (noise).
    #[serde(alias = "free_engine")]
    pub engine: FreeEngine,
    /// Moonshine model directory: a moonshine-voice model (English or German, see
    /// scripts/get-moonshine-model.sh) or an unpacked sherpa-onnx one (English).
    pub moonshine_dir: String,
    /// sherpa-onnx models only.
    pub moonshine_threads: i32,
    /// libmoonshine.so of the moonshine-voice release (for moonshine-voice models).
    /// Empty: lib/moonshine/libmoonshine.so next to the program's bin directory.
    pub moonshine_lib: String,
    /// Words to bias Moonshine towards, besides the wake word (moonshine-voice models).
    pub moonshine_keyterms: Vec<String>,
    /// whisper.cpp ggml model (multilingual, e.g. ggml-base-q5_1.bin).
    pub whisper_model: String,
    pub whisper_threads: i32,
    /// Whisper engines: encode only the spoken audio, not a full 30 s window. Faster but
    /// noticeably less accurate in tests, so off by default.
    pub whisper_short_window: bool,
    /// Give the main engine only the audio after the wake word (Vosk word timestamps).
    /// Off until tested with real voices: with synthetic test speech it helped as often as
    /// it hurt.
    pub cut_wake_word: bool,
    /// Save every command's audio + transcripts here (WAV + TXT) to tune recognition with
    /// real voices. Empty = off.
    pub record_dir: String,
    /// candle Whisper model directory (config*.json, tokenizer*.json, *.safetensors or *.gguf).
    pub candle_model_dir: String,
    /// Give up if nothing is said within this many seconds after the wake word.
    pub initial_silence_secs: f32,
    /// Hard cap on a single command.
    pub max_command_secs: f32,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            model_path: "/opt/uno-assistant/models/vosk-model-small-en-us-0.15".into(),
            grammar: true,
            end_silence_secs: 0.7,
            free_trust_conf: 0.9,
            engine: FreeEngine::Moonshine,
            moonshine_dir: "/opt/uno-assistant/models/moonshine-small-streaming-en".into(),
            moonshine_threads: 2,
            moonshine_lib: String::new(),
            moonshine_keyterms: Vec::new(),
            whisper_model: "/opt/uno-assistant/models/ggml-base-q5_1.bin".into(),
            whisper_threads: 4,
            whisper_short_window: false,
            record_dir: String::new(),
            cut_wake_word: false,
            candle_model_dir: "/opt/uno-assistant/models/candle-whisper-tiny".into(),
            initial_silence_secs: 5.0,
            max_command_secs: 10.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsEngine {
    Piper,
    Espeak,
    /// Only log the reply (handy for testing on a PC).
    Print,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    pub engine: TtsEngine,
    pub piper_bin: String,
    pub piper_model: String,
    /// Must match `audio.sample_rate` in the voice's .onnx.json.
    pub piper_sample_rate: u32,
    pub espeak_voice: String,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            engine: TtsEngine::Piper,
            piper_bin: "/opt/uno-assistant/piper/piper".into(),
            piper_model: "/opt/uno-assistant/models/en_US-lessac-low.onnx".into(),
            piper_sample_rate: 16000,
            espeak_voice: "en-us".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub enabled: bool,
    pub device: String,
    pub photo_dir: String,
    /// Run low-res motion detection to know whether someone is in the room.
    pub presence: bool,
    /// Mean absolute pixel difference (0-255) between frames that counts as motion.
    pub motion_threshold: f32,
    /// Greet when someone shows up after the room was empty this long. 0 disables greeting.
    pub greet_after_absent_mins: u64,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device: "/dev/video0".into(),
            photo_dir: "/var/lib/uno-assistant/photos".into(),
            presence: true,
            motion_threshold: 8.0,
            greet_after_absent_mins: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub enabled: bool,
    /// OpenAI-compatible endpoint of a local `llama-server`.
    pub url: String,
    pub max_tokens: u32,
    pub timeout_secs: u64,
    pub system_prompt: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            max_tokens: 80,
            timeout_secs: 30,
            system_prompt: "You are Computer, a voice assistant running offline on a small \
                board. Your answers are spoken aloud: reply in one or two short plain \
                sentences, no lists, no markdown."
                .into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct StatusConfig {
    /// Web page showing what was heard, done and said.
    pub enabled: bool,
    pub listen: String,
}

impl Default for StatusConfig {
    fn default() -> Self {
        Self { enabled: true, listen: "0.0.0.0:8088".into() }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct HistoryConfig {
    /// Keep every turn in an SQLite database (shown on the status page after restarts).
    pub enabled: bool,
    pub db_path: String,
    /// Keep the recorded command and reply audio of this many most recent turns.
    pub keep_audio: u64,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self { enabled: true, db_path: "/var/lib/uno-assistant/history.db".into(), keep_audio: 200 }
    }
}
