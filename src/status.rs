//! Live status page: what was heard, which intent, what actions ran and what was said,
//! with the recorded command and the spoken reply to play back, and webcam photos.
//! A tiny HTTP server on std::net (the page polls /api/status). Besides reading, it can
//! take a photo, switch continuous transcription and change the Admin settings.

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::camera::Camera;
use crate::history::{History, Sample};
use serde::Serialize;

const PAGE: &str = include_str!("status.html");
const MAX_TURNS: usize = 100;
/// Turns whose audio is kept in memory for playback.
const MAX_AUDIO: usize = 20;
const MAX_PHOTOS: usize = 8;
/// Latency samples kept in memory when there is no database.
const MAX_SAMPLES: usize = 5000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    Starting,
    Idle,
    Listening,
    Thinking,
    Speaking,
    Error,
}

/// One interaction: a wake word + command, a timer going off, a greeting...
#[derive(Debug, Clone, Serialize)]
pub struct Turn {
    pub id: u64,
    /// Unix milliseconds.
    pub at: u64,
    /// "wake word", "timer", "presence"
    pub trigger: String,
    pub heard: Option<String>,
    /// Engine of `heard` ("moonshine", "grammar", ...).
    pub heard_engine: Option<String>,
    /// The other recognizer's transcript, if it differs.
    pub heard_alt: Option<String>,
    pub alt_engine: Option<String>,
    pub intent: Option<String>,
    pub actions: Vec<String>,
    pub said: Option<String>,
    /// Set when the turn ended without a command, e.g. "false wake ignored".
    pub outcome: Option<String>,
    pub duration_ms: Option<u64>,
    /// Audio available at /api/audio/<id>/heard.wav and reply.wav.
    pub heard_audio: bool,
    pub reply_audio: bool,
    /// Photo taken in this turn (file name, served at /api/photo/<name>).
    pub photo: Option<String>,
    /// How long each stage took, in milliseconds ("stt", "tts", "response", ...).
    pub latency: BTreeMap<String, f32>,
    #[serde(skip)]
    pub(crate) started: Instant,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimerInfo {
    pub name: String,
    pub due_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Info {
    pub wake_word: String,
    pub wake_engine: String,
    pub capture_device: String,
    pub playback_device: String,
    pub llm: bool,
    pub camera: bool,
}

#[derive(Serialize)]
struct Inner {
    state: State,
    error: Option<String>,
    /// Live transcript while listening.
    live: String,
    /// Mic level 0..1 (dBFS mapped from -60..0).
    level: f32,
    started_ms: u64,
    now_ms: u64,
    info: Info,
    timers: Vec<TimerInfo>,
    turns: VecDeque<Turn>,
    /// Newest photos first (file names).
    photos: Vec<String>,
    /// Continuous transcription instead of waiting for the wake word.
    continuous: bool,
    /// Build, engines and models (Info page); live figures are added per request.
    #[serde(skip)]
    runtime: serde_json::Value,
    /// Current settings and choices (Admin page).
    #[serde(skip)]
    admin: serde_json::Value,
    #[serde(skip)]
    samples: VecDeque<Sample>,
    #[serde(skip)]
    next_id: u64,
    /// (turn id, heard WAV, reply WAV), newest last.
    #[serde(skip)]
    audio: VecDeque<(u64, Option<Vec<u8>>, Option<Vec<u8>>)>,
}

#[derive(Clone)]
pub struct Status {
    inner: Arc<Mutex<Inner>>,
    /// To hand Admin changes to the main loop.
    control: Option<std::sync::mpsc::Sender<crate::Event>>,
    camera: Option<Arc<Camera>>,
    history: Option<Arc<History>>,
}

pub fn unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

impl Status {
    pub fn new(info: Info, continuous: bool, camera: Option<Arc<Camera>>, history: Option<Arc<History>>) -> Self {
        let photos = camera.as_ref().map(|c| list_photos(c.photo_dir())).unwrap_or_default();
        // Show the latest turns from the database after a restart.
        let turns: VecDeque<Turn> = history
            .as_ref()
            .and_then(|h| h.recent(None, MAX_TURNS).map_err(|e| log::warn!("history: {e:#}")).ok())
            .unwrap_or_default()
            .into();
        let inner = Arc::new(Mutex::new(Inner {
            state: State::Starting,
            error: None,
            live: String::new(),
            level: 0.0,
            started_ms: unix_ms(),
            now_ms: 0,
            info,
            timers: Vec::new(),
            turns,
            photos,
            continuous,
            runtime: serde_json::Value::Null,
            admin: serde_json::Value::Null,
            samples: VecDeque::new(),
            next_id: 1,
            audio: VecDeque::new(),
        }));
        Self { inner, control: None, camera, history }
    }

    /// Writes a turn's text fields to the database.
    fn persist(&self, id: u64) {
        let Some(h) = &self.history else { return };
        let turn = self.with(|i| i.turns.iter().find(|t| t.id == id).cloned());
        if let Some(t) = turn {
            if let Err(e) = h.save(&t) {
                log::warn!("history: {e:#}");
            }
        }
    }

    /// Turns from the database, newest first, before `before`.
    fn older(&self, before: Option<u64>, limit: usize) -> Vec<Turn> {
        self.history
            .as_ref()
            .and_then(|h| h.recent(before, limit).map_err(|e| log::warn!("history: {e:#}")).ok())
            .unwrap_or_default()
    }

    fn with<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        f(&mut self.inner.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Keeps the recorded command (`reply: false`) or the spoken reply for playback.
    pub fn set_audio(&self, id: u64, samples: &[i16], rate: u32, reply: bool) {
        if samples.is_empty() {
            return;
        }
        let wav = crate::audio::wav_bytes(samples, rate);
        if let Some(h) = &self.history {
            if let Err(e) = h.set_audio(id, reply, &wav) {
                log::warn!("history: {e:#}");
            }
        }
        let persisted = self.history.is_some();
        self.with(|i| {
            if !i.audio.iter().any(|(t, ..)| *t == id) {
                i.audio.push_back((id, None, None));
                while i.audio.len() > MAX_AUDIO {
                    let (old, ..) = i.audio.pop_front().unwrap();
                    // still in the database otherwise
                    if persisted {
                        continue;
                    }
                    if let Some(t) = i.turns.iter_mut().find(|t| t.id == old) {
                        t.heard_audio = false;
                        t.reply_audio = false;
                    }
                }
            }
            let entry = i.audio.iter_mut().find(|(t, ..)| *t == id).unwrap();
            if reply { entry.2 = Some(wav) } else { entry.1 = Some(wav) }
            if let Some(t) = i.turns.iter_mut().find(|t| t.id == id) {
                if reply { t.reply_audio = true } else { t.heard_audio = true }
            }
        });
    }

    /// Records a new photo (full path) and attaches it to a turn.
    pub fn add_photo(&self, id: Option<u64>, path: &str) {
        let name = path.rsplit('/').next().unwrap_or(path).to_string();
        self.with(|i| {
            i.photos.insert(0, name.clone());
            i.photos.truncate(MAX_PHOTOS);
            if let Some(t) = id.and_then(|id| i.turns.iter_mut().find(|t| t.id == id)) {
                t.photo = Some(name);
            }
        });
        if let Some(id) = id {
            self.persist(id);
        }
    }

    fn audio(&self, id: u64, reply: bool) -> Option<Vec<u8>> {
        let mem = self.with(|i| {
            let (_, heard, said) = i.audio.iter().find(|(t, ..)| *t == id)?;
            if reply { said.clone() } else { heard.clone() }
        });
        mem.or_else(|| self.history.as_ref()?.audio(id, reply))
    }

    pub fn set_state(&self, state: State) {
        self.with(|i| {
            i.state = state;
            if state != State::Error {
                i.error = None;
            }
            if state != State::Listening {
                i.live.clear();
            }
        });
    }

    pub fn set_error(&self, msg: &str) {
        self.with(|i| {
            i.state = State::Error;
            i.error = Some(msg.to_string());
        });
    }

    pub fn set_live(&self, text: &str) {
        self.with(|i| text.clone_into(&mut i.live));
    }

    pub fn set_level(&self, samples: &[i16]) {
        if samples.is_empty() {
            return;
        }
        let rms = (samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
        let db = 20.0 * (rms.max(1.0) / 32768.0).log10();
        self.with(|i| i.level = ((db + 60.0) / 60.0).clamp(0.0, 1.0) as f32);
    }

    pub fn set_timers(&self, timers: Vec<TimerInfo>) {
        self.with(|i| i.timers = timers);
    }

    /// Starts a new turn and returns its id.
    pub fn begin(&self, trigger: &str) -> u64 {
        // With a database its row id is the turn id, so ids stay unique across restarts.
        let db_id = self.history.as_ref().and_then(|h| {
            h.insert(unix_ms(), trigger).map_err(|e| log::warn!("history: {e:#}")).ok()
        });
        self.with(|i| {
            let id = db_id.unwrap_or(i.next_id);
            i.next_id = id + 1;
            i.turns.push_front(Turn {
                id,
                at: unix_ms(),
                trigger: trigger.into(),
                heard: None,
                heard_engine: None,
                heard_alt: None,
                alt_engine: None,
                intent: None,
                actions: Vec::new(),
                said: None,
                outcome: None,
                duration_ms: None,
                heard_audio: false,
                reply_audio: false,
                photo: None,
                latency: BTreeMap::new(),
                started: Instant::now(),
            });
            i.turns.truncate(MAX_TURNS);
            id
        })
    }

    pub fn update(&self, id: u64, f: impl FnOnce(&mut Turn)) {
        self.with(|i| {
            if let Some(t) = i.turns.iter_mut().find(|t| t.id == id) {
                f(t);
            }
        });
        self.persist(id);
    }

    /// Records how long a stage took (`turn` None for periodic measurements).
    pub fn latency(&self, turn: Option<u64>, stage: &str, ms: f32) {
        let sample = Sample { turn, at: unix_ms(), stage: stage.into(), ms };
        if let Some(h) = &self.history {
            if let Err(e) = h.add_latency(&sample) {
                log::warn!("history: {e:#}");
            }
        }
        self.with(|i| {
            if let Some(t) = turn.and_then(|id| i.turns.iter_mut().find(|t| t.id == id)) {
                t.latency.insert(stage.into(), ms);
            }
            i.samples.push_back(sample);
            if i.samples.len() > MAX_SAMPLES {
                i.samples.pop_front();
            }
        });
    }

    /// Latency samples of the last `hours`.
    fn samples(&self, hours: u64) -> Vec<Sample> {
        let since = unix_ms().saturating_sub(hours * 3_600_000);
        if let Some(h) = &self.history {
            match h.latency_since(since) {
                Ok(s) => return s,
                Err(e) => log::warn!("history: {e:#}"),
            }
        }
        self.with(|i| i.samples.iter().filter(|s| s.at >= since).cloned().collect())
    }

    /// Lets the Admin page send settings to the main loop.
    pub fn set_control(&mut self, tx: std::sync::mpsc::Sender<crate::Event>) {
        self.control = Some(tx);
    }

    pub fn set_runtime(&self, runtime: serde_json::Value) {
        self.with(|i| i.runtime = runtime);
    }

    pub fn set_admin(&self, admin: serde_json::Value) {
        self.with(|i| i.admin = admin);
    }

    pub fn set_wake_word(&self, word: &str) {
        self.with(|i| word.clone_into(&mut i.info.wake_word));
    }

    /// Everything for the Info page: build/engines plus live system figures.
    fn info(&self) -> serde_json::Value {
        let mut v = self.with(|i| i.runtime.clone());
        v["system"] = crate::info::system();
        #[cfg(feature = "llm")]
        if let Some(url) = v["llm"]["url"].as_str().filter(|_| v["llm"]["enabled"] == true) {
            v["llm"]["server"] = crate::llm::server_info(url);
        }
        if let (Some(h), Some(path)) = (&self.history, v["database"]["path"].as_str()) {
            v["database"] = h.stats(path);
        }
        v
    }

    /// Hands new settings to the main loop and waits until they are applied.
    fn apply(&self, body: &[u8]) -> Result<serde_json::Value, String> {
        let settings: crate::admin::Settings = serde_json::from_slice(body).map_err(|e| format!("bad request: {e}"))?;
        let tx = self.control.as_ref().ok_or("settings cannot be changed in this mode")?;
        let (reply_tx, reply) = std::sync::mpsc::channel();
        tx.send(crate::Event::Apply(Box::new(settings), reply_tx)).map_err(|_| "assistant not running")?;
        reply.recv_timeout(Duration::from_secs(120)).map_err(|_| "timed out applying the settings".to_string())?
    }

    pub fn continuous(&self) -> bool {
        self.with(|i| i.continuous)
    }

    pub fn set_continuous(&self, on: bool) {
        self.with(|i| i.continuous = on);
    }

    pub fn finish(&self, id: u64) {
        self.update(id, |t| t.duration_ms = Some(t.started.elapsed().as_millis() as u64));
    }

    fn json(&self) -> String {
        self.with(|i| {
            i.now_ms = unix_ms();
            serde_json::to_string(i).unwrap_or_else(|_| "{}".into())
        })
    }

    /// Serves the page on `addr` (e.g. "0.0.0.0:8088") in a background thread.
    pub fn serve(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).with_context(|| format!("status page: binding {addr}"))?;
        log::info!("status page on http://{addr}/");
        let status = self.clone();
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let status = status.clone();
                thread::spawn(move || {
                    if let Err(e) = handle(stream, &status) {
                        log::debug!("status page: {e}");
                    }
                });
            }
        });
        Ok(())
    }
}

fn handle(mut stream: TcpStream, status: &Status) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut buf = vec![0u8; 4096];
    let mut len = 0;
    let header_end = loop {
        let n = stream.read(&mut buf[len..])?;
        len += n;
        if let Some(i) = buf[..len].windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if n == 0 || len == buf.len() {
            break len;
        }
    };
    let req = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    // request body (Admin settings): small JSON only
    let body_len = req
        .lines()
        .find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("content-length")).map(|(_, v)| v.trim().parse::<usize>().unwrap_or(0)))
        .unwrap_or(0)
        .min(64 * 1024);
    let mut body = buf[header_end..len].to_vec();
    while body.len() < body_len {
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(body_len);
    let mut first = req.split_whitespace();
    let method = first.next().unwrap_or("GET").to_string();
    let path = first.next().unwrap_or("/");
    let (path, query) = path.split_once('?').unwrap_or((path, ""));
    let path = path.to_string();
    let param = |k: &str| query.split('&').find_map(|kv| kv.strip_prefix(k)?.strip_prefix('=')).map(str::to_string);
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    let not_found = || ("404 Not Found", "text/plain", b"not found".to_vec());
    let (code, ctype, body): (&str, &str, Vec<u8>) = match (method.as_str(), parts.as_slice()) {
        ("GET", [""] | ["index.html"]) => ("200 OK", "text/html; charset=utf-8", PAGE.as_bytes().to_vec()),
        ("GET", ["api", "status"]) => ("200 OK", "application/json", status.json().into_bytes()),
        ("GET", ["api", "history"]) => {
            let before = param("before").and_then(|b| b.parse().ok());
            let limit = param("limit").and_then(|l| l.parse().ok()).unwrap_or(30usize).min(200);
            let turns = status.older(before, limit);
            ("200 OK", "application/json", serde_json::to_string(&turns).unwrap_or_default().into_bytes())
        }
        ("GET", ["api", "stats"]) => {
            let hours = param("hours").and_then(|h| h.parse().ok()).unwrap_or(24u64).clamp(1, 24 * 90);
            // compact: [at_ms, stage, ms, turn]
            let rows: Vec<_> =
                status.samples(hours).into_iter().map(|s| serde_json::json!([s.at, s.stage, s.ms, s.turn])).collect();
            ("200 OK", "application/json", serde_json::to_string(&rows).unwrap_or_default().into_bytes())
        }
        ("GET", ["api", "info"]) => ("200 OK", "application/json", status.info().to_string().into_bytes()),
        ("GET", ["api", "admin"]) => ("200 OK", "application/json", status.with(|i| i.admin.clone()).to_string().into_bytes()),
        ("POST", ["api", "admin"]) => match status.apply(&body) {
            Ok(v) => ("200 OK", "application/json", v.to_string().into_bytes()),
            Err(e) => ("400 Bad Request", "text/plain; charset=utf-8", e.into_bytes()),
        },
        ("POST", ["api", "mode"]) => match param("continuous").as_deref() {
            Some(v @ ("0" | "1")) => {
                status.set_continuous(v == "1");
                ("200 OK", "application/json", serde_json::json!({ "continuous": v == "1" }).to_string().into_bytes())
            }
            _ => ("400 Bad Request", "text/plain", b"continuous=0|1".to_vec()),
        },
        ("GET", ["api", "audio", id, file @ ("heard.wav" | "reply.wav")]) => {
            match id.parse().ok().and_then(|id| status.audio(id, *file == "reply.wav")) {
                Some(wav) => ("200 OK", "audio/wav", wav),
                None => not_found(),
            }
        }
        ("GET", ["api", "photo", name]) => match (&status.camera, safe_name(name)) {
            (Some(cam), true) => match std::fs::read(format!("{}/{name}", cam.photo_dir())) {
                Ok(jpg) => ("200 OK", "image/jpeg", jpg),
                Err(_) => not_found(),
            },
            _ => not_found(),
        },
        ("POST", ["api", "photo"]) => match &status.camera {
            None => ("409 Conflict", "text/plain", b"camera disabled".to_vec()),
            Some(cam) => match cam.snapshot() {
                Ok(path) => {
                    status.add_photo(None, &path);
                    ("200 OK", "application/json", serde_json::json!({ "ok": true }).to_string().into_bytes())
                }
                Err(e) => ("500 Internal Server Error", "text/plain", format!("{e:#}").into_bytes()),
            },
        },
        _ => not_found(),
    };
    write!(
        stream,
        "HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)
}

/// Photo file names only: no paths.
fn safe_name(name: &str) -> bool {
    name.ends_with(".jpg") && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) && !name.starts_with('.')
}

fn list_photos(dir: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map(|d| d.flatten().filter_map(|e| e.file_name().into_string().ok()).filter(|n| safe_name(n)).collect())
        .unwrap_or_default();
    names.sort_unstable_by(|a, b| b.cmp(a));
    names.truncate(MAX_PHOTOS);
    names
}
