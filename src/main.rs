//! Offline "Alexa-like" voice assistant for the Arduino UNO Q.
//!
//! mic (arecord) -> wake word ("computer") -> Vosk command transcription -> intent
//! -> skill -> Piper TTS (aplay). Webcam via ffmpeg for photos and presence.

mod admin;
mod audio;
mod camera;
mod config;
mod history;
mod info;
mod intents;
#[cfg(feature = "llm")]
mod llm;
#[cfg(feature = "moonshine")]
mod moonshine;
#[cfg(feature = "moonshine-voice")]
mod moonshine_voice;
#[cfg(feature = "whisper")]
mod whisper;
#[cfg(feature = "candle")]
mod candle_whisper;
mod skills;
mod speech;
mod status;
mod tts;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use config::{Config, ContinuousAct, TtsEngine};
use speech::{CommandListener, Heard};
use status::{State, Status};

const PREROLL_CHUNKS: usize = 15;

/// A command being listened to.
struct Listen {
    /// Status page turn; continuous mode creates it only once words were heard.
    turn: Option<u64>,
    continuous: bool,
    started: Instant,
}

pub enum Event {
    /// 100 ms of 16 kHz mono audio and when it arrived.
    Audio(Instant, Vec<i16>),
    MicStopped,
    PersonArrived,
    /// New settings from the Admin page; the reply says how it went.
    Apply(Box<admin::Settings>, mpsc::Sender<Result<serde_json::Value, String>>),
}

struct Args {
    config: PathBuf,
    wav: Option<String>,
    text: Option<String>,
    print: bool,
    hold: bool,
    /// Only transcribe these WAV files with the main engine (benchmark).
    transcribe: Vec<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        config: PathBuf::from("/etc/uno-assistant/config.toml"),
        wav: None,
        text: None,
        print: false,
        hold: false,
        transcribe: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-c" | "--config" => a.config = it.next().unwrap_or_default().into(),
            "--wav" => a.wav = it.next(),
            "--text" => a.text = it.next(),
            "--print" => a.print = true,
            "--hold" => a.hold = true,
            "--transcribe" => a.transcribe.extend(it.by_ref()),
            "-h" | "--help" => {
                println!(
                    "usage: uno-assistant [--config FILE] [--print] [--wav FILE.wav | --text \"COMMAND\"]\n\
                     \n  --config  config file (default /etc/uno-assistant/config.toml)\
                     \n  --print   print replies instead of speaking\
                     \n  --wav     use a 16 kHz mono WAV instead of the microphone\
                     \n  --hold    keep running (status page) after the WAV ended\
                     \n  --text    run a single command without audio, e.g. --text \"what time is it\"\
                     \n  --transcribe FILE.wav...  transcribe files with the main engine and time it"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other} (try --help)"),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = parse_args()?;
    let mut cfg = Config::load(&args.config)?;
    if args.print {
        cfg.tts.engine = TtsEngine::Print;
    }
    if !args.transcribe.is_empty() {
        let mut model = speech::load_model(&cfg.stt.model_path)?;
        return speech::transcribe_files(&mut model, &cfg, &args.transcribe);
    }

    let (tx, rx) = mpsc::channel();

    let camera = cfg
        .camera
        .enabled
        .then(|| std::sync::Arc::new(camera::Camera::new(cfg.camera.clone(), tx.clone())));
    let mut skills = skills::Skills::new(&cfg, camera.clone());
    let tts = tts::Tts::new(&cfg);

    if let Some(text) = args.text {
        let intent = intents::parse(&text, cfg.language);
        log::info!("intent: {intent:?}");
        if let Some(reply) = skills.handle(intent) {
            tts.say(&reply);
        }
        for a in skills.take_actions() {
            log::info!("action: {a}");
        }
        return Ok(());
    }

    let history = cfg.history.enabled.then(|| {
        history::History::open(&cfg.history.db_path, cfg.history.keep_audio)
            .map(std::sync::Arc::new)
            .map_err(|e| log::error!("history disabled: {e:#}"))
            .ok()
    });
    let history = history.flatten();
    let mut status = Status::new(status::Info {
        wake_word: cfg.wake.word.clone(),
        wake_engine: format!("{:?}", cfg.wake.engine).to_lowercase(),
        capture_device: if args.wav.is_some() { "WAV file".into() } else { cfg.audio.capture_device.clone() },
        playback_device: cfg.audio.playback_device.clone(),
        llm: cfg.llm.enabled,
        camera: cfg.camera.enabled,
    }, cfg.wake.mode == config::WakeMode::Continuous, camera.clone(), history);
    status.set_control(tx.clone());
    if cfg.status.enabled {
        if let Err(e) = status.serve(&cfg.status.listen) {
            log::error!("{e:#}");
        }
    }

    log::info!("loading speech model {}", cfg.stt.model_path);
    let mut model = speech::load_model(&cfg.stt.model_path)?;
    let mut wake = speech::wake_detector(&cfg, &mut model)?;
    // Moonshine/Whisper as main engine always run; a free Vosk recognizer only for the LLM.
    let second_transcriber = cfg.llm.enabled || cfg.stt.engine != config::FreeEngine::Vosk;
    let phrases = intents::grammar(cfg.language);
    let phrases: Vec<&str> = phrases.iter().map(String::as_str).collect();
    let mut listener =
        CommandListener::new(&mut model, &cfg, cfg.stt.grammar.then_some(&phrases[..]), second_transcriber)?;
    // The configuration as changed on the Admin page.
    let mut cur = cfg.clone();
    status.set_runtime(runtime_info(&cur, &listener, &args.config));
    status.set_admin(admin_info(&cur, &args.config));

    let start_mic = |tx| match &args.wav {
        Some(path) => audio::spawn_wav(path, tx),
        None => audio::spawn_mic(&cfg.audio.capture_device, tx),
    };
    start_mic(tx.clone())?;

    let out = cfg.audio.playback_device.clone();
    let mut listening: Option<Listen> = None;
    // The last ~1.5 s of audio, replayed into the command recognizer on wake.
    let mut preroll: VecDeque<Vec<i16>> = VecDeque::with_capacity(PREROLL_CHUNKS + 1);
    // Ignore audio that was recorded while we were talking ourselves.
    let mut mute_until = Instant::now();
    // Background level while idle (dBFS): falls quickly, rises slowly, ignores speech.
    let mut noise_db: f32 = -50.0;
    // Continuous mode: consecutive loud chunks.
    let mut voice_run = 0;
    let mut was_continuous = status.continuous();
    // Wake detector load: compute time per audio processed, logged every minute.
    let (mut wake_cpu, mut wake_samples) = (Duration::ZERO, 0usize);
    // Speaks and returns when the first audio came out.
    let say = |turn: u64, text: &str| -> Instant {
        status.update(turn, |t| t.said = Some(text.to_string()));
        status.set_state(State::Speaking);
        let start = Instant::now();
        // No audio (printing only, or TTS failed): the reply is out now.
        let Some(speech) = tts.say(text) else { return start };
        status.latency(Some(turn), "tts", speech.first_ms);
        status.set_audio(turn, &speech.samples, speech.rate, true);
        start + Duration::from_secs_f32(speech.first_ms / 1000.0)
    };
    status.set_state(State::Idle);
    if was_continuous {
        log::info!("ready, transcribing continuously");
    } else {
        log::info!("ready, say \"{}\"", cfg.wake.word);
    }

    loop {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(Event::Audio(at, samples)) if at >= mute_until => {
                status.set_level(&samples);
                let continuous = status.continuous();
                if continuous != was_continuous {
                    log::info!("continuous transcription {}", if continuous { "on" } else { "off" });
                    was_continuous = continuous;
                    wake.reset();
                    voice_run = 0;
                }
                if listening.is_none() {
                    let db = speech::level_db(&samples);
                    if db < noise_db {
                        noise_db += 0.2 * (db - noise_db);
                    } else if db < noise_db + 10.0 {
                        noise_db += 0.01 * (db - noise_db);
                    }
                }
                if listening.is_none() {
                    let hit = if continuous {
                        // Voice activity instead of the wake word.
                        voice_run = if speech::level_db(&samples) > noise_db + 10.0 { voice_run + 1 } else { 0 };
                        voice_run >= 2
                    } else {
                        let t = Instant::now();
                        let hit = wake.feed(&samples);
                        wake_cpu += t.elapsed();
                        wake_samples += samples.len();
                        if wake_samples >= 60 * audio::SAMPLE_RATE as usize {
                            // ms of compute per second of audio
                            let load = wake_cpu.as_secs_f32() * 1000.0 / (wake_samples as f32 / audio::SAMPLE_RATE as f32);
                            status.latency(None, "wake", load);
                            (wake_cpu, wake_samples) = (Duration::ZERO, 0);
                        }
                        hit
                    };
                    preroll.push_back(samples);
                    if preroll.len() > PREROLL_CHUNKS {
                        preroll.pop_front();
                    }
                    if hit {
                        let turn = if continuous {
                            voice_run = 0;
                            None
                        } else {
                            log::info!("wake word detected");
                            audio::play(&out, audio::wake_chime(), false);
                            Some(status.begin("wake word"))
                        };
                        listener.start(preroll.iter().map(|c| c.as_slice()), noise_db, continuous);
                        preroll.clear();
                        listening = Some(Listen { turn, continuous, started: Instant::now() });
                        status.set_state(State::Listening);
                    }
                    continue;
                }
                let heard = listener.feed(&samples);
                status.set_live(&listener.partial);
                if matches!(heard, Heard::Listening) {
                    continue;
                }
                let listen = listening.take().expect("listening");
                let text = match heard {
                    Heard::Listening => None,
                    Heard::Nothing => {
                        if let Some(turn) = listen.turn {
                            log::info!("heard nothing");
                            status.update(turn, |t| t.outcome = Some("nothing heard".into()));
                            audio::play(&out, audio::done_chime(), true);
                        }
                        None
                    }
                    Heard::FalseWake(t) => {
                        log::info!("false wake ignored ({t:?})");
                        if let Some(turn) = listen.turn {
                            status.update(turn, |t| t.outcome = Some("false wake ignored".into()));
                        }
                        None
                    }
                    Heard::Text(t) => Some(t),
                };
                if let Some(cmd) = text {
                    // With Moonshine/Whisper as main engine, their transcript is "what was heard".
                    let main_is_free = cmd.free_kind != speech::FreeKind::Vosk;
                    let engine = format!("{:?}", cmd.free_kind).to_lowercase();
                    let (heard, heard_alt, main_name, alt_name) = if main_is_free {
                        (cmd.free.clone().unwrap_or_default(), Some(cmd.text.clone()), engine, "grammar".to_string())
                    } else {
                        (cmd.text.clone(), cmd.free.clone(), "grammar".to_string(), engine)
                    };
                    log::info!(
                        "< {heard} [{main_name}]{}",
                        heard_alt.as_ref().map(|a| format!("  ({alt_name}: {a})")).unwrap_or_default()
                    );
                    log::debug!("grammar conf {:.2}{}, free conf {:.2}", cmd.conf, if cmd.unk { ", unk" } else { "" }, cmd.free_conf);
                    // Continuous mode: noise that turned out to be no words leaves no trace.
                    let turn = match listen.turn {
                        Some(turn) => Some(turn),
                        None if heard.trim().is_empty() => None,
                        None => Some(status.begin("continuous")),
                    };
                    if let Some(turn) = turn {
                        status.latency(Some(turn), "endpoint", cmd.endpoint_ms);
                        if cmd.stt_ms > 0.0 {
                            status.latency(Some(turn), "stt", cmd.stt_ms);
                        }
                        status.set_state(State::Thinking);
                        let t = Instant::now();
                        let intent = choose_intent(&cmd, cfg.language, cfg.stt.free_trust_conf);
                        status.latency(Some(turn), "intent", t.elapsed().as_secs_f32() * 1000.0);
                        log::debug!("intent: {intent:?}");
                        if !cfg.stt.record_dir.is_empty() {
                            record_command(&cfg.stt.record_dir, &cmd, &heard, &format!("{intent:?}"));
                        }
                        status.set_audio(turn, &cmd.audio, audio::SAMPLE_RATE, false);
                        status.update(turn, |t| {
                            t.heard = Some(heard.clone());
                            t.heard_engine = Some(main_name.clone());
                            t.heard_alt = heard_alt.clone().filter(|a| *a != heard);
                            t.alt_engine = Some(alt_name.clone());
                            t.intent = Some(format!("{intent:?}"));
                        });
                        let unknown = matches!(intent, intents::Intent::Unknown(_));
                        let act = !listen.continuous
                            || match cur.wake.continuous_act {
                                ContinuousAct::None => false,
                                ContinuousAct::Commands => cmd.addressed || !unknown,
                                ContinuousAct::All => true,
                            };
                        if act {
                            if skills.is_slow(&intent) {
                                audio::play(&out, audio::tones(&[(660.0, 0.06)]), false);
                            }
                            let stage = match intent {
                                intents::Intent::Unknown(_) if cfg.llm.enabled => "llm",
                                intents::Intent::TakePhoto => "camera",
                                _ => "action",
                            };
                            let t = Instant::now();
                            let reply = skills.handle(intent);
                            status.latency(Some(turn), stage, t.elapsed().as_secs_f32() * 1000.0);
                            if let Some(path) = skills.last_photo.take() {
                                status.add_photo(Some(turn), &path);
                            }
                            let actions = skills.take_actions();
                            status.update(turn, |t| t.actions = actions);
                            status.set_timers(skills.timer_infos());
                            let first_sound = match reply {
                                Some(reply) => say(turn, &reply),
                                None => {
                                    let t = Instant::now();
                                    audio::play(&out, audio::done_chime(), true);
                                    t
                                }
                            };
                            // From the end of speech until we answer.
                            let wait = first_sound.saturating_duration_since(cmd.ended).as_secs_f32() * 1000.0;
                            status.latency(Some(turn), "response", cmd.endpoint_ms + wait);
                        } else {
                            status.update(turn, |t| t.outcome = Some("transcribed".into()));
                        }
                        status.latency(Some(turn), "turn", listen.started.elapsed().as_secs_f32() * 1000.0);
                        status.finish(turn);
                    }
                }
                status.set_state(State::Idle);
                wake.reset();
                mute_until = Instant::now() + Duration::from_millis(250);
            }
            Ok(Event::Audio(..)) => {}
            Ok(Event::Apply(settings, reply)) => {
                let result = (|| -> Result<serde_json::Value> {
                    let new = settings.apply_to(&cur)?;
                    let rebuild = new.wake.word != cur.wake.word || new.stt.moonshine_dir != cur.stt.moonshine_dir;
                    if rebuild {
                        // the wake detector also checks the word against the vocabulary
                        let new_wake = speech::wake_detector(&new, &mut model)?;
                        let new_listener =
                            CommandListener::new(&mut model, &new, new.stt.grammar.then_some(&phrases[..]), second_transcriber)?;
                        wake = new_wake;
                        listener = new_listener;
                    }
                    admin::save(&args.config, &admin::Settings::from_config(&new))?;
                    let engine = listener.engine_info(&new);
                    let wanted = std::path::Path::new(&new.stt.moonshine_dir).file_name().map(|n| n.to_string_lossy().into_owned());
                    let warning = (new.stt.engine == config::FreeEngine::Moonshine
                        && engine["model"]["name"].as_str().map(str::to_string) != wanted)
                        .then_some("the speech model could not be loaded, using Vosk instead (see the log)");
                    cur = new;
                    Ok(serde_json::json!({ "ok": true, "engine": engine, "warning": warning }))
                })();
                if result.is_ok() {
                    log::info!("settings changed: wake word \"{}\", {:?} mode, model {}", cur.wake.word, cur.wake.mode, cur.stt.moonshine_dir);
                    status.set_wake_word(&cur.wake.word);
                    status.set_continuous(cur.wake.mode == config::WakeMode::Continuous);
                    status.set_runtime(runtime_info(&cur, &listener, &args.config));
                    status.set_admin(admin_info(&cur, &args.config));
                    // a command in progress belonged to the old recognizers
                    if let Some(l) = listening.take() {
                        if let Some(turn) = l.turn {
                            status.update(turn, |t| t.outcome = Some("settings changed".into()));
                            status.finish(turn);
                        }
                    }
                    preroll.clear();
                    wake.reset();
                    status.set_state(State::Idle);
                }
                let _ = reply.send(result.map_err(|e| format!("{e:#}")));
            }
            Ok(Event::PersonArrived) => {
                if listening.is_none() {
                    let turn = status.begin("presence");
                    say(turn, &skills.welcome_text());
                    status.finish(turn);
                    status.set_state(State::Idle);
                    mute_until = Instant::now() + Duration::from_millis(250);
                }
            }
            Ok(Event::MicStopped) => {
                if args.wav.is_some() {
                    if !args.hold {
                        return Ok(());
                    }
                    log::info!("WAV finished, holding (Ctrl+C to quit)");
                    status.set_level(&[0]);
                    continue;
                }
                // Keep running so the status page can show the problem; retry the mic.
                let mut msg = format!("microphone stream stopped (check audio.capture_device = \"{}\")", cfg.audio.capture_device);
                if info::usb_role().as_deref() == Some("device") {
                    // seen after a reboot: then no USB device (webcam, hub) is powered at all
                    msg.push_str(". The USB-C port is in device mode, so the hub and webcam are not powered: check the hub's power supply and replug it");
                }
                log::error!("{msg}, retrying in 10 s");
                status.set_error(&msg);
                listening = None;
                thread::sleep(Duration::from_secs(10));
                match start_mic(tx.clone()) {
                    Ok(()) => status.set_state(State::Idle),
                    Err(e) => log::error!("{e:#}"),
                }
            }
            Err(RecvTimeoutError::Disconnected) => bail!("event channel closed"),
            Err(RecvTimeoutError::Timeout) => {}
        }

        for t in skills.take_due_timers() {
            log::info!("{} done", t.name());
            let turn = status.begin("timer");
            status.update(turn, |x| x.actions.push(format!("{} rang", t.name())));
            status.set_timers(skills.timer_infos());
            status.set_state(State::Speaking);
            audio::play(&out, audio::alarm(), true);
            say(turn, &skills.timer_done_text(&t));
            status.finish(turn);
            status.set_state(State::Idle);
            mute_until = Instant::now() + Duration::from_millis(250);
        }
    }
}

/// Build, engines, models and devices for the Info page.
fn runtime_info(cfg: &Config, listener: &CommandListener, config_path: &std::path::Path) -> serde_json::Value {
    use serde_json::json;
    let lower = |v: &dyn std::fmt::Debug| format!("{v:?}").to_lowercase();
    json!({
        "build": info::build(),
        "config_path": config_path.display().to_string(),
        "language": lower(&cfg.language),
        "wake": {
            "word": cfg.wake.word, "engine": "vosk grammar", "mode": lower(&cfg.wake.mode),
            "continuous_act": lower(&cfg.wake.continuous_act), "verify": cfg.wake.verify,
        },
        "stt": {
            "main": listener.engine_info(cfg),
            "grammar": { "engine": "vosk", "enabled": cfg.stt.grammar, "model": info::model(&cfg.stt.model_path) },
            "end_silence_secs": cfg.stt.end_silence_secs,
            "cut_wake_word": cfg.stt.cut_wake_word,
            "record_dir": cfg.stt.record_dir,
        },
        "tts": {
            "engine": lower(&cfg.tts.engine),
            "piper": cfg.tts.piper_bin,
            "piper_version": (cfg.tts.engine == TtsEngine::Piper).then(|| info::tool_version(&cfg.tts.piper_bin)).flatten(),
            "voice": info::model(&cfg.tts.piper_model),
        },
        "llm": { "enabled": cfg.llm.enabled, "url": cfg.llm.url, "max_tokens": cfg.llm.max_tokens },
        "camera": { "enabled": cfg.camera.enabled, "device": cfg.camera.device, "presence": cfg.camera.presence },
        "audio": {
            "capture": cfg.audio.capture_device, "playback": cfg.audio.playback_device,
            "mixer": format!("card {} / {}", cfg.audio.mixer_card, cfg.audio.mixer_control),
        },
        "status": { "listen": cfg.status.listen },
        "database": { "enabled": cfg.history.enabled, "path": cfg.history.db_path },
    })
}

/// Current settings and the choices for the Admin page.
fn admin_info(cfg: &Config, config_path: &std::path::Path) -> serde_json::Value {
    serde_json::json!({
        "settings": admin::Settings::from_config(cfg),
        "models": admin::models(cfg),
        "language": format!("{:?}", cfg.language).to_lowercase(),
        "config_path": config_path.display().to_string(),
    })
}

/// Grammar vs. free transcript. The grammar recognizer forces everything onto command
/// words, so "can you recommend a good movie" can come out as "good morning".
fn choose_intent(cmd: &speech::Command, lang: config::Lang, trust: f32) -> intents::Intent {
    use intents::Intent::Unknown;
    let intent = intents::parse(&cmd.text, lang);
    let Some(free) = &cmd.free else { return intent };
    let grammar_ok = !cmd.unk && !matches!(intent, Unknown(_));
    match cmd.free_kind {
        // In noise Vosk's free transcript of a real command is garbage with low
        // confidence, while a clearly heard open question is confident.
        speech::FreeKind::Vosk => {
            if grammar_ok && cmd.free_conf < trust {
                return intent;
            }
            intents::parse(free, lang)
        }
        // Moonshine/Whisper are accurate but give no confidence. A command they also
        // understood wins; otherwise the grammar command, if the transcripts agree on its
        // words and the free one isn't a much longer sentence ("can you recommend a good
        // movie" vs. "good morning"); everything else is a question.
        _ => {
            let free_intent = intents::parse(free, lang);
            if !matches!(free_intent, Unknown(_)) {
                return free_intent;
            }
            // "[unk]" words don't matter here: the word agreement is the check.
            if !matches!(intent, Unknown(_)) && agrees(&cmd.text, free) {
                return intent;
            }
            free_intent
        }
    }
}

/// At least half of the grammar words occur in `free`, and `free` is at most about
/// twice as long.
fn agrees(grammar: &str, free: &str) -> bool {
    let g: Vec<&str> = grammar.split_whitespace().collect();
    let f: std::collections::HashSet<&str> = free.split_whitespace().collect();
    let n_free = free.split_whitespace().count();
    if g.is_empty() {
        return false;
    }
    let shared = g.iter().filter(|w| f.contains(*w)).count() as f32 / g.len() as f32;
    shared >= 0.5 && n_free <= 2 * g.len() + 1
}

#[cfg(test)]
mod tests {
    use super::agrees;

    #[test]
    fn grammar_free_agreement() {
        assert!(agrees("volume down", "so i'll leave him down"));
        assert!(agrees("cancel the timer", "it's a scan it's called the timer"));
        assert!(!agrees("good morning", "can you recommend a good movie"));
        assert!(!agrees("noch ist der timer lauter", "warum ist der himmel blau und so weiter"));
        assert!(agrees("make a picture", "make a big shot"));
    }
}

/// Saves the command audio and transcripts (stt.record_dir), to tune recognition with real
/// voices: <time>.wav (pre-roll included) and <time>.txt.
fn record_command(dir: &str, cmd: &speech::Command, heard: &str, intent: &str) {
    let stem = format!("{dir}/{}", chrono::Local::now().format("%Y%m%d-%H%M%S%.3f"));
    let info = format!(
        "heard: {heard}\ngrammar: {}\nfree ({:?}): {}\nintent: {intent}\ncommand_start_sample: {}\n",
        cmd.text,
        cmd.free_kind,
        cmd.free.as_deref().unwrap_or(""),
        cmd.command_start
    );
    let res = std::fs::create_dir_all(dir)
        .and_then(|_| audio::write_wav(&format!("{stem}.wav"), &cmd.audio))
        .and_then(|_| std::fs::write(format!("{stem}.txt"), info));
    if let Err(e) = res {
        log::warn!("recording command to {dir}: {e}");
    }
}
