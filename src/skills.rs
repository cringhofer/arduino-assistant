//! Executes intents and produces the spoken reply.

use std::process::Command;
use std::time::{Duration, Instant};

use chrono::{Datelike, Local};

use crate::camera::Camera;
use crate::config::{Config, Lang};
use crate::intents::Intent;

pub struct Timer {
    pub label: Option<String>,
    pub due: Instant,
}

impl Timer {
    pub fn name(&self) -> String {
        match &self.label {
            Some(l) => format!("{l} timer"),
            None => "timer".into(),
        }
    }

    fn name_de(&self) -> String {
        match &self.label {
            Some(l) => format!("{}-Timer", capitalize(l)),
            None => "Timer".into(),
        }
    }
}

pub struct Skills {
    cfg: Config,
    pub timers: Vec<Timer>,
    pub camera: Option<std::sync::Arc<Camera>>,
    /// Photo taken by the last `handle` (full path).
    pub last_photo: Option<String>,
    #[cfg(feature = "llm")]
    llm: Option<crate::llm::Llm>,
    joke_idx: usize,
    /// What the last `handle` actually did, for the status page.
    actions: Vec<String>,
}

const JOKES: &[&str] = &[
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "I would tell you a UDP joke, but you might not get it.",
    "Why did the microcontroller break up with the sensor? It felt like it was always being read.",
    "There are ten kinds of people: those who understand binary and those who don't.",
    "Why was the computer cold? It left its Windows open.",
    "I asked my Arduino for a joke. It just kept blinking.",
];

const JOKES_DE: &[&str] = &[
    "Warum mögen Programmierer den Dunkelmodus? Weil Licht Bugs anzieht.",
    "Treffen sich zwei Bytes. Fragt das eine: Bist du krank? Sagt das andere: Nein, nur ein bisschen bit.",
    "Wie nennt man einen Bumerang, der nicht zurückkommt? Stock.",
    "Es gibt zehn Arten von Menschen: Die, die Binär verstehen, und die, die es nicht tun.",
    "Ich wollte meinem Arduino einen Witz erzählen. Er hat nur geblinkt.",
    "Was macht ein Pirat am Computer? Er drückt die Enter-Taste.",
];

const WEEKDAYS_DE: [&str; 7] = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"];
const MONTHS_DE: [&str; 12] = [
    "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember",
];

impl Skills {
    pub fn new(cfg: &Config, camera: Option<std::sync::Arc<Camera>>) -> Self {
        Self {
            cfg: cfg.clone(),
            timers: Vec::new(),
            camera,
            last_photo: None,
            #[cfg(feature = "llm")]
            llm: cfg.llm.enabled.then(|| crate::llm::Llm::new(&cfg.llm, cfg.language)),
            joke_idx: (Local::now().timestamp() as usize) % JOKES.len(),
            actions: Vec::new(),
        }
    }

    /// Whether answering this intent may take a while (worth a "thinking" cue).
    pub fn is_slow(&self, intent: &Intent) -> bool {
        matches!(intent, Intent::TakePhoto) || (matches!(intent, Intent::Unknown(_)) && self.cfg.llm.enabled)
    }

    fn de(&self) -> bool {
        self.cfg.language == Lang::De
    }

    /// Picks the reply for the configured language.
    fn tr(&self, en: impl Into<String>, de: impl Into<String>) -> String {
        if self.de() { de.into() } else { en.into() }
    }

    fn duration(&self, secs: u64) -> String {
        if self.de() { speak_duration_de(secs) } else { speak_duration(secs) }
    }

    fn timer_name(&self, t: &Timer) -> String {
        if self.de() { t.name_de() } else { t.name() }
    }

    pub fn timer_done_text(&self, t: &Timer) -> String {
        self.tr(format!("Your {} is done.", t.name()), format!("Dein {} ist abgelaufen.", t.name_de()))
    }

    pub fn welcome_text(&self) -> String {
        self.tr("Hello! Welcome back.", "Hallo! Schön, dass du wieder da bist.")
    }

    /// Returns the reply to speak, or None for silent acknowledgement.
    pub fn handle(&mut self, intent: Intent) -> Option<String> {
        let now = Local::now();
        let no_timers = self.tr("There are no timers running.", "Es läuft kein Timer.");
        Some(match intent {
            Intent::Time => self.tr(
                format!("It's {}.", now.format("%-I:%M %p")),
                format!("Es ist {} Uhr.", now.format("%-H:%M")),
            ),
            Intent::Date => self.tr(
                format!("Today is {}.", now.format("%A, %B %-d, %Y")),
                format!(
                    "Heute ist {}, der {}. {} {}.",
                    WEEKDAYS_DE[now.weekday().num_days_from_monday() as usize],
                    now.day(),
                    MONTHS_DE[now.month0() as usize],
                    now.year()
                ),
            ),
            Intent::SetTimer { secs, label } => {
                let t = Timer { label, due: Instant::now() + Duration::from_secs(secs) };
                let (name, dur) = (self.timer_name(&t), self.duration(secs));
                let reply = self.tr(
                    format!("{} set for {dur}.", capitalize(&name)),
                    format!("{name} auf {dur} gestellt."),
                );
                let rings = Local::now() + chrono::Duration::seconds(secs as i64);
                self.actions.push(format!(
                    "Started {} for {} (rings at {})",
                    t.name(),
                    speak_duration(secs),
                    rings.format("%H:%M:%S")
                ));
                self.timers.push(t);
                reply
            }
            Intent::CancelTimer => match self.timers.len() {
                0 => no_timers,
                n => {
                    let names: Vec<String> = self.timers.iter().map(|t| t.name()).collect();
                    self.actions.push(format!("Cancelled {}", names.join(", ")));
                    self.timers.clear();
                    match n {
                        1 => self.tr("Timer cancelled.", "Timer gelöscht."),
                        n => self.tr(format!("{n} timers cancelled."), format!("{n} Timer gelöscht.")),
                    }
                }
            },
            Intent::TimerStatus => {
                if self.timers.is_empty() {
                    return Some(no_timers);
                }
                self.timers.sort_by_key(|t| t.due);
                let parts: Vec<String> = self
                    .timers
                    .iter()
                    .map(|t| {
                        let left = self.duration(t.due.saturating_duration_since(Instant::now()).as_secs().max(1));
                        self.tr(
                            format!("{left} left on your {}", t.name()),
                            format!("{} läuft noch {left}", t.name_de()),
                        )
                    })
                    .collect();
                format!("{}.", capitalize(&parts.join(self.tr(", and ", ", und ").as_str())))
            }
            Intent::VolumeUp => return self.mixer("10%+", None),
            Intent::VolumeDown => return self.mixer("10%-", None),
            Intent::SetVolume(n) => {
                let reply = self.tr(format!("Volume set to {n}."), format!("Lautstärke auf {n}."));
                return self.mixer(&format!("{}%", n as u32 * 10), Some(reply));
            }
            Intent::Stop => {
                self.actions.push("Stopped".into());
                return None;
            }
            Intent::Joke => {
                let jokes = if self.de() { JOKES_DE } else { JOKES };
                self.joke_idx = (self.joke_idx + 1) % jokes.len();
                jokes[self.joke_idx].into()
            }
            Intent::Calc { a, op, b } => match op.apply(a, b) {
                Some(r) if self.de() => format!("Das ergibt {}.", speak_number(r).replace('.', ",")),
                Some(r) => format!("That's {}.", speak_number(r)),
                None => self.tr("I can't divide by zero.", "Durch null kann ich nicht teilen."),
            },
            Intent::TakePhoto => match &self.camera {
                None => self.tr("The camera is disabled.", "Die Kamera ist ausgeschaltet."),
                Some(cam) => match cam.snapshot() {
                    Ok(path) => {
                        log::info!("photo saved to {path}");
                        self.actions.push(format!("Saved photo {path}"));
                        self.last_photo = Some(path.clone());
                        self.tr("Done, I took a photo.", "Erledigt, ich habe ein Foto gemacht.")
                    }
                    Err(e) => {
                        log::error!("snapshot: {e:#}");
                        self.actions.push(format!("Camera failed: {e:#}"));
                        self.tr("Sorry, I couldn't use the camera.", "Die Kamera funktioniert leider nicht.")
                    }
                },
            },
            Intent::AnyoneThere => match self.camera.as_ref().and_then(|c| c.secs_since_motion()) {
                None => self.tr(
                    "I haven't seen anyone, or presence detection is off.",
                    "Ich habe niemanden gesehen, oder die Erkennung ist aus.",
                ),
                Some(s) if s < 30 => self.tr("Yes, I can see movement right now.", "Ja, ich sehe gerade Bewegung."),
                Some(s) => {
                    let d = self.duration(s);
                    self.tr(format!("I haven't seen anyone for {d}."), format!("Seit {d} war niemand da."))
                }
            },
            Intent::Greeting => self.tr(
                "Hi! I'm Computer, your offline assistant. How can I help?",
                "Hallo! Ich bin Computer, dein Offline-Assistent. Wie kann ich helfen?",
            ),
            Intent::Help => self.tr(
                "I can tell the time and date, set timers, do simple math, change the volume, \
                 take a photo, tell jokes, and tell you if someone is around.",
                "Ich kann dir Uhrzeit und Datum sagen, Timer stellen, rechnen, die Lautstärke \
                 ändern, Fotos machen, Witze erzählen und sagen, ob jemand da ist.",
            ),
            Intent::Unknown(text) => return Some(self.fallback(&text)),
        })
    }

    #[cfg(feature = "llm")]
    fn fallback(&mut self, text: &str) -> String {
        let Some(llm) = &self.llm else {
            self.actions.push("No matching skill, LLM disabled".into());
            return self.tr("Sorry, I don't know how to help with that.", "Das kann ich leider noch nicht.");
        };
        let started = Instant::now();
        let answer = llm.ask(text);
        let secs = started.elapsed().as_secs_f32();
        match answer {
            Ok(a) => {
                self.actions.push(format!("Asked local LLM ({secs:.1} s)"));
                a
            }
            Err(e) => {
                log::error!("llm: {e:#}");
                self.actions.push(format!("LLM failed after {secs:.1} s: {e:#}"));
                self.tr("Sorry, my brain isn't responding right now.", "Mein Sprachmodell antwortet gerade nicht.")
            }
        }
    }

    #[cfg(not(feature = "llm"))]
    fn fallback(&mut self, _text: &str) -> String {
        self.actions.push("No matching skill".into());
        self.tr("Sorry, I don't know how to help with that.", "Das kann ich leider noch nicht.")
    }

    /// Actions performed since the last call.
    pub fn take_actions(&mut self) -> Vec<String> {
        std::mem::take(&mut self.actions)
    }

    pub fn timer_infos(&self) -> Vec<crate::status::TimerInfo> {
        let (now, now_ms) = (Instant::now(), crate::status::unix_ms());
        self.timers
            .iter()
            .map(|t| crate::status::TimerInfo {
                name: t.name(),
                due_ms: now_ms + t.due.saturating_duration_since(now).as_millis() as u64,
            })
            .collect()
    }

    fn mixer(&mut self, value: &str, reply: Option<String>) -> Option<String> {
        let a = &self.cfg.audio;
        let ok = Command::new("amixer")
            .args(["-q", "-c", &a.mixer_card, "sset", &a.mixer_control, value])
            .status()
            .is_ok_and(|s| s.success());
        self.actions.push(format!(
            "amixer -c {} sset {} {value}: {}",
            a.mixer_card,
            a.mixer_control,
            if ok { "ok" } else { "failed" }
        ));
        if ok { reply } else { Some(self.tr("Sorry, I couldn't change the volume.", "Die Lautstärke kann ich leider nicht ändern.")) }
    }

    /// Removes and returns timers that have expired.
    pub fn take_due_timers(&mut self) -> Vec<Timer> {
        let now = Instant::now();
        let (due, keep) = std::mem::take(&mut self.timers).into_iter().partition(|t| t.due <= now);
        self.timers = keep;
        due
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

fn plural(n: u64, unit: &str) -> String {
    if n == 1 { format!("1 {unit}") } else { format!("{n} {unit}s") }
}

pub fn speak_duration(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, secs % 3600 / 60, secs % 60);
    let mut parts = Vec::new();
    if h > 0 {
        parts.push(plural(h, "hour"));
    }
    if m > 0 {
        parts.push(plural(m, "minute"));
    }
    if s > 0 && h == 0 {
        parts.push(plural(s, "second"));
    }
    parts.join(" and ")
}

pub fn speak_duration_de(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, secs % 3600 / 60, secs % 60);
    let unit = |n: u64, one: &str, many: &str| if n == 1 { format!("1 {one}") } else { format!("{n} {many}") };
    let mut parts = Vec::new();
    if h > 0 {
        parts.push(unit(h, "Stunde", "Stunden"));
    }
    if m > 0 {
        parts.push(unit(m, "Minute", "Minuten"));
    }
    if s > 0 && h == 0 {
        parts.push(unit(s, "Sekunde", "Sekunden"));
    }
    parts.join(" und ")
}

fn speak_number(x: f64) -> String {
    if x.fract().abs() < 1e-9 {
        format!("{}", x as i64)
    } else {
        format!("{x:.2}").trim_end_matches('0').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(speak_duration(300), "5 minutes");
        assert_eq!(speak_duration(5400), "1 hour and 30 minutes");
        assert_eq!(speak_duration(61), "1 minute and 1 second");
        assert_eq!(speak_duration_de(5400), "1 Stunde und 30 Minuten");
        assert_eq!(speak_duration_de(1), "1 Sekunde");
    }

    #[test]
    fn numbers() {
        assert_eq!(speak_number(48.0), "48");
        assert_eq!(speak_number(2.5), "2.5");
        assert_eq!(speak_number(1.0 / 3.0), "0.33");
    }
}
