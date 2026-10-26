//! Turns a transcript (lowercase words, numbers spelled out as Vosk emits them) into an intent.
//! English lives here, German in `intents/de.rs`; both produce the same `Intent`s.

mod de;

use crate::config::Lang;

pub fn parse(text: &str, lang: Lang) -> Intent {
    match lang {
        Lang::En => parse_en(text),
        Lang::De => de::parse(text),
    }
}

/// Phrases for the grammar-restricted command recognizer.
pub fn grammar(lang: Lang) -> Vec<String> {
    match lang {
        Lang::En => GRAMMAR_EN.iter().map(|s| s.to_string()).collect(),
        Lang::De => de::grammar(),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Intent {
    Time,
    Date,
    SetTimer { secs: u64, label: Option<String> },
    CancelTimer,
    TimerStatus,
    VolumeUp,
    VolumeDown,
    SetVolume(u8),
    Stop,
    Joke,
    Calc { a: f64, op: Op, b: f64 },
    TakePhoto,
    AnyoneThere,
    Greeting,
    Help,
    Unknown(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    pub fn apply(self, a: f64, b: f64) -> Option<f64> {
        match self {
            Op::Add => Some(a + b),
            Op::Sub => Some(a - b),
            Op::Mul => Some(a * b),
            Op::Div => (b != 0.0).then(|| a / b),
        }
    }
}

/// Phrases the command recognizer is restricted to in grammar mode. Vosk builds a small
/// language model from them, so words combine freely ("set a pasta timer for twenty
/// five minutes") but likely word orders win. Keep in sync with `parse`.
const GRAMMAR_EN: &[&str] = &[
    // time & date
    "what time is it",
    "what's the time",
    "what is the time",
    "tell me the time",
    "what day is it",
    "what day is it today",
    "what's the date",
    "what is the date today",
    "what's today's date",
    // timers
    "set a timer for five minutes",
    "set a timer for one hour and thirty minutes",
    "start a timer for ten seconds",
    "set a pasta timer for eight minutes",
    "timer for half an hour",
    "timer for an hour and a half",
    "a minute",
    "remind me in ten minutes",
    "cancel the timer",
    "stop the timer",
    "delete all timers",
    "how much time is left",
    "how long is left on my timer",
    "timer status",
    "pasta egg eggs tea pizza oven bread rice coffee laundry cake",
    // numbers
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen \
     fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy \
     eighty ninety hundred thousand and",
    "seconds second minutes minute hours hour",
    // volume
    "volume up",
    "volume down",
    "turn the volume up",
    "turn the volume down",
    "louder",
    "quieter",
    "set the volume to five",
    // math
    "what is two plus two",
    "what's ten minus three",
    "what is six times seven",
    "what is twelve multiplied by four",
    "what is one hundred divided by five",
    "calculate",
    // camera
    "take a photo",
    "take a picture",
    "take a selfie",
    "is anyone there",
    "is anybody here",
    "do you see anyone",
    "is someone there",
    // misc
    "tell me a joke",
    "make me laugh",
    "what can you do",
    "help",
    "stop",
    "cancel",
    "never mind",
    "be quiet",
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening",
    "who are you",
    "how are you",
    "please",
    "[unk]",
];

pub fn parse_en(text: &str) -> Intent {
    let t = text.trim().to_lowercase();
    let words: Vec<&str> = t.split_whitespace().collect();
    let has = |p: &str| format!(" {t} ").contains(&format!(" {p} "));
    let any = |ps: &[&str]| ps.iter().any(|p| has(p));

    if words.is_empty() {
        return Intent::Unknown(String::new());
    }

    // Timers first: "timer" must not be confused with "time".
    if any(&["timer", "timers"]) || (has("remind me in") && parse_duration(&words).is_some()) {
        if any(&["cancel", "stop", "delete", "remove", "clear"]) {
            return Intent::CancelTimer;
        }
        if any(&["how much", "how long", "left", "remaining", "status"]) {
            return Intent::TimerStatus;
        }
        if let Some(secs) = parse_duration(&words) {
            return Intent::SetTimer { secs, label: timer_label(&words) };
        }
        // "timer for ten" (unit lost in noise): minutes
        if let Some(n) = words.iter().find_map(|w| word_value(w)) {
            return Intent::SetTimer { secs: (n * 60.0) as u64, label: timer_label(&words) };
        }
        return Intent::TimerStatus;
    }

    if has("volume") || any(&["louder", "quieter", "softer"]) {
        if let Some(pos) = words.iter().position(|w| *w == "to") {
            if let Some(n) = parse_number(&words[pos + 1..]) {
                return Intent::SetVolume(n.clamp(0.0, 10.0) as u8);
            }
        }
        if any(&["up", "louder", "increase", "raise"]) {
            return Intent::VolumeUp;
        }
        if any(&["down", "quieter", "softer", "decrease", "lower"]) {
            return Intent::VolumeDown;
        }
    }

    if any(&["photo", "picture", "selfie", "snapshot"]) {
        return Intent::TakePhoto;
    }
    if any(&["anyone there", "anybody there", "anyone here", "anybody here", "see anyone", "see anybody", "someone there", "somebody there"]) {
        return Intent::AnyoneThere;
    }
    if let Some(calc) = parse_calc(&words) {
        return calc;
    }
    if any(&["what time", "the time", "time is it"]) || t == "time" {
        return Intent::Time;
    }
    if any(&["what day", "the date", "what's the date", "today's date", "which day"]) || t == "date" {
        return Intent::Date;
    }
    if any(&["joke", "make me laugh", "something funny"]) {
        return Intent::Joke;
    }
    if any(&["what can you do", "help"]) {
        return Intent::Help;
    }
    if matches!(t.as_str(), "stop" | "cancel" | "never mind" | "nevermind" | "shut up" | "be quiet" | "nothing" | "quiet") {
        return Intent::Stop;
    }
    if any(&["hello", "hi", "hey", "good morning", "good evening", "who are you", "how are you"]) {
        return Intent::Greeting;
    }
    Intent::Unknown(t)
}

fn timer_label(words: &[&str]) -> Option<String> {
    // "set a pasta timer for ..." -> "pasta"
    let pos = words.iter().position(|w| *w == "timer")?;
    let w = *words.get(pos.checked_sub(1)?)?;
    (!matches!(w, "a" | "the" | "set" | "start" | "new") && word_value(w).is_none() && !is_unit(w))
        .then(|| w.to_string())
}

fn parse_calc(words: &[&str]) -> Option<Intent> {
    const OPS: &[(&[&str], Op)] = &[
        (&["plus"], Op::Add),
        (&["minus"], Op::Sub),
        (&["times"], Op::Mul),
        (&["multiplied", "by"], Op::Mul),
        (&["divided", "by"], Op::Div),
        (&["over"], Op::Div),
    ];
    // Around an operator these are almost certainly numbers ("twelve times for").
    fn homophone(w: &str) -> &str {
        match w {
            "for" => "four",
            "to" | "too" => "two",
            "won" => "one",
            "ate" => "eight",
            _ => w,
        }
    }
    let is_num = |w: &&str| is_number_word(w) || *w == "and";
    for (pat, op) in OPS {
        let Some(i) = words.windows(pat.len()).position(|w| w == *pat) else { continue };
        // Numbers directly left and right of the operator; ignore surrounding words.
        let left: Vec<&str> = words[..i].iter().map(|w| homophone(w)).collect();
        let start = left.iter().rposition(|w| !is_num(w)).map_or(0, |p| p + 1);
        let a = parse_number(&left[start..])?;
        let right: Vec<&str> =
            words[i + pat.len()..].iter().map(|w| homophone(w)).take_while(is_num).collect();
        let b = parse_number(&right)?;
        return Some(Intent::Calc { a, op: *op, b });
    }
    None
}

fn word_value(w: &str) -> Option<f64> {
    let v = match w {
        "zero" | "oh" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => return w.parse::<f64>().ok(),
    };
    Some(v as f64)
}

fn is_number_word(w: &str) -> bool {
    word_value(w).is_some() || matches!(w, "hundred" | "thousand" | "million")
}

/// "two hundred and fifty five" -> 255. Every word must be part of the number.
pub fn parse_number(words: &[&str]) -> Option<f64> {
    let (mut total, mut current, mut any) = (0.0_f64, 0.0_f64, false);
    for &w in words {
        match w {
            "and" => continue,
            "hundred" => current = current.max(1.0) * 100.0,
            "thousand" => {
                total += current.max(1.0) * 1000.0;
                current = 0.0;
            }
            "million" => {
                total += current.max(1.0) * 1e6;
                current = 0.0;
            }
            _ => current += word_value(w)?,
        }
        any = true;
    }
    any.then_some(total + current)
}

fn unit_secs(w: &str) -> Option<f64> {
    match w {
        "second" | "seconds" | "sec" | "secs" => Some(1.0),
        "minute" | "minutes" | "min" | "mins" => Some(60.0),
        "hour" | "hours" => Some(3600.0),
        _ => None,
    }
}

fn is_unit(w: &str) -> bool {
    unit_secs(w).is_some()
}

/// Value of the words in front of a unit: "five", "a", "one and a half", "half an".
fn amount(buf: &[&str]) -> Option<f64> {
    let half = buf.contains(&"half");
    let nums: Vec<&str> = buf.iter().copied().filter(|w| is_number_word(w)).collect();
    let base = if nums.is_empty() {
        if half { 0.0 } else if buf.iter().any(|w| matches!(*w, "a" | "an")) { 1.0 } else { return None }
    } else {
        parse_number(&nums)?
    };
    Some(base + if half { 0.5 } else { 0.0 })
}

/// "one hour and thirty minutes", "ninety seconds", "an hour and a half" -> seconds.
pub fn parse_duration(words: &[&str]) -> Option<u64> {
    let mut total = 0.0;
    let mut buf: Vec<&str> = Vec::new();
    let mut last_unit = None;
    for &w in words {
        if let Some(mult) = unit_secs(w) {
            if let Some(n) = amount(&buf) {
                total += n * mult;
                last_unit = Some(mult);
            }
            buf.clear();
        } else if is_number_word(w) || matches!(w, "a" | "an" | "half") || (w == "and" && !buf.is_empty()) {
            buf.push(w);
        } else if !(w == "and" && last_unit.is_some()) {
            buf.clear();
        }
    }
    // trailing "... and a half" refers to the last unit
    if let (true, Some(mult)) = (buf.contains(&"half"), last_unit) {
        total += 0.5 * mult;
    }
    (total >= 1.0).then_some(total.round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Vec<&str> {
        s.split_whitespace().collect()
    }

    #[test]
    fn numbers() {
        assert_eq!(parse_number(&w("twenty five")), Some(25.0));
        assert_eq!(parse_number(&w("two hundred and fifty five")), Some(255.0));
        assert_eq!(parse_number(&w("three thousand four hundred")), Some(3400.0));
        assert_eq!(parse_number(&w("five apples")), None);
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration(&w("set a timer for five minutes")), Some(300));
        assert_eq!(parse_duration(&w("timer for one hour and thirty minutes")), Some(5400));
        assert_eq!(parse_duration(&w("timer for ninety seconds")), Some(90));
        assert_eq!(parse_duration(&w("timer for an hour and a half")), Some(5400));
        assert_eq!(parse_duration(&w("timer for one and a half minutes")), Some(90));
        assert_eq!(parse_duration(&w("timer for half an hour")), Some(1800));
        assert_eq!(parse_duration(&w("timer for a minute")), Some(60));
        assert_eq!(parse_duration(&w("set a timer")), None);
    }

    #[test]
    fn intents() {
        assert_eq!(parse_en("what time is it"), Intent::Time);
        assert_eq!(parse_en("what's the date today"), Intent::Date);
        assert_eq!(
            parse_en("set a timer for ten minutes"),
            Intent::SetTimer { secs: 600, label: None }
        );
        assert_eq!(
            parse_en("set a pasta timer for eight minutes"),
            Intent::SetTimer { secs: 480, label: Some("pasta".into()) }
        );
        assert_eq!(parse_en("cancel the timer"), Intent::CancelTimer);
        assert_eq!(parse_en("how much time is left on my timer"), Intent::TimerStatus);
        assert_eq!(parse_en("volume up"), Intent::VolumeUp);
        assert_eq!(parse_en("set the volume to seven"), Intent::SetVolume(7));
        assert_eq!(parse_en("take a picture"), Intent::TakePhoto);
        assert_eq!(parse_en("is anyone there"), Intent::AnyoneThere);
        assert_eq!(parse_en("what is twelve times four"), Intent::Calc { a: 12.0, op: Op::Mul, b: 4.0 });
        assert_eq!(parse_en("ten divided by four"), Intent::Calc { a: 10.0, op: Op::Div, b: 4.0 });
        assert_eq!(parse_en("so what is twelve times for"), Intent::Calc { a: 12.0, op: Op::Mul, b: 4.0 });
        assert_eq!(parse_en("tell me a joke"), Intent::Joke);
        assert_eq!(parse_en("stop"), Intent::Stop);
        assert!(matches!(parse_en("why is the sky blue"), Intent::Unknown(_)));
    }
}
