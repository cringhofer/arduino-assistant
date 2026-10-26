//! German: "wie spät ist es", "stell einen Timer auf fünf Minuten", "was ist zwölf mal vier".
//! Vosk's German model writes numbers as words, often compounds ("fünfundzwanzig").

use super::{Intent, Op};

pub fn parse(text: &str) -> Intent {
    let t = text.trim().to_lowercase();
    let words: Vec<&str> = t.split_whitespace().collect();
    let has = |p: &str| format!(" {t} ").contains(&format!(" {p} "));
    let any = |ps: &[&str]| ps.iter().any(|p| has(p));

    if words.is_empty() {
        return Intent::Unknown(String::new());
    }

    if any(&["timer", "wecker", "countdown"]) || (has("erinnere mich in") && parse_duration(&words).is_some()) {
        if any(&["lösche", "lösch", "löschen", "stopp", "stoppe", "stop", "abbrechen", "brich", "beende", "beenden", "entferne", "aus"]) {
            return Intent::CancelTimer;
        }
        if any(&["wie lange", "wie viel", "wieviel", "noch", "übrig", "restzeit", "status"]) {
            return Intent::TimerStatus;
        }
        if let Some(secs) = parse_duration(&words) {
            return Intent::SetTimer { secs, label: None };
        }
        // "timer auf zehn" (unit lost in noise): minutes
        if let Some(n) = words.iter().find_map(|w| word_value(w).filter(|_| !matches!(*w, "ein" | "eine" | "einen" | "einer" | "einem"))) {
            return Intent::SetTimer { secs: (n * 60.0) as u64, label: None };
        }
        return Intent::TimerStatus;
    }

    if any(&["lautstärke", "lauter", "leiser"]) {
        if let Some(pos) = words.iter().position(|w| *w == "auf") {
            if let Some(n) = parse_number(&words[pos + 1..]) {
                return Intent::SetVolume(n.clamp(0.0, 10.0) as u8);
            }
        }
        if any(&["lauter", "erhöhe", "erhöhen", "hoch", "rauf"]) {
            return Intent::VolumeUp;
        }
        if any(&["leiser", "runter", "verringere", "verringern", "niedriger"]) {
            return Intent::VolumeDown;
        }
    }

    if any(&["foto", "bild", "selfie", "fotografiere", "schnappschuss"]) {
        return Intent::TakePhoto;
    }
    if any(&["jemand da", "jemand hier", "wer da", "siehst du jemand", "siehst du jemanden", "ist wer da"]) {
        return Intent::AnyoneThere;
    }
    if let Some(calc) = parse_calc(&words) {
        return calc;
    }
    if any(&["wie spät", "spät ist es", "wie viel uhr", "wieviel uhr", "uhrzeit", "die zeit"]) {
        return Intent::Time;
    }
    if any(&["welcher tag", "welchen tag", "welches datum", "das datum", "der wievielte", "den wievielten", "wochentag", "datum"]) {
        return Intent::Date;
    }
    if any(&["witz", "witze", "lustiges", "bring mich zum lachen"]) {
        return Intent::Joke;
    }
    if any(&["was kannst du", "hilfe"]) {
        return Intent::Help;
    }
    if matches!(t.as_str(), "stopp" | "stop" | "halt" | "abbrechen" | "vergiss es" | "sei still" | "ruhe" | "nichts" | "schon gut" | "egal") {
        return Intent::Stop;
    }
    if any(&["hallo", "hi", "hey", "servus", "grüß gott", "guten morgen", "guten tag", "guten abend", "wer bist du", "wie geht es dir", "wie gehts", "wie geht's"]) {
        return Intent::Greeting;
    }
    Intent::Unknown(t)
}

const BASIC: &[(&str, u32)] = &[
    ("null", 0),
    ("ein", 1),
    ("eins", 1),
    ("eine", 1),
    ("einen", 1),
    ("einer", 1),
    ("einem", 1),
    ("zwei", 2),
    ("zwo", 2),
    ("drei", 3),
    ("vier", 4),
    ("fünf", 5),
    ("sechs", 6),
    ("sieben", 7),
    ("acht", 8),
    ("neun", 9),
    ("zehn", 10),
    ("elf", 11),
    ("zwölf", 12),
    ("dreizehn", 13),
    ("vierzehn", 14),
    ("fünfzehn", 15),
    ("sechzehn", 16),
    ("siebzehn", 17),
    ("achtzehn", 18),
    ("neunzehn", 19),
    ("zwanzig", 20),
    ("dreißig", 30),
    ("vierzig", 40),
    ("fünfzig", 50),
    ("sechzig", 60),
    ("siebzig", 70),
    ("achtzig", 80),
    ("neunzig", 90),
];

/// One German number word, including compounds: "einundzwanzig", "zweihundertfünfzig".
fn word_value(w: &str) -> Option<f64> {
    if w.is_empty() {
        return None;
    }
    if let Ok(n) = w.parse::<f64>() {
        return Some(n);
    }
    if let Some(&(_, v)) = BASIC.iter().find(|(k, _)| *k == w) {
        return Some(v as f64);
    }
    for (big, mult) in [("tausend", 1000.0), ("hundert", 100.0)] {
        if let Some(i) = w.find(big) {
            let (pre, post) = (&w[..i], &w[i + big.len()..]);
            let a = if pre.is_empty() { 1.0 } else { word_value(pre)? };
            let post = post.strip_prefix("und").unwrap_or(post);
            let b = if post.is_empty() { 0.0 } else { word_value(post)? };
            return Some(a * mult + b);
        }
    }
    // "einundzwanzig" = 1 + 20
    let i = w.find("und")?;
    let (a, b) = (word_value(&w[..i])?, word_value(&w[i + 3..])?);
    (a < 10.0 && b >= 20.0).then_some(a + b)
}

fn is_number_word(w: &str) -> bool {
    word_value(w).is_some()
}

/// "anderthalb" = 1.5, "eineinhalb" = 1.5, "zweieinhalb" = 2.5
fn half_value(w: &str) -> Option<f64> {
    if w == "anderthalb" {
        return Some(1.5);
    }
    Some(word_value(w.strip_suffix("einhalb")?)? + 0.5)
}

/// Every word must be part of the number: "zwei hundert und fünf" -> 205.
pub fn parse_number(words: &[&str]) -> Option<f64> {
    let (mut total, mut current, mut any) = (0.0_f64, 0.0_f64, false);
    for &w in words {
        match w {
            "und" => continue,
            "hundert" => current = current.max(1.0) * 100.0,
            "tausend" => {
                total += current.max(1.0) * 1000.0;
                current = 0.0;
            }
            _ => current += word_value(w).or_else(|| half_value(w))?,
        }
        any = true;
    }
    any.then_some(total + current)
}

fn unit_secs(w: &str) -> Option<f64> {
    match w {
        "sekunde" | "sekunden" => Some(1.0),
        "minute" | "minuten" => Some(60.0),
        "stunde" | "stunden" => Some(3600.0),
        _ => None,
    }
}

fn amount(buf: &[&str]) -> Option<f64> {
    // "eine halbe stunde"
    if buf.iter().any(|w| matches!(*w, "halbe" | "halben")) {
        return Some(0.5);
    }
    parse_number(buf)
}

/// "eine stunde und dreißig minuten", "anderthalb stunden", "eine halbe stunde" -> seconds.
pub fn parse_duration(words: &[&str]) -> Option<u64> {
    let mut total = 0.0;
    let mut buf: Vec<&str> = Vec::new();
    for &w in words {
        if let Some(mult) = unit_secs(w) {
            if let Some(n) = amount(&buf) {
                total += n * mult;
            }
            buf.clear();
        } else if is_number_word(w) || half_value(w).is_some() || matches!(w, "halbe" | "halben") || (w == "und" && !buf.is_empty()) {
            buf.push(w);
        } else if w != "und" {
            buf.clear();
        }
    }
    (total >= 1.0).then_some(total.round() as u64)
}

fn parse_calc(words: &[&str]) -> Option<Intent> {
    const OPS: &[(&[&str], Op)] = &[
        (&["plus"], Op::Add),
        (&["minus"], Op::Sub),
        (&["mal"], Op::Mul),
        (&["multipliziert", "mit"], Op::Mul),
        (&["geteilt", "durch"], Op::Div),
        (&["dividiert", "durch"], Op::Div),
        (&["durch"], Op::Div),
    ];
    let is_num = |w: &&str| is_number_word(w) || matches!(*w, "und" | "hundert" | "tausend");
    for (pat, op) in OPS {
        let Some(i) = words.windows(pat.len()).position(|w| w == *pat) else { continue };
        let left = &words[..i];
        let start = left.iter().rposition(|w| !is_num(w)).map_or(0, |p| p + 1);
        let a = parse_number(&left[start..])?;
        let right: Vec<&str> = words[i + pat.len()..].iter().copied().take_while(is_num).collect();
        let b = parse_number(&right)?;
        return Some(Intent::Calc { a, op: *op, b });
    }
    None
}

const PHRASES: &[&str] = &[
    // Uhrzeit & Datum
    "wie spät ist es",
    "wie viel uhr ist es",
    "wieviel uhr ist es",
    "sag mir die uhrzeit",
    "welcher tag ist heute",
    "welches datum haben wir",
    "der wievielte ist heute",
    "welcher wochentag ist heute",
    // Timer
    "stell einen timer auf fünf minuten",
    "stelle einen timer für zehn minuten",
    "starte einen timer für dreißig sekunden",
    "timer für eine halbe stunde",
    "timer für anderthalb stunden",
    "timer auf eine stunde und dreißig minuten",
    "erinnere mich in zehn minuten",
    "lösche den timer",
    "stopp den timer",
    "timer abbrechen",
    "lösche alle timer",
    "wie lange läuft der timer noch",
    "wie viel zeit ist noch übrig",
    "sekunde sekunden minute minuten stunde stunden halbe halben eineinhalb zweieinhalb",
    // Lautstärke
    "lauter",
    "leiser",
    "mach lauter",
    "mach leiser",
    "lautstärke hoch",
    "lautstärke runter",
    "stell die lautstärke auf fünf",
    // Rechnen
    "was ist zwei plus zwei",
    "was ist zehn minus drei",
    "was ist sechs mal sieben",
    "was ist hundert geteilt durch fünf",
    "rechne zwölf mal vier",
    "wie viel ist acht durch zwei",
    // Kamera
    "mach ein foto",
    "mach ein bild",
    "mach ein selfie",
    "ist jemand da",
    "ist jemand hier",
    "siehst du jemanden",
    // Sonstiges
    "erzähl mir einen witz",
    "erzähl einen witz",
    "was kannst du",
    "hilfe",
    "stopp",
    "halt",
    "abbrechen",
    "vergiss es",
    "sei still",
    "hallo",
    "servus",
    "grüß gott",
    "guten morgen",
    "guten tag",
    "guten abend",
    "wer bist du",
    "wie geht es dir",
    "bitte",
    "danke",
    "[unk]",
];

pub fn grammar() -> Vec<String> {
    let mut g: Vec<String> = PHRASES.iter().map(|s| s.to_string()).collect();
    // All numbers 0-99 as Vosk writes them, plus hundert/tausend.
    let units = ["ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"];
    let tens = ["zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"];
    let mut numbers: Vec<String> = BASIC.iter().map(|(w, _)| w.to_string()).collect();
    for t in tens {
        for u in units {
            numbers.push(format!("{u}und{t}"));
        }
    }
    numbers.extend(["hundert", "tausend", "und"].map(String::from));
    // One phrase per word: a single unknown word must not drop the whole list.
    g.extend(numbers);
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Vec<&str> {
        s.split_whitespace().collect()
    }

    #[test]
    fn numbers() {
        assert_eq!(parse_number(&w("fünfundzwanzig")), Some(25.0));
        assert_eq!(parse_number(&w("zweihundertfünfzig")), Some(250.0));
        assert_eq!(parse_number(&w("zwei hundert und fünf")), Some(205.0));
        assert_eq!(parse_number(&w("dreitausendvierhundert")), Some(3400.0));
        assert_eq!(parse_number(&w("zwölf")), Some(12.0));
        assert_eq!(parse_number(&w("sekunde")), None);
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration(&w("stell einen timer auf fünf minuten")), Some(300));
        assert_eq!(parse_duration(&w("timer für eine stunde und dreißig minuten")), Some(5400));
        assert_eq!(parse_duration(&w("timer für eine halbe stunde")), Some(1800));
        assert_eq!(parse_duration(&w("timer für anderthalb stunden")), Some(5400));
        assert_eq!(parse_duration(&w("timer für zweieinhalb minuten")), Some(150));
        assert_eq!(parse_duration(&w("timer für neunzig sekunden")), Some(90));
        assert_eq!(parse_duration(&w("stell einen timer")), None);
    }

    #[test]
    fn intents() {
        assert_eq!(parse("wie spät ist es"), Intent::Time);
        assert_eq!(parse("wie viel uhr ist es"), Intent::Time);
        assert_eq!(parse("spät ist es"), Intent::Time);
        assert_eq!(parse("stell einen timer auf zehn minus siebzehn"), Intent::SetTimer { secs: 600, label: None });
        assert_eq!(parse("welcher tag ist heute"), Intent::Date);
        assert_eq!(parse("stell einen timer auf zehn minuten"), Intent::SetTimer { secs: 600, label: None });
        assert_eq!(parse("lösche den timer"), Intent::CancelTimer);
        assert_eq!(parse("wie lange läuft der timer noch"), Intent::TimerStatus);
        assert_eq!(parse("mach lauter"), Intent::VolumeUp);
        assert_eq!(parse("stell die lautstärke auf sieben"), Intent::SetVolume(7));
        assert_eq!(parse("mach ein foto"), Intent::TakePhoto);
        assert_eq!(parse("ist jemand da"), Intent::AnyoneThere);
        assert_eq!(parse("was ist zwölf mal vier"), Intent::Calc { a: 12.0, op: Op::Mul, b: 4.0 });
        assert_eq!(parse("was ist hundert geteilt durch fünf"), Intent::Calc { a: 100.0, op: Op::Div, b: 5.0 });
        assert_eq!(parse("erzähl mir einen witz"), Intent::Joke);
        assert_eq!(parse("stopp"), Intent::Stop);
        assert!(matches!(parse("warum ist der himmel blau"), Intent::Unknown(_)));
    }
}
