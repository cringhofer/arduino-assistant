//! Settings that can be changed on the status page's Admin section: applied live by the
//! main loop and saved to the config file (keeping its comments and layout).

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{Config, ContinuousAct, Lang, WakeMode};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub wake_word: String,
    pub mode: WakeMode,
    pub continuous_act: ContinuousAct,
    /// Main speech-to-text model (Moonshine model directory).
    pub moonshine_dir: String,
}

impl Settings {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            wake_word: cfg.wake.word.clone(),
            mode: cfg.wake.mode,
            continuous_act: cfg.wake.continuous_act,
            moonshine_dir: cfg.stt.moonshine_dir.clone(),
        }
    }

    /// `cfg` with these settings. Checks what can be checked without the speech model.
    pub fn apply_to(&self, cfg: &Config) -> Result<Config> {
        let word = crate::speech::normalize(&self.wake_word);
        if word.is_empty() || word.split(' ').count() > 3 {
            bail!("the wake word must be one to three words");
        }
        if word.chars().any(|c| !(c.is_alphabetic() || c == ' ' || c == '\'')) {
            bail!("the wake word may only contain letters");
        }
        if self.moonshine_dir != cfg.stt.moonshine_dir && !models(cfg).iter().any(|m| m["path"] == self.moonshine_dir.as_str()) {
            bail!("unknown speech model {}", self.moonshine_dir);
        }
        let mut c = cfg.clone();
        c.wake.word = word;
        c.wake.mode = self.mode;
        c.wake.continuous_act = self.continuous_act;
        c.stt.moonshine_dir = self.moonshine_dir.clone();
        Ok(c)
    }
}

/// Language of a model directory from its name ("...-de", "...-en-int8"); None if unknown.
fn model_lang(name: &str) -> Option<Lang> {
    let n = name.to_lowercase();
    if n.ends_with("-de") || n.contains("-de-") {
        Some(Lang::De)
    } else if n.ends_with("-en") || n.contains("-en-") {
        Some(Lang::En)
    } else {
        None
    }
}

/// Moonshine models next to the configured one that fit the configured language.
pub fn models(cfg: &Config) -> Vec<Value> {
    let Some(dir) = Path::new(&cfg.stt.moonshine_dir).parent() else { return Vec::new() };
    let mut out: Vec<Value> = std::fs::read_dir(dir)
        .map(|d| d.flatten().map(|e| e.path()).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|p| {
            let name = p.file_name()?.to_string_lossy().into_owned();
            let path = p.to_string_lossy().into_owned();
            if !name.contains("moonshine") || model_lang(&name) != Some(cfg.language) {
                return None;
            }
            #[cfg(feature = "moonshine-voice")]
            let new_runtime = crate::moonshine_voice::is_model_dir(&path);
            #[cfg(not(feature = "moonshine-voice"))]
            let new_runtime = false;
            let sherpa = p.join("tokens.txt").exists();
            if !new_runtime && !(sherpa && cfg!(feature = "moonshine")) {
                return None;
            }
            let size = ["tiny", "base", "small", "medium"].into_iter().find(|s| name.contains(s)).unwrap_or("?");
            let streaming = p.join("streaming_config.json").exists();
            Some(json!({
                "path": path,
                "name": name,
                "size": size,
                "streaming": streaming,
                "runtime": if new_runtime { "moonshine-voice" } else { "sherpa-onnx" },
                "bytes": crate::info::size(&path),
            }))
        })
        .collect();
    let rank = |v: &Value| ["tiny", "base", "small", "medium"].iter().position(|s| v["size"] == *s).unwrap_or(9);
    out.sort_by_key(|v| (rank(v), v["runtime"] != "moonshine-voice"));
    out
}

/// Writes the settings into the config file, keeping everything else as it is.
pub fn save(path: &Path, s: &Settings) -> Result<()> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut doc: toml_edit::DocumentMut = text.parse().with_context(|| format!("parsing {}", path.display()))?;
    let lower = |v: &dyn std::fmt::Debug| format!("{v:?}").to_lowercase();
    doc["wake"]["word"] = toml_edit::value(s.wake_word.clone());
    doc["wake"]["mode"] = toml_edit::value(lower(&s.mode));
    doc["wake"]["continuous_act"] = toml_edit::value(lower(&s.continuous_act));
    doc["stt"]["moonshine_dir"] = toml_edit::value(s.moonshine_dir.clone());
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, doc.to_string()).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_keeps_comments_and_other_values() {
        let dir = std::env::temp_dir().join(format!("uno-admin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "language = \"en\"\n\n[wake]\n# the word\nword = \"computer\"\n\n[stt]\nengine = \"moonshine\"\n").unwrap();
        let s = Settings {
            wake_word: "jarvis".into(),
            mode: WakeMode::Continuous,
            continuous_act: ContinuousAct::None,
            moonshine_dir: "/m/moonshine-small-streaming-en".into(),
        };
        save(&path, &s).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# the word"));
        assert!(text.contains("engine = \"moonshine\""));
        let cfg: Config = toml::from_str(&text).unwrap();
        assert_eq!(Settings::from_config(&cfg), s);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn validates_wake_word() {
        let cfg = Config::default();
        let mut s = Settings::from_config(&cfg);
        s.wake_word = "  Hey Computer ".into();
        assert_eq!(s.apply_to(&cfg).unwrap().wake.word, "hey computer");
        s.wake_word = "".into();
        assert!(s.apply_to(&cfg).is_err());
        s.wake_word = "rm -rf".into();
        assert!(s.apply_to(&cfg).is_err());
    }

    #[test]
    fn model_language() {
        assert_eq!(model_lang("moonshine-small-streaming-de"), Some(Lang::De));
        assert_eq!(model_lang("sherpa-onnx-moonshine-tiny-en-int8"), Some(Lang::En));
        assert_eq!(model_lang("moonshine-tiny-en"), Some(Lang::En));
        assert_eq!(model_lang("something"), None);
    }
}
