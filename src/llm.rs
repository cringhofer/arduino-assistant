//! Optional fallback to a local llama.cpp `llama-server` (OpenAI-compatible API).

use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::config::{Lang, LlmConfig};

pub struct Llm {
    cfg: LlmConfig,
    system: String,
    agent: ureq::Agent,
}

impl Llm {
    pub fn new(cfg: &LlmConfig, lang: Lang) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(cfg.timeout_secs)))
            .build()
            .into();
        let system = match lang {
            Lang::En => cfg.system_prompt.clone(),
            Lang::De => format!("{} Always answer in German.", cfg.system_prompt),
        };
        Self { cfg: cfg.clone(), system, agent }
    }

    pub fn ask(&self, question: &str) -> Result<String> {
        let body = json!({
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": question},
            ],
            "max_tokens": self.cfg.max_tokens,
            "temperature": 0.4,
        });
        let v: Value = self
            .agent
            .post(&self.cfg.url)
            .send_json(&body)
            .context("llama-server request")?
            .body_mut()
            .read_json()?;
        let text = v["choices"][0]["message"]["content"].as_str().context("no answer in response")?;
        Ok(clean_for_speech(text))
    }
}

/// Model and build of the running llama-server (its /props endpoint), for the Info page.
pub fn server_info(chat_url: &str) -> Value {
    let props = chat_url.split("/v1/").next().unwrap_or(chat_url).to_string() + "/props";
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(2))).build().into();
    match agent.get(&props).call().map(|mut r| r.body_mut().read_json::<Value>()) {
        Ok(Ok(p)) => json!({
            "running": true,
            "model": p["model_path"],
            "quantization": p["model_ftype"],
            "build": p["build_info"],
            "slots": p["total_slots"],
            "context": p["default_generation_settings"]["n_ctx"],
        }),
        _ => json!({ "running": false }),
    }
}

/// Strips markdown and cuts after the second sentence; small models ramble.
fn clean_for_speech(text: &str) -> String {
    let t: String = text.chars().filter(|c| !matches!(c, '*' | '#' | '`' | '_')).collect();
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut end = t.len();
    let mut sentences = 0;
    for (i, c) in t.char_indices() {
        if matches!(c, '.' | '!' | '?') {
            sentences += 1;
            if sentences == 2 {
                end = i + 1;
                break;
            }
        }
    }
    t[..end].to_string()
}
