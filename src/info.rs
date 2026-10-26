//! What is running: build, engines and models, and live system figures, for the status
//! page's Info section.

use std::path::Path;

use serde_json::{Value, json};

/// The program itself, fixed at build time (see build.rs).
pub fn build() -> Value {
    // crates of features that are not built in are in Cargo.lock too
    let built = |name: &str| match name {
        "sherpa-onnx" => cfg!(feature = "moonshine"),
        "libloading" => cfg!(feature = "moonshine-voice"),
        "whisper-rs" => cfg!(feature = "whisper"),
        "candle-core" | "candle-transformers" => cfg!(feature = "candle"),
        "ureq" => cfg!(feature = "llm"),
        _ => true,
    };
    let deps: serde_json::Map<String, Value> = env!("BUILD_DEPS")
        .split(',')
        .filter_map(|d| d.split_once('='))
        .filter(|(n, _)| built(n))
        .map(|(n, v)| (n.to_string(), json!(v)))
        .collect();
    let features: Vec<&str> = [
        ("llm", cfg!(feature = "llm")),
        ("moonshine", cfg!(feature = "moonshine")),
        ("moonshine-voice", cfg!(feature = "moonshine-voice")),
        ("whisper", cfg!(feature = "whisper")),
        ("candle", cfg!(feature = "candle")),
    ]
    .into_iter()
    .filter_map(|(f, on)| on.then_some(f))
    .collect();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "built_ms": env!("BUILD_UNIX").parse::<u64>().unwrap_or(0) * 1000,
        "built_on": env!("BUILD_HOST"),
        "target": env!("BUILD_TARGET"),
        "profile": env!("BUILD_PROFILE"),
        "rustc": env!("BUILD_RUSTC"),
        "features": features,
        "crates": deps,
        "sqlite": rusqlite::version(),
        "binary": std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default(),
        "binary_bytes": std::env::current_exe().ok().and_then(|p| p.metadata().ok()).map(|m| m.len()),
    })
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

/// "Name:   value kB" line of /proc/<self|meminfo>.
fn kb(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|l| l.strip_prefix(key)?.trim().strip_suffix("kB")?.trim().parse().ok())
}

/// Live figures of the board and this process.
pub fn system() -> Value {
    let meminfo = read("/proc/meminfo").unwrap_or_default();
    let me = read("/proc/self/status").unwrap_or_default();
    let os = read("/etc/os-release")
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("PRETTY_NAME=").map(|v| v.trim_matches('"').to_string())));
    let temps: Vec<f32> = (0..8)
        .filter_map(|i| read(&format!("/sys/class/thermal/thermal_zone{i}/temp"))?.parse::<f32>().ok())
        .map(|t| t / 1000.0)
        .collect();
    let disk = home_disk();
    json!({
        "usb": usb(),
        "hostname": read("/proc/sys/kernel/hostname"),
        "board": read("/proc/device-tree/model").map(|m| m.trim_end_matches('\0').to_string()),
        "os": os,
        "kernel": read("/proc/sys/kernel/osrelease"),
        "cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "load": read("/proc/loadavg").map(|l| l.split(' ').take(3).collect::<Vec<_>>().join(" ")),
        "uptime_s": read("/proc/uptime").and_then(|u| u.split('.').next()?.parse::<u64>().ok()),
        "mem_total_kb": kb(&meminfo, "MemTotal:"),
        "mem_available_kb": kb(&meminfo, "MemAvailable:"),
        "temp_c": temps.iter().cloned().fold(None, |m: Option<f32>, t| Some(m.map_or(t, |m| m.max(t)))),
        "disk_total_kb": disk.map(|d| d.0),
        "disk_free_kb": disk.map(|d| d.1),
        "process": {
            "pid": std::process::id(),
            "rss_kb": kb(&me, "VmRSS:"),
            "threads": me.lines().find_map(|l| l.strip_prefix("Threads:")?.trim().parse::<u64>().ok()),
        },
    })
}

/// USB-C data role ("host" or "device", the selected one) of the first Type-C port.
pub fn usb_role() -> Option<String> {
    let roles = read("/sys/class/typec/port0/data_role")?;
    let start = roles.find('[')? + 1;
    Some(roles[start..start + roles[start..].find(']')?].to_string())
}

/// The Type-C role and the connected USB devices (webcam, sound card, hub...).
fn usb() -> Value {
    let devices: Vec<String> = std::fs::read_dir("/sys/bus/usb/devices")
        .map(|d| {
            d.flatten()
                .filter_map(|e| {
                    let p = e.path();
                    let product = read(&p.join("product").to_string_lossy())?;
                    let id = format!("{}:{}", read(&p.join("idVendor").to_string_lossy())?, read(&p.join("idProduct").to_string_lossy())?);
                    // skip the root hubs
                    (!product.contains("Host Controller")).then(|| format!("{product} ({id})"))
                })
                .collect()
        })
        .unwrap_or_default();
    json!({ "role": usb_role(), "devices": devices })
}

/// (total, free) KiB of the file system holding the home directory, via `df`.
fn home_disk() -> Option<(u64, u64)> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    let out = std::process::Command::new("df").args(["-Pk", &home]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let f: Vec<&str> = text.lines().nth(1)?.split_whitespace().collect();
    Some((f.get(1)?.parse().ok()?, f.get(3)?.parse().ok()?))
}

/// Total size of a file or directory tree in bytes.
pub fn size(path: &str) -> u64 {
    fn walk(p: &Path) -> u64 {
        match p.metadata() {
            Ok(m) if m.is_dir() => std::fs::read_dir(p).map(|d| d.flatten().map(|e| walk(&e.path())).sum()).unwrap_or(0),
            Ok(m) => m.len(),
            Err(_) => 0,
        }
    }
    walk(Path::new(path))
}

/// A model on disk: path, name and size.
pub fn model(path: &str) -> Value {
    let name = Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let exists = Path::new(path).exists();
    json!({ "path": path, "name": name, "bytes": exists.then(|| size(path)), "exists": exists })
}

/// First line printed by `cmd --version` (e.g. Piper), if any.
pub fn tool_version(cmd: &str) -> Option<String> {
    let out = std::process::Command::new(cmd).arg("--version").output().ok()?;
    let text = if out.stdout.is_empty() { out.stderr } else { out.stdout };
    String::from_utf8_lossy(&text).lines().next().map(|l| l.trim().to_string()).filter(|l| !l.is_empty())
}
