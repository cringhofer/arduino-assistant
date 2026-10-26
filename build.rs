use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // libvosk.so and the statically linked sherpa-onnx (Moonshine) both contain Kaldi and
    // OpenFst. Without this, the executable exports its static copies and libvosk binds to
    // them instead of its own: crash before main(). Keep static symbols private.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,--exclude-libs,ALL");
    }

    // Build information for the status page's Info section.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=build.rs");
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    println!("cargo:rustc-env=BUILD_UNIX={now}");
    println!("cargo:rustc-env=BUILD_TARGET={}", std::env::var("TARGET").unwrap_or_default());
    println!("cargo:rustc-env=BUILD_PROFILE={}", std::env::var("PROFILE").unwrap_or_default());
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc = Command::new(rustc).arg("-V").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    println!("cargo:rustc-env=BUILD_RUSTC={}", rustc.unwrap_or_default());
    let host = Command::new("hostname").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    println!("cargo:rustc-env=BUILD_HOST={}", host.unwrap_or_default());
    // versions of the crates that matter, from Cargo.lock
    let lock = std::fs::read_to_string("Cargo.lock").unwrap_or_default();
    let wanted = ["vosk", "sherpa-onnx", "candle-core", "candle-transformers", "whisper-rs", "rusqlite", "libsqlite3-sys", "libloading", "ureq"];
    let mut deps = Vec::new();
    let mut name = "";
    for line in lock.lines() {
        if let Some(n) = line.strip_prefix("name = ") {
            name = n.trim_matches('"');
        } else if let Some(v) = line.strip_prefix("version = ") {
            if wanted.contains(&name) {
                deps.push(format!("{name}={}", v.trim_matches('"')));
            }
        }
    }
    println!("cargo:rustc-env=BUILD_DEPS={}", deps.join(","));
}
