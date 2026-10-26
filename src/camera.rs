//! Webcam: snapshots and low-res motion detection via `v4l2-ctl` (v4l-utils, part of
//! the UNO Q image): MJPEG frames are saved as JPEG directly, no ffmpeg or decoding.
//! The device can only be opened once, so a snapshot pauses the presence stream.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

use crate::Event;
use crate::config::CameraConfig;

/// Presence stream: YUYV, luma is every other byte.
const W: usize = 160;
const H: usize = 120;

pub struct Camera {
    cfg: CameraConfig,
    paused: Arc<AtomicBool>,
    stream: Arc<Mutex<Option<Child>>>,
    /// Unix seconds of the last detected motion (0 = never).
    last_motion: Arc<AtomicU64>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

impl Camera {
    pub fn new(cfg: CameraConfig, tx: Sender<Event>) -> Self {
        let cam = Self {
            cfg,
            paused: Arc::default(),
            stream: Arc::default(),
            last_motion: Arc::default(),
        };
        if cam.cfg.presence {
            cam.spawn_presence(tx);
        }
        cam
    }

    /// Seconds since motion was last seen, None if presence detection is off or never saw any.
    pub fn secs_since_motion(&self) -> Option<u64> {
        let t = self.last_motion.load(Ordering::Relaxed);
        (self.cfg.presence && t > 0).then(|| now().saturating_sub(t))
    }

    fn spawn_presence(&self, tx: Sender<Event>) {
        let (cfg, paused, stream, last_motion) =
            (self.cfg.clone(), self.paused.clone(), self.stream.clone(), self.last_motion.clone());
        thread::spawn(move || {
            let greet_after = cfg.greet_after_absent_mins * 60;
            let mut missing = false;
            loop {
                if paused.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(200));
                    continue;
                }
                // unplugged (or USB not powered): wait quietly until it shows up again
                if !std::path::Path::new(&cfg.device).exists() {
                    if !missing {
                        log::warn!("camera {} not found, waiting for it", cfg.device);
                        missing = true;
                    }
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }
                if missing {
                    log::info!("camera {} is back", cfg.device);
                    missing = false;
                }
                let child = Command::new("v4l2-ctl")
                    .args(["-d", &cfg.device, "--stream-mmap", "--stream-to=-"])
                    .arg(format!("--set-fmt-video=width={W},height={H},pixelformat=YUYV"))
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .stdin(Stdio::null())
                    .spawn();
                let mut child = match child {
                    Ok(c) => c,
                    Err(e) => {
                        log::error!("presence: cannot start v4l2-ctl: {e}");
                        return;
                    }
                };
                let mut out = child.stdout.take().unwrap();
                *stream.lock().unwrap() = Some(child);

                let mut prev: Option<Vec<u8>> = None;
                let mut frame = vec![0u8; W * H * 2];
                let mut last = std::time::Instant::now();
                while out.read_exact(&mut frame).is_ok() {
                    // ~30 fps from the camera; compare twice a second
                    if last.elapsed() < Duration::from_millis(500) {
                        continue;
                    }
                    last = std::time::Instant::now();
                    let frame: Vec<u8> = frame.iter().step_by(2).copied().collect();
                    if let Some(p) = &prev {
                        let diff = p.iter().zip(&frame).map(|(a, b)| a.abs_diff(*b) as u64).sum::<u64>()
                            as f32
                            / frame.len() as f32;
                        if diff > cfg.motion_threshold {
                            let before = last_motion.swap(now(), Ordering::Relaxed);
                            log::debug!("motion {diff:.1}");
                            if greet_after > 0 && before > 0 && now() - before > greet_after {
                                let _ = tx.send(Event::PersonArrived);
                            }
                        }
                    }
                    prev = Some(frame);
                }
                if let Some(mut c) = stream.lock().unwrap().take() {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                if !paused.load(Ordering::Relaxed) {
                    log::warn!("presence stream ended, restarting in 5s");
                    thread::sleep(Duration::from_secs(5));
                }
            }
        });
    }

    /// Captures a full-resolution JPEG and returns its path.
    pub fn snapshot(&self) -> Result<String> {
        std::fs::create_dir_all(&self.cfg.photo_dir)
            .with_context(|| format!("creating {}", self.cfg.photo_dir))?;
        let path = format!(
            "{}/photo-{}.jpg",
            self.cfg.photo_dir,
            chrono::Local::now().format("%Y%m%d-%H%M%S%.3f")
        );

        self.paused.store(true, Ordering::Relaxed);
        if let Some(c) = self.stream.lock().unwrap().as_mut() {
            let _ = c.kill();
        }
        thread::sleep(Duration::from_millis(300)); // let the device close
        // The camera sends MJPEG: one frame is a JPEG file. Skip frames so auto exposure
        // settles.
        let status = Command::new("v4l2-ctl")
            .args(["-d", &self.cfg.device, "--stream-mmap", "--stream-skip=15", "--stream-count=1"])
            .arg("--set-fmt-video=width=1280,height=720,pixelformat=MJPG")
            .arg(format!("--stream-to={path}"))
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .status();
        self.paused.store(false, Ordering::Relaxed);

        match status {
            Ok(s) if s.success() && std::fs::metadata(&path).is_ok_and(|m| m.len() > 0) => Ok(path),
            Ok(s) => bail!("v4l2-ctl exited with {s}"),
            Err(e) => bail!("cannot run v4l2-ctl (apt install v4l-utils): {e}"),
        }
    }

    pub fn photo_dir(&self) -> &str {
        &self.cfg.photo_dir
    }
}
