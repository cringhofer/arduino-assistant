# uno-assistant

An offline, Alexa-style voice assistant in Rust for the **Arduino UNO Q (2 GB)**, in
English and German. Say **"computer"** (configurable), then a command. Nothing leaves the
board.

```
 mic ─arecord─▶ Vosk wake word ─▶ Vosk grammar recognizer ──┐  (end of command, fallback)
                 ("computer")       + 1.5 s pre-roll         ├─▶ intent ─▶ skill ─▶ Piper ─aplay─▶ speaker
                                  ─▶ main engine: Moonshine ─┘        └─▶ else: local LLM (llama.cpp)
                                     (streamed while you speak)
 webcam ─v4l2─▶  photos, motion ("is anyone there?")        status page: http://<board>:8088
```

## What it can do

| Say "computer, …" | Sag „Computer, …“ | |
|---|---|---|
| what time is it / what's the date | wie spät ist es / welcher Tag ist heute | clock |
| set a (pasta) timer for one hour and thirty minutes | stell einen Timer auf anderthalb Stunden | timers with alarm |
| how much time is left / cancel the timer | wie lange läuft der Timer noch / lösche den Timer | |
| what is twelve times four | was ist zwölf mal vier | + − × ÷ |
| volume up / set the volume to seven | mach lauter / Lautstärke auf sieben | ALSA mixer |
| take a picture / is anyone there? | mach ein Foto / ist jemand da? | webcam (v4l2-ctl) |
| tell me a joke / what can you do / stop | erzähl mir einen Witz / was kannst du / stopp | |
| *anything else* | *alles andere* | local LLM (Qwen2.5-0.5B), if enabled |

A pause after the wake word works ("computer … what day is it"), and so does saying it all
in one breath.

## How recognition works

* **Wake word:** a Vosk recognizer restricted to `["computer", "[unk]"]`. It needs no
  training, and any word in the model vocabulary works (`wake.word`).
* **Grammar recognizer (Vosk):** only knows the phrases the skills understand
  (`intents.rs`, `intents/de.rs`). In noise this is far more reliable than free dictation:
  with real room noise from the UNO Q's webcam mic it got 8/8 commands right, against 3/8
  for free Vosk. It also detects the end of the command (the words stop changing for 0.7 s)
  and confirms the wake word (`wake.verify`).
* **Main engine (`stt.engine = "moonshine"`):** transcribes the command, analysing it
  while you speak.
  * If its text is a command, that command runs.
  * If not, but the grammar heard a command with matching words, the grammar command runs.
  * Anything else is a question and goes to the LLM.

### Speech-to-text: Moonshine

Moonshine (the official moonshine-voice runtime) transcribes every command, streamed while
you speak; Vosk does the wake word, the end of the command and the command grammar. The
runtime (`libmoonshine.so` + `libonnxruntime.so.1` from the moonshine-voice release) is
loaded at run time; models come from `scripts/get-moonshine-model.sh en|de tiny|small DIR`.
Switch models live on the status page (Admin).

Measured on the UNO Q with noisy test commands in the running assistant (14 English, 12
German; response = end of speech until the first sound, including Piper):

| model | commands right | response | |
|---|---|---|---|
| English tiny | 8 / 14 | **~2.6 s** | fast; misses more open questions in noise |
| English small | **11 / 14** | ~5.7 s | most accurate in English (default) |
| German tiny | **9–11 / 12** | **~2.2 s** | best for German (default), with built-in command keywords |
| German small | 8 / 12 | ~5.9 s | not better than tiny in German |

What made it faster: streaming the audio while you speak, giving Moonshine only 0.8 s
before the wake detection (`stt.stream_preroll_secs`), decoding each line once instead of
after every chunk, and keeping Piper running (first sound 0.9 s instead of 3–4 s). The
grammar rescues commands Moonshine mishears when it heard them cleanly and confidently
(`stt.grammar_trust_conf`). The board has no usable accelerator (Adreno 702: Vulkan 1.0
only, no NPU) and its Cortex-A53 cores lack dot-product instructions, so this is CPU work.

The German models are under the non-commercial
[Moonshine Community License](https://www.moonshine.ai/license).
`uno-assistant --config … --transcribe *.wav` benchmarks files.

## Hardware

* Arduino UNO Q, 2 GB. With the LLM running about 1 GB is still free.
* A USB-C hub **with Power Delivery** (the UNO Q has one USB-C port, which also powers it).
* Microphone: e.g. a webcam mic (tested: eMeet C960). A USB speakerphone (Jabra Speak, Anker
  PowerConf, …) is better: far-field mic and echo cancellation.
* Speaker: USB sound card / speaker / speakerphone.

## Build (on an x86_64 Linux or WSL PC)

```bash
rustup target add aarch64-unknown-linux-gnu
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu cmake unzip
./scripts/cross-build.sh          # assistant (features: llm, moonshine)
./scripts/build-llama.sh          # llama-server, only needed for the LLM
```

Cargo features: `llm` and `moonshine` (both default). The Moonshine library
(`libmoonshine.so` + its `libonnxruntime.so.1`) is loaded at run time, so nothing of it is
needed to build; `deploy.sh` downloads the release and installs it on the board.

## Install on the UNO Q

Set up SSH key login once (`ssh-copy-id arduino@<board-ip>`), then from the PC:

```bash
./scripts/deploy.sh arduino@<board-ip> --llm            # English + Moonshine + LLM
./scripts/deploy.sh arduino@<board-ip> --lang de --llm  # German (new install)
```

The script needs no root and is safe to re-run. It installs everything for the `arduino`
user:
* `~/uno-assistant`: binaries, models, Piper
* `~/.config/uno-assistant/config.toml`: kept if it exists
* the systemd user services `uno-assistant` and `llama-server`

Afterwards:

```bash
ssh arduino@<board-ip> 'arecord -L; aplay -L'          # pick mic/speaker for the config
ssh arduino@<board-ip> 'journalctl --user -fu uno-assistant'
```

The status page at `http://<board-ip>:8088` shows the state, mic level and live transcript.
For every command it shows what was heard (main engine and grammar), the intent, the
actions taken and the reply, with the recording and the spoken answer to play back.
The **Latency** tab shows how long each stage takes (end-of-speech detection, speech to
text, intent, command / LLM / camera, voice synthesis, the whole response and the wake
detector's CPU load), as medians, p95, a per-command breakdown and over time. All of it
is stored in the SQLite history (`latency` table).

**Continuous mode** (`wake.mode = "continuous"` or the switch on the page): no wake
word, everything said is transcribed. `wake.continuous_act` decides what is acted on:
`"commands"` (default: recognized commands run; questions only reach the LLM when they
start with the wake word), `"all"` or `"none"`.

Besides reading, the page can take a photo and switch continuous mode. It has no login:
to keep it off the network, set `status.listen = "127.0.0.1:8088"`.

Optional, needs sudo: `sudo timedatectl set-timezone Europe/Vienna` (the board runs on UTC by default).

## Switching to German

In `~/.config/uno-assistant/config.toml`:

```toml
language = "de"
[stt]
model_path = "/home/arduino/uno-assistant/models/vosk-model-small-de-0.15"
engine = "moonshine"
moonshine_dir = "/home/arduino/uno-assistant/models/moonshine-small-streaming-de"
[tts]
piper_model = "/home/arduino/uno-assistant/models/de_DE-thorsten-low.onnx"
```

Get the German models with `scripts/deploy.sh <host> --lang de` (or
`get-moonshine-model.sh de small …`), then run `systemctl --user restart uno-assistant`.
The LLM is told to answer in German.

## Testing without a microphone

```bash
uno-assistant --print --text "set a timer for ten minutes"   # no audio at all
uno-assistant --print --wav command.wav                      # 16 kHz mono WAV, real time
cargo test
```

## Layout

| file | |
|---|---|
| `src/main.rs` | event loop: audio → wake → listen → intent → reply, timers, engine choice |
| `src/speech.rs` | wake word, grammar recognizer, end-of-command, main/free transcriber |
| `src/moonshine.rs` | Moonshine speech-to-text (moonshine-voice library, streaming) |
| `src/intents.rs`, `src/intents/de.rs` | transcript → intent, numbers, durations, grammars |
| `src/skills.rs` | runs intents, replies in English/German, timers, volume, camera, LLM |
| `src/status.rs`, `status.html` | status web page: Status, Stats, Info, Admin |
| `src/admin.rs`, `src/info.rs` | live settings saved to the config; what is running |
| `src/history.rs` | SQLite history and latency, versioned schema migrations |
| `src/audio.rs`, `tts.rs`, `camera.rs`, `llm.rs` | arecord/aplay, Piper/espeak, v4l2-ctl, llama-server |
| `scripts/` | cross-build, llama-server build, deploy |
