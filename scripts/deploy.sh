#!/usr/bin/env bash
# Install or update uno-assistant on an Arduino UNO Q from this PC over SSH.
# Everything goes to the normal user (no root): ~/uno-assistant, ~/.config/uno-assistant,
# systemd user services. Safe to re-run: existing models and config are kept.
#
#   scripts/deploy.sh arduino@192.168.99.210              # English
#   scripts/deploy.sh arduino@192.168.99.210 --lang de    # German config (new installs)
#   scripts/deploy.sh arduino@192.168.99.210 --llm        # + llama-server and Qwen2.5-0.5B
#   ... --no-build                                         # deploy the existing build
#
# Needs SSH key login to the board (ssh-copy-id), and for building see cross-build.sh.
set -euo pipefail

HOST=${1:?usage: deploy.sh user@host [--lang en|de] [--llm] [--no-build]}
shift
LANG_=en LLM=0 BUILD=1
while (($#)); do
    case $1 in
        --lang) LANG_=$2; shift ;;
        --llm) LLM=1 ;;
        --no-build) BUILD=0 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
    shift
done

cd "$(dirname "$0")/.."
BIN=target/aarch64-unknown-linux-gnu/release/uno-assistant
((BUILD)) && ./scripts/cross-build.sh
# moonshine-voice runtime (Moonshine speech-to-text, English and German)
MV=target/moonshine-voice-aarch64
if [[ ! -f $MV/libmoonshine.so ]]; then
    mkdir -p "$MV"
    curl -fsSL https://github.com/moonshine-ai/moonshine/releases/download/v0.1.5/moonshine-voice-linux-arm64.tar.gz |
        tar xz -C "$MV" --strip-components=2 moonshine-voice-linux-arm64/lib
fi
if ((LLM)) && [[ ! -f target/llama-server ]]; then
    ./scripts/build-llama.sh
fi

echo "== copying binaries to $HOST"
ssh "$HOST" 'mkdir -p ~/uno-assistant/bin ~/uno-assistant/lib/moonshine ~/uno-assistant/models ~/uno-assistant/photos ~/.config/uno-assistant ~/.config/systemd/user'
# Upload under .new and rename on the board: running programs keep their old files.
scp -q "$BIN" "$HOST:uno-assistant/bin/uno-assistant.new"
scp -q target/vosk-aarch64/libvosk.so "$HOST:uno-assistant/lib/libvosk.so.new"
scp -q "$MV/libmoonshine.so" "$HOST:uno-assistant/lib/moonshine/libmoonshine.so.new"
scp -q "$MV/libonnxruntime.so.1" "$HOST:uno-assistant/lib/moonshine/libonnxruntime.so.1.new"
scp -q scripts/get-moonshine-model.sh "$HOST:uno-assistant/bin/get-moonshine-model.sh"
if ((LLM)); then
    scp -q target/llama-server "$HOST:uno-assistant/bin/llama-server.new"
fi

echo "== models and services on the board"
ssh "$HOST" LANG_="$LANG_" LLM="$LLM" bash -s <<'REMOTE'
set -euo pipefail
cd ~/uno-assistant
mv bin/uno-assistant.new bin/uno-assistant
mv lib/libvosk.so.new lib/libvosk.so
mv lib/moonshine/libmoonshine.so.new lib/moonshine/libmoonshine.so
mv lib/moonshine/libonnxruntime.so.1.new lib/moonshine/libonnxruntime.so.1
if [[ -f bin/llama-server.new ]]; then
    mv bin/llama-server.new bin/llama-server
    LLAMA_UPDATED=1
fi
M=models
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
get_zip() { [[ -d $M/$1 ]] || { curl -fsSL -o "$T/z.zip" "$2" && unzip -q "$T/z.zip" -d $M; }; }
get() { [[ -f $M/$1 ]] || curl -fsSL -o "$M/$1" "$2"; }
VOICES=https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0

get_zip vosk-model-small-en-us-0.15 https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
get_zip vosk-model-small-de-0.15 https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip
[[ -x piper/piper ]] || curl -fsSL https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz | tar xz
for e in onnx onnx.json; do
    get en_US-lessac-low.$e $VOICES/en/en_US/lessac/low/en_US-lessac-low.$e
    get de_DE-thorsten-low.$e $VOICES/de/de_DE/thorsten/low/de_DE-thorsten-low.$e
done
# Moonshine (new runtime): small = most accurate (~4-5 s per command on the UNO Q),
# tiny = faster but weak in noise. Switch between them on the status page (Admin).
if [[ $LANG_ == de ]]; then
    bin/get-moonshine-model.sh de small $M/moonshine-small-streaming-de >/dev/null
    bin/get-moonshine-model.sh de tiny $M/moonshine-tiny-streaming-de >/dev/null
else
    bin/get-moonshine-model.sh en small $M/moonshine-small-streaming-en >/dev/null
fi
# English fallback with sherpa-onnx: the fastest (~1 s per command)
[[ -d $M/sherpa-onnx-moonshine-tiny-en-int8 ]] || {
    curl -fsSL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2 | tar xj -C $M
    rm -rf $M/sherpa-onnx-moonshine-tiny-en-int8/test_wavs
}
if ((LLM)); then
    get qwen2.5-0.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
fi

C=~/.config/uno-assistant/config.toml
if [[ ! -f $C ]]; then
    H=$HOME/uno-assistant
    if [[ $LANG_ == de ]]; then
        VOSK=vosk-model-small-de-0.15 VOICE=de_DE-thorsten-low
    else
        VOSK=vosk-model-small-en-us-0.15 VOICE=en_US-lessac-low
    fi
    MIC=$(arecord -L 2>/dev/null | grep -m1 '^plughw:CARD=.*DEV=0' || echo default)
    cat > $C <<EOF
# uno-assistant. Full reference: config.example.toml in the repository.
language = "$LANG_"

[wake]
word = "computer"

[audio]
# arecord -L / aplay -L
capture_device = "$MIC"
playback_device = "default"

[stt]
model_path = "$H/models/$VOSK"
# Moonshine transcribes every command (model switchable on the status page, Admin)
engine = "moonshine"
moonshine_dir = "$H/models/moonshine-small-streaming-$LANG_"
moonshine_lib = "$H/lib/moonshine/libmoonshine.so"

[tts]
piper_bin = "$H/piper/piper"
piper_model = "$H/models/$VOICE.onnx"
piper_sample_rate = 16000

[camera]
# uses v4l2-ctl (v4l-utils)
enabled = false
photo_dir = "$H/photos"

[llm]
enabled = $( ((LLM)) && echo true || echo false )

[status]
listen = "0.0.0.0:8088"

[history]
db_path = "$H/history.db"
EOF
    echo "wrote $C (mic: $MIC)"
else
    echo "keeping existing $C"
fi

cat > ~/.config/systemd/user/uno-assistant.service <<'EOF'
[Unit]
Description=Offline voice assistant (wake word "computer")
After=sound.target

[Service]
Environment=LD_LIBRARY_PATH=%h/uno-assistant/lib
Environment=RUST_LOG=info
ExecStart=%h/uno-assistant/bin/uno-assistant --config %h/.config/uno-assistant/config.toml
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
if ((LLM)); then
    cat > ~/.config/systemd/user/llama-server.service <<'EOF'
[Unit]
Description=llama.cpp server for uno-assistant

[Service]
ExecStart=%h/uno-assistant/bin/llama-server -m %h/uno-assistant/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 -c 1024 -t 3 --no-webui
Restart=on-failure
RestartSec=10
Nice=5

[Install]
WantedBy=default.target
EOF
fi

loginctl enable-linger "$USER" 2>/dev/null || echo "!! could not enable lingering: services only run while logged in"
systemctl --user daemon-reload
if ((LLM)); then
    systemctl --user enable llama-server >/dev/null 2>&1
    if [[ ${LLAMA_UPDATED:-0} == 1 ]]; then systemctl --user restart llama-server; else systemctl --user start llama-server; fi
fi
systemctl --user enable uno-assistant >/dev/null 2>&1
systemctl --user restart uno-assistant
sleep 8
systemctl --user is-active uno-assistant
journalctl --user -u uno-assistant -n 4 --no-pager -o cat
REMOTE

IP=${HOST#*@}
echo
echo "Done. Status page: http://$IP:8088"
echo "Optional (needs sudo on the board): the time zone:"
echo "  ssh -t $HOST 'sudo timedatectl set-timezone Europe/Vienna'"
