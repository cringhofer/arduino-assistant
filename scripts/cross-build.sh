#!/usr/bin/env bash
# Cross-compile for the UNO Q on an x86_64 Linux/WSL machine.
# Needs: rustup target add aarch64-unknown-linux-gnu
#        sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu cmake unzip
# libvosk is only downloaded for linking; deploy.sh copies it to the board.
set -euo pipefail

cd "$(dirname "$0")/.."
VOSK_VER=0.3.45
LIBDIR=target/vosk-aarch64

if [[ ! -f $LIBDIR/libvosk.so ]]; then
    mkdir -p "$LIBDIR"
    curl -fL -o "$LIBDIR/vosk.zip" \
        "https://github.com/alphacep/vosk-api/releases/download/v$VOSK_VER/vosk-linux-aarch64-$VOSK_VER.zip"
    unzip -qjo "$LIBDIR/vosk.zip" "*/libvosk.so" -d "$LIBDIR"
fi

# A RUSTFLAGS from the environment would override the target flags below.
unset RUSTFLAGS
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
# whisper.cpp (whisper feature) is built with cmake: use the cross compilers, and the
# bindings bundled with whisper-rs-sys instead of running bindgen for the host.
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
export AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
export WHISPER_DONT_GENERATE_BINDINGS=1
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-L $PWD/$LIBDIR"
cargo build --release --target aarch64-unknown-linux-gnu "$@"

echo "built target/aarch64-unknown-linux-gnu/release/uno-assistant"
