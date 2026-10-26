#!/usr/bin/env bash
# Cross-compile llama.cpp's llama-server for the UNO Q (Cortex-A53: ARMv8.0, no dotprod)
# on an x86_64 Linux/WSL machine. Output: target/llama-server (static libstdc++).
# Needs: sudo apt install g++-aarch64-linux-gnu cmake git
set -euo pipefail

cd "$(dirname "$0")/.."
SRC=target/llama.cpp
[[ -d $SRC ]] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"

cmake -S "$SRC" -B "$SRC/build-arm64" \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a \
    -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
cmake --build "$SRC/build-arm64" --target llama-server -j "$(nproc)"
cp "$SRC/build-arm64/bin/llama-server" target/llama-server
echo "built target/llama-server"
