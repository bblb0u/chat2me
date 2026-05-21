#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
mkdir -p "$MODELS_DIR"

download_and_extract() {
  local name="$1"
  local url="$2"
  local archive="$MODELS_DIR/$name.tar.bz2"

  if [ -d "$MODELS_DIR/$name" ]; then
    echo "$name already exists"
    return
  fi

  echo "Downloading $name"
  curl -fL --retry 5 --connect-timeout 20 "$url" -o "$archive"
  tar -xjf "$archive" -C "$MODELS_DIR"
  rm -f "$archive"
}

download_file() {
  local output="$1"
  local url="$2"

  if [ -f "$output" ]; then
    echo "$(basename "$output") already exists"
    return
  fi

  mkdir -p "$(dirname "$output")"
  echo "Downloading $(basename "$output")"
  curl -fL --retry 5 --connect-timeout 20 "$url" -o "$output"
}

download_and_extract \
  "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20" \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2"

download_and_extract \
  "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20" \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"

PIPER_DIR="$MODELS_DIR/piper/zh_CN-huayan-medium"
download_file \
  "$PIPER_DIR/model.onnx" \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx"
download_file \
  "$PIPER_DIR/model.onnx.json" \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json"

echo "Voice models are ready in $MODELS_DIR"
