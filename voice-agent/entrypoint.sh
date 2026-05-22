#!/bin/sh
set -eu

MODELS_DIR=/models
DEFAULT_CONFIG_DIR="${DEFAULT_CONFIG_DIR:-/defaults/config}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
VOICE_MODELS_REQUIRED="${VOICE_MODELS_REQUIRED:-1}"
KWS_MODEL="$MODELS_DIR/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
ASR_MODEL="$MODELS_DIR/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
PIPER_DIR="$MODELS_DIR/piper/zh_CN-huayan-medium"

init_config() {
  if [ ! -d "$DEFAULT_CONFIG_DIR" ]; then
    return
  fi

  mkdir -p "$CONFIG_DIR"
  for source_file in "$DEFAULT_CONFIG_DIR"/*; do
    [ -f "$source_file" ] || continue
    target_file="$CONFIG_DIR/$(basename "$source_file")"
    if [ ! -e "$target_file" ]; then
      cp "$source_file" "$target_file"
      echo "Initialized config: $target_file"
    fi
  done
}

required_files_ok() {
  for required_file in "$@"; do
    if [ ! -s "$required_file" ]; then
      echo "Missing or empty model file: $required_file"
      return 1
    fi
  done
}

json_file_ok() {
  python3 - "$1" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        json.load(handle)
except Exception as exc:
    print(f"Invalid JSON file: {sys.argv[1]}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

kws_runtime_ok() {
  python3 - "$KWS_MODEL" <<'PY'
import subprocess
import sys
import tempfile
from pathlib import Path

import sherpa_onnx

model_dir = sys.argv[1]
try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_keywords = tmp_path / "keywords_raw.txt"
        keywords = tmp_path / "keywords.txt"
        raw_keywords.write_text("嗨小江 @嗨小江\n", encoding="utf-8")
        subprocess.run(
            [
                "sherpa-onnx-cli",
                "text2token",
                "--tokens",
                f"{model_dir}/tokens.txt",
                "--tokens-type",
                "phone+ppinyin",
                "--lexicon",
                f"{model_dir}/en.phone",
                str(raw_keywords),
                str(keywords),
            ],
            check=True,
        )

        sherpa_onnx.KeywordSpotter(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
            decoder=f"{model_dir}/decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
            joiner=f"{model_dir}/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
            num_threads=1,
            keywords_file=str(keywords),
            provider="cpu",
        )
except Exception as exc:
    print(f"Invalid KWS model: {model_dir}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

asr_runtime_ok() {
  python3 - "$ASR_MODEL" <<'PY'
import sys
import sherpa_onnx

model_dir = sys.argv[1]
try:
    sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{model_dir}/tokens.txt",
        encoder=f"{model_dir}/encoder-epoch-99-avg-1.int8.onnx",
        decoder=f"{model_dir}/decoder-epoch-99-avg-1.int8.onnx",
        joiner=f"{model_dir}/joiner-epoch-99-avg-1.int8.onnx",
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        provider="cpu",
    )
except Exception as exc:
    print(f"Invalid ASR model: {model_dir}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

piper_runtime_ok() {
  python3 - "$PIPER_DIR/model.onnx" "$PIPER_DIR/model.onnx.json" <<'PY'
import sys
from piper.voice import PiperVoice

try:
    PiperVoice.load(sys.argv[1], config_path=sys.argv[2])
except Exception as exc:
    print(f"Invalid Piper model: {sys.argv[1]}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

kws_model_ok() {
  required_files_ok \
    "$KWS_MODEL/tokens.txt" \
    "$KWS_MODEL/en.phone" \
    "$KWS_MODEL/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx" \
    "$KWS_MODEL/decoder-epoch-13-avg-2-chunk-8-left-64.onnx" \
    "$KWS_MODEL/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx" \
    && kws_runtime_ok
}

asr_model_ok() {
  required_files_ok \
    "$ASR_MODEL/tokens.txt" \
    "$ASR_MODEL/encoder-epoch-99-avg-1.int8.onnx" \
    "$ASR_MODEL/decoder-epoch-99-avg-1.int8.onnx" \
    "$ASR_MODEL/joiner-epoch-99-avg-1.int8.onnx" \
    && asr_runtime_ok
}

piper_model_ok() {
  required_files_ok \
    "$PIPER_DIR/model.onnx" \
    "$PIPER_DIR/model.onnx.json" \
    && json_file_ok "$PIPER_DIR/model.onnx.json" \
    && piper_runtime_ok
}

content_length() {
  curl -fsSIL --retry 5 --connect-timeout 20 "$1" \
    | awk 'tolower($1) == "content-length:" { size = $2 } END { gsub("\r", "", size); print size }'
}

print_download_progress() {
  label="$1"
  completed="$2"
  total="$3"

  case "$completed" in ''|*[!0-9]*) completed=0 ;; esac
  case "$total" in ''|*[!0-9]*) total=0 ;; esac

  if [ "$total" -gt 0 ]; then
    awk -v label="$label" -v done="$completed" -v total="$total" '
      BEGIN {
        width = 24
        pct = int(done * 100 / total)
        if (pct > 100) pct = 100
        filled = int(pct * width / 100)
        bar = ""
        for (i = 0; i < width; i++) bar = bar (i < filled ? "#" : "-")
        printf("[models] %s [%s] %3d%% %.1f/%.1f MB\n", label, bar, pct, done / 1048576, total / 1048576)
      }'
  else
    awk -v label="$label" -v done="$completed" '
      BEGIN {
        printf("[models] %s %.1f MB downloaded\n", label, done / 1048576)
      }'
  fi
}

download_with_progress() {
  output="$1"
  url="$2"
  label="$3"
  tmp="$output.download"

  mkdir -p "$(dirname "$output")"
  rm -f "$tmp"
  total="$(content_length "$url" || true)"
  echo "[models] downloading $label"
  print_download_progress "$label" 0 "$total"

  curl -fL --retry 5 --connect-timeout 20 --silent --show-error "$url" -o "$tmp" &
  curl_pid="$!"

  (
    while kill -0 "$curl_pid" 2>/dev/null; do
      if [ -f "$tmp" ]; then
        completed="$(wc -c < "$tmp" | tr -d ' ')"
        print_download_progress "$label" "$completed" "$total"
      fi
      sleep 5
    done
  ) &
  progress_pid="$!"

  if ! wait "$curl_pid"; then
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
    rm -f "$tmp"
    return 1
  fi

  kill "$progress_pid" 2>/dev/null || true
  wait "$progress_pid" 2>/dev/null || true
  completed="$(wc -c < "$tmp" | tr -d ' ')"
  print_download_progress "$label" "$completed" "$total"
  mv "$tmp" "$output"
}

download_and_extract() {
  name="$1"
  url="$2"
  target="$MODELS_DIR/$name"
  archive="$MODELS_DIR/$name.tar.bz2"

  echo "[models] preparing $name"
  rm -rf "$target"
  download_with_progress "$archive" "$url" "$name"
  echo "[models] extracting $name"
  python3 - "$archive" "$MODELS_DIR" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:bz2") as archive:
    archive.extractall(sys.argv[2])
PY
  rm -f "$archive"
  echo "[models] extracted $name"
}

download_file() {
  output="$1"
  url="$2"

  mkdir -p "$(dirname "$output")"
  rm -f "$output"
  download_with_progress "$output" "$url" "$(basename "$output")"
}

ensure_archive_model() {
  name="$1"
  url="$2"
  check_name="$3"

  if "$check_name"; then
    echo "$name is ready"
    return
  fi

  echo "$name is missing or invalid; re-downloading"
  download_and_extract "$name" "$url"

  echo "[models] validating $name"
  if ! "$check_name"; then
    echo "$name is still invalid after download" >&2
    exit 1
  fi
}

ensure_piper_model() {
  if piper_model_ok; then
    echo "piper zh_CN-huayan-medium is ready"
    return
  fi

  echo "piper zh_CN-huayan-medium is missing or invalid; re-downloading"
  rm -rf "$PIPER_DIR"
  download_file \
    "$PIPER_DIR/model.onnx" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx"
  download_file \
    "$PIPER_DIR/model.onnx.json" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json"

  echo "[models] validating piper zh_CN-huayan-medium"
  if ! piper_model_ok; then
    echo "piper zh_CN-huayan-medium is still invalid after download" >&2
    exit 1
  fi
}

init_config

if [ "$VOICE_MODELS_REQUIRED" != "1" ]; then
  exec "$@"
fi

mkdir -p "$MODELS_DIR"
LOCK_DIR="$MODELS_DIR/.download.lock"
while ! mkdir "$LOCK_DIR" 2>/dev/null; do
  echo "[models] waiting for voice model download lock"
  sleep 2
done
echo "[models] voice model download lock acquired"
trap 'rmdir "$LOCK_DIR"' EXIT

ensure_archive_model \
  "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20" \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2" \
  kws_model_ok

ensure_archive_model \
  "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20" \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2" \
  asr_model_ok

ensure_piper_model

trap - EXIT
rmdir "$LOCK_DIR"

exec "$@"
