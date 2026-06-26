#!/bin/sh
set -eu

DEFAULT_MODELS_DIR="${DEFAULT_MODELS_DIR:-/opt/chat2me-default-models}"
CHAT2ME_DOWNLOAD_RETRIES="${CHAT2ME_DOWNLOAD_RETRIES:-10}"
DEFAULT_MODEL_SET="${DEFAULT_MODEL_SET:-kws,asr,homophone}"

KWS_MODEL_NAME="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
KWS_MODEL_URL="${KWS_MODEL_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/${KWS_MODEL_NAME}.tar.bz2}"
KWS_MODEL_SHA256="${KWS_MODEL_SHA256:-68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6}"

ASR_MODEL_ENGINE="sensevoice"
ASR_MODEL_NAME="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09"
ASR_MODEL_URL="${ASR_MODEL_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${ASR_MODEL_NAME}.tar.bz2}"
ASR_MODEL_SHA256="${ASR_MODEL_SHA256:-7305f7905bfcf77fa0b39388a313f3da35c68d971661a65475b56fb2162c8e63}"

ASR_HOMOPHONE_LEXICON_URL="${ASR_HOMOPHONE_LEXICON_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt}"
ASR_HOMOPHONE_LEXICON_SHA256="${ASR_HOMOPHONE_LEXICON_SHA256:-978900e511bc481b8630cb6e4a573c12566fa092c366d5396e2c3823dec9dcb9}"

model_selected() {
  case ",$DEFAULT_MODEL_SET," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

download_file() {
  url="$1"
  target="$2"
  label="$3"
  expected_sha256="$4"
  tmp="$target.download"

  mkdir -p "$(dirname "$target")"
  echo "[default-models] downloading $label: $url"
  curl -fL \
    --retry "$CHAT2ME_DOWNLOAD_RETRIES" \
    --retry-connrefused \
    --connect-timeout 20 \
    --speed-limit 1024 \
    --speed-time 120 \
    --show-error \
    "$url" \
    -o "$tmp"
  actual_sha256="$(sha256sum "$tmp" | awk '{print $1}')"
  if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "SHA256 mismatch for $label: expected $expected_sha256, got $actual_sha256" >&2
    rm -f "$tmp"
    exit 1
  fi
  mv "$tmp" "$target"
}

extract_archive() {
  archive="$1"
  target="$2"
  name="$3"

  rm -rf "$target"
  tmp_dir="$target.extract"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"
  tar -xjf "$archive" -C "$tmp_dir"
  extracted_dir="$tmp_dir/$name"
  if [ -d "$extracted_dir" ]; then
    mkdir -p "$(dirname "$target")"
    mv "$extracted_dir" "$target"
  else
    mkdir -p "$target"
    find "$tmp_dir" -mindepth 1 -maxdepth 1 -exec mv {} "$target"/ \;
  fi
  rm -rf "$tmp_dir"
  rm -f "$archive"
}

rm -rf "$DEFAULT_MODELS_DIR"
mkdir -p "$DEFAULT_MODELS_DIR"

if model_selected kws; then
  download_file \
    "$KWS_MODEL_URL" \
    "/tmp/${KWS_MODEL_NAME}.tar.bz2" \
    "$KWS_MODEL_NAME" \
    "$KWS_MODEL_SHA256"
  extract_archive "/tmp/${KWS_MODEL_NAME}.tar.bz2" "$DEFAULT_MODELS_DIR/$KWS_MODEL_NAME" "$KWS_MODEL_NAME"
fi

if model_selected asr; then
  download_file \
    "$ASR_MODEL_URL" \
    "/tmp/${ASR_MODEL_NAME}.tar.bz2" \
    "$ASR_MODEL_NAME" \
    "$ASR_MODEL_SHA256"
  extract_archive "/tmp/${ASR_MODEL_NAME}.tar.bz2" "$DEFAULT_MODELS_DIR/$ASR_MODEL_ENGINE/$ASR_MODEL_NAME" "$ASR_MODEL_NAME"
fi

if model_selected homophone; then
  download_file \
    "$ASR_HOMOPHONE_LEXICON_URL" \
    "$DEFAULT_MODELS_DIR/homophone/lexicon.txt" \
    "homophone lexicon" \
    "$ASR_HOMOPHONE_LEXICON_SHA256"
fi

find "$DEFAULT_MODELS_DIR" -type f -name '*.download' -delete
