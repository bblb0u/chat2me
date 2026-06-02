#!/bin/sh
set -eu

features="${1:-sensevoice sherpa melotts piper f5-tts}"

for feature in $features; do
  case "$feature" in
    asr-sherpa|sherpa)
      /opt/chat2me-deps/shared/sherpa_onnx.sh
      ;;
    asr-sensevoice|sensevoice)
      /opt/chat2me-deps/asr/sensevoice.sh
      ;;
    tts-piper|piper)
      /opt/chat2me-deps/tts/piper.sh
      ;;
    tts-melotts|melotts)
      /opt/chat2me-deps/shared/sherpa_onnx.sh
      ;;
    tts-f5|f5|f5-tts)
      /opt/chat2me-deps/tts/f5.sh
      ;;
    tts-cosyvoice|cosyvoice)
      /opt/chat2me-deps/tts/cosyvoice.sh
      ;;
    none)
      ;;
    *)
      echo "Unknown voice agent feature: $feature" >&2
      exit 1
      ;;
  esac
done
