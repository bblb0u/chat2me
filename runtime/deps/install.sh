#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 '<feature...|none>'" >&2
  exit 2
fi

features="$1"
if [ -z "$(printf '%s' "$features" | tr -d '[:space:]')" ]; then
  echo "VOICE_RUNTIME_FEATURES must be a space-separated feature list or 'none'" >&2
  exit 2
fi

for feature in $features; do
  case "$feature" in
    speech-kws)
      /opt/chat2me-deps/speech/kws.sh
      ;;
    asr-sherpa)
      /opt/chat2me-deps/asr/sherpa.sh
      ;;
    tts-melotts)
      /opt/chat2me-deps/tts/melotts.sh
      ;;
    tts-sherpa)
      /opt/chat2me-deps/tts/sherpa.sh
      ;;
    none)
      ;;
    *)
      echo "Unknown voice agent feature: $feature" >&2
      exit 1
      ;;
  esac
done
