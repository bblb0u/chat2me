#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETWORK_NAME="chat2m-local"
CONTAINER_NAME="chat2m-voice-agent"
WAKE_WORDS_VALUE="${WAKE_WORDS:-嗨小江,嘿小江,小江}"
WAKE_WORDS_SET=0
AUDIO_CONFIG_MOUNTS=()
DISPLAY_DEVICE_ARGS=()
DISPLAY_SERIAL_PORT_VALUE="${DISPLAY_SERIAL_PORT:-}"

usage() {
  cat <<'EOF'
Usage: ./scripts/start-voice-agent.sh [--wake-word WORD] [--wake-words WORDS]

Options:
  --wake-word WORD    Add one wake word. Can be used more than once.
  --wake-words WORDS  Comma-separated wake words, for example "嗨小江,嘿小江,小江".
  -h, --help          Show this help.
EOF
}

add_wake_word() {
  if [ "$WAKE_WORDS_SET" -eq 0 ]; then
    WAKE_WORDS_VALUE="$1"
    WAKE_WORDS_SET=1
  else
    WAKE_WORDS_VALUE="$WAKE_WORDS_VALUE,$1"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --wake-word)
      add_wake_word "${2:?missing wake word after $1}"
      shift 2
      ;;
    --wake-words)
      WAKE_WORDS_VALUE="${2:?missing wake words after $1}"
      WAKE_WORDS_SET=1
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"

if [ -f /etc/asound.conf ]; then
  AUDIO_CONFIG_MOUNTS+=("-v" "/etc/asound.conf:/etc/asound.conf:ro")
fi

if [ -z "$DISPLAY_SERIAL_PORT_VALUE" ] && [ -e /dev/ttyACM0 ]; then
  DISPLAY_SERIAL_PORT_VALUE="/dev/ttyACM0"
fi

if [ -n "$DISPLAY_SERIAL_PORT_VALUE" ] && [ -e "$DISPLAY_SERIAL_PORT_VALUE" ]; then
  DISPLAY_DEVICE_ARGS+=("--device" "$DISPLAY_SERIAL_PORT_VALUE:$DISPLAY_SERIAL_PORT_VALUE")
fi

if [ ! -d "$ROOT_DIR/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20" ] || \
   [ ! -d "$ROOT_DIR/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20" ] || \
   [ ! -f "$ROOT_DIR/models/piper/zh_CN-huayan-medium/model.onnx" ] || \
   [ ! -f "$ROOT_DIR/models/piper/zh_CN-huayan-medium/model.onnx.json" ]; then
  ./scripts/download-voice-models.sh
fi

docker network inspect "$NETWORK_NAME" >/dev/null 2>&1 || docker network create "$NETWORK_NAME" >/dev/null
docker build -t chat2m/voice-agent:local ./voice-agent

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --network "$NETWORK_NAME" \
  --device /dev/snd:/dev/snd \
  "${DISPLAY_DEVICE_ARGS[@]}" \
  -e AUDIO_INPUT_DEVICE="${AUDIO_INPUT_DEVICE:-ReSpeaker}" \
  -e AUDIO_INPUT_CHANNELS="${AUDIO_INPUT_CHANNELS:-1}" \
  -e AUDIO_INPUT_CHANNEL_INDEX="${AUDIO_INPUT_CHANNEL_INDEX:-0}" \
  -e AUDIO_OUTPUT_DEVICE="${AUDIO_OUTPUT_DEVICE:-plughw:CARD=ArrayUAC10,DEV=0}" \
  -e GATEWAY_URL="${GATEWAY_URL:-http://chat2m-voice-gateway:8080/chat}" \
  -e DISPLAY_SERIAL_PORT="$DISPLAY_SERIAL_PORT_VALUE" \
  -e DISPLAY_SERIAL_BAUD="${DISPLAY_SERIAL_BAUD:-115200}" \
  -e WAKE_WORDS="$WAKE_WORDS_VALUE" \
  -e KWS_KEYWORDS_SCORE="${KWS_KEYWORDS_SCORE:-1.5}" \
  -e KWS_KEYWORDS_THRESHOLD="${KWS_KEYWORDS_THRESHOLD:-0.25}" \
  -e COMMAND_TIMEOUT_SECONDS="${COMMAND_TIMEOUT_SECONDS:-10}" \
  -e COMMAND_LEADING_SILENCE_SECONDS="${COMMAND_LEADING_SILENCE_SECONDS:-4}" \
  -e SPEECH_RMS_THRESHOLD="${SPEECH_RMS_THRESHOLD:-0.006}" \
  -e PIPER_MODEL="${PIPER_MODEL:-/models/piper/zh_CN-huayan-medium/model.onnx}" \
  -e PIPER_CONFIG="${PIPER_CONFIG:-/models/piper/zh_CN-huayan-medium/model.onnx.json}" \
  -e PIPER_SPEAKER="${PIPER_SPEAKER:-0}" \
  -e PIPER_LENGTH_SCALE="${PIPER_LENGTH_SCALE:-0.9}" \
  -e PIPER_VOLUME="${PIPER_VOLUME:-1.0}" \
  -e WAKE_RESPONSE="${WAKE_RESPONSE:-有什么可以帮助您的}" \
  -e SESSION_IDLE_RESPONSE="${SESSION_IDLE_RESPONSE:-}" \
  -e SESSION_END_RESPONSE="${SESSION_END_RESPONSE:-好的，我先待机}" \
  -e SESSION_END_PHRASES="${SESSION_END_PHRASES:-退出,结束,不用了,没事了,再见,拜拜,回到待机,退下,退下吧,你走吧,走吧,下去吧,可以了,先这样}" \
  -e MAX_SESSION_TURNS="${MAX_SESSION_TURNS:-8}" \
  -v "$ROOT_DIR/models:/models" \
  -v "$ROOT_DIR/config:/app/config:ro" \
  "${AUDIO_CONFIG_MOUNTS[@]}" \
  chat2m/voice-agent:local >/dev/null

echo "Chat2M voice agent is running."
echo "Wake words: $WAKE_WORDS_VALUE"
echo "Display serial: ${DISPLAY_SERIAL_PORT_VALUE:-disabled}"
echo "Logs: docker logs -f $CONTAINER_NAME"
