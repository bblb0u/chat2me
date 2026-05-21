#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETWORK_NAME="chat2m-local"
CONTAINER_NAME="chat2m-voice-agent"

cd "$ROOT_DIR"

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
  -e AUDIO_INPUT_DEVICE="${AUDIO_INPUT_DEVICE:-ReSpeaker}" \
  -e AUDIO_INPUT_CHANNELS="${AUDIO_INPUT_CHANNELS:-1}" \
  -e AUDIO_INPUT_CHANNEL_INDEX="${AUDIO_INPUT_CHANNEL_INDEX:-0}" \
  -e AUDIO_OUTPUT_DEVICE="${AUDIO_OUTPUT_DEVICE:-default}" \
  -e GATEWAY_URL="${GATEWAY_URL:-http://chat2m-voice-gateway:8080/chat}" \
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
  -e MAX_SESSION_TURNS="${MAX_SESSION_TURNS:-8}" \
  -v "$ROOT_DIR/models:/models" \
  -v "$ROOT_DIR/config:/app/config:ro" \
  chat2m/voice-agent:local >/dev/null

echo "Chat2M voice agent is running."
echo "Wake word: 嗨小紫"
echo "Logs: docker logs -f $CONTAINER_NAME"
