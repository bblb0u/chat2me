#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${OLLAMA_MODEL:-qwen3:0.6b}"
OLLAMA_IMAGE="${OLLAMA_IMAGE:-ollama/ollama:latest}"
PYTHON_IMAGE="${PYTHON_IMAGE:-python:3.11-slim}"
NETWORK_NAME="chat2m-local"
OLLAMA_CONTAINER="chat2m-ollama"
GATEWAY_CONTAINER="chat2m-voice-gateway"

cd "$ROOT_DIR"

docker network inspect "$NETWORK_NAME" >/dev/null 2>&1 || docker network create "$NETWORK_NAME" >/dev/null
docker volume inspect chat2m-ollama-data >/dev/null 2>&1 || docker volume create chat2m-ollama-data >/dev/null

if ! docker ps --format '{{.Names}}' | grep -qx "$OLLAMA_CONTAINER"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$OLLAMA_CONTAINER"; then
    docker rm "$OLLAMA_CONTAINER" >/dev/null
  fi
  docker run -d \
    --name "$OLLAMA_CONTAINER" \
    --restart unless-stopped \
    --network "$NETWORK_NAME" \
    -p 11434:11434 \
    -v chat2m-ollama-data:/root/.ollama \
    "$OLLAMA_IMAGE" >/dev/null
fi

until docker exec "$OLLAMA_CONTAINER" ollama list >/dev/null 2>&1; do
  sleep 2
done

if ! docker exec "$OLLAMA_CONTAINER" ollama list | awk 'NR > 1 {print $1}' | grep -qx "$MODEL"; then
  docker exec "$OLLAMA_CONTAINER" ollama pull "$MODEL"
fi

docker build \
  --build-arg PYTHON_IMAGE="$PYTHON_IMAGE" \
  -t chat2m/voice-gateway:local ./voice-gateway

if docker ps -a --format '{{.Names}}' | grep -qx "$GATEWAY_CONTAINER"; then
  docker rm -f "$GATEWAY_CONTAINER" >/dev/null
fi

docker run -d \
  --name "$GATEWAY_CONTAINER" \
  --restart unless-stopped \
  --network "$NETWORK_NAME" \
  -p 8080:8080 \
  -e OLLAMA_BASE_URL=http://"$OLLAMA_CONTAINER":11434 \
  -e OLLAMA_MODEL="$MODEL" \
  -e PROFILE_PATH=/app/config/profile.yaml \
  -e SAFETY_PATH=/app/config/safety.yaml \
  -v "$ROOT_DIR/config:/app/config:ro" \
  chat2m/voice-gateway:local >/dev/null

echo "Chat2M local chat is running."
echo "Web UI:  http://localhost:8080"
echo "Health:  http://localhost:8080/health"
echo "Model:   $MODEL"
