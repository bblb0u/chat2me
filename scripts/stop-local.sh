#!/usr/bin/env bash
set -euo pipefail

docker rm -f chat2m-voice-gateway >/dev/null 2>&1 || true
docker rm -f chat2m-ollama >/dev/null 2>&1 || true

echo "Chat2M local chat containers stopped."
