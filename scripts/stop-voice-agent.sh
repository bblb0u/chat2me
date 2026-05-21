#!/usr/bin/env bash
set -euo pipefail

docker rm -f chat2m-voice-agent >/dev/null 2>&1 || true
echo "Chat2M voice agent stopped."
