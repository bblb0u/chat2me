#!/bin/sh
set -eu

python3 - <<'PY'
import httpx
print(httpx.__version__)
PY
ffmpeg -version >/dev/null
