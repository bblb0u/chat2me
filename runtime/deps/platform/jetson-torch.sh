#!/bin/sh
set -eu

. /opt/chat2me-deps/lib.sh

if python3 - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
then
  exit 0
fi

JETSON_TORCH_WHEEL_URL="${JETSON_TORCH_WHEEL_URL:-https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl}"
pip_install --force-reinstall --no-deps "$JETSON_TORCH_WHEEL_URL"
