#!/bin/sh
set -eu

. /opt/chat2me-deps/lib/common.sh

if python3 - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
then
  exit 0
fi

JETSON_TORCH_WHEEL_URL="${JETSON_TORCH_WHEEL_URL:-https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl}"
pip_install --force-reinstall --no-deps "$JETSON_TORCH_WHEEL_URL"
pip_install \
  "filelock==3.13.4" \
  "jinja2==3.1.4" \
  "networkx==3.1" \
  "sympy==1.12" \
  "typing_extensions>=4.13.2"
