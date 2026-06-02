#!/bin/sh
set -eu

. /opt/chat2me-deps/lib/common.sh

pip_install \
  "pypinyin==0.55.0" \
  "sherpa-onnx==1.12.38"
