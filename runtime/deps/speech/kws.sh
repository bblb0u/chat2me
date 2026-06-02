#!/bin/sh
set -eu

. /opt/chat2me-deps/lib/common.sh

pip_install \
  "sherpa-onnx==1.12.38" \
  "click==8.1.8" \
  "sentencepiece==0.2.0" \
  "pypinyin==0.53.0"
