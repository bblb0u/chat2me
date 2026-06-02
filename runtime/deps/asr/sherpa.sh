#!/bin/sh
set -eu

. /opt/chat2me-deps/lib/common.sh

pip_install \
  "sherpa-onnx==1.12.38"
