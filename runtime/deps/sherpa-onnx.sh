#!/bin/sh
set -eu

. /opt/chat2me-deps/lib.sh

pip_install \
  "sherpa-onnx==1.12.38"
