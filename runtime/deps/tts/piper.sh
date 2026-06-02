#!/bin/sh
set -eu

. /opt/chat2me-deps/lib.sh

PIPER_RELEASE="${PIPER_RELEASE:-2023.11.14-2}"
PIPER_ARCHIVE="${PIPER_ARCHIVE:-piper_linux_aarch64.tar.gz}"

download_file \
  "https://github.com/rhasspy/piper/releases/download/${PIPER_RELEASE}/${PIPER_ARCHIVE}" \
  /tmp/piper.tar.gz \
  "Piper runtime"
rm -rf /opt/piper
tar -xzf /tmp/piper.tar.gz -C /opt
rm -f /tmp/piper.tar.gz
ln -sf /opt/piper/piper /usr/local/bin/piper
/usr/local/bin/piper --help >/dev/null
