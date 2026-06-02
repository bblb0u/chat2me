#!/bin/sh

DOWNLOAD_RETRIES="${CHAT2ME_DOWNLOAD_RETRIES:-10}"
GIT_RETRIES="${CHAT2ME_GIT_RETRIES:-5}"
PIP_RETRIES="${CHAT2ME_PIP_RETRIES:-10}"
PIP_TIMEOUT="${CHAT2ME_PIP_TIMEOUT:-120}"
RETRY_SLEEP_SECONDS="${CHAT2ME_RETRY_SLEEP_SECONDS:-5}"
CURL_CONNECT_TIMEOUT="${CHAT2ME_CURL_CONNECT_TIMEOUT:-20}"
CURL_SPEED_LIMIT="${CHAT2ME_CURL_SPEED_LIMIT:-1024}"
CURL_SPEED_TIME="${CHAT2ME_CURL_SPEED_TIME:-120}"

retry_cmd() {
  attempts="$1"
  shift
  n=1
  while :; do
    if "$@"; then
      return 0
    fi
    if [ "$n" -ge "$attempts" ]; then
      return 1
    fi
    echo "Retrying failed command ($n/$attempts): $*" >&2
    n=$((n + 1))
    sleep "$RETRY_SLEEP_SECONDS"
  done
}

pip_install() {
  retry_cmd "$PIP_RETRIES" python3 -m pip install --no-cache-dir \
    --retries "$PIP_RETRIES" \
    --timeout "$PIP_TIMEOUT" \
    "$@"
}

pip_download() {
  retry_cmd "$PIP_RETRIES" python3 -m pip download \
    --retries "$PIP_RETRIES" \
    --timeout "$PIP_TIMEOUT" \
    "$@"
}

_apt_install_packages() {
  rm -rf /var/lib/apt/lists/*
  apt-get update
  apt-get install -y --no-install-recommends "$@"
}

apt_install_packages() {
  retry_cmd "$DOWNLOAD_RETRIES" _apt_install_packages "$@"
}

download_file() {
  url="$1"
  output="$2"
  label="${3:-$2}"
  attempts="${4:-$DOWNLOAD_RETRIES}"
  tmp="$output.download"

  mkdir -p "$(dirname "$output")"
  n=1
  while :; do
    if curl -fL \
      --connect-timeout "$CURL_CONNECT_TIMEOUT" \
      --speed-limit "$CURL_SPEED_LIMIT" \
      --speed-time "$CURL_SPEED_TIME" \
      --continue-at - \
      --show-error \
      "$url" \
      -o "$tmp"; then
      mv "$tmp" "$output"
      return 0
    fi
    if [ "$n" -ge "$attempts" ]; then
      rm -f "$tmp"
      return 1
    fi
    echo "Retrying download for $label ($n/$attempts)" >&2
    n=$((n + 1))
    sleep "$RETRY_SLEEP_SECONDS"
  done
}

git_clone_retry() {
  target="$1"
  shift
  attempts="${1:-$GIT_RETRIES}"
  shift
  n=1
  while :; do
    rm -rf "$target"
    if git clone "$@" "$target"; then
      return 0
    fi
    if [ "$n" -ge "$attempts" ]; then
      return 1
    fi
    echo "Retrying git clone for $target ($n/$attempts)" >&2
    n=$((n + 1))
    sleep "$RETRY_SLEEP_SECONDS"
  done
}
