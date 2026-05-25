#!/bin/sh
set -eu

DEFAULT_CONFIG_DIR="${DEFAULT_CONFIG_DIR:-/defaults/config}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_PATH:-$CONFIG_DIR/runtime.env}"

init_config() {
  if [ ! -d "$DEFAULT_CONFIG_DIR" ]; then
    return
  fi

  mkdir -p "$CONFIG_DIR"
  for source_file in "$DEFAULT_CONFIG_DIR"/*; do
    [ -f "$source_file" ] || continue
    target_file="$CONFIG_DIR/$(basename "$source_file")"
    if [ ! -e "$target_file" ]; then
      cp "$source_file" "$target_file"
      echo "Initialized config: $target_file"
    fi
  done
}

sync_runtime_env_defaults() {
  default_runtime="$DEFAULT_CONFIG_DIR/runtime.env"
  target_runtime="$RUNTIME_CONFIG_PATH"
  [ -f "$default_runtime" ] || return 0
  [ -f "$target_runtime" ] || return 0
  [ -w "$target_runtime" ] || return 0

  sync_lock_dir="$CONFIG_DIR/.runtime-env-sync.lock"
  sync_lock_waited=0
  while ! mkdir "$sync_lock_dir" 2>/dev/null; do
    sleep 1
    sync_lock_waited=$((sync_lock_waited + 1))
    if [ "$sync_lock_waited" -ge 60 ]; then
      echo "Removing stale runtime config sync lock: $sync_lock_dir" >&2
      rmdir "$sync_lock_dir" 2>/dev/null || true
      sync_lock_waited=0
    fi
  done
  trap 'rmdir "$sync_lock_dir" 2>/dev/null || true' EXIT

  appended=0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
      *=*) ;;
      *) continue ;;
    esac
    key="${line%%=*}"
    case "$key" in
      ''|*[!A-Za-z0-9_]*) continue ;;
    esac

    if ! grep -Eq "^[[:space:]]*$key[[:space:]]*=" "$target_runtime"; then
      if [ "$appended" -eq 0 ]; then
        printf '\n# Added by Chat2M image defaults. Existing values are never overwritten.\n' >> "$target_runtime"
        appended=1
      fi
      printf '%s\n' "$line" >> "$target_runtime"
      echo "Added missing runtime config: $key"
    fi
  done < "$default_runtime"

  rmdir "$sync_lock_dir"
  trap - EXIT
}

load_runtime_env() {
  [ -f "$RUNTIME_CONFIG_PATH" ] || return

  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
      *=*) ;;
      *) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      ''|*[!A-Za-z0-9_]*) continue ;;
    esac

    eval "is_set=\${$key+x}"
    if [ -z "$is_set" ]; then
      export "$key=$value"
    fi
  done < "$RUNTIME_CONFIG_PATH"
}

json_text_value() {
  key="$1"
  sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p"
}

json_number_value() {
  key="$1"
  sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\\([0-9][0-9]*\\).*/\\1/p"
}

print_progress() {
  label="$1"
  completed="$2"
  total="$3"
  status="$4"

  case "$completed" in ''|*[!0-9]*) completed=0 ;; esac
  case "$total" in ''|*[!0-9]*) total=0 ;; esac

  if [ "$total" -gt 0 ]; then
    awk -v label="$label" -v done="$completed" -v total="$total" -v status="$status" '
      BEGIN {
        width = 24
        pct = int(done * 100 / total)
        if (pct > 100) pct = 100
        filled = int(pct * width / 100)
        bar = ""
        for (i = 0; i < width; i++) bar = bar (i < filled ? "#" : "-")
        printf("[ollama] %s [%s] %3d%% %.1f/%.1f MB %s\n", label, bar, pct, done / 1048576, total / 1048576, status)
      }'
  else
    awk -v label="$label" -v done="$completed" -v status="$status" '
      BEGIN {
        printf("[ollama] %s %.1f MB %s\n", label, done / 1048576, status)
      }'
  fi
}

pull_model_with_progress() {
  fifo="/tmp/ollama-pull.$$"
  rm -f "$fifo"
  mkfifo "$fifo"

  curl -fsS -N \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$MODEL\",\"stream\":true}" \
    http://127.0.0.1:11434/api/pull >"$fifo" &
  curl_pid="$!"

  api_error=0
  last_key=""
  while IFS= read -r line; do
    error="$(printf '%s\n' "$line" | json_text_value error)"
    if [ -n "$error" ]; then
      echo "[ollama] pull error: $error" >&2
      api_error=1
      continue
    fi

    status="$(printf '%s\n' "$line" | json_text_value status)"
    completed="$(printf '%s\n' "$line" | json_number_value completed)"
    total="$(printf '%s\n' "$line" | json_number_value total)"
    digest="$(printf '%s\n' "$line" | json_text_value digest)"

    if [ -n "$completed" ] && [ -n "$total" ] && [ "$total" -gt 0 ] 2>/dev/null; then
      pct=$((completed * 100 / total))
      bucket=$((pct / 5 * 5))
      key="${digest:-blob}:$bucket"
      if [ "$key" != "$last_key" ] || [ "$pct" -eq 100 ]; then
        print_progress "$MODEL" "$completed" "$total" "$status"
        last_key="$key"
      fi
    elif [ -n "$status" ]; then
      echo "[ollama] $MODEL: $status"
    fi
  done <"$fifo"

  curl_status=0
  wait "$curl_pid" || curl_status="$?"
  rm -f "$fifo"

  [ "$curl_status" -eq 0 ] && [ "$api_error" -eq 0 ]
}

init_config
sync_runtime_env_defaults
load_runtime_env
: "${OLLAMA_MODEL:?OLLAMA_MODEL must be set in runtime.env}"
MODEL="$OLLAMA_MODEL"

model_ok() {
  /bin/ollama show "$MODEL" >/dev/null 2>&1 \
    && /bin/ollama run "$MODEL" "只回答OK" >/dev/null 2>&1
}

/bin/ollama serve &
OLLAMA_PID="$!"

until /bin/ollama list >/dev/null 2>&1; do
  sleep 2
done

(
  echo "[ollama] validating model: $MODEL"
  if model_ok; then
    echo "$MODEL is ready"
    exit 0
  fi

  echo "$MODEL is missing or invalid; re-downloading"
  /bin/ollama rm "$MODEL" >/dev/null 2>&1 || true

  until pull_model_with_progress; do
    echo "Retrying Ollama model pull: $MODEL"
    sleep 30
  done

  echo "[ollama] validating model after download: $MODEL"
  if model_ok; then
    echo "$MODEL is ready"
  else
    echo "$MODEL was downloaded but failed runtime validation" >&2
  fi
) &

wait "$OLLAMA_PID"
