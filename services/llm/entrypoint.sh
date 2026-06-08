#!/bin/sh
set -eu

DEFAULT_CONFIG_DIR="${DEFAULT_CONFIG_DIR:-/defaults/config}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_PATH:-$CONFIG_DIR/runtime.env}"
VOICE_ROLE="${VOICE_ROLE:-chat2me-llm}"

normalize_log_level() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    debug|info|warning|error) printf '%s' "$1" | tr '[:upper:]' '[:lower:]' ;;
    warn) echo "warning" ;;
    err) echo "error" ;;
    *) echo "$2" ;;
  esac
}

log_level_value() {
  case "$(normalize_log_level "$1" info)" in
    debug) echo 10 ;;
    info) echo 20 ;;
    warning) echo 30 ;;
    error) echo 40 ;;
    *) echo 20 ;;
  esac
}

runtime_env_value() {
  key="$1"
  [ -f "$RUNTIME_CONFIG_PATH" ] || return 1
  sed -n "s/^${key}=//p" "$RUNTIME_CONFIG_PATH" | tail -n 1
}

chat2me_log() {
  level="$(normalize_log_level "${1:-info}" info)"
  message="$2"
  file_level="$(normalize_log_level "${CHAT2ME_LOG_LEVEL:-$(runtime_env_value CHAT2ME_LOG_LEVEL || true)}" info)"
  console_level="$(normalize_log_level "${CHAT2ME_CONSOLE_LOG_LEVEL:-$(runtime_env_value CHAT2ME_CONSOLE_LOG_LEVEL || true)}" warning)"
  log_dir="/app/log"
  level_value="$(log_level_value "$level")"
  timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  if [ "$level_value" -ge "$(log_level_value "$file_level")" ] && [ -n "$log_dir" ]; then
    mkdir -p "$log_dir" 2>/dev/null || true
    printf '%s [%s] [%s] %s\n' "$timestamp" "$level" "$VOICE_ROLE" "$message" >> "$log_dir/$VOICE_ROLE.log" 2>/dev/null || true
  fi
  if [ "$level_value" -ge "$(log_level_value "$console_level")" ]; then
    printf '[%s] %s: %s\n' "$VOICE_ROLE" "$level" "$message" >&2
  fi
}

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
      chat2me_log info "Initialized config: $target_file"
    fi
  done
}

load_runtime_env() {
  [ -f "$RUNTIME_CONFIG_PATH" ] || return

  protected_keys="$(mktemp)"
  env | sed 's/=.*//' > "$protected_keys"
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

    if ! grep -Fxq "$key" "$protected_keys"; then
      export "$key=$value"
    fi
  done < "$RUNTIME_CONFIG_PATH"
  rm -f "$protected_keys"
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
        printf("[llm] %s [%s] %3d%% %.1f/%.1f MB %s\n", label, bar, pct, done / 1048576, total / 1048576, status)
      }'
  else
    awk -v label="$label" -v done="$completed" -v status="$status" '
      BEGIN {
        printf("[llm] %s %.1f MB %s\n", label, done / 1048576, status)
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
      echo "[llm] pull error: $error" >&2
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
      echo "[llm] $MODEL: $status"
    fi
  done <"$fifo"

  curl_status=0
  wait "$curl_pid" || curl_status="$?"
  rm -f "$fifo"

  [ "$curl_status" -eq 0 ] && [ "$api_error" -eq 0 ]
}

print_ollama_log_tail() {
  [ -f "$OLLAMA_LOG_FILE" ] || return 0
  echo "[llm] recent LLM log:" >&2
  tail -n 80 "$OLLAMA_LOG_FILE" >&2 || true
}

prefix_ollama_logs() {
  while IFS= read -r line || [ -n "$line" ]; do
    timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf '%s [info] [%s] [ollama] %s\n' "$timestamp" "$VOICE_ROLE" "$line" >> "$OLLAMA_LOG_FILE" 2>/dev/null || true
  done
}

init_config
load_runtime_env
OLLAMA_LOG_FILE="/app/log/$VOICE_ROLE.log"
mkdir -p "$(dirname "$OLLAMA_LOG_FILE")"
: "${OLLAMA_MODEL:?OLLAMA_MODEL must be set in runtime.env}"
MODEL="$OLLAMA_MODEL"

model_ok() {
  /bin/ollama show "$MODEL" >/dev/null 2>&1 \
    && /bin/ollama run "$MODEL" "只回答OK" >/dev/null 2>&1
}

OLLAMA_LOG_PIPE="/tmp/ollama-log.$$"
rm -f "$OLLAMA_LOG_PIPE"
mkfifo "$OLLAMA_LOG_PIPE"
prefix_ollama_logs <"$OLLAMA_LOG_PIPE" &
OLLAMA_LOGGER_PID="$!"

/bin/ollama serve >"$OLLAMA_LOG_PIPE" 2>&1 &
OLLAMA_PID="$!"

until /bin/ollama list >/dev/null 2>&1; do
  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "[llm] Ollama exited before it became ready" >&2
    print_ollama_log_tail
    exit 1
  fi
  sleep 2
done

(
  echo "[llm] validating local fallback model: $MODEL"
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

  echo "[llm] validating model after download: $MODEL"
  if model_ok; then
    echo "$MODEL is ready"
  else
    echo "$MODEL was downloaded but failed runtime validation" >&2
    print_ollama_log_tail
  fi
) &

trap 'kill "$OLLAMA_PID" "$OLLAMA_LOGGER_PID" 2>/dev/null || true; rm -f "$OLLAMA_LOG_PIPE"' INT TERM
exec "$@"
