#!/bin/sh
set -eu

MODEL="${OLLAMA_MODEL:-qwen3:4b-instruct}"

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
