#!/bin/sh
set -eu

DEFAULT_CONFIG_DIR="${DEFAULT_CONFIG_DIR:-/defaults/config}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_PATH:-$CONFIG_DIR/runtime.env}"

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

if [ -d "$DEFAULT_CONFIG_DIR" ]; then
  mkdir -p "$CONFIG_DIR"
  for source_file in "$DEFAULT_CONFIG_DIR"/*; do
    [ -f "$source_file" ] || continue
    target_file="$CONFIG_DIR/$(basename "$source_file")"
    if [ ! -e "$target_file" ]; then
      cp "$source_file" "$target_file"
      echo "Initialized config: $target_file"
    fi
  done
fi

sync_runtime_env_defaults

exec "$@"
