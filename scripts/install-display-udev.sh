#!/bin/sh
set -eu

DISPLAY_DEVICE="${1:-/dev/ttyACM0}"
RULE_PATH="${RULE_PATH:-/etc/udev/rules.d/99-chat2m-display.rules}"
SYMLINK_NAME="${SYMLINK_NAME:-chat2m-display}"

if [ ! -e "$DISPLAY_DEVICE" ]; then
  echo "display device not found: $DISPLAY_DEVICE" >&2
  exit 1
fi

vendor_id="$(udevadm info --query=property --name="$DISPLAY_DEVICE" | awk -F= '$1=="ID_VENDOR_ID"{print $2; exit}')"
model_id="$(udevadm info --query=property --name="$DISPLAY_DEVICE" | awk -F= '$1=="ID_MODEL_ID"{print $2; exit}')"
serial_short="$(udevadm info --query=property --name="$DISPLAY_DEVICE" | awk -F= '$1=="ID_SERIAL_SHORT"{print $2; exit}')"

if [ -z "$vendor_id" ] || [ -z "$model_id" ] || [ -z "$serial_short" ]; then
  echo "failed to read USB identity from $DISPLAY_DEVICE" >&2
  exit 1
fi

cat > "$RULE_PATH" <<EOF
SUBSYSTEM=="tty", ENV{ID_VENDOR_ID}=="$vendor_id", ENV{ID_MODEL_ID}=="$model_id", ENV{ID_SERIAL_SHORT}=="$serial_short", SYMLINK+="$SYMLINK_NAME", MODE="0666"
EOF

udevadm control --reload-rules
udevadm trigger --subsystem-match=tty || true

echo "installed $RULE_PATH"
echo "/dev/$SYMLINK_NAME -> $(readlink "/dev/$SYMLINK_NAME" 2>/dev/null || echo pending)"
