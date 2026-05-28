from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import serial


def load_runtime_env() -> None:
    path = Path(os.getenv("RUNTIME_CONFIG_PATH", "/app/config/runtime.env"))
    if not path.is_file():
        return
    protected_keys = set(os.environ)
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.fullmatch(r"[A-Za-z0-9_]+", key):
            continue
        values[key] = value.strip()
    for key, value in values.items():
        if key not in protected_keys:
            os.environ[key] = value


load_runtime_env()


def quiet_http_client_logging() -> None:
    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


quiet_http_client_logging()


def env_value(key: str, *, allow_empty: bool = False) -> str:
    value = os.getenv(key)
    if value is None:
        raise RuntimeError(f"{key} must be set in runtime.env")
    value = value.strip()
    if not allow_empty and not value:
        raise RuntimeError(f"{key} must not be empty in runtime.env")
    return value


def env_int(key: str) -> int:
    value = env_value(key)
    try:
        return int(value)
    except ValueError:
        raise RuntimeError(f"{key} must be an integer in runtime.env") from None


def env_float(key: str) -> float:
    value = env_value(key)
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(f"{key} must be a number in runtime.env") from None


def env_bool(key: str) -> bool:
    value = env_value(key).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{key} must be a boolean in runtime.env")


def log(message: str) -> None:
    role = os.getenv("VOICE_ROLE", "chat2m-speech")
    print(f"[{role}] {message}", flush=True)


class DisplayClient:
    def __init__(
        self,
        port: str,
        baud: int,
        text_max_chars: int | None = None,
        retry_seconds: float | None = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.text_max_chars = env_int("DISPLAY_TEXT_MAX_CHARS") if text_max_chars is None else text_max_chars
        self.retry_seconds = env_float("DISPLAY_SERIAL_RETRY_SECONDS") if retry_seconds is None else retry_seconds
        self._serial: serial.Serial | None = None
        self._lock = threading.Lock()
        self._disabled_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.port)

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def set_state(self, state: str, text: str = "") -> None:
        if not self.enabled:
            return
        if time.monotonic() < self._disabled_until:
            return
        payload = {
            "state": state,
            "text": text[: self.text_max_chars],
            "ts": int(time.time()),
        }
        line = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            try:
                if self._serial is None or not self._serial.is_open:
                    self._serial = serial.Serial(self.port, self.baud, timeout=0, write_timeout=1)
                    time.sleep(0.1)
                written = self._serial.write(line)
                self._serial.flush()
                if written != len(line):
                    raise serial.SerialTimeoutException(
                        f"display serial partial write: {written}/{len(line)} bytes"
                    )
            except serial.SerialException as exc:
                log(f"display serial write failed: {exc}")
                self._close_locked()
                self._disabled_until = time.monotonic() + self.retry_seconds
