from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from glob import glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from app.runtime import DisplayClient, env_float, env_int, env_value, log


DISPLAY_SERIAL_PORT = env_value("DISPLAY_SERIAL_PORT", allow_empty=True)
DISPLAY_SERIAL_CANDIDATES_ENV = os.getenv("DISPLAY_SERIAL_CANDIDATES", "")
DISPLAY_SERIAL_CANDIDATES = tuple(candidate.strip() for candidate in DISPLAY_SERIAL_CANDIDATES_ENV.split(",") if candidate.strip())
DISPLAY_SERIAL_BAUD = env_int("DISPLAY_SERIAL_BAUD")
DISPLAY_SYNC_SECONDS = env_float("DISPLAY_SYNC_SECONDS")
DISPLAY_TEXT_MAX_CHARS = env_int("DISPLAY_TEXT_MAX_CHARS")
DISPLAY_SERIAL_RETRY_SECONDS = env_float("DISPLAY_SERIAL_RETRY_SECONDS")
STATUS_WAIT_MAX_TIMEOUT_SECONDS = env_float("STATUS_WAIT_MAX_TIMEOUT_SECONDS")
STATUS_HOST = os.getenv("STATUS_HOST", "0.0.0.0")
STATUS_PORT = int(os.getenv("STATUS_PORT", "8091"))


def resolve_display_port(port: str) -> str:
    if port and port.lower() != "auto":
        return port
    candidates = DISPLAY_SERIAL_CANDIDATES or (
        "/host-dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*-if00",
        "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*-if00",
    )
    for candidate in candidates:
        for resolved in glob(candidate):
            if Path(resolved).exists():
                return resolved
    return ""


display_port = resolve_display_port(DISPLAY_SERIAL_PORT)
display = DisplayClient(
    display_port,
    DISPLAY_SERIAL_BAUD,
    text_max_chars=DISPLAY_TEXT_MAX_CHARS,
    retry_seconds=DISPLAY_SERIAL_RETRY_SECONDS,
)
state_lock = threading.Condition()
last_state = {
    "state": "idle",
    "text": "",
    "seq": 0,
    "changed_at": time.time(),
    "non_idle_seq": 0,
}


def set_state(state: str, text: str = "") -> None:
    with state_lock:
        seq = int(last_state["seq"]) + 1
        last_state["state"] = state
        last_state["text"] = text
        last_state["seq"] = seq
        last_state["changed_at"] = time.time()
        if state != "idle":
            last_state["non_idle_seq"] = seq
        state_lock.notify_all()
    display.set_state(state, text)
    log(f"display state: {state}")


def sync_display_state() -> None:
    while True:
        time.sleep(DISPLAY_SYNC_SECONDS)
        with state_lock:
            state = str(last_state["state"])
            text = str(last_state["text"])
        display.set_state(state, text)


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_timeout(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return min(max(float(value), 0.0), STATUS_WAIT_MAX_TIMEOUT_SECONDS)
    except ValueError:
        return STATUS_WAIT_MAX_TIMEOUT_SECONDS


class StatusHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            with state_lock:
                state = dict(last_state)
            self._send_json(200, {"ok": True, "display": bool(display_port), "port": display_port, **state})
            return
        if parsed.path == "/wait":
            self._wait_state(parse_qs(parsed.query))
            return
        else:
            self._send_json(404, {"error": "not found"})
            return

    def _wait_state(self, query: dict[str, list[str]]) -> None:
        after_seq = parse_int(query.get("after_seq", [None])[0])
        after_non_idle_seq = parse_int(query.get("after_non_idle_seq", [None])[0])
        non_idle_seq_at_least = parse_int(query.get("non_idle_seq_at_least", [None])[0])
        desired_state = query.get("state", [None])[0]
        timeout = parse_timeout(query.get("timeout", [None])[0])

        def matched() -> bool:
            if after_seq is not None and int(last_state["seq"]) <= after_seq:
                return False
            if after_non_idle_seq is not None and int(last_state["non_idle_seq"]) <= after_non_idle_seq:
                return False
            if non_idle_seq_at_least is not None and int(last_state["non_idle_seq"]) < non_idle_seq_at_least:
                return False
            if desired_state is not None and last_state["state"] != desired_state:
                return False
            return True

        with state_lock:
            ok = state_lock.wait_for(matched, timeout=timeout)
            state = dict(last_state)
        status = 200 if ok else 408
        self._send_json(status, {"ok": ok, "display": bool(display_port), "port": display_port, **state})

    def do_POST(self) -> None:
        if self.path != "/state":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            state = str(payload["state"])[:24]
            text = str(payload.get("text", ""))[:DISPLAY_TEXT_MAX_CHARS]
        except Exception:
            self._send_json(400, {"error": "invalid state payload"})
            return

        set_state(state, text)
        self._send_json(200, {"ok": True})


def main() -> None:
    log(f"display serial: {display_port or 'disabled'}")
    set_state("idle")
    if display.enabled and DISPLAY_SYNC_SECONDS > 0:
        threading.Thread(target=sync_display_state, daemon=True).start()
        log(f"display sync interval: {DISPLAY_SYNC_SECONDS:g}s")
    server = ThreadingHTTPServer((STATUS_HOST, STATUS_PORT), StatusHandler)
    log(f"status forwarder listening on {STATUS_HOST}:{STATUS_PORT}")
    try:
        server.serve_forever()
    finally:
        display.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
    except Exception as exc:
        log(f"fatal: {exc}")
        sys.exit(1)
