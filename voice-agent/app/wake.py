from __future__ import annotations

import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlencode

import httpx
import sounddevice as sd

from app.agent import (
    INPUT_CHANNELS,
    INPUT_DEVICE,
    KWS_MODEL_DIR,
    SAMPLE_RATE,
    create_kws,
    log,
    read_mono,
    select_input_device,
    wake_words_display,
)


SPEECH_WAKE_URL = os.getenv("SPEECH_WAKE_URL", "http://chat2m-speech:8090/wake")
SPEECH_HEALTH_URL = os.getenv("SPEECH_HEALTH_URL", SPEECH_WAKE_URL.rsplit("/", 1)[0] + "/health")
STATUS_URL = os.getenv("STATUS_URL", "http://chat2m-status:8091/state")
STATUS_HEALTH_URL = os.getenv("STATUS_HEALTH_URL", STATUS_URL.rsplit("/", 1)[0] + "/health")
STATUS_WAIT_URL = os.getenv("STATUS_WAIT_URL", STATUS_URL.rsplit("/", 1)[0] + "/wait")
SPEECH_WAIT_LOG_SECONDS = float(os.getenv("SPEECH_WAIT_LOG_SECONDS", "30"))
WAKE_HEALTH_HOST = os.getenv("WAKE_HEALTH_HOST", "127.0.0.1")
WAKE_HEALTH_PORT = int(os.getenv("WAKE_HEALTH_PORT", "8092"))
ready_event = threading.Event()


class HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return
        status = HTTPStatus.OK if ready_event.is_set() else HTTPStatus.SERVICE_UNAVAILABLE
        body = b'{"ok":true}\n' if ready_event.is_set() else b'{"ok":false}\n'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_health_server() -> None:
    server = ThreadingHTTPServer((WAKE_HEALTH_HOST, WAKE_HEALTH_PORT), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def post_json(url: str, payload: dict[str, str], timeout: float = 2.0) -> bool:
    if not url:
        return False
    try:
        with httpx.Client(timeout=timeout) as client:
            client.post(url, json=payload).raise_for_status()
        return True
    except Exception as exc:
        log(f"post failed: {url}: {exc}")
        return False


def set_state(state: str, text: str = "") -> None:
    post_json(STATUS_URL, {"state": state, "text": text}, timeout=1.0)


def set_idle() -> None:
    set_state("idle")


def get_status(timeout: float = 1.0) -> dict[str, object] | None:
    if not STATUS_HEALTH_URL:
        return None
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(STATUS_HEALTH_URL)
            response.raise_for_status()
            return response.json()
    except Exception:
        return None


def wait_for_status() -> dict[str, object]:
    while True:
        status = get_status()
        if status is not None:
            return status
        log(f"waiting for status service: {STATUS_HEALTH_URL}")
        time.sleep(1)


def trigger_speech() -> int | None:
    if not SPEECH_WAKE_URL:
        return None
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.post(SPEECH_WAKE_URL, json={"event": "wake"})
            return response.status_code
    except Exception as exc:
        log(f"post failed: {SPEECH_WAKE_URL}: {exc}")
        return None


def int_value(value: object) -> int | None:
    return value if isinstance(value, int) else None


def status_wait(params: dict[str, object]) -> dict[str, object] | None:
    if not STATUS_WAIT_URL:
        return None
    url = f"{STATUS_WAIT_URL}?{urlencode(params)}"
    try:
        with httpx.Client(timeout=None) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        log(f"status wait failed: {url}: {exc}")
        return None


def wait_for_status_change(start_non_idle_seq: int | None) -> int | None:
    params: dict[str, object] = {}
    if start_non_idle_seq is not None:
        params["after_non_idle_seq"] = start_non_idle_seq
    while True:
        log(f"waiting for speech session state: {STATUS_WAIT_URL}")
        status = status_wait(params)
        if status is not None:
            non_idle_seq = int_value(status.get("non_idle_seq"))
            state = str(status.get("state", ""))
            if non_idle_seq is not None and (start_non_idle_seq is None or non_idle_seq > start_non_idle_seq):
                log(f"speech session state observed: {state}")
                return non_idle_seq

        time.sleep(1)


def wait_for_status_idle(start_non_idle_seq: int | None) -> None:
    params: dict[str, object] = {"state": "idle"}
    if start_non_idle_seq is not None:
        params["non_idle_seq_at_least"] = start_non_idle_seq
    while True:
        log(f"waiting for speech session to finish: {STATUS_WAIT_URL}")
        status = status_wait(params)
        if status is not None:
            non_idle_seq = int_value(status.get("non_idle_seq"))
            state = str(status.get("state", ""))
            if (
                non_idle_seq is not None
                and (start_non_idle_seq is None or non_idle_seq >= start_non_idle_seq)
                and state == "idle"
            ):
                log(f"speech session finished with state: {state}")
                return

        time.sleep(1)


def status_is_busy(status: dict[str, object] | None) -> bool:
    if status is None:
        return False
    state = str(status.get("state", ""))
    return bool(state and state != "idle")


def wait_for_speech() -> None:
    last_log = 0.0
    set_state("waiting", "speech")
    while True:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(SPEECH_HEALTH_URL)
                if response.is_success:
                    log("speech service is online")
                    set_state("idle")
                    return
        except Exception:
            pass

        now = time.monotonic()
        if now - last_log >= SPEECH_WAIT_LOG_SECONDS:
            log(f"waiting for speech service: {SPEECH_HEALTH_URL}")
            last_log = now
        time.sleep(2)


def main() -> None:
    start_health_server()
    wait_for_speech()
    input_device = select_input_device(INPUT_DEVICE)
    log(f"input device: {input_device if input_device is not None else 'default'}")
    log(f"loading wake-word model: {KWS_MODEL_DIR}")
    kws = create_kws()
    chunk = int(0.1 * SAMPLE_RATE)
    set_state("idle")
    ready_event.set()

    while True:
        stream = kws.create_stream()
        matched = ""
        log(f"wake listener active: {wake_words_display()}")
        with sd.InputStream(
            channels=INPUT_CHANNELS,
            dtype="float32",
            samplerate=SAMPLE_RATE,
            device=input_device,
            blocksize=chunk,
        ) as audio:
            while not matched:
                samples = read_mono(audio, chunk)
                stream.accept_waveform(SAMPLE_RATE, samples)
                while kws.is_ready(stream):
                    kws.decode_stream(stream)
                    result = kws.get_result(stream)
                    if not result:
                        continue
                    matched = result
                    log(f"wake keyword matched: {matched}")
                    break

        before_wake = wait_for_status()
        before_non_idle_seq = int_value(before_wake.get("non_idle_seq")) if before_wake is not None else None
        trigger_status = trigger_speech()
        if trigger_status == 202:
            active_non_idle_seq = wait_for_status_change(before_non_idle_seq)
            wait_for_status_idle(active_non_idle_seq)
        elif trigger_status == 409:
            current_status = get_status() or before_wake
            if status_is_busy(current_status):
                wait_for_status_idle(before_non_idle_seq)
            else:
                set_idle()
                log("speech service is busy but no active speech state was observed")
        else:
            set_idle()
            if trigger_status is None:
                set_state("error", "speech service unavailable")
                wait_for_speech()
            else:
                set_state("error", f"speech wake failed: {trigger_status}")
                wait_for_speech()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
    except Exception as exc:
        log(f"fatal: {exc}")
        sys.exit(1)
