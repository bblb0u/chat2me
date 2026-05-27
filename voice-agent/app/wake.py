from __future__ import annotations

import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import sounddevice as sd

from app.agent import (
    INPUT_CHANNELS,
    INPUT_DEVICE,
    CHUNK_SECONDS,
    KWS_MODEL_DIR,
    SAMPLE_RATE,
    create_kws,
    env_float,
    log,
    read_mono,
    select_input_device,
    wake_words_display,
)


SPEECH_WAKE_URL = os.getenv("SPEECH_WAKE_URL", "http://chat2m-speech:8090/wake")
SPEECH_HEALTH_URL = os.getenv("SPEECH_HEALTH_URL", SPEECH_WAKE_URL.rsplit("/", 1)[0] + "/health")
STATUS_URL = os.getenv("STATUS_URL", "http://chat2m-status:8091/state")
SPEECH_WAIT_LOG_SECONDS = env_float("SPEECH_WAIT_LOG_SECONDS")
SPEECH_WAIT_POLL_SECONDS = env_float("SPEECH_WAIT_POLL_SECONDS")
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


def get_speech_health(timeout: float = 1.0) -> dict[str, object] | None:
    if not SPEECH_HEALTH_URL:
        return None
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(SPEECH_HEALTH_URL)
            response.raise_for_status()
            return response.json()
    except Exception:
        return None


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


def wait_for_speech() -> None:
    last_log = 0.0
    set_state("waiting", "speech")
    while True:
        if get_speech_health(timeout=2.0) is not None:
            log("speech service is online")
            set_state("idle")
            return

        now = time.monotonic()
        if now - last_log >= SPEECH_WAIT_LOG_SECONDS:
            log(f"waiting for speech service: {SPEECH_HEALTH_URL}")
            last_log = now
        time.sleep(SPEECH_WAIT_POLL_SECONDS)


def wait_for_speech_idle(observe_busy_timeout: float = 2.0) -> None:
    start = time.monotonic()
    last_log = 0.0
    saw_busy = False
    while True:
        health = get_speech_health()
        if health is not None:
            busy = bool(health.get("busy"))
            if busy:
                saw_busy = True
            elif saw_busy:
                log("speech session finished")
                return
            elif time.monotonic() - start >= observe_busy_timeout:
                log("speech service is idle; no active session observed")
                return

        now = time.monotonic()
        if now - last_log >= SPEECH_WAIT_LOG_SECONDS:
            log(f"waiting for speech session to finish: {SPEECH_HEALTH_URL}")
            last_log = now
        time.sleep(SPEECH_WAIT_POLL_SECONDS)


def listen_for_wake(kws, input_device: int | str | None, chunk: int) -> str:
    stream = kws.create_stream()
    matched = ""
    last_error_log = 0.0
    log(f"wake listener active: {wake_words_display()}")

    while not matched:
        try:
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
        except Exception as exc:
            now = time.monotonic()
            if now - last_error_log >= SPEECH_WAIT_LOG_SECONDS:
                log(f"wake input stream unavailable; retrying: {exc}")
                last_error_log = now
            set_state("error", "audio input unavailable")
            time.sleep(SPEECH_WAIT_POLL_SECONDS)
            stream = kws.create_stream()

    return matched


def main() -> None:
    start_health_server()
    wait_for_speech()
    input_device = select_input_device(INPUT_DEVICE)
    log(f"input device: {input_device if input_device is not None else 'default'}")
    log(f"loading wake-word model: {KWS_MODEL_DIR}")
    kws = create_kws()
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    set_state("idle")
    ready_event.set()

    while True:
        listen_for_wake(kws, input_device, chunk)
        trigger_status = trigger_speech()
        if trigger_status == 202:
            wait_for_speech_idle()
        elif trigger_status == 409:
            wait_for_speech_idle(observe_busy_timeout=0.0)
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
