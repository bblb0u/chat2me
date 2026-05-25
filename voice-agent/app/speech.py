from __future__ import annotations

import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import sounddevice as sd

from app.agent import (
    ASR_MODEL_DIR,
    DISPLAY_SERIAL_BAUD,
    GATEWAY_URL,
    INPUT_CHANNELS,
    INPUT_DEVICE,
    MAX_SESSION_TURNS,
    POST_RESPONSE_DRAIN_SECONDS,
    SAMPLE_RATE,
    SESSION_END_RESPONSE,
    SESSION_IDLE_RESPONSE,
    WAKE_RESPONSE,
    DisplayClient,
    create_asr,
    create_tts,
    drain_audio,
    handle_conversation_turn,
    listen_command,
    log,
    select_input_device,
    speak_pausing_input,
    write_beep,
)


SPEECH_HOST = os.getenv("SPEECH_HOST", "0.0.0.0")
SPEECH_PORT = int(os.getenv("SPEECH_PORT", "8090"))
STATUS_URL = os.getenv("STATUS_URL", "http://chat2m-status:8091/state")


class StatusClient(DisplayClient):
    def __init__(self, url: str) -> None:
        super().__init__("", DISPLAY_SERIAL_BAUD)
        self.url = url
        self._disabled_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def set_state(self, state: str, text: str = "") -> None:
        if not self.enabled or time.monotonic() < self._disabled_until:
            return
        try:
            with httpx.Client(timeout=2.0) as client:
                client.post(self.url, json={"state": state, "text": text[:80]}).raise_for_status()
        except Exception as exc:
            log(f"status forward failed: {exc}")
            self._disabled_until = time.monotonic() + 2.0


class WakeHandler(BaseHTTPRequestHandler):
    recognizer = None
    voice = None
    tts_config = None
    display = None
    beep_path = None
    busy_lock = threading.Lock()

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        self._send_json({"ok": True, "busy": WakeHandler.busy_lock.locked()})

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/wake":
            self.send_response(404)
            self.end_headers()
            return

        if not WakeHandler.busy_lock.acquire(blocking=False):
            self.send_response(409)
            self.end_headers()
            return
        threading.Thread(
            target=run_session_thread,
            args=(
                WakeHandler.recognizer,
                WakeHandler.voice,
                WakeHandler.tts_config,
                WakeHandler.display,
                WakeHandler.beep_path,
            ),
            daemon=True,
        ).start()
        self.send_response(202)
        self.end_headers()


def run_session_thread(recognizer, voice, tts_config, display: StatusClient, beep_path: Path) -> None:
    try:
        log("wake signal received")
        run_session(recognizer, voice, tts_config, display, beep_path)
    except Exception as exc:
        log(f"session failed: {exc}")
        if display is not None:
            display.set_state("error", str(exc))
    finally:
        if display is not None:
            display.set_state("idle")
        WakeHandler.busy_lock.release()


def run_session(recognizer, voice, tts_config, display: StatusClient, beep_path: Path) -> None:
    input_device = select_input_device(INPUT_DEVICE)
    chunk = int(0.1 * SAMPLE_RATE)
    with sd.InputStream(
        channels=INPUT_CHANNELS,
        dtype="float32",
        samplerate=SAMPLE_RATE,
        device=input_device,
        blocksize=chunk,
    ) as audio:
        display.set_state("listening", "wake")
        speak_pausing_input(audio, WAKE_RESPONSE, voice, tts_config, display)
        drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

        for turn in range(1, MAX_SESSION_TURNS + 1):
            log(f"conversation turn {turn}/{MAX_SESSION_TURNS}")
            command = listen_command(audio, beep_path, recognizer, play_ready_beep=False)
            if not command:
                log("conversation idle timeout")
                if SESSION_IDLE_RESPONSE:
                    speak_pausing_input(audio, SESSION_IDLE_RESPONSE, voice, tts_config, display)
                display.set_state("idle")
                return
            if not handle_conversation_turn(audio, command, voice, tts_config, display):
                return
            drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

        log("conversation reached max turns")
        speak_pausing_input(audio, SESSION_END_RESPONSE, voice, tts_config, display)
        display.set_state("idle")


def main() -> None:
    log(f"gateway url: {GATEWAY_URL}")
    log(f"status url: {STATUS_URL}")
    log(f"loading ASR model: {ASR_MODEL_DIR}")
    recognizer = create_asr()
    log("ASR model ready")
    log("loading Piper TTS model")
    voice, tts_config = create_tts()
    log(f"Piper TTS ready: sample_rate={voice.config.sample_rate}")
    beep_path = Path("/tmp/chat2m_wake.wav")
    write_beep(beep_path)
    display = StatusClient(STATUS_URL)

    WakeHandler.recognizer = recognizer
    WakeHandler.voice = voice
    WakeHandler.tts_config = tts_config
    WakeHandler.display = display
    WakeHandler.beep_path = beep_path

    display.set_state("idle")
    server = ThreadingHTTPServer((SPEECH_HOST, SPEECH_PORT), WakeHandler)
    log(f"speech service listening on {SPEECH_HOST}:{SPEECH_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
    except Exception as exc:
        log(f"fatal: {exc}")
        sys.exit(1)
