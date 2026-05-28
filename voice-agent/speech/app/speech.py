from __future__ import annotations

import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import httpx
import sounddevice as sd

from app.agent import (
    ASR_MODEL_DIR,
    CHUNK_SECONDS,
    DISPLAY_SERIAL_BAUD,
    DISPLAY_SERIAL_RETRY_SECONDS,
    DISPLAY_TEXT_MAX_CHARS,
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
    choose_llm_route,
    create_asr,
    create_tts,
    drain_audio,
    handle_conversation_turn,
    listen_command,
    log,
    preload_tts_cache,
    select_input_device,
    speak_pausing_input,
    start_llm_route_cache,
    warmup_tts,
    write_beep,
)
from app.respeaker import direction_answer, open_respeaker


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
                client.post(self.url, json={"state": state, "text": text[:DISPLAY_TEXT_MAX_CHARS]}).raise_for_status()
        except Exception as exc:
            log(f"status forward failed: {exc}")
            self._disabled_until = time.monotonic() + DISPLAY_SERIAL_RETRY_SECONDS


class WakeHandler(BaseHTTPRequestHandler):
    recognizer = None
    voice = None
    tts_config = None
    display = None
    beep_path = None
    audio_source = None
    busy_lock = threading.Lock()

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({"ok": True, "busy": WakeHandler.busy_lock.locked()})
            return
        if path == "/direction":
            self._send_json(direction_snapshot(WakeHandler.audio_source))
            return
        else:
            self.send_response(404)
            self.end_headers()
            return

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
                WakeHandler.audio_source,
            ),
            daemon=True,
        ).start()
        self.send_response(202)
        self.end_headers()


def direction_snapshot(audio_source) -> dict[str, object]:
    if audio_source is None:
        return {"ok": False, "source": "respeaker", "error": "unavailable", "updated_at": time.time()}
    return audio_source.snapshot()


def run_session_thread(recognizer, voice, tts_config, display: StatusClient, beep_path: Path, audio_source) -> None:
    try:
        log("wake signal received")
        run_session(recognizer, voice, tts_config, display, beep_path, audio_source)
    except Exception as exc:
        log(f"session failed: {exc}")
        if display is not None:
            display.set_state("error", str(exc))
    finally:
        if display is not None:
            display.set_state("idle")
        WakeHandler.busy_lock.release()


def open_session_input(input_device: int | str | None, chunk: int) -> sd.InputStream:
    deadline = time.monotonic() + 5.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        audio: sd.InputStream | None = None
        try:
            audio = sd.InputStream(
                channels=INPUT_CHANNELS,
                dtype="float32",
                samplerate=SAMPLE_RATE,
                device=input_device,
                blocksize=chunk,
            )
            audio.start()
            return audio
        except Exception as exc:
            last_error = exc
            if audio is not None:
                try:
                    audio.close()
                except Exception:
                    pass
            log(f"speech input stream unavailable; retrying: {exc}")
            time.sleep(0.25)

    if last_error is not None:
        raise last_error
    raise RuntimeError("speech input stream unavailable")


def run_session(recognizer, voice, tts_config, display: StatusClient, beep_path: Path, audio_source) -> None:
    input_device = select_input_device(INPUT_DEVICE)
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    audio = open_session_input(input_device, chunk)
    try:
        llm_route = choose_llm_route()
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
            direction_reply = direction_answer(command, audio_source)
            if direction_reply is not None:
                log(f"direction answer: {direction_reply}")
                speak_pausing_input(audio, direction_reply, voice, tts_config, display)
                drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)
                continue
            if not handle_conversation_turn(audio, command, voice, tts_config, display, llm_route):
                return
            drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

        log("conversation reached max turns")
        speak_pausing_input(audio, SESSION_END_RESPONSE, voice, tts_config, display)
        display.set_state("idle")
    finally:
        audio.close()


def main() -> None:
    log(f"gateway url: {GATEWAY_URL}")
    log(f"status url: {STATUS_URL}")
    start_llm_route_cache()
    log(f"loading ASR model: {ASR_MODEL_DIR}")
    recognizer = create_asr()
    log("ASR model ready")
    log("loading TTS model")
    voice, tts_config = create_tts()
    log(f"TTS ready: sample_rate={voice.config.sample_rate}")
    preload_tts_cache(voice, WAKE_RESPONSE, SESSION_END_RESPONSE, SESSION_IDLE_RESPONSE)
    warmup_tts(voice)
    audio_source = open_respeaker()
    beep_path = Path("/tmp/chat2m_wake.wav")
    write_beep(beep_path)
    display = StatusClient(STATUS_URL)

    WakeHandler.recognizer = recognizer
    WakeHandler.voice = voice
    WakeHandler.tts_config = tts_config
    WakeHandler.display = display
    WakeHandler.beep_path = beep_path
    WakeHandler.audio_source = audio_source

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
