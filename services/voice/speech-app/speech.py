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
    CORE_URL,
    INPUT_CHANNELS,
    INPUT_DEVICE,
    INPUT_DEVICE_REQUIRED,
    KWS_MODEL_DIR,
    MAX_SESSION_TURNS,
    POST_RESPONSE_DRAIN_SECONDS,
    SAMPLE_RATE,
    SESSION_END_RESPONSE,
    SESSION_IDLE_RESPONSE,
    WAKE_RESPONSE,
    DisplayClient,
    create_asr,
    create_kws,
    create_remote_asr,
    create_remote_tts,
    create_tts,
    drain_audio,
    env_float,
    handle_conversation_turn,
    listen_command,
    log,
    preload_tts_cache,
    read_mono,
    select_input_device,
    speak_pausing_input,
    start_asr_route_cache,
    start_llm_route_cache,
    start_tts_route_cache,
    warmup_tts,
    wake_words_display,
    write_beep,
)
from app.respeaker import direction_answer_from_snapshot, direction_label, open_respeaker


SPEECH_HOST = os.getenv("SPEECH_HOST", "0.0.0.0")
SPEECH_PORT = int(os.getenv("SPEECH_PORT", "8090"))
DIRECTION_CHANGE_POLL_SECONDS = float(os.getenv("DIRECTION_CHANGE_POLL_SECONDS", "0.5").strip() or "0.5")
SPEECH_WAIT_LOG_SECONDS = env_float("SPEECH_WAIT_LOG_SECONDS")
SPEECH_WAIT_POLL_SECONDS = env_float("SPEECH_WAIT_POLL_SECONDS")


STATE_LOCK = threading.Lock()
CURRENT_STATE: dict[str, object] = {
    "state": "idle",
    "text": "",
    "direction": {
        "ok": False,
        "angle_degrees": None,
        "error": "not_reported",
        "updated_at": time.time(),
    },
    "seq": 0,
    "changed_at": time.time(),
}


class SpeechState(DisplayClient):
    def __init__(self) -> None:
        super().__init__("", DISPLAY_SERIAL_BAUD)
        self.audio_source = None

    @property
    def enabled(self) -> bool:
        return True

    def set_state(self, state: str, text: str = "") -> None:
        text = text[:DISPLAY_TEXT_MAX_CHARS]
        update_speech_state(state=state, text=text, direction=direction_status(self.audio_source))

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
        if path == "/state":
            self._send_json(update_speech_state(direction=direction_status(WakeHandler.audio_source)))
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


def direction_status(audio_source) -> dict[str, object]:
    snapshot = direction_snapshot(audio_source)
    if not snapshot.get("ok"):
        return {
            "ok": False,
            "angle_degrees": None,
            "updated_at": snapshot.get("updated_at", time.time()),
            "error": snapshot.get("error", "unavailable"),
        }
    return {
        "ok": True,
        "angle_degrees": snapshot.get("angle_degrees"),
        "updated_at": snapshot.get("updated_at", time.time()),
    }


def normalize_direction(direction: dict[str, object]) -> dict[str, object]:
    ok = bool(direction.get("ok"))
    angle = direction.get("angle_degrees")
    if ok and isinstance(angle, (int, float)):
        return {
            "ok": True,
            "angle_degrees": int(round(float(angle))) % 360,
            "updated_at": direction.get("updated_at") or time.time(),
        }
    return {
        "ok": False,
        "angle_degrees": None,
        "error": str(direction.get("error") or "unavailable"),
        "updated_at": direction.get("updated_at") or time.time(),
    }


def current_state() -> dict[str, object]:
    with STATE_LOCK:
        payload = dict(CURRENT_STATE)
        payload["direction"] = dict(CURRENT_STATE.get("direction") or {})
        return payload


def direction_bucket(direction: dict[str, object]) -> str | None:
    if not direction.get("ok"):
        return None
    angle = direction.get("angle_degrees")
    if not isinstance(angle, (int, float)):
        return None
    return direction_label(float(angle))


def update_speech_state(
    *,
    state: str | None = None,
    text: str | None = None,
    direction: dict[str, object] | None = None,
    force_seq: bool = False,
) -> dict[str, object]:
    with STATE_LOCK:
        changed = force_seq
        if state is not None and CURRENT_STATE["state"] != state:
            CURRENT_STATE["state"] = state
            changed = True
        if text is not None and CURRENT_STATE["text"] != text:
            CURRENT_STATE["text"] = text
            changed = True
        if direction is not None:
            normalized = normalize_direction(direction)
            old_direction = CURRENT_STATE.get("direction") or {}
            if direction_bucket(old_direction) != direction_bucket(normalized) or old_direction.get("ok") != normalized.get("ok"):
                changed = True
            CURRENT_STATE["direction"] = normalized
        if changed:
            CURRENT_STATE["seq"] = int(CURRENT_STATE["seq"]) + 1
            CURRENT_STATE["changed_at"] = time.time()
        payload = dict(CURRENT_STATE)
        payload["direction"] = dict(CURRENT_STATE.get("direction") or {})
        return payload


def watch_direction_changes(state_store: SpeechState) -> None:
    interval = max(0.2, DIRECTION_CHANGE_POLL_SECONDS)
    last_bucket: str | None = None
    while True:
        payload = update_speech_state(direction=direction_status(state_store.audio_source))
        bucket = direction_bucket(dict(payload.get("direction") or {}))
        if bucket != last_bucket:
            last_bucket = bucket
        time.sleep(interval)


def run_session_thread(recognizer, voice, tts_config, display: SpeechState, beep_path: Path, audio_source) -> None:
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


def listen_for_wake(kws, input_device: int | str | None, chunk: int, display: SpeechState) -> str:
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
            display.set_state("error", "audio input unavailable")
            time.sleep(SPEECH_WAIT_POLL_SECONDS)
            stream = kws.create_stream()

    return matched


def wait_for_input_device(display: SpeechState) -> int | str | None:
    last_log = 0.0
    while True:
        input_device = select_input_device(INPUT_DEVICE)
        if input_device is not None or not INPUT_DEVICE_REQUIRED:
            return input_device

        now = time.monotonic()
        if now - last_log >= SPEECH_WAIT_LOG_SECONDS:
            log(f"waiting for configured input device: {INPUT_DEVICE}")
            last_log = now
        display.set_state("error", "audio input unavailable")
        time.sleep(SPEECH_WAIT_POLL_SECONDS)


def run_embedded_wake_loop(recognizer, voice, tts_config, display: SpeechState, beep_path: Path, audio_source) -> None:
    input_device = wait_for_input_device(display)
    log(f"input device: {input_device if input_device is not None else 'default'}")
    log(f"loading wake-word model: {KWS_MODEL_DIR}")
    kws = create_kws()
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    display.set_state("idle")

    while True:
        listen_for_wake(kws, input_device, chunk, display)
        if not WakeHandler.busy_lock.acquire(blocking=False):
            log("wake ignored because a session is already running")
            continue
        run_session_thread(recognizer, voice, tts_config, display, beep_path, audio_source)


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


def run_session(recognizer, voice, tts_config, display: SpeechState, beep_path: Path, audio_source) -> None:
    input_device = wait_for_input_device(display)
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    audio = open_session_input(input_device, chunk)
    try:
        log_llm_online_cache()
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
            direction_payload = update_speech_state(direction=direction_status(audio_source))
            direction_reply = direction_answer_from_snapshot(command, dict(direction_payload.get("direction") or {}))
            if direction_reply is not None:
                log(f"direction answer: {direction_reply}")
                speak_pausing_input(audio, direction_reply, voice, tts_config, display)
                drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)
                continue
            if not handle_conversation_turn(audio, command, voice, tts_config, display):
                return
            drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

        log("conversation reached max turns")
        speak_pausing_input(audio, SESSION_END_RESPONSE, voice, tts_config, display)
        display.set_state("idle")
    finally:
        audio.close()


def main() -> None:
    log(f"core url: {CORE_URL}")
    start_llm_route_cache()
    start_asr_route_cache()
    start_tts_route_cache()
    log("using remote ASR service")
    recognizer = create_remote_asr()
    log("using remote TTS service")
    voice, tts_config = create_remote_tts()
    audio_source = open_respeaker()
    beep_path = Path("/tmp/chat2me_wake.wav")
    write_beep(beep_path)
    display = SpeechState()
    display.audio_source = audio_source
    threading.Thread(target=watch_direction_changes, args=(display,), daemon=True).start()

    WakeHandler.recognizer = recognizer
    WakeHandler.voice = voice
    WakeHandler.tts_config = tts_config
    WakeHandler.display = display
    WakeHandler.beep_path = beep_path
    WakeHandler.audio_source = audio_source

    display.set_state("idle")
    server = ThreadingHTTPServer((SPEECH_HOST, SPEECH_PORT), WakeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"speech service listening on {SPEECH_HOST}:{SPEECH_PORT}")
    run_embedded_wake_loop(recognizer, voice, tts_config, display, beep_path, audio_source)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
    except Exception as exc:
        log(f"fatal: {exc}")
        sys.exit(1)
    log_llm_online_cache,
