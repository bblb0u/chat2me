from __future__ import annotations

import gc
import json
import os
import re
import threading
import subprocess
import sys
import tempfile
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from piper.config import SynthesisConfig
from piper.voice import PiperVoice
import yaml
import serial
import sherpa_onnx
import sounddevice as sd


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


MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
VOICE_KWS_MODEL_NAME = env_value("VOICE_KWS_MODEL_NAME")
VOICE_ASR_MODEL_NAME = env_value("VOICE_ASR_MODEL_NAME")
VOICE_PIPER_MODEL_NAME = env_value("VOICE_PIPER_MODEL_NAME")
KWS_MODEL_DIR = MODELS_DIR / VOICE_KWS_MODEL_NAME
ASR_MODEL_DIR = MODELS_DIR / VOICE_ASR_MODEL_NAME
GENERATED_KEYWORDS_FILE = MODELS_DIR / "wake_words.txt"
GENERATED_KEYWORDS_RAW = MODELS_DIR / "wake_words_raw.txt"
HOTWORDS_PATH = Path("/app/config/hotwords.yaml")
GENERATED_HOTWORDS_FILE = MODELS_DIR / "hotwords"
WAKE_WORDS_ENV = env_value("WAKE_WORDS")
WAKE_WORDS = tuple(
    word.strip()
    for word in WAKE_WORDS_ENV.split(",")
    if word.strip()
)
if not WAKE_WORDS:
    raise RuntimeError("WAKE_WORDS must contain at least one wake word in runtime.env")

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://chat2m-gateway:8080/chat")
GATEWAY_REACHABILITY_URL = os.getenv("GATEWAY_REACHABILITY_URL", GATEWAY_URL.rsplit("/", 1)[0] + "/llm/reachability")
NETWORK_UNAVAILABLE_RESPONSE = env_value("NETWORK_UNAVAILABLE_RESPONSE")
LLM_ROUTE_CACHE_INTERVAL_SECONDS = env_float("LLM_ROUTE_CACHE_INTERVAL_SECONDS")
DISPLAY_SERIAL_PORT = env_value("DISPLAY_SERIAL_PORT", allow_empty=True)
DISPLAY_SERIAL_BAUD = env_int("DISPLAY_SERIAL_BAUD")
INPUT_DEVICE = env_value("AUDIO_INPUT_DEVICE", allow_empty=True)
OUTPUT_DEVICE = env_value("AUDIO_OUTPUT_DEVICE", allow_empty=True)
SAMPLE_RATE = env_int("AUDIO_SAMPLE_RATE")
CHUNK_SECONDS = env_float("AUDIO_CHUNK_SECONDS")
INPUT_CHANNELS = env_int("AUDIO_INPUT_CHANNELS")
INPUT_CHANNEL_INDEX_RAW = env_value("AUDIO_INPUT_CHANNEL_INDEX")
INPUT_CHANNEL_INDEX_AUTO = INPUT_CHANNEL_INDEX_RAW.lower() == "auto"
if INPUT_CHANNEL_INDEX_AUTO:
    INPUT_CHANNEL_INDEX = 0
else:
    try:
        INPUT_CHANNEL_INDEX = int(INPUT_CHANNEL_INDEX_RAW)
    except ValueError:
        raise RuntimeError("AUDIO_INPUT_CHANNEL_INDEX must be an integer or auto in runtime.env") from None
KWS_THREADS = env_int("KWS_THREADS")
ASR_THREADS = env_int("ASR_THREADS")
ASR_MODEL_PRECISION = env_value("ASR_MODEL_PRECISION")
ASR_DECODING_METHOD = env_value("ASR_DECODING_METHOD")
ASR_MAX_ACTIVE_PATHS = env_int("ASR_MAX_ACTIVE_PATHS")
ASR_MODELING_UNIT = env_value("ASR_MODELING_UNIT")
ASR_HOTWORDS_SCORE = env_float("ASR_HOTWORDS_SCORE")
GATEWAY_REQUEST_TIMEOUT_SECONDS = env_float("GATEWAY_REQUEST_TIMEOUT_SECONDS")
GATEWAY_UNAVAILABLE_RESPONSE = env_value("GATEWAY_UNAVAILABLE_RESPONSE")
ASR_ERROR_RESPONSE = env_value("ASR_ERROR_RESPONSE")
COMMAND_TIMEOUT_SECONDS = env_float("COMMAND_TIMEOUT_SECONDS")
COMMAND_MIN_SECONDS = env_float("COMMAND_MIN_SECONDS")
COMMAND_LEADING_SILENCE_SECONDS = env_float("COMMAND_LEADING_SILENCE_SECONDS")
COMMAND_INITIAL_GRACE_SECONDS = env_float("COMMAND_INITIAL_GRACE_SECONDS")
PRE_BEEP_DRAIN_SECONDS = env_float("PRE_BEEP_DRAIN_SECONDS")
POST_BEEP_DRAIN_SECONDS = env_float("POST_BEEP_DRAIN_SECONDS")
POST_RESPONSE_DRAIN_SECONDS = env_float("POST_RESPONSE_DRAIN_SECONDS")
SPEECH_RMS_THRESHOLD = env_float("SPEECH_RMS_THRESHOLD")
ASR_NOISE_GATE_ENABLED = env_bool("ASR_NOISE_GATE_ENABLED")
ASR_NOISE_CALIBRATION_SECONDS = env_float("ASR_NOISE_CALIBRATION_SECONDS")
ASR_NOISE_GATE_PERCENTILE = env_float("ASR_NOISE_GATE_PERCENTILE")
ASR_NOISE_GATE_RATIO = env_float("ASR_NOISE_GATE_RATIO")
ASR_NOISE_GATE_OFFSET = env_float("ASR_NOISE_GATE_OFFSET")
ASR_PREROLL_SECONDS = env_float("ASR_PREROLL_SECONDS")
PIPER_MODEL = MODELS_DIR / "piper" / VOICE_PIPER_MODEL_NAME / "model.onnx"
PIPER_CONFIG = Path(str(PIPER_MODEL) + ".json")
PIPER_SPEAKER = env_int("PIPER_SPEAKER")
PIPER_LENGTH_SCALE = env_float("PIPER_LENGTH_SCALE")
PIPER_NOISE_SCALE = env_float("PIPER_NOISE_SCALE")
PIPER_NOISE_W_SCALE = env_float("PIPER_NOISE_W_SCALE")
PIPER_VOLUME = env_float("PIPER_VOLUME")
TTS_PLAYER_TIMEOUT_SECONDS = env_float("TTS_PLAYER_TIMEOUT_SECONDS")
DISPLAY_TEXT_MAX_CHARS = env_int("DISPLAY_TEXT_MAX_CHARS")
DISPLAY_SERIAL_RETRY_SECONDS = env_float("DISPLAY_SERIAL_RETRY_SECONDS")
NO_COMMAND_RESPONSE = env_value("NO_COMMAND_RESPONSE")
WAKE_RESPONSE = env_value("WAKE_RESPONSE")
SESSION_IDLE_RESPONSE = env_value("SESSION_IDLE_RESPONSE", allow_empty=True)
SESSION_END_RESPONSE = env_value("SESSION_END_RESPONSE")
SESSION_END_PHRASES_ENV = env_value("SESSION_END_PHRASES")
MAX_SESSION_TURNS = env_int("MAX_SESSION_TURNS")
SESSION_END_PHRASES = tuple(
    phrase.strip()
    for phrase in SESSION_END_PHRASES_ENV.split(",")
    if phrase.strip()
)
if not SESSION_END_PHRASES:
    raise RuntimeError("SESSION_END_PHRASES must contain at least one phrase in runtime.env")
KWS_KEYWORDS_SCORE = env_float("KWS_KEYWORDS_SCORE")
KWS_KEYWORDS_THRESHOLD = env_float("KWS_KEYWORDS_THRESHOLD")
ASR_RULE1_MIN_TRAILING_SILENCE = env_float("ASR_RULE1_MIN_TRAILING_SILENCE")
ASR_RULE2_MIN_TRAILING_SILENCE = env_float("ASR_RULE2_MIN_TRAILING_SILENCE")
ASR_RULE3_MIN_UTTERANCE_LENGTH = env_float("ASR_RULE3_MIN_UTTERANCE_LENGTH")
LLM_ROUTE_CACHE = {
    "route": "local",
    "provider": "",
    "model": "",
    "status": "not_checked",
    "updated_at": 0.0,
}
LLM_ROUTE_CACHE_LOCK = threading.Lock()


def log(message: str) -> None:
    role = os.getenv("VOICE_ROLE", "chat2m-speech")
    print(f"[{role}] {message}", flush=True)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")


class DisplayClient:
    def __init__(self, port: str, baud: int) -> None:
        self.port = port
        self.baud = baud
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
            "text": text[:DISPLAY_TEXT_MAX_CHARS],
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
                self._disabled_until = time.monotonic() + DISPLAY_SERIAL_RETRY_SECONDS


def wake_words_display() -> str:
    return " / ".join(WAKE_WORDS)


def select_input_device(selector: str) -> int | str | None:
    if not selector:
        return None
    if selector.isdigit():
        return int(selector)

    devices = sd.query_devices()
    selector_lower = selector.lower()
    fallback_index: int | None = None
    for index, device in enumerate(devices):
        if selector_lower not in str(device.get("name", "")).lower():
            continue
        if device.get("max_input_channels", 0) > 0:
            return index
        if fallback_index is None:
            fallback_index = index

    if fallback_index is not None:
        log(
            f"input device containing '{selector}' reports no input channels; "
            f"trying device {fallback_index} anyway"
        )
        return fallback_index

    log(f"input device containing '{selector}' not found; using PortAudio default")
    return None


def ensure_keywords_file() -> Path:
    require_file(KWS_MODEL_DIR / "tokens.txt")
    require_file(KWS_MODEL_DIR / "en.phone")

    raw_file = GENERATED_KEYWORDS_RAW
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("".join(f"{word} @{word}\n" for word in WAKE_WORDS), encoding="utf-8")

    keywords_file = GENERATED_KEYWORDS_FILE
    keywords_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sherpa-onnx-cli",
        "text2token",
        "--tokens",
        str(KWS_MODEL_DIR / "tokens.txt"),
        "--tokens-type",
        "phone+ppinyin",
        "--lexicon",
        str(KWS_MODEL_DIR / "en.phone"),
        str(raw_file),
        str(keywords_file),
    ]
    log(f"generating wake keyword tokens: {wake_words_display()}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"wake keyword token generation failed with exit code {result.returncode}")
    return keywords_file


def create_kws() -> sherpa_onnx.KeywordSpotter:
    keywords_file = ensure_keywords_file()
    require_file(KWS_MODEL_DIR / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
    require_file(KWS_MODEL_DIR / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx")
    require_file(KWS_MODEL_DIR / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx")

    return sherpa_onnx.KeywordSpotter(
        tokens=str(KWS_MODEL_DIR / "tokens.txt"),
        encoder=str(KWS_MODEL_DIR / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        decoder=str(KWS_MODEL_DIR / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
        joiner=str(KWS_MODEL_DIR / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        num_threads=KWS_THREADS,
        keywords_file=str(keywords_file),
        keywords_score=KWS_KEYWORDS_SCORE,
        keywords_threshold=KWS_KEYWORDS_THRESHOLD,
        provider="cpu",
    )


def asr_model_file(stem: str) -> Path:
    precision = ASR_MODEL_PRECISION.strip().lower()
    if precision in {"fp32", "float32", "full"}:
        return ASR_MODEL_DIR / f"{stem}.onnx"
    if precision in {"int8", "quantized"}:
        return ASR_MODEL_DIR / f"{stem}.int8.onnx"
    raise RuntimeError("ASR_MODEL_PRECISION must be fp32 or int8 in runtime.env")


def ensure_hotwords_file() -> str:
    if not HOTWORDS_PATH.is_file():
        raise FileNotFoundError(f"missing hotwords file: {HOTWORDS_PATH}")

    with HOTWORDS_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"hotwords file must be a YAML mapping: {HOTWORDS_PATH}")
    raw_hotwords = data.get("hotwords", [])
    if not isinstance(raw_hotwords, list):
        raise RuntimeError(f"hotwords must be a YAML list: {HOTWORDS_PATH}")
    hotwords = tuple(str(word).strip() for word in raw_hotwords if str(word).strip())
    if not hotwords:
        return ""

    require_file(ASR_MODEL_DIR / "tokens.txt")
    bpe_model = ASR_MODEL_DIR / "bpe.model"
    bpe_model_arg = str(bpe_model) if bpe_model.is_file() else None
    try:
        tokenized = sherpa_onnx.text2token(
            list(hotwords),
            str(ASR_MODEL_DIR / "tokens.txt"),
            tokens_type=ASR_MODELING_UNIT,
            bpe_model=bpe_model_arg,
        )
    except Exception as exc:
        raise RuntimeError(f"ASR hotword tokenization failed: {exc}") from exc

    lines = [" ".join(str(token) for token in tokens) for tokens in tokenized if tokens]
    if not lines:
        return ""

    GENERATED_HOTWORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_HOTWORDS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"ASR hotwords active: {' / '.join(hotwords)}")
    return str(GENERATED_HOTWORDS_FILE)


def create_asr() -> sherpa_onnx.OnlineRecognizer:
    require_file(ASR_MODEL_DIR / "tokens.txt")
    encoder = asr_model_file("encoder-epoch-99-avg-1")
    decoder = asr_model_file("decoder-epoch-99-avg-1")
    joiner = asr_model_file("joiner-epoch-99-avg-1")
    require_file(encoder)
    require_file(decoder)
    require_file(joiner)
    hotwords_file = ensure_hotwords_file()
    bpe_vocab = ASR_MODEL_DIR / "bpe.vocab"
    log(
        "ASR config: "
        f"precision={ASR_MODEL_PRECISION} decoding={ASR_DECODING_METHOD} "
        f"max_active_paths={ASR_MAX_ACTIVE_PATHS} hotwords={HOTWORDS_PATH}"
    )

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(ASR_MODEL_DIR / "tokens.txt"),
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        num_threads=ASR_THREADS,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=ASR_RULE1_MIN_TRAILING_SILENCE,
        rule2_min_trailing_silence=ASR_RULE2_MIN_TRAILING_SILENCE,
        rule3_min_utterance_length=ASR_RULE3_MIN_UTTERANCE_LENGTH,
        decoding_method=ASR_DECODING_METHOD,
        max_active_paths=ASR_MAX_ACTIVE_PATHS,
        hotwords_file=hotwords_file,
        hotwords_score=ASR_HOTWORDS_SCORE,
        modeling_unit=ASR_MODELING_UNIT,
        bpe_vocab=str(bpe_vocab) if bpe_vocab.is_file() else "",
        provider="cpu",
    )


def write_beep(path: Path) -> None:
    sample_rate = 16000
    duration = 0.13
    freq = 880
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.25 * np.sin(2 * np.pi * freq * t)
    samples = np.int16(tone * 32767)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())


def play_wav(path: Path) -> None:
    try:
        subprocess.run(["aplay", "-q", "-D", OUTPUT_DEVICE, str(path)], check=False)
    except FileNotFoundError as exc:
        log(f"aplay is unavailable: {exc}")


def create_tts() -> tuple[PiperVoice, SynthesisConfig]:
    require_file(PIPER_MODEL)
    require_file(PIPER_CONFIG)
    voice = PiperVoice.load(PIPER_MODEL, config_path=PIPER_CONFIG)
    config = SynthesisConfig(
        speaker_id=PIPER_SPEAKER,
        length_scale=PIPER_LENGTH_SCALE,
        noise_scale=PIPER_NOISE_SCALE,
        noise_w_scale=PIPER_NOISE_W_SCALE,
        volume=PIPER_VOLUME,
    )
    return voice, config


def audio_chunk_bytes(chunk: Any) -> bytes:
    int16_audio = getattr(chunk, "audio_int16_array", None)
    if int16_audio is not None:
        return np.asarray(int16_audio, dtype=np.int16).tobytes()

    float_audio = getattr(chunk, "audio_float_array", None)
    if float_audio is None:
        raise RuntimeError(f"unsupported Piper audio chunk: {type(chunk)!r}")

    clipped = np.clip(np.asarray(float_audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def speak(text: str, voice: PiperVoice, config: SynthesisConfig) -> None:
    if not text:
        return
    log(f"piper tts: speaker={PIPER_SPEAKER} text={text}")
    command = [
        "aplay",
        "-q",
        "-D",
        OUTPUT_DEVICE,
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(voice.config.sample_rate),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as player:
        assert player.stdin is not None
        try:
            for chunk in voice.synthesize(text, config):
                player.stdin.write(audio_chunk_bytes(chunk))
        finally:
            player.stdin.close()
        return_code = player.wait(timeout=TTS_PLAYER_TIMEOUT_SECONDS)
    if return_code != 0:
        raise RuntimeError(f"aplay exited with status {return_code}")


def speak_pausing_input(
    audio: sd.InputStream,
    text: str,
    voice: PiperVoice,
    config: SynthesisConfig,
    display: DisplayClient,
) -> None:
    if not text:
        return
    display.set_state("speaking", text)
    audio.stop()
    try:
        speak(text, voice, config)
    finally:
        audio.start()
        display.set_state("listening")


def drain_audio(audio: sd.InputStream, seconds: float) -> None:
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        audio.read(chunk)


def mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if INPUT_CHANNEL_INDEX_AUTO:
        channel_rms = np.sqrt(np.mean(np.square(samples), axis=0))
        channel_index = int(np.argmax(channel_rms))
        return samples[:, channel_index]
    channel_index = min(max(INPUT_CHANNEL_INDEX, 0), samples.shape[1] - 1)
    return samples[:, channel_index]


def read_mono(audio: sd.InputStream, frames: int) -> np.ndarray:
    samples, overflowed = audio.read(frames)
    if overflowed:
        log("audio input overflowed; command audio may be clipped")
    return mono(samples).reshape(-1)


def audio_rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def decode_ready_asr(recognizer: sherpa_onnx.OnlineRecognizer, stream: Any) -> str:
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    return str(recognizer.get_result(stream) or "").strip()


def feed_asr(
    recognizer: sherpa_onnx.OnlineRecognizer,
    stream: Any,
    samples: np.ndarray,
    last_text: str,
) -> str:
    stream.accept_waveform(SAMPLE_RATE, samples)
    result = decode_ready_asr(recognizer, stream)
    if result and result != last_text:
        log(f"asr partial: {result}")
        return result
    return last_text


def calibrate_asr_noise(audio: sd.InputStream, frames: int) -> tuple[float, list[tuple[np.ndarray, float]]]:
    chunks: list[tuple[np.ndarray, float]] = []
    if not ASR_NOISE_GATE_ENABLED or ASR_NOISE_CALIBRATION_SECONDS <= 0:
        return SPEECH_RMS_THRESHOLD, chunks

    deadline = time.monotonic() + ASR_NOISE_CALIBRATION_SECONDS
    while time.monotonic() < deadline:
        samples = read_mono(audio, frames)
        chunks.append((samples.copy(), audio_rms(samples)))

    if not chunks:
        return SPEECH_RMS_THRESHOLD, chunks

    percentile = min(max(ASR_NOISE_GATE_PERCENTILE, 0.0), 100.0)
    noise_floor = float(np.percentile([rms for _, rms in chunks], percentile))
    threshold = max(SPEECH_RMS_THRESHOLD, noise_floor * ASR_NOISE_GATE_RATIO + ASR_NOISE_GATE_OFFSET)
    log(f"asr noise gate: floor={noise_floor:.4f} threshold={threshold:.4f}")
    return threshold, chunks


def listen_command(
    audio: sd.InputStream,
    ready_beep_path: Path | None,
    recognizer: sherpa_onnx.OnlineRecognizer,
    play_ready_beep: bool = True,
) -> str:
    stream = recognizer.create_stream()
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    last_text = ""
    speech_started = False
    max_rms = 0.0
    preroll_chunks = max(1, int(max(ASR_PREROLL_SECONDS, CHUNK_SECONDS) / CHUNK_SECONDS))
    pre_roll: deque[tuple[np.ndarray, float]] = deque(maxlen=preroll_chunks)

    if play_ready_beep and ready_beep_path is not None:
        drain_audio(audio, PRE_BEEP_DRAIN_SECONDS)
        play_wav(ready_beep_path)
        drain_audio(audio, POST_BEEP_DRAIN_SECONDS)
    log("listening for command")
    started = time.monotonic()
    gate_threshold, calibration_chunks = calibrate_asr_noise(audio, chunk)
    for samples, rms in calibration_chunks:
        max_rms = max(max_rms, rms)
        pre_roll.append((samples, rms))
    if ASR_NOISE_GATE_ENABLED and pre_roll and max(rms for _, rms in pre_roll) >= gate_threshold:
        speech_started = True
        for samples, _ in pre_roll:
            last_text = feed_asr(recognizer, stream, samples, last_text)
        pre_roll.clear()

    while time.monotonic() - started < COMMAND_TIMEOUT_SECONDS:
        samples = read_mono(audio, chunk)
        rms = audio_rms(samples)
        max_rms = max(max_rms, rms)
        active = rms >= gate_threshold

        if ASR_NOISE_GATE_ENABLED:
            if not speech_started:
                pre_roll.append((samples.copy(), rms))
                if active:
                    speech_started = True
                    for buffered_samples, _ in pre_roll:
                        last_text = feed_asr(recognizer, stream, buffered_samples, last_text)
                    pre_roll.clear()
            else:
                gated_samples = samples if active else np.zeros_like(samples)
                last_text = feed_asr(recognizer, stream, gated_samples, last_text)
        else:
            if rms >= SPEECH_RMS_THRESHOLD:
                speech_started = True
            last_text = feed_asr(recognizer, stream, samples, last_text)

        elapsed = time.monotonic() - started
        if not speech_started and not last_text and elapsed >= COMMAND_LEADING_SILENCE_SECONDS:
            break

        if elapsed < max(COMMAND_MIN_SECONDS, COMMAND_INITIAL_GRACE_SECONDS):
            continue

        if recognizer.is_endpoint(stream) and (speech_started or last_text):
            break

    stream.input_finished()
    final_text = (decode_ready_asr(recognizer, stream) or last_text).strip()
    del stream
    gc.collect()
    log(
        "asr finished: "
        f"text='{final_text}' speech_started={speech_started} max_rms={max_rms:.4f} "
        f"elapsed={time.monotonic() - started:.1f}s"
    )
    return final_text


class OnlineModelUnavailable(RuntimeError):
    pass


def refresh_llm_route_cache() -> None:
    try:
        timeout = min(max(LLM_ROUTE_CACHE_INTERVAL_SECONDS * 0.5, 0.2), 1.0)
        with httpx.Client(timeout=timeout) as client:
            response = client.get(GATEWAY_REACHABILITY_URL)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
    except Exception as exc:
        log(f"llm reachability cache unavailable: {exc}")
        data = {"online": False, "provider": "", "model": "", "status": "unavailable"}

    route = "online" if data.get("online") is True else "local"
    with LLM_ROUTE_CACHE_LOCK:
        LLM_ROUTE_CACHE.update(
            {
                "route": route,
                "provider": str(data.get("provider") or ""),
                "model": str(data.get("model") or ""),
                "status": str(data.get("status") or ""),
                "updated_at": time.time(),
            }
        )


def llm_route_cache_loop() -> None:
    interval = max(0.5, LLM_ROUTE_CACHE_INTERVAL_SECONDS)
    while True:
        refresh_llm_route_cache()
        time.sleep(interval)


def start_llm_route_cache() -> None:
    refresh_llm_route_cache()
    threading.Thread(target=llm_route_cache_loop, daemon=True).start()


def choose_llm_route() -> str:
    with LLM_ROUTE_CACHE_LOCK:
        cached = dict(LLM_ROUTE_CACHE)
    log(
        "llm route selected for session: "
        f"{cached['route']} provider={cached['provider']} model={cached['model']} status={cached['status']}"
    )
    return str(cached["route"])


def ask_gateway(text: str, llm_route: str | None = None) -> str:
    payload: dict[str, str] = {"message": text}
    if llm_route:
        payload["llm_route"] = llm_route
    with httpx.Client(timeout=GATEWAY_REQUEST_TIMEOUT_SECONDS) as client:
        response = client.post(GATEWAY_URL, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            if llm_route == "online":
                raise OnlineModelUnavailable(str(exc)) from exc
            raise
        data: dict[str, Any] = response.json()
    return str(data.get("answer", "")).strip()


def is_session_end(command: str) -> bool:
    normalized = re.sub(r"[\s，。！？,.!?]", "", command)
    return any(phrase in normalized for phrase in SESSION_END_PHRASES)


def handle_conversation_turn(
    audio: sd.InputStream,
    command: str,
    voice: PiperVoice,
    tts_config: SynthesisConfig,
    display: DisplayClient,
    llm_route: str | None = None,
) -> bool:
    if is_session_end(command):
        log(f"session end command: {command}")
        speak_pausing_input(audio, SESSION_END_RESPONSE, voice, tts_config, display)
        display.set_state("idle")
        return False

    log(f"recognized command: {command}")
    display.set_state("thinking", command)
    try:
        answer = ask_gateway(command, llm_route)
    except OnlineModelUnavailable as exc:
        log(f"online model unavailable during session: {exc}")
        display.set_state("error", "network unavailable")
        speak_pausing_input(audio, NETWORK_UNAVAILABLE_RESPONSE, voice, tts_config, display)
        display.set_state("idle")
        return False
    except Exception as exc:
        log(f"gateway request failed: {exc}")
        display.set_state("error", "gateway unavailable")
        speak_pausing_input(audio, GATEWAY_UNAVAILABLE_RESPONSE, voice, tts_config, display)
        return True

    log(f"answer: {answer}")
    speak_pausing_input(audio, answer, voice, tts_config, display)
    return True


def handle_wake(
    audio: sd.InputStream,
    beep_path: Path,
    recognizer: sherpa_onnx.OnlineRecognizer,
    voice: PiperVoice,
    tts_config: SynthesisConfig,
    display: DisplayClient,
) -> None:
    log("wake detected")
    llm_route = choose_llm_route()
    display.set_state("listening", "wake")
    speak_pausing_input(audio, WAKE_RESPONSE, voice, tts_config, display)
    drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

    for turn in range(1, MAX_SESSION_TURNS + 1):
        log(f"conversation turn {turn}/{MAX_SESSION_TURNS}")
        try:
            command = listen_command(audio, beep_path, recognizer, play_ready_beep=False)
        except Exception as exc:
            log(f"asr failed in conversation: {exc}")
            display.set_state("error", "asr failed")
            speak_pausing_input(audio, ASR_ERROR_RESPONSE, voice, tts_config, display)
            return

        if not command:
            log("conversation idle timeout")
            if SESSION_IDLE_RESPONSE:
                speak_pausing_input(audio, SESSION_IDLE_RESPONSE, voice, tts_config, display)
            display.set_state("idle")
            return

        if not handle_conversation_turn(audio, command, voice, tts_config, display, llm_route):
            return
        drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

    log("conversation reached max turns")
    speak_pausing_input(audio, SESSION_END_RESPONSE, voice, tts_config, display)
    display.set_state("idle")


def main() -> None:
    start_llm_route_cache()
    input_device = select_input_device(INPUT_DEVICE)
    log(f"input device: {input_device if input_device is not None else 'default'}")
    log(f"output device: {OUTPUT_DEVICE}")
    display = DisplayClient(DISPLAY_SERIAL_PORT, DISPLAY_SERIAL_BAUD)
    log(f"display serial: {DISPLAY_SERIAL_PORT or 'disabled'}")
    display.set_state("idle")
    log(f"input channels: {INPUT_CHANNELS}, selected channel: {INPUT_CHANNEL_INDEX}")
    log(f"loading Piper TTS model: {PIPER_MODEL}")
    voice, tts_config = create_tts()
    log(
        "Piper TTS ready: "
        f"sample_rate={voice.config.sample_rate} speaker={PIPER_SPEAKER} "
        f"length_scale={PIPER_LENGTH_SCALE}"
    )
    log("loading low-power wake-word model only")
    kws = create_kws()
    stream = kws.create_stream()
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    log("preloading ASR model")
    recognizer = create_asr()
    log("ASR model ready")

    beep_path = Path(tempfile.gettempdir()) / "chat2m_wake.wav"
    write_beep(beep_path)

    log(f"wake listener active: {wake_words_display()}")
    with sd.InputStream(
        channels=INPUT_CHANNELS,
        dtype="float32",
        samplerate=SAMPLE_RATE,
        device=input_device,
        blocksize=chunk,
    ) as audio:
        while True:
            samples = read_mono(audio, chunk)
            stream.accept_waveform(SAMPLE_RATE, samples)
            while kws.is_ready(stream):
                kws.decode_stream(stream)
                result = kws.get_result(stream)
                if result:
                    log(f"wake keyword matched: {result}")
                    kws.reset_stream(stream)
                    try:
                        handle_wake(audio, beep_path, recognizer, voice, tts_config, display)
                    except Exception as exc:
                        log(f"wake handling failed: {exc}")
                        display.set_state("error", str(exc))
                    drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)
                    stream = kws.create_stream()
                    log(f"wake listener active: {wake_words_display()}")
                    break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
    except Exception as exc:
        log(f"fatal: {exc}")
        sys.exit(1)
