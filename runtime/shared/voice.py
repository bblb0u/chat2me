from __future__ import annotations

import gc
import io
import os
import re
import threading
import subprocess
import sys
import time
import wave
from collections import deque
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Protocol

from app.common import DisplayClient, env_bool, env_float, env_int, env_value, log
import httpx
import numpy as np
import yaml


def env_float_compat(primary_key: str, fallback_key: str, default: str) -> float:
    value = os.getenv(primary_key) or os.getenv(fallback_key) or default
    try:
        return float(value.strip())
    except ValueError:
        raise RuntimeError(f"{primary_key} must be a number in runtime.env") from None


def env_float_default(key: str, default: str) -> float:
    value = os.getenv(key, default).strip() or default
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(f"{key} must be a number in runtime.env") from None


MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
VOICE_KWS_MODEL = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
VOICE_ASR_ENGINE = env_value("VOICE_ASR_ENGINE")
VOICE_ASR_MODEL = env_value("VOICE_ASR_MODEL")
VOICE_TTS_MODEL = env_value("VOICE_TTS_MODEL")
VOICE_TTS_ENGINE = env_value("VOICE_TTS_ENGINE").strip().lower()
KWS_MODEL_DIR = MODELS_DIR / VOICE_KWS_MODEL
ASR_MODEL_DIR = MODELS_DIR / VOICE_ASR_ENGINE / VOICE_ASR_MODEL
SENSEVOICE_MODEL_DIR = Path(os.getenv("SENSEVOICE_MODEL_DIR", str(ASR_MODEL_DIR)))
SENSEVOICE_VAD_MODEL_DIR = Path(
    os.getenv("SENSEVOICE_VAD_MODEL_DIR", str(MODELS_DIR / VOICE_ASR_ENGINE / "speech_fsmn_vad_zh-cn-16k-common-onnx"))
)
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

CORE_URL = os.getenv("CORE_URL") or os.getenv("GATEWAY_URL", "http://chat2me-core:8080/chat")
CORE_REACHABILITY_URL = (
    os.getenv("CORE_REACHABILITY_URL")
    or os.getenv("GATEWAY_REACHABILITY_URL")
    or CORE_URL.rsplit("/", 1)[0] + "/llm/reachability"
)
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://chat2me-asr:8092/asr/transcribe")
ASR_REACHABILITY_URL = os.getenv("ASR_REACHABILITY_URL", ASR_SERVICE_URL.rsplit("/", 1)[0] + "/reachability")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://chat2me-tts:8093/tts/speak")
TTS_REACHABILITY_URL = os.getenv("TTS_REACHABILITY_URL", TTS_SERVICE_URL.rsplit("/", 1)[0] + "/reachability")
NETWORK_UNAVAILABLE_RESPONSE = env_value("NETWORK_UNAVAILABLE_RESPONSE")
LLM_ROUTE_CACHE_INTERVAL_SECONDS = env_float("LLM_ROUTE_CACHE_INTERVAL_SECONDS")
ASR_ROUTE_CACHE_INTERVAL_SECONDS = env_float("ASR_ROUTE_CACHE_INTERVAL_SECONDS")
TTS_ROUTE_CACHE_INTERVAL_SECONDS = env_float("TTS_ROUTE_CACHE_INTERVAL_SECONDS")
DISPLAY_SERIAL_BAUD = env_int("DISPLAY_SERIAL_BAUD")
INPUT_DEVICE = env_value("AUDIO_INPUT_DEVICE", allow_empty=True)
INPUT_DEVICE_REQUIRED = bool(INPUT_DEVICE and not INPUT_DEVICE.isdigit())
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
VOICE_KWS_PROVIDER = os.getenv("VOICE_KWS_PROVIDER", "auto").strip().lower() or "auto"
ASR_THREADS = env_int("ASR_THREADS")
ASR_MODEL_PRECISION = env_value("ASR_MODEL_PRECISION")
ASR_DECODING_METHOD = env_value("ASR_DECODING_METHOD")
ASR_MAX_ACTIVE_PATHS = env_int("ASR_MAX_ACTIVE_PATHS")
ASR_MODELING_UNIT = env_value("ASR_MODELING_UNIT")
ASR_HOTWORDS_SCORE = env_float("ASR_HOTWORDS_SCORE")
CORE_REQUEST_TIMEOUT_SECONDS = env_float_compat("CORE_REQUEST_TIMEOUT_SECONDS", "GATEWAY_REQUEST_TIMEOUT_SECONDS", "30")
CORE_UNAVAILABLE_RESPONSE = os.getenv("CORE_UNAVAILABLE_RESPONSE") or env_value("GATEWAY_UNAVAILABLE_RESPONSE")
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
ASR_WARMUP_SECONDS = env_float_default("ASR_WARMUP_SECONDS", "0")
ASR_WARMUP_WAV_PATH_RAW = os.getenv("ASR_WARMUP_WAV_PATH", "").strip()
ASR_WARMUP_WAV_PATH = Path(ASR_WARMUP_WAV_PATH_RAW) if ASR_WARMUP_WAV_PATH_RAW else None
ONLINE_ASR_BASE_URL = os.getenv("ONLINE_ASR_BASE_URL", "").strip().rstrip("/")
ONLINE_ASR_TRANSCRIPTIONS_PATH = os.getenv("ONLINE_ASR_TRANSCRIPTIONS_PATH", "/audio/transcriptions").strip() or "/audio/transcriptions"
if not ONLINE_ASR_TRANSCRIPTIONS_PATH.startswith("/"):
    ONLINE_ASR_TRANSCRIPTIONS_PATH = "/" + ONLINE_ASR_TRANSCRIPTIONS_PATH
ONLINE_ASR_API_KEY = (os.getenv("ONLINE_ASR_API_KEY") or os.getenv("LLM_API_KEY", "")).strip()
ONLINE_ASR_LANGUAGE = os.getenv("ONLINE_ASR_LANGUAGE", "zh").strip()
ONLINE_ASR_PROMPT = os.getenv("ONLINE_ASR_PROMPT", "").strip()
ONLINE_ASR_RESPONSE_FORMAT = os.getenv("ONLINE_ASR_RESPONSE_FORMAT", "json").strip() or "json"
ONLINE_ASR_TIMEOUT_SECONDS = env_float("ONLINE_ASR_TIMEOUT_SECONDS")
ONLINE_ASR_MIN_AUDIO_SECONDS = env_float("ONLINE_ASR_MIN_AUDIO_SECONDS")
ONLINE_ASR_SILENCE_SECONDS = env_float("ONLINE_ASR_SILENCE_SECONDS")
ONLINE_ASR_RMS_THRESHOLD = env_float("ONLINE_ASR_RMS_THRESHOLD")
SHERPA_TTS_THREADS = env_int("SHERPA_TTS_THREADS")
SHERPA_TTS_PROVIDER = os.getenv("SHERPA_TTS_PROVIDER", "auto").strip().lower() or "auto"
SHERPA_TTS_SPEED = env_float("SHERPA_TTS_SPEED")
SHERPA_TTS_LENGTH_SCALE = env_float("SHERPA_TTS_LENGTH_SCALE")
SHERPA_TTS_NOISE_SCALE = env_float("SHERPA_TTS_NOISE_SCALE")
SHERPA_TTS_SILENCE_SCALE = env_float("SHERPA_TTS_SILENCE_SCALE")
SHERPA_TTS_RULE_FSTS = tuple(
    item.strip()
    for item in os.getenv("SHERPA_TTS_RULE_FSTS", "").split(",")
    if item.strip()
)
MELOTTS_THREADS = env_int("MELOTTS_THREADS")
MELOTTS_PROVIDER = os.getenv("MELOTTS_PROVIDER", "auto").strip().lower() or "auto"
MELOTTS_SPEAKER = env_int("MELOTTS_SPEAKER")
MELOTTS_SPEED = env_float("MELOTTS_SPEED")
MELOTTS_LENGTH_SCALE = env_float("MELOTTS_LENGTH_SCALE")
MELOTTS_NOISE_SCALE = env_float("MELOTTS_NOISE_SCALE")
MELOTTS_NOISE_W_SCALE = env_float("MELOTTS_NOISE_W_SCALE")
MELOTTS_SILENCE_SCALE = env_float("MELOTTS_SILENCE_SCALE")
MELOTTS_RULE_FSTS = tuple(
    item.strip()
    for item in os.getenv("MELOTTS_RULE_FSTS", "").split(",")
    if item.strip()
)
TTS_PLAYER_TIMEOUT_SECONDS = env_float("TTS_PLAYER_TIMEOUT_SECONDS")
SPEECH_TTS_MAX_CHARS = env_int("SPEECH_TTS_MAX_CHARS")
TTS_CACHE_ENABLED = env_bool("TTS_CACHE_ENABLED")
TTS_CACHE_MAX_ITEMS = env_int("TTS_CACHE_MAX_ITEMS")
TTS_CACHE_MAX_BYTES = env_int("TTS_CACHE_MAX_BYTES")
TTS_PLAYBACK_MODE = os.getenv("TTS_PLAYBACK_MODE", "buffered").strip().lower()
TTS_PREBUFFER_SECONDS = float(os.getenv("TTS_PREBUFFER_SECONDS", "2.4").strip() or "2.4")
TTS_WARMUP_TEXTS = tuple(
    text.strip()
    for text in os.getenv("TTS_WARMUP_TEXTS", "").split("|")
    if text.strip()
)
TTS_MODEL_DIR = MODELS_DIR / VOICE_TTS_ENGINE / VOICE_TTS_MODEL
VOICE_ASR_DEVICE = os.getenv("VOICE_ASR_DEVICE", "auto").strip().lower()
VOICE_TTS_DEVICE = os.getenv("VOICE_TTS_DEVICE", "auto").strip().lower()
PIPER_MODEL = TTS_MODEL_DIR / "model.onnx"
PIPER_CONFIG = Path(str(PIPER_MODEL) + ".json")
PIPER_ESPEAK_DATA = Path(os.getenv("PIPER_ESPEAK_DATA", "/opt/piper/espeak-ng-data"))
PIPER_SPEAKER = env_int("PIPER_SPEAKER")
PIPER_LENGTH_SCALE = env_float("PIPER_LENGTH_SCALE")
PIPER_NOISE_SCALE = env_float("PIPER_NOISE_SCALE")
PIPER_NOISE_W_SCALE = env_float("PIPER_NOISE_W_SCALE")
F5_TTS_CKPT_FILE = Path(os.getenv("F5_TTS_CKPT_FILE", str(TTS_MODEL_DIR / "model_1250000.safetensors")))
F5_TTS_VOCODER_DIR = Path(os.getenv("F5_TTS_VOCODER_DIR", str(MODELS_DIR / "f5-tts" / "vocos-mel-24khz")))
F5_TTS_REF_AUDIO_RAW = os.getenv("F5_TTS_REF_AUDIO", "").strip()
F5_TTS_REF_AUDIO = Path(F5_TTS_REF_AUDIO_RAW) if F5_TTS_REF_AUDIO_RAW else None
F5_TTS_REF_TEXT = os.getenv("F5_TTS_REF_TEXT", "对，这就是我，万人敬仰的太乙真人。").strip()
F5_TTS_NFE_STEP = env_int("F5_TTS_NFE_STEP")
F5_TTS_CFG_STRENGTH = env_float("F5_TTS_CFG_STRENGTH")
F5_TTS_SWAY_SAMPLING_COEF = env_float("F5_TTS_SWAY_SAMPLING_COEF")
F5_TTS_SPEED = env_float("F5_TTS_SPEED")
F5_TTS_TARGET_RMS = env_float("F5_TTS_TARGET_RMS")
F5_TTS_FP16 = env_bool("F5_TTS_FP16")
F5_TTS_USE_EMA = env_bool("F5_TTS_USE_EMA")
F5_TTS_ODE_METHOD = os.getenv("F5_TTS_ODE_METHOD", "euler").strip() or "euler"
F5_TTS_SEED_RAW = os.getenv("F5_TTS_SEED", "").strip()
F5_TTS_SEED = int(F5_TTS_SEED_RAW) if F5_TTS_SEED_RAW else None
COSYVOICE_CODE_DIR = Path(os.getenv("COSYVOICE_CODE_DIR", "/opt/CosyVoice"))
COSYVOICE_PACKAGE_PATH = os.getenv(
    "COSYVOICE_PACKAGE_PATH",
    f"{COSYVOICE_CODE_DIR}:{COSYVOICE_CODE_DIR / 'third_party' / 'Matcha-TTS'}",
).strip()
COSYVOICE_WHISPER_ASSETS_DIR = Path(os.getenv("COSYVOICE_WHISPER_ASSETS_DIR", "/opt/chat2me-whisper-assets"))
COSYVOICE_SPK_ID = os.getenv("COSYVOICE_SPK_ID", "中文女").strip() or "中文女"
COSYVOICE_INSTRUCT_TEXT = os.getenv("COSYVOICE_INSTRUCT_TEXT", "用自然、清晰、亲切的语气说话。").strip()
COSYVOICE_SPEED = env_float("COSYVOICE_SPEED")
COSYVOICE_TEXT_FRONTEND = env_bool("COSYVOICE_TEXT_FRONTEND")
COSYVOICE_LOAD_JIT = env_bool("COSYVOICE_LOAD_JIT")
COSYVOICE_LOAD_TRT = env_bool("COSYVOICE_LOAD_TRT")
COSYVOICE_FP16 = env_bool("COSYVOICE_FP16")
ONLINE_TTS_BASE_URL = os.getenv("ONLINE_TTS_BASE_URL", "").strip().rstrip("/")
ONLINE_TTS_SPEECH_PATH = os.getenv("ONLINE_TTS_SPEECH_PATH", "/audio/speech").strip() or "/audio/speech"
if not ONLINE_TTS_SPEECH_PATH.startswith("/"):
    ONLINE_TTS_SPEECH_PATH = "/" + ONLINE_TTS_SPEECH_PATH
ONLINE_TTS_API_KEY = (os.getenv("ONLINE_TTS_API_KEY") or os.getenv("LLM_API_KEY", "")).strip()
ONLINE_TTS_VOICE = os.getenv("ONLINE_TTS_VOICE", "alloy").strip() or "alloy"
ONLINE_TTS_INSTRUCTIONS = os.getenv("ONLINE_TTS_INSTRUCTIONS", "").strip()
ONLINE_TTS_RESPONSE_FORMAT = os.getenv("ONLINE_TTS_RESPONSE_FORMAT", "pcm").strip() or "pcm"
ONLINE_TTS_SPEED = env_float("ONLINE_TTS_SPEED")
ONLINE_TTS_SAMPLE_RATE = env_int("ONLINE_TTS_SAMPLE_RATE")
ONLINE_TTS_TIMEOUT_SECONDS = env_float("ONLINE_TTS_TIMEOUT_SECONDS")
DISPLAY_TEXT_MAX_CHARS = env_int("DISPLAY_TEXT_MAX_CHARS")
DISPLAY_SERIAL_RETRY_SECONDS = env_float("DISPLAY_SERIAL_RETRY_SECONDS")
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
    "online": False,
    "route": "local",
    "provider": "",
    "model": "",
    "status": "not_checked",
    "updated_at": 0.0,
}
LLM_ROUTE_CACHE_LOCK = threading.Lock()
ASR_ROUTE_CACHE = {
    "online": False,
    "provider": "",
    "model": "",
    "status": "not_checked",
    "updated_at": 0.0,
}
ASR_ROUTE_CACHE_LOCK = threading.Lock()
TTS_ROUTE_CACHE = {
    "online": False,
    "provider": "",
    "model": "",
    "status": "not_checked",
    "updated_at": 0.0,
}
TTS_ROUTE_CACHE_LOCK = threading.Lock()


class StreamingRecognizer(Protocol):
    def create_stream(self) -> Any:
        ...

    def accept_waveform(self, stream: Any, sample_rate: int, samples: np.ndarray) -> None:
        ...

    def input_finished(self, stream: Any) -> None:
        ...

    def decode_ready(self, stream: Any) -> str:
        ...

    def is_endpoint(self, stream: Any) -> bool:
        ...


class TextToSpeech(Protocol):
    config: Any

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        ...


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")


def sherpa_provider_candidates(setting_name: str, requested: str) -> tuple[str, tuple[str, ...]]:
    value = requested.strip().lower() or "auto"
    if value == "auto":
        return value, ("cuda", "cpu")
    if value in {"cuda", "gpu"}:
        return value, ("cuda",)
    if value == "cpu":
        return value, ("cpu",)
    raise RuntimeError(f"{setting_name} must be auto, cpu, cuda, or gpu")


def sherpa_gpu_build_unavailable() -> bool:
    import sherpa_onnx

    root = Path(sherpa_onnx.__file__).resolve().parent
    needle = b"Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON"
    for lib in (root / "lib").glob("*.so"):
        try:
            if needle in lib.read_bytes():
                return True
        except OSError:
            continue
    return False


def create_with_sherpa_provider(
    label: str,
    setting_name: str,
    requested: str,
    factory: Any,
) -> tuple[Any, str]:
    requested_value, candidates = sherpa_provider_candidates(setting_name, requested)
    cuda_error: Exception | None = None
    for provider in candidates:
        try:
            instance = factory(provider)
        except Exception as exc:
            if requested_value == "auto" and provider == "cuda":
                cuda_error = exc
                continue
            raise RuntimeError(f"{label} failed with {setting_name}={provider}: {exc}") from exc
        if cuda_error is not None and provider == "cpu":
            log(f"{label} CUDA provider is unavailable for auto; using CPU: {cuda_error}")
        if provider == "cuda" and sherpa_gpu_build_unavailable():
            log(f"{label} CUDA provider is not enabled in this sherpa-onnx wheel; using CPU fallback")
            return instance, "cpu-fallback"
        return instance, provider
    raise RuntimeError(f"{label} failed to initialize with {setting_name}=auto")


def onnxruntime_providers(setting_name: str, requested: str, available_providers: Iterable[str]) -> tuple[list[Any], bool]:
    value = requested.strip().lower() or "auto"
    available = set(available_providers)
    cuda_available = "CUDAExecutionProvider" in available
    cuda_provider: Any = "CUDAExecutionProvider"

    if value.startswith("cuda:"):
        device_index = value.split(":", 1)[1]
        if not device_index.isdigit():
            raise RuntimeError(f"{setting_name} must be auto, cpu, cuda, gpu, or cuda:<index>")
        cuda_provider = ("CUDAExecutionProvider", {"device_id": int(device_index)})
        if not cuda_available:
            raise RuntimeError(
                f"{setting_name}={requested} was requested, but onnxruntime CUDAExecutionProvider is not available"
            )
        return [cuda_provider, "CPUExecutionProvider"], True

    if value in {"cuda", "gpu"}:
        if not cuda_available:
            raise RuntimeError(
                f"{setting_name}={requested} was requested, but onnxruntime CUDAExecutionProvider is not available"
            )
        return [cuda_provider, "CPUExecutionProvider"], True

    if value == "auto":
        if cuda_available:
            return [cuda_provider, "CPUExecutionProvider"], True
        return ["CPUExecutionProvider"], False

    if value == "cpu":
        return ["CPUExecutionProvider"], False

    raise RuntimeError(f"{setting_name} must be auto, cpu, cuda, gpu, or cuda:<index>")


def wake_words_display() -> str:
    return " / ".join(WAKE_WORDS)


def select_input_device(selector: str) -> int | str | None:
    if not selector:
        return None
    if selector.isdigit():
        return int(selector)

    import sounddevice as sd

    devices = sd.query_devices()
    selector_lower = selector.lower()
    for index, device in enumerate(devices):
        if selector_lower not in str(device.get("name", "")).lower():
            continue
        if device.get("max_input_channels", 0) > 0:
            return index

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


def create_kws() -> Any:
    import sherpa_onnx

    keywords_file = ensure_keywords_file()
    require_file(KWS_MODEL_DIR / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
    require_file(KWS_MODEL_DIR / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx")
    require_file(KWS_MODEL_DIR / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx")

    def build(provider: str) -> Any:
        return sherpa_onnx.KeywordSpotter(
            tokens=str(KWS_MODEL_DIR / "tokens.txt"),
            encoder=str(KWS_MODEL_DIR / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            decoder=str(KWS_MODEL_DIR / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
            joiner=str(KWS_MODEL_DIR / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            num_threads=KWS_THREADS,
            keywords_file=str(keywords_file),
            keywords_score=KWS_KEYWORDS_SCORE,
            keywords_threshold=KWS_KEYWORDS_THRESHOLD,
            provider=provider,
        )

    spotter, provider = create_with_sherpa_provider(
        "KWS",
        "VOICE_KWS_PROVIDER",
        VOICE_KWS_PROVIDER,
        build,
    )
    log(
        "KWS config: "
        f"model={KWS_MODEL_DIR} provider={provider} requested={VOICE_KWS_PROVIDER} threads={KWS_THREADS}"
    )
    return spotter


def asr_model_file(stem: str) -> Path:
    precision = ASR_MODEL_PRECISION.strip().lower()
    if precision in {"fp32", "float32", "full"}:
        return ASR_MODEL_DIR / f"{stem}.onnx"
    if precision in {"int8", "quantized"}:
        return ASR_MODEL_DIR / f"{stem}.int8.onnx"
    raise RuntimeError("ASR_MODEL_PRECISION must be fp32 or int8 in runtime.env")


def ensure_hotwords_file() -> str:
    import sherpa_onnx

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


class SherpaStreamingRecognizer:
    def __init__(self, recognizer: Any) -> None:
        self.recognizer = recognizer

    def create_stream(self) -> Any:
        return self.recognizer.create_stream()

    def accept_waveform(self, stream: Any, sample_rate: int, samples: np.ndarray) -> None:
        stream.accept_waveform(sample_rate, samples)

    def input_finished(self, stream: Any) -> None:
        stream.input_finished()

    def decode_ready(self, stream: Any) -> str:
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        return str(self.recognizer.get_result(stream) or "").strip()

    def is_endpoint(self, stream: Any) -> bool:
        return bool(self.recognizer.is_endpoint(stream))


class SenseVoiceStreamingRecognizer:
    def __init__(self, model: Any) -> None:
        self.model = model

    def create_stream(self) -> dict[str, Any]:
        StreamingASREventType = load_sensevoice_streaming_module().StreamingASREventType

        stream: dict[str, Any] = {
            "committed": "",
            "partial": "",
            "endpoint": False,
        }

        def on_event(event_type: Any, text: str) -> None:
            if event_type == StreamingASREventType.FINAL_RESULT and text:
                stream["committed"] = str(stream["committed"]) + text
            elif event_type == StreamingASREventType.PARTIAL_RESULT:
                stream["partial"] = text
            elif event_type == StreamingASREventType.SPEECH_END:
                stream["endpoint"] = True

        self.model.set_on_event_callback(on_event)
        return stream

    def accept_waveform(self, stream: dict[str, Any], sample_rate: int, samples: np.ndarray) -> None:
        if sample_rate != 16000:
            raise RuntimeError("SenseVoice input must be 16 kHz; set AUDIO_SAMPLE_RATE=16000")
        if samples.size:
            self.model.accept_audio(np.asarray(samples, dtype=np.float32))

    def input_finished(self, stream: dict[str, Any]) -> None:
        self.model.finalize_utterance()

    def decode_ready(self, stream: dict[str, Any]) -> str:
        return (str(stream.get("committed") or "") + str(stream.get("partial") or "")).strip()

    def is_endpoint(self, stream: dict[str, Any]) -> bool:
        return bool(stream.get("endpoint", False))


class OnlineBatchRecognizer:
    def __init__(self) -> None:
        self.sample_rate = SAMPLE_RATE

    def create_stream(self) -> dict[str, Any]:
        return {
            "chunks": [],
            "seconds": 0.0,
            "active_seconds": 0.0,
            "speech_started": False,
            "endpoint": False,
            "finalized": False,
            "text": "",
        }

    def accept_waveform(self, stream: dict[str, Any], sample_rate: int, samples: np.ndarray) -> None:
        if sample_rate != self.sample_rate:
            raise RuntimeError(f"online ASR input must be {self.sample_rate} Hz")
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not audio.size:
            return
        stream["chunks"].append(audio.copy())
        seconds = len(audio) / max(1, sample_rate)
        stream["seconds"] = float(stream["seconds"]) + seconds
        rms = audio_rms(audio)
        if rms >= ONLINE_ASR_RMS_THRESHOLD:
            stream["speech_started"] = True
            stream["active_seconds"] = float(stream["seconds"])
        elif stream["speech_started"]:
            trailing = float(stream["seconds"]) - float(stream["active_seconds"])
            if float(stream["seconds"]) >= ONLINE_ASR_MIN_AUDIO_SECONDS and trailing >= ONLINE_ASR_SILENCE_SECONDS:
                stream["endpoint"] = True

    def input_finished(self, stream: dict[str, Any]) -> None:
        if stream.get("finalized"):
            return
        stream["finalized"] = True
        chunks = stream.get("chunks") or []
        if not chunks:
            stream["text"] = ""
            return
        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        stream["text"] = transcribe_online_audio(audio, self.sample_rate)

    def decode_ready(self, stream: dict[str, Any]) -> str:
        return str(stream.get("text") or "").strip()

    def is_endpoint(self, stream: dict[str, Any]) -> bool:
        return bool(stream.get("endpoint", False))


class RemoteBatchRecognizer:
    def __init__(self) -> None:
        self.sample_rate = SAMPLE_RATE

    def create_stream(self) -> dict[str, Any]:
        return {
            "chunks": [],
            "seconds": 0.0,
            "active_seconds": 0.0,
            "speech_started": False,
            "endpoint": False,
            "finalized": False,
            "text": "",
        }

    def accept_waveform(self, stream: dict[str, Any], sample_rate: int, samples: np.ndarray) -> None:
        if sample_rate != self.sample_rate:
            raise RuntimeError(f"remote ASR input must be {self.sample_rate} Hz")
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not audio.size:
            return
        stream["chunks"].append(audio.copy())
        seconds = len(audio) / max(1, sample_rate)
        stream["seconds"] = float(stream["seconds"]) + seconds
        rms = audio_rms(audio)
        if rms >= ONLINE_ASR_RMS_THRESHOLD:
            stream["speech_started"] = True
            stream["active_seconds"] = float(stream["seconds"])
        elif stream["speech_started"]:
            trailing = float(stream["seconds"]) - float(stream["active_seconds"])
            if float(stream["seconds"]) >= ONLINE_ASR_MIN_AUDIO_SECONDS and trailing >= ONLINE_ASR_SILENCE_SECONDS:
                stream["endpoint"] = True

    def input_finished(self, stream: dict[str, Any]) -> None:
        if stream.get("finalized"):
            return
        stream["finalized"] = True
        chunks = stream.get("chunks") or []
        if not chunks:
            stream["text"] = ""
            return
        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        stream["text"] = transcribe_remote_audio(audio, self.sample_rate)

    def decode_ready(self, stream: dict[str, Any]) -> str:
        return str(stream.get("text") or "").strip()

    def is_endpoint(self, stream: dict[str, Any]) -> bool:
        return bool(stream.get("endpoint", False))


def online_audio_headers(api_key: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def online_asr_url() -> str:
    if not ONLINE_ASR_BASE_URL:
        raise RuntimeError("ONLINE_ASR_BASE_URL must be set when VOICE_ASR_ENGINE=online")
    return f"{ONLINE_ASR_BASE_URL}{ONLINE_ASR_TRANSCRIPTIONS_PATH}"


def online_tts_url() -> str:
    if not ONLINE_TTS_BASE_URL:
        raise RuntimeError("ONLINE_TTS_BASE_URL must be set when VOICE_TTS_ENGINE=online")
    return f"{ONLINE_TTS_BASE_URL}{ONLINE_TTS_SPEECH_PATH}"


def float_audio_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def transcribe_online_audio(samples: np.ndarray, sample_rate: int) -> str:
    wav_bytes = float_audio_to_wav_bytes(samples, sample_rate)
    data = {
        "model": VOICE_ASR_MODEL,
        "response_format": ONLINE_ASR_RESPONSE_FORMAT,
    }
    if ONLINE_ASR_LANGUAGE:
        data["language"] = ONLINE_ASR_LANGUAGE
    if ONLINE_ASR_PROMPT:
        data["prompt"] = ONLINE_ASR_PROMPT
    files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
    timeout = httpx.Timeout(connect=5.0, read=ONLINE_ASR_TIMEOUT_SECONDS, write=15.0, pool=5.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(online_asr_url(), headers=online_audio_headers(ONLINE_ASR_API_KEY), data=data, files=files)
        response.raise_for_status()
    if ONLINE_ASR_RESPONSE_FORMAT == "json":
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("text") or "").strip()
    return response.text.strip()


def cached_online_available(cache: dict[str, Any], lock: threading.Lock) -> bool:
    with lock:
        return bool(cache.get("online", False))


def transcribe_remote_audio(samples: np.ndarray, sample_rate: int) -> str:
    wav_bytes = float_audio_to_wav_bytes(samples, sample_rate)
    data = {"online_available": "1" if cached_online_available(ASR_ROUTE_CACHE, ASR_ROUTE_CACHE_LOCK) else "0"}
    files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
    timeout = httpx.Timeout(connect=5.0, read=ONLINE_ASR_TIMEOUT_SECONDS + 10, write=15.0, pool=5.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(ASR_SERVICE_URL, data=data, files=files)
        response.raise_for_status()
    payload: dict[str, Any] = response.json()
    route = payload.get("route")
    engine = payload.get("engine")
    model = payload.get("model")
    fallback = payload.get("fallback")
    log(f"remote asr result: route={route} engine={engine} model={model} fallback={fallback}")
    return str(payload.get("text") or "").strip()


def create_sherpa_asr() -> StreamingRecognizer:
    import sherpa_onnx

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
        f"max_active_paths={ASR_MAX_ACTIVE_PATHS} hotwords={HOTWORDS_PATH} "
        f"device={VOICE_ASR_DEVICE}"
    )

    def build(provider: str) -> Any:
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
            provider=provider,
        )

    recognizer, provider = create_with_sherpa_provider(
        "Sherpa ASR",
        "VOICE_ASR_DEVICE",
        VOICE_ASR_DEVICE,
        build,
    )
    log(f"Sherpa ASR loaded: provider={provider} requested={VOICE_ASR_DEVICE}")
    return SherpaStreamingRecognizer(recognizer)


def create_sensevoice_asr() -> StreamingRecognizer:
    import json

    import kaldi_native_fbank as knf
    import onnxruntime
    from sense_voice_streaming_asr.cmvn_utils import load_cmvn

    sensevoice_module = load_sensevoice_streaming_module()
    SenseVoiceStreamingASR = sensevoice_module.SenseVoiceStreamingASR
    StreamingASRConfig = sensevoice_module.StreamingASRConfig

    require_file(SENSEVOICE_MODEL_DIR / "model_quant.onnx")
    require_file(SENSEVOICE_MODEL_DIR / "am.mvn")
    require_file(SENSEVOICE_MODEL_DIR / "tokens.json")
    require_file(SENSEVOICE_VAD_MODEL_DIR / "model_quant.onnx")
    vad_cmvn_path = SENSEVOICE_VAD_MODEL_DIR / "vad.mvn"
    if not vad_cmvn_path.is_file():
        vad_cmvn_path = SENSEVOICE_VAD_MODEL_DIR / "am.mvn"
    require_file(vad_cmvn_path)

    available_providers = onnxruntime.get_available_providers()
    providers, use_cuda = onnxruntime_providers("VOICE_ASR_DEVICE", VOICE_ASR_DEVICE, available_providers)
    config = StreamingASRConfig(
        lang=os.getenv("SENSEVOICE_LANGUAGE", "auto"),
        itn_min_speech_time_ms=env_int("SENSEVOICE_ITN_MIN_SPEECH_MS"),
        vad_start_threshold=env_float("SENSEVOICE_VAD_START_THRESHOLD"),
        vad_end_threshold=env_float("SENSEVOICE_VAD_END_THRESHOLD"),
        vad_start_persistence_ms=env_int("SENSEVOICE_VAD_START_PERSISTENCE_MS"),
        vad_end_persistence_ms=env_int("SENSEVOICE_VAD_END_PERSISTENCE_MS"),
        vad_start_padding_ms=env_int("SENSEVOICE_VAD_START_PADDING_MS"),
        asr_result_trigger_buffer_ms=env_int("SENSEVOICE_ASR_TRIGGER_BUFFER_MS"),
        asr_result_update_interval_ms=env_int("SENSEVOICE_ASR_UPDATE_INTERVAL_MS"),
    )
    log(
        "SenseVoice streaming ASR config: "
        f"model={SENSEVOICE_MODEL_DIR} vad={SENSEVOICE_VAD_MODEL_DIR} "
        f"device={VOICE_ASR_DEVICE} cuda={use_cuda} providers={','.join(available_providers)} "
        f"lang={config.lang} "
        f"vad_start={config.vad_start_threshold} vad_end={config.vad_end_threshold}"
    )

    def make_fbank_options() -> Any:
        fbank_opts = knf.FbankOptions()
        fbank_opts.frame_opts.samp_freq = 16000
        fbank_opts.frame_opts.dither = 0.0
        fbank_opts.frame_opts.window_type = "hamming"
        fbank_opts.frame_opts.frame_shift_ms = 10
        fbank_opts.frame_opts.frame_length_ms = 25
        fbank_opts.mel_opts.num_bins = 80
        fbank_opts.energy_floor = 0
        fbank_opts.frame_opts.snip_edges = True
        fbank_opts.mel_opts.debug_mel = False
        return fbank_opts

    asr_model = SimpleNamespace()
    asr_model.cmvn = load_cmvn(str(SENSEVOICE_MODEL_DIR / "am.mvn"))
    asr_model.sensevoice_tokens = json.loads((SENSEVOICE_MODEL_DIR / "tokens.json").read_text(encoding="utf-8"))

    def ctc_tokens_to_text(tokens: Iterable[Any], prev_token_id: Any = None, filter_special_token: bool = True) -> str:
        merged_tokens: list[int] = []
        previous = int(prev_token_id) if prev_token_id is not None else None
        for token in tokens:
            token_id = int(token)
            if token_id == previous:
                continue
            previous = token_id
            if token_id != 0:
                merged_tokens.append(token_id)
        pieces = [asr_model.sensevoice_tokens[token_id] for token_id in merged_tokens]
        if filter_special_token:
            pieces = [piece for piece in pieces if not str(piece).startswith("<|")]
        return "".join(str(piece) for piece in pieces).replace("▁", " ")

    asr_model.ctc_tokens_to_text = ctc_tokens_to_text
    asr_model.model_inference_session = onnxruntime.InferenceSession(
        str(SENSEVOICE_MODEL_DIR / "model_quant.onnx"),
        providers=providers,
    )
    asr_model.fbank_opts = make_fbank_options()
    asr_model.lfr_m = 7
    asr_model.lfr_n = 6
    asr_model.textnorm_dict = {"withitn": 14, "woitn": 15}
    asr_model.lid_dict = {
        "auto": 0,
        "zh": 3,
        "en": 4,
        "yue": 7,
        "ja": 11,
        "ko": 12,
        "nospeech": 13,
    }

    vad_model = SimpleNamespace()
    vad_model.cmvn = load_cmvn(str(vad_cmvn_path))
    vad_model.model_inference_session = onnxruntime.InferenceSession(
        str(SENSEVOICE_VAD_MODEL_DIR / "model_quant.onnx"),
        providers=providers,
    )
    vad_model.fbank_opts = make_fbank_options()
    vad_model.lfr_m = 5
    vad_model.lfr_n = 1
    vad_model.vad_cache = [np.zeros((1, 128, 19, 1), dtype=np.float32) for _ in range(4)]

    model = SenseVoiceStreamingASR(
        asr_model=asr_model,
        vad_model=vad_model,
        config=config,
    )
    return SenseVoiceStreamingRecognizer(model)


def install_sensevoice_model_data_stub() -> None:
    import types

    module_name = "sense_voice_streaming_asr.model_data"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class SenseVoiceModel:
        pass

    class VadModel:
        pass

    module.SenseVoiceModel = SenseVoiceModel
    module.VadModel = VadModel
    sys.modules[module_name] = module


def load_sensevoice_streaming_module() -> Any:
    module_name = "sense_voice_streaming_asr.sense_voice_streaming_asr"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    import sense_voice_streaming_asr

    install_sensevoice_model_data_stub()
    package_dir = Path(sense_voice_streaming_asr.__file__).parent
    source_path = package_dir / "sense_voice_streaming_asr.py"
    source = source_path.read_text(encoding="utf-8")
    source_lines = [
        line
        for line in source.splitlines()
        if not line.strip().startswith("from __future__ import ")
    ]
    patched_source = "from __future__ import annotations\nimport logging\n" + "\n".join(source_lines) + "\n"
    module = type(sys)(module_name)
    module.__file__ = str(source_path)
    module.__package__ = "sense_voice_streaming_asr"
    sys.modules[module_name] = module
    exec(compile(patched_source, str(source_path), "exec"), module.__dict__)
    return module


def create_asr() -> StreamingRecognizer:
    if VOICE_ASR_ENGINE == "sherpa":
        return create_sherpa_asr()
    if VOICE_ASR_ENGINE == "sensevoice":
        return create_sensevoice_asr()
    if VOICE_ASR_ENGINE == "online":
        log(
            "Online ASR config: "
            f"url={online_asr_url()} model={VOICE_ASR_MODEL} language={ONLINE_ASR_LANGUAGE} "
            f"format={ONLINE_ASR_RESPONSE_FORMAT}"
        )
        return OnlineBatchRecognizer()
    raise RuntimeError(f"VOICE_ASR_ENGINE '{VOICE_ASR_ENGINE}' is not supported")


def create_remote_asr() -> StreamingRecognizer:
    log(f"remote ASR service: {ASR_SERVICE_URL}")
    return RemoteBatchRecognizer()


def read_wav_file(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"ASR warmup WAV must be 16-bit PCM: {path}")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32, copy=False), int(sample_rate)


def warmup_audio_asr(recognizer: StreamingRecognizer, samples: np.ndarray, sample_rate: int) -> tuple[str, float]:
    stream = recognizer.create_stream()
    started = time.monotonic()
    try:
        chunk = max(1, int(sample_rate * CHUNK_SECONDS))
        for offset in range(0, len(samples), chunk):
            recognizer.accept_waveform(stream, sample_rate, samples[offset : offset + chunk])
        recognizer.input_finished(stream)
        text = recognizer.decode_ready(stream)
    finally:
        del stream
    return text, time.monotonic() - started


def warmup_streaming_asr(recognizer: StreamingRecognizer, seconds: float) -> tuple[int, float]:
    samples = np.zeros(max(1, int(SAMPLE_RATE * seconds)), dtype=np.float32)
    _, elapsed = warmup_audio_asr(recognizer, samples, SAMPLE_RATE)
    return len(samples), elapsed


def warmup_asr(recognizer: StreamingRecognizer) -> None:
    seconds = ASR_WARMUP_SECONDS
    if seconds <= 0:
        return

    started = time.monotonic()
    try:
        if ASR_WARMUP_WAV_PATH is not None:
            if not ASR_WARMUP_WAV_PATH.is_file():
                log(f"asr warmup skipped: missing wav={ASR_WARMUP_WAV_PATH}")
                return
            samples, sample_rate = read_wav_file(ASR_WARMUP_WAV_PATH)
            text, elapsed = warmup_audio_asr(recognizer, samples, sample_rate)
            log(
                "asr warmup: "
                f"engine={VOICE_ASR_ENGINE} wav={ASR_WARMUP_WAV_PATH} "
                f"audio_seconds={len(samples) / max(1, sample_rate):.2f} "
                f"text_chars={len(text)} elapsed={elapsed:.2f}s"
            )
            return

        if isinstance(recognizer, (OnlineBatchRecognizer, RemoteBatchRecognizer)):
            log(f"asr warmup skipped: engine={VOICE_ASR_ENGINE}")
            return

        if isinstance(recognizer, SenseVoiceStreamingRecognizer):
            log("asr warmup skipped: SenseVoice requires ASR_WARMUP_WAV_PATH")
            return

        samples, elapsed = warmup_streaming_asr(recognizer, seconds)
        log(
            "asr warmup: "
            f"engine={VOICE_ASR_ENGINE} seconds={seconds:.2f} samples={samples} "
            f"elapsed={elapsed:.2f}s"
        )
    except Exception as exc:
        log(
            "asr warmup failed: "
            f"engine={VOICE_ASR_ENGINE} model={VOICE_ASR_MODEL} "
            f"elapsed={time.monotonic() - started:.2f}s error={exc}"
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


class SherpaTTS:
    def __init__(self, tts: Any) -> None:
        self.tts = tts
        sample_rate = int(getattr(tts, "sample_rate", 16000) or 16000)
        self.config = SimpleNamespace(sample_rate=sample_rate)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        audio = self.tts.generate(text, sid=0, speed=SHERPA_TTS_SPEED)
        yield tensor_audio_bytes(audio.samples)


class F5TextToSpeech:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.config = SimpleNamespace(sample_rate=runtime.sample_rate)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        yield tensor_audio_bytes(self.runtime.generate(text))


class PiperTTS:
    def __init__(self, sample_rate: int) -> None:
        self.config = SimpleNamespace(sample_rate=sample_rate)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        command = [
            "piper",
            "--model",
            str(PIPER_MODEL),
            "--config",
            str(PIPER_CONFIG),
            "--output_raw",
            "--speaker",
            str(PIPER_SPEAKER),
            "--length_scale",
            str(PIPER_LENGTH_SCALE),
            "--noise_scale",
            str(PIPER_NOISE_SCALE),
            "--noise_w",
            str(PIPER_NOISE_W_SCALE),
            "--espeak_data",
            str(PIPER_ESPEAK_DATA),
            "--quiet",
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(text.encode("utf-8"))
            process.stdin.close()
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                yield chunk
            return_code = process.wait(timeout=TTS_PLAYER_TIMEOUT_SECONDS)
        if return_code != 0:
            raise RuntimeError(f"piper exited with status {return_code}")


class OnlineTextToSpeech:
    def __init__(self) -> None:
        self.config = SimpleNamespace(sample_rate=ONLINE_TTS_SAMPLE_RATE)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        audio = synthesize_online_audio(text)
        yield decode_online_tts_audio(audio, ONLINE_TTS_RESPONSE_FORMAT, ONLINE_TTS_SAMPLE_RATE)


class RemoteTextToSpeech:
    def __init__(self) -> None:
        self.config = SimpleNamespace(sample_rate=SAMPLE_RATE)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        wav_bytes = synthesize_remote_wav(text)
        pcm, sample_rate = wav_bytes_to_pcm(wav_bytes)
        self.config.sample_rate = sample_rate
        yield pcm


class CosyVoiceTTS:
    def __init__(self, model: Any) -> None:
        self.model = model
        sample_rate = int(getattr(model, "sample_rate", 22050) or 22050)
        self.config = SimpleNamespace(sample_rate=sample_rate)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        kwargs = {
            "stream": False,
            "speed": COSYVOICE_SPEED,
            "text_frontend": COSYVOICE_TEXT_FRONTEND,
        }
        if VOICE_TTS_MODEL.endswith("-Instruct"):
            chunks = self.model.inference_instruct(text, COSYVOICE_SPK_ID, COSYVOICE_INSTRUCT_TEXT, **kwargs)
        elif VOICE_TTS_MODEL.endswith("-SFT"):
            chunks = self.model.inference_sft(text, COSYVOICE_SPK_ID, **kwargs)
        else:
            raise RuntimeError("use CosyVoice-300M-SFT or CosyVoice-300M-Instruct")
        for chunk in chunks:
            speech = chunk.get("tts_speech") if isinstance(chunk, dict) else chunk
            yield tensor_audio_bytes(speech)


class CachedTextToSpeech:
    def __init__(self, voice: TextToSpeech, max_items: int, max_bytes: int) -> None:
        self.voice = voice
        self.config = voice.config
        self.max_items = max(0, max_items)
        self.max_bytes = max(0, max_bytes)
        self.cache: dict[str, tuple[bytes, ...]] = {}
        self.order: deque[str] = deque()

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        cached = self.cache.get(text)
        if cached is not None:
            for chunk in cached:
                yield chunk
            return

        chunks: list[bytes] = []
        total_bytes = 0
        cacheable = bool(text) and self.max_items > 0 and self.max_bytes > 0
        for chunk in self.voice.synthesize_pcm(text):
            data = bytes(chunk)
            if cacheable:
                total_bytes += len(data)
                if total_bytes <= self.max_bytes:
                    chunks.append(data)
                else:
                    chunks.clear()
                    cacheable = False
            yield data

        if cacheable and chunks:
            self.cache[text] = tuple(chunks)
            self.order.append(text)
            while len(self.order) > self.max_items:
                old_text = self.order.popleft()
                self.cache.pop(old_text, None)

    def preload(self, text: str) -> None:
        if text and text not in self.cache:
            for _ in self.synthesize_pcm(text):
                pass


def create_sherpa_tts() -> TextToSpeech:
    import sherpa_onnx

    if VOICE_TTS_MODEL != "matcha-icefall-zh-en":
        raise RuntimeError("use sherpa TTS model matcha-icefall-zh-en")

    acoustic_model = TTS_MODEL_DIR / "model-steps-3.onnx"
    vocoder = TTS_MODEL_DIR / "vocos-16khz-univ.onnx"
    tokens = TTS_MODEL_DIR / "tokens.txt"
    lexicon = TTS_MODEL_DIR / "lexicon.txt"
    data_dir = TTS_MODEL_DIR / "espeak-ng-data"
    require_file(acoustic_model)
    require_file(vocoder)
    require_file(tokens)
    require_file(lexicon)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing required directory: {data_dir}")
    rule_fsts = []
    for name in SHERPA_TTS_RULE_FSTS:
        path = TTS_MODEL_DIR / name
        require_file(path)
        rule_fsts.append(str(path))

    log(
        "Sherpa TTS config: "
        f"model={TTS_MODEL_DIR} provider={SHERPA_TTS_PROVIDER} threads={SHERPA_TTS_THREADS} "
        f"speed={SHERPA_TTS_SPEED} length_scale={SHERPA_TTS_LENGTH_SCALE} "
        f"noise_scale={SHERPA_TTS_NOISE_SCALE}"
    )
    started = time.monotonic()
    matcha = sherpa_onnx.OfflineTtsMatchaModelConfig(
        acoustic_model=str(acoustic_model),
        vocoder=str(vocoder),
        tokens=str(tokens),
        lexicon=str(lexicon),
        data_dir=str(data_dir),
        length_scale=SHERPA_TTS_LENGTH_SCALE,
        noise_scale=SHERPA_TTS_NOISE_SCALE,
    )

    def build(provider: str) -> Any:
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            matcha=matcha,
            num_threads=SHERPA_TTS_THREADS,
            provider=provider,
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts=",".join(rule_fsts),
            max_num_sentences=1,
            silence_scale=SHERPA_TTS_SILENCE_SCALE,
        )
        return sherpa_onnx.OfflineTts(config)

    tts, provider = create_with_sherpa_provider(
        "Sherpa TTS",
        "SHERPA_TTS_PROVIDER",
        SHERPA_TTS_PROVIDER,
        build,
    )
    log(
        "Sherpa TTS loaded: "
        f"provider={provider} requested={SHERPA_TTS_PROVIDER} "
        f"sample_rate={tts.sample_rate} elapsed={time.monotonic() - started:.2f}s"
    )
    return SherpaTTS(tts)


def create_melotts_tts() -> TextToSpeech:
    import sherpa_onnx

    if VOICE_TTS_MODEL != "vits-melo-tts-zh_en":
        raise RuntimeError("use MeloTTS model vits-melo-tts-zh_en")

    model = TTS_MODEL_DIR / "model.onnx"
    tokens = TTS_MODEL_DIR / "tokens.txt"
    lexicon = TTS_MODEL_DIR / "lexicon.txt"
    dict_dir = TTS_MODEL_DIR / "dict"
    require_file(model)
    require_file(tokens)
    require_file(lexicon)
    if not dict_dir.is_dir():
        raise FileNotFoundError(f"missing required directory: {dict_dir}")
    rule_fsts = []
    for name in MELOTTS_RULE_FSTS:
        path = TTS_MODEL_DIR / name
        require_file(path)
        rule_fsts.append(str(path))

    log(
        "MeloTTS config: "
        f"model={TTS_MODEL_DIR} provider={MELOTTS_PROVIDER} threads={MELOTTS_THREADS} "
        f"speaker={MELOTTS_SPEAKER} speed={MELOTTS_SPEED} length_scale={MELOTTS_LENGTH_SCALE} "
        f"noise_scale={MELOTTS_NOISE_SCALE}"
    )
    started = time.monotonic()
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(model),
        tokens=str(tokens),
        lexicon=str(lexicon),
        data_dir=str(TTS_MODEL_DIR),
        noise_scale=MELOTTS_NOISE_SCALE,
        noise_scale_w=MELOTTS_NOISE_W_SCALE,
        length_scale=MELOTTS_LENGTH_SCALE,
    )

    def build(provider: str) -> Any:
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits,
            num_threads=MELOTTS_THREADS,
            provider=provider,
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts=",".join(rule_fsts),
            max_num_sentences=1,
            silence_scale=MELOTTS_SILENCE_SCALE,
        )
        return sherpa_onnx.OfflineTts(config)

    tts, provider = create_with_sherpa_provider(
        "MeloTTS",
        "MELOTTS_PROVIDER",
        MELOTTS_PROVIDER,
        build,
    )
    log(
        "MeloTTS loaded: "
        f"provider={provider} requested={MELOTTS_PROVIDER} "
        f"sample_rate={tts.sample_rate} elapsed={time.monotonic() - started:.2f}s"
    )
    return SherpaTTSWithSpeaker(tts, MELOTTS_SPEAKER, MELOTTS_SPEED)


def create_cosyvoice_tts() -> TextToSpeech:
    import inspect
    import torch

    from app.engines.cosyvoice import install_cosyvoice_runtime_adapters

    install_cosyvoice_runtime_adapters(
        COSYVOICE_PACKAGE_PATH,
        COSYVOICE_WHISPER_ASSETS_DIR,
        COSYVOICE_TEXT_FRONTEND,
    )
    from cosyvoice.cli.cosyvoice import CosyVoice

    for name in ("cosyvoice.yaml", "flow.pt", "hift.pt", "llm.pt", "campplus.onnx", "speech_tokenizer_v1.onnx"):
        require_file(TTS_MODEL_DIR / name)
    if VOICE_TTS_MODEL.endswith(("-SFT", "-Instruct")):
        require_file(TTS_MODEL_DIR / "spk2info.pt")
    if VOICE_TTS_DEVICE not in {"auto", "cuda", "gpu"} and not VOICE_TTS_DEVICE.startswith("cuda:"):
        raise RuntimeError("CosyVoice requires GPU. Set VOICE_TTS_DEVICE=cuda or auto.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CosyVoice requires torch CUDA, "
            f"but CUDA is not available: device={VOICE_TTS_DEVICE} "
            f"torch={torch.__version__} cuda={torch.version.cuda}"
        )
    if VOICE_TTS_DEVICE.startswith("cuda:"):
        device_index = VOICE_TTS_DEVICE.split(":", 1)[1]
        if not device_index.isdigit():
            raise RuntimeError("VOICE_TTS_DEVICE must be auto, cuda, gpu, or cuda:<index> for CosyVoice")
        if int(device_index) >= torch.cuda.device_count():
            raise RuntimeError(
                f"VOICE_TTS_DEVICE={VOICE_TTS_DEVICE} is not available; "
                f"torch sees {torch.cuda.device_count()} CUDA device(s)"
            )
        torch.cuda.set_device(int(device_index))
    device = "cuda" if VOICE_TTS_DEVICE in {"auto", "gpu"} else VOICE_TTS_DEVICE
    log(
        "CosyVoice TTS config: "
        f"model={TTS_MODEL_DIR} device={device} speaker={COSYVOICE_SPK_ID} "
        f"jit={COSYVOICE_LOAD_JIT} trt={COSYVOICE_LOAD_TRT} fp16={COSYVOICE_FP16}"
    )
    init_kwargs: dict[str, Any] = {}
    signature = inspect.signature(CosyVoice)
    if "load_jit" in signature.parameters:
        init_kwargs["load_jit"] = COSYVOICE_LOAD_JIT
    if "load_trt" in signature.parameters:
        init_kwargs["load_trt"] = COSYVOICE_LOAD_TRT
    if "fp16" in signature.parameters:
        init_kwargs["fp16"] = COSYVOICE_FP16
    if "device" in signature.parameters:
        init_kwargs["device"] = device
    try:
        model = CosyVoice(str(TTS_MODEL_DIR), **init_kwargs)
    except AssertionError as exc:
        if COSYVOICE_LOAD_TRT:
            plan = TTS_MODEL_DIR / f"flow.decoder.estimator.{'fp16' if COSYVOICE_FP16 else 'fp32'}.mygpu.plan"
            raise RuntimeError(
                "CosyVoice TensorRT engine failed to load. "
                f"Plan file: {plan}. "
                "Regenerate the plan inside this image/runtime, or set COSYVOICE_LOAD_TRT=0."
            ) from exc
        raise
    available_spks = list(getattr(getattr(model, "frontend", None), "spk2info", {}).keys())
    if available_spks and COSYVOICE_SPK_ID not in available_spks:
        raise RuntimeError(
            f"CosyVoice speaker '{COSYVOICE_SPK_ID}' is not available. "
            f"Available speakers: {', '.join(available_spks)}"
        )
    return CosyVoiceTTS(model)


class SherpaTTSWithSpeaker(SherpaTTS):
    def __init__(self, tts: Any, speaker: int, speed: float) -> None:
        super().__init__(tts)
        self.speaker = speaker
        self.speed = speed

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        audio = self.tts.generate(text, sid=self.speaker, speed=self.speed)
        yield tensor_audio_bytes(audio.samples)


def create_f5_tts() -> TextToSpeech:
    from app.engines.f5 import F5TTSRuntime

    runtime = F5TTSRuntime(
        model_name=VOICE_TTS_MODEL,
        model_dir=TTS_MODEL_DIR,
        ckpt_file=F5_TTS_CKPT_FILE,
        vocoder_dir=F5_TTS_VOCODER_DIR,
        ref_audio=F5_TTS_REF_AUDIO,
        ref_text=F5_TTS_REF_TEXT,
        device=VOICE_TTS_DEVICE,
        fp16=F5_TTS_FP16,
        use_ema=F5_TTS_USE_EMA,
        ode_method=F5_TTS_ODE_METHOD,
        nfe_step=F5_TTS_NFE_STEP,
        cfg_strength=F5_TTS_CFG_STRENGTH,
        sway_sampling_coef=F5_TTS_SWAY_SAMPLING_COEF,
        speed=F5_TTS_SPEED,
        target_rms=F5_TTS_TARGET_RMS,
        seed=F5_TTS_SEED,
    )
    return F5TextToSpeech(runtime)


def create_tts() -> tuple[TextToSpeech, None]:
    if VOICE_TTS_ENGINE == "piper":
        return wrap_tts(create_piper_tts()), None
    if VOICE_TTS_ENGINE == "melotts":
        return wrap_tts(create_melotts_tts()), None
    if VOICE_TTS_ENGINE == "sherpa":
        return wrap_tts(create_sherpa_tts()), None
    if VOICE_TTS_ENGINE == "f5-tts":
        return wrap_tts(create_f5_tts()), None
    if VOICE_TTS_ENGINE == "cosyvoice":
        return wrap_tts(create_cosyvoice_tts()), None
    if VOICE_TTS_ENGINE == "online":
        return wrap_tts(create_online_tts()), None
    raise RuntimeError(f"VOICE_TTS_ENGINE '{VOICE_TTS_ENGINE}' is not supported")


def create_remote_tts() -> tuple[TextToSpeech, None]:
    log(f"remote TTS service: {TTS_SERVICE_URL}")
    return RemoteTextToSpeech(), None


def wrap_tts(voice: TextToSpeech) -> TextToSpeech:
    if not TTS_CACHE_ENABLED:
        return voice
    return CachedTextToSpeech(voice, TTS_CACHE_MAX_ITEMS, TTS_CACHE_MAX_BYTES)


def preload_tts_cache(voice: TextToSpeech, *texts: str) -> None:
    if isinstance(voice, CachedTextToSpeech):
        for text in texts:
            voice.preload(text)


def warmup_tts(voice: TextToSpeech) -> None:
    for text in TTS_WARMUP_TEXTS:
        started = time.monotonic()
        chunks = 0
        bytes_total = 0
        for chunk in voice.synthesize_pcm(text):
            if chunk:
                chunks += 1
                bytes_total += len(chunk)
        log(
            "tts warmup: "
            f"text_chars={len(text)} chunks={chunks} bytes={bytes_total} "
            f"elapsed={time.monotonic() - started:.2f}s"
        )


def create_piper_tts() -> TextToSpeech:
    require_file(PIPER_MODEL)
    require_file(PIPER_CONFIG)
    if not PIPER_ESPEAK_DATA.is_dir():
        raise FileNotFoundError(f"missing required directory: {PIPER_ESPEAK_DATA}")
    with PIPER_CONFIG.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    audio = config.get("audio") if isinstance(config, dict) else None
    sample_rate = int((audio or {}).get("sample_rate") or 22050)
    log(f"Piper TTS config: model={PIPER_MODEL} sample_rate={sample_rate} speaker={PIPER_SPEAKER}")
    return PiperTTS(sample_rate)


def synthesize_online_audio(text: str) -> bytes:
    payload: dict[str, Any] = {
        "model": VOICE_TTS_MODEL,
        "input": text,
        "voice": ONLINE_TTS_VOICE,
        "response_format": ONLINE_TTS_RESPONSE_FORMAT,
        "speed": ONLINE_TTS_SPEED,
    }
    if ONLINE_TTS_INSTRUCTIONS:
        payload["instructions"] = ONLINE_TTS_INSTRUCTIONS
    timeout = httpx.Timeout(connect=5.0, read=ONLINE_TTS_TIMEOUT_SECONDS, write=15.0, pool=5.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            online_tts_url(),
            headers={"Content-Type": "application/json", **online_audio_headers(ONLINE_TTS_API_KEY)},
            json=payload,
        )
        response.raise_for_status()
    return response.content


def synthesize_remote_wav(text: str) -> bytes:
    payload = {
        "text": text,
        "online_available": cached_online_available(TTS_ROUTE_CACHE, TTS_ROUTE_CACHE_LOCK),
    }
    timeout = httpx.Timeout(connect=5.0, read=ONLINE_TTS_TIMEOUT_SECONDS + 10, write=15.0, pool=5.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(TTS_SERVICE_URL, json=payload)
        response.raise_for_status()
    route = response.headers.get("X-Chat2Me-TTS-Route", "")
    engine = response.headers.get("X-Chat2Me-TTS-Engine", "")
    model = response.headers.get("X-Chat2Me-TTS-Model", "")
    fallback = response.headers.get("X-Chat2Me-TTS-Fallback", "")
    log(f"remote tts result: route={route} engine={engine} model={model} fallback={fallback}")
    return response.content


def wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise RuntimeError("remote TTS must return 16-bit PCM wav")
    if channels == 1:
        return frames, int(sample_rate)
    samples = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    mono_samples = samples.mean(axis=1).astype(np.int16)
    return mono_samples.tobytes(), int(sample_rate)


def decode_online_tts_audio(audio: bytes, response_format: str, sample_rate: int) -> bytes:
    if response_format == "pcm":
        return audio
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, input=audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"online TTS audio decode failed: {detail}")
    return result.stdout


def create_online_tts() -> TextToSpeech:
    log(
        "Online TTS config: "
        f"url={online_tts_url()} model={VOICE_TTS_MODEL} voice={ONLINE_TTS_VOICE} "
        f"format={ONLINE_TTS_RESPONSE_FORMAT} sample_rate={ONLINE_TTS_SAMPLE_RATE}"
    )
    return OnlineTextToSpeech()


def tensor_audio_bytes(audio: Any) -> bytes:
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
    except Exception:
        pass
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def tts_playback_mode() -> str:
    if TTS_PLAYBACK_MODE in {"buffered", "hybrid", "stream"}:
        return TTS_PLAYBACK_MODE
    raise RuntimeError("TTS_PLAYBACK_MODE must be buffered, hybrid, or stream")


def write_player_stdin(player: subprocess.Popen[bytes], chunks: Iterable[bytes]) -> None:
    assert player.stdin is not None
    try:
        for chunk in chunks:
            if chunk:
                player.stdin.write(chunk)
    finally:
        player.stdin.close()


def play_pcm_chunks(command: list[str], chunks: Iterable[bytes]) -> None:
    with subprocess.Popen(command, stdin=subprocess.PIPE) as player:
        write_player_stdin(player, chunks)
        return_code = player.wait(timeout=TTS_PLAYER_TIMEOUT_SECONDS)
    if return_code != 0:
        raise RuntimeError(f"aplay exited with status {return_code}")


def speak(text: str, voice: TextToSpeech, config: Any = None) -> None:
    if not text:
        return
    log(f"{VOICE_TTS_ENGINE} tts: text={text}")

    def playback_command() -> list[str]:
        return [
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

    mode = tts_playback_mode()
    if mode == "buffered":
        started = time.monotonic()
        pcm = b"".join(chunk for chunk in voice.synthesize_pcm(text) if chunk)
        duration = len(pcm) / max(1, voice.config.sample_rate * 2)
        log(f"tts buffered pcm: bytes={len(pcm)} duration={duration:.2f}s synth={time.monotonic() - started:.2f}s")
        result = subprocess.run(playback_command(), input=pcm, check=False, timeout=TTS_PLAYER_TIMEOUT_SECONDS)
        if result.returncode != 0:
            raise RuntimeError(f"aplay exited with status {result.returncode}")
        return

    if mode == "hybrid":
        started = time.monotonic()
        min_bytes = int(max(0.0, TTS_PREBUFFER_SECONDS) * voice.config.sample_rate * 2)
        iterator = iter(voice.synthesize_pcm(text))
        buffered: list[bytes] = []
        buffered_bytes = 0
        for chunk in iterator:
            if not chunk:
                continue
            buffered.append(chunk)
            buffered_bytes += len(chunk)
            if buffered_bytes >= min_bytes:
                break
        duration = buffered_bytes / max(1, voice.config.sample_rate * 2)
        log(f"tts hybrid prebuffer: bytes={buffered_bytes} duration={duration:.2f}s wait={time.monotonic() - started:.2f}s")
        play_pcm_chunks(playback_command(), chain(buffered, iterator))
        return

    play_pcm_chunks(playback_command(), voice.synthesize_pcm(text))


def spoken_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if SPEECH_TTS_MAX_CHARS <= 0 or len(cleaned) <= SPEECH_TTS_MAX_CHARS:
        return cleaned

    parts = re.findall(r"[^。！？!?]+[。！？!?]?", cleaned)
    output = ""
    for part in parts:
        if not part:
            continue
        if len(output) + len(part) > SPEECH_TTS_MAX_CHARS:
            break
        output += part

    if not output:
        output = cleaned[:SPEECH_TTS_MAX_CHARS].rstrip("，,、；;：: ")
    output = output.rstrip()
    if output and output[-1] not in "。！？!?":
        output += "。"
    return output


def speak_pausing_input(
    audio: Any,
    text: str,
    voice: TextToSpeech,
    config: Any,
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


def drain_audio(audio: Any, seconds: float) -> None:
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


def read_mono(audio: Any, frames: int) -> np.ndarray:
    samples, overflowed = audio.read(frames)
    if overflowed:
        log("audio input overflowed; command audio may be clipped")
    return mono(samples).reshape(-1)


def audio_rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def decode_ready_asr(recognizer: StreamingRecognizer, stream: Any) -> str:
    return recognizer.decode_ready(stream)


def feed_asr(
    recognizer: StreamingRecognizer,
    stream: Any,
    samples: np.ndarray,
    last_text: str,
) -> str:
    recognizer.accept_waveform(stream, SAMPLE_RATE, samples)
    result = decode_ready_asr(recognizer, stream)
    if result and result != last_text:
        log(f"asr partial: {result}")
        return result
    return last_text


def calibrate_asr_noise(audio: Any, frames: int) -> tuple[float, list[tuple[np.ndarray, float]]]:
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
    audio: Any,
    ready_beep_path: Path | None,
    recognizer: StreamingRecognizer,
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

    recognizer.input_finished(stream)
    final_text = (decode_ready_asr(recognizer, stream) or last_text).strip()
    del stream
    gc.collect()
    log(
        "asr finished: "
        f"text='{final_text}' speech_started={speech_started} max_rms={max_rms:.4f} "
        f"elapsed={time.monotonic() - started:.1f}s"
    )
    return final_text


def refresh_llm_route_cache() -> None:
    try:
        timeout = min(max(LLM_ROUTE_CACHE_INTERVAL_SECONDS * 0.5, 0.2), 1.0)
        with httpx.Client(timeout=timeout) as client:
            response = client.get(CORE_REACHABILITY_URL)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
    except Exception as exc:
        log(f"llm reachability cache unavailable: {exc}")
        data = {"online": False, "provider": "", "model": "", "status": "unavailable"}

    route = "online" if data.get("online") is True else "local"
    with LLM_ROUTE_CACHE_LOCK:
        LLM_ROUTE_CACHE.update(
            {
                "online": data.get("online") is True,
                "route": route,
                "provider": str(data.get("provider") or ""),
                "model": str(data.get("model") or ""),
                "status": str(data.get("status") or ""),
                "updated_at": time.time(),
            }
        )


def refresh_service_reachability(url: str, cache: dict[str, Any], lock: threading.Lock, label: str) -> None:
    try:
        with httpx.Client(timeout=1.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
    except Exception as exc:
        log(f"{label} reachability cache unavailable: {exc}")
        data = {"online": False, "provider": "", "model": "", "status": "unavailable"}

    with lock:
        cache.update(
            {
                "online": bool(data.get("online") is True),
                "provider": str(data.get("provider") or ""),
                "model": str(data.get("model") or ""),
                "status": str(data.get("status") or ""),
                "updated_at": time.time(),
            }
        )


def service_reachability_loop(
    url: str,
    cache: dict[str, Any],
    lock: threading.Lock,
    label: str,
    interval_seconds: float,
) -> None:
    interval = max(0.5, interval_seconds)
    while True:
        refresh_service_reachability(url, cache, lock, label)
        time.sleep(interval)


def llm_route_cache_loop() -> None:
    interval = max(0.5, LLM_ROUTE_CACHE_INTERVAL_SECONDS)
    while True:
        refresh_llm_route_cache()
        time.sleep(interval)


def start_llm_route_cache() -> None:
    refresh_llm_route_cache()
    threading.Thread(target=llm_route_cache_loop, daemon=True).start()


def start_asr_route_cache() -> None:
    refresh_service_reachability(ASR_REACHABILITY_URL, ASR_ROUTE_CACHE, ASR_ROUTE_CACHE_LOCK, "asr")
    threading.Thread(
        target=service_reachability_loop,
        args=(ASR_REACHABILITY_URL, ASR_ROUTE_CACHE, ASR_ROUTE_CACHE_LOCK, "asr", ASR_ROUTE_CACHE_INTERVAL_SECONDS),
        daemon=True,
    ).start()


def start_tts_route_cache() -> None:
    refresh_service_reachability(TTS_REACHABILITY_URL, TTS_ROUTE_CACHE, TTS_ROUTE_CACHE_LOCK, "tts")
    threading.Thread(
        target=service_reachability_loop,
        args=(TTS_REACHABILITY_URL, TTS_ROUTE_CACHE, TTS_ROUTE_CACHE_LOCK, "tts", TTS_ROUTE_CACHE_INTERVAL_SECONDS),
        daemon=True,
    ).start()


def log_llm_online_cache() -> None:
    with LLM_ROUTE_CACHE_LOCK:
        cached = dict(LLM_ROUTE_CACHE)
    log(
        "llm online cache for session: "
        f"online={cached.get('online')} provider={cached['provider']} model={cached['model']} status={cached['status']}"
    )


def ask_core(text: str) -> str:
    payload: dict[str, Any] = {
        "message": text,
        "online_available": cached_online_available(LLM_ROUTE_CACHE, LLM_ROUTE_CACHE_LOCK),
    }
    with httpx.Client(timeout=CORE_REQUEST_TIMEOUT_SECONDS) as client:
        response = client.post(CORE_URL, json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    return str(data.get("answer", "")).strip()


def is_session_end(command: str) -> bool:
    normalized = re.sub(r"[\s，。！？,.!?]", "", command)
    return any(phrase in normalized for phrase in SESSION_END_PHRASES)


def handle_conversation_turn(
    audio: Any,
    command: str,
    voice: TextToSpeech,
    tts_config: Any,
    display: DisplayClient,
) -> bool:
    if is_session_end(command):
        log(f"session end command: {command}")
        speak_pausing_input(audio, SESSION_END_RESPONSE, voice, tts_config, display)
        display.set_state("idle")
        return False

    log(f"recognized command: {command}")
    display.set_state("thinking", command)
    try:
        answer = ask_core(command)
    except Exception as exc:
        log(f"core request failed: {exc}")
        display.set_state("error", "core unavailable")
        speak_pausing_input(audio, CORE_UNAVAILABLE_RESPONSE, voice, tts_config, display)
        return True

    log(f"answer: {answer}")
    speech_answer = spoken_text(answer)
    if speech_answer != answer:
        log(f"answer shortened for speech: {speech_answer}")
    speak_pausing_input(audio, speech_answer, voice, tts_config, display)
    return True
