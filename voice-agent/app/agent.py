from __future__ import annotations

import gc
import inspect
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
from types import SimpleNamespace
from typing import Any, Iterable, Protocol

from app.runtime import DisplayClient, env_bool, env_float, env_int, env_value, log

import httpx
import numpy as np
import yaml
import sounddevice as sd


MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
VOICE_KWS_MODEL = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
VOICE_ASR_ENGINE = env_value("VOICE_ASR_ENGINE")
VOICE_ASR_MODEL = env_value("VOICE_ASR_MODEL")
VOICE_TTS_MODEL = env_value("VOICE_TTS_MODEL")
VOICE_TTS_ENGINE = env_value("VOICE_TTS_ENGINE")
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
PIPER_MODEL = MODELS_DIR / VOICE_TTS_ENGINE / VOICE_TTS_MODEL / "model.onnx"
PIPER_CONFIG = Path(str(PIPER_MODEL) + ".json")
PIPER_SPEAKER = env_int("PIPER_SPEAKER")
PIPER_LENGTH_SCALE = env_float("PIPER_LENGTH_SCALE")
PIPER_NOISE_SCALE = env_float("PIPER_NOISE_SCALE")
PIPER_NOISE_W_SCALE = env_float("PIPER_NOISE_W_SCALE")
PIPER_VOLUME = env_float("PIPER_VOLUME")
TTS_PLAYER_TIMEOUT_SECONDS = env_float("TTS_PLAYER_TIMEOUT_SECONDS")
SPEECH_TTS_MAX_CHARS = env_int("SPEECH_TTS_MAX_CHARS")
TTS_CACHE_ENABLED = env_bool("TTS_CACHE_ENABLED")
TTS_CACHE_MAX_ITEMS = env_int("TTS_CACHE_MAX_ITEMS")
TTS_CACHE_MAX_BYTES = env_int("TTS_CACHE_MAX_BYTES")
TTS_MODEL_DIR = MODELS_DIR / VOICE_TTS_ENGINE / VOICE_TTS_MODEL
COSYVOICE_SPK_ID = os.getenv("COSYVOICE_SPK_ID", "中文女").strip() or "中文女"
COSYVOICE_INSTRUCT_TEXT = os.getenv("COSYVOICE_INSTRUCT_TEXT", "用自然、清晰、亲切的语气说话。").strip()
COSYVOICE_SPEED = env_float("COSYVOICE_SPEED")
COSYVOICE_TEXT_FRONTEND = env_bool("COSYVOICE_TEXT_FRONTEND")
COSYVOICE_LOAD_JIT = env_bool("COSYVOICE_LOAD_JIT")
COSYVOICE_LOAD_TRT = env_bool("COSYVOICE_LOAD_TRT")
COSYVOICE_FP16 = env_bool("COSYVOICE_FP16")
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


def wake_words_display() -> str:
    return " / ".join(WAKE_WORDS)


def select_input_device(selector: str) -> int | str | None:
    if not selector:
        return None
    if selector.isdigit():
        return int(selector)

    devices = sd.query_devices()
    selector_lower = selector.lower()
    matched_without_input = False
    for index, device in enumerate(devices):
        if selector_lower not in str(device.get("name", "")).lower():
            continue
        if device.get("max_input_channels", 0) > 0:
            return index
        matched_without_input = True

    if matched_without_input:
        log(f"input device containing '{selector}' has no input channels; using PortAudio default")
        return None

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


def create_kws() -> Any:
    import sherpa_onnx

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
        from sense_voice_streaming_asr.sense_voice_streaming_asr import StreamingASREventType

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
        f"max_active_paths={ASR_MAX_ACTIVE_PATHS} hotwords={HOTWORDS_PATH}"
    )

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
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
    return SherpaStreamingRecognizer(recognizer)


def create_sensevoice_asr() -> StreamingRecognizer:
    import json

    import kaldi_native_fbank as knf
    import onnxruntime
    from sense_voice_streaming_asr.cmvn_utils import load_cmvn
    from sense_voice_streaming_asr.model_data import SenseVoiceModel, VadModel
    from sense_voice_streaming_asr.sense_voice_streaming_asr import SenseVoiceStreamingASR, StreamingASRConfig

    require_file(SENSEVOICE_MODEL_DIR / "model_quant.onnx")
    require_file(SENSEVOICE_MODEL_DIR / "am.mvn")
    require_file(SENSEVOICE_MODEL_DIR / "tokens.json")
    require_file(SENSEVOICE_VAD_MODEL_DIR / "model_quant.onnx")
    vad_cmvn_path = SENSEVOICE_VAD_MODEL_DIR / "vad.mvn"
    if not vad_cmvn_path.is_file():
        vad_cmvn_path = SENSEVOICE_VAD_MODEL_DIR / "am.mvn"
    require_file(vad_cmvn_path)

    use_cuda = os.getenv("VOICE_ASR_DEVICE", "cpu").lower().startswith("cuda")
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
        f"model={SENSEVOICE_MODEL_DIR} vad={SENSEVOICE_VAD_MODEL_DIR} cuda={use_cuda} lang={config.lang} "
        f"vad_start={config.vad_start_threshold} vad_end={config.vad_end_threshold}"
    )
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

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

    asr_model = SenseVoiceModel.__new__(SenseVoiceModel)
    asr_model.cmvn = load_cmvn(str(SENSEVOICE_MODEL_DIR / "am.mvn"))
    asr_model.sensevoice_tokens = json.loads((SENSEVOICE_MODEL_DIR / "tokens.json").read_text(encoding="utf-8"))
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

    vad_model = VadModel.__new__(VadModel)
    vad_model.cmvn = load_cmvn(str(vad_cmvn_path))
    vad_model.model_inference_session = onnxruntime.InferenceSession(
        str(SENSEVOICE_VAD_MODEL_DIR / "model_quant.onnx"),
        providers=["CPUExecutionProvider"],
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


def create_asr() -> StreamingRecognizer:
    if VOICE_ASR_ENGINE == "sherpa":
        return create_sherpa_asr()
    if VOICE_ASR_ENGINE == "sensevoice":
        return create_sensevoice_asr()
    raise RuntimeError(f"VOICE_ASR_ENGINE '{VOICE_ASR_ENGINE}' is not supported")


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


class PiperTTS:
    def __init__(self, voice: Any, config: Any) -> None:
        self.voice = voice
        self.piper_config = config
        self.config = voice.config

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        for chunk in self.voice.synthesize(text, self.piper_config):
            yield audio_chunk_bytes(chunk)


class CosyVoiceTTS:
    def __init__(self, model: Any) -> None:
        self.model = model
        sample_rate = int(getattr(model, "sample_rate", 22050) or 22050)
        self.config = SimpleNamespace(sample_rate=sample_rate)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        kwargs = {
            "stream": True,
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


def create_piper_tts() -> TextToSpeech:
    from piper.config import SynthesisConfig
    from piper.voice import PiperVoice

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
    return PiperTTS(voice, config)


def create_cosyvoice_tts() -> TextToSpeech:
    cosyvoice_package = os.getenv("COSYVOICE_PACKAGE_PATH", "").strip()
    if cosyvoice_package:
        for path in reversed([part.strip() for part in cosyvoice_package.split(":") if part.strip()]):
            if path not in sys.path:
                sys.path.insert(0, path)
    from cosyvoice.cli.cosyvoice import CosyVoice

    require_file(TTS_MODEL_DIR / "cosyvoice.yaml")
    require_file(TTS_MODEL_DIR / "flow.pt")
    require_file(TTS_MODEL_DIR / "hift.pt")
    require_file(TTS_MODEL_DIR / "llm.pt")
    require_file(TTS_MODEL_DIR / "campplus.onnx")
    require_file(TTS_MODEL_DIR / "speech_tokenizer_v1.onnx")
    if VOICE_TTS_MODEL.endswith(("-SFT", "-Instruct")):
        require_file(TTS_MODEL_DIR / "spk2info.pt")
    device = os.getenv("VOICE_TTS_DEVICE", "cpu")
    log(
        "CosyVoice TTS config: "
        f"model={TTS_MODEL_DIR} device={device} speaker={COSYVOICE_SPK_ID} "
        f"jit={COSYVOICE_LOAD_JIT} trt={COSYVOICE_LOAD_TRT} fp16={COSYVOICE_FP16}"
    )
    init_kwargs: dict[str, Any] = {}
    signature = inspect.signature(CosyVoice)
    if "load_jit" in signature.parameters:
        init_kwargs["load_jit"] = COSYVOICE_LOAD_JIT
    if "load_onnx" in signature.parameters:
        init_kwargs["load_onnx"] = False
    if "load_trt" in signature.parameters:
        init_kwargs["load_trt"] = COSYVOICE_LOAD_TRT
    if "fp16" in signature.parameters:
        init_kwargs["fp16"] = COSYVOICE_FP16
    if "device" in signature.parameters:
        init_kwargs["device"] = device
    install_cosyvoice_inference_stubs()
    model = CosyVoice(str(TTS_MODEL_DIR), **init_kwargs)
    return CosyVoiceTTS(model)


def install_cosyvoice_inference_stubs() -> None:
    import importlib
    import types

    try:
        dataset_package = importlib.import_module("cosyvoice.dataset")
    except Exception:
        return

    processor = types.ModuleType("cosyvoice.dataset.processor")

    def unused_processor(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CosyVoice dataset processor is not available in inference runtime")

    for name in (
        "parquet_opener",
        "tokenize",
        "filter",
        "resample",
        "compute_fbank",
        "parse_embedding",
        "shuffle",
        "sort",
        "batch",
        "padding",
    ):
        setattr(processor, name, unused_processor)
    sys.modules["cosyvoice.dataset.processor"] = processor
    setattr(dataset_package, "processor", processor)

    pylogger = types.ModuleType("matcha.utils.pylogger")

    def get_pylogger(name: str = __name__) -> Any:
        import logging

        return logging.getLogger(name)

    pylogger.get_pylogger = get_pylogger
    sys.modules["matcha.utils.pylogger"] = pylogger

    def noop(*args: Any, **kwargs: Any) -> Any:
        return None

    instantiators = types.ModuleType("matcha.utils.instantiators")
    instantiators.instantiate_callbacks = lambda *args, **kwargs: []
    instantiators.instantiate_loggers = lambda *args, **kwargs: []

    logging_utils = types.ModuleType("matcha.utils.logging_utils")
    logging_utils.log_hyperparameters = noop

    rich_utils = types.ModuleType("matcha.utils.rich_utils")
    rich_utils.enforce_tags = noop
    rich_utils.print_config_tree = noop

    utils = types.ModuleType("matcha.utils.utils")
    utils.extras = noop
    utils.get_metric_value = lambda *args, **kwargs: 0.0
    utils.task_wrapper = lambda func: func

    sys.modules["matcha.utils.instantiators"] = instantiators
    sys.modules["matcha.utils.logging_utils"] = logging_utils
    sys.modules["matcha.utils.rich_utils"] = rich_utils
    sys.modules["matcha.utils.utils"] = utils


def create_tts() -> tuple[TextToSpeech, None]:
    if VOICE_TTS_ENGINE == "piper":
        return wrap_tts(create_piper_tts()), None
    if VOICE_TTS_ENGINE == "cosyvoice":
        return wrap_tts(create_cosyvoice_tts()), None
    raise RuntimeError(f"VOICE_TTS_ENGINE '{VOICE_TTS_ENGINE}' is not supported")


def wrap_tts(voice: TextToSpeech) -> TextToSpeech:
    if not TTS_CACHE_ENABLED:
        return voice
    return CachedTextToSpeech(voice, TTS_CACHE_MAX_ITEMS, TTS_CACHE_MAX_BYTES)


def preload_tts_cache(voice: TextToSpeech, *texts: str) -> None:
    if isinstance(voice, CachedTextToSpeech):
        for text in texts:
            voice.preload(text)


def audio_chunk_bytes(chunk: Any) -> bytes:
    int16_audio = getattr(chunk, "audio_int16_array", None)
    if int16_audio is not None:
        return np.asarray(int16_audio, dtype=np.int16).tobytes()

    float_audio = getattr(chunk, "audio_float_array", None)
    if float_audio is None:
        raise RuntimeError(f"unsupported Piper audio chunk: {type(chunk)!r}")

    clipped = np.clip(np.asarray(float_audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


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


def speak(text: str, voice: TextToSpeech, config: Any = None) -> None:
    if not text:
        return
    log(f"{VOICE_TTS_ENGINE} tts: text={text}")
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
            for chunk in voice.synthesize_pcm(text):
                if chunk:
                    player.stdin.write(chunk)
        finally:
            player.stdin.close()
        return_code = player.wait(timeout=TTS_PLAYER_TIMEOUT_SECONDS)
    if return_code != 0:
        raise RuntimeError(f"aplay exited with status {return_code}")


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
    audio: sd.InputStream,
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
    voice: TextToSpeech,
    tts_config: Any,
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
    speech_answer = spoken_text(answer)
    if speech_answer != answer:
        log(f"answer shortened for speech: {speech_answer}")
    speak_pausing_input(audio, speech_answer, voice, tts_config, display)
    return True


def handle_wake(
    audio: sd.InputStream,
    beep_path: Path,
    recognizer: StreamingRecognizer,
    voice: TextToSpeech,
    tts_config: Any,
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
    log(f"loading {VOICE_TTS_ENGINE} TTS model: {TTS_MODEL_DIR if VOICE_TTS_ENGINE == 'cosyvoice' else PIPER_MODEL}")
    voice, tts_config = create_tts()
    log(f"{VOICE_TTS_ENGINE} TTS ready: sample_rate={voice.config.sample_rate}")
    preload_tts_cache(voice, WAKE_RESPONSE, SESSION_END_RESPONSE, SESSION_IDLE_RESPONSE)
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
