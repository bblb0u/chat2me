from __future__ import annotations

import gc
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from piper.config import SynthesisConfig
from piper.voice import PiperVoice
import sherpa_onnx
import sounddevice as sd


MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
KWS_MODEL_DIR = Path(
    os.getenv("KWS_MODEL_DIR", str(MODELS_DIR / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"))
)
ASR_MODEL_DIR = Path(
    os.getenv("ASR_MODEL_DIR", str(MODELS_DIR / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"))
)
KEYWORDS_RAW = Path(os.getenv("KEYWORDS_RAW", "/app/config/wake_keywords_raw.txt"))
KEYWORDS_FILE = Path(os.getenv("KEYWORDS_FILE", "/app/config/wake_keywords.txt"))
GENERATED_KEYWORDS_FILE = MODELS_DIR / "wake_keywords.txt"

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://voice-gateway:8080/chat")
INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE", "ReSpeaker")
OUTPUT_DEVICE = os.getenv("AUDIO_OUTPUT_DEVICE", "default")
SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
INPUT_CHANNELS = int(os.getenv("AUDIO_INPUT_CHANNELS", "1"))
INPUT_CHANNEL_INDEX = int(os.getenv("AUDIO_INPUT_CHANNEL_INDEX", "0"))
KWS_THREADS = int(os.getenv("KWS_THREADS", "1"))
ASR_THREADS = int(os.getenv("ASR_THREADS", "2"))
COMMAND_TIMEOUT_SECONDS = float(os.getenv("COMMAND_TIMEOUT_SECONDS", "10"))
COMMAND_MIN_SECONDS = float(os.getenv("COMMAND_MIN_SECONDS", "1.8"))
COMMAND_LEADING_SILENCE_SECONDS = float(os.getenv("COMMAND_LEADING_SILENCE_SECONDS", "4.0"))
COMMAND_INITIAL_GRACE_SECONDS = float(os.getenv("COMMAND_INITIAL_GRACE_SECONDS", "1.2"))
PRE_BEEP_DRAIN_SECONDS = float(os.getenv("PRE_BEEP_DRAIN_SECONDS", "0.05"))
POST_BEEP_DRAIN_SECONDS = float(os.getenv("POST_BEEP_DRAIN_SECONDS", "0.05"))
POST_RESPONSE_DRAIN_SECONDS = float(os.getenv("POST_RESPONSE_DRAIN_SECONDS", "0.5"))
SPEECH_RMS_THRESHOLD = float(os.getenv("SPEECH_RMS_THRESHOLD", "0.006"))
PIPER_MODEL = Path(os.getenv("PIPER_MODEL", str(MODELS_DIR / "piper/zh_CN-huayan-medium/model.onnx")))
PIPER_CONFIG = Path(os.getenv("PIPER_CONFIG", str(PIPER_MODEL) + ".json"))
PIPER_SPEAKER = int(os.getenv("PIPER_SPEAKER", "0"))
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "0.9"))
PIPER_NOISE_SCALE = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
PIPER_NOISE_W_SCALE = float(os.getenv("PIPER_NOISE_W_SCALE", "0.8"))
PIPER_VOLUME = float(os.getenv("PIPER_VOLUME", "1.0"))
NO_COMMAND_RESPONSE = os.getenv("NO_COMMAND_RESPONSE", "请再说一遍")
WAKE_RESPONSE = os.getenv("WAKE_RESPONSE", "有什么可以帮助您的")
SESSION_IDLE_RESPONSE = os.getenv("SESSION_IDLE_RESPONSE", "")
SESSION_END_RESPONSE = os.getenv("SESSION_END_RESPONSE", "好的，我先待机")
MAX_SESSION_TURNS = int(os.getenv("MAX_SESSION_TURNS", "8"))
SESSION_END_PHRASES = tuple(
    phrase.strip()
    for phrase in os.getenv("SESSION_END_PHRASES", "退出,结束,不用了,没事了,再见,拜拜,回到待机").split(",")
    if phrase.strip()
)


def log(message: str) -> None:
    print(f"[voice-agent] {message}", flush=True)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")


def select_input_device(selector: str) -> int | str | None:
    if not selector:
        return None
    if selector.isdigit():
        return int(selector)

    devices = sd.query_devices()
    selector_lower = selector.lower()
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0 and selector_lower in str(device.get("name", "")).lower():
            return index

    log(f"input device containing '{selector}' not found; using PortAudio default")
    return None


def ensure_keywords_file() -> Path:
    require_file(KWS_MODEL_DIR / "tokens.txt")
    require_file(KWS_MODEL_DIR / "en.phone")

    if KEYWORDS_FILE.is_file():
        return KEYWORDS_FILE

    keywords_file = KEYWORDS_FILE
    if not os.getenv("KEYWORDS_FILE"):
        keywords_file = GENERATED_KEYWORDS_FILE
        if keywords_file.is_file():
            return keywords_file

    if not KEYWORDS_RAW.is_file():
        KEYWORDS_RAW.parent.mkdir(parents=True, exist_ok=True)
        KEYWORDS_RAW.write_text("嗨小紫 @嗨小紫\n", encoding="utf-8")

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
        str(KEYWORDS_RAW),
        str(keywords_file),
    ]
    log("generating wake keyword tokens")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 and not keywords_file.is_file():
        raise RuntimeError(f"wake keyword token generation failed with exit code {result.returncode}")
    if result.returncode != 0:
        log(f"keyword token generator returned {result.returncode}; output exists, continuing")
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
        keywords_score=float(os.getenv("KWS_KEYWORDS_SCORE", "1.5")),
        keywords_threshold=float(os.getenv("KWS_KEYWORDS_THRESHOLD", "0.25")),
        provider="cpu",
    )


def create_asr() -> sherpa_onnx.OnlineRecognizer:
    require_file(ASR_MODEL_DIR / "tokens.txt")
    require_file(ASR_MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx")
    require_file(ASR_MODEL_DIR / "decoder-epoch-99-avg-1.int8.onnx")
    require_file(ASR_MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx")

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(ASR_MODEL_DIR / "tokens.txt"),
        encoder=str(ASR_MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(ASR_MODEL_DIR / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(ASR_MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=ASR_THREADS,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=float(os.getenv("ASR_RULE1_MIN_TRAILING_SILENCE", "1.8")),
        rule2_min_trailing_silence=float(os.getenv("ASR_RULE2_MIN_TRAILING_SILENCE", "1.2")),
        rule3_min_utterance_length=float(os.getenv("ASR_RULE3_MIN_UTTERANCE_LENGTH", "8")),
        decoding_method="greedy_search",
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
        return_code = player.wait(timeout=30)
    if return_code != 0:
        raise RuntimeError(f"aplay exited with status {return_code}")


def drain_audio(audio: sd.InputStream, seconds: float) -> None:
    chunk = int(0.1 * SAMPLE_RATE)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        audio.read(chunk)


def mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    channel_index = min(max(INPUT_CHANNEL_INDEX, 0), samples.shape[1] - 1)
    return samples[:, channel_index]


def read_mono(audio: sd.InputStream, frames: int) -> np.ndarray:
    samples, overflowed = audio.read(frames)
    if overflowed:
        log("audio input overflowed; command audio may be clipped")
    return mono(samples).reshape(-1)


def listen_command(
    audio: sd.InputStream,
    ready_beep_path: Path | None,
    recognizer: sherpa_onnx.OnlineRecognizer,
    play_ready_beep: bool = True,
) -> str:
    stream = recognizer.create_stream()
    chunk = int(0.1 * SAMPLE_RATE)
    last_text = ""
    speech_started = False
    max_rms = 0.0

    if play_ready_beep and ready_beep_path is not None:
        drain_audio(audio, PRE_BEEP_DRAIN_SECONDS)
        play_wav(ready_beep_path)
        drain_audio(audio, POST_BEEP_DRAIN_SECONDS)
    log("listening for command")
    started = time.monotonic()
    while time.monotonic() - started < COMMAND_TIMEOUT_SECONDS:
        samples = read_mono(audio, chunk)
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        max_rms = max(max_rms, rms)
        if rms >= SPEECH_RMS_THRESHOLD:
            speech_started = True

        stream.accept_waveform(SAMPLE_RATE, samples)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        if result and result.strip() != last_text:
            last_text = result.strip()
            log(f"asr partial: {last_text}")

        elapsed = time.monotonic() - started
        if not speech_started and not last_text and elapsed >= COMMAND_LEADING_SILENCE_SECONDS:
            break

        if elapsed < max(COMMAND_MIN_SECONDS, COMMAND_INITIAL_GRACE_SECONDS):
            continue

        if recognizer.is_endpoint(stream) and (speech_started or last_text):
            break

    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    final_text = (recognizer.get_result(stream) or last_text).strip()
    del stream
    gc.collect()
    log(
        "asr finished: "
        f"text='{final_text}' speech_started={speech_started} max_rms={max_rms:.4f} "
        f"elapsed={time.monotonic() - started:.1f}s"
    )
    return final_text


def ask_gateway(text: str) -> str:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(GATEWAY_URL, json={"message": text})
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    return str(data.get("answer", "")).strip()


def is_session_end(command: str) -> bool:
    normalized = re.sub(r"[\s，。！？,.!?]", "", command)
    return any(phrase in normalized for phrase in SESSION_END_PHRASES)


def handle_conversation_turn(command: str, voice: PiperVoice, tts_config: SynthesisConfig) -> bool:
    if is_session_end(command):
        log(f"session end command: {command}")
        speak(SESSION_END_RESPONSE, voice, tts_config)
        return False

    log(f"recognized command: {command}")
    try:
        answer = ask_gateway(command)
    except Exception as exc:
        log(f"gateway request failed: {exc}")
        speak("对话服务暂时不可用", voice, tts_config)
        return True

    log(f"answer: {answer}")
    speak(answer, voice, tts_config)
    return True


def handle_wake(
    audio: sd.InputStream,
    beep_path: Path,
    recognizer: sherpa_onnx.OnlineRecognizer,
    voice: PiperVoice,
    tts_config: SynthesisConfig,
) -> None:
    log("wake detected")
    speak(WAKE_RESPONSE, voice, tts_config)
    drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

    for turn in range(1, MAX_SESSION_TURNS + 1):
        log(f"conversation turn {turn}/{MAX_SESSION_TURNS}")
        try:
            command = listen_command(audio, beep_path, recognizer, play_ready_beep=False)
        except Exception as exc:
            log(f"asr failed in conversation: {exc}")
            speak("语音识别出错", voice, tts_config)
            return

        if not command:
            log("conversation idle timeout")
            if SESSION_IDLE_RESPONSE:
                speak(SESSION_IDLE_RESPONSE, voice, tts_config)
            return

        if not handle_conversation_turn(command, voice, tts_config):
            return
        drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

    log("conversation reached max turns")
    speak(SESSION_END_RESPONSE, voice, tts_config)


def main() -> None:
    input_device = select_input_device(INPUT_DEVICE)
    log(f"input device: {input_device if input_device is not None else 'default'}")
    log(f"output device: {OUTPUT_DEVICE}")
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
    chunk = int(0.1 * SAMPLE_RATE)
    log("preloading ASR model")
    recognizer = create_asr()
    log("ASR model ready")

    beep_path = Path(tempfile.gettempdir()) / "chat2m_wake.wav"
    write_beep(beep_path)

    log("wake listener active: 嗨小紫")
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
                        handle_wake(audio, beep_path, recognizer, voice, tts_config)
                    except Exception as exc:
                        log(f"wake handling failed: {exc}")
                    drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)
                    stream = kws.create_stream()
                    log("wake listener active: 嗨小紫")
                    break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
    except Exception as exc:
        log(f"fatal: {exc}")
        sys.exit(1)
