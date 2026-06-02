from __future__ import annotations

import asyncio
import io
import os
import time
import threading
import wave
from dataclasses import dataclass

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app import voice
from app.common import env_float, log


ASR_HOST = os.getenv("ASR_HOST", "0.0.0.0")
ASR_PORT = int(os.getenv("ASR_PORT", "8092"))
VOICE_ASR_FALLBACK_ENGINE = os.getenv("VOICE_ASR_FALLBACK_ENGINE", "sensevoice").strip().lower() or "sensevoice"
VOICE_ASR_FALLBACK_MODEL = os.getenv("VOICE_ASR_FALLBACK_MODEL", "SenseVoiceSmall").strip() or "SenseVoiceSmall"
CONFIGURED_ASR_ENGINE = voice.VOICE_ASR_ENGINE
CONFIGURED_ASR_MODEL = voice.VOICE_ASR_MODEL
ASR_REACHABILITY_INTERVAL_SECONDS = env_float("ASR_REACHABILITY_INTERVAL_SECONDS")
ASR_REACHABILITY_TIMEOUT_SECONDS = env_float("ASR_REACHABILITY_TIMEOUT_SECONDS")
ONLINE_ASR_REACHABILITY_PATH = os.getenv("ONLINE_ASR_REACHABILITY_PATH", "/models").strip() or "/models"
if not ONLINE_ASR_REACHABILITY_PATH.startswith("/"):
    ONLINE_ASR_REACHABILITY_PATH = "/" + ONLINE_ASR_REACHABILITY_PATH


@dataclass(frozen=True)
class Reachability:
    online: bool
    status: str
    checked_at: float | None = None


class HealthResponse(BaseModel):
    ok: bool
    configured_engine: str
    active_online: bool
    online_status: str
    local_engine: str
    local_model: str


class ReachabilityResponse(BaseModel):
    online: bool
    provider: str
    model: str | None
    status: str
    checked_at: float | None = None


class TranscribeResponse(BaseModel):
    text: str
    route: str
    engine: str
    model: str
    fallback: bool = False
    online_status: str | None = None
    latency_ms: int


app = FastAPI(title="Chat2Me ASR", version="0.1.0")
ONLINE_REACHABILITY = Reachability(online=False, status="not_checked")
ONLINE_REACHABILITY_TASK: asyncio.Task[None] | None = None
LOCAL_RECOGNIZER: voice.StreamingRecognizer | None = None
ONLINE_RECOGNIZER: voice.StreamingRecognizer | None = None
ASR_INFERENCE_LOCK = threading.Lock()


def online_enabled() -> bool:
    return CONFIGURED_ASR_ENGINE == "online"


def local_engine() -> str:
    return VOICE_ASR_FALLBACK_ENGINE if online_enabled() else CONFIGURED_ASR_ENGINE


def local_model() -> str:
    return VOICE_ASR_FALLBACK_MODEL if online_enabled() else CONFIGURED_ASR_MODEL


def with_asr_env(engine: str, model: str) -> None:
    voice.VOICE_ASR_ENGINE = engine
    voice.VOICE_ASR_MODEL = model
    voice.ASR_MODEL_DIR = voice.MODELS_DIR / engine / model
    if engine == "sensevoice":
        voice.SENSEVOICE_MODEL_DIR = voice.ASR_MODEL_DIR
        voice.SENSEVOICE_VAD_MODEL_DIR = voice.MODELS_DIR / "sensevoice" / "speech_fsmn_vad_zh-cn-16k-common-onnx"


def create_local_recognizer() -> voice.StreamingRecognizer:
    engine = local_engine()
    model = local_model()
    with_asr_env(engine, model)
    recognizer = voice.create_asr()
    log(f"local ASR ready: engine={engine} model={model}")
    return recognizer


def create_online_recognizer() -> voice.StreamingRecognizer:
    if not online_enabled():
        raise RuntimeError("online ASR is not configured")
    with_asr_env("online", CONFIGURED_ASR_MODEL)
    recognizer = voice.create_asr()
    log(f"online ASR ready: model={CONFIGURED_ASR_MODEL}")
    return recognizer


def read_wav_upload(payload: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise HTTPException(status_code=400, detail="only 16-bit PCM wav is supported")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32, copy=False), int(sample_rate)


def transcribe_with_recognizer(recognizer: voice.StreamingRecognizer, samples: np.ndarray, sample_rate: int) -> str:
    stream = recognizer.create_stream()
    try:
        chunk = max(1, int(sample_rate * voice.CHUNK_SECONDS))
        for offset in range(0, len(samples), chunk):
            recognizer.accept_waveform(stream, sample_rate, samples[offset : offset + chunk])
        recognizer.input_finished(stream)
        return recognizer.decode_ready(stream).strip()
    finally:
        del stream


async def probe_online() -> Reachability:
    if not online_enabled():
        return Reachability(online=False, status="online_engine_disabled", checked_at=time.time())
    if not voice.ONLINE_ASR_BASE_URL:
        return Reachability(online=False, status="config_error:ONLINE_ASR_BASE_URL", checked_at=time.time())
    if not voice.ONLINE_ASR_API_KEY:
        return Reachability(online=False, status="config_error:ONLINE_ASR_API_KEY", checked_at=time.time())
    try:
        async with httpx.AsyncClient(timeout=ASR_REACHABILITY_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f"{voice.ONLINE_ASR_BASE_URL}{ONLINE_ASR_REACHABILITY_PATH}",
                headers=voice.online_audio_headers(voice.ONLINE_ASR_API_KEY),
            )
            status = "ok" if response.is_success else f"http_{response.status_code}"
            return Reachability(online=response.is_success, status=status, checked_at=time.time())
    except httpx.HTTPError:
        return Reachability(online=False, status="unreachable", checked_at=time.time())


async def reachability_loop() -> None:
    global ONLINE_REACHABILITY
    interval = max(0.5, ASR_REACHABILITY_INTERVAL_SECONDS)
    while True:
        ONLINE_REACHABILITY = await probe_online()
        await asyncio.sleep(interval)


@app.on_event("startup")
async def startup() -> None:
    global LOCAL_RECOGNIZER, ONLINE_RECOGNIZER, ONLINE_REACHABILITY, ONLINE_REACHABILITY_TASK
    LOCAL_RECOGNIZER = create_local_recognizer()
    if online_enabled():
        ONLINE_RECOGNIZER = create_online_recognizer()
        ONLINE_REACHABILITY = await probe_online()
        ONLINE_REACHABILITY_TASK = asyncio.create_task(reachability_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    if ONLINE_REACHABILITY_TASK is None:
        return
    ONLINE_REACHABILITY_TASK.cancel()
    try:
        await ONLINE_REACHABILITY_TASK
    except asyncio.CancelledError:
        pass


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        ok=LOCAL_RECOGNIZER is not None,
        configured_engine=CONFIGURED_ASR_ENGINE,
        active_online=online_enabled() and ONLINE_REACHABILITY.online,
        online_status=ONLINE_REACHABILITY.status,
        local_engine=local_engine(),
        local_model=local_model(),
    )


@app.get("/asr/reachability", response_model=ReachabilityResponse)
async def reachability() -> ReachabilityResponse:
    return ReachabilityResponse(
        online=ONLINE_REACHABILITY.online,
        provider="online-asr" if online_enabled() else "local",
        model=CONFIGURED_ASR_MODEL if online_enabled() else None,
        status=ONLINE_REACHABILITY.status if online_enabled() else "online_engine_disabled",
        checked_at=ONLINE_REACHABILITY.checked_at,
    )


@app.post("/asr/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    online_available: bool | None = Form(default=None),
) -> TranscribeResponse:
    started_at = time.perf_counter()
    payload = await file.read()
    samples, sample_rate = read_wav_upload(payload)
    use_online = online_enabled() and (ONLINE_REACHABILITY.online if online_available is None else online_available)

    with ASR_INFERENCE_LOCK:
        if use_online and ONLINE_RECOGNIZER is not None:
            try:
                with_asr_env("online", CONFIGURED_ASR_MODEL)
                text = transcribe_with_recognizer(ONLINE_RECOGNIZER, samples, sample_rate)
                return TranscribeResponse(
                    text=text,
                    route="online",
                    engine="online",
                    model=CONFIGURED_ASR_MODEL,
                    online_status=ONLINE_REACHABILITY.status,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                )
            except Exception as exc:
                log(f"online ASR failed; falling back to local: {exc}")

        if LOCAL_RECOGNIZER is None:
            raise HTTPException(status_code=503, detail="local ASR is not ready")
        with_asr_env(local_engine(), local_model())
        text = transcribe_with_recognizer(LOCAL_RECOGNIZER, samples, sample_rate)
    return TranscribeResponse(
        text=text,
        route="local",
        engine=local_engine(),
        model=local_model(),
        fallback=online_enabled(),
        online_status=ONLINE_REACHABILITY.status,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=ASR_HOST, port=ASR_PORT)
