from __future__ import annotations

import asyncio
import os
import threading
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import voice
from app.common import env_float, log


TTS_HOST = os.getenv("TTS_HOST", "0.0.0.0")
TTS_PORT = int(os.getenv("TTS_PORT", "8093"))
LOCAL_TTS_ENGINE = "melotts"
LOCAL_TTS_MODEL = "MeloTTS-Chinese"
CONFIGURED_TTS_ENGINE = voice.VOICE_TTS_ENGINE
CONFIGURED_TTS_MODEL = voice.VOICE_TTS_MODEL
TTS_REACHABILITY_INTERVAL_SECONDS = env_float("TTS_REACHABILITY_INTERVAL_SECONDS")
TTS_REACHABILITY_TIMEOUT_SECONDS = env_float("TTS_REACHABILITY_TIMEOUT_SECONDS")


@dataclass(frozen=True)
class Reachability:
    online: bool
    status: str
    checked_at: float | None = None


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    online_available: bool | None = None


class HealthResponse(BaseModel):
    ok: bool
    configured_engine: str
    active_online: bool
    online_status: str
    local_engine: str
    local_model: str
    sample_rate: int


class ReachabilityResponse(BaseModel):
    online: bool
    provider: str
    model: str | None
    status: str
    checked_at: float | None = None


ONLINE_REACHABILITY = Reachability(online=False, status="not_checked")
ONLINE_REACHABILITY_TASK: asyncio.Task[None] | None = None
LOCAL_VOICE: voice.TextToSpeech | None = None
ONLINE_VOICE: voice.TextToSpeech | None = None
TTS_INFERENCE_LOCK = threading.Lock()


def online_enabled() -> bool:
    return CONFIGURED_TTS_ENGINE == "online"


def local_engine() -> str:
    return LOCAL_TTS_ENGINE if online_enabled() else CONFIGURED_TTS_ENGINE


def local_model() -> str:
    return LOCAL_TTS_MODEL if online_enabled() else CONFIGURED_TTS_MODEL


def validate_tts_selection() -> None:
    if CONFIGURED_TTS_ENGINE not in {"melotts", "online"}:
        raise RuntimeError("VOICE_TTS_ENGINE must be melotts or online")
    if online_enabled() and CONFIGURED_TTS_MODEL != "edge-tts":
        raise RuntimeError("online TTS only supports VOICE_TTS_MODEL=edge-tts")
    if local_model() != "MeloTTS-Chinese":
        raise RuntimeError("local TTS only supports MeloTTS-Chinese")


def with_tts_env(engine: str, model: str) -> None:
    voice.VOICE_TTS_ENGINE = engine
    voice.VOICE_TTS_MODEL = model
    voice.TTS_MODEL_DIR = voice.MODELS_DIR / engine / model
    voice.MELOTTS_CONFIG_FILE = Path(os.getenv("MELOTTS_CONFIG_FILE", str(voice.TTS_MODEL_DIR / "config.json")))
    voice.MELOTTS_CKPT_FILE = Path(os.getenv("MELOTTS_CKPT_FILE", str(voice.TTS_MODEL_DIR / "checkpoint.pth")))


def create_local_voice() -> voice.TextToSpeech:
    engine = local_engine()
    model = local_model()
    with_tts_env(engine, model)
    tts_voice, _ = voice.create_tts()
    voice.preload_tts_cache(tts_voice, voice.WAKE_RESPONSE, voice.SESSION_END_RESPONSE, voice.SESSION_IDLE_RESPONSE)
    voice.warmup_tts(tts_voice)
    log(f"local TTS ready: engine={engine} model={model} sample_rate={tts_voice.config.sample_rate}")
    return tts_voice


def create_online_voice() -> voice.TextToSpeech:
    if not online_enabled():
        raise RuntimeError("online TTS is not configured")
    with_tts_env("online", CONFIGURED_TTS_MODEL)
    tts_voice, _ = voice.create_tts()
    log(f"online TTS ready: model={CONFIGURED_TTS_MODEL} sample_rate={tts_voice.config.sample_rate}")
    return tts_voice


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def synthesize_wav(tts_voice: voice.TextToSpeech, text: str) -> bytes:
    pcm = b"".join(chunk for chunk in tts_voice.synthesize_pcm(text) if chunk)
    return pcm_to_wav(pcm, int(tts_voice.config.sample_rate))


async def probe_online() -> Reachability:
    if not online_enabled():
        return Reachability(online=False, status="online_engine_disabled", checked_at=time.time())
    try:
        import edge_tts

        await asyncio.wait_for(
            edge_tts.list_voices(proxy=voice.EDGE_TTS_PROXY),
            timeout=TTS_REACHABILITY_TIMEOUT_SECONDS,
        )
        return Reachability(online=True, status="ok", checked_at=time.time())
    except asyncio.TimeoutError:
        return Reachability(online=False, status="timeout", checked_at=time.time())
    except Exception:
        return Reachability(online=False, status="unreachable", checked_at=time.time())


async def reachability_loop() -> None:
    global ONLINE_REACHABILITY
    interval = max(0.5, TTS_REACHABILITY_INTERVAL_SECONDS)
    while True:
        ONLINE_REACHABILITY = await probe_online()
        await asyncio.sleep(interval)


async def startup() -> None:
    global LOCAL_VOICE, ONLINE_VOICE, ONLINE_REACHABILITY, ONLINE_REACHABILITY_TASK
    validate_tts_selection()
    LOCAL_VOICE = create_local_voice()
    if online_enabled():
        ONLINE_VOICE = create_online_voice()
        ONLINE_REACHABILITY = await probe_online()
        ONLINE_REACHABILITY_TASK = asyncio.create_task(reachability_loop())


async def shutdown() -> None:
    if ONLINE_REACHABILITY_TASK is None:
        return
    ONLINE_REACHABILITY_TASK.cancel()
    try:
        await ONLINE_REACHABILITY_TASK
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    try:
        yield
    finally:
        await shutdown()


app = FastAPI(title="Chat2Me TTS", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    voice = ONLINE_VOICE if online_enabled() and ONLINE_REACHABILITY.online and ONLINE_VOICE is not None else LOCAL_VOICE
    return HealthResponse(
        ok=LOCAL_VOICE is not None,
        configured_engine=CONFIGURED_TTS_ENGINE,
        active_online=online_enabled() and ONLINE_REACHABILITY.online,
        online_status=ONLINE_REACHABILITY.status,
        local_engine=local_engine(),
        local_model=local_model(),
        sample_rate=int(getattr(getattr(voice, "config", None), "sample_rate", 0) or 0),
    )


@app.get("/tts/reachability", response_model=ReachabilityResponse)
async def reachability() -> ReachabilityResponse:
    return ReachabilityResponse(
        online=ONLINE_REACHABILITY.online,
        provider="online-tts" if online_enabled() else "local",
        model=CONFIGURED_TTS_MODEL if online_enabled() else None,
        status=ONLINE_REACHABILITY.status if online_enabled() else "online_engine_disabled",
        checked_at=ONLINE_REACHABILITY.checked_at,
    )


@app.post("/tts/speak")
async def speak(request: SpeakRequest) -> Response:
    use_online = online_enabled() and (ONLINE_REACHABILITY.online if request.online_available is None else request.online_available)

    with TTS_INFERENCE_LOCK:
        if use_online and ONLINE_VOICE is not None:
            try:
                with_tts_env("online", CONFIGURED_TTS_MODEL)
                wav = synthesize_wav(ONLINE_VOICE, request.text)
                return Response(
                    content=wav,
                    media_type="audio/wav",
                    headers={
                        "X-Chat2Me-TTS-Route": "online",
                        "X-Chat2Me-TTS-Engine": "online",
                        "X-Chat2Me-TTS-Model": CONFIGURED_TTS_MODEL,
                        "X-Chat2Me-Online-Status": ONLINE_REACHABILITY.status,
                    },
                )
            except Exception as exc:
                log(f"online TTS failed; falling back to local: {exc}")

        if LOCAL_VOICE is None:
            raise HTTPException(status_code=503, detail="local TTS is not ready")
        with_tts_env(local_engine(), local_model())
        wav = synthesize_wav(LOCAL_VOICE, request.text)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Chat2Me-TTS-Route": "local",
            "X-Chat2Me-TTS-Engine": local_engine(),
            "X-Chat2Me-TTS-Model": local_model(),
            "X-Chat2Me-TTS-Fallback": "1" if online_enabled() else "0",
            "X-Chat2Me-Online-Status": ONLINE_REACHABILITY.status,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=TTS_HOST, port=TTS_PORT)
