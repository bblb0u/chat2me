from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.common import log


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


def env_value(key: str, *, allow_empty: bool = False, default: str | None = None) -> str:
    value = os.getenv(key)
    if value is None:
        if default is not None:
            value = default
        else:
            raise RuntimeError(f"{key} must be set in runtime.env")
    value = value.strip()
    if not allow_empty and not value:
        raise RuntimeError(f"{key} must not be empty in runtime.env")
    return value


def env_float(key: str, default: str | None = None) -> float:
    value = env_value(key, default=default)
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(f"{key} must be a number in runtime.env") from None


def env_csv(key: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in env_value(key).split(",") if item.strip())


CORE_LLM_URL = "http://chat2me-llm:8082/chat"
CORE_LLM_REACHABILITY_URL = "http://chat2me-llm:8082/llm/reachability"
CORE_LLM_HEALTH_URL = "http://chat2me-llm:8082/health"
CORE_LLM_TIMEOUT_SECONDS = 180.0
CORE_LLM_REACHABILITY_TIMEOUT_SECONDS = 2.0
EMPTY_ANSWER_RESPONSE = env_value("EMPTY_ANSWER_RESPONSE")
SPEECH_STATE_URL = "http://chat2me-speech:8090/state"
WAKE_WORDS = env_csv("WAKE_WORDS")
PROFILE_PATH = Path(os.getenv("PROFILE_PATH", "/app/config/profile.yaml"))
SAFETY_PATH = Path(os.getenv("SAFETY_PATH", "/app/config/safety.yaml"))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    route: str
    model: str | None = None
    latency_ms: int
    fallback: bool = False
    online_status: str | None = None


class HealthResponse(BaseModel):
    status: str
    llm: str


class ReachabilityResponse(BaseModel):
    online: bool
    provider: str
    model: str | None
    status: str
    checked_at: float | None = None


class DirectionResponse(BaseModel):
    ok: bool
    source: str = "speech"
    raw_angle_degrees: int | None = None
    angle_degrees: int | None = None
    sector: str | None = None
    label: str | None = None
    voice_activity: bool | None = None
    coordinate: dict[str, Any] | None = None
    updated_at: float
    error: str | None = None


app = FastAPI(title="Chat2Me Core", version="0.1.0")


@app.on_event("startup")
async def startup() -> None:
    log(f"core service ready: llm_url={CORE_LLM_URL}")


MATCH_REMOVE_PATTERN = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()【】\[\]{}<>《》]")


def build_match_prefix_pattern() -> re.Pattern[str]:
    prefixes = ("请问", "那个", "嗯", "啊", "你好", "您好", *WAKE_WORDS)
    escaped = "|".join(re.escape(prefix) for prefix in sorted(set(prefixes), key=len, reverse=True) if prefix)
    return re.compile(rf"^({escaped})+") if escaped else re.compile(r"a^")


MATCH_PREFIX_PATTERN = build_match_prefix_pattern()


def load_yaml(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        return default
    return data


def profile() -> dict[str, Any]:
    return load_yaml(PROFILE_PATH, {})


def safety() -> dict[str, Any]:
    return load_yaml(SAFETY_PATH, {"blocked_keywords": [], "blocked_response": "这个问题暂时不能回答。"})


def contains_blocked_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(str(keyword).lower() in lowered for keyword in safety().get("blocked_keywords", []))


def blocked_response() -> str:
    return str(safety().get("blocked_response", "这个问题暂时不能回答。"))


def normalize_for_match(text: str) -> str:
    normalized = text.strip().lower().replace("您", "你")
    normalized = MATCH_REMOVE_PATTERN.sub("", normalized)
    normalized = MATCH_PREFIX_PATTERN.sub("", normalized)
    return normalized


def match_fixed_qa(message: str) -> str | None:
    normalized = normalize_for_match(message)
    for item in profile().get("fixed_qa", []):
        patterns = item.get("patterns", [])
        normalized_patterns = [normalize_for_match(str(pattern)) for pattern in patterns]
        if any(
            pattern and (pattern in normalized or (len(normalized) >= 2 and normalized in pattern))
            for pattern in normalized_patterns
        ):
            return str(item.get("answer", "")).strip() or None
    return None


async def call_llm_service(request: ChatRequest) -> ChatResponse:
    system_prompt = str(profile().get("system_prompt", "")).strip()
    payload: dict[str, Any] = {
        "message": request.message.strip(),
        "system_prompt": system_prompt,
    }
    timeout = httpx.Timeout(connect=5.0, read=CORE_LLM_TIMEOUT_SECONDS, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(CORE_LLM_URL, json=payload)
        response.raise_for_status()
    return ChatResponse(**response.json())


@app.get("/llm/reachability", response_model=ReachabilityResponse)
async def llm_reachability() -> ReachabilityResponse:
    try:
        async with httpx.AsyncClient(timeout=CORE_LLM_REACHABILITY_TIMEOUT_SECONDS) as client:
            response = await client.get(CORE_LLM_REACHABILITY_URL)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError:
        return ReachabilityResponse(online=False, provider="unknown", model=None, status="unreachable")
    return ReachabilityResponse(**data)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(CORE_LLM_HEALTH_URL)
            llm_status = "ok" if response.is_success else f"http_{response.status_code}"
    except httpx.HTTPError:
        llm_status = "unreachable"
    return HealthResponse(status="ok" if llm_status == "ok" else "degraded", llm=llm_status)


@app.get("/direction", response_model=DirectionResponse)
async def direction() -> DirectionResponse:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(SPEECH_STATE_URL)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        return DirectionResponse(
            ok=False,
            source="speech",
            error=f"speech_direction_unavailable:{exc.__class__.__name__}",
            updated_at=time.time(),
        )
    except ValueError:
        return DirectionResponse(
            ok=False,
            source="speech",
            error="speech_direction_invalid_json",
            updated_at=time.time(),
        )

    if not isinstance(data, dict):
        return DirectionResponse(
            ok=False,
            source="speech",
            error="speech_direction_invalid_payload",
            updated_at=time.time(),
        )
    direction_data = data.get("direction")
    if not isinstance(direction_data, dict):
        return DirectionResponse(
            ok=False,
            source="speech",
            error="speech_state_missing_direction",
            updated_at=time.time(),
        )
    return DirectionResponse(**direction_data)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    started_at = time.perf_counter()
    message = request.message.strip()

    if contains_blocked_keyword(message):
        return ChatResponse(
            answer=blocked_response(),
            route="blocked_input",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )

    fixed_answer = match_fixed_qa(message)
    if fixed_answer:
        return ChatResponse(
            answer=fixed_answer,
            route="fixed_qa",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )

    try:
        result = await call_llm_service(request)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:800]
        log(f"LLM service returned HTTP {exc.response.status_code}: {detail}", level="warning")
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.ReadTimeout as exc:
        log("LLM service timed out", level="warning")
        raise HTTPException(status_code=504, detail="大模型生成超时。") from exc
    except httpx.HTTPError as exc:
        log(f"LLM service unavailable: {exc}", level="warning")
        raise HTTPException(status_code=502, detail=f"LLM 服务不可用：{exc}") from exc

    if contains_blocked_keyword(result.answer):
        answer = blocked_response()
        route = "blocked_output"
    else:
        answer = result.answer
        route = result.route

    return ChatResponse(
        answer=answer or EMPTY_ANSWER_RESPONSE,
        route=route,
        model=result.model,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        fallback=result.fallback,
        online_status=result.online_status,
    )
