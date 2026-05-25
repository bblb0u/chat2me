from __future__ import annotations

import os
import re
import time
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def load_runtime_env() -> None:
    path = Path(os.getenv("RUNTIME_CONFIG_PATH", "/app/config/runtime.env"))
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.fullmatch(r"[A-Za-z0-9_]+", key) or key in os.environ:
            continue
        os.environ[key] = value.strip()


load_runtime_env()

DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"
FINAL_ANSWER_PROMPT = "只输出给用户听的最终答案，不要分析题目，不要复述用户问题，不要解释你的输出规则。"
OPENAI_COMPATIBLE_PROVIDERS = {"openai", "deepseek", "openai_compatible"}


def env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def normalize_provider(value: str) -> str:
    provider = value.strip().lower().replace("-", "_")
    aliases = {
        "": "ollama",
        "local": "ollama",
        "openai_compatible": "openai_compatible",
        "compatible": "openai_compatible",
        "custom": "openai_compatible",
    }
    return aliases.get(provider, provider)


LLM_PROVIDER = normalize_provider(os.getenv("LLM_PROVIDER", "ollama"))
OLLAMA_BASE_URL = env_first("OLLAMA_BASE_URL", default="http://ollama:11434").rstrip("/")
OLLAMA_MODEL = env_first("OLLAMA_MODEL", default=DEFAULT_OLLAMA_MODEL)
LLM_MODEL = (
    env_first("LLM_MODEL", "OLLAMA_MODEL", default=OLLAMA_MODEL)
    if LLM_PROVIDER == "ollama"
    else env_first("LLM_MODEL")
)
LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.2)
LLM_TOP_P = env_float("LLM_TOP_P", 0.9)
LLM_MAX_TOKENS = env_int("LLM_MAX_TOKENS", 128)
LLM_CONNECT_TIMEOUT_SECONDS = env_float("LLM_CONNECT_TIMEOUT_SECONDS", 5.0)
LLM_TIMEOUT_SECONDS = env_float("LLM_TIMEOUT_SECONDS", 180.0)
LLM_REACHABILITY_INTERVAL_SECONDS = env_float("LLM_REACHABILITY_INTERVAL_SECONDS", 5.0)
LLM_REACHABILITY_TIMEOUT_SECONDS = env_float("LLM_REACHABILITY_TIMEOUT_SECONDS", 1.5)
OLLAMA_NUM_CTX = env_int("OLLAMA_NUM_CTX", 2048)
OLLAMA_NUM_THREAD = env_int("OLLAMA_NUM_THREAD", 8)
PROFILE_PATH = Path(os.getenv("PROFILE_PATH", "/app/config/profile.yaml"))
SAFETY_PATH = Path(os.getenv("SAFETY_PATH", "/app/config/safety.yaml"))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    stream: bool = False
    llm_route: str | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    model: str | None = None
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str | None
    llm: str
    ollama: str | None = None


class ReachabilityResponse(BaseModel):
    online: bool
    provider: str
    model: str | None
    status: str
    checked_at: float | None = None


app = FastAPI(title="Chat2M Voice Gateway", version="0.1.0")


MATCH_REMOVE_PATTERN = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()【】\[\]{}<>《》]")
MATCH_PREFIX_PATTERN = re.compile(r"^(请问|那个|嗯|啊|你好|您好|小江|嗨小江|嘿小江)+")


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


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


class LLMConfigError(Exception):
    pass


@dataclass(frozen=True)
class LLMResult:
    answer: str
    route: str
    model: str | None


@dataclass(frozen=True)
class LLMReachability:
    online: bool
    status: str
    checked_at: float | None = None


REMOTE_REACHABILITY = LLMReachability(online=False, status="not_checked")
REMOTE_REACHABILITY_TASK: asyncio.Task[None] | None = None


def chat_messages(message: str) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        item
        for item in (
            str(profile().get("system_prompt", "")).strip(),
            FINAL_ANSWER_PROMPT,
        )
        if item
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]


def remote_base_url() -> str:
    if LLM_PROVIDER == "openai":
        base_url = env_first("LLM_BASE_URL", "LLM_API_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1")
    elif LLM_PROVIDER == "deepseek":
        base_url = env_first("LLM_BASE_URL", "LLM_API_BASE_URL", "DEEPSEEK_BASE_URL", default="https://api.deepseek.com")
    else:
        base_url = env_first("LLM_BASE_URL", "LLM_API_BASE_URL", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL")

    base_url = base_url.rstrip("/")
    suffix = "/chat/completions"
    if base_url.endswith(suffix):
        base_url = base_url[: -len(suffix)]
    return base_url


def remote_api_key() -> str:
    return env_first("LLM_API_KEY")


def remote_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = remote_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def require_remote_config() -> str:
    if not LLM_MODEL:
        raise LLMConfigError("远程大模型未配置 LLM_MODEL。")
    base_url = remote_base_url()
    if not base_url:
        raise LLMConfigError("远程大模型未配置 LLM_BASE_URL。")
    if LLM_PROVIDER in {"openai", "deepseek"} and not remote_api_key():
        raise LLMConfigError("远程大模型未配置 LLM_API_KEY。")
    return base_url


def extract_chat_completion_answer(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message_data = choices[0].get("message") or {}
    content = message_data.get("content") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def short_detail(text: str, limit: int = 800) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


async def call_ollama(message: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": chat_messages(message),
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "num_predict": LLM_MAX_TOKENS,
            "num_thread": OLLAMA_NUM_THREAD,
        },
    }
    timeout = httpx.Timeout(connect=LLM_CONNECT_TIMEOUT_SECONDS, read=LLM_TIMEOUT_SECONDS, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
    data = response.json()
    message_data = data.get("message") or {}
    answer = message_data.get("content") or data.get("response") or ""
    return strip_thinking(str(answer))


async def call_openai_compatible(message: str) -> str:
    base_url = require_remote_config()
    payload = {
        "model": LLM_MODEL,
        "messages": chat_messages(message),
        "stream": False,
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
    }
    if LLM_PROVIDER == "openai":
        payload["max_completion_tokens"] = LLM_MAX_TOKENS
    else:
        payload["max_tokens"] = LLM_MAX_TOKENS
    timeout = httpx.Timeout(connect=LLM_CONNECT_TIMEOUT_SECONDS, read=LLM_TIMEOUT_SECONDS, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url}/chat/completions", headers=remote_headers(), json=payload)
        response.raise_for_status()
    return strip_thinking(extract_chat_completion_answer(response.json()))


def normalize_llm_route(route: str | None) -> str:
    value = (route or "auto").strip().lower().replace("-", "_")
    if value in {"", "auto"}:
        return "auto"
    if value in {"local", "ollama"}:
        return "local"
    if value in {"online", "remote"}:
        return "online"
    raise LLMConfigError(f"不支持的 llm_route：{route}")


async def call_llm(message: str, route: str | None = None) -> LLMResult:
    requested_route = normalize_llm_route(route)
    if requested_route == "local":
        answer = await call_ollama(message, OLLAMA_MODEL)
        return LLMResult(answer=answer, route="local", model=OLLAMA_MODEL)

    if requested_route == "online":
        if LLM_PROVIDER not in OPENAI_COMPATIBLE_PROVIDERS:
            raise LLMConfigError("当前未配置在线大模型 provider。")
        answer = await call_openai_compatible(message)
        return LLMResult(answer=answer, route="online", model=LLM_MODEL)

    if LLM_PROVIDER == "ollama":
        answer = await call_ollama(message, LLM_MODEL)
        return LLMResult(answer=answer, route="local", model=LLM_MODEL)
    if LLM_PROVIDER in OPENAI_COMPATIBLE_PROVIDERS:
        answer = await call_openai_compatible(message)
        return LLMResult(answer=answer, route="online", model=LLM_MODEL)
    raise LLMConfigError(f"不支持的 LLM_PROVIDER：{LLM_PROVIDER}")


async def check_ollama_health() -> str:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return "ok" if response.is_success else f"http_{response.status_code}"
    except httpx.HTTPError:
        return "unreachable"


async def probe_openai_compatible_health() -> LLMReachability:
    try:
        base_url = require_remote_config()
    except LLMConfigError as exc:
        return LLMReachability(online=False, status=f"config_error:{exc}", checked_at=time.time())

    try:
        async with httpx.AsyncClient(timeout=LLM_REACHABILITY_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{base_url}/models", headers=remote_headers())
            status = "ok" if response.is_success else f"http_{response.status_code}"
            return LLMReachability(online=response.is_success, status=status, checked_at=time.time())
    except httpx.HTTPError:
        return LLMReachability(online=False, status="unreachable", checked_at=time.time())


async def remote_reachability_loop() -> None:
    global REMOTE_REACHABILITY

    interval = max(0.5, LLM_REACHABILITY_INTERVAL_SECONDS)
    while True:
        REMOTE_REACHABILITY = await probe_openai_compatible_health()
        await asyncio.sleep(interval)


@app.on_event("startup")
async def start_remote_reachability_loop() -> None:
    global REMOTE_REACHABILITY_TASK
    if LLM_PROVIDER in OPENAI_COMPATIBLE_PROVIDERS:
        global REMOTE_REACHABILITY
        REMOTE_REACHABILITY = await probe_openai_compatible_health()
        REMOTE_REACHABILITY_TASK = asyncio.create_task(remote_reachability_loop())


@app.on_event("shutdown")
async def stop_remote_reachability_loop() -> None:
    if REMOTE_REACHABILITY_TASK is None:
        return
    REMOTE_REACHABILITY_TASK.cancel()
    try:
        await REMOTE_REACHABILITY_TASK
    except asyncio.CancelledError:
        pass


@app.get("/llm/reachability", response_model=ReachabilityResponse)
async def llm_reachability() -> ReachabilityResponse:
    if LLM_PROVIDER not in OPENAI_COMPATIBLE_PROVIDERS:
        return ReachabilityResponse(online=False, provider=LLM_PROVIDER, model=LLM_MODEL or None, status="online_provider_disabled")

    reachability = REMOTE_REACHABILITY
    return ReachabilityResponse(
        online=reachability.online,
        provider=LLM_PROVIDER,
        model=LLM_MODEL or None,
        status=reachability.status,
        checked_at=reachability.checked_at,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if LLM_PROVIDER == "ollama":
        llm_status = await check_ollama_health()
        status = "ok" if llm_status == "ok" else "degraded"
        return HealthResponse(status=status, provider=LLM_PROVIDER, model=LLM_MODEL, llm=llm_status, ollama=llm_status)

    if LLM_PROVIDER in OPENAI_COMPATIBLE_PROVIDERS:
        llm_reachability = await probe_openai_compatible_health()
        ollama_status = await check_ollama_health()
        status = "ok" if llm_reachability.online or ollama_status == "ok" else "degraded"
        return HealthResponse(
            status=status,
            provider=LLM_PROVIDER,
            model=LLM_MODEL or None,
            llm=llm_reachability.status,
            ollama=ollama_status,
        )

    return HealthResponse(
        status="degraded",
        provider=LLM_PROVIDER,
        model=LLM_MODEL or None,
        llm="unsupported_provider",
        ollama="skipped",
    )


def request_route(request: ChatRequest) -> str:
    try:
        return normalize_llm_route(request.llm_route)
    except LLMConfigError:
        return "auto"


def model_not_found_detail(request: ChatRequest, detail: str) -> str:
    if "model" not in detail.lower() or "not found" not in detail.lower():
        return detail

    route = request_route(request)
    if route == "local" or (route == "auto" and LLM_PROVIDER == "ollama"):
        return f"模型 {OLLAMA_MODEL} 还未下载完成，Ollama 容器会在后台自动拉取，请稍后重试。"
    return detail


def request_timeout_detail(request: ChatRequest) -> str:
    route = request_route(request)
    if route == "local" or (route == "auto" and LLM_PROVIDER == "ollama"):
        return "Ollama 生成超时，建议先用固定问答或更短问题测试。"
    return f"{LLM_PROVIDER} 生成超时，建议先用固定问答或更短问题测试。"


def request_service_unavailable_detail(request: ChatRequest, exc: httpx.HTTPError) -> str:
    detail = str(exc) or exc.__class__.__name__
    route = request_route(request)
    if route == "local" or (route == "auto" and LLM_PROVIDER == "ollama"):
        return f"Ollama 服务不可用：{detail}"
    return f"{LLM_PROVIDER} 服务不可用：{detail}"


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
        result = await call_llm(message, request.llm_route)
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        detail = short_detail(exc.response.text)
        raise HTTPException(status_code=502, detail=model_not_found_detail(request, detail)) from exc
    except httpx.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail=request_timeout_detail(request)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=request_service_unavailable_detail(request, exc)) from exc

    if contains_blocked_keyword(result.answer):
        answer = blocked_response()
        route = "blocked_output"
    else:
        answer = result.answer
        route = result.route

    return ChatResponse(
        answer=answer or "我暂时没有生成有效回答。",
        route=route,
        model=result.model,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )
