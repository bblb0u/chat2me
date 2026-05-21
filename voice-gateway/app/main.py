from __future__ import annotations

import html
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
PROFILE_PATH = Path(os.getenv("PROFILE_PATH", "/app/config/profile.yaml"))
SAFETY_PATH = Path(os.getenv("SAFETY_PATH", "/app/config/safety.yaml"))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    route: str
    model: str | None = None
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    model: str
    ollama: str


app = FastAPI(title="Chat2M Voice Gateway", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


def match_fixed_qa(message: str) -> str | None:
    normalized = message.strip().lower()
    for item in profile().get("fixed_qa", []):
        patterns = item.get("patterns", [])
        if any(str(pattern).lower() in normalized for pattern in patterns):
            return str(item.get("answer", "")).strip() or None
    return None


def build_prompt(message: str) -> str:
    data = profile()
    robot = data.get("robot", {})
    facts = "\n".join(f"- {fact}" for fact in data.get("fixed_facts", []))
    system_prompt = data.get("system_prompt", "")
    name = robot.get("name", "Chat2M")
    company = robot.get("company", "待定公司")
    persona = robot.get("persona", "")

    return "\n".join(
        [
            str(system_prompt).strip(),
            "",
            f"机器人名：{name}",
            f"公司：{company}",
            f"人格：{persona}",
            "",
            "固定事实：",
            facts,
            "",
            f"用户：{message}",
            "助手：",
        ]
    ).strip()


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


async def call_ollama(message: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_prompt(message),
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_predict": 256,
        },
    }
    timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
    answer = response.json().get("response", "")
    return strip_thinking(str(answer))


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_status = "unreachable"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = "ok" if response.is_success else f"http_{response.status_code}"
    except httpx.HTTPError:
        ollama_status = "unreachable"
    status = "ok" if ollama_status == "ok" else "degraded"
    return HealthResponse(status=status, model=OLLAMA_MODEL, ollama=ollama_status)


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
        answer = await call_ollama(message)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        if "model" in detail.lower() and "not found" in detail.lower():
            detail = f"模型 {OLLAMA_MODEL} 还未下载，请先执行模型初始化或 ollama pull。"
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama 服务不可用：{exc}") from exc

    if contains_blocked_keyword(answer):
        answer = blocked_response()
        route = "blocked_output"
    else:
        route = "ollama"

    return ChatResponse(
        answer=answer or "我暂时没有生成有效回答。",
        route=route,
        model=OLLAMA_MODEL,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )


@app.get("/config/preview", response_class=HTMLResponse)
async def config_preview() -> str:
    data = {
        "profile_path": str(PROFILE_PATH),
        "safety_path": str(SAFETY_PATH),
        "model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
    }
    rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in data.items())
    return f"<table>{rows}</table>"
