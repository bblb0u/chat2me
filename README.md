# Chat2M 本地小模型对话 MVP

这个仓库当前先落地项目规划里的“对话路由”最小闭环：

- `ollama` 容器运行本地小模型，默认 `qwen3:0.6b`。
- `voice-gateway` 容器提供 FastAPI 对话接口和浏览器聊天页。
- `config/profile.yaml` 放机器人固定信息、固定问答和系统提示词。
- `config/safety.yaml` 放第一层敏感词拦截。

## 快速启动

当前机器没有安装 Docker Compose 插件时，可以直接运行：

```bash
./scripts/start-local.sh
```

启动后打开：

```text
http://localhost:8080
```

命令行测试：

```bash
curl -s http://localhost:8080/health
curl -s http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"你叫什么？"}'
```

停止服务：

```bash
./scripts/stop-local.sh
```

## 使用 Docker Compose

如果目标机器安装了 Docker Compose v2：

```bash
docker compose up -d ollama
docker compose --profile init run --rm ollama-model-init
docker compose up -d voice-gateway
```

如需换更大的模型：

```bash
OLLAMA_MODEL=qwen3:1.7b ./scripts/start-local.sh
```

如果目标网络拉 Docker Hub 很慢，可以临时替换镜像来源：

```bash
OLLAMA_IMAGE=<ollama镜像> PYTHON_IMAGE=<python镜像> ./scripts/start-local.sh
```

## API

### `GET /health`

检查网关和 Ollama 状态。

### `POST /chat`

请求：

```json
{
  "message": "介绍一下你自己"
}
```

响应：

```json
{
  "answer": "我可以先完成本地文字对话。后续会接入语音识别、语音合成和 ESP32 状态面屏。",
  "route": "fixed_qa",
  "model": null,
  "latency_ms": 1
}
```

`route` 说明：

- `fixed_qa`：命中固定问答，没有调用模型。
- `ollama`：调用本地小模型。
- `blocked_input`：输入命中敏感词。
- `blocked_output`：模型输出命中敏感词。

## 后续接入点

下一步可以在 `voice-gateway` 里继续接：

- ASR 输入：把识别文本 POST 到 `/chat`。
- TTS 输出：把 `answer` 送到本地或在线 TTS。
- CAN 状态屏：请求开始时发 `THINKING`，播放语音时发 `SPEAKING`，结束后发 `IDLE`。
