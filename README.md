# Chat2M 本地小模型对话 MVP

这个仓库当前先落地项目规划里的“对话路由”最小闭环：

- `ollama` 容器运行本地小模型，默认 `qwen3:0.6b`。
- `voice-gateway` 容器提供 FastAPI 对话接口。
- `voice-agent` 容器提供 ReSpeaker 唤醒、离线 ASR、连续对话和本地 Piper TTS。
- `config/profile.yaml` 放机器人固定信息、固定问答和系统提示词。
- `config/safety.yaml` 放第一层敏感词拦截。

## 快速启动

当前机器没有安装 Docker Compose 插件时，可以直接运行：

```bash
./scripts/start-local.sh
```

Jetson 上会默认用 Docker 的 `nvidia` runtime 启动 Ollama，并设置 `JETSON_JETPACK=5` 与 `cuda_jetpack5` backend。

启动后接口：

```text
http://localhost:8080/chat
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

## 语音链路

当前语音链路：

- 唤醒监听：`voice-agent` 常驻监听“嗨小紫 / 嘿小紫 / 小紫”。
- ASR 输入：唤醒后使用 sherpa-onnx streaming ASR，把识别文本 POST 到 `/chat`。
- 连续对话：唤醒后先播放“有什么可以帮助您的”，之后最多连续 8 轮，不需要每轮重复唤醒。
- TTS 输出：Piper 本地中文 `zh_CN-huayan-medium`，合成 PCM 后直接通过 ALSA 播放。
- 待接入：CAN 状态屏，请求开始时发 `THINKING`，播放语音时发 `SPEAKING`，结束后发 `IDLE`。

## 语音唤醒

先确认 `voice-gateway` 和 Ollama 已启动：

```bash
./scripts/start-local.sh
```

启动后台唤醒监听：

```bash
./scripts/start-voice-agent.sh
docker logs -f chat2m-voice-agent
```

首次启动会自动下载 sherpa-onnx KWS/ASR 模型和 Piper 中文 TTS 模型到 `models/`。这些模型文件只保留在本地，不提交到 Git。

默认输入设备匹配 `ReSpeaker`，默认输出设备用 ALSA `default`。可以覆盖：

```bash
AUDIO_INPUT_DEVICE=ReSpeaker AUDIO_OUTPUT_DEVICE=hw:0,0 ./scripts/start-voice-agent.sh
```

Piper 语速可以用 `PIPER_LENGTH_SCALE` 调整，数值越大越慢：

```bash
PIPER_LENGTH_SCALE=1.0 ./scripts/start-voice-agent.sh
```

停止：

```bash
./scripts/stop-voice-agent.sh
```
