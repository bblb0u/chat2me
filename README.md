# Chat2M 本地小模型对话 MVP

这个仓库当前先落地项目规划里的“对话路由”最小闭环：

- `ollama` 容器运行本地小模型；也可以在 `runtime.env` 切到任意 OpenAI-compatible 在线接口。
- `chat2m-gateway` 容器提供 FastAPI 对话接口。
- `chat2m-wake` 容器负责麦克风唤醒词监听。
- `chat2m-speech` 容器负责离线 ASR、连续对话和本地 Piper TTS。
- `chat2m-status` 容器负责把状态转发到 ESP32 显示屏。
- `config/` 放默认配置模板；运行时配置会初始化到 `data/config/`。

## 快速启动

直接启动整套服务：

```bash
docker compose up -d
```

Jetson 上会默认用 Docker 的 `nvidia` runtime 启动 Ollama，并设置 `JETSON_JETPACK=5` 与 `cuda_jetpack5` backend。
`ollama` 容器启动后会在后台检查 `OLLAMA_MODEL`，可用则复用，不可用会删除后重新拉取。切到远程 provider 时仍会保留这个本地模型，供离线会话使用。

服务默认只在 Docker Compose 内部网络通信，不向宿主机暴露端口。

停止服务：

```bash
docker compose down
```

如果只想启动文字对话网关：

```bash
./scripts/start-local.sh
```

默认本地模型在 `config/runtime.env` 里配置为 Qwen3 4B Instruct 非思考版，比 1.7B 和 `qwen2.5:3b` 更强，同时不会输出 `<think>` 思考块，更适合实时 TTS 语音播报。

## 大模型配置

运行时请改 `data/config/runtime.env`，改完重启相关容器：

```bash
docker compose up -d --force-recreate ollama chat2m-gateway chat2m-speech chat2m-wake
```

本地 Ollama：

```env
LLM_PROVIDER=ollama
LLM_MODEL=
OLLAMA_MODEL=qwen3:4b-instruct
```

OpenAI-compatible 在线接口：

```env
LLM_PROVIDER=remote
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5-mini
LLM_API_KEY=sk-...
OLLAMA_MODEL=qwen3:4b-instruct
```

如果要用 DeepSeek 或自建模型，只改地址、模型名和密钥：

```env
LLM_PROVIDER=remote
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-...
OLLAMA_MODEL=qwen3:4b-instruct
```

如果在线网关要求根地址加 `/v1` 路径，例如 `https://sub2api.canghai.org/v1/chat/completions`，可以这样写：

```env
LLM_PROVIDER=remote
LLM_BASE_URL=https://sub2api.canghai.org
LLM_CHAT_COMPLETIONS_PATH=/v1/chat/completions
LLM_REACHABILITY_PATH=/v1/models
LLM_MODEL=gpt-5.5
LLM_API_KEY=sk-...
OLLAMA_MODEL=qwen3:4b-instruct
```

`LLM_PROVIDER=ollama` 或 `local` 表示本地；其他任意值都表示在线接口。代码不会内置 OpenAI、DeepSeek 或其他供应商地址，实际调用只看 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY`。

屏蔽词、固定问答和 `system_prompt` 仍然由 `chat2m-gateway` 统一处理。切换 provider 只替换最终生成答案的大模型后端；输入会先过 `safety.yaml` 和 `profile.yaml`，模型输出后也会再过一次屏蔽词检查。

远程 provider 的可达性由 `chat2m-gateway` 后台周期探测，默认每 5 秒探一次，超时 1.5 秒；`chat2m-speech` 默认每 2 秒同步一次这个结果到自己的内存缓存。语音唤醒后只读取 speech 内存缓存，不做任何网络探测：缓存在线则本轮会话固定调用在线模型，缓存离线则本轮会话固定调用本地 `OLLAMA_MODEL`。如果本轮选择在线模型但中途网络不可用，会播报“网络连接不可用”并结束本轮会话，等待下一次唤醒。

## API

### `GET /health`

检查网关和 Ollama 状态。

### `GET /direction`

读取 ReSpeaker Mic Array v3.0 的声源方向。这个接口用于后续头部自由度或其他外部控制模块；语音里问“我在你的哪边”只是调用同一份方向数据做验证。

```json
{
  "ok": true,
  "source": "respeaker",
  "raw_angle_degrees": 122,
  "angle_degrees": 122,
  "sector": "back_right",
  "label": "右后方",
  "voice_activity": false,
  "coordinate": {
    "zero": "front",
    "positive": "clockwise",
    "unit": "degrees"
  },
  "updated_at": 1779775407.9472184
}
```

`angle_degrees` 是校准后的角度：`0` 表示正前方，顺时针增加。`RESPEAKER_DOA_FRONT_OFFSET_DEGREES` 用来校准设备正前方，`RESPEAKER_DOA_CLOCKWISE` 用来修正左右方向。

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

- `fixed_qa`：命中固定问答，没有调用模型。会对 ASR 文本做基础归一化，去掉空格、标点和常见前缀后再匹配。
- `local`：调用本地 Ollama 模型。
- `online`：调用远程 OpenAI-compatible 模型。
- `blocked_input`：输入命中敏感词。
- `blocked_output`：模型输出命中敏感词。

## 语音链路

当前语音链路：

- 唤醒监听：`chat2m-wake` 默认监听“嗨小江 / 嘿小江 / 小江”，用于提升实际唤醒稳定性。
- ASR 输入：唤醒后使用 ReSpeaker Mic Array v3.0 处理后的采集音频和 sherpa-onnx streaming ASR，把识别文本 POST 到 `/chat`。
- 声源方向：`chat2m-speech` 通过 ReSpeaker 官方 USB 控制接口读取 DOA/VAD；统一接口为 `GET http://chat2m-gateway:8080/direction`，内部直连为 `GET http://chat2m-speech:8090/direction`，问“我在你的哪边”会直接读取该接口数据回答。
- 连续对话：唤醒后先播放“有什么可以帮助您的”，之后最多连续 8 轮，不需要每轮重复唤醒。
- 退出会话：说“退下吧”“你走吧”“走吧”“不用了”“再见”等会回到待机。
- TTS 输出：Piper 本地中文 `zh_CN-huayan-medium`，合成 PCM 后直接通过 ALSA 播放。
- 状态屏：Waveshare ESP32-S3-Touch-LCD-3.5 通过 USB 串口接收 `idle` / `listening` / `thinking` / `speaking` / `error` 状态。

## 语音唤醒

启动整套语音链路：

```bash
docker compose up -d
docker compose logs -f chat2m-wake chat2m-speech chat2m-status
```

默认唤醒词是“嗨小江 / 嘿小江 / 小江”。如果要更换唤醒词、音频设备、显示屏串口、Ollama 模型或 Piper 语速，改 `data/config/runtime.env`；完整配置说明见 `.env.example`。

首次启动会自动检查 sherpa-onnx KWS/ASR 模型和 Piper 中文 TTS 模型。模型关键文件可用则复用，不可用或为空会删除对应模型后重新下载。

状态屏串口默认通过宿主机 `/dev/chat2m-display` 传入容器，避免绑定 `/dev/ttyACM0` 这类会随 USB 插拔顺序漂移的端口。首次部署或更换显示屏后，在宿主机执行：

```bash
sudo scripts/install-display-udev.sh /dev/ttyACM0
```

脚本会根据当前显示屏的 USB VID/PID/serial 写入 udev 规则。当前这块 ESP32-S3 显示屏会匹配 `303a:1001` 和 serial `44:1B:F6:85:CF:34`，生成 `/dev/chat2m-display`。

## 数据目录

可迁移运行时数据统一放在 `data/`，不提交到 Git：

- `data/config/`：运行时配置，从仓库 `config/` 默认模板初始化。
- `data/models/`：唤醒词、ASR、TTS 模型。
- `data/ollama/`：Ollama 模型、manifest 和本地运行数据。

换机器时迁移 `/opt/chat2m/data` 即可。语音模型按 `runtime.env` 里的模型名放到 `data/models/` 的独立子目录；更换模型名会使用新目录，不会覆盖旧模型。`/dev`、`/dev/snd`、`/etc/asound.conf` 是宿主机设备和系统音频配置，不放进项目数据目录。

停止：

```bash
docker compose down
```
