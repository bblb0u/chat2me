# Chat2Me

Chat2Me 是一套面向机器人本体部署的中文语音交互系统。当前目标平台是 NVIDIA Jetson / ARM64，默认外设包括 ReSpeaker USB 麦克风阵列、ALSA 或 PulseAudio 扬声器，以及通过 USB Serial/JTAG 连接的 ESP32-S3 状态屏。

系统由六个容器组成：`chat2me-speech` 负责唤醒、录音、播放和状态；`chat2me-asr` 负责语音识别；`chat2me-core` 负责编排、安全策略、固定问答和意图路由；`chat2me-llm` 负责本地 Ollama 与在线 OpenAI-compatible 模型的路由；`chat2me-tts` 负责语音合成；`chat2me-relay` 负责把语音状态同步到显示屏等外设。

## 当前能力

- 中文唤醒词监听，默认 `小江同学`、`小江`。
- 多轮语音会话，支持唤醒回应、空闲退出、口令退出和最大轮次限制。
- Sherpa ONNX SenseVoice ASR，支持同音词替换和中文专有词修正。
- 固定问答、关键词安全过滤和本地意图分类。
- 方向类问题本地处理，可回答“我在你哪边”这类请求。
- LLM 支持本地 Ollama，也支持 OpenAI-compatible 在线接口；在线不可用时回落本地模型。
- TTS 支持在线 EdgeTTS 优先、本地 MeloTTS 回落。
- ReSpeaker DOA/VAD/AGC/AEC/降噪参数控制。
- 通过 Relay 将 `idle/listening/thinking/speaking/error` 等状态同步到 ESP32-S3 屏幕。

## 架构

```text
语音链路

ReSpeaker
  -> chat2me-speech
  -> chat2me-asr
  -> chat2me-core
  -> chat2me-llm
  -> chat2me-tts
  -> chat2me-speech
  -> 扬声器

状态链路

chat2me-speech /state
  -> chat2me-relay
  -> USB Serial/JTAG
  -> ESP32-S3 显示屏

文本链路

client
  -> chat2me-core /chat
  -> chat2me-llm
```

Compose 服务：

| 服务 | 端口 | 作用 |
| --- | --- | --- |
| `chat2me-llm` | `8082` | Ollama 网关、在线 LLM 路由、本地意图模型接口 |
| `chat2me-core` | `8080` | 对话编排、安全过滤、固定问答、意图路由 |
| `chat2me-asr` | `8092` | SenseVoice ASR |
| `chat2me-tts` | `8093` | 在线/本地 TTS |
| `chat2me-speech` | `8090` | 唤醒、录音、播放、会话状态 |
| `chat2me-relay` | `8091` | 外设状态转发 |

## 运行流程

### 语音会话

1. `chat2me-speech` 打开 ReSpeaker 音频输入和 USB tuning 接口，应用 AGC、降噪、AEC、VAD、DOA 等参数。
2. 唤醒监听只取 ReSpeaker 6 通道音频里的 channel 0，即官方定义的 processed audio for ASR。
3. 命中唤醒词后播放 `WAKE_RESPONSE`，随后进入多轮会话。
4. 每轮开始时进行短时底噪校准，结合 RMS gate、硬件 VAD 和 `SPEECHDETECTED` 判断有效语音。
5. 录音片段发送到 `chat2me-asr`，ASR 返回文本后可执行同音词替换。
6. `chat2me-speech` 先处理本地退出词；普通文本送到 `chat2me-core /chat`。
7. Core 返回答案后，Speech 调用 `chat2me-tts` 合成 WAV，并暂停输入流播放，避免扬声器声音被重新识别。
8. 回答播放结束后等待 `POST_RESPONSE_DRAIN_SECONDS`，再进入下一轮收音。

当前语音开头的底噪校准窗口为 `ASR_NOISE_CALIBRATION_SECONDS=0.2`。这个值不宜过长，否则用户刚开口时的语音可能被计入“环境噪声”，导致开头几个字被门限吃掉。

### Core 编排

`chat2me-core /chat` 的处理顺序如下：

1. 读取 `runtime.env`、`profile.yaml`、`safety.yaml`。
2. 输入命中 `blocked_keywords` 时直接返回阻断回复。
3. 对用户文本做空白、标点、唤醒词和“您/你”归一化，优先匹配 `fixed_qa[*].patterns`。
4. 固定问答未命中时，Core 调用 `chat2me-llm /intent` 做本地意图分类。
5. 意图分类只允许返回 `blocked`、`fixed_qa`、`direction`、`session_end`、`chat`。
6. `blocked`、`fixed_qa`、`direction`、`session_end` 由 Core 直接执行；`chat` 才进入通用 LLM。
7. LLM 输出仍会经过一次安全关键词过滤。

意图目录来自 `profile.yaml` 的 `intent_router` 和 `fixed_qa[*].intent`。固定问答建议维护稳定的 `id`，因为意图模型返回的是 `fixed_qa_id`，Core 会用它映射到标准答案。

### LLM 路由

`chat2me-llm` 支持两种模式：

- `LLM_ENGINE=ollama` 或 `local`：直接调用本地 Ollama。
- `LLM_ENGINE=online`：后台探测在线 OpenAI-compatible 接口；在线可用时走在线模型，请求失败或不可达时回落到 `OLLAMA_MODEL`。

本地意图分类始终走 Ollama，由 `INTENT_MODEL` 指定。它不受 `LLM_ENGINE=online` 影响，这样在线链路不可用时仍能处理固定问答、方向、结束会话和安全拦截。

### ASR 和 TTS

`chat2me-asr` 接收 16-bit PCM WAV，使用 Sherpa ONNX SenseVoice 返回文本。启动时可根据 `homophones.yaml` 生成同音替换 FST，用于把常见同音误识别修正为业务词。

`chat2me-tts` 默认在线优先。`VOICE_TTS_ENGINE=online` 时服务后台探活 EdgeTTS；在线请求失败时自动回落到本地 MeloTTS。TTS 输出统一为 `audio/wav`，由 `chat2me-speech` 播放。

### 状态和显示

`chat2me-speech` 维护 `/state`，包含当前状态、显示文本、序号和方向信息。`chat2me-relay` 轮询该接口，只在序号变化时写入串口。

ESP32-S3 显示固件通过 USB Serial/JTAG 接收单行 JSON：

```json
{"state":"speaking","text":"回答内容"}
```

当前固件使用 `state` 更新 LVGL 状态动画，`text` 保留给后续显示文本、动作提示或其他外设使用。

## 快速启动

准备目录和初始配置：

```bash
mkdir -p data/config data/models data/ollama data/log
docker compose run --rm --no-deps chat2me-core true
```

按现场设备修改运行配置：

```bash
vim data/config/runtime.env
vim data/config/profile.yaml
vim data/config/safety.yaml
vim data/config/homophones.yaml
```

启动文字链路：

```bash
docker compose up -d chat2me-llm chat2me-core
```

启动完整语音链路：

```bash
docker compose up -d
```

预下载或校验语音模型：

```bash
docker compose run --rm --no-deps -e VOICE_ROLE=chat2me-asr chat2me-asr true
docker compose run --rm --no-deps -e VOICE_ROLE=chat2me-tts chat2me-tts true
```

查看日志：

```bash
tail -f data/log/chat2me-*.log
docker compose logs -f
```

停止：

```bash
docker compose down
```

## 配置文件

仓库内 `config/` 是默认模板。容器首次启动时会复制到 `data/config/`；之后运行时读取 `data/config/`。已有部署请直接改 `data/config/*`，只改 `config/*` 不会覆盖现有运行配置。

| 文件 | 作用 |
| --- | --- |
| `data/config/runtime.env` | 当前机器实际生效的环境配置 |
| `data/config/profile.yaml` | 机器人身份、固定事实、固定问答、意图目录、系统提示词 |
| `data/config/safety.yaml` | 输入/输出安全关键词和阻断回复 |
| `data/config/homophones.yaml` | ASR 同音词替换词库 |

常用 LLM 配置：

```env
LLM_ENGINE=ollama
OLLAMA_MODEL=qwen3:4b-instruct
INTENT_CLASSIFIER_ENABLED=1
INTENT_MODEL=qwen3:0.6b
```

在线 LLM 配置：

```env
LLM_ENGINE=online
LLM_ONLINE_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5-mini
LLM_API_KEY=sk-...
OLLAMA_MODEL=qwen3:4b-instruct
```

语音识别配置：

```env
VOICE_ASR_ENGINE=sensevoice
VOICE_ASR_MODEL=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09
SENSEVOICE_LANGUAGE=auto
SENSEVOICE_USE_ITN=1
ASR_HOMOPHONE_REPLACER_ENABLED=1
ASR_NOISE_GATE_ENABLED=1
ASR_NOISE_CALIBRATION_SECONDS=0.2
ASR_PREROLL_SECONDS=0.5
```

语音合成配置：

```env
VOICE_TTS_ENGINE=online
VOICE_TTS_MODEL=edge-tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
EDGE_TTS_RATE=+0%
EDGE_TTS_VOLUME=+0%
EDGE_TTS_PITCH=+0Hz
MELOTTS_DISABLE_BERT=1
```

常用中文 EdgeTTS 音色：

| 配置值 | 说明 |
| --- | --- |
| `zh-CN-XiaoxiaoNeural` | 普通话女声 |
| `zh-CN-XiaoyiNeural` | 普通话女声 |
| `zh-CN-YunjianNeural` | 普通话男声 |
| `zh-CN-YunxiNeural` | 普通话男声 |
| `zh-CN-YunxiaNeural` | 普通话男声 |
| `zh-CN-YunyangNeural` | 普通话男声 |
| `zh-CN-liaoning-XiaobeiNeural` | 辽宁方言女声 |
| `zh-CN-shaanxi-XiaoniNeural` | 陕西方言女声 |
| `zh-HK-HiuGaaiNeural` | 香港粤语女声 |
| `zh-HK-HiuMaanNeural` | 香港粤语女声 |
| `zh-HK-WanLungNeural` | 香港粤语男声 |
| `zh-TW-HsiaoChenNeural` | 台湾国语女声 |
| `zh-TW-HsiaoYuNeural` | 台湾国语女声 |
| `zh-TW-YunJheNeural` | 台湾国语男声 |

## API

### chat2me-core

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | Core 与 LLM 健康状态 |
| `GET` | `/direction` | 当前声源方向 |
| `GET` | `/llm/reachability` | 在线 LLM 可达性 |
| `POST` | `/chat` | 文本对话入口 |

### chat2me-llm

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 本地/在线模型状态 |
| `GET` | `/llm/reachability` | 在线模型探活结果 |
| `POST` | `/chat` | LLM 问答 |
| `POST` | `/intent` | 本地意图分类 |

### chat2me-asr

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | ASR 健康状态 |
| `GET` | `/asr/reachability` | ASR 可用性 |
| `POST` | `/asr/transcribe` | WAV 转文本 |

### chat2me-tts

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | TTS 健康状态 |
| `GET` | `/tts/reachability` | 在线 TTS 可达性 |
| `POST` | `/tts/speak` | 文本转 WAV |

### chat2me-speech

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | Speech 健康状态 |
| `GET` | `/state` | 当前语音状态和方向 |
| `POST` | `/wake` | 手动触发会话 |
| `POST` | `/diagnostics/turn` | 单轮诊断，可提交文本或 base64 WAV |

### chat2me-relay

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | Relay 与显示端状态 |

文本调用示例：

```bash
docker compose exec chat2me-core python - <<'PY'
import json, urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8080/chat",
    data=json.dumps({"message": "你是谁"}).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode())
PY
```

## 数据目录

| 路径 | 说明 |
| --- | --- |
| `data/config/` | 运行配置，当前机器实际读取这里 |
| `data/models/` | KWS、ASR、TTS 模型缓存，以及生成的唤醒词和同音替换文件 |
| `data/ollama/` | Ollama 模型、缓存和密钥 |
| `data/log/` | 服务日志，按 `chat2me-*.log` 写入 |

迁移机器时保留 `data/`，可以复用配置和已下载模型。`data/` 是运行时目录，不参与镜像构建。

## 仓库结构

```text
config/                 默认配置模板
firmware/display/       ESP32-S3 显示屏固件
runtime/entrypoints/    镜像入口脚本
runtime/shared/         各服务共享运行时代码
runtime/tools/          同音词 FST 生成工具
services/asr/           ASR 服务
services/core/          Core 编排服务
services/llm/           LLM 网关服务
services/relay/         外设状态转发服务
services/speech/        唤醒、录音、播放和状态服务
services/tts/           TTS 服务
.github/workflows/      ARM64 镜像构建与发布
```

关键文件：

| 文件 | 说明 |
| --- | --- |
| `docker-compose.yml` | 六个运行服务、设备挂载、数据卷和健康检查 |
| `runtime/shared/common.py` | 环境变量、日志、显示串口客户端 |
| `runtime/shared/voice.py` | 语音模型、远程 ASR/TTS 适配、音频门控、播放、Core 调用 |
| `services/core/app/main.py` | 安全过滤、固定问答、意图路由、方向查询、LLM 转发 |
| `services/llm/app/main.py` | Ollama、在线 LLM、意图模型和回落逻辑 |
| `services/speech/app/main.py` | 唤醒监听、会话循环、状态接口、诊断接口 |
| `services/speech/app/respeaker.py` | ReSpeaker tuning、DOA/VAD 读取和方向话术 |
| `services/relay/app/main.py` | 轮询 Speech 状态并写入串口外设 |

## 显示屏固件

固件位于 `firmware/display`，面向 Waveshare ESP32-S3 LCD 状态屏。

构建和烧录：

```bash
. /opt/esp-idf-v5.5.4/export.sh
cd firmware/display
idf.py build
idf.py -p /dev/ttyACM0 flash
```

容器侧默认自动查找 ESP32 USB Serial/JTAG：

```env
DISPLAY_SERIAL_PORT=auto
DISPLAY_SERIAL_CANDIDATES=/host-dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*-if00
DISPLAY_SERIAL_BAUD=115200
```

## 运行注意

- `docker-compose.yml` 默认使用 ARM64 镜像，LLM/TTS 服务启用 NVIDIA runtime。
- `chat2me-speech` 需要访问 `/dev/snd`、`/dev/bus/usb`、宿主机 ALSA 配置和用户 PulseAudio socket。
- `chat2me-relay` 挂载 `/dev` 到 `/host-dev`，用于查找 ESP32 或其他串口外设；不启动 Relay 不影响语音主链路。
- 在线 LLM/TTS 的可达性由服务后台探活维护，请求路径使用最近一次探活状态。
- `config/*` 只影响新初始化部署；已经运行过的机器请改 `data/config/*`。

## 参考

- Seeed Studio ReSpeaker USB 4-Mic Array XVF3000 v3.0：<https://wiki.seeedstudio.com/cn/respeaker_mic_array_v3.0/>
- ReSpeaker `usb_4_mic_array`：<https://github.com/respeaker/usb_4_mic_array>
- ReSpeaker MicArrayV3 firmware：<https://github.com/respeaker/usb_4_mic_array/tree/master/MicArrayV3_firmware>
- ReSpeaker `tuning.py`：<https://github.com/respeaker/usb_4_mic_array/blob/master/tuning.py>
- ReSpeaker `pixel_ring`：<https://github.com/respeaker/pixel_ring>
