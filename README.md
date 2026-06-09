# Chat2Me

Chat2Me 是一套部署在机器人硬件上的中文语音交互系统，默认面向 NVIDIA Jetson / ARM64 平台。典型外设包括 ReSpeaker 麦克风阵列、ALSA 扬声器和 ESP32-S3 触摸屏。

系统按职责拆分为唤醒、语音识别、对话编排、LLM、语音合成、声源方向和外设状态同步等服务，通过 Docker Compose 组合运行。LLM 支持本地 Ollama 和 OpenAI-compatible 在线服务；ASR 使用 Sherpa ONNX SenseVoice；TTS 支持本地 MeloTTS，也可以通过第三方 `edge-tts` 包调用 Microsoft Edge 在线语音服务，在线不可用时回落到本地模型。

## 硬件和平台

默认配置按这套设备准备：

- NVIDIA Jetson / ARM64，Docker 使用 NVIDIA runtime。
- ReSpeaker USB 麦克风阵列，用于唤醒、录音、声源方向和降噪参数控制。
- ALSA 音频输出设备，用于播放 TTS。
- Waveshare ESP32-S3-Touch-LCD-3.5 或同类串口外设，用于显示状态、文本和后续交互信号。

其他硬件可通过配置适配；当前部署目标仍以 Jetson 机器人硬件为主。

## 系统职责

- 通过 ReSpeaker/麦克风持续监听唤醒词，例如 `小江同学`。
- 唤醒后录制用户语音，调用 ASR 服务转成文本。
- 在 `chat2me-core` 中做安全关键词过滤、固定问答匹配、角色提示词注入。
- 通过 `chat2me-llm` 调用本地 Ollama 或在线 OpenAI-compatible LLM。
- 将回答交给 TTS 服务合成语音，并通过 ALSA 播放。
- 读取 ReSpeaker DOA 声源方向，能回答“我在你哪边”这类问题。
- 通过可选的 `chat2me-relay` 把状态、文本、动作提示等交互信号转发到屏幕、灯带、控制板或其他外设。

## 总体架构

```text
文本对话:
client -> chat2me-core -> chat2me-llm -> Ollama 或在线 LLM

语音对话:
ReSpeaker/麦克风
  -> chat2me-speech
  -> chat2me-asr
  -> chat2me-core
  -> chat2me-llm
  -> chat2me-tts
  -> chat2me-speech
  -> 扬声器

外设交互:
chat2me-relay 主动读取 chat2me-speech /state -> USB Serial/JTAG 或其他输出通道 -> ESP32-S3/灯带/控制板

声源方向:
ReSpeaker USB 控制接口 -> chat2me-speech /state.direction -> chat2me-core /direction
```

Compose 中实际运行 6 个容器：

- `chat2me-llm`：LLM 网关和本地 Ollama。
- `chat2me-core`：业务编排、固定问答、安全过滤、对外 `/chat`。
- `chat2me-relay`：可选外设交互桥，主动读取 `chat2me-speech /state`，并转发状态、文本和动作提示等信号。
- `chat2me-asr`：Sherpa ONNX SenseVoice ASR 服务。
- `chat2me-tts`：TTS 服务，支持在线优先、本地回落。
- `chat2me-speech`：麦克风监听、唤醒、会话流程、播放和状态接口。

## 技术栈

- 后端语言：Python 3.11、Python 3.8/Ubuntu 20.04 ASR/TTS/Speech 镜像运行环境。
- Web/API：FastAPI、Uvicorn、Pydantic、httpx；语音主循环和状态服务使用 `http.server`。
- 本地 LLM：Ollama，默认 `qwen3:4b-instruct`。
- 在线模型接口：OpenAI-compatible HTTP 接口。
- ASR：Sherpa ONNX 离线 SenseVoice ASR。
- TTS：本地 MeloTTS；在线模式通过第三方 `edge-tts` 包调用 Microsoft Edge 在线语音服务。
- 音频：sounddevice、ALSA `aplay`、ffmpeg、PyUSB、ReSpeaker USB tuning/DOA。
- 容器：Docker Compose、Docker Hub ARM64 镜像、GitHub Actions Buildx 发布。
- 固件：ESP-IDF 5.5.x、C/C++、LVGL、ST7796 LCD、FT6336 Touch、AXP2101 PMU。

## 快速启动

准备目录和默认配置：

```bash
mkdir -p data/config data/models data/ollama data/log
docker compose run --rm --no-deps chat2me-core true
```

然后按需修改：

```bash
vim data/config/runtime.env
vim data/config/profile.yaml
vim data/config/safety.yaml
```

启动文字对话链路：

```bash
docker compose up -d chat2me-llm chat2me-core
```

启动完整语音链路：

```bash
docker compose up -d
```

预下载/校验语音模型：

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

## 配置说明

运行配置以根目录 `config/` 为模板。容器首次启动时，entrypoint 会把默认文件复制到 `data/config/`，之后使用 `data/config/` 下的副本。

日志等级：

```env
CHAT2ME_LOG_LEVEL=info
CHAT2ME_CONSOLE_LOG_LEVEL=warning
```

支持 `error`、`warning`、`info`、`debug` 四档。默认把 `info` 及以上写入容器内目录 `/app/log`，由 compose 映射到 `data/log/chat2me-tts.log`、`data/log/chat2me-core.log` 这类服务日志；Docker 控制台显示 `warning/error`。模型下载和校验进度仍会直接输出到 `docker compose logs`，方便冷启动观察。

常用 LLM 配置：

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b-instruct
```

在线 LLM，失败回落本地 Ollama：

```env
LLM_PROVIDER=online
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5-mini
LLM_API_KEY=sk-...
OLLAMA_MODEL=qwen3:4b-instruct
```

默认本地 ASR/TTS：

```env
VOICE_ASR_ENGINE=sensevoice
VOICE_ASR_MODEL=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09
SENSEVOICE_LANGUAGE=auto
SENSEVOICE_USE_ITN=1
VOICE_TTS_ENGINE=melotts
VOICE_TTS_MODEL=MeloTTS-Chinese
VOICE_TTS_DEVICE=auto
MELOTTS_LANGUAGE=ZH
MELOTTS_SPEAKER=ZH
MELOTTS_DISABLE_BERT=1
```

Sherpa ONNX 的 KWS 和 SenseVoice ASR 使用 CPU 运行。MeloTTS 使用 PyTorch device，默认 `auto`，支持 `auto/cpu/cuda/gpu/cuda:<index>`。

在线 TTS，失败时回落本地：

```env
VOICE_TTS_ENGINE=online
VOICE_TTS_MODEL=edge-tts
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
EDGE_TTS_RATE=+0%
EDGE_TTS_VOLUME=+0%
EDGE_TTS_PITCH=+0Hz
EDGE_TTS_PROXY=
```

`edge-tts` 是第三方 Python 包；TTS 镜像依赖为 `edge-tts==7.2.8`，用于调用 Microsoft Edge 在线语音服务。

常用中文音色：

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

角色、固定问答和安全策略：

- `profile.yaml`：机器人名称、公司、人设、固定事实、固定问答、系统提示词。
- `safety.yaml`：输入/输出敏感关键词和阻断回复。
- `hotwords.yaml`：旧版 ASR 热词配置；SenseVoice ASR 当前不读取。

## 运行流程

`chat2me-speech` 是语音会话入口。它启动后会：

1. 周期拉取 LLM/TTS reachability，把在线可用性缓存在本地。
2. 打开 ReSpeaker 控制接口，按配置写入 AGC、降噪、VAD、AEC 参数。
3. 用 Sherpa ONNX KWS 模型监听唤醒词。
4. 唤醒后播放 `WAKE_RESPONSE`，进入多轮会话。
5. 录音时先做噪声门限校准，再把音频送入远程 ASR 服务。
6. 维护并暴露 `/state`，读取时会刷新当前方向；方向数据放在 `state.direction` 里。
7. 其他问题调用 `chat2me-core`，拿到回答后调用远程 TTS 服务。
8. 播放期间暂停输入流，避免扬声器声音被继续识别。

`chat2me-core` 的 `/chat` 处理顺序：

1. 加载 `runtime.env`、`profile.yaml`、`safety.yaml`。
2. 输入命中 `blocked_keywords` 时直接返回阻断回复。
3. 对用户文本做标点/空白/唤醒词归一化，优先匹配 `fixed_qa`。
4. 未命中固定问答时，把用户问题和 `system_prompt` 转发给 `chat2me-llm`。
5. 对 LLM 输出再做一次安全过滤。

`chat2me-llm` 负责在线 LLM 与本地 Ollama 的路由：

- `LLM_PROVIDER=ollama/local` 时使用本地 Ollama。
- `LLM_PROVIDER=online` 时，后台周期访问 reachability 接口。
- 在线可用时调用在线模型，不可用或请求失败时回落 `OLLAMA_MODEL`。
- 支持 OpenAI Chat Completions 风格响应，也兼容 Responses API 的 `output_text/output` 字段。

`chat2me-asr` 和 `chat2me-tts` 独立运行：

- `chat2me-asr` 启动时创建 Sherpa ONNX SenseVoice 识别实例。
- `chat2me-tts` 启动时创建本地 TTS；配置 `VOICE_TTS_ENGINE=online` 时同时创建在线实例并后台探活。
- TTS 请求优先使用调用方传入的 `online_available` 缓存值，避免每次请求阻塞在探活流程。
- 在线 TTS 请求失败会自动回落到本地 `melotts/MeloTTS-Chinese`。
- ASR 输入是 16-bit PCM WAV；TTS 输出是 `audio/wav`。

`chat2me-relay` 是语音主链路之外的外设桥。它把 `chat2me-speech /state` 暴露出来的交互状态转成外设可消费的消息，当前默认写给 ESP32-S3 屏幕固件，也可以扩展成灯光、舵机、触摸板、上位机状态同步或其他机器人交互通道。

ESP32 显示固件通过 USB Serial/JTAG 从标准输入读取一行 JSON：

```json
{"state":"speaking","text":"回答内容"}
```

当前固件解析 `state` 字段，并用 LVGL 动画展示不同颜色/节奏的状态 UI；Relay 侧保留 `text` 等字段，方便外设按需要消费更多交互信息。

## API

`chat2me-core`：

- `GET /health`
- `GET /direction`
- `GET /llm/reachability`
- `POST /chat`

`chat2me-llm`：

- `GET /health`
- `GET /llm/reachability`
- `POST /chat`

`chat2me-asr`：

- `GET /health`
- `GET /asr/reachability`
- `POST /asr/transcribe`

`chat2me-tts`：

- `GET /health`
- `GET /tts/reachability`
- `POST /tts/speak`

`chat2me-speech`：

- `GET /health`
- `GET /state`
- `POST /wake`
- `POST /diagnostics/turn`：内部诊断回合，可提交 `audio_wav_base64` 或 `text`，返回 ASR、core/LLM、TTS 的路由和耗时；用于无麦克风时验证远端镜像链路。

`chat2me-relay`：

- `GET /health`

它作为语音主链路之外的服务，轮询 `chat2me-speech /state` 并转发到外设。

示例：

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

- `data/config/`：运行时配置副本，真正生效的是这里。
- `data/models/`：KWS/ASR/TTS 模型缓存，以及生成的唤醒词、热词文件。
- `data/ollama/`：Ollama 模型、密钥和运行数据。
- `data/log/`：各服务文件日志，文件名为 `chat2me-llm.log`、`chat2me-core.log`、`chat2me-asr.log` 等。

迁移机器时保留 `data/`，即可复用配置和已下载模型。`data/` 是运行时目录，不参与镜像构建。

## 文件说明

### 根目录

| 文件 | 作用 |
| --- | --- |
| `README.md` | 项目说明文档。 |
| `docker-compose.yml` | 定义 LLM、Core、Relay、ASR、TTS、Speech 六个服务及设备/卷/健康检查；语音镜像支持本地 build。 |
| `config/runtime.env` | 默认运行配置模板。 |
| `config/profile.yaml` | 默认机器人身份、固定问答和系统提示词。 |
| `config/safety.yaml` | 默认敏感关键词与阻断回复。 |
| `config/hotwords.yaml` | 旧版 ASR 热词配置；SenseVoice ASR 当前不读取。 |
| `.env.example` | 全量运行环境变量参考。 |
| `.dockerignore` | Docker build 时忽略无关文件。 |
| `.gitignore` | Git 忽略规则。 |

### services/core

| 文件 | 作用 |
| --- | --- |
| `services/core/Dockerfile` | 构建 Core 镜像，安装 FastAPI 依赖并复制默认配置。 |
| `services/core/requirements.txt` | Core Python 依赖。 |
| `services/core/app/main.py` | Core FastAPI 应用：安全过滤、固定问答、LLM 转发、方向查询。 |

### services/llm

| 文件 | 作用 |
| --- | --- |
| `services/llm/Dockerfile` | 基于 Ubuntu 复制 Ollama 二进制，构建 LLM 网关镜像。 |
| `services/llm/entrypoint.sh` | 初始化配置、启动 Ollama、下载/校验 `OLLAMA_MODEL`。 |
| `services/llm/requirements.txt` | LLM 网关 Python 依赖。 |
| `services/llm/app/main.py` | LLM FastAPI 应用：本地/在线路由、探活、回落、响应解析。 |

### services/asr

| 文件 | 作用 |
| --- | --- |
| `services/asr/Dockerfile` | 构建 ASR 镜像，默认 `VOICE_ROLE=chat2me-asr`，安装 Sherpa ONNX SenseVoice ASR 服务和 WAV 上传接口所需依赖。 |
| `services/asr/requirements.txt` | ASR 服务基础 Python 依赖。 |
| `services/asr/app/main.py` | ASR FastAPI 服务入口：WAV 上传、本地识别和 reachability。 |

### services/tts

| 文件 | 作用 |
| --- | --- |
| `services/tts/Dockerfile` | 构建 TTS 镜像，默认 `VOICE_ROLE=chat2me-tts`，安装 TTS 服务、在线合成和本地 TTS 引擎所需依赖。 |
| `services/tts/requirements.txt` | TTS 服务基础 Python 依赖。 |
| `services/tts/app/main.py` | TTS FastAPI 服务入口：文本合成 WAV、在线/本地合成、回落和 reachability。 |

### services/speech

| 文件 | 作用 |
| --- | --- |
| `services/speech/Dockerfile` | 构建 Speech 镜像，默认 `VOICE_ROLE=chat2me-speech`，安装唤醒、麦克风输入、扬声器播放、ReSpeaker 和远程调用所需依赖。 |
| `services/speech/requirements.txt` | Speech 服务基础 Python 依赖。 |
| `services/speech/app/main.py` | Speech 服务入口：唤醒监听、会话循环、HTTP `/wake`、状态接口、诊断回合和远程 ASR/TTS 调用。 |
| `services/speech/app/respeaker.py` | ReSpeaker USB 参数读写、降噪/AGC/AEC tuning、DOA 角度和方向话术。 |

### services/relay

| 文件 | 作用 |
| --- | --- |
| `services/relay/Dockerfile` | 构建 Relay 镜像，默认 `VOICE_ROLE=chat2me-relay`，安装状态轮询和串口转发所需依赖。 |
| `services/relay/requirements.txt` | Relay 服务基础 Python 依赖。 |
| `services/relay/app/main.py` | Relay 服务入口：主动读取 `chat2me-speech /state`，状态变化时写入串口外设，后续可扩展灯光、动作提示、状态同步等输出。 |

### runtime

| 文件 | 作用 |
| --- | --- |
| `runtime/shared/common.py` | 各 Python 服务共享运行时工具：读取 `runtime.env`、环境变量解析、分级文件日志、串口显示客户端。 |
| `runtime/shared/voice.py` | ASR/TTS/Speech 共享语音逻辑：模型创建、远程 ASR/TTS 适配、音频读写、噪声门限、播放、服务探活缓存。 |
| `runtime/entrypoints/config.sh` | Core 和 Relay 共用入口：首次启动时初始化 `/app/config`。 |
| `runtime/entrypoints/audio.sh` | ASR/TTS/Speech 镜像入口：初始化配置、解析模型选择、下载/校验 KWS/ASR/TTS 模型。 |

### services/tts build helper

| 文件 | 作用 |
| --- | --- |
| `services/tts/install_melotts.sh` | TTS 镜像专用构建脚本：下载指定版本 MeloTTS 源码并应用中文本地模型所需补丁。Python/apt 依赖由 `services/tts/requirements.txt` 和 `services/tts/Dockerfile` 显式安装。 |

### firmware/display

| 文件 | 作用 |
| --- | --- |
| `firmware/display/README.md` | 显示屏固件单独构建/烧录说明。 |
| `firmware/display/CMakeLists.txt` | ESP-IDF 项目定义。 |
| `firmware/display/main/main.cpp` | 固件主程序：初始化 PMU/LCD/Touch/LVGL，读取串口 JSON 状态并更新动画。 |
| `firmware/display/main/CMakeLists.txt` | 主组件构建定义。 |
| `firmware/display/main/idf_component.yml` | ESP-IDF 组件依赖声明。 |
| `firmware/display/sdkconfig.defaults` | ESP-IDF 默认配置。 |
| `firmware/display/partitions.csv` | ESP32 分区表。 |
| `firmware/display/dependencies.lock` | ESP-IDF 组件依赖锁定文件。 |
| `firmware/display/components/esp_bsp/*` | 板级支持代码：I2C、AXP2101、电源、LCD、触摸、背光。 |
| `firmware/display/components/esp_lcd_st7796/*` | ST7796 LCD 驱动组件。 |
| `firmware/display/components/esp_lcd_touch_ft6336/*` | FT6336 触摸驱动组件。 |
| `firmware/display/components/esp_lv_port/*` | LVGL 与 ESP LCD/FreeRTOS 的移植层。 |
| `firmware/display/components/XPowersLib/*` | AXP2101 PMU 驱动库。 |

### CI 和运行时数据

| 路径 | 作用 |
| --- | --- |
| `.github/workflows/docker-publish.yml` | GitHub Actions：构建并推送 `chat2me-core`、`chat2me-llm`、`chat2me-relay`、`chat2me-asr`、`chat2me-tts`、`chat2me-speech` ARM64 镜像。 |
| `data/config/*` | 当前机器实际生效的配置。 |
| `data/models/*` | 已下载模型和运行时生成文件。 |
| `data/ollama/*` | Ollama 模型、缓存和密钥。 |
| `data/log/*` | 服务日志文件，默认按服务名写入 `chat2me-*.log`。 |

## 支持的模型

ASR：

- `sensevoice`：`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09`

TTS：

- `melotts`：`MeloTTS-Chinese`
- `online`：`edge-tts`，第三方 Python 包，调用 Microsoft Edge 在线语音服务。

LLM：

- 本地 Ollama：由 `OLLAMA_MODEL` 指定。
- 在线 OpenAI-compatible：由 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` 指定。

## 显示屏固件

固件位于 `firmware/display`，面向 Waveshare ESP32-S3-Touch-LCD-3.5。构建烧录：

```bash
. /opt/esp-idf-v5.5.4/export.sh
cd firmware/display
idf.py build
idf.py -p /dev/ttyACM0 flash
```

容器侧默认通过：

```env
DISPLAY_SERIAL_PORT=auto
DISPLAY_SERIAL_CANDIDATES=/host-dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*-if00
DISPLAY_SERIAL_BAUD=115200
```

自动查找 ESP32 USB Serial/JTAG 设备。

## 注意事项

- `docker-compose.yml` 使用 ARM64 镜像；LLM/TTS 服务默认使用 `runtime: nvidia`，ASR/KWS 使用 CPU。
- `chat2me-speech` 需要访问 `/dev/snd`、`/dev/bus/usb` 和宿主机 ALSA 配置。
- `chat2me-relay` 通过挂载 `/dev` 到 `/host-dev` 查找 ESP32 或其他串口外设；不启动它不影响唤醒、收音、ASR/TTS 和对话。
- LLM/TTS 在线模式不代表每次请求都先探测网络；服务会后台探活，请求使用最近缓存状态。
- 修改 `config/*` 会影响新初始化的配置；已有部署请改 `data/config/*`。
