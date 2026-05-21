# ESP32 面部显示 + Jetson 语音对话规划

## 1. 目标边界

机器人由两条链路组成：

- 显示链路：ESP32-S3 只做 Figure AI 类抽象状态面屏，通过 CAN 接收 Jetson 的状态动画控制命令。
- 语音链路：Jetson AGX 做轻量语音网关，有网络时调用局域网或公网自建模型服务，无网络时只保留简单离线对话和控制指令。

这样做的原因：

- ESP32-S3 适合显示、触摸、简单状态机，不适合 ASR/LLM/TTS 推理。
- Jetson AGX 还要做运控和操作，语音服务必须轻量、可限额、可降级。
- 重模型部署在外部机器后，模型能力和硬件规模可以独立升级，不影响机器人实时控制。
- 全链路使用开源可自部署项目，避免绑定第三方闭源云服务。

## 2. 硬件与固件层

### 2.1 ESP32 面部显示屏

硬件：Waveshare ESP32-S3-Touch-LCD-3.5B / 3.5B-C。

关键信息：

- ESP32-S3R8，双核 240MHz。
- 320 x 480，3.5 寸 IPS 触摸屏。
- 8MB PSRAM，16MB Flash。
- QSPI 显示接口，AXS15231B 驱动。
- 板载 MicroSD，可选用于存放较大的动画资源。
- ESP32-S3 自带 TWAI 控制器，但 CAN 总线还需要外接 CAN 收发器，例如 SN65HVD230、TJA1051、TCAN1051。

技术栈：

- ESP-IDF：主固件框架。
- LVGL：抽象状态动画、简单图形、状态页。
- ESP-IDF TWAI driver：CAN 通信。
- 可选 MicroSD：存放较大的动画资源。

部署方式：

- ESP32 固件单独编译和烧录。
- 状态动画资源跟随固件打包，或放入 SD 卡。
- 固件内置默认状态动画，Jetson 掉线时仍可显示待机/离线状态。

依据：

- Waveshare 官方资料确认该屏支持 ESP-IDF、LVGL、QSPI 屏、MicroSD、触摸和相关 demo。
- Espressif 官方文档中 ESP32-S3 支持 TWAI，即 CAN 2.0 控制器，但工程上需要外接物理层收发器。

### 2.2 CAN 状态显示协议

CAN 只传控制命令，不传图片、音频或大文本。

推荐参数：

- CAN 2.0B。
- 500 kbps 起步，线短且干扰低可以用 1 Mbps。
- 标准帧 11-bit 即可。
- Jetson 和 ESP32 都做心跳。

推荐帧定义：

| CAN ID | 方向 | 名称 | 用途 |
|---|---|---|---|
| 0x100 | Jetson -> ESP32 | SET_ANIMATION | 设置动画 ID、强度、持续时间 |
| 0x101 | Jetson -> ESP32 | DISPLAY_STATE | 对话状态：待机、聆听、思考、惊讶、说话、成功、错误、离线 |
| 0x102 | Jetson -> ESP32 | DISPLAY_HINT | 可选，短提示类型或图标类型，不传文本 |
| 0x180 | ESP32 -> Jetson | HEARTBEAT | 屏端心跳、当前状态、FPS、错误码 |
| 0x181 | ESP32 -> Jetson | INPUT_EVENT | 触摸或按键事件，可选 |
| 0x182 | ESP32 -> Jetson | DISPLAY_ERROR | 屏端错误 |

示例数据：

```text
0x101 DISPLAY_STATE
byte0: state_id
byte1: animation_id
byte2: priority
byte3: duration_lsb
byte4: duration_msb
byte5-7: reserved

0x102 DISPLAY_HINT
byte0: hint_id
byte1: severity, 0-100
byte2: flags，例如是否闪烁、是否覆盖当前状态
byte3-7: reserved
```

显示策略：

- 屏幕不做复杂脸部表情，也不做唇形同步。
- 参考 Figure AI 类显示风格，用抽象线条、点阵、光环、波形和几何图形表达状态。
- 每个状态做 2-5 秒循环小动画，例如思考用旋转点阵，惊讶用扩散圆环，错误用红色脉冲或叉号。
- `speaking` 状态只显示说话中的通用小动画，不随音量逐帧变化，降低 CAN 频率和 ESP32 渲染复杂度。

## 3. Jetson 语音网关层

Jetson 上不直接跑重模型，主要运行轻量、可限额的服务。

### 3.1 基础技术栈

- Docker Compose：统一部署。
- Python + FastAPI/asyncio：实现 voice-gateway。
- SocketCAN + python-can：收发 CAN 帧。
- ALSA/PipeWire/PulseAudio：接入麦克风和扬声器。
- systemd：保证关键容器随系统启动。
- cgroups/Docker limits：限制语音服务 CPU/内存，避免影响运控。

部署原则：

- 运控、操作等核心进程优先级最高。
- 语音容器默认不占 GPU，除非明确允许。
- 所有在线服务都必须有超时、健康检查和离线降级路径。

### 3.2 唤醒和 VAD

推荐项目：

- openWakeWord：开源唤醒词检测，支持自定义唤醒词。
- sherpa-onnx VAD：离线 VAD，适合嵌入式和边缘设备。

部署：

- Jetson 上常驻一个 `wake-vad` 容器。
- 低采样率音频常驻监听。
- 只有检测到唤醒词或有效人声后，才启动 ASR 流程。

选择理由：

- 常驻资源占用小。
- 离线可用。
- 触发式架构能降低 Jetson 压力。

### 3.3 ASR

推荐项目：

- sherpa-onnx：优先用于轻量离线中文/中英流式 ASR。
- FunASR / SenseVoice：识别效果更好，但资源占用可能更高，可放 Jetson 或外部服务器。

部署：

- Jetson 离线模式：`sherpa-onnx` 小模型。
- 在线模式：优先调用外部服务器上的 FunASR/SenseVoice。
- voice-gateway 统一封装为 `/asr/stream` 或内部 async 接口。

选择理由：

- sherpa-onnx 是纯离线、ONNX Runtime 推理，部署简单，适合边缘设备。
- FunASR/SenseVoice 中文效果强，适合作为在线增强能力。

### 3.4 对话路由

核心服务：`voice-gateway`。

职责：

- 接收 ASR 文本。
- 发 CAN：`LISTENING`、`THINKING`、`SURPRISED`、`SPEAKING`、`ERROR`、`OFFLINE`。
- 做网络检测和服务健康检查。
- 判断走本地规则、Jetson 小模型还是外部大模型。
- 拼接人格配置和 RAG 检索结果。
- 做输入/输出过滤。
- 调用 TTS 并播放。

推荐路由策略：

```text
唤醒 -> VAD -> ASR -> 输入过滤
  -> 本地控制指令？执行并回复
  -> 网络可用且在线模型健康？调用外部 LLM
  -> 否则走离线固定问答/小模型
  -> 输出过滤 -> TTS -> 播放
  -> 同步 CAN 状态动画
```

### 3.5 离线对话

离线能力不追求复杂聊天，目标是稳定回答固定问题和执行基本指令。

推荐组成：

- YAML 固定事实和固定问答。
- 意图识别规则：关键词、正则、小型分类器。
- 可选 llama.cpp + Qwen 小模型 GGUF 量化版。

示例能力：

- “你叫什么？”
- “你是哪家公司做的？”
- “介绍一下你自己。”
- “现在能联网吗？”
- “回到待机。”
- “开始/停止某个演示。”

选择理由：

- 离线规则稳定可控。
- 避免 Jetson 被本地大模型长期占用。
- 关键介绍和公司信息不依赖模型生成，降低乱答风险。

### 3.6 TTS

推荐项目：

- Piper：轻量本地 TTS，资源占用低。
- MeloTTS：多语言能力较好。
- CosyVoice：中文质量高，适合部署在外部 GPU 机器。

部署策略：

- Jetson 离线：固定语音包 + Piper/MeloTTS 小模型。
- 在线：外部服务器 CosyVoice，返回音频流给 Jetson 播放。

屏幕同步：

- Jetson 播放 TTS 前发送 `DISPLAY_STATE=SPEAKING`。
- ESP32 播放通用 speaking 小动画，不需要逐帧嘴型数据。
- TTS 结束后发送 `IDLE` 或 `LISTENING`。
- 这样能减少 CAN 总线占用，也减少 ESP32 固件复杂度。

## 4. 在线重模型服务层

### 4.1 LLM 服务

推荐项目：

- vLLM：高吞吐推理服务，支持 OpenAI-compatible API。
- SGLang：也适合自建高性能推理服务。
- llama.cpp server：适合 CPU 或轻量 GPU 服务器，支持 GGUF 量化模型。

部署：

- 独立 GPU 服务器上用 Docker 启动。
- 暴露 OpenAI 兼容接口。
- LAN 优先，公网必须通过 WireGuard/VPN 或 HTTPS + API key。

推荐模型：

- Qwen3 / Qwen3.6 系列：优先，Apache 2.0 开放权重，中文能力强。
- DeepSeek-R1-Distill-Qwen：需要复杂推理时作为可选模型。

选择理由：

- Qwen 系列中文能力和工具调用生态成熟。
- Apache 2.0 对商业和私有部署更友好。
- OpenAI-compatible API 便于后续替换模型或服务框架。

### 4.2 在线 ASR/TTS

推荐组合：

- FunASR/SenseVoice：在线高质量中文 ASR。
- CosyVoice：在线高质量中文 TTS。

部署：

- 与 LLM 同一台或另一台 GPU 服务器。
- 独立容器，分别暴露 ASR/TTS API。
- Jetson 的 voice-gateway 通过统一配置决定是否调用。

## 5. 知识植入和安全过滤

### 5.1 人格和固定信息

先不要微调模型，先做配置化注入。

示例配置：

```yaml
robot:
  name: "待定机器人名"
  company: "待定公司名"
  persona: "你是公司的接待机器人，回答简短、准确、礼貌。"

fixed_facts:
  - "机器人由待定公司研发。"
  - "机器人支持语音交互、状态面屏显示和基础任务控制。"

allowed_topics:
  - "公司公开介绍"
  - "机器人功能"
  - "展厅导览"

blocked_topics:
  - "未公开价格"
  - "客户名单"
  - "内部源代码"
  - "未发布硬件方案"
```

实现：

- voice-gateway 每次请求模型前读取配置。
- 固定问答优先匹配，不交给 LLM 自由发挥。
- 公司资料多后再接 RAG。

选择理由：

- 可控、可审计、上线快。
- 修改机器人名、公司名、禁答内容不需要重新训练。
- 能避免微调数据不足导致的错误记忆。

### 5.2 RAG 知识库

推荐项目：

- Qdrant：开源向量数据库，Docker 部署简单。
- Qwen3-Embedding 或 BGE-M3：中文/多语言 embedding。

部署：

- Jetson 上可以放小规模 Qdrant。
- 资料较多时放外部服务器。
- 文档入库时加权限标签，例如 `public`、`internal`、`confidential`。

检索策略：

- 默认只检索 `public` 文档。
- 机器人对外交流不使用 `internal/confidential`。
- 检索结果进入 prompt 前先做敏感检查。

### 5.3 敏感信息屏蔽

推荐项目：

- Microsoft Presidio：PII 检测和脱敏。
- LLM Guard：LLM 输入/输出扫描。
- 自定义敏感词和正则：用于公司内部特定词。

三层过滤：

- 输入过滤：ASR 文本先检查敏感请求、注入攻击、违法或越权内容。
- 检索过滤：RAG 只返回允许公开的文档片段。
- 输出过滤：模型回答播放前再次检查，命中则替换为固定拒答。

选择理由：

- 提示词不能作为唯一安全边界。
- 输入、检索、输出三层都可记录日志，便于追责和调试。

## 6. Docker 部署规划

### 6.1 Jetson Compose 服务

建议服务：

```text
voice-gateway     Python/FastAPI，主控流程和路由
can-bridge        SocketCAN/python-can，也可合并进 voice-gateway
wake-vad          openWakeWord 或 sherpa-onnx VAD
local-asr         sherpa-onnx
local-tts         Piper 或 MeloTTS，可选
qdrant            小规模知识库，可选
safety-filter     Presidio/LLM Guard，可合并进 gateway
```

资源限制：

- `voice-gateway`：CPU 1-2 核，内存 512MB-1GB。
- `wake-vad`：CPU 1 核以内。
- `local-asr`：按模型限制 CPU/内存，不默认占 GPU。
- `local-tts`：离线只用于短句或固定话术。
- `qdrant`：如果 Jetson 资源紧张，迁到外部服务器。

### 6.2 外部服务器 Compose 服务

建议服务：

```text
llm-server        vLLM 或 SGLang
online-asr        FunASR/SenseVoice
online-tts        CosyVoice
embedding         Qwen3-Embedding/BGE-M3
qdrant            资料较多时部署在这里
api-gateway       Nginx/Caddy，TLS/API Key/限流
wireguard         公网访问时使用
```

## 7. 实施 Todo List

### 阶段 0：硬件确认

- [ ] 确认 ESP32 屏幕具体型号：3.5B 或 3.5B-C。
- [ ] 确认是否需要摄像头；如果只做脸部显示，暂不使用 OV5640。
- [ ] 选型 CAN 收发器：SN65HVD230、TJA1051 或 TCAN1051。
- [ ] 确认 Jetson 上 CAN 接口来源：原生 CAN、USB-CAN、MCP2515 扩展板。
- [ ] 确认机器人麦克风和扬声器方案：USB 声卡、阵列麦、I2S 音频板。

### 阶段 1：ESP32 面屏最小闭环

- [ ] 搭建 ESP-IDF 工程。
- [ ] 跑通 Waveshare LCD/LVGL demo。
- [ ] 制作基础状态动画：idle、listening、thinking、surprised、speaking、success、error、offline。
- [ ] 实现状态动画机。
- [ ] 接入 TWAI/CAN 接收任务。
- [ ] 实现 `DISPLAY_STATE` 和 `SET_ANIMATION`。
- [ ] 实现 `DISPLAY_HINT`，用于错误、惊讶、低电量等临时提示。
- [ ] 实现 `HEARTBEAT` 上报。
- [ ] 做 Jetson 掉线超时逻辑，自动进入 offline/idle。

### 阶段 2：Jetson CAN 和状态联调

- [ ] 配置 SocketCAN。
- [ ] 用 `can-utils` 跑通 `candump` / `cansend`。
- [ ] 写 `can-bridge` Python 模块。
- [ ] 封装显示 API：`set_display_state()`、`set_animation()`、`set_display_hint()`。
- [ ] 建立状态映射表：语音流程状态 -> CAN 帧。
- [ ] 连续运行 24 小时，检查丢帧、重连、心跳错误。

### 阶段 3：语音基础闭环

- [ ] Jetson 接入麦克风和扬声器。
- [ ] 实现音频采集和播放。
- [ ] 接入 openWakeWord 或 sherpa-onnx VAD。
- [ ] 接入本地 ASR：优先 sherpa-onnx。
- [ ] 实现固定问答和本地指令。
- [ ] 接入本地 TTS 或固定语音包。
- [ ] TTS 播放前后通过 CAN 切换 speaking/idle/listening 状态。
- [ ] 完成离线闭环：唤醒 -> 识别 -> 固定回答 -> 播放 -> 状态动画同步。

### 阶段 4：在线大模型闭环

- [ ] 外部服务器部署 vLLM 或 SGLang。
- [ ] 选择第一版在线模型：建议 Qwen3/Qwen3.6 Instruct。
- [ ] 暴露 OpenAI-compatible API。
- [ ] Jetson 配置在线模型地址、API key、超时。
- [ ] voice-gateway 实现在线/离线路由。
- [ ] 在线失败时自动降级到本地固定问答。
- [ ] 接入在线 ASR/TTS，可选 FunASR + CosyVoice。

### 阶段 5：知识和安全

- [ ] 编写 `profile.yaml`，填入机器人名、公司名、固定事实。
- [ ] 编写敏感词和禁答规则。
- [ ] 接入输入过滤。
- [ ] 接入输出过滤。
- [ ] 部署 Qdrant。
- [ ] 选择 embedding 模型：Qwen3-Embedding 或 BGE-M3。
- [ ] 导入公开资料、FAQ、产品介绍。
- [ ] 实现按权限标签检索。
- [ ] 做 50-100 条问答评测集，验证不会泄露内部信息。

### 阶段 6：工程化和稳定性

- [ ] 写 Docker Compose for Jetson。
- [ ] 写 Docker Compose for 外部服务器。
- [ ] systemd 设置开机自启动。
- [ ] 加健康检查：ASR、LLM、TTS、CAN、音频设备。
- [ ] 加日志：ASR 文本、路由结果、模型耗时、安全过滤命中、CAN 状态。
- [ ] 加资源限制，确保语音服务不会抢占运控资源。
- [ ] 做断网、服务器宕机、麦克风断开、CAN 断开测试。
- [ ] 做展会/现场环境噪声测试。
- [ ] 固化版本：模型版本、镜像版本、ESP32 固件版本。

## 8. 第一版 MVP 建议

第一版不要一次做太复杂。建议 MVP 只做：

1. ESP32 显示 6-8 个抽象状态动画，并能通过 CAN 切换。
2. Jetson 用 SocketCAN 控制状态动画。
3. 离线固定问答能回答机器人名、公司名、功能介绍。
4. 在线调用外部 Qwen 模型回答开放问题。
5. TTS 播放时屏幕显示 speaking 动画，播放结束自动回到待机或聆听状态。
6. 输入/输出各做一层自定义敏感词过滤。

MVP 跑通后，再逐步接 RAG、CosyVoice、高质量 ASR 和更细的安全策略。

## 9. 参考来源

- Waveshare ESP32-S3-Touch-LCD-3.5B Wiki: https://www.waveshare.com/wiki/ESP32-S3-Touch-LCD-3.5B
- Espressif ESP32-S3 TWAI/CAN 文档: https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-reference/peripherals/twai.html
- Linux SocketCAN can-utils: https://github.com/linux-can/can-utils
- python-can 文档: https://python-can.readthedocs.io/
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- openWakeWord: https://github.com/dscripka/openWakeWord
- FunASR: https://github.com/modelscope/FunASR
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- Piper: https://github.com/rhasspy/piper
- vLLM OpenAI Compatible Server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- SGLang: https://github.com/sgl-project/sglang
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Qwen: https://github.com/QwenLM
- Qdrant: https://github.com/qdrant/qdrant
- Presidio: https://github.com/microsoft/presidio
- LLM Guard: https://github.com/protectai/llm-guard
