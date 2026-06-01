FROM ubuntu:20.04

ARG JETSON_L4T_PLATFORM=t234
ARG JETSON_TORCH_WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
ARG COSYVOICE_GIT_REF=v2.0
ARG VOICE_AGENT_FEATURES="sensevoice sherpa piper melotts f5-tts cosyvoice online"
ARG CHAT2ME_DOWNLOAD_RETRIES=10
ARG CHAT2ME_GIT_RETRIES=5
ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHAT2ME_DOWNLOAD_RETRIES=${CHAT2ME_DOWNLOAD_RETRIES} \
    CHAT2ME_GIT_RETRIES=${CHAT2ME_GIT_RETRIES} \
    CHAT2ME_PIP_RETRIES=${CHAT2ME_PIP_RETRIES} \
    CHAT2ME_PIP_TIMEOUT=${CHAT2ME_PIP_TIMEOUT}

WORKDIR /app

COPY infra/image-deps/lib.sh /opt/chat2me-deps/lib.sh
COPY infra/image-deps/base/apt.sh /opt/chat2me-deps/base/apt.sh
RUN chmod +x /opt/chat2me-deps/lib.sh /opt/chat2me-deps/base/apt.sh \
    && /opt/chat2me-deps/base/apt.sh

COPY packages/common/voice-requirements.txt requirements.txt
RUN . /opt/chat2me-deps/lib.sh \
    && pip_install -r requirements.txt \
    && pip_install fastapi==0.115.6 pydantic==2.10.4 python-multipart==0.0.20 uvicorn[standard]==0.33.0

COPY infra/image-deps /opt/chat2me-deps
RUN find /opt/chat2me-deps -type f -name "*.sh" -exec chmod +x {} +

RUN JETSON_L4T_PLATFORM="$JETSON_L4T_PLATFORM" \
    JETSON_TORCH_WHEEL_URL="$JETSON_TORCH_WHEEL_URL" \
    COSYVOICE_GIT_REF="$COSYVOICE_GIT_REF" \
    /opt/chat2me-deps/install-selected.sh "$VOICE_AGENT_FEATURES"

RUN python3 -m pip install --no-cache-dir eval_type_backport==0.2.2

COPY infra/default-config /defaults/config
COPY packages/common/voice-entrypoint.sh /entrypoint.sh
COPY packages/common/app/runtime.py ./app/runtime.py
COPY services/voice/speech-app/agent.py ./app/agent.py
COPY services/voice/speech-app/cosyvoice_runtime.py ./app/cosyvoice_runtime.py
COPY services/voice/speech-app/f5_tts_runtime.py ./app/f5_tts_runtime.py
COPY services/voice/speech-app/respeaker.py ./app/respeaker.py
COPY services/voice/speech-app/speech.py ./app/speech.py
COPY services/voice/asr-app/asr.py ./app/asr.py
COPY services/voice/tts-app/tts.py ./app/tts.py
COPY services/voice/relay-app/relay.py ./app/relay.py
RUN chmod +x /entrypoint.sh

ENV COSYVOICE_CODE_DIR=/opt/CosyVoice \
    COSYVOICE_PACKAGE_PATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
    COSYVOICE_WHISPER_ASSETS_DIR=/opt/chat2me-whisper-assets \
    PIPER_ESPEAK_DATA=/opt/piper/espeak-ng-data \
    LD_LIBRARY_PATH=/opt/piper:/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra \
    PATH=/usr/src/tensorrt/bin:$PATH

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "app.speech"]
