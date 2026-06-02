FROM ubuntu:20.04

ARG JETSON_L4T_PLATFORM=t234
ARG JETSON_TORCH_WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
ARG COSYVOICE_GIT_REF=v2.0
ARG VOICE_RUNTIME_FEATURES="piper melotts f5-tts cosyvoice online"
ARG CHAT2ME_DOWNLOAD_RETRIES=10
ARG CHAT2ME_GIT_RETRIES=5
ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV DEBIAN_FRONTEND=noninteractive \
    VOICE_ROLE=chat2me-tts \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHAT2ME_DOWNLOAD_RETRIES=${CHAT2ME_DOWNLOAD_RETRIES} \
    CHAT2ME_GIT_RETRIES=${CHAT2ME_GIT_RETRIES} \
    CHAT2ME_PIP_RETRIES=${CHAT2ME_PIP_RETRIES} \
    CHAT2ME_PIP_TIMEOUT=${CHAT2ME_PIP_TIMEOUT}

WORKDIR /app

COPY runtime/deps/lib /opt/chat2me-deps/lib
RUN chmod +x /opt/chat2me-deps/lib/common.sh \
    && . /opt/chat2me-deps/lib/common.sh \
    && apt_install_packages \
       ca-certificates \
       curl \
       ffmpeg \
       git \
       libgomp1 \
       libopenblas-base \
       libsndfile1 \
       python3 \
       python3-dev \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY services/tts/requirements.txt requirements.txt
RUN . /opt/chat2me-deps/lib/common.sh \
    && pip_install -r requirements.txt \
    && pip_install fastapi==0.115.6 pydantic==2.10.4 python-multipart==0.0.20 uvicorn[standard]==0.33.0

COPY runtime/deps/install.sh /opt/chat2me-deps/install.sh
COPY runtime/deps/shared /opt/chat2me-deps/shared
COPY runtime/deps/tts /opt/chat2me-deps/tts
COPY runtime/deps/platform /opt/chat2me-deps/platform
COPY runtime/deps/online /opt/chat2me-deps/online
RUN find /opt/chat2me-deps -type f -name "*.sh" -exec chmod +x {} + \
    && JETSON_L4T_PLATFORM="$JETSON_L4T_PLATFORM" \
       JETSON_TORCH_WHEEL_URL="$JETSON_TORCH_WHEEL_URL" \
       COSYVOICE_GIT_REF="$COSYVOICE_GIT_REF" \
       /opt/chat2me-deps/install.sh "$VOICE_RUNTIME_FEATURES"

RUN python3 -m pip install --no-cache-dir eval_type_backport==0.2.2

COPY config /defaults/config
COPY runtime/entrypoints/audio.sh /entrypoint.sh
COPY runtime/shared/common.py ./app/common.py
COPY runtime/shared/voice.py ./app/voice.py
COPY services/tts/app/engines ./app/engines
COPY services/tts/app/main.py ./app/main.py
RUN chmod +x /entrypoint.sh

ENV COSYVOICE_CODE_DIR=/opt/CosyVoice \
    COSYVOICE_PACKAGE_PATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
    COSYVOICE_WHISPER_ASSETS_DIR=/opt/chat2me-whisper-assets \
    PIPER_ESPEAK_DATA=/opt/piper/espeak-ng-data \
    LD_LIBRARY_PATH=/opt/piper:/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra \
    PATH=/usr/src/tensorrt/bin:$PATH

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "app.main"]
