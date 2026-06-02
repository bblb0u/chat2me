FROM ubuntu:20.04

ARG VOICE_AGENT_FEATURES="sensevoice sherpa online"
ARG CHAT2ME_DOWNLOAD_RETRIES=10
ARG CHAT2ME_GIT_RETRIES=5
ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV DEBIAN_FRONTEND=noninteractive \
    VOICE_ROLE=chat2me-asr \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHAT2ME_DOWNLOAD_RETRIES=${CHAT2ME_DOWNLOAD_RETRIES} \
    CHAT2ME_GIT_RETRIES=${CHAT2ME_GIT_RETRIES} \
    CHAT2ME_PIP_RETRIES=${CHAT2ME_PIP_RETRIES} \
    CHAT2ME_PIP_TIMEOUT=${CHAT2ME_PIP_TIMEOUT}

WORKDIR /app

COPY runtime/deps/lib.sh /opt/chat2me-deps/lib.sh
RUN chmod +x /opt/chat2me-deps/lib.sh \
    && . /opt/chat2me-deps/lib.sh \
    && apt_install_packages \
       ca-certificates \
       curl \
       ffmpeg \
       libgomp1 \
       libopenblas-base \
       python3 \
       python3-dev \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY services/asr/requirements.txt requirements.txt
RUN . /opt/chat2me-deps/lib.sh \
    && pip_install -r requirements.txt \
    && pip_install fastapi==0.115.6 pydantic==2.10.4 python-multipart==0.0.20 uvicorn[standard]==0.33.0

RUN python3 -m pip install --no-cache-dir eval_type_backport==0.2.2

COPY runtime/deps/install-selected.sh /opt/chat2me-deps/install-selected.sh
COPY runtime/deps/sherpa-onnx.sh /opt/chat2me-deps/sherpa-onnx.sh
COPY runtime/deps/asr /opt/chat2me-deps/asr
COPY runtime/deps/online /opt/chat2me-deps/online
RUN find /opt/chat2me-deps -type f -name "*.sh" -exec chmod +x {} + \
    && /opt/chat2me-deps/install-selected.sh "$VOICE_AGENT_FEATURES"

COPY config /defaults/config
COPY runtime/audio-entrypoint.sh /entrypoint.sh
COPY runtime/app/runtime.py ./app/runtime.py
COPY runtime/app/agent.py ./app/agent.py
COPY services/asr/app/main.py ./app/main.py
RUN chmod +x /entrypoint.sh

ENV LD_LIBRARY_PATH=/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "app.main"]
