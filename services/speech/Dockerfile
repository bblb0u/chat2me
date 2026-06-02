FROM ubuntu:20.04

ARG VOICE_AGENT_FEATURES="sherpa"
ARG CHAT2ME_DOWNLOAD_RETRIES=10
ARG CHAT2ME_GIT_RETRIES=5
ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV DEBIAN_FRONTEND=noninteractive \
    VOICE_ROLE=chat2me-speech \
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
       alsa-utils \
       ca-certificates \
       curl \
       libasound2 \
       libgomp1 \
       libopenblas-base \
       libportaudio2 \
       libusb-1.0-0 \
       portaudio19-dev \
       python3 \
       python3-dev \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY services/speech/requirements.txt requirements.txt
RUN . /opt/chat2me-deps/lib.sh \
    && pip_install -r requirements.txt

COPY runtime/deps/install-selected.sh /opt/chat2me-deps/install-selected.sh
COPY runtime/deps/sherpa-onnx.sh /opt/chat2me-deps/sherpa-onnx.sh
RUN find /opt/chat2me-deps -type f -name "*.sh" -exec chmod +x {} + \
    && /opt/chat2me-deps/install-selected.sh "$VOICE_AGENT_FEATURES"

COPY config /defaults/config
COPY runtime/audio-entrypoint.sh /entrypoint.sh
COPY runtime/app/runtime.py ./app/runtime.py
COPY runtime/app/agent.py ./app/agent.py
COPY services/speech/app/respeaker.py ./app/respeaker.py
COPY services/speech/app/speech.py ./app/speech.py
RUN chmod +x /entrypoint.sh

ENV LD_LIBRARY_PATH=/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "app.speech"]
