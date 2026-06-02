FROM ubuntu:20.04

ARG VOICE_RUNTIME_FEATURES="sherpa"
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

COPY runtime/deps/lib /opt/chat2me-deps/lib
RUN chmod +x /opt/chat2me-deps/lib/common.sh \
    && . /opt/chat2me-deps/lib/common.sh \
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
RUN . /opt/chat2me-deps/lib/common.sh \
    && pip_install -r requirements.txt

COPY runtime/deps/install.sh /opt/chat2me-deps/install.sh
COPY runtime/deps/shared /opt/chat2me-deps/shared
RUN find /opt/chat2me-deps -type f -name "*.sh" -exec chmod +x {} + \
    && /opt/chat2me-deps/install.sh "$VOICE_RUNTIME_FEATURES"

COPY config /defaults/config
COPY runtime/entrypoints/audio.sh /entrypoint.sh
COPY runtime/shared/common.py ./app/common.py
COPY runtime/shared/voice.py ./app/voice.py
COPY services/speech/app/respeaker.py ./app/respeaker.py
COPY services/speech/app/main.py ./app/main.py
RUN chmod +x /entrypoint.sh

ENV LD_LIBRARY_PATH=/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "app.main"]
