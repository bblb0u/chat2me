FROM ubuntu:20.04

ARG CHAT2ME_DOWNLOAD_RETRIES=10
ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV DEBIAN_FRONTEND=noninteractive \
    VOICE_ROLE=chat2me-asr \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHAT2ME_DOWNLOAD_RETRIES=${CHAT2ME_DOWNLOAD_RETRIES} \
    CHAT2ME_PIP_RETRIES=${CHAT2ME_PIP_RETRIES} \
    CHAT2ME_PIP_TIMEOUT=${CHAT2ME_PIP_TIMEOUT}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       libgomp1 \
       libopenblas-base \
       python3 \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY services/asr/requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      -r requirements.txt

COPY config /defaults/config
COPY runtime/entrypoints/audio.sh /entrypoint.sh
COPY runtime/shared/common.py ./app/common.py
COPY runtime/shared/voice.py ./app/voice.py
COPY services/asr/app/main.py ./app/main.py
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "app.main"]
