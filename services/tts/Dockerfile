FROM ubuntu:20.04

ARG JETSON_TORCH_WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
ARG MELOTTS_SOURCE_REF=209145371cff8fc3bd60d7be902ea69cbdb7965a
ARG CHAT2ME_DOWNLOAD_RETRIES=10
ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV DEBIAN_FRONTEND=noninteractive \
    VOICE_ROLE=chat2me-tts \
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
       ffmpeg \
       libgomp1 \
       libopenblas-base \
       libsndfile1 \
       python3 \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY services/tts/requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      -r requirements.txt \
    && python3 -m pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      --force-reinstall \
      --no-deps \
      "$JETSON_TORCH_WHEEL_URL" \
    && python3 -m pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      "filelock==3.13.4" \
      "jinja2==3.1.4" \
      "networkx==3.1" \
      "sympy==1.12" \
      "typing_extensions==4.13.2"

COPY services/tts/install_melotts.sh /opt/chat2me-install-melotts.sh
RUN chmod +x /opt/chat2me-install-melotts.sh \
    && MELOTTS_SOURCE_REF="$MELOTTS_SOURCE_REF" \
       /opt/chat2me-install-melotts.sh

COPY config /defaults/config
COPY runtime/entrypoints/audio.sh /entrypoint.sh
COPY runtime/shared/common.py ./app/common.py
COPY runtime/shared/voice.py ./app/voice.py
COPY services/tts/app/main.py ./app/main.py
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/opt/MeloTTS \
    LD_LIBRARY_PATH=/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra \
    PATH=/usr/src/tensorrt/bin:$PATH

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "app.main"]
