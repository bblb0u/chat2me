FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    VOICE_ROLE=chat2me-relay \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir httpx==0.28.1 pyserial==3.5

COPY config /defaults/config
COPY runtime/entrypoints/config.sh /entrypoint.sh
COPY runtime/shared/common.py ./app/common.py
COPY services/relay/app/main.py ./app/main.py
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "app.main"]
