FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/core/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config /defaults/config
COPY runtime/entrypoints/config.sh /entrypoint.sh
COPY services/core/app ./app
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log", "--log-level", "warning"]
