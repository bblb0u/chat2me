FROM ollama/ollama:latest AS upstream

FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=upstream /usr/bin/ollama /usr/bin/ollama
COPY --from=upstream /usr/lib/ollama /usr/lib/ollama
COPY config /defaults/config
COPY ollama/entrypoint.sh /entrypoint.sh

ENV LD_LIBRARY_PATH=/usr/lib/ollama:/usr/lib/ollama/cuda_jetpack5:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_VISIBLE_DEVICES=all \
    OLLAMA_HOST=0.0.0.0:11434

ENTRYPOINT ["/entrypoint.sh"]
