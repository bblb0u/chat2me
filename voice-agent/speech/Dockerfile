FROM ubuntu:20.04

ARG JETSON_L4T_PLATFORM=t234
ARG JETSON_TORCH_WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
ARG COSYVOICE_GIT_REF=v2.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      alsa-utils \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libasound2 \
      libgomp1 \
      libopenblas-base \
      libportaudio2 \
      libsndfile1 \
      libusb-1.0-0 \
      portaudio19-dev \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --retry 5 --connect-timeout 20 https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
      -o /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
    && printf '%s\n' \
      'deb https://repo.download.nvidia.com/jetson/common r35.6 main' \
      "deb https://repo.download.nvidia.com/jetson/${JETSON_L4T_PLATFORM} r35.6 main" \
      > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      cuda-cudart-11-4 \
      cuda-cupti-11-4 \
      cuda-nvrtc-11-4 \
      cuda-nvtx-11-4 \
      libcublas-11-4 \
      libcudla-11-4 \
      libcufft-11-4 \
      libcudnn8 \
      libcurand-11-4 \
      libcusolver-11-4 \
      libcusparse-11-4 \
      libnpp-11-4 \
      libnvinfer-bin \
      libnvinfer-plugin8 \
      libnvinfer8 \
      libnvonnxparsers8 \
      libnvparsers8 \
      python3-libnvinfer \
      tensorrt-libs \
    && rm -rf /var/lib/apt/lists/*

COPY voice-agent/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps "$JETSON_TORCH_WHEEL_URL" \
    && python3 -m pip install --no-cache-dir \
      "accelerate==0.34.2" \
      "conformer==0.3.2" \
      "diffusers==0.29.0" \
      "einops==0.8.0" \
      "hydra-core" \
      "HyperPyYAML==1.2.3" \
      "inflect==7.3.1" \
      "kaldi_native_fbank" \
      "librosa==0.10.2" \
      "modelscope==1.20.0" \
      "omegaconf==2.3.0" \
      "onnx==1.16.1" \
      "onnxruntime==1.16.3" \
      "openai-whisper==20231117" \
      "regex" \
      "scipy==1.10.1" \
      "sense-voice-streaming-asr==0.1.1" \
      "sentencepiece" \
      "sherpa-onnx==1.12.38" \
      "tiktoken==0.7.0" \
      "transformers==4.45.2"

RUN git clone --depth 1 --branch "$COSYVOICE_GIT_REF" https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
    && git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS.git /opt/CosyVoice/third_party/Matcha-TTS

COPY config /defaults/config
COPY voice-agent/entrypoint.sh /entrypoint.sh
COPY voice-agent/common/app ./app
COPY voice-agent/speech/app ./app

ENV COSYVOICE_CODE_DIR=/opt/CosyVoice \
    COSYVOICE_PACKAGE_PATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
    LD_LIBRARY_PATH=/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra \
    PATH=/usr/src/tensorrt/bin:$PATH \
    PYTHON_RUNTIME_PREFIX=/opt/chat2m-runtime/python

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "app.speech"]
