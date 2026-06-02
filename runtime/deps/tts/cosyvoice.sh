#!/bin/sh
set -eu

. /opt/chat2me-deps/lib.sh

/opt/chat2me-deps/platform/jetson-gpu.sh
/opt/chat2me-deps/platform/jetson-torch.sh

COSYVOICE_GIT_REF="${COSYVOICE_GIT_REF:-v2.0}"

pip_install \
  "conformer==0.3.2" \
  "diffusers==0.29.0" \
  "einops==0.8.0" \
  "hydra-core" \
  "HyperPyYAML==1.2.3" \
  "inflect==7.3.1" \
  "librosa==0.10.2" \
  "modelscope==1.20.0" \
  "omegaconf==2.3.0" \
  "onnx==1.16.1" \
  "onnxruntime==1.16.3" \
  "regex" \
  "safetensors" \
  "scipy==1.10.1" \
  "soundfile" \
  "tiktoken==0.7.0" \
  "transformers==4.45.2"

rm -rf /opt/CosyVoice
git_clone_retry /opt/CosyVoice "$GIT_RETRIES" --depth 1 --branch "$COSYVOICE_GIT_REF" https://github.com/FunAudioLLM/CosyVoice.git
git_clone_retry /opt/CosyVoice/third_party/Matcha-TTS "$GIT_RETRIES" --depth 1 https://github.com/shivammehta25/Matcha-TTS.git

pip_download --no-deps "openai-whisper==20231117" -d /tmp/chat2me-whisper
mkdir -p /opt/chat2me-whisper-assets
tar -xzf /tmp/chat2me-whisper/openai-whisper-20231117.tar.gz -C /tmp/chat2me-whisper \
  openai-whisper-20231117/whisper/assets/gpt2.tiktoken \
  openai-whisper-20231117/whisper/assets/multilingual.tiktoken
mv /tmp/chat2me-whisper/openai-whisper-20231117/whisper/assets/*.tiktoken /opt/chat2me-whisper-assets/
rm -rf /tmp/chat2me-whisper
