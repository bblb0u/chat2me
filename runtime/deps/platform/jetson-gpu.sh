#!/bin/sh
set -eu

. /opt/chat2me-deps/lib.sh

if [ -f /opt/chat2me-deps/.jetson-gpu-installed ]; then
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
JETSON_L4T_PLATFORM="${JETSON_L4T_PLATFORM:-t234}"

download_file \
  "https://repo.download.nvidia.com/jetson/jetson-ota-public.asc" \
  /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
  "Jetson apt key"
printf '%s\n' \
  'deb https://repo.download.nvidia.com/jetson/common r35.6 main' \
  "deb https://repo.download.nvidia.com/jetson/${JETSON_L4T_PLATFORM} r35.6 main" \
  > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

apt_install_packages \
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
  tensorrt-libs
rm -rf /var/lib/apt/lists/*
touch /opt/chat2me-deps/.jetson-gpu-installed
