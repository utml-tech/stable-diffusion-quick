FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /workspaces/

ENV TZ "Asia/Tokyo"
ENV PYTHONUNBUFFERED 1
ENV OPENCV_LOG_LEVEL ERROR
ENV OPENCV_VIDEOIO_DEBUG 0
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    fontconfig \
    git \
    git-lfs \
    gnupg \
    libasound2 \
    libcublas-dev-11-8 \
    libcudnn8-dev \
    libfreetype6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgoogle-perftools-dev \
    nodejs \
    nvidia-gds-11-8 \
    openssh-client \
    python3 \
    python3-pip \
    python3-venv \
    software-properties-common \
    sysstat \
    tree \
    tmux \
    tzdata \
    unp \
    unrar-free \
    wget \
    xkb-data
RUN apt-get clean

RUN git lfs install

RUN python3 -m pip install -U pip

ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt
