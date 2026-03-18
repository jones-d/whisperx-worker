FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
WORKDIR /

# System packages: Python 3.12, FFmpeg 6.1, utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        ffmpeg wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create cache directories
RUN mkdir -p /cache/models /root/.cache/torch

# Copy only requirements file first to leverage Docker cache
COPY builder/requirements.txt /builder/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --break-system-packages hf_transfer \
 && python3 -m pip install --break-system-packages --no-cache-dir -r /builder/requirements.txt

# Copy the local VAD model to the expected location
COPY models/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin

# Copy the rest of the builder files
COPY builder /builder

# Download Faster Whisper Models
RUN chmod +x /builder/download_models.sh
RUN --mount=type=secret,id=hf_token /builder/download_models.sh

# Permanently upgrade Lightning checkpoint to silence startup warning
RUN python3 -m lightning.pytorch.utilities.upgrade_checkpoint \
    /usr/local/lib/python3.12/dist-packages/whisperx/assets/pytorch_model.bin || true

# Copy source code
COPY src .

CMD [ "python3", "-u", "/rp_handler.py" ]
