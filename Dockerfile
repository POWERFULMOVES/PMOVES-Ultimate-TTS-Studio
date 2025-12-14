FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# System deps for audio + minimal runtime utilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      espeak-ng \
      libsndfile1 \
      libportaudio2 \
      portaudio19-dev \
      libaio-dev \
      git \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (keep minimal; optional engines are gated behind try/except in launch.py).
COPY docker/requirements.txt /app/docker/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/docker/requirements.txt

# App sources
COPY . /app

# Persist models/cache outside the container.
ENV HF_HOME=/data/hf \
    HUGGINGFACE_HUB_CACHE=/data/hf \
    XDG_CACHE_HOME=/data/cache

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${GRADIO_SERVER_PORT}/" >/dev/null || exit 1

CMD ["python", "launch.py"]

