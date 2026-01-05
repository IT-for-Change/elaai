FROM python:3.10-slim

# -------------------------
# System dependencies (single layer)
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# -------------------------
# Create virtual envs
# -------------------------
RUN mkdir -p /opt/venvs \
 && python -m venv /opt/venvs/whisper \
 && python -m venv /opt/venvs/pyannote \
 && python -m venv /opt/venvs/spacy

# -------------------------
# Whisper environment (single layer)
# -------------------------
RUN /opt/venvs/whisper/bin/pip install -U pip \
 && /opt/venvs/whisper/bin/pip install \
        torch \
        openai-whisper \
        ffmpeg-python \
        requests

# -------------------------
# pyannote.audio environment (single layer)
# -------------------------
RUN /opt/venvs/pyannote/bin/pip install -U pip \
 && /opt/venvs/pyannote/bin/pip install \
        torch \
        pyannote.audio

# -------------------------
# spaCy environment (single optimized layer)
# -------------------------
RUN /opt/venvs/spacy/bin/pip install --no-cache-dir -U \
        pip setuptools wheel \
 && /opt/venvs/spacy/bin/pip install --no-cache-dir \
        spacy \
        errant==3.0.0 \
        pyphen==0.15.0 \
        happytransformer==3.0.0 \
        requests \
        httpx \
 && /opt/venvs/spacy/bin/python -m spacy download en_core_web_trf
 
ENV PYTHONUNBUFFERED=1

# -------------------------
# Entrypoint
# -------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
