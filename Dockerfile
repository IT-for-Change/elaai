############################
# Builder stage
############################
FROM python:3.10-slim AS builder

# System deps needed only for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade tooling
RUN pip install --no-cache-dir -U pip setuptools wheel

# Install CPU-only torch first
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    openai-whisper \
    pyannote.audio \
    ffmpeg-python \
    spacy \
    errant==3.0.0 \
    pyphen==0.15.0 \
    happytransformer==3.0.0 \
    requests \
    httpx

# Download spaCy transformer model
RUN python -m spacy download en_core_web_trf

# Cleanup, just in case!
RUN rm -rf /root/.cache/pip


############################
# Runtime stage
############################
FROM python:3.10-slim

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
