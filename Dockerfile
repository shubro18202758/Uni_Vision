# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Uni_Vision — Multi-stage Production Dockerfile
#
# Stage 1: dependency builder  (installs wheels into a venv)
# Stage 2: runtime             (minimal CUDA image with app code)
#
# Build:
#   docker build -t uni-vision:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 --env-file .env uni-vision:latest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for wheel compilation (asyncpg, opencv, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

# Install into an isolated venv so we can copy it cleanly
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir ".[inference]"


# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        libpq5 \
        libgl1 \
        libglib2.0-0 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code & config
COPY src/ src/
COPY config/ config/

# Create non-root user
RUN groupadd -r univision && \
    useradd -r -g univision -d /app -s /sbin/nologin univision && \
    chown -R univision:univision /app

USER univision

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start the API server
CMD ["python", "-m", "uni_vision.api"]
