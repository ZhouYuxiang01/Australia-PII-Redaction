FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libglib2.0-0 \
        libgl1 \
        poppler-utils \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt /app/requirements-docker.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install "torch==2.11.0" \
    && python3 -m pip install -r /app/requirements-docker.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY opf-runtime /app/opf-runtime
RUN python3 -m pip install /app/opf-runtime

COPY pii-redaction-service /app/pii-redaction-service
COPY hybrid-pii-model-runtime /app/hybrid-pii-model-runtime

ENV WRAPPER_HOST=0.0.0.0 \
    WRAPPER_PORT=8090 \
    WRAPPER_PYTHON=python3 \
    WRAPPER_BACKEND_CONFIG=/app/pii-redaction-service/configs/backends/hybrid-opf-qwen9b-hn.json \
    WRAPPER_POLICY_CONFIG=/app/pii-redaction-service/configs/policies/hybrid-80class-v2-4b.json \
    WRAPPER_SERVICE_ROOT=/app/pii-redaction-service \
    REDACTION_PII_PROJECT_ROOT=/app/hybrid-pii-model-runtime \
    REDACTION_QWEN_BACKBONE=/models/Qwen3.5-9B-Base \
    WRAPPER_QWEN_VL_MODEL=/models/Qwen3.5-9B-Base \
    PYTHONPATH=/app/pii-redaction-service:/app/opf-runtime

WORKDIR /app/pii-redaction-service
EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${WRAPPER_PORT}/api/health >/dev/null || exit 1

CMD ["./scripts/run_server.sh"]
