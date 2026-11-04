# syntax=docker/dockerfile:1
# Multi-arch enterprise build (amd64/arm64 via buildx).
# Targets: base -> api | worker | dashboard | cli
#   CPU:  docker build --target api -t provenance-api .
#   CUDA: docker build --target worker --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t provenance-worker:gpu .
#   Multi-arch: docker buildx build --platform linux/amd64,linux/arm64 ...
# Offline: bake models/ + ollama layer, then run with --network none.
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

FROM python:${PYTHON_VERSION}-slim AS base
ARG TARGETARCH
ARG TORCH_INDEX_URL
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    WORK_DIR=/tmp/opencode/provenance REPORT_OUT_DIR=/app/reports \
    OLLAMA_HOST=http://ollama:11434 VECTOR_BACKEND=memory

# OpenCV + video/audio demuxing + health probes.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 ffmpeg tini curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --extra-index-url ${TORCH_INDEX_URL} -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY src/ ./src/
COPY worker.py main.py ./
COPY security/ ./security/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY data/reference_events.json ./data/reference_events.json
COPY models/ ./models/

RUN useradd -m -u 10001 analyst && mkdir -p /app/reports /app/data /tmp/opencode \
 && chown -R analyst:analyst /app /tmp/opencode
USER analyst
# TARGETARCH available here for arch-specific wheels if needed.

FROM base AS api
EXPOSE 8000
ENTRYPOINT ["tini", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# GPU note: run with --gpus all (nvidia-docker); torch auto-selects cuda.

FROM base AS worker
ENTRYPOINT ["tini", "--", "celery", "-A", "worker.app", "worker", "--loglevel=info", "--concurrency=2"]
# GPU note: same image under nvidia runtime; prefetch=1 set in worker.py.

FROM base AS dashboard
USER root
RUN pip install --no-cache-dir streamlit requests && rm -rf /root/.cache/pip
USER analyst
COPY dashboard/ ./dashboard/
EXPOSE 8501
ENTRYPOINT ["tini", "--", "streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM base AS cli
ENTRYPOINT ["tini", "--", "python", "main.py"]
CMD ["--help"]
