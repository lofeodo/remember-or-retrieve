# Linux CPU environment for the distributed smoke test (the dev machine is Windows).
# Build:  docker build -f docker/smoke.Dockerfile -t ror-smoke .
# Run:    docker run --rm -v "$PWD":/work -w /work ror-smoke pytest -m slow tests/test_training_smoke.py
FROM python:3.11-slim-bookworm
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/work/.hf-cache
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml /tmp/pyproject.toml
RUN pip install "ray[train]==2.59.0" transformers==5.19.0 peft==0.21.2 \
    safetensors==0.8.0 pynvml==13.0.1 numpy pyyaml pytest ruff
WORKDIR /work
