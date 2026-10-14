# Linux CPU environment for the JAX stage smoke test (Python 3.12: gemma 4.x needs it).
# Build:  docker build -f docker/jax.Dockerfile -t ror-jax .
# Run:    docker run --rm -v "$PWD":/work -w /work ror-jax pytest -m slow tests/test_jax_smoke.py
FROM python:3.12-slim-bookworm
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work/src
RUN pip install flax==0.12.10 gemma==4.0.1 jax==0.11.2 kauldron==1.4.4 optax==0.2.8 \
    orbax-checkpoint==0.12.7 pynvml==13.0.1 numpy pyyaml pytest ruff
WORKDIR /work
