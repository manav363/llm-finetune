# Serving image for the llm-finetune FastAPI endpoint (M5).
#
# The default build serves the offline **mock** engine — it builds and runs with
# no GPU and no model download, which is what CI and a cold `docker run` need.
# For a real backend, install one extra at build time and set the engine env,
# e.g.:  --build-arg EXTRA=requirements/mac.txt  +  -e LLM_FINETUNE_ENGINE=mlx
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_FINETUNE_ENGINE=mock

WORKDIR /app

# Install dependencies first for better layer caching.
ARG EXTRA=requirements/base.txt
COPY requirements/ ./requirements/
COPY pyproject.toml ./
RUN pip install -r ${EXTRA}

# Then the source + config, and install the package.
COPY src/ ./src/
COPY config/ ./config/
RUN pip install --no-deps -e .

EXPOSE 8000
CMD ["uvicorn", "llm_finetune.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
