"""FastAPI inference endpoint over the fine-tuned model (M5).

`POST /generate` takes a domain question (+ optional grounding context) and
returns the model's answer. The model is loaded **once** into a singleton engine
at startup and reused across requests. Which engine runs is chosen by env so the
same app image serves any backend:

    LLM_FINETUNE_CONFIG   config path        (default: config/qa_domain.yaml)
    LLM_FINETUNE_ENGINE   mock|mlx|cuda|gguf (default: mock)
    LLM_FINETUNE_ADAPTER  adapter dir or .gguf path (default: train.output_dir)

Run:
    uvicorn llm_finetune.serve.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_finetune.config import load_config
from llm_finetune.serve.engine import DEFAULT_MAX_TOKENS, GREEDY_TEMPERATURE, InferenceEngine
from llm_finetune.serve.registry import select_engine

DEFAULT_CONFIG = "config/qa_domain.yaml"
DEFAULT_ENGINE = "mock"


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Domain question to answer.")
    context: str = Field("", description="Optional grounding context.")
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, gt=0, le=4096)
    temperature: float = Field(GREEDY_TEMPERATURE, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    answer: str
    engine: str


class HealthResponse(BaseModel):
    status: str
    engine: str


def engine_from_env() -> InferenceEngine:
    """Build the inference engine described by the environment."""
    config = load_config(os.environ.get("LLM_FINETUNE_CONFIG", DEFAULT_CONFIG))
    name = os.environ.get("LLM_FINETUNE_ENGINE", DEFAULT_ENGINE)
    adapter_env = os.environ.get("LLM_FINETUNE_ADAPTER")
    if adapter_env:
        adapter: Path | None = Path(adapter_env)
    elif name == "mock":
        adapter = None
    else:
        adapter = config.train.output_dir
    return select_engine(name, config, adapter=adapter)


def create_app(engine: InferenceEngine | None = None) -> FastAPI:
    """Build the FastAPI app.

    A preloaded ``engine`` can be injected (tests, embedding). Otherwise the
    engine is built from env and loaded once at startup.
    """
    resolved = engine if engine is not None else engine_from_env()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        resolved.load()  # load model weights once, before serving traffic
        yield

    app = FastAPI(title="llm-finetune serving", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", engine=resolved.name)

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        try:
            answer = resolved.generate(
                req.question,
                req.context,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
        except Exception as exc:  # surface engine failures as a 503, not a 500
            raise HTTPException(status_code=503, detail=f"generation failed: {exc}") from exc
        return GenerateResponse(answer=answer, engine=resolved.name)

    return app


# Module-level app for `uvicorn llm_finetune.serve.api:app` and the container.
app = create_app()
