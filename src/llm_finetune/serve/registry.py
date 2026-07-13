"""Engine selection — the single place a backend name maps to an engine.

Mirrors `train.select_backend` / `quantize.select_exporter`. Concrete engines
are imported lazily so importing the serving app never drags in torch / mlx /
llama.cpp; only the selected engine's deps are touched.

`gguf` is offered alongside the training backends because the M4 artifact is a
GGUF file — the natural thing to serve on a Mac (llama.cpp) once quantized.
"""

from __future__ import annotations

from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.serve.engine import InferenceEngine

VALID_ENGINES = ("mock", "cuda", "mlx", "gguf")


def select_engine(name: str, config: Config, *, adapter: Path | None) -> InferenceEngine:
    """Build the inference engine for ``name`` from config.

    ``adapter`` is the fine-tuned artifact to serve: a LoRA adapter dir for
    mlx/cuda, or a `.gguf` file for the gguf engine. ``None`` serves the base
    model (used by the mock engine and for smoke checks).
    """
    if name == "mock":
        from llm_finetune.serve.backend_mock import MockEngine

        return MockEngine(config.model.name, adapter=adapter)
    if name == "mlx":
        from llm_finetune.serve.backend_mlx import MlxEngine

        return MlxEngine(config.model.name, adapter=adapter)
    if name == "cuda":
        from llm_finetune.serve.backend_cuda import CudaEngine

        return CudaEngine(config.model.name, adapter=adapter)
    if name == "gguf":
        from llm_finetune.serve.backend_gguf import GgufEngine

        if adapter is None:
            raise ValueError("gguf engine needs a .gguf artifact path (--adapter)")
        return GgufEngine(adapter)
    raise ValueError(f"unknown engine {name!r}; choose from {VALID_ENGINES}")
