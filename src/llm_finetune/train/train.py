"""Backend dispatcher: pick a TrainBackend by name and run it.

This is the seam the whole project is built around — `select_backend` is the
only place that knows which concrete backend a config string maps to.
"""

from __future__ import annotations

from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.train.backend_base import TrainBackend, TrainResult
from llm_finetune.train.backend_cuda import CudaBackend
from llm_finetune.train.backend_mlx import MlxBackend
from llm_finetune.train.backend_mock import MockBackend


def select_backend(name: str) -> TrainBackend:
    """Map a config backend string to a concrete backend instance."""
    backends: dict[str, TrainBackend] = {
        "mock": MockBackend(),
        "cuda": CudaBackend(),
        "mlx": MlxBackend(),
    }
    if name not in backends:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(backends)}")
    return backends[name]


def run_training(config: Config, train_path: Path, val_path: Path) -> TrainResult:
    """Dispatch to the configured backend and run training."""
    backend = select_backend(config.backend)
    return backend.train(config, train_path, val_path)
