"""The training-backend contract.

Every backend (mock, cuda, mlx) implements `TrainBackend`. The dispatcher in
`train.py` selects one by name from config — this is the single seam that lets
the same pipeline train on an NVIDIA GPU (QLoRA) or Apple Silicon (MLX LoRA)
without any other code changing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llm_finetune.config import Config


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a training run: where the adapter landed and a summary."""

    backend: str
    adapter_dir: Path
    n_train: int
    n_val: int
    steps: int
    note: str = ""


class TrainBackend(Protocol):
    """A fine-tuning backend. Implementations must not mutate their inputs."""

    name: str

    def train(
        self,
        config: Config,
        train_path: Path,
        val_path: Path,
    ) -> TrainResult:
        """Fine-tune from the given split files and write an adapter to disk."""
        ...


class BackendUnavailable(RuntimeError):
    """Raised when a backend's runtime deps (CUDA / MLX) are not installed."""
