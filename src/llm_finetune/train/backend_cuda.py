"""CUDA backend: QLoRA fine-tuning via transformers + peft + bitsandbytes + trl.

M0 ships the interface and dependency guard only; the real training loop lands
in M2. Importing heavy deps is deferred to call time so the package imports
cleanly on a machine without torch/CUDA installed.
"""

from __future__ import annotations

from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.train.backend_base import BackendUnavailable, TrainResult


class CudaBackend:
    name = "cuda"

    def train(self, config: Config, train_path: Path, val_path: Path) -> TrainResult:
        try:
            import bitsandbytes  # noqa: F401
            import peft  # noqa: F401
            import torch
            import transformers  # noqa: F401
            import trl  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "cuda backend requires the CUDA extras: "
                "pip install -r requirements/cuda.txt"
            ) from exc

        if not torch.cuda.is_available():  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "cuda backend selected but no CUDA device is available"
            )

        # M2: load 4-bit base model, attach LoRA adapters, run trl SFTTrainer,
        # save the adapter to config.train.output_dir, return a real TrainResult.
        raise NotImplementedError(
            "cuda QLoRA training loop is implemented in M2; "
            "run with backend: mock for the M0 dry-run"
        )
