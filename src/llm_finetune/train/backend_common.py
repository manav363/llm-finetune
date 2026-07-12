"""Shared, dependency-light helpers used by the real training backends.

These are the parts of M2 that can be exercised offline with no torch / MLX /
model download: deterministic seeding, converting the split files into the
chat-formatted JSONL that both `trl` and `mlx-lm` consume, computing the step
budget, and writing a reproducibility record. Keeping them here means the CUDA
and MLX backends share one formatting/repro path and stay small.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.schema import QAExample, load_jsonl


def set_all_seeds(seed: int) -> None:
    """Seed every RNG that could affect a training run (best-effort, guarded)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy always present via base deps
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - env-dependent
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a cuda-only extra
        pass


def to_chat_records(examples: list[QAExample]) -> list[dict[str, object]]:
    """Render examples as chat records: ``{"messages": [...]}`` per example."""
    return [{"messages": ex.to_chat()} for ex in examples]


def write_chat_jsonl(examples: list[QAExample], path: str | Path) -> Path:
    """Write chat-formatted training records (one JSON object per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in to_chat_records(examples):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out


def training_iters(n_train: int, batch_size: int, epochs: int, max_steps: int) -> int:
    """Resolve the number of optimizer steps for a run.

    ``max_steps > 0`` caps the run (smoke trains); otherwise the budget is the
    number of steps needed to cover ``epochs`` passes over the data.
    """
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_steps > 0:
        return max_steps
    steps_per_epoch = max(1, -(-n_train // batch_size))  # ceil division
    return steps_per_epoch * max(1, epochs)


def library_versions(modules: list[str]) -> dict[str, str]:
    """Best-effort installed-version lookup for reproducibility records."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for module in modules:
        try:
            versions[module] = version(module)
        except PackageNotFoundError:
            versions[module] = "not-installed"
    return versions


def write_run_metadata(
    config: Config,
    *,
    backend: str,
    steps: int,
    n_train: int,
    n_val: int,
    package_names: list[str],
    path: str | Path,
) -> Path:
    """Write a versioned reproducibility record next to the adapter."""
    meta = {
        "backend": backend,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "base_model": config.model.name,
        "max_seq_len": config.model.max_seq_len,
        "lora": {
            "r": config.lora.r,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "target_modules": list(config.lora.target_modules),
        },
        "train": {
            "epochs": config.train.epochs,
            "batch_size": config.train.batch_size,
            "grad_accum": config.train.grad_accum,
            "learning_rate": config.train.learning_rate,
            "max_steps": config.train.max_steps,
            "resolved_steps": steps,
        },
        "data": {"n_train": n_train, "n_val": n_val},
        "wandb": {"enabled": config.wandb.enabled, "project": config.wandb.project},
        "versions": library_versions(package_names),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def load_split(path: str | Path) -> list[QAExample]:
    """Load and validate a split file into examples."""
    return load_jsonl(str(path))
