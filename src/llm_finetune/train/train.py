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


def main() -> None:
    """CLI: train from the split files produced by `data.split` (or the pipeline)."""
    import argparse

    from llm_finetune.config import load_config

    parser = argparse.ArgumentParser(description="Fine-tune via the configured backend.")
    parser.add_argument("--config", default="config/qa_domain.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_path = config.data.splits_dir / "train.jsonl"
    val_path = config.data.splits_dir / "val.jsonl"
    for path in (train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing split file {path}; run `python -m llm_finetune.data.split` first"
            )
    result = run_training(config, train_path, val_path)
    print(f"backend {result.backend} · train/val {result.n_train}/{result.n_val} "
          f"· steps {result.steps} · adapter {result.adapter_dir}")


if __name__ == "__main__":
    main()
