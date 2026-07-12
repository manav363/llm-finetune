"""MLX backend: LoRA fine-tuning via mlx-lm on Apple Silicon.

M0 ships the interface and dependency guard only; the real training loop lands
in M2. Heavy imports are deferred to call time so the package imports cleanly
on a machine without MLX installed (e.g. a CUDA box or CI).
"""

from __future__ import annotations

from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.train.backend_base import BackendUnavailable, TrainResult


class MlxBackend:
    name = "mlx"

    def train(self, config: Config, train_path: Path, val_path: Path) -> TrainResult:
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "mlx backend requires the Apple-Silicon extras: "
                "pip install -r requirements/mac.txt"
            ) from exc

        # M2: convert splits to the mlx-lm chat format, run LoRA training,
        # save the adapter to config.train.output_dir, return a real TrainResult.
        raise NotImplementedError(
            "mlx LoRA training loop is implemented in M2; "
            "run with backend: mock for the M0 dry-run"
        )
