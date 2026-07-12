"""MLX backend: LoRA fine-tuning via mlx-lm on Apple Silicon.

Writes the splits as chat-formatted `train.jsonl` / `valid.jsonl` into a data
directory, then drives `mlx-lm`'s LoRA trainer (its documented, version-stable
entry point) to produce an adapter. Heavy imports are deferred so the package
imports cleanly on a non-Mac / CI box; on such a machine `train()` raises
`BackendUnavailable` before doing any work.

Like the CUDA loop, the real training can't run in the offline suite (it needs
MLX and a model download); its testable seams live in `backend_common`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.train import backend_common as common
from llm_finetune.train.backend_base import BackendUnavailable, TrainResult

_PACKAGES = ["mlx", "mlx-lm"]
_DEFAULT_LORA_LAYERS = 16


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

        common.set_all_seeds(config.seed)

        train_examples = common.load_split(train_path)
        val_examples = common.load_split(val_path)
        iters = common.training_iters(
            len(train_examples),
            config.train.batch_size,
            config.train.epochs,
            config.train.max_steps,
        )

        adapter_dir = Path(config.train.output_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # mlx-lm reads a data directory containing train.jsonl and valid.jsonl.
        data_dir = adapter_dir / "mlx_data"
        common.write_chat_jsonl(train_examples, data_dir / "train.jsonl")
        common.write_chat_jsonl(val_examples, data_dir / "valid.jsonl")

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "--model", config.model.name,
            "--train",
            "--data", str(data_dir),
            "--adapter-path", str(adapter_dir),
            "--batch-size", str(config.train.batch_size),
            "--iters", str(iters),
            "--num-layers", str(_DEFAULT_LORA_LAYERS),
            "--learning-rate", str(config.train.learning_rate),
            "--grad-accumulation-steps", str(config.train.grad_accum),
            "--max-seq-length", str(config.model.max_seq_len),
            "--seed", str(config.seed),
        ]
        if config.wandb.enabled:
            cmd += ["--report-to", "wandb", "--project-name", config.wandb.project]

        result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
        if result.returncode != 0:  # pragma: no cover - env-dependent
            raise RuntimeError(
                f"mlx-lm LoRA training failed (exit {result.returncode}):\n{result.stderr}"
            )

        common.write_run_metadata(
            config,
            backend=self.name,
            steps=iters,
            n_train=len(train_examples),
            n_val=len(val_examples),
            package_names=_PACKAGES,
            path=adapter_dir / "run.json",
        )

        return TrainResult(
            backend=self.name,
            adapter_dir=adapter_dir,
            n_train=len(train_examples),
            n_val=len(val_examples),
            steps=iters,
            note=f"MLX LoRA adapter saved to {adapter_dir}",
        )
