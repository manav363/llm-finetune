"""Mock backend: no real training, no GPU, no model download.

Used for the offline dry-run and CI. It validates that the split files load,
counts the examples, and writes a small `adapter.json` marker so the rest of
the pipeline (eval wiring, serving stubs) has a concrete artifact to point at.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.schema import load_jsonl
from llm_finetune.train.backend_base import TrainResult


class MockBackend:
    name = "mock"

    def train(self, config: Config, train_path: Path, val_path: Path) -> TrainResult:
        train_rows = load_jsonl(str(train_path))
        val_rows = load_jsonl(str(val_path))

        adapter_dir = Path(config.train.output_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        marker = {
            "backend": self.name,
            "base_model": config.model.name,
            "lora": {
                "r": config.lora.r,
                "alpha": config.lora.alpha,
                "dropout": config.lora.dropout,
            },
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "trained": False,
            "note": "mock adapter — no weights were trained",
        }
        (adapter_dir / "adapter.json").write_text(
            json.dumps(marker, indent=2), encoding="utf-8"
        )

        return TrainResult(
            backend=self.name,
            adapter_dir=adapter_dir,
            n_train=len(train_rows),
            n_val=len(val_rows),
            steps=0,
            note="mock run: wrote adapter.json marker, no weights trained",
        )
