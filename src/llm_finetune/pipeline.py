"""End-to-end dry-run entrypoint: prepare -> split -> train (backend dispatch).

With `backend: mock` in the config this runs fully offline — no GPU, no model
download — and is what M0's acceptance check exercises. Real training happens
by flipping `backend` to `cuda` or `mlx` (implemented in M2).

Usage:
    python -m llm_finetune.pipeline --config config/qa_domain.yaml
"""

from __future__ import annotations

import argparse

from llm_finetune.config import Config, load_config
from llm_finetune.data.prepare import prepare
from llm_finetune.data.split import split_examples, write_splits
from llm_finetune.train.backend_base import TrainResult
from llm_finetune.train.train import run_training


def run_pipeline(config: Config) -> TrainResult:
    """Run prepare -> split -> train and return the training result."""
    examples = prepare(config.data.raw_path, config.data.processed_path)
    splits = split_examples(
        examples,
        val_frac=config.data.val_frac,
        test_frac=config.data.test_frac,
        seed=config.seed,
    )
    paths = write_splits(splits, config.data.splits_dir)
    return run_training(config, paths["train"], paths["val"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fine-tuning pipeline dry-run.")
    parser.add_argument("--config", default="config/qa_domain.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_pipeline(config)

    print(f"backend      : {result.backend}")
    print(f"train / val  : {result.n_train} / {result.n_val}")
    print(f"adapter dir  : {result.adapter_dir}")
    print(f"note         : {result.note}")


if __name__ == "__main__":
    main()
