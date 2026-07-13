"""One-shot M6 builder: register the run and (re)write the model card.

Reads the committed artifacts, computes the deterministic run id, upserts a
registry entry, and writes MODEL_CARD.md — the single command a maintainer runs
to refresh the reproducibility surface after a new eval/export.

    python -m llm_finetune.repro.build --config config/qa_domain.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_finetune.config import load_config
from llm_finetune.repro.model_card import (
    DEFAULT_CARD_PATH,
    collect_and_render,
    write_model_card,
)
from llm_finetune.repro.registry import DEFAULT_REGISTRY, register_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the run and write the model card.")
    parser.add_argument("--config", default="config/qa_domain.yaml")
    parser.add_argument("--card", default=str(DEFAULT_CARD_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args()

    config = load_config(args.config)
    card, record = collect_and_render(config)
    card_path = write_model_card(card, args.card)
    register_run(record, Path(args.registry))

    print(f"run_id   {record.run_id}")
    print(f"card     {card_path}")
    print(f"registry {args.registry}")


if __name__ == "__main__":
    main()
