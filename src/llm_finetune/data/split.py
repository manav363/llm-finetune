"""Deterministic, leak-safe train/val/test splitting.

The split is seeded (reproducible) and carries a hard guarantee that no id
appears in more than one split — the LLM equivalent of preventing lookahead
leakage. `assert_no_leakage` makes that guarantee testable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from llm_finetune.schema import QAExample, write_jsonl


class SplitError(ValueError):
    """Raised when a split would be empty or would leak ids across splits."""


@dataclass(frozen=True)
class DatasetSplits:
    train: list[QAExample]
    val: list[QAExample]
    test: list[QAExample]

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def assert_no_leakage(splits: DatasetSplits) -> None:
    """Raise if any id appears in more than one split."""
    train_ids = {ex.id for ex in splits.train}
    val_ids = {ex.id for ex in splits.val}
    test_ids = {ex.id for ex in splits.test}
    overlaps = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlaps:
        raise SplitError(f"leakage: ids in multiple splits: {sorted(overlaps)}")


def split_examples(
    examples: list[QAExample],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> DatasetSplits:
    """Shuffle deterministically and partition into train/val/test."""
    n = len(examples)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    if n_val == 0 or n_test == 0 or (n - n_val - n_test) <= 0:
        raise SplitError(
            f"{n} examples too few for val_frac={val_frac}, test_frac={test_frac}; "
            "every split must be non-empty"
        )

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]

    splits = DatasetSplits(train=train, val=val, test=test)
    assert_no_leakage(splits)
    return splits


def write_splits(splits: DatasetSplits, splits_dir: str | Path) -> dict[str, Path]:
    """Write each split to <splits_dir>/<name>.jsonl and return the paths."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, rows in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        path = out_dir / f"{name}.jsonl"
        write_jsonl(str(path), rows)
        paths[name] = path
    return paths
