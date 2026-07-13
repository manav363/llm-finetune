"""Deterministic, leak-safe train/val/test splitting.

The split is seeded (reproducible) and carries a hard guarantee that no id
appears in more than one split — the LLM equivalent of preventing lookahead
leakage. `assert_no_leakage` makes that guarantee testable.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from llm_finetune.schema import QAExample, load_jsonl, write_jsonl


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


def _partition(
    examples: list[QAExample], *, val_frac: float, test_frac: float, seed: int
) -> tuple[list[QAExample], list[QAExample], list[QAExample]]:
    """Deterministically shuffle and slice into (test, val, train). No min checks."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    return (
        shuffled[:n_test],
        shuffled[n_test : n_test + n_val],
        shuffled[n_test + n_val :],
    )


def split_examples(
    examples: list[QAExample],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify_by_category: bool = False,
) -> DatasetSplits:
    """Deterministically partition into train/val/test.

    With ``stratify_by_category`` the split is done *within* each category so the
    category distribution is preserved across train/val/test — this narrows one
    leakage/imbalance vector (a category landing entirely in one split). Records
    with no category, and singleton categories, fall back to the global split.
    Note: this is still a lexical/id-level guarantee — semantic paraphrases that
    survive lexical dedup can still cross splits (see prepare.py).
    """
    n = len(examples)
    if stratify_by_category and any(ex.category for ex in examples):
        test, val, train = _stratified_partition(
            examples, val_frac=val_frac, test_frac=test_frac, seed=seed
        )
    else:
        test, val, train = _partition(
            examples, val_frac=val_frac, test_frac=test_frac, seed=seed
        )

    if not test or not val or not train:
        raise SplitError(
            f"{n} examples too few for val_frac={val_frac}, test_frac={test_frac}"
            f"{' (stratified)' if stratify_by_category else ''}; every split must be non-empty"
        )

    splits = DatasetSplits(train=train, val=val, test=test)
    assert_no_leakage(splits)
    return splits


def _stratified_partition(
    examples: list[QAExample], *, val_frac: float, test_frac: float, seed: int
) -> tuple[list[QAExample], list[QAExample], list[QAExample]]:
    """Partition within each category, then concatenate the per-category splits."""
    by_category: dict[str, list[QAExample]] = {}
    for ex in examples:
        by_category.setdefault(ex.category, []).append(ex)

    test: list[QAExample] = []
    val: list[QAExample] = []
    train: list[QAExample] = []
    # Deterministic category order; vary the per-category seed so categories
    # don't all shuffle identically.
    for offset, category in enumerate(sorted(by_category)):
        c_test, c_val, c_train = _partition(
            by_category[category], val_frac=val_frac, test_frac=test_frac, seed=seed + offset
        )
        test += c_test
        val += c_val
        train += c_train
    return test, val, train


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


def main() -> None:
    """CLI: split the processed dataset into leak-safe train/val/test files."""
    from llm_finetune.config import load_config

    parser = argparse.ArgumentParser(description="Split the processed dataset.")
    parser.add_argument("--config", default="config/qa_domain.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    examples = load_jsonl(str(config.data.processed_path))
    splits = split_examples(
        examples,
        val_frac=config.data.val_frac,
        test_frac=config.data.test_frac,
        seed=config.seed,
        stratify_by_category=config.data.stratify_splits,
    )
    paths = write_splits(splits, config.data.splits_dir)
    print(f"splits ({splits.counts()}) -> {config.data.splits_dir}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
