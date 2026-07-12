"""Dataset statistics report: counts, length distribution, category balance.

Given a list of QAExamples, produce a compact, serializable summary of the
dataset's shape. Lengths are measured in whitespace tokens (a transparent,
tokenizer-independent proxy) over the question, answer, and context fields.
Category balance counts examples per category, bucketing empty categories under
``uncategorized`` so the report always sums to the dataset size.

The report is pure data (`DatasetStats`); rendering (`format_report`) and
persistence (`write_report`) are separate so callers choose the output.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_finetune.schema import QAExample, load_jsonl

UNCATEGORIZED = "uncategorized"


@dataclass(frozen=True)
class LengthSummary:
    """Token-length distribution for one text field."""

    min: int
    max: int
    mean: float
    median: float
    p90: int


@dataclass(frozen=True)
class DatasetStats:
    count: int
    question_tokens: LengthSummary
    answer_tokens: LengthSummary
    context_tokens: LengthSummary
    category_balance: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _token_count(text: str) -> int:
    return len(text.split())


def _percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile (q in [0, 1]) over a non-empty sorted list."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


def _summarize(values: list[int]) -> LengthSummary:
    ordered = sorted(values)
    n = len(ordered)
    total = sum(ordered)
    mid = n // 2
    if n % 2 == 1:
        median = float(ordered[mid])
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2
    return LengthSummary(
        min=ordered[0],
        max=ordered[-1],
        mean=round(total / n, 2),
        median=median,
        p90=_percentile(ordered, 0.9),
    )


def compute_stats(examples: list[QAExample]) -> DatasetStats:
    """Compute counts, length distributions, and category balance."""
    if not examples:
        raise ValueError("cannot compute stats on an empty dataset")

    balance: dict[str, int] = {}
    for ex in examples:
        key = ex.category or UNCATEGORIZED
        balance[key] = balance.get(key, 0) + 1

    return DatasetStats(
        count=len(examples),
        question_tokens=_summarize([_token_count(e.question) for e in examples]),
        answer_tokens=_summarize([_token_count(e.answer) for e in examples]),
        context_tokens=_summarize([_token_count(e.context) for e in examples]),
        category_balance=dict(sorted(balance.items())),
    )


def format_report(stats: DatasetStats) -> str:
    """Render stats as a human-readable plain-text report."""
    lines = [
        f"Dataset stats — {stats.count} examples",
        "",
        f"{'field':<10} {'min':>5} {'median':>7} {'mean':>7} {'p90':>5} {'max':>5}",
    ]
    for field, summary in (
        ("question", stats.question_tokens),
        ("answer", stats.answer_tokens),
        ("context", stats.context_tokens),
    ):
        lines.append(
            f"{field:<10} {summary.min:>5} {summary.median:>7} "
            f"{summary.mean:>7} {summary.p90:>5} {summary.max:>5}"
        )
    lines.append("")
    lines.append("category balance:")
    width = max((len(k) for k in stats.category_balance), default=0)
    for category, count in stats.category_balance.items():
        pct = 100 * count / stats.count
        lines.append(f"  {category:<{width}}  {count:>3}  ({pct:4.1f}%)")
    return "\n".join(lines)


def write_report(stats: DatasetStats, path: str | Path) -> Path:
    """Write the stats report as JSON and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Report dataset statistics for a JSONL file.")
    parser.add_argument("jsonl", help="Path to a processed QA JSONL file.")
    parser.add_argument("--json-out", help="Optional path to write the JSON report to.")
    args = parser.parse_args()

    stats = compute_stats(load_jsonl(args.jsonl))
    print(format_report(stats))
    if args.json_out:
        write_report(stats, args.json_out)


if __name__ == "__main__":
    main()
