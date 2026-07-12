"""Intrinsic text metrics for base-vs-fine-tuned answer comparison.

Pure functions over (prediction, reference) strings — no model, no network — so
they run in the offline suite. Each returns a score in [0, 1] where higher is
better. Perplexity is model-based and lives with the generators, not here.
"""

from __future__ import annotations

import re
import string

_WS = re.compile(r"\s+")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace (SQuAD-style)."""
    lowered = text.lower().translate(_PUNCT)
    return _WS.sub(" ", lowered).strip()


def _tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def exact_match(prediction: str, reference: str) -> float:
    """1.0 iff the trimmed strings are identical, else 0.0."""
    return float(prediction.strip() == reference.strip())


def normalized_match(prediction: str, reference: str) -> float:
    """1.0 iff normalized strings match (case/punctuation/whitespace-insensitive)."""
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-overlap F1 between prediction and reference."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    ref_counts: dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    pred_counts: dict[str, int] = {}
    for tok in pred:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    overlap = sum(min(count, ref_counts.get(tok, 0)) for tok, count in pred_counts.items())

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1: LCS-based recall/precision harmonic mean over tokens."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


_METRICS = {
    "exact_match": exact_match,
    "normalized_match": normalized_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
}


def all_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Compute every intrinsic metric for one (prediction, reference) pair."""
    return {name: fn(prediction, reference) for name, fn in _METRICS.items()}


def mean_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Mean of each intrinsic metric over paired predictions/references."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not predictions:
        raise ValueError("cannot average metrics over zero items")
    totals: dict[str, float] = {name: 0.0 for name in _METRICS}
    for pred, ref in zip(predictions, references, strict=True):
        for name, value in all_metrics(pred, ref).items():
            totals[name] += value
    n = len(predictions)
    return {name: total / n for name, total in totals.items()}
