"""Paired bootstrap for the base-vs-fine-tuned quality delta.

We resample the *paired* per-item score differences (tuned - base) with
replacement to get a confidence interval on the mean delta, plus a preliminary
two-sided bootstrap p-value for "the delta is non-zero".

Honesty note: `tasks.md` scopes the *validated* paired bootstrap + p-value to
the AI Eval Pipeline's M5. Until that lands, treat `p_value` here as
preliminary and `p_value_status = "pending-validated-judge"`: the CI is sound,
but the significance claim is not final until it runs against the validated
judge and the project's agreed method.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

P_VALUE_PENDING = "pending-validated-judge"


@dataclass(frozen=True)
class DeltaEstimate:
    """A mean paired delta with a bootstrap CI and preliminary significance."""

    delta: float
    ci_low: float
    ci_high: float
    level: float
    p_value: float
    p_value_status: str
    n_items: int
    n_resamples: int

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 1]) over a sorted list."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def paired_bootstrap(
    base_scores: list[float],
    tuned_scores: list[float],
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> DeltaEstimate:
    """Bootstrap the mean of (tuned - base) over paired items.

    Returns the observed mean delta, a `level` CI from the resample
    distribution, and a preliminary two-sided p-value (flagged as pending the
    validated judge). Positive delta means the fine-tuned model scored higher.
    """
    if len(base_scores) != len(tuned_scores):
        raise ValueError("base and tuned score lists must be the same length")
    n = len(base_scores)
    if n == 0:
        raise ValueError("need at least one paired item")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")

    diffs = [t - b for t, b in zip(tuned_scores, base_scores, strict=True)]
    observed = sum(diffs) / n

    rng = random.Random(seed)
    means: list[float] = []
    at_or_below_zero = 0
    for _ in range(n_resamples):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += diffs[rng.randrange(n)]
        m = resample_sum / n
        means.append(m)
        if m <= 0.0:
            at_or_below_zero += 1

    means.sort()
    tail = (1.0 - level) / 2.0
    ci_low = _percentile(means, tail)
    ci_high = _percentile(means, 1.0 - tail)

    # Two-sided bootstrap p-value: 2x the smaller tail mass around zero.
    frac_le = at_or_below_zero / n_resamples
    frac_ge = 1.0 - frac_le
    p_value = min(1.0, 2.0 * min(frac_le, frac_ge))

    return DeltaEstimate(
        delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        level=level,
        p_value=p_value,
        p_value_status=P_VALUE_PENDING,
        n_items=n,
        n_resamples=n_resamples,
    )
