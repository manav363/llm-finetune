"""Eval-as-CI: fail the build when a candidate regresses vs the promoted run.

The promoted checkpoint's headline quality is the bar. A new candidate's eval
report is compared against it on the same held-out test set; if the candidate's
mean judge quality drops by more than `tolerance`, the gate fails (non-zero
exit) so CI blocks the regression. With no promoted baseline yet, the gate
passes — the first run is promoted by default.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TOLERANCE = 0.02


@dataclass(frozen=True)
class GateResult:
    passed: bool
    candidate_quality: float
    promoted_quality: float | None
    delta: float | None
    tolerance: float
    reason: str


def _headline_quality(report: dict[str, object]) -> float:
    headline = report.get("headline")
    if not isinstance(headline, dict) or "tuned_mean" not in headline:
        raise ValueError("eval report missing headline.tuned_mean")
    return float(headline["tuned_mean"])


def evaluate_gate(
    candidate: dict[str, object],
    promoted: dict[str, object] | None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GateResult:
    """Compare candidate vs promoted headline quality; gate on the drop."""
    candidate_q = _headline_quality(candidate)
    if promoted is None:
        return GateResult(
            passed=True,
            candidate_quality=candidate_q,
            promoted_quality=None,
            delta=None,
            tolerance=tolerance,
            reason="no promoted baseline — candidate promoted by default",
        )
    promoted_q = _headline_quality(promoted)
    delta = candidate_q - promoted_q
    passed = delta >= -tolerance
    reason = (
        f"candidate {candidate_q:.3f} vs promoted {promoted_q:.3f} "
        f"(Δ {delta:+.3f}, tolerance -{tolerance:.3f}) — "
        + ("PASS" if passed else "REGRESSION: build should fail")
    )
    return GateResult(
        passed=passed,
        candidate_quality=candidate_q,
        promoted_quality=promoted_q,
        delta=delta,
        tolerance=tolerance,
        reason=reason,
    )


def _load(path: str | Path) -> dict[str, object]:
    data: dict[str, object] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Eval-as-CI gate vs a promoted run.")
    parser.add_argument("--candidate", default="reports/eval_report.json")
    parser.add_argument("--promoted", help="Promoted eval report JSON (omit for first run).")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    candidate = _load(args.candidate)
    promoted = _load(args.promoted) if args.promoted and Path(args.promoted).is_file() else None
    result = evaluate_gate(candidate, promoted, tolerance=args.tolerance)
    print(result.reason)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
