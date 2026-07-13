"""M4 sanity check: does quantization hold quality vs the merged fp16 model?

This re-runs a slice of the M3 evaluation, but with a different pairing: the
*merged fp16* model is the baseline and the *quantized* model is the candidate.
It reuses `eval.report.build_report` wholesale — same intrinsic metrics, same
judge, same paired bootstrap — then applies a tolerance gate: the quantized
model passes if its headline (mean judge) quality dropped by no more than
`tolerance` versus merged fp16. A drop beyond that fails the check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llm_finetune.eval.generate import Generation
from llm_finetune.eval.judge import Judge
from llm_finetune.eval.report import EvalReport, build_report


@dataclass(frozen=True)
class SanityResult:
    """Outcome of the quantized-vs-merged quality check."""

    n_items: int
    quant: str
    tolerance: float
    merged_mean: float
    quantized_mean: float
    delta: float
    passed: bool
    verdict: str
    report: EvalReport


def run_sanity_check(
    merged: list[Generation],
    quantized: list[Generation],
    judge: Judge,
    *,
    base_model: str,
    quant: str,
    tolerance: float,
    merged_generator: str,
    quantized_generator: str,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> SanityResult:
    """Score quantized vs merged fp16 and gate on the headline quality drop."""
    report = build_report(
        merged,
        quantized,
        judge,
        base_model=base_model,
        adapter=None,
        base_generator=merged_generator,
        tuned_generator=quantized_generator,
        n_resamples=n_resamples,
        level=level,
        seed=seed,
    )
    # In this pairing the report's "base" is merged fp16 and "tuned" is
    # quantized, so delta = quantized - merged. A pass means quality did not
    # drop by more than the tolerance.
    delta = report.headline.delta.delta
    passed = delta >= -tolerance
    verdict = (
        f"PASS — quantized ({quant}) holds quality within tolerance: "
        f"Δ {delta:+.3f} ≥ -{tolerance:.3f}"
        if passed
        else (
            f"FAIL — quantization ({quant}) dropped quality beyond tolerance: "
            f"Δ {delta:+.3f} < -{tolerance:.3f}"
        )
    )
    return SanityResult(
        n_items=report.n_items,
        quant=quant,
        tolerance=tolerance,
        merged_mean=report.headline.base_mean,
        quantized_mean=report.headline.tuned_mean,
        delta=delta,
        passed=passed,
        verdict=verdict,
        report=report,
    )


def render_sanity_markdown(result: SanityResult) -> str:
    r = result
    return "\n".join(
        [
            "## Quantization sanity check",
            "",
            f"- **Quant level:** `{r.quant}`",
            f"- **Test-slice items:** {r.n_items}",
            f"- **Tolerance (max headline drop):** {r.tolerance:.3f}",
            f"- **Merged fp16 (mean judge quality):** {r.merged_mean:.3f}",
            f"- **Quantized (mean judge quality):** {r.quantized_mean:.3f}",
            f"- **Δ (quantized − merged):** {r.delta:+.3f}",
            "",
            f"**{r.verdict}**",
            "",
        ]
    )


def _sanity_to_dict(result: SanityResult) -> dict[str, object]:
    return {
        "n_items": result.n_items,
        "quant": result.quant,
        "tolerance": result.tolerance,
        "merged_mean": result.merged_mean,
        "quantized_mean": result.quantized_mean,
        "delta": result.delta,
        "passed": result.passed,
        "verdict": result.verdict,
    }


def write_sanity_json(result: SanityResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanity_to_dict(result), indent=2) + "\n", encoding="utf-8")
    return out
