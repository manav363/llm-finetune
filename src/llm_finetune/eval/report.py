"""Assemble and render the base-vs-fine-tuned evaluation report.

Given aligned base/tuned generations plus a judge, compute per-dimension means,
paired deltas with bootstrap CIs, and an honest verdict — one that can say the
fine-tune did *not* beat the base model. Intrinsic metrics and judge dimensions
are treated identically: point estimates + a CI on the paired delta. The
significance claim stays flagged pending the validated judge.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_finetune.eval.bootstrap import DeltaEstimate, paired_bootstrap
from llm_finetune.eval.generate import Generation
from llm_finetune.eval.judge import DIMENSIONS, Judge
from llm_finetune.eval.metrics import all_metrics

INTRINSIC_METRICS = ("exact_match", "normalized_match", "token_f1", "rouge_l")


@dataclass(frozen=True)
class DimensionResult:
    name: str
    base_mean: float
    tuned_mean: float
    delta: DeltaEstimate


@dataclass(frozen=True)
class EvalReport:
    n_items: int
    base_model: str
    adapter: str | None
    base_generator: str
    tuned_generator: str
    judge: str
    p_value_status: str
    headline: DimensionResult
    intrinsic: list[DimensionResult]
    judged: list[DimensionResult]
    verdict: str


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dimension(
    name: str,
    base_scores: list[float],
    tuned_scores: list[float],
    *,
    n_resamples: int,
    level: float,
    seed: int,
) -> DimensionResult:
    delta = paired_bootstrap(
        base_scores, tuned_scores, n_resamples=n_resamples, level=level, seed=seed
    )
    return DimensionResult(
        name=name,
        base_mean=_mean(base_scores),
        tuned_mean=_mean(tuned_scores),
        delta=delta,
    )


def _verdict(headline: DimensionResult) -> str:
    d = headline.delta
    if d.ci_excludes_zero and d.delta > 0:
        return (
            "fine-tuned model scored higher (CI excludes zero; "
            "significance pending validated judge)"
        )
    if d.ci_excludes_zero and d.delta < 0:
        return "fine-tuned model REGRESSED vs base (CI excludes zero)"
    return (
        "no measurable difference — CI includes zero; "
        "cannot claim the fine-tune beat the base model"
    )


def build_report(
    base: list[Generation],
    tuned: list[Generation],
    judge: Judge,
    *,
    base_model: str,
    adapter: str | None,
    base_generator: str,
    tuned_generator: str,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> EvalReport:
    """Compute intrinsic + judged deltas and an honest verdict."""
    if len(base) != len(tuned):
        raise ValueError("base and tuned generation lists must be the same length")
    if not base:
        raise ValueError("cannot build a report over zero items")
    for b, t in zip(base, tuned, strict=True):
        if b.id != t.id:
            raise ValueError(f"generation id mismatch: {b.id!r} vs {t.id!r}")

    # Intrinsic metrics: per-item vectors for base and tuned.
    intrinsic: list[DimensionResult] = []
    for metric in INTRINSIC_METRICS:
        base_scores = [all_metrics(g.answer, g.reference)[metric] for g in base]
        tuned_scores = [all_metrics(g.answer, g.reference)[metric] for g in tuned]
        intrinsic.append(
            _dimension(
                metric, base_scores, tuned_scores,
                n_resamples=n_resamples, level=level, seed=seed,
            )
        )

    # Judge dimensions: per-item scores for base and tuned.
    base_judged = [
        judge.score(question=g.question, context=g.context, reference=g.reference, answer=g.answer)
        for g in base
    ]
    tuned_judged = [
        judge.score(question=g.question, context=g.context, reference=g.reference, answer=g.answer)
        for g in tuned
    ]
    judged: list[DimensionResult] = []
    for dim in DIMENSIONS:
        base_scores = [s.as_dict()[dim] for s in base_judged]
        tuned_scores = [s.as_dict()[dim] for s in tuned_judged]
        judged.append(
            _dimension(
                dim, base_scores, tuned_scores,
                n_resamples=n_resamples, level=level, seed=seed,
            )
        )

    # Headline = mean of the three judge dimensions per item (overall quality).
    base_overall = [_mean(list(s.as_dict().values())) for s in base_judged]
    tuned_overall = [_mean(list(s.as_dict().values())) for s in tuned_judged]
    headline = _dimension(
        "judge_overall", base_overall, tuned_overall,
        n_resamples=n_resamples, level=level, seed=seed,
    )

    return EvalReport(
        n_items=len(base),
        base_model=base_model,
        adapter=adapter,
        base_generator=base_generator,
        tuned_generator=tuned_generator,
        judge=judge.name,
        p_value_status=headline.delta.p_value_status,
        headline=headline,
        intrinsic=intrinsic,
        judged=judged,
        verdict=_verdict(headline),
    )


def _row(d: DimensionResult) -> str:
    e = d.delta
    return (
        f"| {d.name} | {d.base_mean:.3f} | {d.tuned_mean:.3f} | {e.delta:+.3f} | "
        f"[{e.ci_low:+.3f}, {e.ci_high:+.3f}] | {e.p_value:.3f} |"
    )


def render_markdown(report: EvalReport) -> str:
    r = report
    lines = [
        "# Evaluation: base vs fine-tuned",
        "",
        f"- **Base model:** {r.base_model}",
        f"- **Adapter:** {r.adapter or '(none)'}",
        f"- **Generation:** base=`{r.base_generator}` · "
        f"tuned=`{r.tuned_generator}` (greedy, temp=0)",
        f"- **Judge:** `{r.judge}`",
        f"- **Test items:** {r.n_items}",
        f"- **Significance:** p-value `{r.p_value_status}` "
        "(local bootstrap CIs shown; validated judge + p-value land with eval-pipeline M5)",
        "",
        f"## Verdict\n\n**{r.verdict}**",
        "",
        f"Headline (mean judge quality): base {r.headline.base_mean:.3f} → "
        f"tuned {r.headline.tuned_mean:.3f}, Δ {r.headline.delta.delta:+.3f} "
        f"(95% CI [{r.headline.delta.ci_low:+.3f}, {r.headline.delta.ci_high:+.3f}]).",
        "",
        "## Judge dimensions",
        "",
        "| dimension | base | tuned | Δ | 95% CI | p (prelim) |",
        "|---|---|---|---|---|---|",
        *[_row(d) for d in r.judged],
        "",
        "## Intrinsic metrics",
        "",
        "| metric | base | tuned | Δ | 95% CI | p (prelim) |",
        "|---|---|---|---|---|---|",
        *[_row(d) for d in r.intrinsic],
        "",
        "> The judge here is a lexical **placeholder**, not the AI Eval Pipeline's "
        "validated judge. Deltas and CIs are real; the significance verdict is "
        "preliminary until the validated judge is wired in.",
        "",
    ]
    return "\n".join(lines)


def _report_to_dict(report: EvalReport) -> dict[str, object]:
    return asdict(report)


def write_report(
    report: EvalReport, md_path: str | Path, json_path: str | Path
) -> tuple[Path, Path]:
    """Write the report as Markdown and JSON; return both paths."""
    md = Path(md_path)
    js = Path(json_path)
    md.parent.mkdir(parents=True, exist_ok=True)
    js.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_markdown(report), encoding="utf-8")
    js.write_text(json.dumps(_report_to_dict(report), indent=2) + "\n", encoding="utf-8")
    return md, js
