"""Assemble MODEL_CARD.md from the committed artifacts.

The model card is the human-facing summary of a run: what base model, which
dataset version, the LoRA + training config, the evaluation result *with its
uncertainty*, the quantized artifact, and the honest limitations. It reads the
JSON that earlier milestones already commit (stats, eval report, export
manifest) so the card can't drift from the numbers — it renders them.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_finetune.config import Config, load_config
from llm_finetune.repro import version as ver
from llm_finetune.repro.registry import RunRecord, build_record, eval_summary_from_report

DEFAULT_CARD_PATH = Path("MODEL_CARD.md")


def _load_json(path: str | Path) -> dict[str, object] | None:
    p = Path(path)
    if not p.is_file():
        return None
    data: dict[str, object] = json.loads(p.read_text(encoding="utf-8"))
    return data


def _fmt(value: object, spec: str = "") -> str:
    if value is None:
        return "—"
    if spec and isinstance(value, int | float):
        return format(value, spec)
    return str(value)


def _dataset_section(stats: dict[str, object] | None, data_version: str) -> list[str]:
    lines = ["## Dataset", "", f"- **Version (content hash):** `{data_version}`"]
    if stats is None:
        lines.append("- (stats unavailable — run the data pipeline to generate stats.json)")
        return lines
    lines.append(f"- **Examples:** {_fmt(stats.get('count'))}")
    balance = stats.get("category_balance")
    if isinstance(balance, dict) and balance:
        pretty = ", ".join(f"{k}: {v}" for k, v in sorted(balance.items()))
        lines.append(f"- **Category balance:** {pretty}")
    q = stats.get("question_tokens")
    a = stats.get("answer_tokens")
    if isinstance(q, dict) and isinstance(a, dict):
        lines.append(
            f"- **Token lengths (mean):** question {_fmt(q.get('mean'))}, "
            f"answer {_fmt(a.get('mean'))}"
        )
    return lines


def _eval_section(eval_report: dict[str, object] | None) -> list[str]:
    if eval_report is None:
        return ["## Evaluation", "", "- (no eval report committed yet)"]
    summary = eval_summary_from_report(eval_report)
    lines = [
        "## Evaluation (base vs fine-tuned, held-out test)",
        "",
        f"- **Judge:** `{_fmt(eval_report.get('judge'))}` "
        f"(placeholder pending the validated judge)",
        f"- **Test items:** {_fmt(eval_report.get('n_items'))}",
        f"- **Headline (mean judge quality):** base {_fmt(summary['base_mean'], '.3f')} "
        f"→ tuned {_fmt(summary['tuned_mean'], '.3f')}",
        f"- **Δ:** {_fmt(summary['delta'], '+.3f')} "
        f"(95% CI [{_fmt(summary['ci_low'], '+.3f')}, {_fmt(summary['ci_high'], '+.3f')}])",
        f"- **Significance:** p-value `{_fmt(summary['p_value_status'])}`",
        "",
        f"**Verdict:** {_fmt(summary['verdict'])}",
    ]
    return lines


def _export_section(export_manifest: dict[str, object] | None) -> list[str]:
    if export_manifest is None:
        return []
    return [
        "## Quantized artifact (M4)",
        "",
        f"- **Quant level:** `{_fmt(export_manifest.get('quant'))}`",
        f"- **Size:** {_fmt(export_manifest.get('size_human'))} "
        f"({_fmt(export_manifest.get('size_bytes'))} bytes)",
        f"- **GGUF:** `{_fmt(export_manifest.get('gguf_path'))}`",
    ]


def _limitations_section(eval_report: dict[str, object] | None) -> list[str]:
    return [
        "## Known limitations",
        "",
        "- **Judge is a lexical placeholder**, not the AI Eval Pipeline's validated "
        "judge; the significance verdict is preliminary until that lands.",
        "- **Small synthetic sample** — the bundled dataset is a 20-item demo corpus "
        "(a 3-item test split); deltas and bootstrap CIs demonstrate the *pipeline*, "
        "not a real quality improvement. Do not read them as a product claim.",
        "- **Leakage control is lexical/id-level.** Dedup uses token-Jaccard and the "
        "split guards duplicate ids (optionally category-stratified); semantic "
        "paraphrases and shared-source records can still cross train/test.",
        "- **The run_id does not pin dependency or base-model *revisions*** — only the "
        "config + data content. Installed library versions are recorded per run (see the "
        "registry), but a real hardware run can differ across package/model versions.",
        "- **Backends:** real QLoRA/MLX training, GGUF export, and non-mock serving are "
        "implemented behind runtime guards but exercised on hardware, not in CI.",
    ]


def build_model_card(
    config: Config,
    *,
    data_version: str,
    run_id: str,
    git_commit: str | None,
    stats: dict[str, object] | None,
    eval_report: dict[str, object] | None,
    export_manifest: dict[str, object] | None,
) -> str:
    """Render the full MODEL_CARD.md content (pure — inputs already loaded)."""
    lines = [
        f"# Model card — {config.model.name} (domain-QA LoRA)",
        "",
        "> Generated from committed artifacts by `python -m llm_finetune.repro.model_card`. "
        "Numbers are read from the eval/export JSON, not hand-written.",
        "",
        "## Run",
        "",
        f"- **Run id:** `{run_id}` (deterministic from config + data version)",
        f"- **Git commit:** `{_fmt(git_commit)}`",
        f"- **Backend:** `{config.backend}`",
        f"- **Base model:** `{config.model.name}` (max_seq_len {config.model.max_seq_len})",
        "",
        "## LoRA + training config",
        "",
        f"- **LoRA:** r={config.lora.r}, alpha={config.lora.alpha}, "
        f"dropout={config.lora.dropout}, targets={list(config.lora.target_modules)}",
        f"- **Training:** {config.train.epochs} epochs, batch {config.train.batch_size}, "
        f"grad_accum {config.train.grad_accum}, lr {config.train.learning_rate}"
        f"{f', max_steps {config.train.max_steps}' if config.train.max_steps else ''}",
        f"- **Seed:** {config.seed}",
        "",
        *_dataset_section(stats, data_version),
        "",
        *_eval_section(eval_report),
        "",
        *_export_section(export_manifest),
        *([""] if export_manifest else []),
        *_limitations_section(eval_report),
        "",
        "## Reproduce",
        "",
        "```bash",
        "pytest -q                                        # data · leak · eval · quant · serve",
        "python -m llm_finetune.pipeline --config config/qa_domain.yaml",
        "python -m llm_finetune.eval.evaluate --mode mock",
        "python -m llm_finetune.quantize.export --mode mock",
        "python -m llm_finetune.repro.build               # re-registers the run + this card",
        "```",
        "",
        f"The `run_id` above is recomputed from the config + data version, so a cold "
        f"clone that regenerates it and gets `{run_id}` is looking at the same run.",
        "",
    ]
    return "\n".join(lines)


def collect_and_render(
    config: Config,
    *,
    reports_dir: Path = Path("reports"),
    stats_path: Path = Path("data/processed/stats.json"),
) -> tuple[str, RunRecord]:
    """Load committed artifacts, compute the run id, and render the card + record."""
    data_ver = ver.data_version(config.data.processed_path)
    stats = _load_json(stats_path)
    eval_report = _load_json(reports_dir / "eval_report.json")
    export_manifest = _load_json(reports_dir / "export.json")

    rid = ver.run_id(config, data_ver)
    card = build_model_card(
        config,
        data_version=data_ver,
        run_id=rid,
        git_commit=ver.git_commit(),
        stats=stats,
        eval_report=eval_report,
        export_manifest=export_manifest,
    )
    eval_summary = eval_summary_from_report(eval_report) if eval_report else {}
    record = build_record(
        config,
        data_version=data_ver,
        artifacts=_artifact_pointers(config, reports_dir, export_manifest),
        eval_summary=eval_summary,
        note="registered by repro.model_card",
    )
    return card, record


def _artifact_pointers(
    config: Config, reports_dir: Path, export_manifest: dict[str, object] | None
) -> dict[str, str]:
    pointers = {
        "adapter": str(config.train.output_dir),
        "eval_report": str(reports_dir / "eval_report.json"),
    }
    if export_manifest and export_manifest.get("gguf_path"):
        pointers["gguf"] = str(export_manifest["gguf_path"])
    return pointers


def write_model_card(card: str, path: str | Path = DEFAULT_CARD_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card, encoding="utf-8")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate MODEL_CARD.md from artifacts.")
    parser.add_argument("--config", default="config/qa_domain.yaml")
    parser.add_argument("--out", default=str(DEFAULT_CARD_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    card, _ = collect_and_render(config)
    path = write_model_card(card, args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
