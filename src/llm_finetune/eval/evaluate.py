"""Evaluate base vs fine-tuned on the held-out test split and write a report.

Generates answers for the *same* test items from the base model and the
fine-tuned adapter (greedy, temp=0), scores both with intrinsic metrics and the
judge, bootstraps the paired deltas, and writes a Markdown + JSON report.

Modes:
    mock  offline demo — base echoes the question, tuned echoes the reference,
          so the full metric/judge/bootstrap/report path runs with no model.
    mlx   real generation via mlx-lm (base = no adapter, tuned = adapter).
    cuda  real generation via transformers + peft.

Usage:
    python -m llm_finetune.eval.evaluate --config config/qa_domain.yaml --mode mock
    python -m llm_finetune.eval.evaluate --mode mlx --tuned outputs/adapter
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_finetune.config import Config, load_config
from llm_finetune.eval.generate import (
    CudaGenerator,
    Generator,
    MlxGenerator,
    MockGenerator,
    write_generations,
)
from llm_finetune.eval.judge import HeuristicJudge
from llm_finetune.eval.report import EvalReport, build_report, render_markdown, write_report
from llm_finetune.schema import QAExample, load_jsonl

DEFAULT_OUT_DIR = Path("reports")


def _generators(
    mode: str, config: Config, adapter: Path
) -> tuple[Generator, Generator, str | None]:
    """Return (base_generator, tuned_generator, adapter_str) for the mode."""
    if mode == "mock":
        base = MockGenerator(name="mock-base (echo-question)", transform=lambda ex: ex.question)
        tuned = MockGenerator(name="mock-tuned (echo-reference)", transform=lambda ex: ex.answer)
        return base, tuned, None
    if mode == "mlx":
        return (
            MlxGenerator(config.model.name, adapter_path=None),
            MlxGenerator(config.model.name, adapter_path=adapter),
            str(adapter),
        )
    if mode == "cuda":
        return (
            CudaGenerator(config.model.name, adapter_path=None),
            CudaGenerator(config.model.name, adapter_path=adapter),
            str(adapter),
        )
    raise ValueError(f"unknown mode {mode!r}; choose from mock, mlx, cuda")


def run_evaluation(
    config: Config,
    *,
    mode: str,
    adapter: Path,
    out_dir: Path,
    n_resamples: int = 10_000,
) -> EvalReport:
    """Generate base/tuned answers, score, bootstrap, and write the report."""
    test_path = config.data.splits_dir / "test.jsonl"
    examples: list[QAExample] = load_jsonl(str(test_path))

    base_gen, tuned_gen, adapter_str = _generators(mode, config, adapter)
    base = base_gen.generate(examples)
    tuned = tuned_gen.generate(examples)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_generations(base, out_dir / "base.jsonl")
    write_generations(tuned, out_dir / "tuned.jsonl")

    report = build_report(
        base,
        tuned,
        HeuristicJudge(),
        base_model=config.model.name,
        adapter=adapter_str,
        base_generator=base_gen.name,
        tuned_generator=tuned_gen.name,
        n_resamples=n_resamples,
        level=0.95,
        seed=config.seed,
    )
    write_report(report, out_dir / "eval_report.md", out_dir / "eval_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned.")
    parser.add_argument("--config", default="config/qa_domain.yaml")
    parser.add_argument("--mode", choices=["mock", "mlx", "cuda"], default="mock")
    parser.add_argument("--tuned", help="Adapter dir (default: train.output_dir from config).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = Path(args.tuned) if args.tuned else config.train.output_dir
    report = run_evaluation(
        config,
        mode=args.mode,
        adapter=adapter,
        out_dir=Path(args.out_dir),
        n_resamples=args.resamples,
    )
    print(render_markdown(report))


if __name__ == "__main__":
    main()
