"""M4 entrypoint: merge the winning LoRA, quantize to GGUF, sanity-check.

Runs the two M4 steps end to end and writes one report:

    1. Export — dispatch to the mode's exporter to merge the adapter into the
       base weights and produce a single quantized GGUF; record size + quant
       level in `export.json`.
    2. Sanity — re-run a slice of the M3 evaluation with the merged fp16 model
       as baseline and the quantized model as candidate, gating on a tolerance.

Modes mirror the eval CLI:
    mock  offline — placeholder merge + GGUF, identical merged/quantized answers
          so the whole path runs with no model or llama.cpp.
    mlx   real fuse + GGUF export on Apple Silicon.
    cuda  real peft merge + llama.cpp quantize on an NVIDIA GPU.

Usage:
    python -m llm_finetune.quantize.export --config config/qa_domain.yaml --mode mock
    python -m llm_finetune.quantize.export --mode mlx --adapter outputs/adapter
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from llm_finetune.config import Config, load_config
from llm_finetune.eval.generate import (
    CudaGenerator,
    Generator,
    GgufGenerator,
    MlxGenerator,
    MockGenerator,
)
from llm_finetune.eval.judge import HeuristicJudge
from llm_finetune.quantize.artifact import ExportResult, build_manifest
from llm_finetune.quantize.exporter import select_exporter
from llm_finetune.quantize.sanity import (
    SanityResult,
    render_sanity_markdown,
    run_sanity_check,
    write_sanity_json,
)
from llm_finetune.schema import QAExample, load_jsonl
from llm_finetune.train import backend_common as common

DEFAULT_OUT_DIR = Path("reports")
DEFAULT_SLICE = 8
_MANIFEST_PACKAGES = ["torch", "transformers", "peft", "mlx-lm", "llama-cpp-python"]


def _sanity_generators(
    mode: str, config: Config, result: ExportResult
) -> tuple[Generator, Generator]:
    """Return (merged_fp16_generator, quantized_generator) for the mode."""
    if mode == "mock":
        # A faithful quant preserves answers: both echo the reference, so the
        # measured drop is zero and the tolerance gate passes.
        return (
            MockGenerator(name="mock-merged-fp16", transform=lambda ex: ex.answer),
            MockGenerator(name="mock-quantized", transform=lambda ex: ex.answer),
        )
    if mode == "mlx":
        return (
            MlxGenerator(str(result.merged_dir), adapter_path=None, name="mlx-merged-fp16"),
            GgufGenerator(result.gguf_path),
        )
    if mode == "cuda":
        return (
            CudaGenerator(str(result.merged_dir), adapter_path=None, name="cuda-merged-fp16"),
            GgufGenerator(result.gguf_path),
        )
    raise ValueError(f"unknown mode {mode!r}; choose from mock, mlx, cuda")


def run_export_and_sanity(
    config: Config,
    *,
    mode: str,
    adapter: Path,
    out_dir: Path,
    slice_n: int = DEFAULT_SLICE,
    n_resamples: int = 10_000,
) -> tuple[ExportResult, SanityResult]:
    """Export the quantized artifact, then sanity-check it against merged fp16."""
    result = select_exporter(mode).export(config, adapter)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        result,
        base_model=config.model.name,
        adapter=str(adapter),
        seed=config.seed,
        versions=common.library_versions(_MANIFEST_PACKAGES),
    )
    _write_json(manifest, out_dir / "export.json")

    test_path = config.data.splits_dir / "test.jsonl"
    examples: list[QAExample] = load_jsonl(str(test_path))[:slice_n]
    if not examples:
        raise ValueError(f"no test items to sanity-check (looked in {test_path})")

    merged_gen, quant_gen = _sanity_generators(mode, config, result)
    sanity = run_sanity_check(
        merged_gen.generate(examples),
        quant_gen.generate(examples),
        HeuristicJudge(),
        base_model=config.model.name,
        quant=result.quant,
        tolerance=config.quantize.tolerance,
        merged_generator=merged_gen.name,
        quantized_generator=quant_gen.name,
        n_resamples=n_resamples,
        seed=config.seed,
    )
    write_sanity_json(sanity, out_dir / "sanity.json")
    (out_dir / "export_report.md").write_text(
        _render_report(result, sanity), encoding="utf-8"
    )
    return result, sanity


def _render_report(result: ExportResult, sanity: SanityResult) -> str:
    return "\n".join(
        [
            "# M4 — optimize & export",
            "",
            f"- **Backend:** `{result.backend}`",
            f"- **Merged fp16:** `{result.merged_dir}`",
            f"- **Quantized artifact:** `{result.gguf_path}` "
            f"({result.quant}, {result.size_human})",
            f"- **Note:** {result.note}",
            "",
            render_sanity_markdown(sanity),
        ]
    )


def _write_json(obj: dict[str, object], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Fold CLI overrides into config.quantize (frozen -> dataclasses.replace)."""
    q = config.quantize
    quantize = dataclasses.replace(
        q,
        quant=args.quant or q.quant,
        gguf_path=Path(args.gguf) if args.gguf else q.gguf_path,
        merged_dir=Path(args.merged) if args.merged else q.merged_dir,
        tolerance=args.tolerance if args.tolerance is not None else q.tolerance,
        llama_cpp_dir=Path(args.llama_cpp_dir) if args.llama_cpp_dir else q.llama_cpp_dir,
    )
    return dataclasses.replace(config, quantize=quantize)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge, quantize, and sanity-check (M4).")
    parser.add_argument("--config", default="config/qa_domain.yaml")
    parser.add_argument("--mode", choices=["mock", "mlx", "cuda"], default="mock")
    parser.add_argument("--adapter", help="Adapter dir (default: train.output_dir).")
    parser.add_argument("--quant", help="Override quant level (e.g. Q4_K_M).")
    parser.add_argument("--gguf", help="Override GGUF output path.")
    parser.add_argument("--merged", help="Override merged-weights output dir.")
    parser.add_argument("--tolerance", type=float, help="Override sanity tolerance.")
    parser.add_argument("--llama-cpp-dir", help="Path to a local llama.cpp checkout.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--slice", type=int, default=DEFAULT_SLICE, dest="slice_n")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    config = _apply_overrides(load_config(args.config), args)
    adapter = Path(args.adapter) if args.adapter else config.train.output_dir
    result, sanity = run_export_and_sanity(
        config,
        mode=args.mode,
        adapter=adapter,
        out_dir=Path(args.out_dir),
        slice_n=args.slice_n,
        n_resamples=args.resamples,
    )
    print(_render_report(result, sanity))


if __name__ == "__main__":
    main()
