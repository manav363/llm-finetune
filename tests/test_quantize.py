"""Offline tests for M4 optimize/export.

The real merge + GGUF quantization need a GPU / Apple-Silicon MLX plus the
llama.cpp toolchain, so they can't run here. What we verify offline: the
artifact/manifest helpers, the full mock export path (merged marker + placeholder
GGUF + manifest), the sanity gate (reused M3 report machinery) both passing and
catching a regression, the exporter registry, and that the real exporters raise
cleanly when their deps are absent.
"""

import importlib.util
import json

import pytest

from llm_finetune.config import (
    Config,
    DataConfig,
    LoraConfig,
    ModelConfig,
    QuantizeConfig,
    TrainConfig,
    WandbConfig,
    load_config,
)
from llm_finetune.eval.generate import Generation
from llm_finetune.eval.judge import HeuristicJudge
from llm_finetune.quantize import artifact
from llm_finetune.quantize.backend_cuda import CudaExporter
from llm_finetune.quantize.backend_mlx import MlxExporter
from llm_finetune.quantize.backend_mock import GGUF_MAGIC, MockExporter
from llm_finetune.quantize.export import run_export_and_sanity
from llm_finetune.quantize.exporter import run_export, select_exporter
from llm_finetune.quantize.sanity import run_sanity_check
from llm_finetune.schema import QAExample
from llm_finetune.train.backend_base import BackendUnavailable

_RESAMPLES = 200  # small: these tests care about the gate, not CI precision


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _config(tmp_path, *, tolerance: float = 0.05) -> Config:
    return Config(
        seed=42,
        backend="mock",
        model=ModelConfig(name="Qwen/Qwen2.5-3B-Instruct", max_seq_len=512),
        data=DataConfig(
            raw_path=tmp_path / "raw.jsonl",
            processed_path=tmp_path / "proc.jsonl",
            splits_dir=tmp_path / "splits",
            val_frac=0.15,
            test_frac=0.15,
        ),
        lora=LoraConfig(r=16, alpha=32, dropout=0.05, target_modules=("q_proj", "v_proj")),
        train=TrainConfig(
            epochs=1, batch_size=4, grad_accum=1, learning_rate=2e-4,
            output_dir=tmp_path / "adapter",
        ),
        wandb=WandbConfig(enabled=False, project="llm-finetune"),
        quantize=QuantizeConfig(
            merged_dir=tmp_path / "merged",
            gguf_path=tmp_path / "model.gguf",
            quant="Q4_K_M",
            tolerance=tolerance,
        ),
    )


def _write_test_split(tmp_path, n: int = 6) -> None:
    splits = tmp_path / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    with (splits / "test.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            ex = QAExample(
                id=f"id{i}", question=f"What is {i}?", answer=f"It is {i}.", context=f"ctx {i}"
            )
            fh.write(json.dumps(ex.to_dict()) + "\n")


def _gens(answers: dict[str, str]) -> list[Generation]:
    return [
        Generation(id=k, question="q", context="c", reference="the correct answer", answer=v)
        for k, v in answers.items()
    ]


# --- artifact / manifest -------------------------------------------------


def test_human_size_scales_units():
    assert artifact.human_size(0) == "0 B"
    assert artifact.human_size(1536) == "1.5 KB"
    assert artifact.human_size(5 * 1024**3) == "5.0 GB"


def test_file_size_bytes_sums_directory(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 5)
    assert artifact.file_size_bytes(tmp_path) == 15


def test_build_manifest_records_size_and_quant():
    result = artifact.ExportResult(
        backend="mock", merged_dir=artifact.Path("m"), gguf_path=artifact.Path("g.gguf"),
        quant="Q4_K_M", size_bytes=2048, note="n",
    )
    manifest = artifact.build_manifest(
        result, base_model="base", adapter="adp", seed=7, versions={"x": "1.0"},
    )
    assert manifest["quant"] == "Q4_K_M"
    assert manifest["size_bytes"] == 2048
    assert manifest["size_human"] == "2.0 KB"
    assert manifest["base_model"] == "base"
    assert manifest["adapter"] == "adp"


# --- mock exporter -------------------------------------------------------


def test_mock_export_writes_merged_marker_and_gguf(tmp_path):
    config = _config(tmp_path)
    config.train.output_dir.mkdir(parents=True)
    (config.train.output_dir / "adapter.json").write_text("{}")

    result = MockExporter().export(config, config.train.output_dir)

    assert result.gguf_path.read_bytes().startswith(GGUF_MAGIC)
    assert result.size_bytes > 0
    assert result.quant == "Q4_K_M"
    marker = json.loads((result.merged_dir / "merged.json").read_text())
    assert marker["merged"] is False


def test_mock_export_requires_adapter(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError):
        MockExporter().export(config, tmp_path / "missing")


# --- sanity gate ---------------------------------------------------------


def test_sanity_passes_when_quantized_matches_merged():
    merged = _gens({"a": "the correct answer", "b": "the correct answer"})
    quantized = _gens({"a": "the correct answer", "b": "the correct answer"})
    result = run_sanity_check(
        merged, quantized, HeuristicJudge(),
        base_model="base", quant="Q4_K_M", tolerance=0.05,
        merged_generator="merged", quantized_generator="quant",
        n_resamples=_RESAMPLES, seed=0,
    )
    assert result.passed
    assert result.delta == pytest.approx(0.0, abs=1e-9)


def test_sanity_fails_when_quantized_collapses():
    merged = _gens({"a": "the correct answer", "b": "the correct answer"})
    quantized = _gens({"a": "garbage nonsense", "b": "totally wrong"})
    result = run_sanity_check(
        merged, quantized, HeuristicJudge(),
        base_model="base", quant="Q2_K", tolerance=0.05,
        merged_generator="merged", quantized_generator="quant",
        n_resamples=_RESAMPLES, seed=0,
    )
    assert not result.passed
    assert result.delta < -0.05
    assert "FAIL" in result.verdict


# --- end-to-end mock export + sanity ------------------------------------


def test_run_export_and_sanity_end_to_end(tmp_path):
    config = _config(tmp_path)
    config.train.output_dir.mkdir(parents=True)
    (config.train.output_dir / "adapter.json").write_text("{}")
    _write_test_split(tmp_path)

    out_dir = tmp_path / "reports"
    result, sanity = run_export_and_sanity(
        config, mode="mock", adapter=config.train.output_dir,
        out_dir=out_dir, slice_n=4, n_resamples=_RESAMPLES,
    )

    assert sanity.passed
    assert sanity.n_items == 4
    manifest = json.loads((out_dir / "export.json").read_text())
    assert manifest["quant"] == "Q4_K_M"
    assert manifest["size_bytes"] == result.size_bytes
    assert (out_dir / "sanity.json").is_file()
    assert "M4 — optimize & export" in (out_dir / "export_report.md").read_text()


# --- registry + config + guards -----------------------------------------


def test_exporter_registry_names():
    assert select_exporter("mock").name == "mock"
    assert select_exporter("mlx").name == "mlx"
    assert select_exporter("cuda").name == "cuda"
    with pytest.raises(ValueError):
        select_exporter("nope")


def test_run_export_dispatches_by_backend(tmp_path):
    config = _config(tmp_path)
    config.train.output_dir.mkdir(parents=True)
    (config.train.output_dir / "adapter.json").write_text("{}")
    result = run_export(config, config.train.output_dir)
    assert result.backend == "mock"


def test_config_defaults_quantize_when_absent():
    # An older config with no `quantize:` section still loads with M4 defaults.
    config = load_config("config/qa_domain.yaml")
    assert config.quantize.quant  # present via YAML or default
    assert 0.0 <= config.quantize.tolerance <= 1.0


@pytest.mark.skipif(_installed("torch"), reason="torch installed; guard can't be exercised")
def test_cuda_exporter_unavailable_without_torch(tmp_path):
    config = _config(tmp_path)
    config.train.output_dir.mkdir(parents=True)
    with pytest.raises(BackendUnavailable):
        CudaExporter().export(config, config.train.output_dir)


@pytest.mark.skipif(_installed("mlx"), reason="mlx installed; guard can't be exercised")
def test_mlx_exporter_unavailable_without_mlx(tmp_path):
    config = _config(tmp_path)
    config.train.output_dir.mkdir(parents=True)
    with pytest.raises(BackendUnavailable):
        MlxExporter().export(config, config.train.output_dir)
