"""Tests for M6 reproducibility: versioning, registry, model card, eval-as-CI.

These run fully offline. The core claim under test is determinism — the same
config + data always yield the same run_id — plus the registry round-trip, the
card rendering the committed numbers, and the CI gate catching a regression.
"""

import dataclasses
import json

import pytest

from llm_finetune.config import (
    Config,
    DataConfig,
    LoraConfig,
    ModelConfig,
    TrainConfig,
    WandbConfig,
)
from llm_finetune.repro import version as ver
from llm_finetune.repro.gate import evaluate_gate
from llm_finetune.repro.model_card import build_model_card
from llm_finetune.repro.registry import (
    build_record,
    eval_summary_from_report,
    find_run,
    load_registry,
    register_run,
)


def _config(tmp_path, *, seed: int = 42) -> Config:
    return Config(
        seed=seed,
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
    )


def _eval_report(tuned_mean: float, delta: float = 0.0) -> dict:
    return {
        "n_items": 3,
        "judge": "heuristic-lexical",
        "verdict": "no measurable difference",
        "headline": {
            "name": "judge_overall",
            "base_mean": tuned_mean - delta,
            "tuned_mean": tuned_mean,
            "delta": {
                "delta": delta, "ci_low": -0.01, "ci_high": 0.01,
                "p_value_status": "pending-validated-judge",
            },
        },
    }


# --- versioning: determinism --------------------------------------------


def test_run_id_is_deterministic_for_same_inputs(tmp_path):
    config = _config(tmp_path)
    assert ver.run_id(config, "abc123") == ver.run_id(config, "abc123")


def test_run_id_changes_with_seed(tmp_path):
    a = ver.run_id(_config(tmp_path, seed=1), "v")
    b = ver.run_id(_config(tmp_path, seed=2), "v")
    assert a != b


def test_run_id_changes_with_data_version(tmp_path):
    config = _config(tmp_path)
    assert ver.run_id(config, "v1") != ver.run_id(config, "v2")


def test_data_version_tracks_content(tmp_path):
    p = tmp_path / "proc.jsonl"
    p.write_text('{"a": 1}\n')
    v1 = ver.data_version(p)
    p.write_text('{"a": 2}\n')
    v2 = ver.data_version(p)
    assert v1 != v2
    assert len(v1) == 12


def test_config_fingerprint_excludes_output_paths(tmp_path):
    # Same training inputs, different output dir -> identical fingerprint.
    c1 = _config(tmp_path)
    c2 = dataclasses.replace(
        c1, train=dataclasses.replace(c1.train, output_dir=tmp_path / "elsewhere")
    )
    assert ver.config_fingerprint(c1) == ver.config_fingerprint(c2)


# --- registry round-trip -------------------------------------------------


def test_register_and_find_roundtrip(tmp_path):
    config = _config(tmp_path)
    record = build_record(config, data_version="dv", note="first")
    path = tmp_path / "registry.jsonl"
    register_run(record, path)

    found = find_run(record.run_id, path)
    assert found is not None
    assert found.run_id == record.run_id
    assert found.base_model == "Qwen/Qwen2.5-3B-Instruct"


def test_register_upserts_by_run_id(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / "registry.jsonl"
    register_run(build_record(config, data_version="dv", note="v1"), path)
    register_run(build_record(config, data_version="dv", note="v2"), path)

    records = load_registry(path)
    assert len(records) == 1  # same inputs -> same run_id -> upsert, no dup
    assert records[0].note == "v2"


def test_eval_summary_from_report_extracts_headline():
    summary = eval_summary_from_report(_eval_report(0.71, delta=0.05))
    assert summary["tuned_mean"] == 0.71
    assert summary["delta"] == 0.05
    assert summary["p_value_status"] == "pending-validated-judge"


# --- model card ----------------------------------------------------------


def test_model_card_renders_key_facts(tmp_path):
    config = _config(tmp_path)
    card = build_model_card(
        config,
        data_version="deadbeef1234",
        run_id="run0123abcd",
        git_commit="abc1234",
        stats={"count": 20, "category_balance": {"api": 2}},
        eval_report=_eval_report(0.66, delta=0.09),
        export_manifest={
            "quant": "Q4_K_M", "size_human": "1.8 GB", "size_bytes": 1, "gguf_path": "m.gguf",
        },
    )
    assert "Qwen/Qwen2.5-3B-Instruct" in card
    assert "run0123abcd" in card
    assert "deadbeef1234" in card
    assert "Q4_K_M" in card
    assert "+0.090" in card  # eval delta rendered from the report
    assert "Known limitations" in card


def test_model_card_handles_missing_optional_artifacts(tmp_path):
    config = _config(tmp_path)
    card = build_model_card(
        config, data_version="v", run_id="r", git_commit=None,
        stats=None, eval_report=None, export_manifest=None,
    )
    assert "no eval report committed yet" in card
    assert "Quantized artifact" not in card  # export section omitted when absent


# --- eval-as-CI gate -----------------------------------------------------


def test_gate_passes_without_promoted_baseline():
    result = evaluate_gate(_eval_report(0.5), None)
    assert result.passed
    assert result.promoted_quality is None


def test_gate_passes_when_candidate_holds():
    result = evaluate_gate(_eval_report(0.70), _eval_report(0.71), tolerance=0.02)
    assert result.passed  # 0.70 vs 0.71 is within tolerance


def test_gate_fails_on_regression():
    result = evaluate_gate(_eval_report(0.60), _eval_report(0.71), tolerance=0.02)
    assert not result.passed
    assert result.delta is not None and result.delta < -0.02
    assert "REGRESSION" in result.reason


def test_gate_rejects_report_without_headline():
    with pytest.raises(ValueError):
        evaluate_gate({"headline": {}}, None)


def test_gate_json_reports_roundtrip(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_eval_report(0.8)))
    loaded = json.loads(path.read_text())
    assert evaluate_gate(loaded, None).candidate_quality == 0.8
