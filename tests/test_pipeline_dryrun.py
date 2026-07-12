"""M0 acceptance: the full pipeline runs offline on the sample via the mock backend."""

from dataclasses import replace

import pytest

from llm_finetune.config import load_config
from llm_finetune.pipeline import run_pipeline
from llm_finetune.train.backend_base import BackendUnavailable
from llm_finetune.train.train import run_training, select_backend


def _mock_config(tmp_path):
    cfg = load_config("config/qa_domain.yaml")
    # Redirect all writes into tmp so tests never touch the repo working tree.
    data = replace(
        cfg.data,
        processed_path=tmp_path / "processed.jsonl",
        splits_dir=tmp_path / "splits",
    )
    train = replace(cfg.train, output_dir=tmp_path / "adapter")
    return replace(cfg, backend="mock", data=data, train=train)


def test_dry_run_end_to_end_on_sample(tmp_path):
    cfg = _mock_config(tmp_path)
    result = run_pipeline(cfg)

    assert result.backend == "mock"
    assert result.n_train + result.n_val > 0
    assert (result.adapter_dir / "adapter.json").is_file()


def test_select_backend_maps_names():
    assert select_backend("mock").name == "mock"
    assert select_backend("cuda").name == "cuda"
    assert select_backend("mlx").name == "mlx"


def test_select_backend_rejects_unknown():
    with pytest.raises(ValueError):
        select_backend("tpu")


def test_real_backends_are_stubbed_not_silently_passing(tmp_path):
    """cuda/mlx must fail loudly (unavailable or not-implemented), never no-op."""
    cfg = replace(_mock_config(tmp_path), backend="cuda")
    train_p = tmp_path / "t.jsonl"
    val_p = tmp_path / "v.jsonl"
    train_p.write_text('{"id": "1", "question": "q", "answer": "a"}\n', encoding="utf-8")
    val_p.write_text('{"id": "2", "question": "q", "answer": "a"}\n', encoding="utf-8")

    with pytest.raises((BackendUnavailable, NotImplementedError)):
        run_training(cfg, train_p, val_p)
