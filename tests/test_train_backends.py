"""Offline tests for M2 training: the shared seams plus backend availability.

The real QLoRA / MLX loops need a GPU or Apple-Silicon MLX plus a model
download, so they cannot run here. What we *can* verify offline: chat
formatting, deterministic seeding, the step-budget math, the reproducibility
record, and that each real backend raises BackendUnavailable when its runtime
deps are absent (which they are in CI / on this box).
"""

import importlib.util
import json
import random

import pytest

from llm_finetune.config import (
    Config,
    DataConfig,
    LoraConfig,
    ModelConfig,
    TrainConfig,
    WandbConfig,
)
from llm_finetune.schema import QAExample
from llm_finetune.train import backend_common as common
from llm_finetune.train.backend_base import BackendUnavailable
from llm_finetune.train.backend_cuda import CudaBackend
from llm_finetune.train.backend_mlx import MlxBackend
from llm_finetune.train.train import select_backend


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _config(tmp_path, *, max_steps: int = 0, wandb: bool = False) -> Config:
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
            epochs=3,
            batch_size=4,
            grad_accum=4,
            learning_rate=2e-4,
            output_dir=tmp_path / "adapter",
            max_steps=max_steps,
        ),
        wandb=WandbConfig(enabled=wandb, project="llm-finetune"),
    )


def _examples(n: int) -> list[QAExample]:
    return [
        QAExample(id=f"id{i}", question=f"q{i}", answer=f"a{i}", context="ctx")
        for i in range(n)
    ]


def test_to_chat_records_wraps_messages():
    records = common.to_chat_records(_examples(2))
    assert len(records) == 2
    roles = [m["role"] for m in records[0]["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_write_chat_jsonl_roundtrips(tmp_path):
    path = common.write_chat_jsonl(_examples(3), tmp_path / "train.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["messages"][-1]["content"] == "a0"


def test_set_all_seeds_is_deterministic():
    common.set_all_seeds(123)
    a = [random.random() for _ in range(5)]
    common.set_all_seeds(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_training_iters_uses_epochs_by_default():
    # 10 examples, batch 4 -> ceil = 3 steps/epoch, 2 epochs -> 6
    assert common.training_iters(10, 4, 2, 0) == 6


def test_training_iters_respects_max_steps():
    assert common.training_iters(10, 4, 5, 2) == 2


@pytest.mark.parametrize("bad", [(0, 4), (10, 0)])
def test_training_iters_rejects_nonpositive(bad):
    n_train, batch = bad
    with pytest.raises(ValueError):
        common.training_iters(n_train, batch, 1, 0)


def test_library_versions_marks_missing():
    versions = common.library_versions(["pytest", "definitely-not-a-real-pkg-xyz"])
    assert versions["pytest"] != "not-installed"
    assert versions["definitely-not-a-real-pkg-xyz"] == "not-installed"


def test_write_run_metadata_captures_config(tmp_path):
    config = _config(tmp_path, max_steps=2, wandb=True)
    path = common.write_run_metadata(
        config,
        backend="cuda",
        steps=2,
        n_train=14,
        n_val=3,
        package_names=["pytest"],
        path=tmp_path / "run.json",
    )
    meta = json.loads(path.read_text())
    assert meta["backend"] == "cuda"
    assert meta["seed"] == 42
    assert meta["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert meta["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert meta["train"]["resolved_steps"] == 2
    assert meta["wandb"]["enabled"] is True
    assert "pytest" in meta["versions"]


@pytest.mark.skipif(_installed("torch"), reason="torch installed; guard can't be exercised")
def test_cuda_backend_unavailable_without_torch(tmp_path):
    # With the CUDA extras absent, the guarded import must surface cleanly.
    with pytest.raises(BackendUnavailable):
        CudaBackend().train(_config(tmp_path), tmp_path / "t.jsonl", tmp_path / "v.jsonl")


@pytest.mark.skipif(_installed("mlx"), reason="mlx installed; guard can't be exercised")
def test_mlx_backend_unavailable_without_mlx(tmp_path):
    with pytest.raises(BackendUnavailable):
        MlxBackend().train(_config(tmp_path), tmp_path / "t.jsonl", tmp_path / "v.jsonl")


def test_backend_names_match_registry():
    assert select_backend("cuda").name == "cuda"
    assert select_backend("mlx").name == "mlx"
    assert select_backend("mock").name == "mock"
