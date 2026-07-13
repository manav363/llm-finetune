"""Offline tests for M5 serving.

The real engines (mlx / cuda / gguf) need a model + runtime, so they can't run
here. What we verify offline: the prompt seam matches the training format, the
mock engine answers, the FastAPI app serves `POST /generate` + `/health` end to
end via TestClient, request validation, the engine registry, and that the real
engines raise cleanly when their runtimes are absent.
"""

import importlib.util

import pytest
from fastapi.testclient import TestClient

from llm_finetune.config import (
    Config,
    DataConfig,
    LoraConfig,
    ModelConfig,
    TrainConfig,
    WandbConfig,
)
from llm_finetune.schema import INSTRUCTION_SYSTEM
from llm_finetune.serve import engine as engine_mod
from llm_finetune.serve.api import create_app
from llm_finetune.serve.backend_cuda import CudaEngine
from llm_finetune.serve.backend_gguf import GgufEngine
from llm_finetune.serve.backend_mlx import MlxEngine
from llm_finetune.serve.backend_mock import MockEngine
from llm_finetune.serve.registry import select_engine
from llm_finetune.train.backend_base import BackendUnavailable


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _config(tmp_path) -> Config:
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
        lora=LoraConfig(r=16, alpha=32, dropout=0.05, target_modules=("q_proj",)),
        train=TrainConfig(
            epochs=1, batch_size=4, grad_accum=1, learning_rate=2e-4,
            output_dir=tmp_path / "adapter",
        ),
        wandb=WandbConfig(enabled=False, project="llm-finetune"),
    )


# --- prompt seam ---------------------------------------------------------


def test_build_messages_matches_training_format():
    msgs = engine_mod.build_messages("What is X?", "X is a thing.")
    assert msgs[0] == {"role": "system", "content": INSTRUCTION_SYSTEM}
    assert msgs[1]["role"] == "user"
    assert "X is a thing." in msgs[1]["content"]
    assert "What is X?" in msgs[1]["content"]


def test_build_messages_without_context_omits_context_block():
    msgs = engine_mod.build_messages("Just a question?")
    assert msgs[1]["content"].startswith("Question:")


# --- mock engine ---------------------------------------------------------


def test_mock_engine_grounds_answer_in_context():
    eng = MockEngine("some-model")
    eng.load()
    answer = eng.generate("What color?", "The sky is blue. More text.")
    assert "The sky is blue" in answer
    assert "What color?" in answer


def test_mock_engine_without_context():
    eng = MockEngine("some-model")
    assert eng.generate("Hello?") == "(mock answer to: Hello?)"


# --- FastAPI app (the acceptance path) -----------------------------------


def _client() -> TestClient:
    return TestClient(create_app(engine=MockEngine("test-model")))


def test_health_ok():
    with _client() as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "engine": "mock"}


def test_generate_returns_domain_answer():
    with _client() as client:
        resp = client.post(
            "/generate",
            json={"question": "What is the capital?", "context": "Paris is the capital."},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"] == "mock"
    assert "Paris is the capital" in body["answer"]


def test_generate_rejects_empty_question():
    with _client() as client:
        resp = client.post("/generate", json={"question": ""})
    assert resp.status_code == 422


def test_generate_surfaces_engine_failure_as_503():
    class BoomEngine:
        name = "boom"

        def load(self) -> None:
            pass

        def generate(self, question, context="", *, max_tokens=256, temperature=0.0) -> str:
            raise RuntimeError("model exploded")

    with TestClient(create_app(engine=BoomEngine())) as client:
        resp = client.post("/generate", json={"question": "hi"})
    assert resp.status_code == 503
    assert "model exploded" in resp.json()["detail"]


# --- registry + guards ---------------------------------------------------


def test_registry_builds_each_engine(tmp_path):
    config = _config(tmp_path)
    assert select_engine("mock", config, adapter=None).name == "mock"
    assert select_engine("mlx", config, adapter=tmp_path / "a").name == "mlx"
    assert select_engine("cuda", config, adapter=tmp_path / "a").name == "cuda"
    assert select_engine("gguf", config, adapter=tmp_path / "m.gguf").name == "gguf"


def test_registry_rejects_unknown_and_gguf_without_path(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError):
        select_engine("nope", config, adapter=None)
    with pytest.raises(ValueError):
        select_engine("gguf", config, adapter=None)


def test_gguf_engine_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        GgufEngine(tmp_path / "missing.gguf").load()


@pytest.mark.skipif(_installed("mlx_lm"), reason="mlx installed; guard can't be exercised")
def test_mlx_engine_unavailable_without_mlx():
    with pytest.raises(BackendUnavailable):
        MlxEngine("some-model").load()


@pytest.mark.skipif(_installed("torch"), reason="torch installed; guard can't be exercised")
def test_cuda_engine_unavailable_without_torch():
    with pytest.raises(BackendUnavailable):
        CudaEngine("some-model").load()
