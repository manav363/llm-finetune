import pytest

from llm_finetune.config import ConfigError, load_config, parse_config


def _valid_raw() -> dict:
    return {
        "seed": 42,
        "backend": "mock",
        "model": {"name": "Qwen/Qwen2.5-3B-Instruct", "max_seq_len": 1024},
        "data": {
            "raw_path": "data/sample/domain_qa.jsonl",
            "processed_path": "data/processed/domain_qa.jsonl",
            "splits_dir": "data/processed/splits",
            "val_frac": 0.15,
            "test_frac": 0.15,
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj"]},
        "train": {
            "epochs": 3,
            "batch_size": 4,
            "grad_accum": 4,
            "learning_rate": 0.0002,
            "output_dir": "outputs/adapter",
        },
        "wandb": {"enabled": False, "project": "llm-finetune"},
    }


def test_parses_valid_config():
    cfg = parse_config(_valid_raw())
    assert cfg.backend == "mock"
    assert cfg.lora.target_modules == ("q_proj",)
    assert cfg.data.train_frac == pytest.approx(0.70)


def test_loads_shipped_config_file():
    cfg = load_config("config/qa_domain.yaml")
    assert cfg.backend in ("mock", "cuda", "mlx")
    assert cfg.model.name


def test_rejects_unknown_backend():
    raw = _valid_raw()
    raw["backend"] = "tpu"
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_rejects_missing_section():
    raw = _valid_raw()
    del raw["lora"]
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_rejects_impossible_split_fractions():
    raw = _valid_raw()
    raw["data"]["val_frac"] = 0.6
    raw["data"]["test_frac"] = 0.6
    with pytest.raises(ConfigError):
        parse_config(raw)
