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


def test_near_dup_threshold_defaults_when_omitted():
    cfg = parse_config(_valid_raw())
    assert cfg.data.near_dup_threshold == 0.85


def test_near_dup_threshold_is_read_from_config():
    raw = _valid_raw()
    raw["data"]["near_dup_threshold"] = 0.7
    assert parse_config(raw).data.near_dup_threshold == pytest.approx(0.7)


def test_rejects_out_of_range_near_dup_threshold():
    raw = _valid_raw()
    raw["data"]["near_dup_threshold"] = 1.5
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_loads_shipped_config_file():
    cfg = load_config("config/qa_domain.yaml")
    assert cfg.backend in ("mock", "cuda", "mlx")
    assert cfg.model.name


def test_max_steps_defaults_to_zero():
    assert parse_config(_valid_raw()).train.max_steps == 0


def test_max_steps_is_read_from_config():
    raw = _valid_raw()
    raw["train"]["max_steps"] = 2
    assert parse_config(raw).train.max_steps == 2


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


@pytest.mark.parametrize(
    "section,key,bad",
    [
        ("train", "epochs", 0),
        ("train", "batch_size", 0),
        ("train", "grad_accum", -1),
        ("train", "learning_rate", 0.0),
        ("train", "max_steps", -5),
        ("lora", "r", 0),
        ("lora", "alpha", -3),
    ],
)
def test_rejects_nonpositive_training_values(section, key, bad):
    raw = _valid_raw()
    raw[section][key] = bad
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_rejects_out_of_range_dropout():
    raw = _valid_raw()
    raw["lora"]["dropout"] = 1.0
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_rejects_string_target_modules():
    # A bare string would silently become individual characters ('q','_',...).
    raw = _valid_raw()
    raw["lora"]["target_modules"] = "q_proj"
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_rejects_empty_target_modules():
    raw = _valid_raw()
    raw["lora"]["target_modules"] = []
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_stratify_splits_defaults_false_and_reads_true():
    assert parse_config(_valid_raw()).data.stratify_splits is False
    raw = _valid_raw()
    raw["data"]["stratify_splits"] = True
    assert parse_config(raw).data.stratify_splits is True
