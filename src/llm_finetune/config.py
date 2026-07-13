"""Typed, validated configuration loaded from YAML.

Validation happens once, at load time (a system boundary). Everything
downstream receives frozen dataclasses and can trust the values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_BACKENDS = ("mock", "cuda", "mlx")


class ConfigError(ValueError):
    """Raised when a config file is missing required fields or has bad values."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    max_seq_len: int


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    processed_path: Path
    splits_dir: Path
    val_frac: float
    test_frac: float
    near_dup_threshold: float = 0.85

    @property
    def train_frac(self) -> float:
        return 1.0 - self.val_frac - self.test_frac


@dataclass(frozen=True)
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    output_dir: Path
    max_steps: int = 0  # 0 = run full epochs; >0 caps steps (smoke runs)


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str


@dataclass(frozen=True)
class QuantizeConfig:
    """M4 optimize/export settings: where merged + quantized artifacts land,
    the llama.cpp quant level, and the max quality drop the quantized model may
    show versus the merged fp16 model in the sanity check."""

    merged_dir: Path
    gguf_path: Path
    quant: str
    tolerance: float
    # Path to a local llama.cpp checkout for the CUDA/HF GGUF-conversion step.
    # None -> discover its scripts/binaries on PATH. Ignored by the MLX exporter
    # (mlx-lm fuses + exports GGUF itself) and the offline mock exporter.
    llama_cpp_dir: Path | None = None


def _default_quantize() -> QuantizeConfig:
    return QuantizeConfig(
        merged_dir=Path("outputs/merged"),
        gguf_path=Path("outputs/model.gguf"),
        quant="Q4_K_M",
        tolerance=0.05,
        llama_cpp_dir=None,
    )


@dataclass(frozen=True)
class Config:
    seed: int
    backend: str
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    train: TrainConfig
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(False, "llm-finetune"))
    quantize: QuantizeConfig = field(default_factory=_default_quantize)


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required field '{section}.{key}'")
    return mapping[key]


def _as_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = _require(raw, key, "<root>")
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping, got {type(value).__name__}")
    return value


def parse_config(raw: dict[str, Any]) -> Config:
    """Validate a raw dict into a frozen Config. Raises ConfigError on any problem."""
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    backend = _require(raw, "backend", "<root>")
    if backend not in VALID_BACKENDS:
        raise ConfigError(f"backend must be one of {VALID_BACKENDS}, got {backend!r}")

    model_raw = _as_section(raw, "model")
    data_raw = _as_section(raw, "data")
    lora_raw = _as_section(raw, "lora")
    train_raw = _as_section(raw, "train")
    wandb_raw = raw.get("wandb", {"enabled": False, "project": "llm-finetune"})

    val_frac = float(_require(data_raw, "val_frac", "data"))
    test_frac = float(_require(data_raw, "test_frac", "data"))
    if not (0.0 < val_frac < 1.0) or not (0.0 < test_frac < 1.0):
        raise ConfigError("data.val_frac and data.test_frac must each be in (0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ConfigError("data.val_frac + data.test_frac must be < 1.0 (need a train split)")

    near_dup_threshold = float(data_raw.get("near_dup_threshold", 0.85))
    if not 0.0 < near_dup_threshold <= 1.0:
        raise ConfigError("data.near_dup_threshold must be in (0, 1]")

    quantize = _parse_quantize(raw.get("quantize", {}))

    return Config(
        seed=int(_require(raw, "seed", "<root>")),
        backend=str(backend),
        model=ModelConfig(
            name=str(_require(model_raw, "name", "model")),
            max_seq_len=int(_require(model_raw, "max_seq_len", "model")),
        ),
        data=DataConfig(
            raw_path=Path(_require(data_raw, "raw_path", "data")),
            processed_path=Path(_require(data_raw, "processed_path", "data")),
            splits_dir=Path(_require(data_raw, "splits_dir", "data")),
            val_frac=val_frac,
            test_frac=test_frac,
            near_dup_threshold=near_dup_threshold,
        ),
        lora=LoraConfig(
            r=int(_require(lora_raw, "r", "lora")),
            alpha=int(_require(lora_raw, "alpha", "lora")),
            dropout=float(_require(lora_raw, "dropout", "lora")),
            target_modules=tuple(_require(lora_raw, "target_modules", "lora")),
        ),
        train=TrainConfig(
            epochs=int(_require(train_raw, "epochs", "train")),
            batch_size=int(_require(train_raw, "batch_size", "train")),
            grad_accum=int(_require(train_raw, "grad_accum", "train")),
            learning_rate=float(_require(train_raw, "learning_rate", "train")),
            output_dir=Path(_require(train_raw, "output_dir", "train")),
            max_steps=int(train_raw.get("max_steps", 0)),
        ),
        wandb=WandbConfig(
            enabled=bool(wandb_raw.get("enabled", False)),
            project=str(wandb_raw.get("project", "llm-finetune")),
        ),
        quantize=quantize,
    )


def _parse_quantize(raw: Any) -> QuantizeConfig:
    """Parse the optional `quantize` section, filling M4 defaults."""
    if not isinstance(raw, dict):
        raise ConfigError("'quantize' must be a mapping")
    default = _default_quantize()
    tolerance = float(raw.get("tolerance", default.tolerance))
    if not 0.0 <= tolerance <= 1.0:
        raise ConfigError("quantize.tolerance must be in [0, 1]")
    llama_cpp_dir = raw.get("llama_cpp_dir")
    return QuantizeConfig(
        merged_dir=Path(raw.get("merged_dir", default.merged_dir)),
        gguf_path=Path(raw.get("gguf_path", default.gguf_path)),
        quant=str(raw.get("quant", default.quant)),
        tolerance=tolerance,
        llama_cpp_dir=Path(llama_cpp_dir) if llama_cpp_dir else None,
    )


def load_config(path: str | Path) -> Config:
    """Read and validate a YAML config file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return parse_config(raw)
