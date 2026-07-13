"""Deterministic versioning — the backbone of "reproducible from config".

A run is named by a `run_id` computed purely from the inputs that determine its
result: the reproducibility-relevant config fields and a content hash of the
training data. Same config + same data -> same `run_id`, on any machine, so a
cold clone can recompute the id recorded in the registry and know it's looking
at the same run. Git commit is captured too (best-effort) for provenance.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from llm_finetune.config import Config

_SHORT = 12  # hex chars kept for short ids — 48 bits, ample for this scale


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Full SHA-256 hex digest of a file's bytes."""
    return _sha256_hex(Path(path).read_bytes())


def data_version(processed_path: str | Path) -> str:
    """Short content hash of the processed dataset — the dataset's version."""
    return file_sha256(processed_path)[:_SHORT]


def config_fingerprint(config: Config) -> dict[str, object]:
    """The reproducibility-relevant config fields (excludes output paths/wandb).

    Two configs with the same fingerprint should produce the same model, so only
    inputs that affect the result are included — not where artifacts are written.
    """
    return {
        "seed": config.seed,
        "backend": config.backend,
        "base_model": config.model.name,
        "max_seq_len": config.model.max_seq_len,
        "lora": {
            "r": config.lora.r,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "target_modules": list(config.lora.target_modules),
        },
        "train": {
            "epochs": config.train.epochs,
            "batch_size": config.train.batch_size,
            "grad_accum": config.train.grad_accum,
            "learning_rate": config.train.learning_rate,
            "max_steps": config.train.max_steps,
        },
    }


def _canonical(obj: object) -> str:
    """Stable, sorted JSON string for hashing (no whitespace ambiguity)."""
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def fingerprint_hash(fingerprint: dict[str, object]) -> str:
    """Short deterministic hash of a config fingerprint."""
    return _sha256_hex(_canonical(fingerprint).encode("utf-8"))[:_SHORT]


def run_id(config: Config, data_version: str) -> str:
    """Deterministic run id from the config fingerprint + the data version."""
    payload = {"config": config_fingerprint(config), "data_version": data_version}
    return _sha256_hex(_canonical(payload).encode("utf-8"))[:_SHORT]


def git_commit() -> str | None:
    """Current git commit hash, or None if unavailable (best-effort)."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):  # pragma: no cover - git almost always present
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
