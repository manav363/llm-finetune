"""The exporter contract and dispatcher — the M4 analogue of `train.py`.

Every exporter (mock, mlx, cuda) turns a trained adapter into a merged fp16
model plus one quantized GGUF artifact. `select_exporter` is the single place
that maps a backend name to a concrete implementation; `run_export` dispatches
to it. This mirrors the training seam so the same config `backend` field drives
where the export runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from llm_finetune.config import Config
from llm_finetune.quantize.artifact import ExportResult


class Exporter(Protocol):
    """Merges an adapter into the base weights and quantizes to GGUF.

    Implementations read output paths and the quant level from
    ``config.quantize`` and must not mutate their inputs.
    """

    name: str

    def export(self, config: Config, adapter_dir: Path) -> ExportResult:
        ...


def select_exporter(name: str) -> Exporter:
    """Map a config backend string to a concrete exporter instance."""
    # Imported here to keep heavy/guarded backends off the module import path.
    from llm_finetune.quantize.backend_cuda import CudaExporter
    from llm_finetune.quantize.backend_mlx import MlxExporter
    from llm_finetune.quantize.backend_mock import MockExporter

    exporters: dict[str, Exporter] = {
        "mock": MockExporter(),
        "cuda": CudaExporter(),
        "mlx": MlxExporter(),
    }
    if name not in exporters:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(exporters)}")
    return exporters[name]


def run_export(config: Config, adapter_dir: Path) -> ExportResult:
    """Dispatch to the configured backend's exporter and run it."""
    exporter = select_exporter(config.backend)
    return exporter.export(config, adapter_dir)
