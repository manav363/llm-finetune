"""The export artifact: its result record and the reproducibility manifest.

An `ExportResult` says what M4 produced — where the merged fp16 weights and the
single quantized GGUF landed, the quant level, and the artifact size (the two
numbers `tasks.md` asks M4 to record). `build_manifest` renders that plus the
provenance (base model, adapter, library versions) into the `export.json` that
sits next to the artifact so the export is reproducible from config.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ExportResult:
    """Outcome of an export: the merged weights, the quantized artifact, size."""

    backend: str
    merged_dir: Path
    gguf_path: Path
    quant: str
    size_bytes: int
    note: str = ""

    @property
    def size_human(self) -> str:
        return human_size(self.size_bytes)


def file_size_bytes(path: str | Path) -> int:
    """Size of a file in bytes, or the summed size of a directory tree."""
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def human_size(n_bytes: int) -> str:
    """Render a byte count as a short human-readable string (e.g. '1.8 GB')."""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"  # pragma: no cover - unreachable, loop returns first


def build_manifest(
    result: ExportResult,
    *,
    base_model: str,
    adapter: str | None,
    seed: int,
    versions: dict[str, str],
) -> dict[str, object]:
    """Assemble the export manifest dict (pure — no I/O)."""
    return {
        "backend": result.backend,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "base_model": base_model,
        "adapter": adapter,
        "merged_dir": str(result.merged_dir),
        "gguf_path": str(result.gguf_path),
        "quant": result.quant,
        "size_bytes": result.size_bytes,
        "size_human": result.size_human,
        "versions": versions,
        "note": result.note,
    }
