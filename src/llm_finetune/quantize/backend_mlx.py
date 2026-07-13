"""MLX exporter: fuse the LoRA into the base weights and export GGUF.

`mlx_lm fuse` both merges the adapter and writes an f16 GGUF in one call. mlx's
GGUF export is f16-only, so a smaller target quant is finished with llama.cpp's
`llama-quantize`. Heavy imports are deferred and guarded: on a non-Mac / CI box
`export()` raises `BackendUnavailable` before doing any work, mirroring the MLX
training backend. The real run awaits Apple-Silicon MLX.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.quantize.artifact import ExportResult, file_size_bytes
from llm_finetune.quantize.llama_cpp import PASSTHROUGH_QUANTS, quantize_gguf
from llm_finetune.train.backend_base import BackendUnavailable

_F16_GGUF_NAME = "ggml-model-f16.gguf"


class MlxExporter:
    name = "mlx"

    def export(self, config: Config, adapter_dir: Path) -> ExportResult:
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "mlx export requires the Apple-Silicon extras: "
                "pip install -r requirements/mac.txt"
            ) from exc

        q = config.quantize
        merged_dir = Path(q.merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)

        # 1) Fuse the adapter into the base weights and export an f16 GGUF.
        cmd = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", config.model.name,
            "--adapter-path", str(adapter_dir),
            "--save-path", str(merged_dir),
            "--export-gguf",
            "--gguf-path", _F16_GGUF_NAME,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
        if result.returncode != 0:  # pragma: no cover - env-dependent
            raise RuntimeError(
                f"mlx-lm fuse failed (exit {result.returncode}):\n{result.stderr}"
            )

        # 2) mlx exports f16; quantize to the target level unless it's f16/f32.
        f16_gguf = merged_dir / _F16_GGUF_NAME
        gguf_path = Path(q.gguf_path)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        if q.quant.upper() in PASSTHROUGH_QUANTS:
            f16_gguf.replace(gguf_path)
        else:  # pragma: no cover - env-dependent
            quantize_gguf(f16_gguf, gguf_path, q.quant, q.llama_cpp_dir)

        return ExportResult(
            backend=self.name,
            merged_dir=merged_dir,
            gguf_path=gguf_path,
            quant=q.quant,
            size_bytes=file_size_bytes(gguf_path),
            note=f"mlx fuse -> f16 GGUF -> {q.quant}",
        )
