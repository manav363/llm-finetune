"""Mock exporter: no real weights, no GPU, no llama.cpp — the offline path.

It proves the M4 wiring end-to-end for CI: it checks the adapter exists, writes
a `merged.json` marker where the fused weights would land, and writes a tiny
deterministic GGUF-shaped file (a real `GGUF` magic header plus the recorded
quant tag) so the size + quant level are concrete and the sanity check has an
artifact to point at. No quantization actually happens.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.quantize.artifact import ExportResult, file_size_bytes

GGUF_MAGIC = b"GGUF"


class MockExporter:
    name = "mock"

    def export(self, config: Config, adapter_dir: Path) -> ExportResult:
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"adapter dir not found: {adapter_dir} (train an adapter first)"
            )

        q = config.quantize

        merged_dir = Path(q.merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "merged.json").write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "base_model": config.model.name,
                    "adapter": str(adapter_dir),
                    "merged": False,
                    "note": "mock merge — no weights were fused",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        gguf_path = Path(q.gguf_path)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        # A deterministic, GGUF-shaped placeholder: magic + quant tag + filler.
        payload = GGUF_MAGIC + b"\x00" + q.quant.encode("utf-8") + b"\x00" + b"mock" * 64
        gguf_path.write_bytes(payload)

        return ExportResult(
            backend=self.name,
            merged_dir=merged_dir,
            gguf_path=gguf_path,
            quant=q.quant,
            size_bytes=file_size_bytes(gguf_path),
            note="mock export: wrote merged.json marker + placeholder GGUF, no quantization",
        )
