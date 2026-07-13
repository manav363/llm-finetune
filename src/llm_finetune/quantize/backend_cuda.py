"""CUDA/HF exporter: merge the LoRA with peft, then quantize via llama.cpp.

`merge_and_unload()` fuses the adapter into the base weights; the merged fp16
model is saved with its tokenizer, then converted to an f16 GGUF and quantized
to the target level with the llama.cpp toolchain. Heavy imports are deferred and
guarded so the package still imports on a CPU-only / CI box, where `export()`
raises `BackendUnavailable` before doing any work — mirroring the CUDA training
backend. The real run awaits a GPU box with llama.cpp available.
"""

from __future__ import annotations

from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.quantize.artifact import ExportResult, file_size_bytes
from llm_finetune.quantize.llama_cpp import (
    PASSTHROUGH_QUANTS,
    convert_hf_to_gguf,
    quantize_gguf,
)
from llm_finetune.train.backend_base import BackendUnavailable


class CudaExporter:
    name = "cuda"

    def export(self, config: Config, adapter_dir: Path) -> ExportResult:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "cuda export requires the CUDA extras: pip install -r requirements/cuda.txt"
            ) from exc

        q = config.quantize
        merged_dir = Path(q.merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)

        # 1) Merge the LoRA into the base weights and save the fp16 model.
        base = AutoModelForCausalLM.from_pretrained(  # pragma: no cover - env-dependent
            config.model.name, torch_dtype=torch.float16
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        AutoTokenizer.from_pretrained(config.model.name).save_pretrained(str(merged_dir))

        # 2) Convert to f16 GGUF, then quantize to the target level.
        gguf_path = Path(q.gguf_path)
        f16_gguf = gguf_path.with_name(gguf_path.stem + ".f16.gguf")
        convert_hf_to_gguf(merged_dir, f16_gguf, q.llama_cpp_dir)
        if q.quant.upper() in PASSTHROUGH_QUANTS:
            f16_gguf.replace(gguf_path)
        else:
            quantize_gguf(f16_gguf, gguf_path, q.quant, q.llama_cpp_dir)
            f16_gguf.unlink(missing_ok=True)

        return ExportResult(  # pragma: no cover - env-dependent
            backend=self.name,
            merged_dir=merged_dir,
            gguf_path=gguf_path,
            quant=q.quant,
            size_bytes=file_size_bytes(gguf_path),
            note=f"merged fp16 -> GGUF {q.quant} via llama.cpp",
        )
