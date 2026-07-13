"""Thin wrappers over the external llama.cpp GGUF toolchain.

GGUF conversion + quantization live in llama.cpp, not in any pip package, so the
real exporters shell out to it: `convert_hf_to_gguf.py` turns a merged HF model
into an f16 GGUF, and `llama-quantize` compresses that to the target quant
level. These helpers locate the tools (on PATH or under a configured llama.cpp
checkout) and run them. They are env-dependent and not exercised by the offline
suite; a missing toolchain raises a clear, actionable error.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Quant levels that are just the direct GGUF conversion output — no separate
# llama-quantize pass is needed.
PASSTHROUGH_QUANTS = frozenset({"F16", "FP16", "F32", "FP32"})

_CONVERT_SCRIPTS = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
_QUANTIZE_BINARIES = ("llama-quantize", "quantize")


class LlamaCppUnavailable(RuntimeError):
    """Raised when the llama.cpp GGUF toolchain cannot be located."""


def _find(candidates: tuple[str, ...], llama_cpp_dir: Path | None) -> Path | None:
    if llama_cpp_dir is not None:
        for name in candidates:
            for hit in Path(llama_cpp_dir).rglob(name):
                if hit.is_file():
                    return hit
    for name in candidates:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _run(cmd: list[str]) -> None:  # pragma: no cover - env-dependent
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def convert_hf_to_gguf(
    model_dir: Path, out_gguf: Path, llama_cpp_dir: Path | None
) -> Path:  # pragma: no cover - env-dependent
    """Convert a merged HF model directory to an f16 GGUF file."""
    script = _find(_CONVERT_SCRIPTS, llama_cpp_dir)
    if script is None:
        raise LlamaCppUnavailable(
            "llama.cpp convert_hf_to_gguf.py not found. Clone llama.cpp and set "
            "quantize.llama_cpp_dir in the config (or put its scripts on PATH)."
        )
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    import sys

    _run([sys.executable, str(script), str(model_dir), "--outfile", str(out_gguf),
          "--outtype", "f16"])
    return out_gguf


def quantize_gguf(
    in_gguf: Path, out_gguf: Path, quant: str, llama_cpp_dir: Path | None
) -> Path:  # pragma: no cover - env-dependent
    """Quantize an f16 GGUF to the target quant level via llama-quantize."""
    binary = _find(_QUANTIZE_BINARIES, llama_cpp_dir)
    if binary is None:
        raise LlamaCppUnavailable(
            "llama.cpp llama-quantize not found. Build llama.cpp and set "
            "quantize.llama_cpp_dir (or put llama-quantize on PATH)."
        )
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    _run([str(binary), str(in_gguf), str(out_gguf), quant])
    return out_gguf
