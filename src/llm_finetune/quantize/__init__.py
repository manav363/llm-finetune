"""M4 optimize/export: merge the winning LoRA, quantize to GGUF, sanity-check.

`export.py` is the entrypoint. Like the training layer, the concrete work sits
behind an `Exporter` seam selected by config: `mock` runs fully offline for CI,
while `mlx` and `cuda` do the real merge + GGUF quantization behind runtime
guards. `sanity.py` re-runs a slice of the M3 evaluation to prove quantization
didn't collapse quality versus the merged fp16 model.
"""
