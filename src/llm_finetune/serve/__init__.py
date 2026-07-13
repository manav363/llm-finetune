"""M5 serving: a FastAPI inference endpoint over the fine-tuned model.

`api.py` is the app. Like training and export, the concrete inference work sits
behind an `InferenceEngine` seam selected by config/env: `mock` answers offline
for CI, while `mlx`, `cuda` (vLLM/transformers), and `gguf` (llama.cpp) load the
real model once at startup. The engine is a singleton — loaded on first use and
reused across requests.
"""
