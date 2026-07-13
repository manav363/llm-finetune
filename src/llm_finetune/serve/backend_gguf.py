"""GGUF engine: serve the quantized M4 artifact via llama.cpp (llama-cpp-python).

The natural way to serve on a Mac (or any CPU box) once the adapter is merged
and quantized to GGUF. Loads the `.gguf` once and reuses it. The runtime is an
optional extra, so the import is deferred and guarded; without it, `load` raises
`BackendUnavailable`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_finetune.serve.engine import DEFAULT_MAX_TOKENS, GREEDY_TEMPERATURE, build_messages
from llm_finetune.train.backend_base import BackendUnavailable


class GgufEngine:
    name = "gguf"

    def __init__(self, gguf_path: Path) -> None:
        self.gguf_path = gguf_path
        self._llm: Any = None

    def load(self) -> None:
        if not Path(self.gguf_path).is_file():
            raise FileNotFoundError(f"GGUF artifact not found: {self.gguf_path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "gguf serving requires llama-cpp-python: pip install llama-cpp-python"
            ) from exc
        self._llm = Llama(  # pragma: no cover - env-dependent
            model_path=str(self.gguf_path), n_ctx=0, verbose=False
        )

    def generate(
        self,
        question: str,
        context: str = "",
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = GREEDY_TEMPERATURE,
    ) -> str:
        if self._llm is None:
            self.load()
        out = self._llm.create_chat_completion(  # pragma: no cover - env-dependent
            messages=build_messages(question, context),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content: str = out["choices"][0]["message"]["content"] or ""  # pragma: no cover
        return content.strip()  # pragma: no cover - env-dependent
