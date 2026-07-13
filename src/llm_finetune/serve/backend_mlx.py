"""MLX engine: Apple-Silicon serving via mlx-lm, optionally with a LoRA adapter.

Loads the model + tokenizer once (`load`) and reuses them across requests. Heavy
imports are deferred and guarded so importing the app on a non-Mac box is safe;
`load`/`generate` raise `BackendUnavailable` there. Real serving awaits MLX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_finetune.serve.engine import DEFAULT_MAX_TOKENS, GREEDY_TEMPERATURE, build_messages
from llm_finetune.train.backend_base import BackendUnavailable


class MlxEngine:
    name = "mlx"

    def __init__(self, model_name: str, *, adapter: Path | None = None) -> None:
        self.model_name = model_name
        self.adapter = adapter
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        try:
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "mlx serving requires the Apple-Silicon extras: "
                "pip install -r requirements/mac.txt"
            ) from exc
        adapter = str(self.adapter) if self.adapter else None
        self._model, self._tokenizer = load(  # pragma: no cover - env-dependent
            self.model_name, adapter_path=adapter
        )

    def generate(
        self,
        question: str,
        context: str = "",
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = GREEDY_TEMPERATURE,
    ) -> str:
        if self._model is None:
            self.load()
        from mlx_lm import generate  # pragma: no cover - env-dependent
        from mlx_lm.sample_utils import make_sampler

        prompt = self._tokenizer.apply_chat_template(  # pragma: no cover - env-dependent
            build_messages(question, context), add_generation_prompt=True, tokenize=False
        )
        text: str = generate(  # pragma: no cover - env-dependent
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        return text.strip()  # pragma: no cover - env-dependent
