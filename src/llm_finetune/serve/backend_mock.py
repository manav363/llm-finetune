"""Mock engine: deterministic offline answers, no model, no GPU.

Serves the API cold in CI and local dev. It builds the real prompt messages (so
the request path is exercised end to end) but returns a canned, deterministic
answer instead of running a model. When context is supplied it echoes the first
sentence of that context — a stand-in for the grounded answer the real model
would give — so `POST /generate` returns something shaped like a domain answer.
"""

from __future__ import annotations

from pathlib import Path

from llm_finetune.serve.engine import DEFAULT_MAX_TOKENS, GREEDY_TEMPERATURE, build_messages


class MockEngine:
    name = "mock"

    def __init__(self, model_name: str, *, adapter: Path | None = None) -> None:
        self.model_name = model_name
        self.adapter = adapter
        self._loaded = False

    def load(self) -> None:
        # Nothing to load; touch the prompt path once so failures surface early.
        build_messages("warmup")
        self._loaded = True

    def generate(
        self,
        question: str,
        context: str = "",
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = GREEDY_TEMPERATURE,
    ) -> str:
        if not self._loaded:
            self.load()
        grounded = context.strip().split(".")[0].strip() if context.strip() else ""
        if grounded:
            return f"{grounded}. (mock answer to: {question.strip()})"
        return f"(mock answer to: {question.strip()})"
