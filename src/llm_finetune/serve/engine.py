"""The inference-engine contract and the prompt seam shared by all engines.

An `InferenceEngine` loads a model once (`load`) and answers many single-prompt
requests (`generate`). Prompts are built through `build_messages`, which reuses
the exact system + user formatting the model was fine-tuned on (`QAExample.
to_prompt_messages`) so serving and training stay in lockstep. Generation is
greedy (temp=0) by default for reproducibility, matching the eval harness.
"""

from __future__ import annotations

from typing import Protocol

from llm_finetune.schema import QAExample

# Greedy by default so a given (question, context) maps to a stable answer,
# mirroring the eval harness. Callers may override per request.
GREEDY_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256


def build_messages(question: str, context: str = "") -> list[dict[str, str]]:
    """Render the system + user turns for a request (no answer) — for inference.

    Reuses `QAExample.to_prompt_messages` so the served prompt is byte-identical
    to the fine-tuning prompt format.
    """
    example = QAExample(id="request", question=question, answer="", context=context)
    return example.to_prompt_messages()


class InferenceEngine(Protocol):
    """A loaded, reusable model that answers one prompt at a time."""

    name: str

    def load(self) -> None:
        """Load the model into memory. Idempotent; called once at startup."""
        ...

    def generate(
        self,
        question: str,
        context: str = "",
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = GREEDY_TEMPERATURE,
    ) -> str:
        """Return the model's answer for a single question (+ optional context)."""
        ...
