"""Dataset schema and instruction formatting.

A QAExample is the atomic unit of the dataset: a domain question, optional
grounding context, and a reference answer. Validation is strict so malformed
rows fail fast at prep time rather than surfacing as silent training noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

INSTRUCTION_SYSTEM = (
    "You are a precise domain assistant. Answer using only the provided context. "
    "If the context does not contain the answer, say you don't know."
)


class SchemaError(ValueError):
    """Raised when a raw record cannot be parsed into a valid QAExample."""


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    answer: str
    context: str = ""
    category: str = ""

    @staticmethod
    def from_raw(raw: dict[str, Any]) -> QAExample:
        """Parse and validate one raw record. Raises SchemaError on any problem."""
        for key in ("id", "question", "answer"):
            if key not in raw:
                raise SchemaError(f"record missing required field '{key}': {raw!r}")
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise SchemaError(f"field '{key}' must be a non-empty string: {raw!r}")

        context = raw.get("context", "")
        if not isinstance(context, str):
            raise SchemaError(f"field 'context' must be a string: {raw!r}")

        category = raw.get("category", "")
        if not isinstance(category, str):
            raise SchemaError(f"field 'category' must be a string: {raw!r}")

        return QAExample(
            id=raw["id"].strip(),
            question=raw["question"].strip(),
            answer=raw["answer"].strip(),
            context=context.strip(),
            category=category.strip(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "category": self.category,
        }

    def to_chat(self) -> list[dict[str, str]]:
        """Render as chat messages for instruction fine-tuning (SFT)."""
        if self.context:
            user = f"Context:\n{self.context}\n\nQuestion: {self.question}"
        else:
            user = f"Question: {self.question}"
        return [
            {"role": "system", "content": INSTRUCTION_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": self.answer},
        ]


def load_jsonl(path: str) -> list[QAExample]:
    """Load and validate a JSONL file of QA records, enforcing unique ids."""
    examples: list[QAExample] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            example = QAExample.from_raw(raw)
            if example.id in seen:
                raise SchemaError(f"{path}:{lineno} duplicate id {example.id!r}")
            seen.add(example.id)
            examples.append(example)
    if not examples:
        raise SchemaError(f"{path} contained no records")
    return examples


def write_jsonl(path: str, examples: list[QAExample]) -> None:
    """Write examples to JSONL (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
