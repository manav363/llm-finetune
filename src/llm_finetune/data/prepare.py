"""Clean raw QA records into a validated, deduplicated processed dataset.

M0 keeps this deliberately small: whitespace normalization + exact-answer
dedup + schema validation. M1 extends it with near-duplicate detection and
richer cleaning. Behavior is pure/immutable: inputs are never mutated.
"""

from __future__ import annotations

import re
from pathlib import Path

from llm_finetune.schema import QAExample, load_jsonl, write_jsonl

_WS = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def clean_example(example: QAExample) -> QAExample:
    """Return a new example with normalized whitespace (never mutates input)."""
    return QAExample(
        id=example.id,
        question=_normalize_ws(example.question),
        answer=_normalize_ws(example.answer),
        context=_normalize_ws(example.context),
    )


def dedup_by_answer(examples: list[QAExample]) -> list[QAExample]:
    """Drop later examples whose (question, answer) pair already appeared."""
    seen: set[tuple[str, str]] = set()
    kept: list[QAExample] = []
    for ex in examples:
        key = (ex.question.lower(), ex.answer.lower())
        if key in seen:
            continue
        seen.add(key)
        kept.append(ex)
    return kept


def prepare(raw_path: str | Path, processed_path: str | Path) -> list[QAExample]:
    """Load raw JSONL, clean + dedup, write processed JSONL, return the result."""
    raw = load_jsonl(str(raw_path))
    cleaned = [clean_example(ex) for ex in raw]
    deduped = dedup_by_answer(cleaned)

    out = Path(processed_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out), deduped)
    return deduped
