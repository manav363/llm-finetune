"""Clean raw QA records into a validated, deduplicated processed dataset.

The prep pass is pure/immutable — inputs are never mutated — and layered:

1. whitespace normalization
2. exact dedup by (question, answer)
3. near-duplicate dedup by lexical Jaccard similarity (M1)

Near-dup detection is a transparent, dependency-free lexical baseline: it
tokenizes question+answer and drops later records whose token set is at least
`threshold` similar to a kept record. It is *not* semantic — paraphrases with
little lexical overlap will survive — but it reliably catches reworded copies
that exact dedup misses. The pass is O(n^2) in the kept set, which is fine at
the dataset sizes this pipeline targets.
"""

from __future__ import annotations

import re
from pathlib import Path

from llm_finetune.schema import QAExample, load_jsonl, write_jsonl

_WS = re.compile(r"\s+")

DEFAULT_NEAR_DUP_THRESHOLD = 0.85


def _normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def clean_example(example: QAExample) -> QAExample:
    """Return a new example with normalized whitespace (never mutates input)."""
    return QAExample(
        id=example.id,
        question=_normalize_ws(example.question),
        answer=_normalize_ws(example.answer),
        context=_normalize_ws(example.context),
        category=_normalize_ws(example.category),
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


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _WS.split(text.lower()) if t)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def near_dedup(
    examples: list[QAExample],
    *,
    threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
) -> list[QAExample]:
    """Drop later examples that are near-duplicates of an already-kept example.

    Similarity is Jaccard over the token set of ``question + " " + answer``.
    An example is dropped when its similarity to any kept example is
    ``>= threshold``. A threshold of 1.0 keeps everything but token-identical rows.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    kept: list[QAExample] = []
    kept_tokens: list[frozenset[str]] = []
    for ex in examples:
        toks = _tokens(f"{ex.question} {ex.answer}")
        if any(_jaccard(toks, kt) >= threshold for kt in kept_tokens):
            continue
        kept.append(ex)
        kept_tokens.append(toks)
    return kept


def prepare(
    raw_path: str | Path,
    processed_path: str | Path,
    *,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
) -> list[QAExample]:
    """Load raw JSONL, clean + exact/near dedup, write processed JSONL, return it."""
    raw = load_jsonl(str(raw_path))
    cleaned = [clean_example(ex) for ex in raw]
    deduped = dedup_by_answer(cleaned)
    deduped = near_dedup(deduped, threshold=near_dup_threshold)

    out = Path(processed_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out), deduped)
    return deduped
