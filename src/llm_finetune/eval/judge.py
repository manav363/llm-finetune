"""Answer judging along correctness / faithfulness / relevance.

`tasks.md` calls for scoring with *the AI Eval Pipeline's validated judge*. That
project isn't built yet, so this module ships:

* `Judge` — the protocol the validated judge will implement, and
* `HeuristicJudge` — a deterministic, offline, lexical **placeholder** so the
  pipeline runs cold and the report has scores to show.

The heuristic is explicitly NOT the validated judge: it approximates each
dimension with token overlap. Any significance claim stays flagged as pending
(see `bootstrap.P_VALUE_PENDING`) until the real judge is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from llm_finetune.eval.metrics import _tokens, rouge_l, token_f1

DIMENSIONS = ("correctness", "faithfulness", "relevance")


@dataclass(frozen=True)
class JudgeScore:
    """Per-dimension scores in [0, 1]; higher is better."""

    correctness: float
    faithfulness: float
    relevance: float

    def as_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
        }


class Judge(Protocol):
    """Scores an answer against its question / context / reference."""

    name: str

    def score(
        self, *, question: str, context: str, reference: str, answer: str
    ) -> JudgeScore:
        ...


def _coverage(source: str, target: str) -> float:
    """Fraction of `source` content tokens that appear in `target` (0..1)."""
    src = _tokens(source)
    if not src:
        return 0.0
    tgt = set(_tokens(target))
    hits = sum(1 for tok in src if tok in tgt)
    return hits / len(src)


class HeuristicJudge:
    """Deterministic lexical placeholder for the validated judge.

    * correctness — token-F1 of the answer against the reference answer.
    * faithfulness — how much of the answer is grounded in the context
      (answer-token coverage by context); 1.0 when there is no context to
      contradict, since the task allows context-free items.
    * relevance — LCS overlap (ROUGE-L) between the answer and the question.
    """

    name = "heuristic-placeholder"

    def score(
        self, *, question: str, context: str, reference: str, answer: str
    ) -> JudgeScore:
        correctness = token_f1(answer, reference)
        faithfulness = _coverage(answer, context) if context.strip() else 1.0
        relevance = rouge_l(answer, question)
        return JudgeScore(
            correctness=correctness,
            faithfulness=faithfulness,
            relevance=relevance,
        )
