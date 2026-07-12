"""End-to-end offline evaluation: mock generation -> judge -> bootstrap -> report."""

import json

import pytest

from llm_finetune.eval.generate import Generation, MockGenerator
from llm_finetune.eval.judge import HeuristicJudge, JudgeScore
from llm_finetune.eval.report import build_report, render_markdown, write_report
from llm_finetune.schema import QAExample


def _examples() -> list[QAExample]:
    return [
        QAExample(id="1", question="What is the file limit?", answer="Two gigabytes.",
                  context="The limit is two gigabytes."),
        QAExample(id="2", question="How long are logs kept?", answer="Thirty days.",
                  context="Logs are kept thirty days."),
        QAExample(id="3", question="Which auth is supported?", answer="API keys and OAuth.",
                  context="We support API keys and OAuth."),
    ]


def test_heuristic_judge_scores_in_unit_range():
    ex = _examples()[0]
    score = HeuristicJudge().score(
        question=ex.question, context=ex.context, reference=ex.answer, answer=ex.answer
    )
    for value in score.as_dict().values():
        assert 0.0 <= value <= 1.0


def test_heuristic_judge_faithfulness_one_without_context():
    score = HeuristicJudge().score(
        question="q", context="", reference="a", answer="totally unrelated"
    )
    assert score.faithfulness == 1.0


def test_judge_score_as_dict_keys():
    s = JudgeScore(correctness=0.1, faithfulness=0.2, relevance=0.3)
    assert s.as_dict() == {"correctness": 0.1, "faithfulness": 0.2, "relevance": 0.3}


def test_build_report_detects_improvement():
    examples = _examples()
    base = MockGenerator(transform=lambda ex: ex.question).generate(examples)  # weak
    tuned = MockGenerator(transform=lambda ex: ex.answer).generate(examples)  # perfect
    report = build_report(
        base, tuned, HeuristicJudge(),
        base_model="test-model", adapter=None,
        base_generator="mock-base", tuned_generator="mock-tuned",
        n_resamples=300, seed=1,
    )
    assert report.n_items == 3
    # tuned echoes the reference -> correctness delta is positive.
    assert report.headline.delta.delta > 0
    # exact_match on tuned should be perfect.
    em = next(d for d in report.intrinsic if d.name == "exact_match")
    assert em.tuned_mean == 1.0
    assert "higher" in report.verdict or "no measurable" in report.verdict


def test_build_report_no_difference_gives_honest_verdict():
    examples = _examples()
    gens = MockGenerator(transform=lambda ex: ex.answer).generate(examples)
    report = build_report(
        gens, gens, HeuristicJudge(),
        base_model="m", adapter=None,
        base_generator="a", tuned_generator="b",
        n_resamples=200, seed=1,
    )
    assert report.headline.delta.delta == 0.0
    assert "no measurable difference" in report.verdict


def test_build_report_rejects_id_mismatch():
    a = [Generation(id="1", question="q", context="", reference="r", answer="x")]
    b = [Generation(id="2", question="q", context="", reference="r", answer="x")]
    with pytest.raises(ValueError):
        build_report(a, b, HeuristicJudge(), base_model="m", adapter=None,
                     base_generator="a", tuned_generator="b", n_resamples=10, seed=1)


def test_write_report_emits_md_and_json(tmp_path):
    examples = _examples()
    base = MockGenerator(transform=lambda ex: ex.question).generate(examples)
    tuned = MockGenerator(transform=lambda ex: ex.answer).generate(examples)
    report = build_report(
        base, tuned, HeuristicJudge(),
        base_model="m", adapter="outputs/adapter",
        base_generator="mock-base", tuned_generator="mock-tuned",
        n_resamples=200, seed=1,
    )
    md_path, json_path = write_report(report, tmp_path / "r.md", tmp_path / "r.json")
    assert "Evaluation: base vs fine-tuned" in md_path.read_text()
    data = json.loads(json_path.read_text())
    assert data["n_items"] == 3
    assert data["p_value_status"] == "pending-validated-judge"


def test_render_markdown_mentions_placeholder_caveat():
    examples = _examples()
    gens = MockGenerator().generate(examples)
    report = build_report(
        gens, gens, HeuristicJudge(),
        base_model="m", adapter=None,
        base_generator="a", tuned_generator="b", n_resamples=100, seed=1,
    )
    md = render_markdown(report)
    assert "placeholder" in md
    assert "pending" in md
