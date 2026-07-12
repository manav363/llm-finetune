import json

import pytest

from llm_finetune.data.stats import (
    compute_stats,
    format_report,
    write_report,
)
from llm_finetune.schema import QAExample, load_jsonl


def _ex(id_: str, question: str, answer: str, category: str = "") -> QAExample:
    return QAExample(id=id_, question=question, answer=answer, category=category)


def test_compute_stats_counts_and_category_balance():
    exs = [
        _ex("1", "a b c", "one", category="api"),
        _ex("2", "a b", "two words", category="api"),
        _ex("3", "a", "three", category="security"),
    ]
    stats = compute_stats(exs)
    assert stats.count == 3
    assert stats.category_balance == {"api": 2, "security": 1}


def test_compute_stats_buckets_missing_category_as_uncategorized():
    stats = compute_stats([_ex("1", "q", "a"), _ex("2", "q", "a", category="x")])
    assert stats.category_balance == {"uncategorized": 1, "x": 1}


def test_length_summary_token_counts():
    exs = [
        _ex("1", "one", "a"),
        _ex("2", "one two", "a"),
        _ex("3", "one two three four five", "a"),
    ]
    q = compute_stats(exs).question_tokens
    assert q.min == 1
    assert q.max == 5
    assert q.median == 2.0
    assert q.mean == round((1 + 2 + 5) / 3, 2)


def test_category_balance_sums_to_count():
    exs = [_ex(str(i), "q", "a", category=("api" if i % 2 else "")) for i in range(10)]
    stats = compute_stats(exs)
    assert sum(stats.category_balance.values()) == stats.count == 10


def test_compute_stats_rejects_empty():
    with pytest.raises(ValueError):
        compute_stats([])


def test_write_report_is_valid_json(tmp_path):
    stats = compute_stats([_ex("1", "q here", "an answer", category="api")])
    path = write_report(stats, tmp_path / "stats.json")
    loaded = json.loads(path.read_text())
    assert loaded["count"] == 1
    assert loaded["category_balance"] == {"api": 1}
    assert loaded["question_tokens"]["max"] == 2


def test_format_report_mentions_count_and_categories():
    stats = compute_stats([_ex("1", "q", "a", category="security")])
    report = format_report(stats)
    assert "1 examples" in report
    assert "security" in report


def test_stats_on_sample_dataset():
    stats = compute_stats(load_jsonl("data/sample/domain_qa.jsonl"))
    assert stats.count == 20
    assert sum(stats.category_balance.values()) == 20
    assert "uncategorized" not in stats.category_balance  # every sample row is labeled
