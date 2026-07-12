import pytest

from llm_finetune.data.prepare import (
    clean_example,
    dedup_by_answer,
    near_dedup,
    prepare,
)
from llm_finetune.data.split import (
    SplitError,
    assert_no_leakage,
    split_examples,
    write_splits,
)
from llm_finetune.schema import QAExample, load_jsonl


def _examples(n: int) -> list[QAExample]:
    return [QAExample(id=f"id{i}", question=f"q{i}", answer=f"a{i}") for i in range(n)]


def test_clean_example_normalizes_whitespace_without_mutating():
    original = QAExample(id="x", question="a\n\n  b", answer="c   d", context="")
    cleaned = clean_example(original)
    assert cleaned.question == "a b"
    assert cleaned.answer == "c d"
    assert original.question == "a\n\n  b"  # input untouched


def test_dedup_by_answer_drops_repeat_pairs():
    exs = [
        QAExample(id="1", question="Q", answer="A"),
        QAExample(id="2", question="q", answer="a"),  # same pair, different case
        QAExample(id="3", question="Q2", answer="A2"),
    ]
    kept = dedup_by_answer(exs)
    assert [e.id for e in kept] == ["1", "3"]


def test_clean_example_normalizes_category():
    original = QAExample(id="x", question="q", answer="a", category="  api ")
    assert clean_example(original).category == "api"


def test_near_dedup_drops_reworded_duplicate():
    exs = [
        QAExample(id="1", question="How do I reset my password today", answer="Use the reset link"),
        # near-identical: one extra token vs the first
        QAExample(id="2", question="How do I reset my password", answer="Use the reset link"),
        QAExample(id="3", question="What is the storage limit", answer="It is two gigabytes"),
    ]
    kept = near_dedup(exs, threshold=0.85)
    assert [e.id for e in kept] == ["1", "3"]


def test_near_dedup_keeps_distinct_examples():
    exs = [
        QAExample(id="1", question="How do I reset my password", answer="Use the reset link"),
        QAExample(id="2", question="What regions are supported", answer="us-east and eu-west"),
    ]
    kept = near_dedup(exs, threshold=0.85)
    assert [e.id for e in kept] == ["1", "2"]


def test_near_dedup_threshold_one_keeps_non_identical():
    exs = [
        QAExample(id="1", question="alpha beta gamma", answer="x"),
        QAExample(id="2", question="alpha beta gamma delta", answer="x"),  # not token-identical
    ]
    assert [e.id for e in near_dedup(exs, threshold=1.0)] == ["1", "2"]


def test_near_dedup_rejects_out_of_range_threshold():
    with pytest.raises(ValueError):
        near_dedup(_examples(2), threshold=0.0)


def test_prepare_writes_processed_file(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        '{"id": "1", "question": "Q  1", "answer": "A"}\n'
        '{"id": "2", "question": "Q 1", "answer": "a"}\n',  # dup after normalization
        encoding="utf-8",
    )
    out = tmp_path / "processed.jsonl"
    result = prepare(raw, out)
    assert len(result) == 1
    assert load_jsonl(str(out))[0].id == "1"


def test_split_is_deterministic_for_a_seed():
    exs = _examples(20)
    a = split_examples(exs, val_frac=0.15, test_frac=0.15, seed=7)
    b = split_examples(exs, val_frac=0.15, test_frac=0.15, seed=7)
    assert [e.id for e in a.train] == [e.id for e in b.train]
    assert [e.id for e in a.test] == [e.id for e in b.test]


def test_split_has_no_leakage_and_covers_all():
    exs = _examples(20)
    splits = split_examples(exs, val_frac=0.15, test_frac=0.15, seed=1)
    assert_no_leakage(splits)  # must not raise
    total = splits.train + splits.val + splits.test
    assert {e.id for e in total} == {e.id for e in exs}


def test_leakage_is_detected():
    dup = QAExample(id="shared", question="q", answer="a")
    from llm_finetune.data.split import DatasetSplits

    bad = DatasetSplits(train=[dup], val=[dup], test=_examples(1))
    with pytest.raises(SplitError):
        assert_no_leakage(bad)


def test_split_rejects_too_few_examples():
    with pytest.raises(SplitError):
        split_examples(_examples(3), val_frac=0.15, test_frac=0.15, seed=1)


def test_write_splits_creates_three_files(tmp_path):
    splits = split_examples(_examples(20), val_frac=0.15, test_frac=0.15, seed=1)
    paths = write_splits(splits, tmp_path / "splits")
    for name in ("train", "val", "test"):
        assert paths[name].is_file()
