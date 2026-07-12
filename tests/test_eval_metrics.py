import pytest

from llm_finetune.eval.metrics import (
    all_metrics,
    exact_match,
    mean_metrics,
    normalize_text,
    normalized_match,
    rouge_l,
    token_f1,
)


def test_normalize_text_strips_case_punct_ws():
    assert normalize_text("  The  CAT, sat! ") == "the cat sat"


def test_exact_match_is_strict():
    assert exact_match("hello", "hello") == 1.0
    assert exact_match("hello", "Hello") == 0.0


def test_normalized_match_ignores_case_and_punct():
    assert normalized_match("The cat.", "the cat") == 1.0
    assert normalized_match("a dog", "a cat") == 0.0


def test_token_f1_partial_overlap():
    # pred={a,b,c}, ref={a,b} -> overlap 2, p=2/3, r=1 -> F1 = 0.8
    assert token_f1("a b c", "a b") == pytest.approx(0.8)


def test_token_f1_no_overlap_is_zero():
    assert token_f1("x y", "a b") == 0.0


def test_rouge_l_rewards_subsequence_order():
    # LCS of "a b c d" vs "a c d" is "a c d" (len 3); p=3/4, r=1 -> 0.857
    assert rouge_l("a b c d", "a c d") == pytest.approx(6 / 7)


def test_rouge_l_identical_is_one():
    assert rouge_l("same words here", "same words here") == 1.0


def test_all_metrics_keys():
    m = all_metrics("a b", "a b")
    assert set(m) == {"exact_match", "normalized_match", "token_f1", "rouge_l"}
    assert all(v == 1.0 for v in m.values())


def test_mean_metrics_averages_over_items():
    preds = ["a b", "x"]
    refs = ["a b", "y"]
    means = mean_metrics(preds, refs)
    assert means["exact_match"] == pytest.approx(0.5)


def test_mean_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mean_metrics(["a"], ["a", "b"])
