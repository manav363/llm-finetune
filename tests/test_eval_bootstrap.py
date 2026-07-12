import pytest

from llm_finetune.eval.bootstrap import P_VALUE_PENDING, paired_bootstrap


def test_zero_delta_when_scores_equal():
    est = paired_bootstrap([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], n_resamples=200, seed=1)
    assert est.delta == 0.0
    assert est.ci_low == 0.0 and est.ci_high == 0.0
    assert not est.ci_excludes_zero


def test_positive_delta_detected():
    base = [0.0] * 20
    tuned = [1.0] * 20
    est = paired_bootstrap(base, tuned, n_resamples=500, seed=1)
    assert est.delta == pytest.approx(1.0)
    assert est.ci_low > 0.0  # unanimous improvement -> CI above zero
    assert est.ci_excludes_zero


def test_p_value_is_flagged_pending():
    est = paired_bootstrap([0.1, 0.2], [0.3, 0.4], n_resamples=100, seed=1)
    assert est.p_value_status == P_VALUE_PENDING


def test_is_deterministic_for_seed():
    a = paired_bootstrap([0.1, 0.9, 0.5], [0.2, 0.8, 0.6], n_resamples=300, seed=7)
    b = paired_bootstrap([0.1, 0.9, 0.5], [0.2, 0.8, 0.6], n_resamples=300, seed=7)
    assert (a.ci_low, a.ci_high, a.p_value) == (b.ci_low, b.ci_high, b.p_value)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1], [0.2, 0.3], n_resamples=10, seed=1)


def test_rejects_empty():
    with pytest.raises(ValueError):
        paired_bootstrap([], [], n_resamples=10, seed=1)
