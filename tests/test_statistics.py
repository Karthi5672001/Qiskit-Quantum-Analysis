"""Tests for statistical utilities."""

import math

import pytest

from quantum_qec.statistics import (
    approximate_binomial_interval,
    binomial_standard_error,
    repetition_code_logical_failure_probability,
    rule_of_three_upper_bound,
    trials_for_expected_failures,
)


def test_repetition_code_failure_at_zero():
    assert (
        repetition_code_logical_failure_probability(0.0)
        == 0.0
    )


def test_repetition_code_failure_at_half():
    assert math.isclose(
        repetition_code_logical_failure_probability(0.5),
        0.5,
    )


def test_repetition_code_failure_at_one():
    assert (
        repetition_code_logical_failure_probability(1.0)
        == 1.0
    )


def test_rule_of_three():
    assert math.isclose(
        rule_of_three_upper_bound(1_000_000),
        3e-6,
    )


def test_binomial_standard_error_is_positive():
    assert (
        binomial_standard_error(
            estimated_probability=0.1,
            number_of_trials=10_000,
        )
        > 0.0
    )


def test_zero_failure_interval_uses_rule_of_three():
    interval = approximate_binomial_interval(
        failure_count=0,
        number_of_trials=1_000_000,
    )

    assert interval.estimate == 0.0
    assert interval.lower == 0.0
    assert math.isclose(
        interval.upper,
        3e-6,
    )
    assert interval.standard_error is None
    assert interval.method == "rule_of_three"


def test_nonzero_interval_uses_normal_approximation():
    interval = approximate_binomial_interval(
        failure_count=100,
        number_of_trials=10_000,
    )

    assert math.isclose(
        interval.estimate,
        0.01,
    )

    assert interval.lower < interval.estimate
    assert interval.upper > interval.estimate
    assert interval.standard_error is not None

    assert (
        interval.method
        == "normal_approximation"
    )


def test_failure_count_cannot_exceed_trials():
    with pytest.raises(ValueError):
        approximate_binomial_interval(
            failure_count=11,
            number_of_trials=10,
        )


def test_trials_for_expected_failures():
    assert (
        trials_for_expected_failures(
            failure_probability=1e-6,
            expected_failures=100,
        )
        == 100_000_000
    )