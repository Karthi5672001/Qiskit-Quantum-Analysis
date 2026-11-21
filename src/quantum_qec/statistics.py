"""Statistical utilities for logical-error analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BinomialInterval:
    """A probability estimate and approximate 95% limits."""

    estimate: float
    lower: float
    upper: float
    standard_error: float | None
    method: str


def validate_probability(
    probability: float,
    parameter_name: str = "probability",
) -> None:
    """Validate a probability in the inclusive interval [0, 1]."""

    if not isinstance(
        probability,
        (int, float),
    ):
        raise TypeError(
            f"{parameter_name} must be numeric"
        )

    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError(
            f"{parameter_name} must be between zero and one"
        )


def binomial_standard_error(
    estimated_probability: float,
    number_of_trials: int,
) -> float:
    """Return the standard error of a binomial estimate."""

    validate_probability(
        estimated_probability,
        "estimated_probability",
    )

    if number_of_trials <= 0:
        raise ValueError(
            "number_of_trials must be positive"
        )

    return math.sqrt(
        estimated_probability
        * (
            1.0
            - estimated_probability
        )
        / number_of_trials
    )


def rule_of_three_upper_bound(
    number_of_trials: int,
) -> float:
    """Return the approximate 95% upper bound after zero failures."""

    if number_of_trials <= 0:
        raise ValueError(
            "number_of_trials must be positive"
        )

    return min(
        1.0,
        3.0 / number_of_trials,
    )


def approximate_binomial_interval(
    failure_count: int,
    number_of_trials: int,
) -> BinomialInterval:
    """Return an approximate 95% interval or zero-failure bound."""

    if number_of_trials <= 0:
        raise ValueError(
            "number_of_trials must be positive"
        )

    if failure_count < 0:
        raise ValueError(
            "failure_count cannot be negative"
        )

    if failure_count > number_of_trials:
        raise ValueError(
            "failure_count cannot exceed "
            "number_of_trials"
        )

    estimate = (
        failure_count
        / number_of_trials
    )

    if failure_count == 0:
        return BinomialInterval(
            estimate=0.0,
            lower=0.0,
            upper=(
                rule_of_three_upper_bound(
                    number_of_trials
                )
            ),
            standard_error=None,
            method="rule_of_three",
        )

    standard_error = (
        binomial_standard_error(
            estimate,
            number_of_trials,
        )
    )

    lower = max(
        0.0,
        estimate
        - 1.96 * standard_error,
    )

    upper = min(
        1.0,
        estimate
        + 1.96 * standard_error,
    )

    return BinomialInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        standard_error=standard_error,
        method="normal_approximation",
    )


def repetition_code_logical_failure_probability(
    physical_error_probability: float,
) -> float:
    """Return 3p^2 - 2p^3 for the three-bit repetition code."""

    validate_probability(
        physical_error_probability,
        "physical_error_probability",
    )

    p = float(
        physical_error_probability
    )

    return (
        3.0 * p**2
        - 2.0 * p**3
    )


def trials_for_expected_failures(
    failure_probability: float,
    expected_failures: float,
) -> int:
    """Return trials needed for a target expected failure count."""

    if not 0.0 < failure_probability <= 1.0:
        raise ValueError(
            "failure_probability must be "
            "greater than zero and no greater "
            "than one"
        )

    if expected_failures <= 0.0:
        raise ValueError(
            "expected_failures must be positive"
        )

    return math.ceil(
        expected_failures
        / failure_probability
    )