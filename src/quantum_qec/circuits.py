"""Circuit-generation utilities for quantum error-correction studies.

This module provides validated constructors for Stim repetition-code and
rotated-surface-code memory circuits. It supports both symmetric and
asymmetric circuit-level Pauli-noise models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import stim


MemoryBasis = Literal["x", "z"]


@dataclass(frozen=True)
class CircuitNoiseModel:
    """Circuit-level Pauli-noise probabilities.

    Attributes
    ----------
    clifford_depolarisation:
        Depolarising probability applied after Clifford operations.
    data_depolarisation:
        Data-qubit depolarising probability applied before each round.
    measurement_flip:
        Probability that a measurement result is flipped.
    reset_flip:
        Probability that a reset prepares the wrong computational state.
    """

    clifford_depolarisation: float = 0.0
    data_depolarisation: float = 0.0
    measurement_flip: float = 0.0
    reset_flip: float = 0.0

    def __post_init__(self) -> None:
        """Validate all stored probabilities."""

        for field_name, probability in (
            (
                "clifford_depolarisation",
                self.clifford_depolarisation,
            ),
            (
                "data_depolarisation",
                self.data_depolarisation,
            ),
            (
                "measurement_flip",
                self.measurement_flip,
            ),
            (
                "reset_flip",
                self.reset_flip,
            ),
        ):
            validate_probability(
                probability,
                field_name,
            )

    @classmethod
    def symmetric(
        cls,
        probability: float,
    ) -> "CircuitNoiseModel":
        """Create a noise model using one probability for every process."""

        validate_probability(
            probability,
            "probability",
        )

        return cls(
            clifford_depolarisation=probability,
            data_depolarisation=probability,
            measurement_flip=probability,
            reset_flip=probability,
        )

    def with_scaled_parameter(
        self,
        parameter_name: str,
        multiplier: float,
    ) -> "CircuitNoiseModel":
        """Return a new model with one probability multiplied.

        Parameters
        ----------
        parameter_name:
            One of the four dataclass field names.
        multiplier:
            Non-negative scale factor.
        """

        valid_parameters = {
            "clifford_depolarisation",
            "data_depolarisation",
            "measurement_flip",
            "reset_flip",
        }

        if parameter_name not in valid_parameters:
            raise ValueError(
                f"Unknown noise parameter: {parameter_name}"
            )

        if multiplier < 0.0:
            raise ValueError(
                "multiplier must be non-negative"
            )

        values = {
            "clifford_depolarisation": (
                self.clifford_depolarisation
            ),
            "data_depolarisation": (
                self.data_depolarisation
            ),
            "measurement_flip": (
                self.measurement_flip
            ),
            "reset_flip": (
                self.reset_flip
            ),
        }

        scaled_probability = (
            values[parameter_name]
            * multiplier
        )

        validate_probability(
            scaled_probability,
            parameter_name,
        )

        values[parameter_name] = (
            scaled_probability
        )

        return CircuitNoiseModel(
            **values
        )

    def as_dict(self) -> dict[str, float]:
        """Return the noise model as a plain dictionary."""

        return {
            "clifford_depolarisation": (
                self.clifford_depolarisation
            ),
            "data_depolarisation": (
                self.data_depolarisation
            ),
            "measurement_flip": (
                self.measurement_flip
            ),
            "reset_flip": (
                self.reset_flip
            ),
        }


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


def validate_odd_distance(
    distance: int,
) -> None:
    """Validate an odd code distance of at least three."""

    if not isinstance(
        distance,
        int,
    ):
        raise TypeError(
            "distance must be an integer"
        )

    if distance < 3:
        raise ValueError(
            "distance must be at least 3"
        )

    if distance % 2 == 0:
        raise ValueError(
            "distance must be odd"
        )


def validate_rounds(
    rounds: int,
) -> None:
    """Validate a positive integer number of syndrome rounds."""

    if not isinstance(
        rounds,
        int,
    ):
        raise TypeError(
            "rounds must be an integer"
        )

    if rounds <= 0:
        raise ValueError(
            "rounds must be positive"
        )


def validate_memory_basis(
    memory_basis: str,
) -> MemoryBasis:
    """Validate and normalise a logical memory basis."""

    if not isinstance(
        memory_basis,
        str,
    ):
        raise TypeError(
            "memory_basis must be a string"
        )

    normalised_basis = (
        memory_basis.strip().lower()
    )

    if normalised_basis not in {
        "x",
        "z",
    }:
        raise ValueError(
            "memory_basis must be 'x' or 'z'"
        )

    return normalised_basis  # type: ignore[return-value]


def create_repetition_memory_circuit(
    distance: int,
    rounds: int,
    noise_model: CircuitNoiseModel,
) -> stim.Circuit:
    """Create a noisy Stim repetition-code memory circuit."""

    validate_odd_distance(
        distance
    )

    validate_rounds(
        rounds
    )

    if not isinstance(
        noise_model,
        CircuitNoiseModel,
    ):
        raise TypeError(
            "noise_model must be a CircuitNoiseModel"
        )

    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=(
            noise_model.clifford_depolarisation
        ),
        before_round_data_depolarization=(
            noise_model.data_depolarisation
        ),
        before_measure_flip_probability=(
            noise_model.measurement_flip
        ),
        after_reset_flip_probability=(
            noise_model.reset_flip
        ),
    )


def create_rotated_surface_code_circuit(
    distance: int,
    rounds: int,
    noise_model: CircuitNoiseModel,
    memory_basis: MemoryBasis = "x",
) -> stim.Circuit:
    """Create a noisy rotated-surface-code memory circuit."""

    validate_odd_distance(
        distance
    )

    validate_rounds(
        rounds
    )

    basis = validate_memory_basis(
        memory_basis
    )

    if not isinstance(
        noise_model,
        CircuitNoiseModel,
    ):
        raise TypeError(
            "noise_model must be a CircuitNoiseModel"
        )

    code_task = (
        "surface_code:rotated_memory_"
        f"{basis}"
    )

    return stim.Circuit.generated(
        code_task,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=(
            noise_model.clifford_depolarisation
        ),
        before_round_data_depolarization=(
            noise_model.data_depolarisation
        ),
        before_measure_flip_probability=(
            noise_model.measurement_flip
        ),
        after_reset_flip_probability=(
            noise_model.reset_flip
        ),
    )


def build_detector_error_model(
    circuit: stim.Circuit,
    *,
    decompose_errors: bool = True,
) -> stim.DetectorErrorModel:
    """Construct a detector error model from an annotated Stim circuit."""

    if not isinstance(
        circuit,
        stim.Circuit,
    ):
        raise TypeError(
            "circuit must be a stim.Circuit"
        )

    if circuit.num_detectors <= 0:
        raise ValueError(
            "circuit must contain at least one detector"
        )

    if circuit.num_observables <= 0:
        raise ValueError(
            "circuit must contain at least one logical observable"
        )

    return circuit.detector_error_model(
        decompose_errors=decompose_errors
    )