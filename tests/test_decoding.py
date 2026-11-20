"""Tests for decoding utilities."""

import numpy as np
import pytest
import pymatching

from quantum_qec.circuits import (
    CircuitNoiseModel,
    build_detector_error_model,
    create_repetition_memory_circuit,
)
from quantum_qec.decoding import (
    calculate_logical_failure_count,
    calculate_logical_failure_probability,
    calculate_logical_failures,
    create_matching_decoder,
    decode_detection_events,
)


def test_logical_failure_array():
    predicted = np.array(
        [
            [False],
            [True],
            [False],
        ]
    )

    actual = np.array(
        [
            [False],
            [False],
            [False],
        ]
    )

    failures = calculate_logical_failures(
        predicted_observables=predicted,
        actual_observables=actual,
    )

    assert failures.tolist() == [
        False,
        True,
        False,
    ]


def test_logical_failure_count():
    predicted = np.array(
        [
            [False],
            [True],
            [True],
        ]
    )

    actual = np.array(
        [
            [False],
            [False],
            [True],
        ]
    )

    count = calculate_logical_failure_count(
        predicted,
        actual,
    )

    assert count == 1


def test_logical_failure_probability():
    predicted = np.array(
        [
            [False],
            [True],
            [True],
            [False],
        ]
    )

    actual = np.array(
        [
            [False],
            [False],
            [True],
            [True],
        ]
    )

    probability = (
        calculate_logical_failure_probability(
            predicted,
            actual,
        )
    )

    assert np.isclose(
        probability,
        0.5,
    )


def test_mismatched_observable_shapes_are_rejected():
    predicted = np.zeros(
        (3, 1),
        dtype=bool,
    )

    actual = np.zeros(
        (4, 1),
        dtype=bool,
    )

    with pytest.raises(ValueError):
        calculate_logical_failures(
            predicted,
            actual,
        )


def test_matching_decoder_can_be_created():
    circuit = create_repetition_memory_circuit(
        distance=3,
        rounds=3,
        noise_model=(
            CircuitNoiseModel.symmetric(
                0.001
            )
        ),
    )

    detector_error_model = (
        build_detector_error_model(
            circuit
        )
    )

    decoder = create_matching_decoder(
        detector_error_model
    )

    assert isinstance(
        decoder,
        pymatching.Matching,
    )

    assert (
        decoder.num_detectors
        == circuit.num_detectors
    )


def test_detection_event_width_is_validated():
    circuit = create_repetition_memory_circuit(
        distance=3,
        rounds=3,
        noise_model=(
            CircuitNoiseModel.symmetric(
                0.001
            )
        ),
    )

    detector_error_model = (
        build_detector_error_model(
            circuit
        )
    )

    decoder = create_matching_decoder(
        detector_error_model
    )

    invalid_samples = np.zeros(
        (
            10,
            decoder.num_detectors + 1,
        ),
        dtype=bool,
    )

    with pytest.raises(ValueError):
        decode_detection_events(
            decoder,
            invalid_samples,
        )