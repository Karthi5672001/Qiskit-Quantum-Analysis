"""Tests for Stim sampling and decoding workflows."""

import numpy as np
import stim

from quantum_qec.circuits import (
    CircuitNoiseModel,
    create_repetition_memory_circuit,
)
from quantum_qec.simulation import (
    compile_detector_sampler,
    sample_detection_events,
    simulate_and_decode,
)


def create_test_circuit():
    return create_repetition_memory_circuit(
        distance=3,
        rounds=3,
        noise_model=(
            CircuitNoiseModel.symmetric(
                0.001
            )
        ),
    )


def test_compiled_sampler_type():
    circuit = create_test_circuit()

    sampler = compile_detector_sampler(
        circuit,
        seed=42,
    )

    assert isinstance(
        sampler,
        stim.CompiledDetectorSampler,
    )


def test_sampling_shapes():
    circuit = create_test_circuit()

    sampler = compile_detector_sampler(
        circuit,
        seed=42,
    )

    samples = sample_detection_events(
        sampler,
        number_of_shots=100,
    )

    assert (
        samples.detection_events.shape
        == (
            100,
            circuit.num_detectors,
        )
    )

    assert (
        samples.actual_observables.shape
        == (
            100,
            circuit.num_observables,
        )
    )

    assert samples.shots == 100


def test_sampling_is_reproducible_with_fixed_seed():
    circuit = create_test_circuit()

    first_sampler = compile_detector_sampler(
        circuit,
        seed=123,
    )

    second_sampler = compile_detector_sampler(
        circuit,
        seed=123,
    )

    first_samples = sample_detection_events(
        first_sampler,
        number_of_shots=100,
    )

    second_samples = sample_detection_events(
        second_sampler,
        number_of_shots=100,
    )

    assert np.array_equal(
        first_samples.detection_events,
        second_samples.detection_events,
    )

    assert np.array_equal(
        first_samples.actual_observables,
        second_samples.actual_observables,
    )


def test_simulate_and_decode_result():
    circuit = create_test_circuit()

    result = simulate_and_decode(
        circuit,
        number_of_shots=1_000,
        seed=42,
    )

    assert result.shots == 1_000

    assert (
        result.predicted_observables.shape
        == (
            1_000,
            circuit.num_observables,
        )
    )

    assert (
        result.logical_failures.shape
        == (1_000,)
    )

    assert (
        0.0
        <= result.logical_failure_probability
        <= 1.0
    )

    assert result.sample_seconds > 0.0
    assert result.decode_seconds > 0.0
    assert result.sampling_throughput > 0.0
    assert result.decoding_throughput > 0.0
    assert result.decoder_microseconds_per_shot > 0.0


def test_failure_count_matches_boolean_array():
    circuit = create_test_circuit()

    result = simulate_and_decode(
        circuit,
        number_of_shots=1_000,
        seed=42,
    )

    assert (
        result.logical_failure_count
        == int(
            result.logical_failures.sum()
        )
    )