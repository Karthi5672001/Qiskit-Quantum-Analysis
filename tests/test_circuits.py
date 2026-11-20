"""Tests for QEC circuit-generation utilities."""

import pytest
import stim

from quantum_qec.circuits import (
    CircuitNoiseModel,
    build_detector_error_model,
    create_repetition_memory_circuit,
    create_rotated_surface_code_circuit,
    validate_memory_basis,
    validate_odd_distance,
    validate_probability,
    validate_rounds,
)


def test_distance_three_is_valid():
    validate_odd_distance(3)


def test_even_distance_is_invalid():
    with pytest.raises(ValueError):
        validate_odd_distance(4)


def test_distance_below_three_is_invalid():
    with pytest.raises(ValueError):
        validate_odd_distance(1)


def test_non_integer_distance_is_invalid():
    with pytest.raises(TypeError):
        validate_odd_distance(3.0)


def test_positive_round_count_is_valid():
    validate_rounds(3)


def test_zero_round_count_is_invalid():
    with pytest.raises(ValueError):
        validate_rounds(0)


def test_probability_bounds_are_valid():
    validate_probability(0.0)
    validate_probability(0.5)
    validate_probability(1.0)


def test_probability_outside_bounds_is_invalid():
    with pytest.raises(ValueError):
        validate_probability(-0.01)

    with pytest.raises(ValueError):
        validate_probability(1.01)


def test_memory_basis_is_normalised():
    assert validate_memory_basis("X") == "x"
    assert validate_memory_basis(" z ") == "z"


def test_invalid_memory_basis_is_rejected():
    with pytest.raises(ValueError):
        validate_memory_basis("y")


def test_symmetric_noise_model():
    noise_model = CircuitNoiseModel.symmetric(
        0.005
    )

    assert noise_model.clifford_depolarisation == 0.005
    assert noise_model.data_depolarisation == 0.005
    assert noise_model.measurement_flip == 0.005
    assert noise_model.reset_flip == 0.005


def test_noise_model_rejects_invalid_probability():
    with pytest.raises(ValueError):
        CircuitNoiseModel(
            measurement_flip=1.2
        )


def test_scaled_noise_model_changes_only_selected_parameter():
    baseline = CircuitNoiseModel(
        clifford_depolarisation=0.001,
        data_depolarisation=0.0005,
        measurement_flip=0.005,
        reset_flip=0.002,
    )

    modified = baseline.with_scaled_parameter(
        "measurement_flip",
        2.0,
    )

    assert modified.measurement_flip == 0.01

    assert (
        modified.clifford_depolarisation
        == baseline.clifford_depolarisation
    )

    assert (
        modified.data_depolarisation
        == baseline.data_depolarisation
    )

    assert (
        modified.reset_flip
        == baseline.reset_flip
    )


def test_repetition_memory_circuit_has_detectors_and_observable():
    circuit = create_repetition_memory_circuit(
        distance=3,
        rounds=3,
        noise_model=(
            CircuitNoiseModel.symmetric(
                0.001
            )
        ),
    )

    assert isinstance(
        circuit,
        stim.Circuit,
    )

    assert circuit.num_qubits > 0
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 1


@pytest.mark.parametrize(
    "memory_basis",
    [
        "x",
        "z",
    ],
)
def test_surface_code_circuit_has_detectors_and_observable(
    memory_basis,
):
    circuit = create_rotated_surface_code_circuit(
        distance=3,
        rounds=3,
        noise_model=(
            CircuitNoiseModel.symmetric(
                0.001
            )
        ),
        memory_basis=memory_basis,
    )

    assert isinstance(
        circuit,
        stim.Circuit,
    )

    assert circuit.num_qubits > 0
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 1


def test_surface_x_and_z_circuits_use_same_number_of_qubits():
    noise_model = CircuitNoiseModel.symmetric(
        0.001
    )

    x_circuit = (
        create_rotated_surface_code_circuit(
            distance=3,
            rounds=3,
            noise_model=noise_model,
            memory_basis="x",
        )
    )

    z_circuit = (
        create_rotated_surface_code_circuit(
            distance=3,
            rounds=3,
            noise_model=noise_model,
            memory_basis="z",
        )
    )

    assert (
        x_circuit.num_qubits
        == z_circuit.num_qubits
    )


def test_detector_error_model_can_be_constructed():
    circuit = create_rotated_surface_code_circuit(
        distance=3,
        rounds=3,
        noise_model=(
            CircuitNoiseModel.symmetric(
                0.001
            )
        ),
        memory_basis="x",
    )

    detector_error_model = (
        build_detector_error_model(
            circuit
        )
    )

    assert isinstance(
        detector_error_model,
        stim.DetectorErrorModel,
    )

    assert (
        detector_error_model.num_detectors
        == circuit.num_detectors
    )

    assert (
        detector_error_model.num_observables
        == circuit.num_observables
    )