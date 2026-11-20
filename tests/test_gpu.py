"""Tests for GPU detector-event preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_qec.gpu import (
    benchmark_event_counting,
    choose_safe_gpu_batch_size,
    cpu_event_counts,
    cpu_event_density,
    cpu_pack_detector_bits,
    cpu_zero_event_mask,
    cupy_available,
    estimate_detector_matrix_bytes,
    get_gpu_device_information,
    gpu_event_counts,
    gpu_event_density,
    gpu_pack_detector_bits,
    gpu_zero_event_mask,
    validate_detector_events,
)


GPU_AVAILABLE = cupy_available()

gpu_required = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="A working CuPy CUDA device is required.",
)


def create_detector_matrix() -> np.ndarray:
    """Create deterministic detector-event samples."""

    rng = np.random.default_rng(42)

    return (
        rng.random(
            size=(1_000, 257)
        )
        < 0.05
    )


def test_validate_detector_events_accepts_boolean_matrix():
    events = create_detector_matrix()

    validated = validate_detector_events(
        events
    )

    assert validated.shape == (
        1_000,
        257,
    )

    assert validated.dtype == np.bool_


def test_validate_detector_events_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        validate_detector_events(
            np.array(
                [True, False]
            )
        )


def test_cpu_event_count_matches_direct_sum():
    events = create_detector_matrix()

    counts = cpu_event_counts(
        events
    )

    expected = events.sum(
        axis=1,
        dtype=np.int64,
    )

    assert np.array_equal(
        counts,
        expected,
    )


def test_cpu_density_matches_direct_mean():
    events = create_detector_matrix()

    density = cpu_event_density(
        events
    )

    expected = events.mean(
        axis=1,
    )

    assert np.allclose(
        density,
        expected,
    )


def test_cpu_zero_event_mask():
    events = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ],
        dtype=bool,
    )

    mask = cpu_zero_event_mask(
        events
    )

    assert mask.tolist() == [
        True,
        False,
        True,
    ]


def test_cpu_bit_packing_reduces_width():
    events = create_detector_matrix()

    packed = cpu_pack_detector_bits(
        events
    )

    assert packed.shape == (
        events.shape[0],
        33,
    )

    assert packed.dtype == np.uint8


def test_storage_estimate():
    assert estimate_detector_matrix_bytes(
        shots=1_000,
        detectors=2_000,
    ) == 2_000_000


def test_safe_batch_size_is_positive():
    batch_size = choose_safe_gpu_batch_size(
        detectors=1_000,
        available_memory_bytes=1_000_000,
        memory_fraction=0.5,
        working_array_multiplier=2.0,
    )

    assert batch_size == 250


@gpu_required
def test_gpu_device_information():
    information = get_gpu_device_information()

    assert information.available
    assert information.device_count >= 1
    assert information.name is not None
    assert information.total_memory_bytes is not None
    assert information.total_memory_bytes > 0


@gpu_required
def test_gpu_event_counts_match_cpu():
    events = create_detector_matrix()

    cpu_result = cpu_event_counts(
        events
    )

    gpu_result = gpu_event_counts(
        events
    )

    assert np.array_equal(
        cpu_result,
        gpu_result,
    )


@gpu_required
def test_gpu_density_matches_cpu():
    events = create_detector_matrix()

    cpu_result = cpu_event_density(
        events
    )

    gpu_result = gpu_event_density(
        events
    )

    assert np.allclose(
        cpu_result,
        gpu_result,
    )


@gpu_required
def test_gpu_zero_event_mask_matches_cpu():
    events = create_detector_matrix()

    cpu_result = cpu_zero_event_mask(
        events
    )

    gpu_result = gpu_zero_event_mask(
        events
    )

    assert np.array_equal(
        cpu_result,
        gpu_result,
    )


@gpu_required
def test_gpu_bit_packing_matches_cpu():
    events = create_detector_matrix()

    cpu_result = cpu_pack_detector_bits(
        events
    )

    gpu_result = gpu_pack_detector_bits(
        events
    )

    assert np.array_equal(
        cpu_result,
        gpu_result,
    )


@gpu_required
def test_gpu_benchmark_returns_positive_timings():
    events = create_detector_matrix()

    timing = benchmark_event_counting(
        events,
        repeats=3,
        warm_up_repeats=1,
    )

    assert timing.shots == 1_000
    assert timing.detectors == 257
    assert timing.cpu_seconds > 0.0
    assert timing.host_to_device_seconds > 0.0
    assert timing.gpu_kernel_seconds > 0.0
    assert timing.device_to_host_seconds > 0.0
    assert timing.gpu_end_to_end_seconds > 0.0
    assert timing.cpu_shots_per_second > 0.0
    assert timing.gpu_kernel_shots_per_second > 0.0
    assert timing.gpu_end_to_end_shots_per_second > 0.0