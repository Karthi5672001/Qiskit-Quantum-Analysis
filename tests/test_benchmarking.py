# Tests for decoder-benchmarking utilities

import numpy as np

from quantum_qec.benchmarking import (
    benchmark_batch_decoding,
    benchmark_batch_sizes,
    benchmark_individual_decoding,
    summarise_latency,
)
from quantum_qec.circuits import (
    CircuitNoiseModel,
    build_detector_error_model,
    create_repetition_memory_circuit,
)
from quantum_qec.decoding import (
    create_matching_decoder,
)
from quantum_qec.simulation import (
    compile_detector_sampler,
    sample_detection_events,
)


def create_benchmark_inputs():
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

    sampler = compile_detector_sampler(
        circuit,
        seed=42,
    )

    samples = sample_detection_events(
        sampler,
        number_of_shots=1_000,
    )

    return (
        decoder,
        samples.detection_events,
    )


def test_batch_benchmark_is_positive():
    decoder, detection_events = (
        create_benchmark_inputs()
    )

    benchmark = benchmark_batch_decoding(
        decoder,
        detection_events,
        repeats=3,
    )

    assert benchmark.batch_size == 1_000
    assert benchmark.repeats == 3
    assert benchmark.median_seconds > 0.0

    assert (
        benchmark
        .median_microseconds_per_shot
        > 0.0
    )

    assert (
        benchmark
        .median_shots_per_second
        > 0.0
    )


def test_individual_benchmark_shape():
    decoder, detection_events = (
        create_benchmark_inputs()
    )

    timing_df = (
        benchmark_individual_decoding(
            decoder,
            detection_events,
            maximum_shots=100,
            warm_up_shots=10,
        )
    )

    assert len(timing_df) == 100

    assert (
        timing_df[
            "latency_microseconds"
        ] > 0.0
    ).all()


def test_latency_summary_contains_percentiles():
    decoder, detection_events = (
        create_benchmark_inputs()
    )

    timing_df = (
        benchmark_individual_decoding(
            decoder,
            detection_events,
            maximum_shots=100,
            warm_up_shots=10,
        )
    )

    summary = summarise_latency(
        timing_df
    )

    assert summary["samples"] == 100

    assert (
        summary["p99_microseconds"]
        >= summary["median_microseconds"]
    )


def test_batch_size_benchmark():
    decoder, detection_events = (
        create_benchmark_inputs()
    )

    batch_size_df = benchmark_batch_sizes(
        decoder,
        detection_events,
        batch_sizes=[
            1,
            10,
            100,
            1_000,
        ],
        repeats=2,
    )

    assert batch_size_df[
        "batch_size"
    ].tolist() == [
        1,
        10,
        100,
        1_000,
    ]

    assert (
        batch_size_df[
            "median_microseconds_per_shot"
        ] > 0.0
    ).all()