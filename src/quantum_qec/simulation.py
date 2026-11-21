"""Stim sampling and PyMatching simulation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pymatching
import stim

from quantum_qec.circuits import (
    build_detector_error_model,
)
from quantum_qec.decoding import (
    calculate_logical_failures,
    create_matching_decoder,
    decode_detection_events,
)
from quantum_qec.statistics import (
    BinomialInterval,
    approximate_binomial_interval,
)


@dataclass(frozen=True)
class DetectorSamples:
    """Detector and logical-observable samples."""

    detection_events: np.ndarray
    actual_observables: np.ndarray

    @property
    def shots(self) -> int:
        """Return the sampled shot count."""

        return int(
            self.detection_events.shape[0]
        )

    @property
    def mean_detection_events(self) -> float:
        """Return the mean detector-event count per shot."""

        return float(
            self.detection_events.sum(
                axis=1
            ).mean()
        )


@dataclass(frozen=True)
class DecodingResult:
    """Decoded logical-observable results."""

    predicted_observables: np.ndarray
    logical_failures: np.ndarray
    logical_failure_count: int
    logical_failure_probability: float
    interval_95: BinomialInterval
    sample_seconds: float
    decode_seconds: float
    mean_detection_events: float

    @property
    def shots(self) -> int:
        """Return the number of decoded shots."""

        return int(
            self.logical_failures.shape[0]
        )

    @property
    def sampling_throughput(self) -> float:
        """Return sampled shots per second."""

        return (
            self.shots
            / self.sample_seconds
        )

    @property
    def decoding_throughput(self) -> float:
        """Return decoded shots per second."""

        return (
            self.shots
            / self.decode_seconds
        )

    @property
    def decoder_microseconds_per_shot(
        self,
    ) -> float:
        """Return effective batch decode time per shot."""

        return (
            1e6
            * self.decode_seconds
            / self.shots
        )


def validate_shot_count(
    number_of_shots: int,
) -> None:
    """Validate a positive integer shot count."""

    if not isinstance(
        number_of_shots,
        int,
    ):
        raise TypeError(
            "number_of_shots must be an integer"
        )

    if number_of_shots <= 0:
        raise ValueError(
            "number_of_shots must be positive"
        )


def compile_detector_sampler(
    circuit: stim.Circuit,
    *,
    seed: int | None = None,
) -> stim.CompiledDetectorSampler:
    """Compile a Stim detector-event sampler."""

    if not isinstance(
        circuit,
        stim.Circuit,
    ):
        raise TypeError(
            "circuit must be a stim.Circuit"
        )

    if circuit.num_detectors <= 0:
        raise ValueError(
            "circuit must contain detectors"
        )

    return circuit.compile_detector_sampler(
        seed=seed
    )


def sample_detection_events(
    sampler: stim.CompiledDetectorSampler,
    number_of_shots: int,
) -> DetectorSamples:
    """Sample detector events and actual observables."""

    if not isinstance(
        sampler,
        stim.CompiledDetectorSampler,
    ):
        raise TypeError(
            "sampler must be a "
            "stim.CompiledDetectorSampler"
        )

    validate_shot_count(
        number_of_shots
    )

    detection_events, actual_observables = (
        sampler.sample(
            shots=number_of_shots,
            separate_observables=True,
        )
    )

    if detection_events.ndim != 2:
        raise RuntimeError(
            "Stim returned invalid detector samples"
        )

    if actual_observables.ndim != 2:
        raise RuntimeError(
            "Stim returned invalid observable samples"
        )

    if (
        detection_events.shape[0]
        != actual_observables.shape[0]
    ):
        raise RuntimeError(
            "Detector and observable shot counts "
            "do not match"
        )

    return DetectorSamples(
        detection_events=(
            detection_events
        ),
        actual_observables=(
            actual_observables
        ),
    )


def simulate_and_decode(
    circuit: stim.Circuit,
    number_of_shots: int,
    *,
    seed: int | None = None,
    warm_up_shots: int = 10,
) -> DecodingResult:
    """Sample and decode a circuit-level QEC experiment."""

    validate_shot_count(
        number_of_shots
    )

    if warm_up_shots < 0:
        raise ValueError(
            "warm_up_shots cannot be negative"
        )

    detector_error_model = (
        build_detector_error_model(
            circuit
        )
    )

    decoder: pymatching.Matching = (
        create_matching_decoder(
            detector_error_model
        )
    )

    sampler = compile_detector_sampler(
        circuit,
        seed=seed,
    )

    sample_start = perf_counter()

    samples = sample_detection_events(
        sampler,
        number_of_shots,
    )

    sample_end = perf_counter()

    actual_warm_up_shots = min(
        warm_up_shots,
        number_of_shots,
    )

    if actual_warm_up_shots > 0:
        decoder.decode_batch(
            samples.detection_events[
                :actual_warm_up_shots
            ]
        )

    decode_start = perf_counter()

    predicted_observables = (
        decode_detection_events(
            decoder,
            samples.detection_events,
        )
    )

    decode_end = perf_counter()

    logical_failures = (
        calculate_logical_failures(
            predicted_observables,
            samples.actual_observables,
        )
    )

    logical_failure_count = int(
        logical_failures.sum()
    )

    logical_failure_probability = float(
        logical_failures.mean()
    )

    interval_95 = (
        approximate_binomial_interval(
            logical_failure_count,
            number_of_shots,
        )
    )

    return DecodingResult(
        predicted_observables=(
            predicted_observables
        ),
        logical_failures=(
            logical_failures
        ),
        logical_failure_count=(
            logical_failure_count
        ),
        logical_failure_probability=(
            logical_failure_probability
        ),
        interval_95=interval_95,
        sample_seconds=(
            sample_end - sample_start
        ),
        decode_seconds=(
            decode_end - decode_start
        ),
        mean_detection_events=(
            samples.mean_detection_events
        ),
    )