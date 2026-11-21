"""Decoder benchmarking and workload-analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import pymatching


@dataclass(frozen=True)
class BatchBenchmark:
    """Summary of repeated batch-decoding measurements."""

    batch_size: int
    repeats: int
    elapsed_seconds: np.ndarray

    @property
    def median_seconds(self) -> float:
        return float(
            np.median(self.elapsed_seconds)
        )

    @property
    def mean_seconds(self) -> float:
        return float(
            np.mean(self.elapsed_seconds)
        )

    @property
    def median_microseconds_per_shot(
        self,
    ) -> float:
        return (
            1e6
            * self.median_seconds
            / self.batch_size
        )

    @property
    def median_shots_per_second(
        self,
    ) -> float:
        return (
            self.batch_size
            / self.median_seconds
        )


def validate_detection_events(
    decoder: pymatching.Matching,
    detection_events: np.ndarray,
) -> np.ndarray:
    """Validate detector samples for a decoder."""

    if not isinstance(
        decoder,
        pymatching.Matching,
    ):
        raise TypeError(
            "decoder must be a "
            "pymatching.Matching instance"
        )

    samples = np.asarray(
        detection_events
    )

    if samples.ndim != 2:
        raise ValueError(
            "detection_events must be "
            "two-dimensional"
        )

    if samples.shape[0] == 0:
        raise ValueError(
            "detection_events must contain "
            "at least one shot"
        )

    if (
        samples.shape[1]
        != decoder.num_detectors
    ):
        raise ValueError(
            "detection-event width does not "
            "match decoder.num_detectors"
        )

    return samples


def benchmark_batch_decoding(
    decoder: pymatching.Matching,
    detection_events: np.ndarray,
    *,
    repeats: int = 10,
    warm_up_shots: int = 10,
) -> BatchBenchmark:
    """Measure repeated batch-decoding performance."""

    samples = validate_detection_events(
        decoder,
        detection_events,
    )

    if repeats <= 0:
        raise ValueError(
            "repeats must be positive"
        )

    if warm_up_shots < 0:
        raise ValueError(
            "warm_up_shots cannot be negative"
        )

    actual_warm_up_shots = min(
        warm_up_shots,
        samples.shape[0],
    )

    if actual_warm_up_shots > 0:
        decoder.decode_batch(
            samples[
                :actual_warm_up_shots
            ]
        )

    elapsed_times = np.empty(
        repeats,
        dtype=float,
    )

    for repeat_index in range(repeats):
        start_time = perf_counter()

        decoder.decode_batch(
            samples
        )

        end_time = perf_counter()

        elapsed_times[repeat_index] = (
            end_time - start_time
        )

    return BatchBenchmark(
        batch_size=int(
            samples.shape[0]
        ),
        repeats=repeats,
        elapsed_seconds=elapsed_times,
    )


def benchmark_individual_decoding(
    decoder: pymatching.Matching,
    detection_events: np.ndarray,
    *,
    maximum_shots: int | None = None,
    warm_up_shots: int = 50,
) -> pd.DataFrame:
    """Measure individual decoder call latency.

    The returned timings include Python-call and
    operating-system overhead.
    """

    samples = validate_detection_events(
        decoder,
        detection_events,
    )

    if maximum_shots is not None:
        if maximum_shots <= 0:
            raise ValueError(
                "maximum_shots must be positive"
            )

        selected_count = min(
            maximum_shots,
            samples.shape[0],
        )

        selected_indices = np.linspace(
            0,
            samples.shape[0] - 1,
            selected_count,
            dtype=int,
        )

        selected_samples = samples[
            selected_indices
        ]
    else:
        selected_indices = np.arange(
            samples.shape[0]
        )

        selected_samples = samples

    actual_warm_up_shots = min(
        warm_up_shots,
        selected_samples.shape[0],
    )

    for sample in selected_samples[
        :actual_warm_up_shots
    ]:
        decoder.decode(sample)

    records = []

    for selected_position, sample in enumerate(
        selected_samples
    ):
        start_time = perf_counter()

        decoder.decode(sample)

        end_time = perf_counter()

        records.append(
            {
                "sample_index": int(
                    selected_indices[
                        selected_position
                    ]
                ),
                "detection_events": int(
                    sample.sum()
                ),
                "latency_microseconds": (
                    1e6
                    * (
                        end_time - start_time
                    )
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def summarise_latency(
    latency_dataframe: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarise individual decoding latencies."""

    required_columns = {
        "detection_events",
        "latency_microseconds",
    }

    missing_columns = (
        required_columns
        - set(latency_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "latency_dataframe is missing: "
            f"{sorted(missing_columns)}"
        )

    if latency_dataframe.empty:
        raise ValueError(
            "latency_dataframe cannot be empty"
        )

    latency = latency_dataframe[
        "latency_microseconds"
    ]

    pearson = latency_dataframe[
        [
            "detection_events",
            "latency_microseconds",
        ]
    ].corr(
        method="pearson"
    ).iloc[0, 1]

    spearman = latency_dataframe[
        [
            "detection_events",
            "latency_microseconds",
        ]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    return {
        "samples": int(
            len(latency_dataframe)
        ),
        "mean_microseconds": float(
            latency.mean()
        ),
        "median_microseconds": float(
            latency.median()
        ),
        "p90_microseconds": float(
            latency.quantile(0.90)
        ),
        "p95_microseconds": float(
            latency.quantile(0.95)
        ),
        "p99_microseconds": float(
            latency.quantile(0.99)
        ),
        "maximum_microseconds": float(
            latency.max()
        ),
        "pearson_event_latency": float(
            pearson
        ),
        "spearman_event_latency": float(
            spearman
        ),
    }


def benchmark_batch_sizes(
    decoder: pymatching.Matching,
    detection_events: np.ndarray,
    batch_sizes: list[int],
    *,
    repeats: int = 10,
) -> pd.DataFrame:
    """Benchmark effective cost across batch sizes."""

    samples = validate_detection_events(
        decoder,
        detection_events,
    )

    if not batch_sizes:
        raise ValueError(
            "batch_sizes cannot be empty"
        )

    records = []

    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError(
                "batch sizes must be positive"
            )

        if batch_size > samples.shape[0]:
            raise ValueError(
                "batch size exceeds available shots"
            )

        benchmark = benchmark_batch_decoding(
            decoder,
            samples[:batch_size],
            repeats=repeats,
        )

        records.append(
            {
                "batch_size": batch_size,
                "repeats": repeats,
                "median_seconds": (
                    benchmark.median_seconds
                ),
                "mean_seconds": (
                    benchmark.mean_seconds
                ),
                "median_microseconds_per_shot": (
                    benchmark
                    .median_microseconds_per_shot
                ),
                "median_shots_per_second": (
                    benchmark
                    .median_shots_per_second
                ),
            }
        )

    return pd.DataFrame(
        records
    )
