"""Classical-decoder queue and backlog models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_worker_utilisation(
    mean_service_time_microseconds: float,
    arrival_interval_microseconds: float,
    number_of_workers: int = 1,
) -> float:
    """Calculate mean utilisation per decoder worker."""

    if mean_service_time_microseconds <= 0.0:
        raise ValueError(
            "mean_service_time_microseconds "
            "must be positive"
        )

    if arrival_interval_microseconds <= 0.0:
        raise ValueError(
            "arrival_interval_microseconds "
            "must be positive"
        )

    if number_of_workers <= 0:
        raise ValueError(
            "number_of_workers must be positive"
        )

    return (
        mean_service_time_microseconds
        / arrival_interval_microseconds
        / number_of_workers
    )


def minimum_workers_for_utilisation(
    mean_service_time_microseconds: float,
    arrival_interval_microseconds: float,
    target_utilisation: float = 0.70,
) -> int:
    """Return workers needed not to exceed target utilisation."""

    if not 0.0 < target_utilisation < 1.0:
        raise ValueError(
            "target_utilisation must be "
            "strictly between zero and one"
        )

    offered_load = calculate_worker_utilisation(
        mean_service_time_microseconds,
        arrival_interval_microseconds,
        number_of_workers=1,
    )

    return max(
        1,
        math.ceil(
            offered_load
            / target_utilisation
        ),
    )


def simulate_decoder_queue(
    service_times_microseconds,
    arrival_interval_microseconds: float,
    number_of_workers: int = 1,
    deadline_microseconds: float | None = None,
) -> pd.DataFrame:
    """Simulate deterministic arrivals and parallel workers."""

    service_times = np.asarray(
        service_times_microseconds,
        dtype=float,
    )

    if service_times.ndim != 1:
        raise ValueError(
            "service_times_microseconds must "
            "be one-dimensional"
        )

    if len(service_times) == 0:
        raise ValueError(
            "at least one service time is required"
        )

    if np.any(
        service_times <= 0.0
    ):
        raise ValueError(
            "all service times must be positive"
        )

    if arrival_interval_microseconds <= 0.0:
        raise ValueError(
            "arrival_interval_microseconds "
            "must be positive"
        )

    if number_of_workers <= 0:
        raise ValueError(
            "number_of_workers must be positive"
        )

    if deadline_microseconds is None:
        deadline_microseconds = (
            arrival_interval_microseconds
        )

    if deadline_microseconds <= 0.0:
        raise ValueError(
            "deadline_microseconds must be positive"
        )

    worker_available_times = np.zeros(
        number_of_workers,
        dtype=float,
    )

    number_of_jobs = len(
        service_times
    )

    arrival_times = (
        np.arange(
            number_of_jobs,
            dtype=float,
        )
        * arrival_interval_microseconds
    )

    start_times = np.empty(
        number_of_jobs,
        dtype=float,
    )

    completion_times = np.empty(
        number_of_jobs,
        dtype=float,
    )

    waiting_times = np.empty(
        number_of_jobs,
        dtype=float,
    )

    response_times = np.empty(
        number_of_jobs,
        dtype=float,
    )

    assigned_workers = np.empty(
        number_of_jobs,
        dtype=int,
    )

    for job_index in range(
        number_of_jobs
    ):
        worker_index = int(
            np.argmin(
                worker_available_times
            )
        )

        start_time = max(
            arrival_times[job_index],
            worker_available_times[
                worker_index
            ],
        )

        completion_time = (
            start_time
            + service_times[job_index]
        )

        worker_available_times[
            worker_index
        ] = completion_time

        start_times[job_index] = (
            start_time
        )

        completion_times[job_index] = (
            completion_time
        )

        waiting_times[job_index] = (
            start_time
            - arrival_times[job_index]
        )

        response_times[job_index] = (
            completion_time
            - arrival_times[job_index]
        )

        assigned_workers[job_index] = (
            worker_index
        )

    deadline_missed = (
        response_times
        > deadline_microseconds
    )

    return pd.DataFrame(
        {
            "job_index": np.arange(
                number_of_jobs
            ),
            "arrival_time_microseconds": (
                arrival_times
            ),
            "worker": assigned_workers,
            "service_time_microseconds": (
                service_times
            ),
            "start_time_microseconds": (
                start_times
            ),
            "completion_time_microseconds": (
                completion_times
            ),
            "waiting_time_microseconds": (
                waiting_times
            ),
            "response_time_microseconds": (
                response_times
            ),
            "deadline_microseconds": (
                deadline_microseconds
            ),
            "deadline_missed": (
                deadline_missed
            ),
        }
    )


def calculate_backlog_at_arrivals(
    queue_dataframe: pd.DataFrame,
) -> np.ndarray:
    """Calculate unfinished prior jobs at each arrival."""

    required_columns = {
        "arrival_time_microseconds",
        "completion_time_microseconds",
    }

    missing_columns = (
        required_columns
        - set(queue_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "queue_dataframe is missing: "
            f"{sorted(missing_columns)}"
        )

    if queue_dataframe.empty:
        raise ValueError(
            "queue_dataframe cannot be empty"
        )

    arrival_times = queue_dataframe[
        "arrival_time_microseconds"
    ].to_numpy()

    completion_times = np.sort(
        queue_dataframe[
            "completion_time_microseconds"
        ].to_numpy()
    )

    completed_by_arrival = np.searchsorted(
        completion_times,
        arrival_times,
        side="right",
    )

    previously_arrived_jobs = np.arange(
        len(queue_dataframe)
    )

    backlog = (
        previously_arrived_jobs
        - completed_by_arrival
    )

    return np.maximum(
        backlog,
        0,
    )


def summarise_queue(
    queue_dataframe: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarise queue waiting, response and backlog."""

    required_columns = {
        "waiting_time_microseconds",
        "response_time_microseconds",
        "deadline_missed",
    }

    missing_columns = (
        required_columns
        - set(queue_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "queue_dataframe is missing: "
            f"{sorted(missing_columns)}"
        )

    if queue_dataframe.empty:
        raise ValueError(
            "queue_dataframe cannot be empty"
        )

    backlog = calculate_backlog_at_arrivals(
        queue_dataframe
    )

    waiting = queue_dataframe[
        "waiting_time_microseconds"
    ]

    response = queue_dataframe[
        "response_time_microseconds"
    ]

    return {
        "jobs": int(
            len(queue_dataframe)
        ),
        "mean_waiting_time_microseconds": (
            float(
                waiting.mean()
            )
        ),
        "p95_waiting_time_microseconds": (
            float(
                waiting.quantile(0.95)
            )
        ),
        "p99_waiting_time_microseconds": (
            float(
                waiting.quantile(0.99)
            )
        ),
        "maximum_waiting_time_microseconds": (
            float(
                waiting.max()
            )
        ),
        "mean_response_time_microseconds": (
            float(
                response.mean()
            )
        ),
        "p99_response_time_microseconds": (
            float(
                response.quantile(0.99)
            )
        ),
        "maximum_response_time_microseconds": (
            float(
                response.max()
            )
        ),
        "deadline_miss_rate": float(
            queue_dataframe[
                "deadline_missed"
            ].mean()
        ),
        "maximum_backlog": int(
            backlog.max()
        ),
        "final_backlog": int(
            backlog[-1]
        ),
    }