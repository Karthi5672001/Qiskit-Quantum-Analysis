"""Tests for decoder queue utilities."""

import numpy as np

from quantum_qec.queueing import (
    calculate_backlog_at_arrivals,
    calculate_worker_utilisation,
    minimum_workers_for_utilisation,
    simulate_decoder_queue,
    summarise_queue,
)


def test_worker_utilisation():
    utilisation = calculate_worker_utilisation(
        mean_service_time_microseconds=2.0,
        arrival_interval_microseconds=4.0,
        number_of_workers=1,
    )

    assert np.isclose(
        utilisation,
        0.5,
    )


def test_parallel_workers_reduce_utilisation():
    one_worker = calculate_worker_utilisation(
        mean_service_time_microseconds=4.0,
        arrival_interval_microseconds=2.0,
        number_of_workers=1,
    )

    four_workers = calculate_worker_utilisation(
        mean_service_time_microseconds=4.0,
        arrival_interval_microseconds=2.0,
        number_of_workers=4,
    )

    assert four_workers < one_worker


def test_minimum_workers_for_target_utilisation():
    workers = minimum_workers_for_utilisation(
        mean_service_time_microseconds=7.0,
        arrival_interval_microseconds=2.0,
        target_utilisation=0.70,
    )

    assert workers == 5


def test_stable_queue_has_no_waiting():
    queue = simulate_decoder_queue(
        service_times_microseconds=[
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        arrival_interval_microseconds=2.0,
        number_of_workers=1,
        deadline_microseconds=2.0,
    )

    assert np.allclose(
        queue[
            "waiting_time_microseconds"
        ],
        0.0,
    )

    assert not queue[
        "deadline_missed"
    ].any()


def test_overloaded_queue_accumulates_waiting():
    queue = simulate_decoder_queue(
        service_times_microseconds=[
            2.0,
            2.0,
            2.0,
            2.0,
        ],
        arrival_interval_microseconds=1.0,
        number_of_workers=1,
        deadline_microseconds=2.0,
    )

    assert (
        queue[
            "waiting_time_microseconds"
        ].iloc[-1]
        > 0.0
    )


def test_backlog_for_stable_queue_is_zero():
    queue = simulate_decoder_queue(
        service_times_microseconds=[
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        arrival_interval_microseconds=2.0,
        number_of_workers=1,
    )

    backlog = calculate_backlog_at_arrivals(
        queue
    )

    assert np.all(
        backlog == 0
    )


def test_queue_summary_reports_deadline_misses():
    queue = simulate_decoder_queue(
        service_times_microseconds=[
            3.0,
            3.0,
            3.0,
            3.0,
        ],
        arrival_interval_microseconds=1.0,
        number_of_workers=1,
        deadline_microseconds=2.0,
    )

    summary = summarise_queue(
        queue
    )

    assert summary["jobs"] == 4

    assert (
        summary["deadline_miss_rate"]
        > 0.0
    )

    assert (
        summary["maximum_backlog"]
        > 0
    )