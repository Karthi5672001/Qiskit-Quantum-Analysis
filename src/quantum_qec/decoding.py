"""Decoder construction and logical-failure utilities."""

from __future__ import annotations

import numpy as np
import pymatching
import stim


def create_matching_decoder(
    detector_error_model: stim.DetectorErrorModel,
) -> pymatching.Matching:
    """Create a PyMatching decoder from a detector error model."""

    if not isinstance(
        detector_error_model,
        stim.DetectorErrorModel,
    ):
        raise TypeError(
            "detector_error_model must be a "
            "stim.DetectorErrorModel"
        )

    if detector_error_model.num_detectors <= 0:
        raise ValueError(
            "detector_error_model must contain detectors"
        )

    return (
        pymatching.Matching
        .from_detector_error_model(
            detector_error_model
        )
    )


def validate_observable_arrays(
    predicted_observables: np.ndarray,
    actual_observables: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return two observable arrays."""

    predicted = np.asarray(
        predicted_observables
    )

    actual = np.asarray(
        actual_observables
    )

    if predicted.shape != actual.shape:
        raise ValueError(
            "predicted_observables and "
            "actual_observables must have "
            "identical shapes"
        )

    if predicted.ndim != 2:
        raise ValueError(
            "observable arrays must be "
            "two-dimensional"
        )

    if predicted.shape[0] == 0:
        raise ValueError(
            "observable arrays must contain "
            "at least one shot"
        )

    return predicted, actual


def calculate_logical_failures(
    predicted_observables: np.ndarray,
    actual_observables: np.ndarray,
) -> np.ndarray:
    """Return one Boolean logical-failure value per shot."""

    predicted, actual = (
        validate_observable_arrays(
            predicted_observables,
            actual_observables,
        )
    )

    return np.any(
        predicted != actual,
        axis=1,
    )


def calculate_logical_failure_count(
    predicted_observables: np.ndarray,
    actual_observables: np.ndarray,
) -> int:
    """Return the number of logical decoding failures."""

    failures = calculate_logical_failures(
        predicted_observables,
        actual_observables,
    )

    return int(
        failures.sum()
    )


def calculate_logical_failure_probability(
    predicted_observables: np.ndarray,
    actual_observables: np.ndarray,
) -> float:
    """Return the observed logical-failure probability."""

    failures = calculate_logical_failures(
        predicted_observables,
        actual_observables,
    )

    return float(
        failures.mean()
    )


def decode_detection_events(
    decoder: pymatching.Matching,
    detection_events: np.ndarray,
) -> np.ndarray:
    """Decode a batch of detector samples."""

    if not isinstance(
        decoder,
        pymatching.Matching,
    ):
        raise TypeError(
            "decoder must be a "
            "pymatching.Matching instance"
        )

    detector_samples = np.asarray(
        detection_events
    )

    if detector_samples.ndim != 2:
        raise ValueError(
            "detection_events must be "
            "two-dimensional"
        )

    if detector_samples.shape[0] == 0:
        raise ValueError(
            "detection_events must contain "
            "at least one shot"
        )

    if (
        detector_samples.shape[1]
        != decoder.num_detectors
    ):
        raise ValueError(
            "detection-event width does not "
            "match the decoder detector count"
        )

    return decoder.decode_batch(
        detector_samples
    )