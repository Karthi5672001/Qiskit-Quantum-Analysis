"""GPU-accelerated detector-event preprocessing.

The functions in this module compare NumPy CPU processing with CuPy GPU
processing. GPU timing is explicitly synchronised because CUDA operations
are asynchronous.

CuPy is optional. Importing quantum_qec must continue to work on systems
without CUDA or CuPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


@dataclass(frozen=True)
class GpuDeviceInformation:
    """Information about one CUDA device."""

    available: bool
    device_count: int
    device_id: int | None
    name: str | None
    compute_capability: str | None
    total_memory_bytes: int | None
    free_memory_bytes: int | None
    runtime_version: int | None
    driver_version: int | None


@dataclass(frozen=True)
class PreprocessingTiming:
    """Timing results for one preprocessing operation."""

    operation: str
    shots: int
    detectors: int
    input_bytes: int
    repeats: int
    cpu_seconds: float
    host_to_device_seconds: float
    gpu_kernel_seconds: float
    device_to_host_seconds: float
    gpu_end_to_end_seconds: float

    @property
    def resident_gpu_speedup(self) -> float:
        """Return CPU time divided by resident GPU kernel time."""

        if self.gpu_kernel_seconds <= 0.0:
            return float("nan")

        return (
            self.cpu_seconds
            / self.gpu_kernel_seconds
        )

    @property
    def end_to_end_gpu_speedup(self) -> float:
        """Return CPU time divided by total GPU pipeline time."""

        if self.gpu_end_to_end_seconds <= 0.0:
            return float("nan")

        return (
            self.cpu_seconds
            / self.gpu_end_to_end_seconds
        )

    @property
    def cpu_shots_per_second(self) -> float:
        """Return CPU preprocessing throughput."""

        return self.shots / self.cpu_seconds

    @property
    def gpu_kernel_shots_per_second(self) -> float:
        """Return resident GPU preprocessing throughput."""

        return self.shots / self.gpu_kernel_seconds

    @property
    def gpu_end_to_end_shots_per_second(self) -> float:
        """Return end-to-end GPU preprocessing throughput."""

        return self.shots / self.gpu_end_to_end_seconds


def cupy_available() -> bool:
    """Return whether CuPy can access at least one CUDA device."""

    if cp is None:
        return False

    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def require_cupy() -> Any:
    """Return CuPy or raise a clear runtime error."""

    if cp is None:
        raise RuntimeError(
            "CuPy is not installed. Install the optional GPU dependencies."
        )

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        raise RuntimeError(
            "CuPy is installed but CUDA initialisation failed."
        ) from error

    if device_count <= 0:
        raise RuntimeError(
            "CuPy is installed but no CUDA device is available."
        )

    return cp


def synchronise_gpu() -> None:
    """Wait for all work on CuPy's default CUDA stream."""

    cupy_module = require_cupy()
    cupy_module.cuda.Stream.null.synchronize()


def get_gpu_device_information(
    device_id: int = 0,
) -> GpuDeviceInformation:
    """Return information about one CUDA device."""

    if not cupy_available():
        return GpuDeviceInformation(
            available=False,
            device_count=0,
            device_id=None,
            name=None,
            compute_capability=None,
            total_memory_bytes=None,
            free_memory_bytes=None,
            runtime_version=None,
            driver_version=None,
        )

    cupy_module = require_cupy()

    device_count = int(
        cupy_module.cuda.runtime.getDeviceCount()
    )

    if not 0 <= device_id < device_count:
        raise ValueError(
            "device_id is outside the available CUDA-device range"
        )

    properties = (
        cupy_module.cuda.runtime.getDeviceProperties(
            device_id
        )
    )

    raw_name = properties["name"]

    if isinstance(raw_name, bytes):
        device_name = raw_name.decode()
    else:
        device_name = str(raw_name)

    major = int(properties["major"])
    minor = int(properties["minor"])

    with cupy_module.cuda.Device(device_id):
        free_memory, total_memory = (
            cupy_module.cuda.runtime.memGetInfo()
        )

    return GpuDeviceInformation(
        available=True,
        device_count=device_count,
        device_id=device_id,
        name=device_name,
        compute_capability=f"{major}.{minor}",
        total_memory_bytes=int(total_memory),
        free_memory_bytes=int(free_memory),
        runtime_version=int(
            cupy_module.cuda.runtime.runtimeGetVersion()
        ),
        driver_version=int(
            cupy_module.cuda.runtime.driverGetVersion()
        ),
    )


def validate_detector_events(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Validate and normalise a CPU detector-event matrix."""

    events = np.asarray(detection_events)

    if events.ndim != 2:
        raise ValueError(
            "detection_events must be a two-dimensional array"
        )

    if events.shape[0] == 0:
        raise ValueError(
            "detection_events must contain at least one shot"
        )

    if events.shape[1] == 0:
        raise ValueError(
            "detection_events must contain at least one detector"
        )

    if events.dtype == np.bool_:
        return events

    if not np.issubdtype(events.dtype, np.integer):
        raise TypeError(
            "detection_events must contain Boolean or integer values"
        )

    if np.any((events != 0) & (events != 1)):
        raise ValueError(
            "integer detection_events must contain only zero and one"
        )

    return events.astype(
        np.bool_,
        copy=False,
    )


def cpu_event_counts(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Count active detectors in each shot using NumPy."""

    events = validate_detector_events(
        detection_events
    )

    return events.sum(
        axis=1,
        dtype=np.int64,
    )


def cpu_event_density(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Calculate detector-event density for each shot using NumPy."""

    events = validate_detector_events(
        detection_events
    )

    return events.mean(
        axis=1,
        dtype=np.float64,
    )


def cpu_zero_event_mask(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Return True for shots containing no detector events."""

    events = validate_detector_events(
        detection_events
    )

    return ~events.any(
        axis=1
    )


def cpu_pack_detector_bits(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Pack detector bits along each shot using NumPy."""

    events = validate_detector_events(
        detection_events
    )

    return np.packbits(
        events,
        axis=1,
        bitorder="little",
    )


def gpu_event_counts(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Count active detectors per shot using CuPy."""

    cupy_module = require_cupy()

    events = validate_detector_events(
        detection_events
    )

    gpu_events = cupy_module.asarray(
        events
    )

    gpu_counts = gpu_events.sum(
        axis=1,
        dtype=cupy_module.int64,
    )

    return cupy_module.asnumpy(
        gpu_counts
    )


def gpu_event_density(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Calculate detector-event density per shot using CuPy."""

    cupy_module = require_cupy()

    events = validate_detector_events(
        detection_events
    )

    gpu_events = cupy_module.asarray(
        events
    )

    gpu_density = gpu_events.mean(
        axis=1,
        dtype=cupy_module.float64,
    )

    return cupy_module.asnumpy(
        gpu_density
    )


def gpu_zero_event_mask(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Return a CPU mask identifying zero-event shots using CuPy."""

    cupy_module = require_cupy()

    events = validate_detector_events(
        detection_events
    )

    gpu_events = cupy_module.asarray(
        events
    )

    gpu_mask = ~gpu_events.any(
        axis=1
    )

    return cupy_module.asnumpy(
        gpu_mask
    )


def gpu_pack_detector_bits(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Pack detector bits along each shot using CuPy.

    CuPy 14.2 does not support the ``axis`` argument of
    ``cupy.packbits``. This implementation therefore packs each
    detector row explicitly on the GPU.

    Detectors are padded with zeros to a multiple of eight. Each
    consecutive group of eight detector values is converted into one
    byte using little-endian bit order, matching:

        numpy.packbits(events, axis=1, bitorder="little")
    """

    cupy_module = require_cupy()

    events = validate_detector_events(
        detection_events
    )

    gpu_events = cupy_module.asarray(
        events,
        dtype=cupy_module.uint8,
    )

    shot_count = int(
        gpu_events.shape[0]
    )

    detector_count = int(
        gpu_events.shape[1]
    )

    padding_bits = (
        -detector_count
    ) % 8

    if padding_bits:
        gpu_events = cupy_module.pad(
            gpu_events,
            pad_width=(
                (0, 0),
                (0, padding_bits),
            ),
            mode="constant",
            constant_values=0,
        )

    packed_width = int(
        gpu_events.shape[1] // 8
    )

    grouped_bits = gpu_events.reshape(
        shot_count,
        packed_width,
        8,
    )

    bit_weights = cupy_module.asarray(
        [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
        ],
        dtype=cupy_module.uint16,
    )

    gpu_packed = (
        grouped_bits.astype(
            cupy_module.uint16,
            copy=False,
        )
        * bit_weights
    ).sum(
        axis=2,
        dtype=cupy_module.uint16,
    ).astype(
        cupy_module.uint8,
        copy=False,
    )

    return cupy_module.asnumpy(
        gpu_packed
    )
def _median_elapsed_seconds(
    operation,
    repeats: int,
) -> float:
    """Return median elapsed wall-clock time."""

    if repeats <= 0:
        raise ValueError(
            "repeats must be positive"
        )

    elapsed = np.empty(
        repeats,
        dtype=np.float64,
    )

    for repeat_index in range(repeats):
        start_time = perf_counter()

        operation()

        end_time = perf_counter()

        elapsed[repeat_index] = (
            end_time - start_time
        )

    return float(
        np.median(elapsed)
    )


def benchmark_event_counting(
    detection_events: np.ndarray,
    *,
    repeats: int = 10,
    warm_up_repeats: int = 3,
) -> PreprocessingTiming:
    """Benchmark CPU and GPU event counting.

    GPU timings are separated into:

    - host-to-device transfer;
    - resident GPU kernel execution;
    - device-to-host result transfer;
    - end-to-end processing.

    The end-to-end result is the relevant comparison when Stim has produced
    detector data in host memory.
    """

    cupy_module = require_cupy()

    events = validate_detector_events(
        detection_events
    )

    if repeats <= 0:
        raise ValueError(
            "repeats must be positive"
        )

    if warm_up_repeats < 0:
        raise ValueError(
            "warm_up_repeats cannot be negative"
        )

    for _ in range(warm_up_repeats):
        cpu_event_counts(events)

    cpu_seconds = _median_elapsed_seconds(
        lambda: cpu_event_counts(events),
        repeats,
    )

    gpu_events = cupy_module.asarray(
        events
    )

    for _ in range(warm_up_repeats):
        gpu_events.sum(
            axis=1,
            dtype=cupy_module.int64,
        )
        synchronise_gpu()

    host_to_device_seconds = _median_elapsed_seconds(
        lambda: _timed_host_to_device(events),
        repeats,
    )

    def resident_kernel() -> None:
        gpu_events.sum(
            axis=1,
            dtype=cupy_module.int64,
        )
        synchronise_gpu()

    gpu_kernel_seconds = _median_elapsed_seconds(
        resident_kernel,
        repeats,
    )

    gpu_counts = gpu_events.sum(
        axis=1,
        dtype=cupy_module.int64,
    )
    synchronise_gpu()

    def device_to_host() -> None:
        cupy_module.asnumpy(
            gpu_counts
        )
        synchronise_gpu()

    device_to_host_seconds = _median_elapsed_seconds(
        device_to_host,
        repeats,
    )

    def end_to_end() -> None:
        local_gpu_events = cupy_module.asarray(
            events
        )

        local_gpu_counts = local_gpu_events.sum(
            axis=1,
            dtype=cupy_module.int64,
        )

        cupy_module.asnumpy(
            local_gpu_counts
        )

        synchronise_gpu()

    gpu_end_to_end_seconds = _median_elapsed_seconds(
        end_to_end,
        repeats,
    )

    return PreprocessingTiming(
        operation="event_count",
        shots=int(events.shape[0]),
        detectors=int(events.shape[1]),
        input_bytes=int(events.nbytes),
        repeats=repeats,
        cpu_seconds=cpu_seconds,
        host_to_device_seconds=host_to_device_seconds,
        gpu_kernel_seconds=gpu_kernel_seconds,
        device_to_host_seconds=device_to_host_seconds,
        gpu_end_to_end_seconds=gpu_end_to_end_seconds,
    )


def _timed_host_to_device(
    events: np.ndarray,
) -> None:
    """Copy an array to the GPU and synchronise."""

    cupy_module = require_cupy()

    cupy_module.asarray(events)

    synchronise_gpu()


def estimate_detector_matrix_bytes(
    shots: int,
    detectors: int,
    bytes_per_event: int = 1,
) -> int:
    """Estimate detector-matrix storage in bytes."""

    if shots <= 0:
        raise ValueError(
            "shots must be positive"
        )

    if detectors <= 0:
        raise ValueError(
            "detectors must be positive"
        )

    if bytes_per_event <= 0:
        raise ValueError(
            "bytes_per_event must be positive"
        )

    return (
        shots
        * detectors
        * bytes_per_event
    )


def choose_safe_gpu_batch_size(
    detectors: int,
    *,
    available_memory_bytes: int | None = None,
    memory_fraction: float = 0.50,
    working_array_multiplier: float = 2.0,
) -> int:
    """Estimate a conservative maximum Boolean detector batch size.

    This is a planning estimate. Actual memory use depends on CuPy's memory
    pool, temporary arrays, result arrays, driver allocations and other
    processes using the GPU.
    """

    if detectors <= 0:
        raise ValueError(
            "detectors must be positive"
        )

    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError(
            "memory_fraction must be greater than zero and at most one"
        )

    if working_array_multiplier < 1.0:
        raise ValueError(
            "working_array_multiplier must be at least one"
        )

    if available_memory_bytes is None:
        device_information = (
            get_gpu_device_information()
        )

        if not device_information.available:
            raise RuntimeError(
                "No CUDA device is available."
            )

        available_memory_bytes = (
            device_information.free_memory_bytes
        )

    if (
        available_memory_bytes is None
        or available_memory_bytes <= 0
    ):
        raise ValueError(
            "available_memory_bytes must be positive"
        )

    safe_bytes = (
        available_memory_bytes
        * memory_fraction
    )

    bytes_per_shot = (
        detectors
        * working_array_multiplier
    )

    return max(
        1,
        int(
            safe_bytes
            // bytes_per_shot
        ),
    )