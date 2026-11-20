$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Karthi\Qauntum"

if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    throw "Project directory does not exist: $ProjectRoot"
}

Set-Location -LiteralPath $ProjectRoot

Write-Host ""
Write-Host "Creating project structure in:" -ForegroundColor Cyan
Write-Host $ProjectRoot -ForegroundColor Cyan
Write-Host ""

$Directories = @(
    "src",
    "src\quantum_qec",
    "tests",
    "notebooks",
    "figures",
    "results",
    "reports",
    "scripts",
    "data",
    "data\processed"
)

foreach ($Directory in $Directories) {
    $FullPath = Join-Path $ProjectRoot $Directory

    if (-not (Test-Path -LiteralPath $FullPath)) {
        New-Item `
            -ItemType Directory `
            -Path $FullPath `
            -Force | Out-Null

        Write-Host "Created directory: $Directory" -ForegroundColor Green
    }
    else {
        Write-Host "Directory exists: $Directory" -ForegroundColor DarkGray
    }
}

function New-FileIfMissing {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,

        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $FullPath = Join-Path $ProjectRoot $RelativePath
    $ParentDirectory = Split-Path -Parent $FullPath

    if (
        $ParentDirectory -and
        -not (Test-Path -LiteralPath $ParentDirectory)
    ) {
        New-Item `
            -ItemType Directory `
            -Path $ParentDirectory `
            -Force | Out-Null
    }

    if (-not (Test-Path -LiteralPath $FullPath)) {
        $Utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)

[System.IO.File]::WriteAllText(
    $FullPath,
    $Content,
    $Utf8WithoutBom
)

        Write-Host "Created file: $RelativePath" -ForegroundColor Green
    }
    else {
        Write-Host "Preserved existing file: $RelativePath" -ForegroundColor Yellow
    }
}

$InitContent = @'
"""Quantum error-correction simulation and decoder-analysis package."""

__version__ = "0.1.0"
'@

$CircuitsContent = @'
"""Circuit-generation utilities.

This module will contain reusable functions for:

- Qiskit repetition-code circuits
- Stim repetition-code memory circuits
- Stim rotated surface-code memory circuits
- asymmetric circuit-level noise models
"""

from __future__ import annotations


def validate_odd_distance(distance: int) -> None:
    """Validate an odd code distance of at least three."""

    if not isinstance(distance, int):
        raise TypeError("distance must be an integer")

    if distance < 3:
        raise ValueError("distance must be at least 3")

    if distance % 2 == 0:
        raise ValueError("distance must be odd")
'@

$DecodingContent = @'
"""Decoder construction and logical-failure utilities."""

from __future__ import annotations

import numpy as np


def calculate_logical_failures(
    predicted_observables: np.ndarray,
    actual_observables: np.ndarray,
) -> np.ndarray:
    """Return one Boolean logical-failure value per shot."""

    predicted = np.asarray(predicted_observables)
    actual = np.asarray(actual_observables)

    if predicted.shape != actual.shape:
        raise ValueError(
            "predicted_observables and actual_observables "
            "must have identical shapes"
        )

    if predicted.ndim != 2:
        raise ValueError(
            "observable arrays must be two-dimensional"
        )

    return np.any(
        predicted != actual,
        axis=1,
    )
'@

$SimulationContent = @'
"""Sampling and parameter-sweep utilities.

Functions from the circuit-level Stim, Sinter and PyMatching notebooks
will be consolidated here.
"""

from __future__ import annotations
'@

$StatisticsContent = @'
"""Statistical utilities for logical-error analysis."""

from __future__ import annotations

import math


def binomial_standard_error(
    estimated_probability: float,
    number_of_trials: int,
) -> float:
    """Return the standard error of a binomial probability estimate."""

    if not 0.0 <= estimated_probability <= 1.0:
        raise ValueError(
            "estimated_probability must be between zero and one"
        )

    if number_of_trials <= 0:
        raise ValueError(
            "number_of_trials must be positive"
        )

    return math.sqrt(
        estimated_probability
        * (1.0 - estimated_probability)
        / number_of_trials
    )


def rule_of_three_upper_bound(
    number_of_trials: int,
) -> float:
    """Return the approximate 95% upper bound after zero failures."""

    if number_of_trials <= 0:
        raise ValueError(
            "number_of_trials must be positive"
        )

    return 3.0 / number_of_trials


def repetition_code_logical_failure_probability(
    physical_error_probability: float,
) -> float:
    """Return 3p^2 - 2p^3 for the three-bit repetition code."""

    if not 0.0 <= physical_error_probability <= 1.0:
        raise ValueError(
            "physical_error_probability must be between zero and one"
        )

    p = physical_error_probability

    return 3.0 * p**2 - 2.0 * p**3
'@

$BenchmarkingContent = @'
"""Decoder timing and workload-analysis utilities."""

from __future__ import annotations
'@

$QueueingContent = @'
"""Classical-decoder queue and backlog models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_worker_utilisation(
    mean_service_time_microseconds: float,
    arrival_interval_microseconds: float,
    number_of_workers: int = 1,
) -> float:
    """Calculate the mean utilisation of each decoder worker."""

    if mean_service_time_microseconds <= 0.0:
        raise ValueError(
            "mean_service_time_microseconds must be positive"
        )

    if arrival_interval_microseconds <= 0.0:
        raise ValueError(
            "arrival_interval_microseconds must be positive"
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


def simulate_decoder_queue(
    service_times_microseconds,
    arrival_interval_microseconds: float,
    number_of_workers: int = 1,
    deadline_microseconds: float | None = None,
) -> pd.DataFrame:
    """Simulate fixed-rate arrivals and parallel decoder workers."""

    service_times = np.asarray(
        service_times_microseconds,
        dtype=float,
    )

    if service_times.ndim != 1:
        raise ValueError(
            "service_times_microseconds must be one-dimensional"
        )

    if len(service_times) == 0:
        raise ValueError(
            "at least one service time is required"
        )

    if np.any(service_times <= 0.0):
        raise ValueError(
            "all service times must be positive"
        )

    if arrival_interval_microseconds <= 0.0:
        raise ValueError(
            "arrival_interval_microseconds must be positive"
        )

    if number_of_workers <= 0:
        raise ValueError(
            "number_of_workers must be positive"
        )

    if deadline_microseconds is None:
        deadline_microseconds = arrival_interval_microseconds

    worker_available_times = np.zeros(
        number_of_workers,
        dtype=float,
    )

    number_of_jobs = len(service_times)

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

    for job_index in range(number_of_jobs):
        worker_index = int(
            np.argmin(worker_available_times)
        )

        start_time = max(
            arrival_times[job_index],
            worker_available_times[worker_index],
        )

        completion_time = (
            start_time
            + service_times[job_index]
        )

        worker_available_times[worker_index] = (
            completion_time
        )

        start_times[job_index] = start_time
        completion_times[job_index] = completion_time
        waiting_times[job_index] = (
            start_time
            - arrival_times[job_index]
        )
        response_times[job_index] = (
            completion_time
            - arrival_times[job_index]
        )
        assigned_workers[job_index] = worker_index

    return pd.DataFrame(
        {
            "job_index": np.arange(number_of_jobs),
            "arrival_time_microseconds": arrival_times,
            "worker": assigned_workers,
            "service_time_microseconds": service_times,
            "start_time_microseconds": start_times,
            "completion_time_microseconds": completion_times,
            "waiting_time_microseconds": waiting_times,
            "response_time_microseconds": response_times,
            "deadline_microseconds": deadline_microseconds,
            "deadline_missed": (
                response_times
                > deadline_microseconds
            ),
        }
    )
'@

$TestStatisticsContent = @'
"""Tests for statistical utilities."""

import math

from quantum_qec.statistics import (
    binomial_standard_error,
    repetition_code_logical_failure_probability,
    rule_of_three_upper_bound,
)


def test_repetition_code_failure_at_zero():
    assert (
        repetition_code_logical_failure_probability(0.0)
        == 0.0
    )


def test_repetition_code_failure_at_half():
    assert math.isclose(
        repetition_code_logical_failure_probability(0.5),
        0.5,
    )


def test_repetition_code_failure_at_one():
    assert (
        repetition_code_logical_failure_probability(1.0)
        == 1.0
    )


def test_rule_of_three():
    assert math.isclose(
        rule_of_three_upper_bound(1_000_000),
        3e-6,
    )


def test_binomial_standard_error_is_positive():
    assert (
        binomial_standard_error(
            estimated_probability=0.1,
            number_of_trials=10_000,
        )
        > 0.0
    )
'@

$TestQueueingContent = @'
"""Tests for decoder queue utilities."""

import numpy as np

from quantum_qec.queueing import (
    calculate_worker_utilisation,
    simulate_decoder_queue,
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
        queue["waiting_time_microseconds"],
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
'@

$TestDecodingContent = @'
"""Tests for logical-failure calculations."""

import numpy as np

from quantum_qec.decoding import (
    calculate_logical_failures,
)


def test_logical_failure_array():
    predicted = np.array(
        [
            [False],
            [True],
            [False],
        ]
    )

    actual = np.array(
        [
            [False],
            [False],
            [False],
        ]
    )

    failures = calculate_logical_failures(
        predicted_observables=predicted,
        actual_observables=actual,
    )

    assert failures.tolist() == [
        False,
        True,
        False,
    ]
'@

$TestCircuitsContent = @'
"""Tests for circuit-validation utilities."""

import pytest

from quantum_qec.circuits import (
    validate_odd_distance,
)


def test_distance_three_is_valid():
    validate_odd_distance(3)


def test_even_distance_is_invalid():
    with pytest.raises(ValueError):
        validate_odd_distance(4)


def test_distance_below_three_is_invalid():
    with pytest.raises(ValueError):
        validate_odd_distance(1)
'@

$GitIgnoreContent = @'
# Python
__pycache__/
*.py[cod]
*.pyd
*.so
.pytest_cache/
.coverage
htmlcov/

# Virtual environments
.qiskit/
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# Build and package output
build/
dist/
*.egg-info/

# Editors
.vscode/
.idea/

# Operating-system files
Thumbs.db
.DS_Store

# Temporary files
*.tmp
*.temp
*.log
~$*

# Large or generated raw sampling files
results/*.b8
results/*.dets
results/*.obs
results/*.r8
results/*.hits

# Keep CSV summaries, figures, notebooks and reports tracked
'@

$PyProjectContent = @'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "quantum-qec-analysis"
version = "0.1.0"
description = "Circuit-level quantum error-correction simulation and classical decoder performance analysis."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "numpy",
    "pandas",
    "matplotlib",
    "qiskit",
    "qiskit-aer",
    "stim",
    "sinter",
    "pymatching",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "jupyterlab",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
'@

$ReadmeContent = @'
# Circuit-Level Quantum Error Correction and Decoder Analysis

This project investigates quantum error correction as a quantum-classical
systems problem.

The project uses:

- Qiskit and Qiskit Aer for introductory circuit and noise experiments
- Stim for circuit-level stabilizer simulation
- Sinter for adaptive parallel Monte Carlo collection
- PyMatching for minimum-weight perfect-matching decoding
- Python for logical-error, latency, throughput and queue analysis

## Current research subjects

- quantum-circuit and noise preparation
- three-qubit repetition-code implementation
- syndrome extraction and correction
- circuit-level repetition-code simulation
- rotated surface-code memory simulation
- preliminary surface-code threshold estimation
- classical decoder latency and backlog modelling
- decoder complexity analysis
- asymmetric-noise sensitivity analysis

## Project status

The simulation and analysis notebooks are complete. Core functions are
being consolidated into a reusable Python package with automated tests.

## Installation

From PowerShell:

    python -m pip install -e ".[dev]"

## Tests

    python -m pytest

## Important limitations

The threshold estimate is preliminary and depends on the selected
circuit-level Pauli noise model, finite code distances and PyMatching.

CPU decoder timings depend on the host computer, Python environment,
package versions and benchmark methodology.

FPGA acceleration is future work and has not yet been implemented.
'@

$ResultsReadmeContent = @'
# Results directory

This directory contains generated CSV summaries, Stim circuit files,
detector error models and exported notebook reports.

Important final results should include:

- repetition-code logical-error sweep
- surface-code threshold summary
- decoder latency summary
- decoder queue sweep
- decoder complexity summary
- asymmetric-noise sensitivity summary

Large intermediate binary samples should not be committed to Git.
'@

$FiguresReadmeContent = @'
# Figures directory

This directory contains final plots generated by the analysis notebooks.

Recommended headline figures:

1. repetition-code physical versus logical error probability
2. surface-code threshold curves
3. decoder latency or backlog
4. decoder complexity scaling
5. asymmetric-noise sensitivity ranking
'@

$ReportReadmeContent = @'
# Reports directory

The final technical report should cover:

1. introduction and motivation
2. repetition-code validation
3. circuit-level simulation and decoding
4. surface-code threshold study
5. decoder latency and backlog
6. asymmetric-noise sensitivity
7. limitations and future FPGA work
'@

New-FileIfMissing `
    -RelativePath "src\quantum_qec\__init__.py" `
    -Content $InitContent

New-FileIfMissing `
    -RelativePath "src\quantum_qec\circuits.py" `
    -Content $CircuitsContent

New-FileIfMissing `
    -RelativePath "src\quantum_qec\decoding.py" `
    -Content $DecodingContent

New-FileIfMissing `
    -RelativePath "src\quantum_qec\simulation.py" `
    -Content $SimulationContent

New-FileIfMissing `
    -RelativePath "src\quantum_qec\statistics.py" `
    -Content $StatisticsContent

New-FileIfMissing `
    -RelativePath "src\quantum_qec\benchmarking.py" `
    -Content $BenchmarkingContent

New-FileIfMissing `
    -RelativePath "src\quantum_qec\queueing.py" `
    -Content $QueueingContent

New-FileIfMissing `
    -RelativePath "tests\test_statistics.py" `
    -Content $TestStatisticsContent

New-FileIfMissing `
    -RelativePath "tests\test_queueing.py" `
    -Content $TestQueueingContent

New-FileIfMissing `
    -RelativePath "tests\test_decoding.py" `
    -Content $TestDecodingContent

New-FileIfMissing `
    -RelativePath "tests\test_circuits.py" `
    -Content $TestCircuitsContent

New-FileIfMissing `
    -RelativePath ".gitignore" `
    -Content $GitIgnoreContent

New-FileIfMissing `
    -RelativePath "pyproject.toml" `
    -Content $PyProjectContent

New-FileIfMissing `
    -RelativePath "README.md" `
    -Content $ReadmeContent

New-FileIfMissing `
    -RelativePath "results\README.md" `
    -Content $ResultsReadmeContent

New-FileIfMissing `
    -RelativePath "figures\README.md" `
    -Content $FiguresReadmeContent

New-FileIfMissing `
    -RelativePath "reports\README.md" `
    -Content $ReportReadmeContent

Write-Host ""
Write-Host "Installing the project in editable mode..." -ForegroundColor Cyan

python -m pip install -e ".[dev]"

if ($LASTEXITCODE -ne 0) {
    throw "Editable package installation failed."
}

Write-Host ""
Write-Host "Running the initial automated tests..." -ForegroundColor Cyan

python -m pytest

if ($LASTEXITCODE -ne 0) {
    throw "One or more initial tests failed."
}

Write-Host ""
Write-Host "Saving the current Python environment..." -ForegroundColor Cyan

python -m pip freeze |
    Set-Content `
        -LiteralPath (
            Join-Path `
                $ProjectRoot `
                "requirements-lock.txt"
        ) `
        -Encoding UTF8

if ($LASTEXITCODE -ne 0) {
    throw "Could not create requirements-lock.txt."
}

Write-Host ""
Write-Host "Project structure created successfully." -ForegroundColor Green
Write-Host ""

Write-Host "Top-level contents:" -ForegroundColor Cyan

Get-ChildItem `
    -LiteralPath $ProjectRoot |
    Sort-Object Name |
    Format-Table `
        Mode, `
        LastWriteTime, `
        Length, `
        Name `
        -AutoSize

Write-Host ""
Write-Host "Python package files:" -ForegroundColor Cyan

Get-ChildItem `
    -LiteralPath (
        Join-Path `
            $ProjectRoot `
            "src\quantum_qec"
    ) `
    -File |
    Sort-Object Name |
    Select-Object Name

Write-Host ""
Write-Host "Test files:" -ForegroundColor Cyan

Get-ChildItem `
    -LiteralPath (
        Join-Path `
            $ProjectRoot `
            "tests"
    ) `
    -File |
    Sort-Object Name |
    Select-Object Name

Write-Host ""
Write-Host "Next command:" -ForegroundColor Cyan
Write-Host 'python -m pytest -v'