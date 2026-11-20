# The audit records:
#
# - Python and operating-system information
# - installed package versions
# - automated-test status
# - package source files
# - notebooks and execution state
# - generated results and figures
# - missing headline artifacts
# - absolute-path references inside notebooks
# - file sizes and modification times
#
# Run from the project root:
#
#     python3 scripts/audit_project.py

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(
    r"C:\Users\Karthi\Qauntum"
).resolve()

SOURCE_DIR = PROJECT_ROOT / "src" / "quantum_qec"
TEST_DIR = PROJECT_ROOT / "tests"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

AUDIT_JSON_PATH = RESULTS_DIR / "project_audit.json"
AUDIT_TEXT_PATH = RESULTS_DIR / "project_audit.txt"

PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "matplotlib",
    "qiskit",
    "qiskit-aer",
    "stim",
    "sinter",
    "pymatching",
    "cupy-cuda13x",
    "pytest",
    "jupyterlab",
]

REQUIRED_FILES = [
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / ".gitignore",
    PROJECT_ROOT / "requirements-lock.txt",
    SOURCE_DIR / "__init__.py",
    SOURCE_DIR / "circuits.py",
    SOURCE_DIR / "decoding.py",
    SOURCE_DIR / "simulation.py",
    SOURCE_DIR / "statistics.py",
    SOURCE_DIR / "benchmarking.py",
    SOURCE_DIR / "queueing.py",
    SOURCE_DIR / "gpu.py",
]

EXPECTED_NOTEBOOK_SUBJECTS = {
    "quantum_noise": [
        "quantum",
        "noise",
    ],
    "repetition_code": [
        "repetition",
    ],
    "circuit_level_qec": [
        "stim",
    ],
    "surface_code": [
        "surface",
    ],
    "threshold": [
        "threshold",
    ],
    "decoder_latency": [
        "latency",
    ],
    "decoder_complexity": [
        "complexity",
    ],
    "asymmetric_noise": [
        "asymmetric",
    ],
    "gpu_detector_preprocessing": [
        "gpu",
        "detector",
        "preprocessing",
    ],
}

HEADLINE_RESULTS = {
    "threshold": [
        "surface_code_threshold_summary.csv",
        "surface_code_threshold_sinter.csv",
    ],
    "latency": [
        "decoder_latency_summary.csv",
        "decoder_single_shot_latency.csv",
    ],
    "queueing": [
        "decoder_queue_sweep.csv",
    ],
    "complexity": [
        "decoder_complexity_summary.csv",
    ],
    "asymmetric_noise": [
        "asymmetric_noise_sensitivity.csv",
    ],
    "gpu_dense_preprocessing": [
        "gpu_detector_preprocessing.csv",
        "gpu_preprocessing_crossover.csv",
    ],
    "gpu_packed_preprocessing": [
        "gpu_packed_detector_preprocessing.csv",
        "gpu_packed_preprocessing_summary.csv",
    ],
    "gpu_hybrid_filtering": [
        "gpu_hybrid_filtering.csv",
        "gpu_hybrid_filtering_summary.csv",
    ],
}

HEADLINE_FIGURES = {
    "threshold": [
        "surface_code_threshold_sinter.png",
    ],
    "latency": [
        "decoder_latency_distribution.png",
    ],
    "backlog": [
        "decoder_backlog_comparison.png",
    ],
    "complexity": [
        "decoder_batch_cost_by_distance.png",
        "decoder_batch_size_efficiency.png",
    ],
    "asymmetric_noise": [
        "asymmetric_noise_sensitivity_ranking.png",
    ],
    "gpu_dense_preprocessing": [
        "gpu_event_count_crossover.png",
        "gpu_speedup_by_distance.png",
        "gpu_transfer_breakdown.png",
    ],
    "gpu_packed_preprocessing": [
        "gpu_packed_transfer_comparison.png",
    ],
    "gpu_hybrid_filtering": [
        "gpu_hybrid_filtering_pilot.png",
        "gpu_hybrid_filtering_opportunity.png",
    ],
}


def relative_path(path: Path) -> str:
    """Return a repository-relative path."""

    try:
        return path.resolve().relative_to(
            PROJECT_ROOT
        ).as_posix()
    except ValueError:
        return str(path.resolve())


def file_information(path: Path) -> dict[str, object]:
    """Return basic metadata about one file."""

    stat = path.stat()

    return {
        "path": relative_path(path),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def collect_files(
    directory: Path,
    patterns: list[str],
) -> list[Path]:
    """Collect unique files matching the patterns."""

    if not directory.exists():
        return []

    discovered: set[Path] = set()

    for pattern in patterns:
        for path in directory.rglob(pattern):
            if path.is_file():
                discovered.add(path.resolve())

    return sorted(
        discovered,
        key=lambda path: str(path).lower(),
    )


def package_versions() -> dict[str, str | None]:
    """Return package versions."""

    versions: dict[str, str | None] = {}

    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = (
                importlib.metadata.version(
                    package_name
                )
            )
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None

    return versions


def run_tests() -> dict[str, object]:
    """Run the complete pytest suite."""

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def inspect_notebook(
    notebook_path: Path,
) -> dict[str, object]:
    """Inspect notebook execution and saved errors."""

    try:
        notebook = json.loads(
            notebook_path.read_text(
                encoding="utf-8-sig"
            )
        )
    except Exception as error:
        return {
            "path": relative_path(notebook_path),
            "readable": False,
            "error": str(error),
        }

    cells = notebook.get("cells", [])

    code_cells = [
        cell
        for cell in cells
        if cell.get("cell_type") == "code"
    ]

    executed_cells = [
        cell
        for cell in code_cells
        if cell.get("execution_count") is not None
    ]

    saved_errors = []
    absolute_path_cells = []

    for cell_index, cell in enumerate(cells):
        source = "".join(
            cell.get("source", [])
        )

        if (
            r"c:\users\karthi\qauntum"
            in source.lower()
        ):
            absolute_path_cells.append(
                cell_index
            )

        if cell.get("cell_type") != "code":
            continue

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                saved_errors.append(
                    {
                        "cell_index": cell_index,
                        "name": output.get("ename"),
                        "message": output.get("evalue"),
                    }
                )

    return {
        "path": relative_path(notebook_path),
        "readable": True,
        "total_cells": len(cells),
        "code_cells": len(code_cells),
        "executed_code_cells": len(
            executed_cells
        ),
        "completely_executed": (
            len(code_cells) > 0
            and len(executed_cells) == len(code_cells)
        ),
        "saved_errors": saved_errors,
        "absolute_path_cells": absolute_path_cells,
    }


def match_notebook_subjects(
    notebooks: list[Path],
) -> dict[str, list[str]]:
    """Match notebook filenames to expected subjects."""

    matches: dict[str, list[str]] = {}

    for subject, keywords in (
        EXPECTED_NOTEBOOK_SUBJECTS.items()
    ):
        subject_matches = []

        for notebook_path in notebooks:
            filename = notebook_path.stem.lower()

            if all(
                keyword in filename
                for keyword in keywords
            ):
                subject_matches.append(
                    relative_path(notebook_path)
                )

        matches[subject] = subject_matches

    return matches


def check_artifact_groups(
    directory: Path,
    expected_groups: dict[str, list[str]],
) -> dict[str, dict[str, object]]:
    """Check whether expected artifacts exist."""

    results: dict[str, dict[str, object]] = {}

    for group_name, filenames in (
        expected_groups.items()
    ):
        existing = []
        missing = []

        for filename in filenames:
            path = directory / filename

            if (
                path.exists()
                and path.stat().st_size > 0
            ):
                existing.append(
                    relative_path(path)
                )
            else:
                missing.append(
                    relative_path(path)
                )

        results[group_name] = {
            "present": len(existing) > 0,
            "existing": existing,
            "missing": missing,
        }

    return results


def build_audit() -> dict[str, object]:
    """Build the complete project audit."""

    notebooks = collect_files(
        NOTEBOOK_DIR,
        ["*.ipynb"],
    )

    results = collect_files(
        RESULTS_DIR,
        [
            "*.csv",
            "*.stim",
            "*.dem",
            "*.html",
            "*.json",
            "*.txt",
        ],
    )

    figures = collect_files(
        FIGURE_DIR,
        [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.svg",
        ],
    )

    source_files = collect_files(
        SOURCE_DIR,
        ["*.py"],
    )

    test_files = collect_files(
        TEST_DIR,
        ["test_*.py"],
    )

    all_files = (
        notebooks
        + results
        + figures
        + source_files
        + test_files
    )

    empty_files = [
        relative_path(path)
        for path in all_files
        if path.stat().st_size == 0
    ]

    return {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "environment": {
            "python_version": (
                platform.python_version()
            ),
            "python_executable": sys.executable,
            "operating_system": (
                platform.platform()
            ),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "packages": package_versions(),
        },
        "tests": run_tests(),
        "required_files": {
            relative_path(path): (
                path.exists()
                and path.stat().st_size > 0
            )
            for path in REQUIRED_FILES
        },
        "inventory": {
            "notebook_count": len(notebooks),
            "result_count": len(results),
            "figure_count": len(figures),
            "source_count": len(source_files),
            "test_count": len(test_files),
            "notebooks": [
                file_information(path)
                for path in notebooks
            ],
            "results": [
                file_information(path)
                for path in results
            ],
            "figures": [
                file_information(path)
                for path in figures
            ],
            "source_files": [
                file_information(path)
                for path in source_files
            ],
            "test_files": [
                file_information(path)
                for path in test_files
            ],
            "empty_files": empty_files,
        },
        "notebooks": {
            "subject_matches": (
                match_notebook_subjects(
                    notebooks
                )
            ),
            "inspection": [
                inspect_notebook(path)
                for path in notebooks
            ],
        },
        "headline_results": check_artifact_groups(
            RESULTS_DIR,
            HEADLINE_RESULTS,
        ),
        "headline_figures": check_artifact_groups(
            FIGURE_DIR,
            HEADLINE_FIGURES,
        ),
    }

# Create a readable audit report
def build_text_report(
    audit: dict[str, object],
) -> str:


    lines: list[str] = []

    lines.extend(
        [
            "QUANTUM QEC PROJECT AUDIT",
            "=" * 72,
            f"Generated: {audit['generated_utc']}",
            f"Project: {audit['project_root']}",
            "",
            "ENVIRONMENT",
            "-" * 72,
        ]
    )

    environment = audit["environment"]

    lines.append(
        f"Python: {environment['python_version']}"
    )

    lines.append(
        f"Executable: "
        f"{environment['python_executable']}"
    )

    lines.append(
        f"Operating system: "
        f"{environment['operating_system']}"
    )

    lines.append(
        f"Machine: {environment['machine']}"
    )

    lines.append(
        f"Logical CPUs: "
        f"{environment['logical_cpu_count']}"
    )

    lines.extend(
        [
            "",
            "PACKAGE VERSIONS",
            "-" * 72,
        ]
    )

    for package_name, version in (
        environment["packages"].items()
    ):
        lines.append(
            f"{package_name}: {version}"
        )

    tests = audit["tests"]

    lines.extend(
        [
            "",
            "TEST SUITE",
            "-" * 72,
            f"Passed: {tests['passed']}",
            f"Return code: {tests['return_code']}",
            str(tests["stdout"]),
        ]
    )

    if tests["stderr"]:
        lines.append(str(tests["stderr"]))

    inventory = audit["inventory"]

    lines.extend(
        [
            "",
            "INVENTORY",
            "-" * 72,
            f"Notebooks: "
            f"{inventory['notebook_count']}",
            f"Results: "
            f"{inventory['result_count']}",
            f"Figures: "
            f"{inventory['figure_count']}",
            f"Source modules: "
            f"{inventory['source_count']}",
            f"Test files: "
            f"{inventory['test_count']}",
            f"Empty files: "
            f"{len(inventory['empty_files'])}",
            "",
            "REQUIRED FILES",
            "-" * 72,
        ]
    )

    for path, present in (
        audit["required_files"].items()
    ):
        status = "OK" if present else "MISSING"
        lines.append(
            f"[{status}] {path}"
        )

    lines.extend(
        [
            "",
            "NOTEBOOK SUBJECTS",
            "-" * 72,
        ]
    )

    for subject, matches in (
        audit["notebooks"][
            "subject_matches"
        ].items()
    ):
        if matches:
            lines.append(
                f"[OK] {subject}: "
                f"{', '.join(matches)}"
            )
        else:
            lines.append(
                f"[MISSING] {subject}"
            )

    lines.extend(
        [
            "",
            "NOTEBOOK EXECUTION",
            "-" * 72,
        ]
    )

    for notebook in (
        audit["notebooks"]["inspection"]
    ):
        if not notebook.get(
            "readable",
            False,
        ):
            lines.append(
                f"[UNREADABLE] "
                f"{notebook['path']}: "
                f"{notebook.get('error')}"
            )
            continue

        status = (
            "COMPLETE"
            if notebook[
                "completely_executed"
            ]
            else "INCOMPLETE"
        )

        lines.append(
            f"[{status}] "
            f"{notebook['path']} | "
            f"code cells "
            f"{notebook['executed_code_cells']}/"
            f"{notebook['code_cells']} | "
            f"saved errors "
            f"{len(notebook['saved_errors'])} | "
            f"absolute-path cells "
            f"{len(notebook['absolute_path_cells'])}"
        )

    lines.extend(
        [
            "",
            "HEADLINE RESULTS",
            "-" * 72,
        ]
    )

    for group_name, group in (
        audit["headline_results"].items()
    ):
        status = (
            "OK"
            if group["present"]
            else "MISSING"
        )

        lines.append(
            f"[{status}] {group_name}"
        )

        for path in group["existing"]:
            lines.append(f"    {path}")

    lines.extend(
        [
            "",
            "HEADLINE FIGURES",
            "-" * 72,
        ]
    )

    for group_name, group in (
        audit["headline_figures"].items()
    ):
        status = (
            "OK"
            if group["present"]
            else "MISSING"
        )

        lines.append(
            f"[{status}] {group_name}"
        )

        for path in group["existing"]:
            lines.append(f"    {path}")

    lines.extend(
        [
            "",
            "EMPTY FILES",
            "-" * 72,
        ]
    )

    if inventory["empty_files"]:
        lines.extend(
            inventory["empty_files"]
        )
    else:
        lines.append("None")

    return "\n".join(lines)


def main() -> None:
    """Run and save the audit."""

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Project root not found: "
            f"{PROJECT_ROOT}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit = build_audit()
    text_report = build_text_report(audit)

    AUDIT_JSON_PATH.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    AUDIT_TEXT_PATH.write_text(
        text_report,
        encoding="utf-8",
    )

    print(text_report)

    print()
    print(
        f"JSON audit saved to: "
        f"{AUDIT_JSON_PATH}"
    )

    print(
        f"Text audit saved to: "
        f"{AUDIT_TEXT_PATH}"
    )

    if not audit["tests"]["passed"]:
        raise SystemExit(
            "Audit completed, but tests failed."
        )


if __name__ == "__main__":
    main()