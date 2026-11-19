#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv-mac"
BUILD_DIR="${PROJECT_ROOT}/build/mac_notebooks"
RESULTS_DIR="${PROJECT_ROOT}/results"
REPORTS_DIR="${PROJECT_ROOT}/reports"

PYTHON_COMMAND="${PYTHON_COMMAND:-}"
MODE="${1:-test}"

NON_GPU_NOTEBOOKS=(
    "quantum_noise_preparation.ipynb"
    "repetition_code.ipynb"
    "circuit_level_qec_stim.ipynb"
    "surface_code_memory.ipynb"
    "surface_code_threshold_analysis.ipynb"
    "decoder_latency_backlog.ipynb"
    "decoder_complexity_analysis.ipynb"
    "asymmetric_noise_sensitivity.ipynb"
)

print_heading() {
    printf "\n"
    printf '%s\n' "============================================================"
    printf '%s\n' "$1"
    printf '%s\n' "============================================================"
}

find_python() {
    if [[ -n "${PYTHON_COMMAND}" ]]; then
        if command -v "${PYTHON_COMMAND}" >/dev/null 2>&1; then
            printf '%s\n' "${PYTHON_COMMAND}"
            return 0
        fi

        printf 'Requested Python command not found: %s\n' \
            "${PYTHON_COMMAND}" >&2
        return 1
    fi

    local candidate

    for candidate in \
        python3.13 \
        python3.12 \
        python3 \
        python
    do
        if command -v "${candidate}" >/dev/null 2>&1; then
            if "${candidate}" -c \
                'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' \
                >/dev/null 2>&1
            then
                printf '%s\n' "${candidate}"
                return 0
            fi
        fi
    done

    printf '%s\n' \
        "No suitable Python installation was found." >&2

    printf '%s\n' \
        "Install Python 3.12 or 3.13, then rerun this script." >&2

    printf '%s\n' \
        "Homebrew example: brew install python@3.12" >&2

    return 1
}

create_environment() {
    local base_python="$1"

    print_heading "Creating the Mac virtual environment"

    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        "${base_python}" -m venv "${VENV_DIR}"
        printf 'Created: %s\n' "${VENV_DIR}"
    else
        printf 'Environment already exists: %s\n' "${VENV_DIR}"
    fi
}

install_project() {
    local python_path="${VENV_DIR}/bin/python"

    print_heading "Installing CPU project dependencies"

    "${python_path}" -m pip install \
        --upgrade \
        pip \
        setuptools \
        wheel

    # Install the package and development tools only.
    # The GPU optional dependency is deliberately omitted.
    "${python_path}" -m pip install \
        -e "${PROJECT_ROOT}[dev]"

    printf '%s\n' \
        "Installed the project without the GPU optional dependency."
}

check_cpu_environment() {
    local python_path="${VENV_DIR}/bin/python"

    print_heading "Checking the CPU environment"

    "${python_path}" - <<'PY'
from __future__ import annotations

import importlib.util
import platform
import sys

import matplotlib
import numpy
import pandas
import pymatching
import qiskit
import stim

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"Machine: {platform.machine()}")
print(f"NumPy: {numpy.__version__}")
print(f"pandas: {pandas.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Qiskit: {qiskit.__version__}")
print(f"Stim: {stim.__version__}")
print(
    "PyMatching:",
    getattr(pymatching, "__version__", "unknown"),
)

cupy_spec = importlib.util.find_spec("cupy")

if cupy_spec is None:
    print("CuPy: not installed, as expected on the Mac CPU environment")
else:
    print(
        "CuPy is installed, but it is not required for the non-GPU work."
    )

import quantum_qec

print(f"quantum_qec: {quantum_qec.__version__}")
PY
}

save_environment_report() {
    local python_path="${VENV_DIR}/bin/python"

    print_heading "Saving the Mac environment report"

    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${REPORTS_DIR}"

    "${python_path}" - <<PY > "${RESULTS_DIR}/mac_cpu_environment.txt"
from __future__ import annotations

import importlib.metadata
import platform
import sys

packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "qiskit",
    "qiskit-aer",
    "stim",
    "sinter",
    "pymatching",
    "pytest",
    "jupyterlab",
]

print(f"Python executable: {sys.executable}")
print(f"Python version: {platform.python_version()}")
print(f"Operating system: {platform.platform()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print()

print("Package versions")
print("----------------")

for package_name in packages:
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"

    print(f"{package_name}: {version}")
PY

    "${python_path}" -m pip freeze \
        > "${RESULTS_DIR}/requirements-mac-cpu.txt"

    printf 'Saved: %s\n' \
        "${RESULTS_DIR}/mac_cpu_environment.txt"

    printf 'Saved: %s\n' \
        "${RESULTS_DIR}/requirements-mac-cpu.txt"
}

run_tests() {
    local python_path="${VENV_DIR}/bin/python"

    print_heading "Running the complete test suite"

    export MPLBACKEND=Agg

    set +e

    "${python_path}" -m pytest \
        -v \
        "${PROJECT_ROOT}/tests" \
        2>&1 | tee "${RESULTS_DIR}/mac_cpu_pytest.txt"

    local pytest_status="${PIPESTATUS[0]}"

    set -e

    if [[ "${pytest_status}" -ne 0 ]]; then
        printf '%s\n' \
            "The Mac CPU test suite failed." >&2
        return "${pytest_status}"
    fi

    printf '%s\n' \
        "The Mac CPU test suite completed successfully."

    printf '%s\n' \
        "CUDA-dependent tests should appear as skipped."
}

create_portable_notebook_copies() {
    local python_path="${VENV_DIR}/bin/python"

    print_heading "Creating portable Mac notebook copies"

    mkdir -p "${BUILD_DIR}"

    PROJECT_ROOT_ENV="${PROJECT_ROOT}" \
    BUILD_DIR_ENV="${BUILD_DIR}" \
    NOTEBOOK_LIST_ENV="$(
        printf '%s\n' "${NON_GPU_NOTEBOOKS[@]}"
    )" \
    "${python_path}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

project_root = Path(
    os.environ["PROJECT_ROOT_ENV"]
).resolve()

build_dir = Path(
    os.environ["BUILD_DIR_ENV"]
).resolve()

notebook_names = [
    name.strip()
    for name in os.environ[
        "NOTEBOOK_LIST_ENV"
    ].splitlines()
    if name.strip()
]

source_directory = project_root / "notebooks"

windows_path_variants = [
    r"C:\Users\Karthi\Qauntum",
    "C:\\\\Users\\\\Karthi\\\\Qauntum",
    r"C:/Users/Karthi/Qauntum",
]

replacement_path = project_root.as_posix()

created = []
missing = []
replacement_counts = {}

for notebook_name in notebook_names:
    source_path = (
        source_directory
        / notebook_name
    )

    destination_path = (
        build_dir
        / notebook_name
    )

    if not source_path.exists():
        missing.append(notebook_name)
        continue

    notebook = json.loads(
        source_path.read_text(
            encoding="utf-8-sig"
        )
    )

    replacements = 0

    for cell in notebook.get(
        "cells",
        [],
    ):
        source = cell.get(
            "source",
            [],
        )

        if isinstance(source, str):
            source_lines = [
                source
            ]
        else:
            source_lines = list(
                source
            )

        updated_lines = []

        for line in source_lines:
            updated_line = line

            for old_path in windows_path_variants:
                count = updated_line.count(
                    old_path
                )

                if count:
                    updated_line = (
                        updated_line.replace(
                            old_path,
                            replacement_path,
                        )
                    )

                    replacements += count

            updated_lines.append(
                updated_line
            )

        cell["source"] = updated_lines

    metadata = notebook.setdefault(
        "metadata",
        {},
    )

    metadata[
        "mac_portable_copy"
    ] = {
        "source_notebook": (
            source_path.name
        ),
        "project_root": (
            replacement_path
        ),
        "path_replacements": (
            replacements
        ),
    }

    destination_path.write_text(
        json.dumps(
            notebook,
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    created.append(
        destination_path
    )

    replacement_counts[
        notebook_name
    ] = replacements

print(
    f"Created {len(created)} "
    "portable notebook copies."
)

for notebook_path in created:
    notebook_name = notebook_path.name

    print(
        f"  {notebook_name}: "
        f"{replacement_counts[notebook_name]} "
        "Windows-path replacements"
    )

if missing:
    print("Missing notebooks:")

    for notebook_name in missing:
        print(
            f"  {notebook_name}"
        )

    raise SystemExit(1)
PY
}

execute_notebooks() {
    local python_path="${VENV_DIR}/bin/python"

    print_heading "Executing the non-GPU notebooks"

    export MPLBACKEND=Agg
    export QUANTUM_QEC_PROJECT_ROOT="${PROJECT_ROOT}"

    local notebook_name
    local notebook_path

    for notebook_name in "${NON_GPU_NOTEBOOKS[@]}"
    do
        notebook_path="${BUILD_DIR}/${notebook_name}"

        printf '\nExecuting: %s\n' "${notebook_name}"

        "${python_path}" -m jupyter nbconvert \
            --to notebook \
            --execute \
            --inplace \
            --ExecutePreprocessor.timeout=7200 \
            --ExecutePreprocessor.kernel_name=python3 \
            "${notebook_path}"
    done

    printf '%s\n' \
        "All non-GPU notebook copies executed successfully."
}

validate_notebooks() {
    local python_path="${VENV_DIR}/bin/python"

    print_heading "Validating executed notebook copies"

    BUILD_DIR_ENV="${BUILD_DIR}" \
    NOTEBOOK_LIST_ENV="$(
        printf '%s\n' "${NON_GPU_NOTEBOOKS[@]}"
    )" \
    "${python_path}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

build_dir = Path(
    os.environ["BUILD_DIR_ENV"]
).resolve()

notebook_names = [
    name.strip()
    for name in os.environ[
        "NOTEBOOK_LIST_ENV"
    ].splitlines()
    if name.strip()
]

failures = []

for notebook_name in notebook_names:
    notebook_path = (
        build_dir
        / notebook_name
    )

    notebook = json.loads(
        notebook_path.read_text(
            encoding="utf-8-sig"
        )
    )

    unexecuted = []
    saved_errors = []

    for cell_index, cell in enumerate(
        notebook.get(
            "cells",
            [],
        )
    ):
        if (
            cell.get("cell_type")
            != "code"
        ):
            continue

        if (
            cell.get("execution_count")
            is None
        ):
            unexecuted.append(
                cell_index
            )

        for output in cell.get(
            "outputs",
            [],
        ):
            if (
                output.get("output_type")
                == "error"
            ):
                saved_errors.append(
                    {
                        "cell": cell_index,
                        "name": output.get(
                            "ename"
                        ),
                        "message": output.get(
                            "evalue"
                        ),
                    }
                )

    print(
        f"{notebook_name}: "
        f"unexecuted={len(unexecuted)}, "
        f"errors={len(saved_errors)}"
    )

    if unexecuted or saved_errors:
        failures.append(
            {
                "notebook": notebook_name,
                "unexecuted": unexecuted,
                "errors": saved_errors,
            }
        )

if failures:
    print()
    print(
        "Notebook validation failures:"
    )

    print(
        json.dumps(
            failures,
            indent=2,
        )
    )

    raise SystemExit(1)

print()
print(
    "All portable notebook copies are "
    "fully executed with no saved errors."
)
PY
}

export_notebooks_to_html() {
    local python_path="${VENV_DIR}/bin/python"

    print_heading "Exporting executed notebooks to HTML"

    local export_dir="${REPORTS_DIR}/mac_notebooks"

    mkdir -p "${export_dir}"

    local notebook_name

    for notebook_name in "${NON_GPU_NOTEBOOKS[@]}"
    do
        "${python_path}" -m jupyter nbconvert \
            --to html \
            --output-dir "${export_dir}" \
            "${BUILD_DIR}/${notebook_name}"
    done

    printf 'HTML reports saved under: %s\n' \
        "${export_dir}"
}

show_usage() {
    cat <<'USAGE'
Usage:

    ./run_non_gpu_mac.sh setup
    ./run_non_gpu_mac.sh test
    ./run_non_gpu_mac.sh notebooks
    ./run_non_gpu_mac.sh all

Modes:

    setup
        Create .venv-mac, install CPU dependencies,
        and save the Mac environment report.

    test
        Perform setup and run the full test suite.
        CUDA-dependent tests should be skipped.

    notebooks
        Perform setup, create portable copies of all
        non-GPU notebooks, execute them, validate them,
        and export them to HTML.

    all
        Run setup, tests, notebook execution, validation,
        and HTML export.

Optional environment variable:

    PYTHON_COMMAND=python3.12 ./run_non_gpu_mac.sh all

The original notebooks are not modified. Portable executed copies
are written under:

    build/mac_notebooks

HTML exports are written under:

    reports/mac_notebooks
USAGE
}

main() {
    case "${MODE}" in
        setup|test|notebooks|all)
            ;;
        -h|--help|help)
            show_usage
            exit 0
            ;;
        *)
            printf 'Unknown mode: %s\n\n' "${MODE}" >&2
            show_usage >&2
            exit 2
            ;;
    esac

    local base_python

    base_python="$(find_python)"

    printf 'Project root: %s\n' "${PROJECT_ROOT}"
    printf 'Base Python: %s\n' "${base_python}"
    printf 'Mode: %s\n' "${MODE}"

    create_environment "${base_python}"
    install_project
    check_cpu_environment
    save_environment_report

    case "${MODE}" in
        setup)
            ;;
        test)
            run_tests
            ;;
        notebooks)
            create_portable_notebook_copies
            execute_notebooks
            validate_notebooks
            export_notebooks_to_html
            ;;
        all)
            run_tests
            create_portable_notebook_copies
            execute_notebooks
            validate_notebooks
            export_notebooks_to_html
            ;;
    esac

    print_heading "Completed"

    printf '%s\n' \
        "Mac non-GPU workflow completed successfully."

    printf 'Environment: %s\n' "${VENV_DIR}"
}

main "$@"