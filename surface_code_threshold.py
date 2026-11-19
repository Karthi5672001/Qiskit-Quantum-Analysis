# Monte Carlo threshold simulation for a rotated surface code
# Uses:
# Stim - circuit simulation
# Sinter - parallel sampling
# PyMatching - minimum-weight perfect matching decoding
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sinter
import stim

# Handling files and directories for data read/write
#==================================================================
PROJECT_DIR = Path(r"/")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV_PATH = (
    RESULTS_DIR / "surface_code_threshold_sinter.csv"
)

SUMMARY_CSV_PATH = (
    RESULTS_DIR / "surface_code_threshold_summary.csv"
)

FIGURE_PATH = (
    FIGURE_DIR / "surface_code_threshold_sinter.png"
)
# Handling files and directories for data read/write
#==================================================================

# Parameter Grid
DISTANCES = [3, 5, 7, 9]
# 10 physical error states for 40 total configurations
PHYSICAL_ERROR_PROBABILITIES = [
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.010,
    0.012,
    0.015,
]
# Stopping criteria
# A shot is one execution and measurement of a circuit
MAX_SHOTS = 1_000_000
MAX_ERRORS = 1_000

# Leave one or two logical CPU cores free for Windows and other applications
# So I can refer to book/Qiskit documentation or google
# Keep this in or you may crash/freeze
NUM_WORKERS = max(
    1,
    (os.cpu_count() or 2) - 2,
)

# Generate a rotated surface-code logical X-memory circuit with rounds equal to code distance
# applies a uniform circuit-level noise model (p) to Clifford gates, idle data qubits, measurements, and resets
def create_surface_code_circuit(
    distance: int,
    physical_error_probability: float,
) -> stim.Circuit:

    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=(
            physical_error_probability
        ),
        before_round_data_depolarization=(
            physical_error_probability
        ),
        before_measure_flip_probability=(
            physical_error_probability
        ),
        after_reset_flip_probability=(
            physical_error_probability
        ),
    )

# Yield one Sinter task for every distance and physical-error-probability combination
# attaches json_metadata so results can be grouped and traced after parallel execution
def generate_tasks():

    for distance in DISTANCES:
        for physical_error_probability in (
            PHYSICAL_ERROR_PROBABILITIES
        ):
            yield sinter.Task(
                circuit=create_surface_code_circuit(
                    distance=distance,
                    physical_error_probability=(
                        physical_error_probability
                    ),
                ),
                json_metadata={
                    "code": (
                        "surface_code:"
                        "rotated_memory_x"
                    ),
                    "distance": distance,
                    "rounds": distance,
                    "physical_error_probability": (
                        physical_error_probability
                    ),
                },
            )

# Convert Sinter TaskStats values into a clean pandas DataFrame.
def stats_to_dataframe(
    collected_stats,
) -> pd.DataFrame:

    rows = []

    for stat in collected_stats:
        metadata = stat.json_metadata

        shots = int(stat.shots)
        errors = int(stat.errors)
        discards = int(stat.discards)
        seconds = float(stat.seconds)

        retained_shots = shots - discards

        # errors > 0: Computes standard error: SE = sqrt{P_L(1 - P_L)/N}
        # and a 95% normal confidence interval (P_L +/- 1.96 * SE).
        if retained_shots <= 0:
            logical_error_rate = np.nan
        else:
            # Logical Error Rate (P_L): Computed as errors / retained_shots
            logical_error_rate = (
                errors / retained_shots
            )

        if (
            retained_shots > 0
            and errors > 0
        ):
            standard_error = np.sqrt(
                logical_error_rate
                * (1.0 - logical_error_rate)
                / retained_shots
            )
        else:
            standard_error = np.nan
        # When errors = 0: Uses the Rule of Three (3 / N)
        # to estimate a 95% confidence upper bound on the failure rate
        if errors == 0 and retained_shots > 0:
            upper_bound_95 = (
                3.0 / retained_shots
            )
        else:
            upper_bound_95 = np.nan

        rows.append(
            {
                "code": metadata["code"],
                "distance": int(
                    metadata["distance"]
                ),
                "rounds": int(
                    metadata["rounds"]
                ),
                "physical_error_probability": (
                    float(
                        metadata[
                            "physical_error_probability"
                        ]
                    )
                ),
                "decoder": stat.decoder,
                "shots": shots,
                "discards": discards,
                "retained_shots": retained_shots,
                "logical_errors": errors,
                "logical_error_rate": (
                    logical_error_rate
                ),
                "standard_error": (
                    standard_error
                ),
                "confidence_lower_95": (
                    max(
                        0.0,
                        logical_error_rate
                        - 1.96 * standard_error,
                    )
                    if errors > 0
                    else 0.0
                ),
                "confidence_upper_95": (
                    min(
                        1.0,
                        logical_error_rate
                        + 1.96 * standard_error,
                    )
                    if errors > 0
                    else upper_bound_95
                ),
                "zero_error_upper_bound_95": (
                    upper_bound_95
                ),
                "cpu_seconds": seconds,
                "shots_per_cpu_second": (
                    shots / seconds
                    if seconds > 0.0
                    else np.nan
                ),
                "strong_id": stat.strong_id,
                "metadata_json": json.dumps(
                    metadata,
                    sort_keys=True,
                ),
            }
        )

    dataframe = pd.DataFrame(rows)

    dataframe = dataframe.sort_values(
        [
            "distance",
            "physical_error_probability",
        ]
    ).reset_index(drop=True)

    return dataframe

# Plot logical error rate against physical error probability for each code distance
def plot_threshold(
    dataframe: pd.DataFrame,
) -> None:

    dataframe = dataframe.copy()

    dataframe[
        "logical_error_rate_for_plot"
    ] = np.where(
        dataframe["logical_errors"] > 0,
        dataframe["logical_error_rate"],
        dataframe[
            "zero_error_upper_bound_95"
        ],
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    colour_map = plt.get_cmap("viridis")

    for distance_index, distance in enumerate(
        DISTANCES
    ):
        subset = dataframe[
            dataframe["distance"] == distance
        ].sort_values(
            "physical_error_probability"
        )

        colour = colour_map(
            distance_index
            / max(1, len(DISTANCES) - 1)
        )

        measured_subset = subset[
            subset["logical_errors"] > 0
        ]

        upper_bound_subset = subset[
            subset["logical_errors"] == 0
        ]

        axis.plot(
            subset[
                "physical_error_probability"
            ],
            subset[
                "logical_error_rate_for_plot"
            ],
            linewidth=2,
            color=colour,
            label=f"Distance {distance}",
        )

        axis.scatter(
            measured_subset[
                "physical_error_probability"
            ],
            measured_subset[
                "logical_error_rate"
            ],
            marker="o",
            s=45,
            color=colour,
        )

        if not upper_bound_subset.empty:
            axis.scatter(
                upper_bound_subset[
                    "physical_error_probability"
                ],
                upper_bound_subset[
                    "logical_error_rate_for_plot"
                ],
                marker="v",
                s=65,
                facecolors="none",
                edgecolors=colour,
            )

    axis.set_xscale("log")
    axis.set_yscale("log")

    axis.set_xlabel(
        "Circuit-level physical error probability"
    )

    axis.set_ylabel(
        "Logical decoding-failure probability"
    )

    axis.set_title(
        "Rotated surface-code threshold study"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

# Print the main results in a readable format
def print_summary(
    dataframe: pd.DataFrame,
) -> None:

    print()
    print("Threshold collection complete")
    print("=" * 72)

    display_columns = [
        "distance",
        "physical_error_probability",
        "shots",
        "logical_errors",
        "logical_error_rate",
        "cpu_seconds",
    ]

    print(
        dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Raw Sinter results: {RESULTS_CSV_PATH}"
    )

    print(
        f"Summary CSV: {SUMMARY_CSV_PATH}"
    )

    print(
        f"Threshold plot: {FIGURE_PATH}"
    )


def main() -> None:
    print(
        f"Using {NUM_WORKERS} worker processes."
    )

    print(
        f"Distances: {DISTANCES}"
    )

    print(
        "Physical error probabilities: "
        f"{PHYSICAL_ERROR_PROBABILITIES}"
    )

    print(
        f"Maximum shots per configuration: "
        f"{MAX_SHOTS:,}"
    )

    print(
        f"Maximum logical errors per "
        f"configuration: {MAX_ERRORS:,}"
    )
    # Runs parallel decoding with pymatching
    # Passing save_resume_filepath allows the script to checkpoint progress to CSV and resume safely if interrupted
    collected_stats = sinter.collect(
        num_workers=NUM_WORKERS,
        tasks=generate_tasks(),
        decoders=["pymatching"],
        max_shots=MAX_SHOTS,
        max_errors=MAX_ERRORS,
        save_resume_filepath=(
            RESULTS_CSV_PATH
        ),
        print_progress=True,
    )

    dataframe = stats_to_dataframe(
        collected_stats
    )

    dataframe.to_csv(
        SUMMARY_CSV_PATH,
        index=False,
    )

    plot_threshold(
        dataframe
    )

    print_summary(
        dataframe
    )

# Guard: Without it, each worker can recursively start more workers.
if __name__ == "__main__":
    main()