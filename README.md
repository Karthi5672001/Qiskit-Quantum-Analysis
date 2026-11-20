# Qiskit-Quantum-Analysis
# Circuit-Level Quantum Error Correction and Heterogeneous Classical Processing

This project investigates quantum error correction as a quantum-classical systems problem. It combines circuit-level simulation of repetition and rotated surface codes with minimum-weight perfect-matching decoding, statistical error analysis, classical decoder benchmarking, queue-based latency modelling, and GPU-accelerated detector-event preprocessing.

## Software stack

- **Qiskit 2.6.2** and **Qiskit Aer 0.18.4** for introductory circuits, statevectors and noise modelling
- **Stim 1.16.0** for high-performance stabilizer-circuit simulation and detector sampling
- **Sinter 1.16.0** for parallel and adaptive Monte Carlo collection
- **PyMatching 2.4.0** for minimum-weight perfect-matching decoding
- **NumPy**, **pandas** and **Matplotlib** for numerical analysis and visualisation
- **pytest 9.2.0** for automated verification
- **Python 3.14.6** on Windows 11

## Repository structure
Below is how I structure my project on my device:
```text
Qauntum/
├── src/
│   └── quantum_qec/
│       ├── __init__.py
│       ├── benchmarking.py
│       ├── circuits.py
│       ├── decoding.py
│       ├── queueing.py
│       ├── simulation.py
│       └── statistics.py
├── tests/
│   ├── test_benchmarking.py
│   ├── test_circuits.py
│   ├── test_decoding.py
│   ├── test_queueing.py
│   ├── test_simulation.py
│   └── test_statistics.py
├── notebooks/
│   ├── quantum_noise_preparation.ipynb
│   ├── repetition_code.ipynb
│   ├── circuit_level_qec_stim.ipynb
│   ├── surface_code_memory.ipynb
│   ├── surface_code_threshold_analysis.ipynb
│   ├── decoder_latency_backlog.ipynb
│   ├── decoder_complexity_analysis.ipynb
│   └── asymmetric_noise_sensitivity.ipynb
├── scripts/
├── figures/
├── results/
├── reports/
├── pyproject.toml
├── requirements-lock.txt
└── README.md
```

## Installation

The verified environment used Python 3.14.6 on Windows 11.

Create and activate a virtual environment:

```powershell
python -m venv .qiskit
.qiskit\Scripts\Activate.ps1
```

Install the project and development dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Alternatively, recreate the recorded environment:

```powershell
python -m pip install -r requirements-lock.txt
```

## Running the tests

Run the complete test suite:

```powershell
python -m pytest -v
```

Current verified status:

```text
49 passed
```

## Limitations

- The threshold is a preliminary finite-distance crossing estimate rather than a full finite-size-scaling result.
- The noise models are stochastic Pauli models and do not include leakage, coherent calibration errors, crosstalk or temporally correlated faults.
- CPU timing results contain Python-call, operating-system and measurement overhead.
