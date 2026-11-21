# Qiskit-Quantum-Analysis
# Circuit-Level Quantum Error Correction and Heterogeneous Classical Processing

This project investigates quantum error correction as a quantum-classical systems problem. It progresses from introductory Qiskit studies of quantum states and noise to repetition-code validation, circuit-level rotated surface-code simulation with Stim, and minimum-weight perfect-matching decoding with PyMatching. The work includes a preliminary finite-distance threshold study, asymmetric-noise analysis, decoder latency and backlog modelling, and a reusable Python package supported by 63 automated tests.

The project also evaluates GPU-accelerated detector-event preprocessing using CuPy on an NVIDIA RTX 5090. Dense and bit-packed representations, CPU-GPU crossover behaviour, transfer overhead, and zero-event filtering before CPU decoding are compared. The results show that GPU kernels can provide substantial throughput for large detector matrices, but data movement and orchestration determine whether that acceleration benefits the complete pipeline. The broader conclusion is that practical fault-tolerant quantum computing requires the quantum code, decoder algorithm, communication path, and classical hardware architecture to be designed together.

## Software stack

- **Qiskit 2.6.2** and **Qiskit Aer 0.18.4** for introductory circuits, statevectors and noise modelling
- **Stim 1.16.0** for high-performance stabilizer-circuit simulation and detector sampling
- **Sinter 1.16.0** for parallel and adaptive Monte Carlo collection
- **PyMatching 2.4.0** for minimum-weight perfect-matching decoding
- **NumPy**, **pandas** and **Matplotlib** for numerical analysis and visualisation
- **pytest 9.2.0** for automated verification
- **Python 3.14.6** on Windows 11

# Project Overview

## Part I: Quantum error correction and CPU analysis

### Quantum circuit and noise preparation

[Quantum Noise Preparation](quantum_noise_preparation.ipynb)

The introductory work establishes the concepts required for later error-correction experiments:

- computational-basis states;
- superposition and phase;
- finite-shot measurement statistics;
- Bell-state preparation and correlation;
- readout errors;
- single- and two-qubit depolarising noise.

These experiments provide ideal baselines before redundancy and error correction are introduced.

### Three-qubit repetition code

[Three-qubit repetition code](repetition_code.ipynb)

A three-qubit bit-flip repetition code was implemented from first principles:

$$
|0_L\rangle = |000\rangle,
\qquad
|1_L\rangle = |111\rangle.
$$

The study includes:

- logical-state encoding;
- deterministic physical-error injection;
- ancilla-based parity-syndrome extraction;
- syndrome-controlled correction;
- majority-vote logical decoding;
- exhaustive zero-, one-, two-, and three-error testing;
- syndrome ambiguity;
- Monte Carlo validation;
- rare-event sampling limitations.

For independent physical bit-flip probability $p$, the theoretical logical failure probability is:

$$
P_L = 3p^2(1-p)+p^3
    = 3p^2-2p^3.
$$

The numerical simulations reproduce this analytical expression.

The experiment also demonstrates that a distance-three repetition code corrects one arbitrary bit flip, but cannot reliably correct arbitrary two-qubit bit flips

### Circuit-level quantum error correction

[Circuit-level quantum error correction](circuit_level_qec_stim_fixed.ipynb)

The project then moves from idealised data errors to circuit-level simulations using Stim.

The simulated fault mechanisms include:

- noisy Clifford operations;
- data errors between syndrome rounds;
- reset errors;
- measurement errors;
- repeated syndrome extraction;
- logical-observable changes.

Stim produces detector-event samples and logical-observable outcomes. PyMatching constructs a minimum-weight perfect-matching decoder from the corresponding detector error model.

A logical decoding failure is recorded when the decoder’s predicted logical-observable change differs from the value sampled by Stim.

The principal workflow is:

```text
Noisy Stim circuit
        ↓
Detector-event samples
        ↓
Detector error model
        ↓
PyMatching decoder
        ↓
Logical-failure estimate
```


### Decoder backlog modelling

[Decoder backlog modelling](decoder_latency_backlog.ipynb)

The classical decoder was modelled as a queue in which syndrome jobs arrive at a fixed interval and require variable service times.

For $c$ equivalent decoder workers, mean utilisation is:

$$
\rho = \frac{\mathbb{E}[S]}{c T_a},
$$

where:

- $\mathbb{E}[S]$ is the mean service time;
- $T_a$ is the syndrome arrival interval;
- $c$ is the number of parallel decoder workers.

A configuration with $\rho \geq 1$ is overloaded on average and develops a growing backlog. A configuration with $\rho < 1$ can still miss short deadlines because of service-time variability and tail-latency events.

The queue model evaluates:

- mean and percentile waiting times;
- mean and percentile response times;
- deadline-miss probability;
- maximum backlog;
- required worker count;
- stable and overloaded operating regions.

![Classical decoder backlog comparison](figures/decoder_backlog_comparison.png)

## Rotated surface-code memory experiments

[Rotated surface-code memory experiments](surface_code_memory.ipynb)

Logical $X$- and $Z$-memory circuits were studied across multiple code distances, syndrome-round counts, and circuit-level error probabilities.

The analysis records:

- physical-qubit count;
- detector and measurement count;
- matching-graph structure;
- logical failure probability;
- detector-event activity;
- decoder workload.

Increasing code distance raises the minimum weight of an undetectable logical operator, but also increases the number of qubits, checks, gates, and potential fault locations.

## Preliminary surface-code threshold study

[Preliminary surface-code threshold study](surface_code_threshold_analysis.ipynb)

An adaptive Sinter experiment was performed for rotated surface-code logical-memory circuits at:

$$
d \in \{3, 5, 7, 9\}.
$$

Circuit-level physical error probabilities between approximately $0.3\%$ and $1.5\%$ were sampled. Each configuration stopped after reaching either:

- 1,000 logical failures; or
- 1,000,000 shots.

The logical-error curves show a preliminary finite-distance crossing region of approximately:

$$
p_{\mathrm{cross}} \approx 0.65\% \text{ to } 0.70\%.
$$

At a physical error probability of $0.3\%$, the measured logical failure rates were:

| Code distance | Logical failure rate |
|---:|---:|
| 3 | $6.913 \times 10^{-3}$ |
| 5 | $3.742 \times 10^{-3}$ |
| 7 | $1.733 \times 10^{-3}$ |
| 9 | $7.30 \times 10^{-4}$ |

Increasing the distance from 3 to 9 reduced the measured logical failure rate by approximately $9.5\times$ for this simulated configuration.

This is a **preliminary finite-distance crossing estimate**, not a precision asymptotic threshold. A stronger estimate would require:

- additional code distances;
- denser sampling around the crossing;
- uncertainty-aware finite-size scaling;
- comparison of multiple circuit and noise models.

## CPU decoder performance

[CPU decoder performance part 1](decoder_latency_backlog.ipynb)
[CPU decoder performance part 2](decoder_complexity_analysis.ipynb)

PyMatching was benchmarked in both batch and individual-call modes.

One representative distance-three benchmark produced approximately:

| Measurement | Observed value |
|---|---:|
| Batched effective cost | $0.0583     \mu\mathrm{s}$ per shot |
| Median individual-call latency | $2.80     \mu\mathrm{s}$ |
| 99th-percentile individual-call latency | $5.00     \mu\mathrm{s}$ |

The individual-call median was approximately $48\times$ the effective batch cost per shot.

This does not imply that the matching algorithm becomes $48\times$ more complex. The difference includes:

- Python-call overhead;
- batch amortisation;
- timing resolution;
- memory behaviour;
- operating-system scheduling.

The results distinguish between:

- **batch throughput**, relevant to offline simulation;
- **individual latency**, relevant to prompt responses;
- **tail latency**, relevant to deadlines and temporary backlog.

## Queue and backlog modelling

[Queue and backlog modelling](decoder_latency_backlog.ipynb)

The decoder was modelled as a multi-worker queue receiving syndrome jobs at a fixed interval.

For $c$ equivalent decoder workers, mean utilisation is:

$$
\rho = \frac{\mathbb{E}[S]}{c T_a},
$$

where:

- $\mathbb{E}[S]$ is mean service time;
- $T_a$ is the syndrome arrival interval;
- $c$ is the number of workers.

A configuration with:

$$
\rho \geq 1
$$

is overloaded on average and develops a growing backlog.

A configuration with:

$$
\rho < 1
$$

can still miss short deadlines because of service-time variability and latency-tail events.

The model evaluates:

- waiting-time distributions;
- response-time distributions;
- maximum and final backlog;
- deadline-miss probability;
- stable and overloaded operating regions;
- minimum worker count.

## Decoder-complexity analysis

[Decoder-complexity analysis](decoder_complexity_analysis.ipynb)

Decoder cost was analysed against:

- surface-code distance;
- detector count;
- matching-graph edge count;
- physical error probability;
- detector-event count;
- detector-event density;
- decoder batch size.

Detector-event count alone did not fully determine latency. Other influences include:

- event geometry;
- matching-graph structure;
- Python and library-call overhead;
- memory behaviour;
- operating-system variability.

## Asymmetric-noise sensitivity

[Asymmetric-noise sensitivity](asymmetric_noise_sensitivity.ipynb)

The circuit-level noise model was separated into:

- Clifford-gate depolarisation;
- data-qubit depolarisation;
- measurement-result flips;
- reset errors.

Each component was varied independently while the remaining probabilities were held at baseline values.

The analysis compares:

- logical failure probability;
- detector-event activity;
- proportional degradation from baseline;
- local sensitivity;
- distance scaling.

This demonstrates why one aggregate physical error probability may conceal which physical operations dominate logical risk or classical detector traffic.

This demonstrates that an aggregate physical error rate can conceal which operations dominate detector traffic and logical risk.

![Asymmetric-noise sensitivity ranking](figures/asymmetric_noise_sensitivity_ranking.png)

# Part II: GPU and heterogeneous processing

## GPU-Accelerated Detector-Event Preprocessing
 
The complete analysis, executable code, saved benchmark outputs, and interpretation are available in:
 
[GPU and heterogeneous processing](gpu_detector_preprocessing.ipynb)

## Motivation

The GPU study investigates whether detector-event preprocessing can benefit from an NVIDIA accelerator after accounting for data movement and pipeline overhead.

The tested environment used:

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| GPU memory | Approximately 32 GB |
| Compute capability | 12.0 |
| CuPy | 14.2.0 |
| CuPy-linked CUDA runtime | 13.2 |
| NVIDIA driver CUDA capability | 13.5 |
| Local CUDA Toolkit | 13.3 |
| Python | 3.14.6 |

This is a preprocessing and systems study. PyMatching remains a CPU decoder, so the work is **not** a GPU implementation of surface-code minimum-weight perfect matching.

## GPU preprocessing module

The reusable GPU module provides:

- CUDA availability checks;
- GPU device information;
- CPU and GPU event counting;
- event-density calculation;
- zero-event masking;
- dense and bit-packed detector processing;
- synchronised CUDA timing;
- host-to-device timing;
- resident-kernel timing;
- device-to-host timing;
- end-to-end timing;
- conservative GPU batch-size estimation.

## Correctness verification

CPU and GPU implementations were verified to agree for:

- detector-event counts;
- event density;
- zero-event masks;
- bit-packed detector representations.

CuPy 14.2.0 does not support `packbits(..., axis=1)`. The project therefore implements explicit row-wise GPU bit packing using:

- zero padding to a whole number of bytes;
- groups of eight detector values;
- little-endian bit weights;
- GPU multiplication and reduction.

The output was verified against NumPy’s row-wise `packbits` operation.

## Synthetic preprocessing baseline

A synthetic dense Boolean detector matrix containing:

$100{,}000 \times 1{,}000$ entries, approximately $95.37\ \mathrm{MiB}$, produced the following representative result:

| Component | Median time |
|---|---:|
| NumPy CPU event count | $39.40\ \mathrm{ms}$ |
| Host-to-device transfer | $4.71\ \mathrm{ms}$ |
| Resident GPU kernel | $0.084\ \mathrm{ms}$ |
| Device-to-host transfer | $0.065\ \mathrm{ms}$ |
| Complete GPU pipeline | $4.49\ \mathrm{ms}$ |
| Resident-data speedup | $467.7\times$ |
| End-to-end speedup | $8.77\times$ |

The resident kernel was extremely fast, but host-to-device transfer dominated the complete GPU path.

This means that further kernel optimisation alone would have limited value unless data transfer is also reduced, overlapped, or avoided.

## Real Stim detector workloads

Real detector matrices were generated for rotated surface-code circuits at:

$$
d \in \{3, 5, 7, 9, 11\},
$$

with batch sizes:

$$
N \in \{1{,}000,\ 10{,}000,\ 100{,}000,\ 500{,}000\}.
$$

The 20-configuration sweep measured:

- detector width;
- matrix size;
- mean event count;
- zero-event fraction;
- CPU event-counting time;
- host-to-device transfer;
- resident GPU kernel time;
- device-to-host transfer;
- complete GPU time;
- resident speedup;
- transfer-inclusive speedup;
- CPU–GPU crossover by distance and batch size.

The resident measurements describe a system in which detector data is already on the accelerator. The end-to-end measurements describe the current Stim workflow, where detector samples are created in CPU memory.

## CPU–GPU Crossover Analysis

For each surface-code distance, the first tested batch size satisfying:

$$
S_{\mathrm{end\text{-}to\text{-}end}} > 1
$$

was recorded as the measured CPU–GPU crossover.

This is the first crossover observed within the tested batch-size grid, rather than an exact analytical crossover point. More closely spaced batch sizes would be needed to locate the transition precisely.

The crossover depends on the total detector-matrix size:

$$
B_{\mathrm{dense}} = N_{\mathrm{shots}} N_{\mathrm{detectors}} \ \mathrm{bytes},
$$

because increasing either batch size or code distance exposes more parallel work to the GPU while also increasing the required data transfer.

The following outputs record the dense-processing results:

- `results/gpu_detector_preprocessing.csv`
- `results/gpu_preprocessing_crossover.csv`
- `figures/gpu_event_count_crossover.png`
- `figures/gpu_speedup_by_distance.png`
- `figures/gpu_transfer_breakdown.png`

The crossover and timing-breakdown figures distinguish between:

- CPU processing;
- resident GPU computation;
- host-to-device transfer;
- device-to-host result transfer;
- complete transfer-inclusive GPU processing.

## Bit-Packed Detector Processing

Dense NumPy Boolean arrays use approximately one byte for each detector value, although each detector contains only one bit of information.

For $N_{\mathrm{shots}}$ samples and $N_{\mathrm{detectors}}$ detectors, the packed representation requires:

$$B_{\mathrm{packed}} = N_{\mathrm{shots}} \left\lceil \frac{N_{\mathrm{detectors}}}{8} \right\rceil \ \mathrm{bytes}.$$

The maximum compression ratio approaches

$$\frac{B_{\mathrm{dense}}}{B_{\mathrm{packed}}} \approx 8,$$

although row padding can reduce the achieved ratio when the detector count is not divisible by eight.

The packed pipeline performs:

1. CPU-side row-wise bit packing;
2. transfer of packed bytes to GPU memory;
3. GPU population counting;
4. transfer of the per-sample event counts back to the CPU.

The GPU counts active detector bits without reconstructing the original dense matrix. A 256-entry lookup table stores the number of active bits in every possible byte value ($0 \leq b \leq 255$). Packed bytes are used as lookup-table indices, and the resulting byte-level counts are reduced across each detector row.

The complete packed-processing time includes:

$$T_{\mathrm{packed,total}} = T_{\mathrm{CPU,pack}} + T_{\mathrm{H2D,packed}} + T_{\mathrm{GPU,popcount}} + T_{\mathrm{D2H}}.$$

The packed-versus-dense GPU speedup is:

$$S_{\mathrm{packed/dense}} = \frac{T_{\mathrm{dense,GPU,total}}}{T_{\mathrm{packed,GPU,total}}}.$$

A reduction in transfer size does not automatically produce an equivalent reduction in total runtime. CPU packing and GPU population counting must also be included.

The packed representation is more attractive in an architecture where an FPGA, digitiser, or readout processor produces packed detector records directly. In that case, the synchronous CPU packing stage would be removed.

The packed benchmarks use the same 20 real Stim configurations as the dense sweep.

The corresponding outputs are:

- `results/gpu_packed_smoke_benchmark.csv`
- `results/gpu_packed_detector_preprocessing.csv`
- `results/gpu_packed_preprocessing_summary.csv`
- `figures/gpu_packed_transfer_comparison.png`

## GPU Zero-Event Filtering Before CPU Decoding

At low physical error probabilities, many detector samples contain no detection events. These samples can be identified before CPU decoding.

The tested heterogeneous pipeline is:

```text
Stim detector matrix in CPU memory
        ↓
GPU identifies nonzero-event rows
        ↓
Selection mask returned to CPU
        ↓
CPU selects retained rows
        ↓
PyMatching decodes retained rows
        ↓
Predictions reconstructed in original order
```

Before applying this optimisation, the notebook verifies that an all-zero detector row produces an all-zero logical prediction for the tested PyMatching decoder.

The filtered pipeline is accepted only when:

```python
np.array_equal(
    full_predictions,
    filtered_predictions,
)
```

is true.

For a distance-three pilot with physical error probability:

$$
p = 0.001
$$

and $10{,}000$ samples:

- $83.68\%$ of detector rows contained no events;
- $1{,}632$ of $10{,}000$ rows were passed to PyMatching;
- reconstructed predictions exactly matched full-batch PyMatching predictions.

A zero-event row does not prove that the underlying physical process contained no logical fault. It means that the decoder receives an empty detector record and returns the corresponding prediction under the tested matching graph.

Hybrid speedup is defined as:

$$S_{\mathrm{hybrid}} = \frac{T_{\mathrm{full\,CPU\,decode}}}{T_{\mathrm{hybrid,total}}},$$

where:

$$T_{\mathrm{hybrid,total}} = T_{\mathrm{GPU,filter}} + T_{\mathrm{CPU,selection}} + T_{\mathrm{filtered\,decode}} + T_{\mathrm{reconstruction}}.$$

The complete hybrid measurement includes:

- transfer of the detector matrix to the GPU;
- GPU identification of nonzero-event rows;
- transfer of the selection mask to the CPU;
- NumPy Boolean row selection;
- PyMatching decoding of retained rows;
- reconstruction of predictions in their original order.

A high zero-event fraction represents an opportunity to reduce decoder work, but it does not guarantee an end-to-end speedup. Transfer, filtering, indexing, reconstruction, and smaller decoder batches can outweigh the saved PyMatching work.

A focused hybrid sweep was performed over:

$$p \in \{0.001, 0.003\}, \qquad d \in \{3, 7\}, \qquad N \in \{10{,}000, 100{,}000\}.$$

This produced eight configurations. Every configuration was checked for exact prediction agreement before its timing result was accepted.

The corresponding outputs are:

- `results/gpu_hybrid_filtering_pilot.csv`
- `results/gpu_hybrid_filtering.csv`
- `results/gpu_hybrid_filtering_summary.csv`
- `figures/gpu_hybrid_filtering_pilot.png`
- `figures/gpu_hybrid_filtering_opportunity.png`

## Per-Patch Latency and System-Scale Throughput

For an individual small surface-code patch, each syndrome-extraction round produces a relatively small detector record. Invoking a discrete GPU independently for every patch and every correction cycle is unlikely to be efficient because transfer, launch, synchronisation, and scheduling overhead can dominate the limited computational payload.

The aggregate classical-processing challenge arises from continuous repetition and system-wide scale. Stabilizer measurements are generated every correction cycle across active data, ancilla, routing, and magic-state patches.

The approximate raw syndrome-data rate is:

$$R_{\mathrm{syndrome}} \approx N_{\mathrm{patches}} \, N_{\mathrm{checks/patch}} \, f_{\mathrm{cycle}} \, b_{\mathrm{check}},$$

where:

- $N_{\mathrm{patches}}$ is the number of active code patches;
- $N_{\mathrm{checks/patch}}$ is the number of measured checks per patch;
- $f_{\mathrm{cycle}}$ is the syndrome-extraction frequency;
- $b_{\mathrm{check}}$ is the number of transported bits associated with each check.

The actual transported bandwidth also depends on:

- measurement multiplexing;
- detector extraction close to the readout hardware;
- dense, sparse, or bit-packed representation;
- timestamps and framing;
- compression;
- buffering;
- communication-protocol overhead.

A real system must satisfy both throughput and latency requirements. Average decoder capacity must keep pace with the syndrome stream, while results required for feed-forward must become available before the corresponding logical operation consumes them.

## Interpreting the Large-Batch Results

The large-batch GPU benchmarks are most representative of:

- offline Monte Carlo simulation;
- simultaneous preprocessing across many independent code patches;
- aggregation across several syndrome windows;
- calibration and detector-workload analysis;
- workloads already resident on an accelerator;
- systems capable of streaming packed detector records efficiently.

They should not be interpreted as evidence that a discrete GPU is optimal for the lowest-latency, single-patch real-time decoding path.

Temporal batching can increase throughput but introduces waiting latency. It may therefore be unsuitable before:

- adaptive logical measurements;
- lattice-surgery feed-forward;
- non-Clifford corrections;
- operations requiring an immediate Pauli-frame decision.

The implemented surface-code path remains:

```text
Stim sampling on the CPU
        ↓
CuPy preprocessing on the GPU
        ↓
PyMatching decoding on the CPU
```

The notebook demonstrates GPU preprocessing and workload reduction, not GPU minimum-weight perfect matching.

## Heterogeneous Control Architecture

A realistic fault-tolerant quantum-classical system may distribute work across several specialised tiers:

- **Readout and local FPGA/ASIC layer:** Performs readout classification, detection-event extraction, bit packing, framing, buffering, and potentially low-latency decoding close to the quantum readout and control electronics.
- **CPU and control-runtime layer:** Coordinates experiments, manages decoder configuration and Pauli frames, schedules logical operations, handles exceptional cases, and integrates the decoder with the wider control stack.
- **GPU or accelerator layer:** Processes aggregated detector workloads, parallel multi-patch preprocessing, large simulation batches, calibration telemetry, and decoder-parameter optimisation.
- **Storage and analysis layer:** Retains selected detector traces, logical-failure records, calibration data, and benchmark outputs for retrospective analysis.

The appropriate division depends on:

- physical-qubit technology;
- error-correcting code;
- decoder algorithm;
- feed-forward deadline;
- number of active code patches;
- interconnect bandwidth and latency;
- power and cooling constraints;
- physical placement of the classical hardware.

Conventional discrete-GPU systems require data to cross a host–device interconnect. Coherent CPU–GPU systems, unified-memory processors, direct peer-to-peer DMA, and shared-memory FPGA–CPU systems can reduce explicit software-managed copying.

These alternatives do not eliminate the physical cost of data movement. Memory location, bandwidth, latency, cache coherence, synchronisation, and contention remain important. They must therefore be evaluated using complete-pipeline measurements rather than kernel execution time alone.

## Main GPU Findings

The GPU study produced the following principal conclusions:

1. **Resident GPU reductions are extremely fast.**
   Once detector data is present in GPU memory, event-count reductions can substantially outperform the equivalent NumPy operation.

2. **Data movement is the dominant bottleneck.**
   For the tested RTX 5090 path, host-to-device transfer was much more expensive than the resident counting kernel.

3. **The useful crossover depends on total matrix size.**
   Batch size and detector count jointly determine whether sufficient parallel work exists to compensate for fixed GPU overheads.

4. **Bit packing reduces transfer volume but adds processing cost.**
   Its complete benefit depends on whether packed data must be created synchronously on the CPU or is produced directly by upstream hardware.

5. **Zero-event filtering can greatly reduce decoder input.**
   In the distance-three pilot, $83.68\%$ of rows bypassed PyMatching while preserving its predictions exactly.

6. **Reduced decoder input does not guarantee reduced total runtime.**
   Transfer, mask generation, row selection, smaller decoder batches, and reconstruction must all be included.

7. **The most plausible architecture is heterogeneous.**
   FPGA or ASIC logic is better suited to the strict low-latency path, while GPUs are more appropriate for aggregated and throughput-oriented workloads.

## Limitations of the GPU Study

- The benchmark results are specific to the tested RTX 5090, CuPy, CUDA, Python, NumPy, Stim, PyMatching, and Windows environment.
- Stim generates detector matrices in CPU memory, so the measured end-to-end path includes explicit host-to-device transfer.
- The synthetic Bernoulli workload does not reproduce circuit-level detector correlations or geometry.
- The GPU performs preprocessing rather than minimum-weight perfect-matching decoding.
- PyMatching remains CPU-based.
- Large-batch throughput does not directly represent single-cycle real-time latency.
- The experiment does not directly benchmark FPGA-to-GPU DMA, coherent CPU–GPU systems, CXL-based accelerators, or FPGA-resident decoding.
- Timing measurements include operating-system, runtime, allocation, synchronisation, and memory-pool effects.
- The empty-row filtering result depends on physical error probability, code distance, batch size, and detector activity.
- No claim is made that the tested discrete-GPU architecture is optimal for a fault-tolerant quantum computer.

---

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
│   ├── asymmetric_noise_sensitivity.ipynb
│   └── gpu_detector_preprocessing.ipynb
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
