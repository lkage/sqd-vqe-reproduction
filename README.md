# sqd-vqe-reproduction

Python reproduction of the VQE algorithm portion of:

> Kim et al., "Qudit-based variational quantum eigensolver using photonic
> orbital angular momentum states", *Sci. Adv.* **10**, eado3472 (2024).

The paper implements VQE on a single photonic qudit encoded in orbital
angular momentum states. This repo reproduces the classical algorithm
side — Hamiltonian construction, the angle-parameterized ansatz, and
COBYLA optimization — in simulation.

Sister repo: [`qudit-simulator-cpp`](../qudit-simulator-cpp) — a
general-purpose C++ qudit simulator, cross-validated against this
implementation in week 7.

## Results

### H2 — 4-dimensional qudit (2 qubits), 6 angle parameters

![H2 convergence](results/h2_convergence.png)
![H2 PEC](results/h2_pec.png)

The Hamiltonian matches the paper's Table S1 to six decimal places at
every interatomic distance tested. VQE reaches chemical accuracy at 19
of 21 distances, with a mean error of 3.8e-4 Ha.

The two misses are at R <= 0.2 A. There the Hamiltonian coefficients are
an order of magnitude larger than at bonding length (Table S1: II = 4.76
at R = 0.1 versus -0.33 at R = 0.73). COBYLA's termination criterion
|a_{n+1} - a_n| < 0.01 — the paper's value — measures distance in
parameter space, so larger coefficients turn the same parameter error
into a larger energy error. Tightening the tolerance fixes this but
departs from the paper's setup, so it was left as is.

### LiH — 16-dimensional qudit (4 qubits), 30 angle parameters

![LiH convergence](results/lih_convergence.png)
![LiH PEC](results/lih_pec.png)

The Hamiltonian matches Table S2 (100 Pauli strings) to six decimal
places. VQE reaches chemical accuracy at all 14 distances, mean error
4.3e-4 Ha, using five random restarts per point.

A single run succeeds only about half the time in this 30-dimensional
non-convex landscape — failures cluster around -7.8632 Ha, a distinct
local minimum well above the true -7.8815 Ha. The paper likewise ran
each distance several times and reported the best result (Fig. 4B
caption).

For reference, the paper's LiH experiment had a mean error of 0.036 Ha
and did not reach chemical accuracy, limited by the purity and fidelity
of the OAM states (Figs. S2, S3). This repo simulates noiselessly, so
better accuracy is expected; what is being reproduced is the algorithm's
behavior, not the experimental error.

### What "exact" means here

The black curve is the lowest eigenvalue of the *same reduced
Hamiltonian the VQE optimizes* — not a full-CI result and not an
experimental reference. It measures how well COBYLA found the minimum in
the ansatz space, nothing more. The Hamiltonian's own accuracy is
verified separately against Tables S1 and S2.

## Method

| | H2 | LiH |
|---|---|---|
| Basis | STO-3G | STO-3G |
| Spin orbitals | 4 | 12 -> 6 (active space) |
| Active space | full | 2 electrons, 3 orbitals |
| Mapping | Parity + 2-qubit reduction | Parity + 2-qubit reduction |
| Qubits (dimension) | 2 (4D) | 4 (16D) |
| Pauli strings | 5 | 100 |
| Ansatz parameters | 6 | 30 |
| Bonding length | 0.73 A | 1.55 A |
| Restarts per point | 3 | 5 |

LiH's active space follows the standard reduction: freeze the Li 1s
core and drop the 2p_x and 2p_y orbitals, which do not participate in
bonding along z. One subtlety — with an active space the constant term
splits in two, nuclear repulsion (+1.02 Ha) and frozen-core energy
(-7.82 Ha), and both must be folded into the identity coefficient to
match Table S2.

Both ansatz forms — the paper's Eq. (5) for 4D and Eq. (9) for 16D —
are the same binary tree: each internal node splits amplitude by
cos/sin of a theta, and the node's omega enters the phase only along
the sin branch. One generalized function covers both, and the existing
H2 tests serve as its check.

## Setup

```bash
uv sync
uv pip install -e .
uv run pytest tests/ -v    # 92 tests
```

Requires Python 3.11 and `uv`. Developed on WSL2 Ubuntu 24.04.

## Reproducing the figures

```bash
uv run python examples/figures/h2_convergence_curve.py    # Fig. 3A
uv run python examples/figures/h2_potential_curve.py      # Fig. 3B
uv run python examples/figures/lih_convergence_curve.py   # Fig. 4A
uv run python examples/figures/lih_potential_curve.py     # Fig. 4B
```

The LiH sweep takes about five minutes; the others are faster.

## Project structure

```
sqd_vqe/
  hamiltonian.py   H2 and LiH Hamiltonians, plus a JSON cache
  ansatz.py        generalized d-dimensional ansatz, Eqs. (5) and (9)
  expectation.py   Pauli expectation values (reference implementation)
  vqe.py           COBYLA loop, single-run and multi-start
  sweep.py         interatomic distance sweeps
  data/            cached Hamiltonians (see below)
tests/             92 tests
examples/
  figures/         the four paper figures
  exploration/     week 3 learning scripts
  diagnostics/     tools written to track down specific problems
results/           generated figures and data
```

## A note on reproducibility

Qiskit's fermion-to-Pauli conversion sums coefficients for identical
Pauli strings in an order that varies between processes. Sixteen of
LiH's hundred coefficients come out differing by up to 152 ULP
(relative error 2e-14) from one run to the next — all of them terms
acting on only two qubits, which receive contributions from the most
fermionic terms and so accumulate the most reordering.

This is physically meaningless and never affects agreement with Table
S2. But the 30-dimensional COBYLA optimization amplifies it over
thousands of evaluations into convergence to entirely different local
minima. Hamiltonians are therefore generated once and cached as JSON in
`sqd_vqe/data/`, which is also what the C++ cross-validation will read.

The scripts in `examples/diagnostics/` document how this was tracked
down — PySCF turned out to be bit-for-bit deterministic, so the problem
had to lie downstream.
