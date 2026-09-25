# sqd-vqe-reproduction

Python reproduction of the VQE algorithm portion of:

> Kim et al., "Qudit-based variational quantum eigensolver using photonic
> orbital angular momentum states", *Sci. Adv.* **10**, eado3472 (2024).

Sister repo: [`qudit-simulator-cpp`](../qudit-simulator-cpp) — general-purpose
C++ qudit simulator used for cross-validation in week 7.

## Status

- **Week 3 (done)**: Python environment + H2 Hamiltonian matches paper Table S1
  to 6 decimal places across multiple interatomic distances.
- **Week 4 (next)**: H2 ansatz (Eq. 5), COBYLA optimization, Fig. 3 reproduction.

## Setup

```bash
uv sync
uv run pytest tests/ -v
```

Requires Python 3.11 and `uv`.

## Project structure

```
sqd_vqe/             # main package
  hamiltonian.py     # H2 Hamiltonian builder (Qiskit Nature + Parity mapper)
tests/               # pytest suite
examples/            # exploratory scripts kept as learning trail
```