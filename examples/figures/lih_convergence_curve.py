"""R=1.55 Å에서 LiH VQE의 수렴 궤적 (논문 Fig. 4A 재현).

논문 Fig. 4A와 같은 형태: 하나의 axes에 에너지(왼쪽 y축)와
30개 각도 파라미터(오른쪽 y축)를 겹쳐 표시.

"Exact diag."는 VQE가 최적화하는 것과 동일한 4-qubit(16x16) 축소
Hamiltonian의 최소 고윳값이다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sqd_vqe.ansatz import LIH_NUM_PARAMS, lih_ansatz_state
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import get_lih_hamiltonian
from sqd_vqe.vqe import run_vqe_multistart


N_RESTARTS = 5


def main():
    R = 1.55
    print(f"Running LiH VQE at R = {R} Å (multi-start {N_RESTARTS}회)...")

    H = get_lih_hamiltonian(distance=R)
    exact = float(np.linalg.eigvalsh(hamiltonian_matrix(H))[0])
    result = run_vqe_multistart(
        H, ansatz=lih_ansatz_state, n_restarts=N_RESTARTS,
        seed=42, n_params=LIH_NUM_PARAMS, max_iter=3000,
    )

    print(f"Best restart seed: {result.best_restart_seed}")
    print(f"Converged in {result.n_iterations} function evaluations")
    print(f"Final energy: {result.energy:.6f} Hartree")
    print(f"Exact energy: {exact:.6f} Hartree")
    print(f"Error: {abs(result.energy - exact):.2e} Hartree")

    evaluations = np.arange(1, result.n_iterations + 1)
    params_arr = np.array(result.param_history)

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(evaluations, result.energy_history,
             "-", color="black", linewidth=1.8,
             label=r"$\langle H \rangle$", zorder=3)
    ax1.axhline(exact, color="black", linestyle=":",
                linewidth=1.2, alpha=0.6,
                label=f"Exact diag. ({exact:.4f} Ha)")
    ax1.set_xlabel("Function evaluation")
    ax1.set_ylabel(r"$\langle H \rangle$ (Hartree)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    colors = plt.cm.viridis(np.linspace(0, 1, LIH_NUM_PARAMS))
    for i in range(LIH_NUM_PARAMS):
        ax2.plot(evaluations, params_arr[:, i],
                 color=colors[i], linewidth=0.6, alpha=0.6)
    ax2.set_ylabel(f"Angle parameters (rad)  [{LIH_NUM_PARAMS} params]")

    ax1.legend(loc="center right", framealpha=0.9)
    ax1.set_title(
        rf"LiH VQE convergence at $R = {R}$ Å "
        f"(Fig. 4A reproduction, {LIH_NUM_PARAMS} parameters)"
    )
    plt.tight_layout()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / "lih_convergence.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()