"""R=0.73 Å에서 H2 VQE의 수렴 궤적 (논문 Fig. 3A 재현).

논문 Fig. 3A와 같은 형태: 하나의 axes에 에너지(왼쪽 y축)와
6개 각도 파라미터(오른쪽 y축)를 겹쳐 표시.

"Exact diag."는 VQE가 최적화하는 것과 동일한 2-qubit 축소 Hamiltonian의
최소 고윳값이다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import get_h2_hamiltonian
from sqd_vqe.vqe import run_vqe


PARAM_LABELS = [
    r"$\theta_0$", r"$\theta_1$", r"$\theta_2$",
    r"$\omega_0$", r"$\omega_1$", r"$\omega_2$",
]


def main():
    R = 0.73
    print(f"Running H2 VQE at R = {R} Å...")

    H = get_h2_hamiltonian(distance=R)
    exact = float(np.linalg.eigvalsh(hamiltonian_matrix(H))[0])
    result = run_vqe(H, seed=42)

    print(f"Converged in {result.n_iterations} function evaluations")
    print(f"Final energy: {result.energy:.6f} Hartree")
    print(f"Exact energy: {exact:.6f} Hartree")
    print(f"Error: {abs(result.energy - exact):.2e} Hartree")

    evaluations = np.arange(1, result.n_iterations + 1)
    params_arr = np.array(result.param_history)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(evaluations, result.energy_history,
             "o-", color="black", markersize=3, linewidth=1.8,
             label=r"$\langle H \rangle$", zorder=3)
    ax1.axhline(exact, color="black", linestyle=":",
                linewidth=1.2, alpha=0.6,
                label=f"Exact diag. ({exact:.4f} Ha)")
    ax1.set_xlabel("Function evaluation")
    ax1.set_ylabel(r"$\langle H \rangle$ (Hartree)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    param_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    for i in range(6):
        ax2.plot(evaluations, params_arr[:, i],
                 color=param_colors[i], linewidth=1.0, alpha=0.8,
                 label=PARAM_LABELS[i])
    ax2.set_ylabel("Parameter value (rad)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", ncol=2, fontsize=9, framealpha=0.9)

    ax1.set_title(rf"$H_2$ VQE convergence at $R = {R}$ Å "
                  "(Fig. 3A reproduction)")
    plt.tight_layout()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / "h2_convergence.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()