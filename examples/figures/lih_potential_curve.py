"""LiH potential energy curve (논문 Fig. 4B 재현).

논문 Figure S11의 14개 R 값에서 multi-start VQE를 실행한다.

"Exact diagonalization"은 VQE가 최적화하는 것과 동일한 4-qubit(16x16)
축소 Hamiltonian의 최소 고윳값이다. Active space는 논문과 동일하게
2 electrons in 3 orbitals(Li 1s frozen, 2px/2py 제거)다.

참고: 논문의 LiH 실험 결과는 평균 오차 0.036 Hartree로 chemical accuracy를
달성하지 못했다(OAM 상태의 purity/fidelity 한계, Figs. S2·S3). 우리는 노이즈
없는 시뮬레이션이므로 훨씬 정확하다. 재현 대상은 알고리즘의 거동이지 실험
오차가 아니다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from sqd_vqe.sweep import PAPER_LIH_DISTANCES, sweep_lih_distances


CHEMICAL_ACCURACY = 1.6e-3  # Hartree
N_RESTARTS = 5


def main():
    print(f"Running LiH VQE for {len(PAPER_LIH_DISTANCES)} distances "
          f"(multi-start {N_RESTARTS}회)...")
    points = sweep_lih_distances(
        PAPER_LIH_DISTANCES, n_restarts=N_RESTARTS, seed=42, verbose=True
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "lih_pec.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["distance_angstrom", "vqe_energy", "exact_energy",
             "error", "n_evaluations"]
        )
        for p in points:
            writer.writerow(
                [p.distance, p.vqe_energy, p.exact_energy,
                 p.error, p.n_iterations]
            )
    print(f"\nSaved results to {csv_path}")

    errors = [p.error for p in points]
    evals = [p.n_iterations for p in points]
    n_ok = sum(e < CHEMICAL_ACCURACY for e in errors)

    print(f"\nMax error:  {max(errors):.2e} Hartree")
    print(f"Mean error: {sum(errors)/len(errors):.2e} Hartree")
    print(f"Within chemical accuracy: {n_ok}/{len(errors)}")
    print(f"Evaluations: mean={sum(evals)/len(evals):.0f}, max={max(evals)}")
    print("(논문 실험: 평균 0.036 Hartree, chemical accuracy 미달성)")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    distances = [p.distance for p in points]

    ax1.plot(distances, [p.exact_energy for p in points],
             "k-", label="Exact diagonalization", linewidth=1.5)
    ax1.plot(distances, [p.vqe_energy for p in points],
             "o", color="tab:red", label="VQE (simulation)", markersize=6)
    ax1.axvline(1.55, color="gray", linestyle="--", linewidth=1,
                alpha=0.6, label="Bonding length (1.55 Å)")
    ax1.set_ylabel(r"$\langle H \rangle_{\min}$ (Hartree)")
    ax1.set_title("LiH Potential Energy Curve (Fig. 4B reproduction)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(distances, errors, "o-", color="tab:red", markersize=4)
    ax2.axhline(CHEMICAL_ACCURACY, color="blue", linestyle="--",
                label=f"Chemical accuracy ({CHEMICAL_ACCURACY*1000:.1f} mHa)")
    ax2.set_xlabel(r"Interatomic distance $R$ (Å)")
    ax2.set_ylabel(r"$|E_{\rm VQE} - E_{\rm exact}|$ (Ha)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plot_path = out_dir / "lih_pec.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()