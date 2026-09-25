"""H2 potential energy curve (논문 Fig. 3B 재현).

논문 Figure S10의 21개 R 값에서 multi-start VQE를 실행한다.
결과를 results/h2_pec.csv로도 저장한다 (7주차 C++ cross-check용).

"Exact diagonalization"은 VQE가 최적화하는 것과 동일한 2-qubit 축소
Hamiltonian의 최소 고윳값이다. 실험 참값이나 full CI가 아니다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from sqd_vqe.sweep import PAPER_H2_DISTANCES, sweep_h2_distances


CHEMICAL_ACCURACY = 1.6e-3  # Hartree
N_RESTARTS = 3


def main():
    print(f"Running H2 VQE for {len(PAPER_H2_DISTANCES)} distances "
          f"(multi-start {N_RESTARTS}회)...")
    points = sweep_h2_distances(
        PAPER_H2_DISTANCES, n_restarts=N_RESTARTS, seed=42, verbose=True
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "h2_pec.csv"
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

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    distances = [p.distance for p in points]

    ax1.plot(distances, [p.exact_energy for p in points],
             "k-", label="Exact diagonalization", linewidth=1.5)
    ax1.plot(distances, [p.vqe_energy for p in points],
             "o", color="tab:red", label="VQE", markersize=6)
    ax1.axvline(0.73, color="gray", linestyle="--", linewidth=1,
                alpha=0.6, label="Bonding length (0.73 Å)")
    ax1.set_ylabel(r"$\langle H \rangle_{\min}$ (Hartree)")
    ax1.set_title(r"$H_2$ Potential Energy Curve (Fig. 3B reproduction)")
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
    plot_path = out_dir / "h2_pec.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()