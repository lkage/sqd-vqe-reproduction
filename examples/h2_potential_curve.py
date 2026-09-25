"""H2 potential energy curve를 생성하고 논문 Fig. 3B와 비교.

논문 Table S1의 21개 R 값에서 VQE를 실행하여 potential energy curve를
그린다. 결과를 results/h2_pec.csv로도 저장한다 (7주차 C++ cross-check용).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from sqd_vqe.sweep import PAPER_DISTANCES, sweep_distances


CHEMICAL_ACCURACY = 1.6e-3  # Hartree


def main():
    print(f"Running VQE for {len(PAPER_DISTANCES)} interatomic distances...")
    points = sweep_distances(PAPER_DISTANCES, seed=42, verbose=True)

    # CSV 저장
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "h2_pec.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["distance_angstrom", "vqe_energy", "exact_energy",
             "error", "n_iterations"]
        )
        for p in points:
            writer.writerow(
                [p.distance, p.vqe_energy, p.exact_energy,
                 p.error, p.n_iterations]
            )
    print(f"\nSaved results to {csv_path}")

    # 통계
    errors = [p.error for p in points]
    iters = [p.n_iterations for p in points]
    print(f"\nMax error: {max(errors):.2e} Hartree")
    print(f"Mean error: {sum(errors)/len(errors):.2e} Hartree")
    print(f"Chemical accuracy threshold: {CHEMICAL_ACCURACY:.2e}")
    print(f"Points within chemical accuracy: "
          f"{sum(e < CHEMICAL_ACCURACY for e in errors)}/{len(errors)}")
    print(f"Iterations: mean={sum(iters)/len(iters):.0f}, max={max(iters)}")

    # 플롯 — 논문 Fig. 3B 스타일
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    distances = [p.distance for p in points]

    # 상단: PEC
    ax1.plot(distances, [p.exact_energy for p in points],
             "k-", label="Theory (exact)", linewidth=1.5)
    ax1.plot(distances, [p.vqe_energy for p in points],
             "o", color="tab:red", label="VQE", markersize=6)
    ax1.set_ylabel(r"$\langle H \rangle_{\min}$ (Hartree)")
    ax1.set_title(r"$H_2$ Potential Energy Curve (Fig. 3B reproduction)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 하단: error
    ax2.semilogy(distances, errors, "o-", color="tab:red", markersize=4)
    ax2.axhline(CHEMICAL_ACCURACY, color="blue", linestyle="--",
            label=f"Chemical accuracy ({CHEMICAL_ACCURACY*1000:.1f} mHa)")
    ax2.set_xlabel(r"Interatomic distance $R$ (Å)")
    ax2.set_ylabel(r"$\Delta E$ (Hartree)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plot_path = out_dir / "h2_pec.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()