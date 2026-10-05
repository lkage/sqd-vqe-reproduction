"""LiH potential energy curve (논문 Fig. 4B 재현).

논문 Figure S11의 14개 R 값에서 multi-start VQE를 실행한다.

"Exact diagonalization"은 VQE가 최적화하는 것과 동일한 4-qubit(16x16)
축소 Hamiltonian의 최소 고윳값이다. Active space는 논문과 동일하게
2 electrons in 3 orbitals(Li 1s frozen, 2px/2py 제거)다.

H2판(Fig. 3B)과 구조는 같고 세 가지가 다르다:
  - R이 21개가 아니라 14개다 (Figure S11 기준)
  - restart가 3회가 아니라 5회다. 30차원은 단일 실행 성공률이 ~50%다
  - 실행 시간이 훨씬 길다. 14 × 5 × 평균 2000여 평가로 5분 정도 걸린다

참고: 논문의 LiH 실험 결과는 평균 오차 0.036 Hartree로 chemical
accuracy를 달성하지 못했다. OAM 상태의 purity/fidelity가 한계였다
(Figs. S2, S3). 우리는 노이즈 없는 시뮬레이션이므로 훨씬 정확하다.
재현 대상은 알고리즘의 거동이지 실험 오차가 아니다 — 우리 결과가 논문보다
좋게 나오는 것이 정상이고, 그렇지 않다면 오히려 구현을 의심해야 한다.

사용법:
    uv run python examples/figures/lih_potential_curve.py
    PYTHONPATH=<cpp_repo>/build-py/python \
      uv run python examples/figures/lih_potential_curve.py --kernel cpp
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from sqd_vqe.sweep import KERNELS, PAPER_LIH_DISTANCES, sweep_lih_distances


CHEMICAL_ACCURACY = 1.6e-3  # Hartree
# 30차원 비볼록 지형. 3회는 간혹 실패했고 5회는 실측상 항상 통과했다.
N_RESTARTS = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel", choices=sorted(KERNELS), default="numpy",
        help="energy evaluation kernel (default: numpy). "
             "cpp requires the qudit_simulator module on PYTHONPATH.",
    )
    args = parser.parse_args()

    # 파일명과 제목을 커널별로 분리한다. 5분짜리 작업이므로 실수로
    # 덮어쓰는 비용이 특히 크다.
    suffix = "" if args.kernel == "numpy" else f"_{args.kernel}"
    kernel_note = "" if args.kernel == "numpy" else ", C++ kernel"

    print(f"Running LiH VQE for {len(PAPER_LIH_DISTANCES)} distances "
          f"(multi-start {N_RESTARTS}회, kernel: {args.kernel})...")
    # verbose=True로 진행 상황을 찍는다. 5분 걸리는 작업이라 중간 출력이
    # 없으면 멈춘 것인지 도는 것인지 알 수 없다.
    points = sweep_lih_distances(
        PAPER_LIH_DISTANCES, n_restarts=N_RESTARTS, seed=42,
        verbose=True, kernel=args.kernel,
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # --- CSV 저장 -------------------------------------------------------
    # C++ cross-validation이 이 파일의 (R, E) 쌍을 기준으로 삼는다.
    # 5분짜리 작업이므로 결과를 파일에 남겨 두는 것이 특히 중요하다.
    csv_path = out_dir / f"lih_pec{suffix}.csv"
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

    # --- 요약 통계 ------------------------------------------------------
    errors = [p.error for p in points]
    evals = [p.n_iterations for p in points]
    n_ok = sum(e < CHEMICAL_ACCURACY for e in errors)

    print(f"\nMax error:  {max(errors):.2e} Hartree")
    print(f"Mean error: {sum(errors)/len(errors):.2e} Hartree")
    print(f"Within chemical accuracy: {n_ok}/{len(errors)}")
    print(f"Evaluations: mean={sum(evals)/len(evals):.0f}, max={max(evals)}")
    # 논문 수치를 함께 찍어 비교 맥락을 남긴다. 우리가 더 정확한 것이
    # 이상한 일이 아니라는 점을 출력에서도 알 수 있게 한다.
    print("(논문 실험: 평균 0.036 Hartree, chemical accuracy 미달성)")

    # --- 플롯 -----------------------------------------------------------
    # H2판과 동일한 레이아웃. 두 그림을 나란히 놓고 볼 때 형식이 같아야
    # 분자 간 차이가 눈에 들어온다.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    distances = [p.distance for p in points]

    ax1.plot(distances, [p.exact_energy for p in points],
             "k-", label="Exact diagonalization", linewidth=1.5)
    # 범례에 "(simulation)"을 붙인다. 논문의 Fig. 4B는 실험 데이터이므로,
    # 같은 모양의 그림이지만 출처가 다르다는 점을 명시한다.
    ax1.plot(distances, [p.vqe_energy for p in points],
             "o", color="tab:red", label="VQE (simulation)", markersize=6)
    ax1.axvline(1.55, color="gray", linestyle="--", linewidth=1,
                alpha=0.6, label="Bonding length (1.55 Å)")
    ax1.set_ylabel(r"$\langle H \rangle_{\min}$ (Hartree)")
    ax1.set_title("LiH Potential Energy Curve "
                  f"(Fig. 4B reproduction{kernel_note})")
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
    plot_path = out_dir / f"lih_pec{suffix}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()