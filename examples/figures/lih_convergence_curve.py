"""R=1.55 Å에서 LiH VQE의 수렴 궤적 (논문 Fig. 4A 재현).

논문 Fig. 4A와 같은 형태: 하나의 axes에 에너지(왼쪽 y축)와
30개 각도 파라미터(오른쪽 y축)를 겹쳐 표시.

H2판(6개 파라미터)과 다른 점:
  - 파라미터가 30개라 개별 추적이 불가능하다. 범례도 달 수 없다. 그래서
    선을 얇고 반투명하게 두고, 전체가 언제 안정화되는지만 보이게 한다.
    논문 Fig. 4A도 같은 인상이다.
  - 에너지 선에 마커를 찍지 않는다. 평가가 수천 번이라 마커를 찍으면
    선이 뭉개진다.
  - multi-start 5회 중 최선의 궤적을 그린다. 단일 실행은 LiH에서 절반쯤
    local minimum에 갇히므로, 그 궤적을 그리면 "수렴 과정"이 아니라
    "실패 과정"을 보여주게 된다.

run_vqe_multistart가 반환하는 history는 여러 시도를 이어 붙인 것이 아니라
최선이었던 그 한 번의 것이다. 따라서 수렴 곡선으로서 의미가 유지된다.

"Exact diag."는 VQE가 최적화하는 것과 동일한 4-qubit(16x16) 축소
Hamiltonian의 최소 고윳값이다.

사용법:
    uv run python examples/figures/lih_convergence_curve.py
    PYTHONPATH=<cpp_repo>/build-py/python \
      uv run python examples/figures/lih_convergence_curve.py --kernel cpp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sqd_vqe.ansatz import LIH_NUM_PARAMS, lih_ansatz_state
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import get_lih_hamiltonian
from sqd_vqe.vqe import run_vqe_multistart, run_vqe_multistart_cpp


# 실측상 3회는 간혹 실패하고 5회는 항상 성공했다.
# (examples/diagnostics/lih_vqe_landscape.py 참조)
N_RESTARTS = 5

KERNELS = {
    "numpy": run_vqe_multistart,
    "cpp": run_vqe_multistart_cpp,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel", choices=sorted(KERNELS), default="numpy",
        help="energy evaluation kernel (default: numpy). "
             "cpp requires the qudit_simulator module on PYTHONPATH.",
    )
    args = parser.parse_args()
    run = KERNELS[args.kernel]

    # 파일명과 제목을 커널별로 분리한다. 같은 이름을 쓰면 나중에 돌린 쪽이
    # 앞서 만든 그림을 덮어쓴다.
    suffix = "" if args.kernel == "numpy" else f"_{args.kernel}"
    kernel_note = "" if args.kernel == "numpy" else ", C++ kernel"

    R = 1.55  # LiH의 결합 길이. 논문 Table S2와 Fig. 4A의 기준 거리
    print(f"Running LiH VQE at R = {R} Å "
          f"(multi-start {N_RESTARTS}회, kernel: {args.kernel})...")

    # 캐시된 Hamiltonian을 쓴다. 캐시 없이 돌리면 Qiskit의 ULP 비결정성
    # 때문에 실행마다 다른 그림이 나온다.
    H = get_lih_hamiltonian(distance=R)
    # 커널과 무관하게 항상 NumPy로 구한다 — VQE를 재는 잣대이기 때문이다.
    exact = float(np.linalg.eigvalsh(hamiltonian_matrix(H))[0])
    result = run(
        H, ansatz=lih_ansatz_state, n_restarts=N_RESTARTS,
        seed=42, n_params=LIH_NUM_PARAMS, max_iter=3000,
    )

    # best_restart_seed를 찍어 두면 그림을 다시 만들 때 어느 시도였는지
    # 추적할 수 있다. 커널을 바꾸면 이 값이 달라질 수도 있다 — 목적 함수가
    # 몇 ULP 다르므로 어느 restart가 이기는지가 뒤집힐 수 있기 때문이다.
    print(f"Best restart seed: {result.best_restart_seed}")
    print(f"Converged in {result.n_iterations} function evaluations")
    print(f"Final energy: {result.energy:.6f} Hartree")
    print(f"Exact energy: {exact:.6f} Hartree")
    print(f"Error: {abs(result.energy - exact):.2e} Hartree")

    evaluations = np.arange(1, result.n_iterations + 1)
    # shape (n_evals, 30). params_arr[:, i]가 i번째 파라미터의 궤적이다.
    params_arr = np.array(result.param_history)

    # H2판보다 가로로 길게 잡는다. 평가 횟수가 수천이라 x축이 길어야
    # 초반 요동 구간이 뭉개지지 않는다.
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # --- 왼쪽 y축: 에너지 ----------------------------------------------
    # 마커 없이 선만. 30개 파라미터 선 위에서도 눈에 띄도록 굵게 두고
    # zorder로 맨 앞에 놓는다.
    ax1.plot(evaluations, result.energy_history,
             "-", color="black", linewidth=1.8,
             label=r"$\langle H \rangle$", zorder=3)
    ax1.axhline(exact, color="black", linestyle=":",
                linewidth=1.2, alpha=0.6,
                label=f"Exact diag. ({exact:.4f} Ha)")
    ax1.set_xlabel("Function evaluation")
    ax1.set_ylabel(r"$\langle H \rangle$ (Hartree)")
    ax1.grid(True, alpha=0.3)

    # --- 오른쪽 y축: 30개 파라미터 -------------------------------------
    ax2 = ax1.twinx()
    # viridis를 쓴다. 30개를 구분할 수는 없지만 연속 컬러맵이라 전체가
    # 하나의 다발로 읽히고, 개별 선을 추적하려는 시도를 유도하지 않는다.
    colors = plt.cm.viridis(np.linspace(0, 1, LIH_NUM_PARAMS))
    for i in range(LIH_NUM_PARAMS):
        # 얇고 반투명하게. 30개가 겹쳐도 밀도가 보이도록.
        ax2.plot(evaluations, params_arr[:, i],
                 color=colors[i], linewidth=0.6, alpha=0.6)
    # 범례 대신 축 라벨에 개수를 적는다. 30개를 범례에 넣을 수는 없다.
    ax2.set_ylabel(f"Angle parameters (rad)  [{LIH_NUM_PARAMS} params]")

    # 파라미터 선에는 label을 주지 않았으므로 ax1의 범례만 그리면 된다.
    # H2판처럼 두 axes의 핸들을 합칠 필요가 없다.
    ax1.legend(loc="center right", framealpha=0.9)
    ax1.set_title(
        rf"LiH VQE convergence at $R = {R}$ Å "
        f"(Fig. 4A reproduction, {LIH_NUM_PARAMS} parameters{kernel_note})"
    )
    plt.tight_layout()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / f"lih_convergence{suffix}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()