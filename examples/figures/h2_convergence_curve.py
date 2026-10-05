"""R=0.73 Å에서 H2 VQE의 수렴 궤적 (논문 Fig. 3A 재현).

논문 Fig. 3A와 같은 형태: 하나의 axes에 에너지(왼쪽 y축)와
6개 각도 파라미터(오른쪽 y축)를 겹쳐 표시.

두 panel로 나누지 않고 twinx로 겹치는 것이 논문의 형태다. 겹쳐 놓으면
"에너지가 떨어지는 구간"과 "파라미터가 요동치는 구간"이 같은 x좌표에서
읽히므로, 수렴이 어떻게 일어나는지가 한눈에 들어온다.

"Exact diag."는 VQE가 최적화하는 것과 동일한 2-qubit 축소 Hamiltonian의
최소 고윳값이다. 실험 참값도 full CI도 아니다.

multi-start가 아니라 단일 실행을 쓴다. 이 그림의 목적은 한 번의 최적화가
어떻게 진행되는지 보여주는 것이므로, 여러 시도 중 최선을 고르면 그 과정이
가려진다.

사용법:
    uv run python examples/figures/h2_convergence_curve.py
    PYTHONPATH=<cpp_repo>/build-py/python \
      uv run python examples/figures/h2_convergence_curve.py --kernel cpp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sqd_vqe.ansatz import H2_NUM_PARAMS, h2_ansatz_state
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import get_h2_hamiltonian
from sqd_vqe.vqe import run_vqe, run_vqe_cpp


# 식 (5)의 파라미터 순서와 맞춘다. params[:3]이 θ, params[3:]이 ω다.
PARAM_LABELS = [
    r"$\theta_0$", r"$\theta_1$", r"$\theta_2$",
    r"$\omega_0$", r"$\omega_1$", r"$\omega_2$",
]

#: 커널 이름 → 단일 실행 함수. sweep.py의 KERNELS와 같은 패턴이지만
#: 이쪽은 multi-start가 아니라 단일 실행이다.
KERNELS = {
    "numpy": run_vqe,
    "cpp": run_vqe_cpp,
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

    # 파일명을 커널별로 분리한다. 같은 이름을 쓰면 나중에 돌린 쪽이
    # 앞서 만든 그림을 덮어쓰고, README가 어느 쪽을 가리키는지도 모호해진다.
    suffix = "" if args.kernel == "numpy" else f"_{args.kernel}"
    # 제목에도 커널을 적는다. 그림 파일만 떼어 봐도 출처를 알 수 있어야 한다.
    kernel_note = "" if args.kernel == "numpy" else ", C++ kernel"

    R = 0.73  # H2의 결합 길이. 논문 Fig. 3A도 이 지점이다
    print(f"Running H2 VQE at R = {R} Å  (kernel: {args.kernel})...")

    H = get_h2_hamiltonian(distance=R)
    # eigvalsh는 Hermitian 전용이라 고윳값을 오름차순으로 돌려준다.
    # 따라서 [0]이 ground state energy다. 이 값은 커널과 무관하게 항상
    # NumPy로 구한다 — VQE의 산출물이 아니라 VQE를 재는 잣대이기 때문이다.
    exact = float(np.linalg.eigvalsh(hamiltonian_matrix(H))[0])
    result = run(H, ansatz=h2_ansatz_state, seed=42,
                 n_params=H2_NUM_PARAMS)

    print(f"Converged in {result.n_iterations} function evaluations")
    print(f"Final energy: {result.energy:.6f} Hartree")
    print(f"Exact energy: {exact:.6f} Hartree")
    print(f"Error: {abs(result.energy - exact):.2e} Hartree")

    # x축은 1부터 시작한다. "첫 번째 평가"가 0번이 아니라 1번으로 읽히는
    # 편이 자연스럽다.
    evaluations = np.arange(1, result.n_iterations + 1)
    # param_history는 (n_evals,) 길이의 배열 리스트다. 2차원 배열로 바꾸면
    # params_arr[:, i]로 i번째 파라미터의 궤적을 한 번에 꺼낼 수 있다.
    params_arr = np.array(result.param_history)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 왼쪽 y축: 에너지 (주된 데이터) --------------------------------
    # 검은 굵은 선 + 마커로 강조한다. 파라미터 6개와 겹쳐 그리므로
    # 시각적 위계를 분명히 해야 한다.
    ax1.plot(evaluations, result.energy_history,
             "o-", color="black", markersize=3, linewidth=1.8,
             label=r"$\langle H \rangle$", zorder=3)
    # 기준선은 점선 + 반투명. 배경 참조선이지 데이터가 아니다.
    ax1.axhline(exact, color="black", linestyle=":",
                linewidth=1.2, alpha=0.6,
                label=f"Exact diag. ({exact:.4f} Ha)")
    # "Iteration"이 아니라 "Function evaluation"이다. SciPy COBYLA의
    # maxiter는 실제로 함수 평가 횟수이며, 논문의 실험 iteration과는
    # 단위가 다르다.
    ax1.set_xlabel("Function evaluation")
    ax1.set_ylabel(r"$\langle H \rangle$ (Hartree)")
    ax1.grid(True, alpha=0.3)

    # --- 오른쪽 y축: 6개 파라미터 (부차 데이터) ------------------------
    ax2 = ax1.twinx()
    # tab10에서 앞쪽 60% 구간만 쓴다. 뒤쪽 색은 회색/갈색 계열이라
    # 검은 에너지 선과 구분이 약해진다.
    param_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    for i in range(6):
        ax2.plot(evaluations, params_arr[:, i],
                 color=param_colors[i], linewidth=1.0, alpha=0.8,
                 label=PARAM_LABELS[i])
    ax2.set_ylabel("Parameter value (rad)")

    # twinx로 만든 두 axes는 legend가 따로 생긴다. 핸들을 모아 하나로
    # 합치지 않으면 범례 상자가 두 개 겹쳐 그려진다.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", ncol=2, fontsize=9, framealpha=0.9)

    ax1.set_title(rf"$H_2$ VQE convergence at $R = {R}$ Å "
                  f"(Fig. 3A reproduction{kernel_note})")
    plt.tight_layout()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / f"h2_convergence{suffix}.png"
    # dpi=150이면 README에 임베드했을 때 선명하면서 파일이 과하게 커지지 않는다.
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()