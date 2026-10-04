"""두 커널로 R 스윕 전체를 돌리고 각각을 exact 값과 비교한다.

두 커널이 서로 일치할 필요는 없다. 목적 함수가 1~4 ULP 다르고, COBYLA는
값을 비교해서 움직이는 알고리즘이다. 두 후보의 에너지 차이가 그 폭 안으로
좁혀지는 지점에서 비교가 뒤집히면 이후 궤적이 갈린다. 예상된 거동이다.

대신 각 커널을 같은 축소 Hamiltonian의 정확한 최소 고윳값과 비교한다.
이것이 재는 것은 하나뿐이다 — COBYLA가 이 파라미터 공간에서 최소점을
얼마나 잘 찾았는가. Hamiltonian 자체가 물리적으로 옳은지는 별개 문제이고,
Table S1/S2 비교 테스트가 담당한다.

모드가 둘이고, 통과 기준이 다르다:

    single      거리마다 고정 seed로 한 번씩. LiH 단일 실행은 local minimum을
                절반쯤 탈출하지 못한다 — multi-start가 존재하는 이유다 —
                그래서 절대 정확도는 여기서 맞는 잣대가 아니다. 대신 두
                커널이 심하게 어긋나지 않는지를 본다. 한쪽은 chemical
                accuracy 안에 들어왔는데 다른 쪽이 그 10배 넘게 빗나간
                지점이라면, 흔한 local minimum 추첨이 아니라 실제 결함이다.

    multistart  n_restarts 중 최선. 논문 그림이 쓰는 설정이다. 여기서는 두
                커널 모두에게 절대적인 chemical accuracy를 요구한다.

Fidelity |<psi_py|psi_cpp>|^2는 출력하되 통과/실패 판정에는 쓰지 않는다.
COBYLA는 파라미터 공간에서 tol=0.01에 멈추므로 각 각도가 그 정도 폭까지만
고정된다. LiH의 30개 파라미터라면 infidelity 하한이 대략
30 * (0.01/2)^2 = 7.5e-4다. 즉 fidelity 0.999 언저리가 이 설정에서 나올 수
있는 최선이다. 그래도 출력에 남기는 이유는, 그 하한보다 한참 낮은 값이
나오면 두 실행이 서로 다른 basin에 안착했다는 신호이기 때문이다.

사용법:
    PYTHONPATH=<cpp_repo>/build/python uv run python tools/compare_sweep.py
    PYTHONPATH=... uv run python tools/compare_sweep.py --mode multistart
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from sqd_vqe.ansatz import (
    H2_NUM_PARAMS, LIH_NUM_PARAMS, h2_ansatz_state, lih_ansatz_state,
)
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import get_h2_hamiltonian, get_lih_hamiltonian
from sqd_vqe.sweep import PAPER_H2_DISTANCES, PAPER_LIH_DISTANCES
from sqd_vqe.vqe import (
    run_vqe, run_vqe_cpp, run_vqe_multistart, run_vqe_multistart_cpp,
)

CASES = [
    ("h2", PAPER_H2_DISTANCES, get_h2_hamiltonian,
     h2_ansatz_state, H2_NUM_PARAMS, 3),
    ("lih", PAPER_LIH_DISTANCES, get_lih_hamiltonian,
     lih_ansatz_state, LIH_NUM_PARAMS, 5),
]

SEED = 42
MAX_ITER = 3000
CHEMICAL_ACCURACY = 1.6e-3

#: single 모드에서, 한 커널은 통과했는데 다른 커널이 chemical accuracy의 이
#: 배수를 넘겨 빗나가면 '심한 불일치'로 센다. 임계값을 사이에 두고 걸치는
#: 정도(1.6e-3 대 1.5e-3)는 같은 결과를 날카로운 컷오프로 본 것일 뿐이지
#: 커널이 깨졌다는 증거가 아니다.
GROSS_FACTOR = 10.0


def first_divergence(ha: list[float], hb: list[float]) -> int | None:
    """두 history가 처음으로 달라지는 인덱스.

    비트 단위 비교라 1 ULP 차이도 잡는다. 갈라지는 것 자체는 예외가 아니라
    기본값이다. 이 인덱스가 유용한 건, 목적 함수 차이가 처음으로 비교를
    뒤집기까지 두 실행이 얼마나 오래 보조를 맞췄는지 보여주기 때문이다.
    """
    for i in range(min(len(ha), len(hb))):
        if ha[i] != hb[i]:
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("single", "multistart"), default="single",
        help="single: one fixed-seed run per distance (default). "
             "multistart: best of n_restarts, as the paper figures use.",
    )
    args = parser.parse_args()
    multistart = args.mode == "multistart"

    max_err = {"py": 0.0, "cpp": 0.0}
    worst_at = {"py": "", "cpp": ""}
    fails = {"py": 0, "cpp": 0}
    min_fidelity = 1.0
    min_fidelity_at = ""
    gross: list[str] = []
    same_outcome = 0
    lockstep = 0
    total_runs = 0
    maxiter_hits: list[str] = []
    t_total = {"py": 0.0, "cpp": 0.0}

    header = (f"{'R':>7s} {'exact':>16s} {'err_py':>10s} {'err_cpp':>10s} "
              f"{'n_py':>6s} {'n_cpp':>6s} {'fid':>9s}")
    if not multistart:
        header += f" {'split@':>7s}"

    for name, distances, ham_fn, ansatz, n_params, n_restarts in CASES:
        label = (f"multi-start {n_restarts}" if multistart
                 else f"single run, seed {SEED}")
        print(f"\n{name}  ({len(distances)} distances, {label})")
        print(header)

        for R in distances:
            H = ham_fn(distance=float(R))
            exact = float(np.linalg.eigvalsh(hamiltonian_matrix(H))[0])

            kw = dict(ansatz=ansatz, seed=SEED, n_params=n_params,
                      max_iter=MAX_ITER)

            t0 = time.perf_counter()
            if multistart:
                a = run_vqe_multistart(H, n_restarts=n_restarts, **kw)
            else:
                a = run_vqe(H, **kw)
            t1 = time.perf_counter()
            if multistart:
                b = run_vqe_multistart_cpp(H, n_restarts=n_restarts, **kw)
            else:
                b = run_vqe_cpp(H, **kw)
            t2 = time.perf_counter()

            t_total["py"] += t1 - t0
            t_total["cpp"] += t2 - t1
            total_runs += 1

            err = {"py": abs(a.energy - exact), "cpp": abs(b.energy - exact)}
            passed = {k: err[k] <= CHEMICAL_ACCURACY for k in err}

            for k in ("py", "cpp"):
                if err[k] > max_err[k]:
                    max_err[k] = err[k]
                    worst_at[k] = f"{name} R={R:.2f}"
                if not passed[k]:
                    fails[k] += 1

            if passed["py"] == passed["cpp"]:
                same_outcome += 1
            else:
                loser = "cpp" if passed["py"] else "py"
                if err[loser] > GROSS_FACTOR * CHEMICAL_ACCURACY:
                    gross.append(f"{name} R={R:.2f} ({loser} off by "
                                 f"{err[loser]:.3e})")

            fid = float(abs(np.vdot(ansatz(a.params), ansatz(b.params))) ** 2)
            if fid < min_fidelity:
                min_fidelity = fid
                min_fidelity_at = f"{name} R={R:.2f}"

            flag = ""
            if not passed["py"]:
                flag += " PY"
            if not passed["cpp"]:
                flag += " CPP"
            if a.n_iterations >= MAX_ITER or b.n_iterations >= MAX_ITER:
                flag += " MAXITER"
                maxiter_hits.append(f"{name} R={R:.2f}")

            row = (f"{R:>7.2f} {exact:>16.12f} "
                   f"{err['py']:>10.3e} {err['cpp']:>10.3e} "
                   f"{a.n_iterations:>6d} {b.n_iterations:>6d} "
                   f"{fid:>9.6f}")

            if not multistart:
                split = first_divergence(a.energy_history, b.energy_history)
                if split is None:
                    lockstep += 1
                    row += f" {'same':>7s}"
                else:
                    row += f" {split:>7d}"

            print(row + flag)

    print(f"\nmax |E - exact|:  py {max_err['py']:.3e} ({worst_at['py']}), "
          f"cpp {max_err['cpp']:.3e} ({worst_at['cpp']})")
    print(f"missed chemical accuracy ({CHEMICAL_ACCURACY:.1e}):  "
          f"py {fails['py']}/{total_runs}, cpp {fails['cpp']}/{total_runs}")
    print(f"same pass/fail outcome: {same_outcome}/{total_runs}")
    print(f"min fidelity:     {min_fidelity:.9f}  ({min_fidelity_at})"
          f"   [reported, not a pass criterion]")

    if not multistart:
        print(f"bitwise-identical trajectories: {lockstep}/{total_runs}")

    if maxiter_hits:
        print(f"hit max_iter={MAX_ITER}: {', '.join(maxiter_hits)}"
              f"   [not converged; these points say little]")
    else:
        print(f"hit max_iter={MAX_ITER}: none")

    print(f"total time: numpy {t_total['py']:.1f}s, cpp {t_total['cpp']:.1f}s")
    if multistart or lockstep < total_runs:
        print("  (not a fair comparison: diverged trajectories mean the two "
              "kernels did different amounts of work)")
    else:
        print(f"  ({t_total['py'] / t_total['cpp']:.2f}x, comparable)")

    if multistart:
        ok = fails["py"] == 0 and fails["cpp"] == 0
        criterion = "both kernels within chemical accuracy everywhere"
    else:
        ok = not gross
        criterion = (f"no point where one kernel passes and the other misses "
                     f"by more than {GROSS_FACTOR:g}x chemical accuracy")
        if gross:
            print(f"gross disagreements: {'; '.join(gross)}")

    print(f"criterion: {criterion}")
    print(f"result: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())