"""에너지 평가 커널의 마이크로 벤치마크.

스윕 비교로는 시간을 잴 수 없다. 두 커널의 궤적이 갈리는 순간부터 평가하는
지점의 개수가 달라지므로, 벽시계 총합은 속도가 아니라 작업량을 재게 된다.
이 스크립트는 옵티마이저를 완전히 걷어내고 실제로 다른 그 단계만 잰다.

네 경로를 재서 두 축이 깔끔하게 분리되게 한다:

                    matvec                  termwise
    numpy           state.conj()@M@state    expectation_value()
    cpp             expectation_dense()     expectation()

세로로 비교하면 언어 차이가, 가로로 비교하면 알고리즘 차이가 분리된다.

측정상 주의한 것:

  - 같은 배열을 반복하지 않고 state 풀을 순환시킨다. 배열 하나를 계속 쓰면
    캐시에 그대로 남아 실제 작업량보다 빠르게 나온다. COBYLA는 호출마다
    다른 지점을 넘긴다.
  - 첫 호출들은 버린다. BLAS 초기화, pybind type-caster 셋업, 콜드 캐시가
    전부 거기에 몰린다.
  - 평균이 아니라 반복 중 최솟값을 쓴다. 스케줄링 노이즈는 느리게만 만들 수
    있으므로, 최솟값이 방해받지 않은 비용에 가장 가깝다.

두 번째 절에서는 두 커널이 동일한 궤적을 따르는 거리에 대해 end-to-end VQE를
잰다. 커널 속도 향상이 그대로 전체 속도가 되지는 않는다. ansatz 생성은 양쪽
모두 Python 재귀이고 COBYLA 자체의 부기도 영향을 받지 않으므로, end-to-end
수치는 전체 실행 시간 중 커널이 실제로 차지하는 비중을 보여준다.

사용법:
    PYTHONPATH=<cpp_repo>/build-py/python uv run python tools/benchmark_kernels.py
"""

from __future__ import annotations

import time
import timeit

import numpy as np

import qudit_simulator as qs
from sqd_vqe.ansatz import (
    H2_NUM_PARAMS, h2_ansatz_state, num_params_for_dim, qudit_ansatz_state,
)
from sqd_vqe.expectation import expectation_value, hamiltonian_matrix
from sqd_vqe.hamiltonian import get_h2_hamiltonian, get_lih_hamiltonian
from sqd_vqe.vqe import run_vqe, run_vqe_cpp

# 풀이 L1에 넉넉히 들어가지 않을 만큼은 많아야, 순환이 실제 최적화 경로를
# 흉내 낸다.
POOL_SIZE = 64
WARMUP = 200
REPEATS = 7
STATE_SEED = 12345

CASES = [
    ("h2", 0.73, get_h2_hamiltonian, 4),
    ("lih", 1.55, get_lih_hamiltonian, 16),
]

# 스윕에서 두 커널이 동일한 궤적을 보인 거리들. 그래야 end-to-end 벽시계가
# 비교 가능하다.
LOCKSTEP_H2_DISTANCES = [0.40, 0.50, 1.00, 1.50, 2.00]


def make_state_pool(dim: int, count: int) -> list[np.ndarray]:
    """물리적으로 그럴듯한 state: 무작위 파라미터의 ansatz 출력.

    Haar-random 벡터도 되지만, 커널이 실제로 보는 것은 ansatz state이고
    진폭 분포가 다르다.
    """
    rng = np.random.default_rng(STATE_SEED)
    n_params = num_params_for_dim(dim)
    return [
        qudit_ansatz_state(rng.uniform(0, 2 * np.pi, size=n_params), dim)
        for _ in range(count)
    ]


def time_calls(fn, states: list[np.ndarray], number: int) -> float:
    """호출당 나노초. REPEATS 중 최솟값."""
    n = len(states)
    counter = {"i": 0}

    def call():
        i = counter["i"]
        counter["i"] = (i + 1) % n
        return fn(states[i])

    for _ in range(WARMUP):
        call()

    timings = timeit.repeat(call, repeat=REPEATS, number=number)
    return min(timings) / number * 1e9


def pick_number(dim: int) -> int:
    """한 라운드당 호출 횟수. 타이머 해상도를 묻을 만큼은 되어야 한다."""
    return 20000 if dim <= 4 else 5000


def benchmark_kernels() -> None:
    for name, distance, ham_fn, dim in CASES:
        H = ham_fn(distance=distance)
        M = hamiltonian_matrix(H)
        H_cpp = qs.Hamiltonian([(c, label) for label, c in H.terms])

        states = make_state_pool(dim, POOL_SIZE)
        number = pick_number(dim)

        def numpy_matvec(state, M=M):
            return float(complex(state.conj() @ M @ state).real)

        def numpy_termwise(state, H=H):
            return expectation_value(state, H)

        paths = {
            "numpy matvec": numpy_matvec,
            "numpy termwise": numpy_termwise,
            "cpp dense": H_cpp.expectation_dense,
            "cpp termwise": H_cpp.expectation,
        }

        # 시간을 재기 전에 값부터 확인한다. 틀린 답을 빠르게 내는 건 의미가 없다.
        values = {k: fn(states[0]) for k, fn in paths.items()}
        spread = max(values.values()) - min(values.values())

        baseline = time_calls(numpy_matvec, states, number)

        print(f"\n{name}  dim={dim}, {len(H.terms)} terms, "
              f"R={distance}  ({number} calls/round, min of {REPEATS})")
        print(f"  {'path':<16s} {'ns/call':>10s} {'speedup':>9s}")
        for label, fn in paths.items():
            ns = time_calls(fn, states, number)
            print(f"  {label:<16s} {ns:>10.0f} {baseline / ns:>8.2f}x")
        print(f"  (max spread across paths on one state: {spread:.3e})")


def benchmark_construction() -> None:
    print("\nHamiltonian construction (one-off, reused across restarts)")
    print(f"  {'case':<30s} {'us':>10s}")
    for name, distance, ham_fn, dim in CASES:
        H = ham_fn(distance=distance)
        terms = [(c, label) for label, c in H.terms]

        t_np = min(timeit.repeat(
            lambda H=H: hamiltonian_matrix(H), repeat=5, number=200)) / 200
        t_cpp = min(timeit.repeat(
            lambda terms=terms: qs.Hamiltonian(terms),
            repeat=5, number=200)) / 200

        print(f"  {'numpy hamiltonian_matrix  ' + name:<30s} "
              f"{t_np * 1e6:>10.1f}")
        print(f"  {'cpp qs.Hamiltonian        ' + name:<30s} "
              f"{t_cpp * 1e6:>10.1f}")


def benchmark_end_to_end() -> None:
    print("\nEnd-to-end VQE on lockstep H2 distances")
    print(f"  {'R':>6s} {'n':>5s} {'py (ms)':>10s} {'cpp (ms)':>10s} "
          f"{'speedup':>9s}")

    t_py_sum = t_cpp_sum = 0.0
    for R in LOCKSTEP_H2_DISTANCES:
        H = get_h2_hamiltonian(distance=R)
        kw = dict(ansatz=h2_ansatz_state, seed=42, n_params=H2_NUM_PARAMS)

        t0 = time.perf_counter(); a = run_vqe(H, **kw);     t1 = time.perf_counter()
        t2 = time.perf_counter(); b = run_vqe_cpp(H, **kw); t3 = time.perf_counter()

        if a.n_iterations != b.n_iterations:
            note = f"  [diverged: {a.n_iterations} vs {b.n_iterations}]"
        else:
            note = ""

        t_py, t_cpp = (t1 - t0) * 1e3, (t3 - t2) * 1e3
        t_py_sum += t_py
        t_cpp_sum += t_cpp
        print(f"  {R:>6.2f} {a.n_iterations:>5d} {t_py:>10.1f} "
              f"{t_cpp:>10.1f} {t_py / t_cpp:>8.2f}x{note}")

    print(f"  {'total':>6s} {'':>5s} {t_py_sum:>10.1f} {t_cpp_sum:>10.1f} "
          f"{t_py_sum / t_cpp_sum:>8.2f}x")


def main() -> None:
    benchmark_kernels()
    benchmark_construction()
    benchmark_end_to_end()


if __name__ == "__main__":
    main()