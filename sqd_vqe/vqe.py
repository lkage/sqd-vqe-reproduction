"""VQE (Variational Quantum Eigensolver) 메인 루프.

논문 Kim et al., Sci. Adv. 10, eado3472 (2024)의 VQE 알고리즘을
COBYLA optimizer로 구현한다.

핵심 흐름:
    1. 초기 각도 파라미터 α_0
    2. ansatz(α_n) → state vector
    3. expectation_value(state, H) → 에너지
    4. COBYLA로 α_{n+1} 갱신
    5. |α_{n+1} - α_n| < 0.01 수렴 시 종료

이 모듈은 ansatz와 H가 분리되어 있어 추후 LiH(16D)로 확장 시 ansatz만
교체하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from sqd_vqe.ansatz import h2_ansatz_state
from sqd_vqe.expectation import expectation_value
from sqd_vqe.hamiltonian import PauliHamiltonian


@dataclass
class VQEResult:
    """VQE 최적화 결과.

    energy: 최종 ground state energy 추정값
    params: 수렴한 각도 파라미터
    energy_history: iteration별 에너지 (수렴 곡선용)
    param_history: iteration별 파라미터 (논문 Fig. 3A 재현용)
    n_iterations: 총 iteration 수
    """
    energy: float
    params: NDArray[np.float64]
    energy_history: list[float]
    param_history: list[NDArray[np.float64]]
    n_iterations: int
    n_restarts: int = 1          # 추가
    best_restart_seed: int | None = None   # 추가


from sqd_vqe.expectation import hamiltonian_matrix


def run_vqe(
    hamiltonian: PauliHamiltonian,
    ansatz: Callable[[NDArray[np.float64]], NDArray[np.complex128]] = h2_ansatz_state,
    initial_params: NDArray[np.float64] | None = None,
    seed: int = 42,
    n_params: int = 6,
    tolerance: float = 0.01,
    max_iter: int = 3000,
) -> VQEResult:
    """(기존 docstring 유지)

    성능: Hamiltonian을 d×d 행렬로 한 번 빌드해서 재사용한다.
    Pauli string을 매번 순회하는 것보다 LiH(100 terms)에서 약 100배 빠르다.
    결과는 수치적으로 동일 (test_expectation.py가 두 경로의 일치를 보장).
    """
    if initial_params is None:
        rng = np.random.default_rng(seed)
        initial_params = rng.uniform(0, 2 * np.pi, size=n_params)

    # Hamiltonian 행렬을 한 번만 빌드
    M = hamiltonian_matrix(hamiltonian)

    energy_history: list[float] = []
    param_history: list[NDArray[np.float64]] = []

    def objective(params: NDArray[np.float64]) -> float:
        state = ansatz(params)
        energy = complex(state.conj() @ M @ state)
        if abs(energy.imag) > 1e-9:
            raise ValueError(f"Non-real expectation value: {energy}")
        e = float(energy.real)
        energy_history.append(e)
        param_history.append(params.copy())
        return e

    result = minimize(
        objective,
        x0=initial_params,
        method="COBYLA",
        options={
            "rhobeg": 1.0,
            "tol": tolerance,
            "maxiter": max_iter,
            "disp": False,
        },
    )

    return VQEResult(
        energy=float(result.fun),
        params=np.asarray(result.x),
        energy_history=energy_history,
        param_history=param_history,
        n_iterations=len(energy_history),
    )

def run_vqe_multistart(
    hamiltonian: PauliHamiltonian,
    ansatz: Callable[[NDArray[np.float64]], NDArray[np.complex128]] = h2_ansatz_state,
    n_restarts: int = 10,
    seed: int = 42,
    n_params: int = 6,
    tolerance: float = 0.01,
    max_iter: int = 3000,
) -> VQEResult:
    """여러 무작위 초기값에서 VQE를 실행하고 최선의 결과를 반환.

    고차원 ansatz(LiH의 30차원)에서 COBYLA는 local minimum에 자주
    갇힌다. 논문도 각 interatomic distance에서 실험을 여러 번 수행하고
    에너지 차이가 최소인 결과를 채택했다 (Fig. 4B 캡션).

    n_restarts: 서로 다른 초기값으로 시도할 횟수.
    seed: 각 restart의 시드는 seed, seed+1, ... 로 결정 (재현 가능).

    반환: 가장 낮은 에너지를 얻은 시도의 VQEResult.
    """
    best: VQEResult | None = None

    for i in range(n_restarts):
        result = run_vqe(
            hamiltonian,
            ansatz=ansatz,
            seed=seed + i,
            n_params=n_params,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        if best is None or result.energy < best.energy:
            best = result
            best_seed = seed + i

    best.n_restarts = n_restarts
    best.best_restart_seed = best_seed
    return best