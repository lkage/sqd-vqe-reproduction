"""VQE (Variational Quantum Eigensolver) 메인 루프.

논문 Kim et al., Sci. Adv. 10, eado3472 (2024)의 VQE 알고리즘을
SciPy COBYLA로 구현한다.

흐름:
    1. 초기 각도 파라미터 α_0
    2. ansatz(α_n) → state vector
    3. ⟨ψ|H|ψ⟩ → 에너지
    4. COBYLA로 α_{n+1} 갱신
    5. |α_{n+1} − α_n| < 0.01(논문 명시값) 수렴 시 종료

ansatz와 Hamiltonian이 분리되어 있어 H2(4D)와 LiH(16D)를 같은 루프로 다룬다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from sqd_vqe.ansatz import H2_NUM_PARAMS, h2_ansatz_state
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import PauliHamiltonian


AnsatzFn = Callable[[NDArray[np.float64]], NDArray[np.complex128]]


@dataclass
class VQEResult:
    """VQE 최적화 결과.

    energy: 최종 ground state energy 추정값 (평가된 값 중 최소)
    params: 그 지점의 각도 파라미터
    energy_history: 함수 평가별 에너지 (수렴 곡선용)
    param_history: 함수 평가별 파라미터 (Fig. 3A/4A 재현용)
    n_iterations: 총 함수 평가 횟수. SciPy COBYLA의 maxiter는 실제로
        함수 평가 횟수이며, 논문의 실험 iteration(H2 48±6.8,
        LiH 242±29.3)과는 다른 단위다
    n_restarts: multi-start 시도 횟수 (단일 실행이면 1)
    best_restart_seed: 최선의 결과를 낸 restart의 시드
    """
    energy: float
    params: NDArray[np.float64]
    energy_history: list[float]
    param_history: list[NDArray[np.float64]]
    n_iterations: int
    n_restarts: int = 1
    best_restart_seed: int | None = None


def run_vqe(
    hamiltonian: PauliHamiltonian,
    ansatz: AnsatzFn = h2_ansatz_state,
    initial_params: NDArray[np.float64] | None = None,
    seed: int = 42,
    n_params: int = H2_NUM_PARAMS,
    tolerance: float = 0.01,
    max_iter: int = 3000,
) -> VQEResult:
    """단일 초기값에서 VQE를 실행한다.

    hamiltonian: PauliHamiltonian
    ansatz: params → state vector. 기본은 H2 ansatz (식 5)
    initial_params: 초기값. None이면 seed 기반 난수
    seed: 난수 시드 (재현 가능성)
    tolerance: COBYLA 종료 기준(= rhoend). 논문 명시값 0.01.
        rhobeg는 SciPy 기본값 1.0을 쓴다 — 작게 주면 trust region이
        충분히 줄어들지 못해 max_iter까지 종료되지 않는다
    max_iter: 함수 평가 상한. 안전장치

    성능: Hamiltonian을 d×d 행렬로 한 번 빌드해 재사용한다. Pauli string을
    매번 순회하는 것보다 LiH(100 terms)에서 약 100배 빠르며, 결과는 수치적으로
    동일하다 (test_expectation.py가 두 경로의 일치를 검증).
    """
    if initial_params is None:
        rng = np.random.default_rng(seed)
        initial_params = rng.uniform(0, 2 * np.pi, size=n_params)

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
    ansatz: AnsatzFn = h2_ansatz_state,
    n_restarts: int = 3,
    seed: int = 42,
    n_params: int = H2_NUM_PARAMS,
    tolerance: float = 0.01,
    max_iter: int = 3000,
) -> VQEResult:
    """여러 무작위 초기값에서 VQE를 실행하고 최선의 결과를 반환한다.

    고차원 ansatz에서 COBYLA는 local minimum에 자주 갇힌다. LiH(30차원)의
    단일 실행 성공률은 약 50%이며, 5회 restart면 실측상 항상 chemical
    accuracy에 도달한다. 논문도 각 interatomic distance에서 실험을 여러 번
    수행하고 에너지 차이가 최소인 결과를 채택했다 (Figs. 3B, 4B 캡션).

    n_restarts: 서로 다른 초기값으로 시도할 횟수
    seed: 각 restart의 시드는 seed, seed+1, ... (재현 가능)

    반환: 가장 낮은 에너지를 얻은 시도의 VQEResult. history도 그 시도의 것.
    """
    best: VQEResult | None = None
    best_seed: int | None = None

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

    assert best is not None
    best.n_restarts = n_restarts
    best.best_restart_seed = best_seed
    return best