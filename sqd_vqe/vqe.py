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


def run_vqe(
    hamiltonian: PauliHamiltonian,
    ansatz: Callable[[NDArray[np.float64]], NDArray[np.complex128]] = h2_ansatz_state,
    initial_params: NDArray[np.float64] | None = None,
    seed: int = 42,
    n_params: int = 6,
    tolerance: float = 0.01,
    max_iter: int = 500,
) -> VQEResult:
    """VQE 메인 루프.

    hamiltonian: PauliHamiltonian. build_h2_hamiltonian() 등으로 생성.
    ansatz: params → state vector. 기본은 H2 ansatz (식 5).
    initial_params: 초기값. None이면 seed 기반 난수.
    seed: 난수 시드 (재현 가능성).
    tolerance: COBYLA 종료 기준. 논문은 |α_{n+1} - α_n| < 0.01.
    max_iter: COBYLA 최대 iteration. 안전장치.
    """
    if initial_params is None:
        rng = np.random.default_rng(seed)
        initial_params = rng.uniform(0, 2 * np.pi, size=n_params)

    energy_history: list[float] = []
    param_history: list[NDArray[np.float64]] = []

    def objective(params: NDArray[np.float64]) -> float:
        state = ansatz(params)
        energy = expectation_value(state, hamiltonian)
        energy_history.append(energy)
        param_history.append(params.copy())
        return energy

    result = minimize(
        objective,
        x0=initial_params,
        method="COBYLA",
        options={
            "rhobeg": 0.5,          # 초기 step 크게 (한 자릿수 정도)
            "tol": tolerance,        # 종료 기준 (= rhoend)
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