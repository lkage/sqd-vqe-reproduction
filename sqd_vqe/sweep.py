"""Interatomic distance 스윕을 통한 potential energy curve 생성.

각 R에 대해 Hamiltonian을 만들고 VQE로 ground state energy를 추정한다.
논문 Fig. 3B(H2), Fig. 4B(LiH)에 해당하는 데이터를 생성한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sqd_vqe.ansatz import (
    H2_NUM_PARAMS,
    LIH_NUM_PARAMS,
    h2_ansatz_state,
    lih_ansatz_state,
)
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import get_h2_hamiltonian, get_lih_hamiltonian
from sqd_vqe.vqe import run_vqe_multistart


# 논문 Figure S10의 H2 21개 R 값 (단위 Å)
PAPER_H2_DISTANCES: NDArray[np.float64] = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.73, 0.8, 0.9,
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 2.5, 3.0,
])

# 논문 Figure S11의 LiH(Qiskit) 14개 R 값 (단위 Å)
PAPER_LIH_DISTANCES: NDArray[np.float64] = np.array([
    0.1, 0.5, 0.8, 1.1, 1.3, 1.4, 1.5,
    1.55, 1.6, 1.7, 1.8, 2.2, 2.8, 3.4,
])


@dataclass
class PECPoint:
    """Potential energy curve의 한 점.

    exact_energy는 VQE가 최적화하는 것과 동일한 축소 Hamiltonian 행렬의
    최소 고윳값이다. 즉 이 ansatz 공간에서 도달 가능한 이론적 최선이며,
    full CI나 실험 참값이 아니다. 따라서 error가 재는 것은 "COBYLA가 이
    공간에서 최소점을 얼마나 잘 찾았는가" 하나뿐이다. Hamiltonian 자체의
    정확성은 논문 Table S1/S2와의 비교 테스트가 별도로 담당한다.
    """
    distance: float       # Å
    vqe_energy: float     # VQE 추정값
    exact_energy: float   # 축소 Hamiltonian의 최소 고윳값
    n_iterations: int     # 최선 restart의 함수 평가 횟수

    @property
    def error(self) -> float:
        """VQE 오차 (절대값)."""
        return abs(self.vqe_energy - self.exact_energy)


def _sweep(
    distances: NDArray[np.float64],
    hamiltonian_fn,
    ansatz,
    n_params: int,
    n_restarts: int,
    seed: int,
    verbose: bool,
) -> list[PECPoint]:
    """R 스윕의 공통 루프."""
    results: list[PECPoint] = []

    for R in distances:
        H = hamiltonian_fn(distance=float(R))
        exact = float(np.linalg.eigvalsh(hamiltonian_matrix(H))[0])
        vqe_result = run_vqe_multistart(
            H,
            ansatz=ansatz,
            n_restarts=n_restarts,
            seed=seed,
            n_params=n_params,
        )

        point = PECPoint(
            distance=float(R),
            vqe_energy=vqe_result.energy,
            exact_energy=exact,
            n_iterations=vqe_result.n_iterations,
        )
        results.append(point)

        if verbose:
            print(
                f"  R={R:.2f} Å: VQE={point.vqe_energy:+.6f}, "
                f"exact={point.exact_energy:+.6f}, "
                f"err={point.error:.2e}, evals={point.n_iterations}"
            )

    return results


def sweep_h2_distances(
    distances: NDArray[np.float64] = PAPER_H2_DISTANCES,
    n_restarts: int = 3,
    seed: int = 42,
    verbose: bool = False,
) -> list[PECPoint]:
    """H2에 대해 R 스윕. 각 R에서 multi-start VQE 실행.

    단일 실행은 COBYLA tol=0.01(논문 명시값)의 한계로 간혹 chemical
    accuracy를 근소하게 넘긴다. 논문도 각 R에서 여러 번 실험하고
    최선을 채택했다 (Fig. 3B 캡션).
    """
    return _sweep(
        distances, get_h2_hamiltonian, h2_ansatz_state,
        H2_NUM_PARAMS, n_restarts, seed, verbose,
    )


def sweep_lih_distances(
    distances: NDArray[np.float64] = PAPER_LIH_DISTANCES,
    n_restarts: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> list[PECPoint]:
    """LiH에 대해 R 스윕. 각 R에서 multi-start VQE 실행.

    30차원 ansatz는 local minimum이 많아 단일 실행 성공률이 ~50%다.
    Hamiltonian은 캐시에서 읽는다 (hamiltonian.py의 재현성 주석 참조).
    """
    return _sweep(
        distances, get_lih_hamiltonian, lih_ansatz_state,
        LIH_NUM_PARAMS, n_restarts, seed, verbose,
    )