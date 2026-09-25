"""Interatomic distance 스윕을 통한 potential energy curve 생성.

각 R에 대해 H2 Hamiltonian을 새로 만들고 VQE로 ground state energy를
추정한다. 정확한 lowest eigenvalue도 함께 계산해서 비교 기준 제공.

논문 Fig. 3B에 해당하는 데이터를 생성한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import build_h2_hamiltonian
from sqd_vqe.vqe import run_vqe


@dataclass
class PECPoint:
    """Potential energy curve의 한 점."""
    distance: float       # Å
    vqe_energy: float     # VQE 추정값
    exact_energy: float   # Hamiltonian 행렬의 lowest eigenvalue
    n_iterations: int     # VQE iteration 수

    @property
    def error(self) -> float:
        """VQE 오차 (절대값)."""
        return abs(self.vqe_energy - self.exact_energy)


def sweep_distances(
    distances: NDArray[np.float64],
    seed: int = 42,
    verbose: bool = False,
) -> list[PECPoint]:
    """주어진 R 값들에 대해 VQE를 실행하고 결과 모음을 반환.

    distances: R 값 배열 (단위 Å).
    seed: 각 R 지점에서 사용할 초기 파라미터 시드. 모든 R에 같은 시드 사용.
    verbose: True면 진행상황 출력.
    """
    results: list[PECPoint] = []

    for R in distances:
        H = build_h2_hamiltonian(distance=float(R))
        exact = float(np.linalg.eigvalsh(hamiltonian_matrix(H))[0])
        vqe_result = run_vqe(H, seed=seed)

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
                f"err={point.error:.2e}, iter={point.n_iterations}"
            )

    return results


# 논문 Table S1의 21개 R 값
PAPER_DISTANCES: NDArray[np.float64] = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.73, 0.8, 0.9,
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 2.5, 3.0,
])