"""d차원 ansatz qudit state.

논문 Kim et al., Sci. Adv. 10, eado3472 (2024) 식 (5)(4D)와 식 (9)(16D)를
하나의 일반화된 함수로 구현한다. 두 식은 동일한 이진 트리 구조이며,
d = 2^k 차원에 대해 2d-2개 각도 파라미터를 사용한다.

구조:
  - 각 내부 노드에서 θ로 cos/sin 분기 (진폭)
  - sin 방향으로 갈 때만 그 노드의 ω를 phase에 누적
  - 노드 인덱싱: 크기 m 서브트리가 k에서 시작하면
    루트=k, 왼쪽 서브트리=k+1, 오른쪽 서브트리=k+m/2

이 파라미터화는 해석적으로 정규화되므로 COBYLA에 제약 없이 전달 가능.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def num_params_for_dim(dim: int) -> int:
    """d차원 ansatz가 요구하는 파라미터 개수 (2d-2).

    d=4 → 6 (H2, 식 5), d=16 → 30 (LiH, 식 9).
    """
    if dim < 2 or (dim & (dim - 1)) != 0:
        raise ValueError(f"dim must be a power of 2 and >= 2, got {dim}")
    return 2 * dim - 2


def qudit_ansatz_state(
    params: NDArray[np.float64],
    dim: int,
) -> NDArray[np.complex128]:
    """2d-2개 각도 파라미터를 d차원 복소 단위 벡터로 변환.

    params: shape (2*dim-2,) 실수 배열.
        params[:dim-1]  = theta_0 ... theta_{dim-2}  (elevation)
        params[dim-1:]  = omega_0 ... omega_{dim-2}  (azimuthal)

    반환: shape (dim,) complex128. ‖psi‖ = 1 (해석적으로 보장).
    """
    expected = num_params_for_dim(dim)
    if params.shape != (expected,):
        raise ValueError(
            f"params must have shape ({expected},) for dim={dim}, "
            f"got {params.shape}"
        )

    n_nodes = dim - 1
    theta = params[:n_nodes]
    omega = params[n_nodes:]

    psi = np.zeros(dim, dtype=np.complex128)

    def descend(node: int, size: int, leaf: int,
                amplitude: float, phase: float) -> None:
        """트리를 내려가며 각 leaf에 진폭과 위상을 기록.

        node: theta/omega 인덱스
        size: 이 서브트리의 leaf 개수
        leaf: 이 서브트리의 첫 leaf 인덱스
        """
        if size == 1:
            psi[leaf] = amplitude * np.exp(1j * phase)
            return

        half = size // 2
        c = np.cos(theta[node] / 2)
        s = np.sin(theta[node] / 2)

        # 왼쪽(cos): phase 변화 없음
        descend(node + 1, half, leaf, amplitude * c, phase)
        # 오른쪽(sin): 이 노드의 omega를 phase에 누적
        descend(node + half, half, leaf + half,
                amplitude * s, phase + omega[node])

    descend(0, dim, 0, 1.0, 0.0)
    return psi


def h2_ansatz_state(params: NDArray[np.float64]) -> NDArray[np.complex128]:
    """H2용 4차원 ansatz (논문 식 5). 6개 파라미터."""
    return qudit_ansatz_state(params, dim=4)


def lih_ansatz_state(params: NDArray[np.float64]) -> NDArray[np.complex128]:
    """LiH용 16차원 ansatz (논문 식 9). 30개 파라미터."""
    return qudit_ansatz_state(params, dim=16)


def num_params() -> int:
    """(하위 호환) H2 ansatz의 파라미터 개수."""
    return num_params_for_dim(4)