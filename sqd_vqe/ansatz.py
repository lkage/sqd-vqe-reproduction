"""d차원 ansatz qudit state.

논문 Kim et al., Sci. Adv. 10, eado3472 (2024) 식 (5)(4D)와 식 (9)(16D)를
하나의 일반화된 함수로 구현한다. d = 2^k 차원에 대해 2d-2개 각도 파라미터를
사용한다.

구조 (두 식이 동일한 이진 트리):
  - 각 내부 노드에서 θ로 cos/sin 분기 (진폭)
  - sin 방향으로 갈 때만 그 노드의 ω를 phase에 누적
  - 노드 인덱싱: 크기 m 서브트리가 k에서 시작하면
    루트=k, 왼쪽 서브트리=k+1, 오른쪽 서브트리=k+m/2

d=16일 때 트리는 이렇게 생겼다 (숫자는 θ/ω 인덱스):

                        θ0
              ┌─────────┴─────────┐
             θ1                   θ8
        ┌─────┴─────┐       ┌─────┴─────┐
       θ2           θ5     θ9          θ12
     ┌──┴──┐     ┌──┴──┐ ┌──┴──┐     ┌──┴──┐
    θ3    θ4    θ6    θ7 θ10  θ11   θ13   θ14

왼쪽 서브트리가 루트 바로 다음 번호를 쓰고, 오른쪽 서브트리는 왼쪽이 다
쓰고 난 다음 번호부터 시작한다. 그래서 루트 0의 오른쪽이 8이 된다 —
왼쪽 서브트리(루트 1)가 1~7의 일곱 개를 쓰기 때문이다.

이 파라미터화는 해석적으로 정규화되므로 COBYLA에 제약 없이 전달 가능하다.
삼각함수 항등식이 각 분기마다 cos²+sin²=1을 보장하고, 그것이 트리 전체에서
곱해지며 ‖psi‖=1로 수렴한다. 옵티마이저에게 정규화 제약을 따로 걸 필요가
없다는 뜻이고, COBYLA가 제약 없는 문제를 푸는 편이 훨씬 안정적이다.

참고: 논문 식 (9)의 α₆, α₁₃, α₁₄에서 ω 아래첨자 n이 누락된 표기 오류가
있으나, 같은 파라미터를 가리킨다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def num_params_for_dim(dim: int) -> int:
    """d차원 ansatz가 요구하는 파라미터 개수 (2d-2).

    d=4 → 6 (H2, 식 5), d=16 → 30 (LiH, 식 9).

    왜 2d-2인가: d차원 복소 단위 벡터는 복소수 d개이므로 실수 2d개지만,
    정규화 조건이 1개, 전역 위상이 1개를 묶어 자유도가 2d-2로 줄어든다.
    논문 식 (5)/(9)의 파라미터 개수와 정확히 일치한다.
    """
    # 2의 거듭제곱이 아니면 이진 트리가 균형 잡히지 않아 알고리즘이 성립하지
    # 않는다. (dim & (dim-1)) == 0 은 비트가 하나뿐인지 보는 표준 관용구다.
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

    # 내부 노드 개수 = leaf 개수 - 1. θ와 ω가 각각 이만큼 있다.
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
        amplitude: 여기까지 곱해온 실수 진폭
        phase: 여기까지 누적한 위상
        """
        # leaf에 도달. 지금까지 곱해온 진폭과 누적 위상으로 성분을 확정한다.
        if size == 1:
            psi[leaf] = amplitude * np.exp(1j * phase)
            return

        half = size // 2
        # θ/2를 쓰는 것은 Bloch sphere 관례와 같다. θ가 0에서 π로 갈 때
        # 진폭이 cos 쪽에서 sin 쪽으로 완전히 넘어간다.
        c = np.cos(theta[node] / 2)
        s = np.sin(theta[node] / 2)

        # 왼쪽(cos): 위상은 그대로. 왼쪽 서브트리는 바로 다음 노드 번호부터
        # 시작하므로 node+1.
        descend(node + 1, half, leaf, amplitude * c, phase)

        # 오른쪽(sin): 이 노드의 omega를 위상에 누적한다. 여기가 식 (5)/(9)의
        # 핵심 — 위상이 더해지는 것은 sin 방향 분기에서만이다.
        # 오른쪽 서브트리의 시작 번호가 node+half인 이유: 왼쪽 서브트리가
        # 자기 루트 1개와 그 아래 half-1개, 합쳐 half-1개의 내부 노드를
        # 쓰므로 node+1부터 node+half-1까지가 소진된다.
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


# 편의 상수. 호출부에서 num_params_for_dim(4)를 반복해 쓰지 않도록 둔다.
H2_NUM_PARAMS = num_params_for_dim(4)    # 6
LIH_NUM_PARAMS = num_params_for_dim(16)  # 30