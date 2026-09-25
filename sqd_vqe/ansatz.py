"""H2의 4차원 ansatz qudit state.

논문 Kim et al., Sci. Adv. 10, eado3472 (2024) 식 (5)를 따른다.
6개의 각도 파라미터 (theta_0, theta_1, theta_2, omega_0, omega_1, omega_2)
를 4차원 복소 단위 벡터로 매핑한다.

자동으로 정규화되는 파라미터화이므로 SciPy 최적화기에 제약 없이 그대로
파라미터를 넘길 수 있다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def h2_ansatz_state(params: NDArray[np.float64]) -> NDArray[np.complex128]:
    """식 (5)에 따라 6개 각도 파라미터를 4차원 ansatz state로 변환.

    params: shape (6,) 실수 배열.
        params[0] = theta_0
        params[1] = theta_1
        params[2] = theta_2
        params[3] = omega_0
        params[4] = omega_1
        params[5] = omega_2

    반환: shape (4,) complex128 배열. ‖psi‖ = 1 보장됨 (해석적으로).
    """
    if params.shape != (6,):
        raise ValueError(f"params must have shape (6,), got {params.shape}")

    theta_0, theta_1, theta_2, omega_0, omega_1, omega_2 = params

    c0 = np.cos(theta_0 / 2)
    s0 = np.sin(theta_0 / 2)
    c1 = np.cos(theta_1 / 2)
    s1 = np.sin(theta_1 / 2)
    c2 = np.cos(theta_2 / 2)
    s2 = np.sin(theta_2 / 2)

    psi = np.array([
        c0 * c1,
        c0 * s1 * np.exp(1j * omega_1),
        s0 * c2 * np.exp(1j * omega_0),
        s0 * s2 * np.exp(1j * (omega_0 + omega_2)),
    ], dtype=np.complex128)

    return psi


def num_params() -> int:
    """이 ansatz가 요구하는 파라미터 개수. (H2: 6)"""
    return 6