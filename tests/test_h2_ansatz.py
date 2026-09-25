"""H2 4차원 ansatz state 검증 (논문 식 5).

식 (5)의 핵심 성질:
1. 항상 정규화 (단위 벡터)
2. 특정 파라미터 값에서 알려진 basis state 복원
3. 4차원 공간 전체를 커버 (단일 basis state로 갇히지 않음)
"""

import math

import numpy as np
import pytest

from sqd_vqe.ansatz import H2_NUM_PARAMS, h2_ansatz_state


def test_num_params_is_six():
    """논문 명시: 4차원 ansatz는 6개 각도 파라미터 (2d-2)."""
    assert H2_NUM_PARAMS == 6


def test_state_shape():
    psi = h2_ansatz_state(np.zeros(6))
    assert psi.shape == (4,)
    assert psi.dtype == np.complex128


def test_state_is_normalized_at_zero():
    """모든 각도가 0이면 cos(0)=1, sin(0)=0 → (1,0,0,0)."""
    psi = h2_ansatz_state(np.zeros(6))
    expected = np.array([1, 0, 0, 0], dtype=np.complex128)
    assert np.allclose(psi, expected)


def test_state_normalized_at_pi():
    """theta_0 = pi → 위쪽 두 성분이 0, 아래쪽만 남음."""
    params = np.array([np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
    psi = h2_ansatz_state(params)
    # c0=0, s0=1이고 theta_2=0이므로 세번째 성분만 1
    expected = np.array([0, 0, 1, 0], dtype=np.complex128)
    assert np.allclose(psi, expected)


@pytest.mark.parametrize("seed", [0, 1, 42, 100, 12345])
def test_state_always_normalized(seed):
    """무작위 파라미터에서도 ‖psi‖ = 1."""
    rng = np.random.default_rng(seed)
    params = rng.uniform(-2 * np.pi, 2 * np.pi, size=6)
    psi = h2_ansatz_state(params)
    assert math.isclose(np.vdot(psi, psi).real, 1.0, abs_tol=1e-12)


def test_state_can_be_complex():
    """비자명한 phase에서 실제로 복소수 값을 가진다."""
    params = np.array([np.pi/2, np.pi/2, np.pi/2, 0.3, 0.5, 0.7])
    psi = h2_ansatz_state(params)
    assert np.any(np.abs(psi.imag) > 1e-10)


def test_state_covers_basis_vectors():
    """4개 기저 상태 각각을 만들 수 있다.

    이게 통과해야 ansatz가 4차원 공간 전체에 도달 가능하다.
    """
    # |0> = (1,0,0,0): 모두 0
    assert np.allclose(h2_ansatz_state(np.zeros(6)), [1, 0, 0, 0])

    # |1> = (0,1,0,0): theta_1 = pi
    psi1 = h2_ansatz_state(np.array([0, np.pi, 0, 0, 0, 0]))
    assert np.allclose(np.abs(psi1), [0, 1, 0, 0])

    # |2> = (0,0,1,0): theta_0 = pi, theta_2 = 0
    psi2 = h2_ansatz_state(np.array([np.pi, 0, 0, 0, 0, 0]))
    assert np.allclose(np.abs(psi2), [0, 0, 1, 0])

    # |3> = (0,0,0,1): theta_0 = pi, theta_2 = pi
    psi3 = h2_ansatz_state(np.array([np.pi, 0, np.pi, 0, 0, 0]))
    assert np.allclose(np.abs(psi3), [0, 0, 0, 1])


def test_rejects_wrong_shape():
    """잘못된 입력 shape는 즉시 에러."""
    with pytest.raises(ValueError):
        h2_ansatz_state(np.zeros(5))
    with pytest.raises(ValueError):
        h2_ansatz_state(np.zeros(7))