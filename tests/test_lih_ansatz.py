"""16D ansatz (논문 식 9) 검증.

식 (5)와 식 (9)가 같은 재귀 구조이므로, 일반화 함수가 d=4에서
기존 H2 결과와 일치하는지도 함께 확인한다.
"""

import math

import numpy as np
import pytest

from sqd_vqe.ansatz import (
    h2_ansatz_state,
    lih_ansatz_state,
    num_params_for_dim,
    qudit_ansatz_state,
)


def test_num_params_matches_paper():
    """논문 명시: 4D는 6개, 16D는 30개 (2d-2)."""
    assert num_params_for_dim(4) == 6
    assert num_params_for_dim(16) == 30


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        num_params_for_dim(6)
    with pytest.raises(ValueError):
        num_params_for_dim(1)


def test_lih_state_shape():
    psi = lih_ansatz_state(np.zeros(30))
    assert psi.shape == (16,)
    assert psi.dtype == np.complex128


def test_lih_state_at_zero_is_first_basis():
    """모든 각도가 0 → cos만 곱해짐 → |0⟩."""
    psi = lih_ansatz_state(np.zeros(30))
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 1.0
    assert np.allclose(psi, expected)


@pytest.mark.parametrize("seed", [0, 1, 42, 100, 12345])
def test_lih_state_always_normalized(seed):
    rng = np.random.default_rng(seed)
    params = rng.uniform(-2 * np.pi, 2 * np.pi, size=30)
    psi = lih_ansatz_state(params)
    assert math.isclose(np.vdot(psi, psi).real, 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("basis_index", range(16))
def test_lih_covers_all_basis_states(basis_index):
    """16개 기저 상태를 각각 만들 수 있다.

    basis_index의 트리 경로를 따라 필요한 θ만 π로 설정.
    (θ=0 → cos=1, sin=0 이므로 왼쪽 / θ=π → 오른쪽)
    """
    params = np.zeros(30)
    theta = params[:15]

    # 트리를 내려가며 basis_index에 도달하는 경로의 θ 설정
    node, size, leaf = 0, 16, 0
    while size > 1:
        half = size // 2
        if basis_index < leaf + half:
            # 왼쪽 → θ=0 (기본값 유지)
            node, size = node + 1, half
        else:
            # 오른쪽 → θ=π
            theta[node] = np.pi
            node, size, leaf = node + half, half, leaf + half

    psi = lih_ansatz_state(params)
    assert math.isclose(abs(psi[basis_index]), 1.0, abs_tol=1e-12), (
        f"basis {basis_index}: |psi| = {np.abs(psi)}"
    )


def test_generalized_matches_h2_for_dim_four():
    """일반화 함수가 d=4에서 식 (5)와 동일한 결과.

    식 (5)를 직접 하드코딩한 버전과 비교. 일반화가 옳은지 검증.
    """
    rng = np.random.default_rng(7)
    for _ in range(20):
        p = rng.uniform(-2 * np.pi, 2 * np.pi, size=6)
        t0, t1, t2, w0, w1, w2 = p

        # 식 (5) 직접 계산
        c0, s0 = np.cos(t0 / 2), np.sin(t0 / 2)
        c1, s1 = np.cos(t1 / 2), np.sin(t1 / 2)
        c2, s2 = np.cos(t2 / 2), np.sin(t2 / 2)
        expected = np.array([
            c0 * c1,
            c0 * s1 * np.exp(1j * w1),
            s0 * c2 * np.exp(1j * w0),
            s0 * s2 * np.exp(1j * (w0 + w2)),
        ], dtype=np.complex128)

        assert np.allclose(qudit_ansatz_state(p, dim=4), expected)
        assert np.allclose(h2_ansatz_state(p), expected)


def test_lih_specific_components_match_equation_9():
    """식 (9)의 몇몇 성분을 직접 계산해서 대조.

    논문 식 (9)에서 골라낸 성분:
      α_3  = c0 c1 s2 s4 · e^{i(ω2+ω4)}
      α_6  = c0 s1 s5 c7 · e^{i(ω1+ω5)}
      α_9  = s0 c8 c9 s10 · e^{i(ω0+ω10)}
      α_14 = s0 s8 s12 c14 · e^{i(ω0+ω8+ω12)}
      α_15 = s0 s8 s12 s14 · e^{i(ω0+ω8+ω12+ω14)}
    """
    rng = np.random.default_rng(99)
    p = rng.uniform(0, np.pi, size=30)
    th, om = p[:15], p[15:]
    c = lambda i: np.cos(th[i] / 2)
    s = lambda i: np.sin(th[i] / 2)

    psi = lih_ansatz_state(p)

    expected = {
        3:  c(0) * c(1) * s(2) * s(4) * np.exp(1j * (om[2] + om[4])),
        6:  c(0) * s(1) * s(5) * c(7) * np.exp(1j * (om[1] + om[5])),
        9:  s(0) * c(8) * c(9) * s(10) * np.exp(1j * (om[0] + om[10])),
        14: s(0) * s(8) * s(12) * c(14)
            * np.exp(1j * (om[0] + om[8] + om[12])),
        15: s(0) * s(8) * s(12) * s(14)
            * np.exp(1j * (om[0] + om[8] + om[12] + om[14])),
    }

    for idx, val in expected.items():
        assert np.isclose(psi[idx], val), (
            f"alpha_{idx}: got {psi[idx]}, expected {val}"
        )