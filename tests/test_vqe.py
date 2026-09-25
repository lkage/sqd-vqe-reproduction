"""VQE COBYLA 최적화의 수렴 검증.

논문 Fig. 3 재현을 위한 핵심 단계. 단일 R에서 ground state energy로
수렴하는지 확인.
"""

import math

import numpy as np
import pytest

from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import build_h2_hamiltonian
from sqd_vqe.vqe import run_vqe


# Chemical accuracy: 1 kcal/mol ≈ 1.6 mHa
# 논문도 이 기준으로 정확도 평가 (Fig. 3B의 "Chemical Accuracy" 라인)
CHEMICAL_ACCURACY = 1.6e-3  # Hartree


@pytest.fixture
def h2_at_bonding_length():
    """R=0.73 Å에서의 H2 Hamiltonian + 정확한 ground state energy."""
    H = build_h2_hamiltonian(distance=0.73)
    M = hamiltonian_matrix(H)
    exact_energy = float(np.linalg.eigvalsh(M)[0])
    return H, exact_energy


def test_vqe_converges_to_ground_state(h2_at_bonding_length):
    """VQE가 R=0.73 Å에서 chemical accuracy 이내로 수렴.

    이게 통과해야 4주차 핵심 목표 달성. Fig. 3A의 정성적 거동 재현.
    """
    H, exact_energy = h2_at_bonding_length
    result = run_vqe(H, seed=42)

    error = abs(result.energy - exact_energy)
    assert error < CHEMICAL_ACCURACY, (
        f"VQE energy: {result.energy:.6f}, "
        f"exact: {exact_energy:.6f}, "
        f"error: {error:.6f} Hartree "
        f"(chemical accuracy = {CHEMICAL_ACCURACY})"
    )


def test_vqe_respects_variational_principle(h2_at_bonding_length):
    """최적화 결과가 정확한 ground state energy보다 작을 수 없다."""
    H, exact_energy = h2_at_bonding_length
    result = run_vqe(H, seed=42)

    # 수치 오차 여유 1e-9
    assert result.energy >= exact_energy - 1e-9


def test_vqe_is_reproducible(h2_at_bonding_length):
    """같은 seed → 같은 결과."""
    H, _ = h2_at_bonding_length
    r1 = run_vqe(H, seed=42)
    r2 = run_vqe(H, seed=42)
    assert math.isclose(r1.energy, r2.energy, abs_tol=1e-12)
    assert np.allclose(r1.params, r2.params)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 100])
def test_vqe_robust_to_initial_params(h2_at_bonding_length, seed):
    """여러 무작위 초기값에서 모두 chemical accuracy 이내로 수렴.

    논문은 H2에 평균 48 iteration으로 수렴함을 보고. 우리는 max_iter 500.
    여러 seed에서 안정적으로 수렴해야 알고리즘이 견고한 것.
    """
    H, exact_energy = h2_at_bonding_length
    result = run_vqe(H, seed=seed)

    error = abs(result.energy - exact_energy)
    assert error < CHEMICAL_ACCURACY, (
        f"seed={seed}: error={error:.6f} Hartree, "
        f"iterations={result.n_iterations}"
    )


def test_vqe_history_consistent(h2_at_bonding_length):
    """history가 결과와 일관성 있게 기록됨.

    COBYLA는 trust region 탐색 과정에서 best가 아닌 점도 평가하므로
    energy_history[-1] != result.energy일 수 있다.
    대신 result.energy는 정의상 평가된 모든 값 중 최소여야 한다.
    """
    H, _ = h2_at_bonding_length
    result = run_vqe(H, seed=42)

    assert len(result.energy_history) == result.n_iterations
    assert len(result.param_history) == result.n_iterations

    # result.energy는 평가된 모든 에너지 중 최소
    best_in_history = min(result.energy_history)
    assert math.isclose(best_in_history, result.energy, abs_tol=1e-9), (
        f"result.energy={result.energy}, "
        f"min(history)={best_in_history}"
    )


def test_vqe_iteration_count_reasonable(h2_at_bonding_length):
    """수렴 iteration 수가 합리적 범위.

    논문: H2 평균 48 iteration. 우리 구현이 한 자릿수~수백 정도면 정상.
    수천 iteration 가면 뭔가 잘못된 것.
    """
    H, _ = h2_at_bonding_length
    result = run_vqe(H, seed=42)

    assert 5 < result.n_iterations < 500, (
        f"Iteration count: {result.n_iterations} "
        f"(논문 H2 평균: 48)"
    )