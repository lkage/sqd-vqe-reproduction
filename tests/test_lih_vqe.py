"""LiH 30차원 VQE 수렴 검증.

단일 실행은 성공률 ~50%이므로 multi-start를 기준으로 한다.
전체 R 스윕은 examples/figures/lih_potential_curve.py에서 수행.
"""

import numpy as np
import pytest

from sqd_vqe.ansatz import LIH_NUM_PARAMS, lih_ansatz_state
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import get_lih_hamiltonian
from sqd_vqe.vqe import run_vqe_multistart


CHEMICAL_ACCURACY = 1.6e-3


@pytest.fixture(scope="module")
def lih_at_bonding_length():
    """R=1.55 Å의 LiH Hamiltonian + 정확한 ground state energy."""
    H = get_lih_hamiltonian(distance=1.55)
    exact = float(np.linalg.eigvalsh(hamiltonian_matrix(H))[0])
    return H, exact


def test_lih_hamiltonian_is_cached_and_stable():
    """캐시에서 읽은 Hamiltonian이 호출마다 비트 단위로 동일.

    Qiskit 직접 생성은 프로세스마다 계수가 최대 ~152 ULP 달라진다.
    캐시가 이를 차단하는지 확인.
    """
    H1 = get_lih_hamiltonian(distance=1.55)
    H2 = get_lih_hamiltonian(distance=1.55)
    assert H1.terms == H2.terms


def test_lih_vqe_converges_with_multistart(lih_at_bonding_length):
    """multi-start 5회로 R=1.55 Å에서 chemical accuracy 달성."""
    H, exact = lih_at_bonding_length
    result = run_vqe_multistart(
        H,
        ansatz=lih_ansatz_state,
        n_restarts=5,
        seed=42,
        n_params=LIH_NUM_PARAMS,
        max_iter=3000,
    )

    error = abs(result.energy - exact)
    assert error < CHEMICAL_ACCURACY, (
        f"VQE energy: {result.energy:.6f}, exact: {exact:.6f}, "
        f"error: {error:.2e} Hartree"
    )


def test_lih_vqe_respects_variational_principle(lih_at_bonding_length):
    """결과가 정확한 ground state energy보다 작을 수 없다."""
    H, exact = lih_at_bonding_length
    result = run_vqe_multistart(
        H, ansatz=lih_ansatz_state, n_restarts=3,
        seed=42, n_params=LIH_NUM_PARAMS, max_iter=3000,
    )
    assert result.energy >= exact - 1e-9


def test_lih_vqe_is_reproducible(lih_at_bonding_length):
    """같은 seed → 같은 결과 (캐시된 Hamiltonian 덕분)."""
    H, _ = lih_at_bonding_length
    r1 = run_vqe_multistart(
        H, ansatz=lih_ansatz_state, n_restarts=3,
        seed=42, n_params=LIH_NUM_PARAMS, max_iter=3000,
    )
    r2 = run_vqe_multistart(
        H, ansatz=lih_ansatz_state, n_restarts=3,
        seed=42, n_params=LIH_NUM_PARAMS, max_iter=3000,
    )
    assert r1.energy == r2.energy
    assert np.array_equal(r1.params, r2.params)


def test_lih_multistart_reports_best_seed(lih_at_bonding_length):
    """multi-start가 어느 restart에서 최선을 얻었는지 기록한다."""
    H, _ = lih_at_bonding_length
    result = run_vqe_multistart(
        H, ansatz=lih_ansatz_state, n_restarts=3,
        seed=42, n_params=LIH_NUM_PARAMS, max_iter=3000,
    )
    assert result.n_restarts == 3
    assert result.best_restart_seed in (42, 43, 44)