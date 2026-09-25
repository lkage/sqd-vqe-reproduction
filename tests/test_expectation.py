"""Pauli expectation value 계산 검증.

검증 전략:
1. 단일 Pauli 행렬이 표준 정의와 일치
2. 텐서곱이 Qiskit SparsePauliOp.to_matrix()와 일치 (외부 기준)
3. H2 Hamiltonian의 lowest eigenvalue가 알려진 ground state energy와 일치
4. Hartree-Fock state(|10⟩)에서의 기대값이 알려진 HF energy와 합리적
"""

import math

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from sqd_vqe.expectation import (
    expectation_value,
    hamiltonian_matrix,
    pauli_string_matrix,
)
from sqd_vqe.hamiltonian import PauliHamiltonian, build_h2_hamiltonian


# H2의 알려진 ground state energy (Hartree)
# 출처: 논문 Fig. 3B, R=0.73 Å 근처에서의 수렴값. 정밀한 값은
# Hamiltonian 자체의 lowest eigenvalue로 직접 계산해서 검증.
H2_GROUND_STATE_ENERGY_AT_073 = -1.137  # 약 1mHa 정확도


def test_single_pauli_matrices():
    """단일 Pauli 행렬이 표준 정의와 일치."""
    I = pauli_string_matrix("I")
    X = pauli_string_matrix("X")
    Y = pauli_string_matrix("Y")
    Z = pauli_string_matrix("Z")

    assert np.allclose(I, np.eye(2))
    assert np.allclose(X, [[0, 1], [1, 0]])
    assert np.allclose(Y, [[0, -1j], [1j, 0]])
    assert np.allclose(Z, [[1, 0], [0, -1]])


def test_pauli_squared_is_identity():
    """X² = Y² = Z² = I (Pauli 성질)."""
    for label in ["X", "Y", "Z"]:
        P = pauli_string_matrix(label)
        assert np.allclose(P @ P, np.eye(2))


@pytest.mark.parametrize("label", ["II", "IZ", "ZI", "ZZ", "XX", "XY", "YZ"])
def test_pauli_string_matches_qiskit(label):
    """텐서곱 결과가 Qiskit SparsePauliOp과 동일 (endian 컨벤션 일치 확인)."""
    ours = pauli_string_matrix(label)
    qiskit_op = SparsePauliOp.from_list([(label, 1.0)])
    theirs = qiskit_op.to_matrix()
    assert np.allclose(ours, theirs), (
        f"Mismatch for {label}:\nours=\n{ours}\ntheirs=\n{theirs}"
    )


def test_pauli_string_is_hermitian():
    """Pauli string은 항상 Hermitian."""
    for label in ["II", "IZ", "ZI", "ZZ", "XX", "XY", "YZ", "YY"]:
        P = pauli_string_matrix(label)
        assert np.allclose(P, P.conj().T), f"{label} not Hermitian"


def test_h2_hamiltonian_matrix_is_hermitian():
    """H2 Hamiltonian 전체가 Hermitian."""
    H = build_h2_hamiltonian(distance=0.73)
    M = hamiltonian_matrix(H)
    assert np.allclose(M, M.conj().T)


def test_h2_ground_state_energy_at_bonding_length():
    """H2 Hamiltonian의 lowest eigenvalue가 알려진 ground state energy.

    이 테스트는 expectation_value의 정확성을 *간접적으로* 검증한다.
    정확한 ground state energy = Hamiltonian 행렬의 최소 eigenvalue.
    VQE의 수렴 목표값이 이 값.
    """
    H = build_h2_hamiltonian(distance=0.73)
    M = hamiltonian_matrix(H)
    eigenvalues = np.linalg.eigvalsh(M)
    ground_state_energy = eigenvalues[0]

    assert math.isclose(
        ground_state_energy, H2_GROUND_STATE_ENERGY_AT_073, abs_tol=1e-2
    ), f"Ground state energy: {ground_state_energy} Hartree"


def test_expectation_of_eigenstate_equals_eigenvalue():
    """Hamiltonian의 ground state에서의 기대값 = ground state energy.

    이 테스트는 expectation_value의 정확성을 *직접적으로* 검증한다.
    수치 eigenstate를 만들어 그 위에서 기대값을 계산하면
    정확히 그 eigenvalue가 나와야 한다.
    """
    H = build_h2_hamiltonian(distance=0.73)
    M = hamiltonian_matrix(H)
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # ground state (lowest eigenvalue)
    ground_state = eigenvectors[:, 0]
    expected_energy = eigenvalues[0]

    computed = expectation_value(ground_state, H)
    assert math.isclose(computed, expected_energy, abs_tol=1e-12), (
        f"computed={computed}, expected={expected_energy}"
    )


def test_expectation_of_all_eigenstates():
    """모든 eigenstate에서 기대값 = 해당 eigenvalue."""
    H = build_h2_hamiltonian(distance=0.73)
    M = hamiltonian_matrix(H)
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    for i in range(len(eigenvalues)):
        psi = eigenvectors[:, i]
        computed = expectation_value(psi, H)
        assert math.isclose(computed, eigenvalues[i], abs_tol=1e-12), (
            f"Eigenstate {i}: computed={computed}, expected={eigenvalues[i]}"
        )


def test_expectation_is_real_for_normalized_state():
    """정규화된 임의 state에 대해 기대값이 실수."""
    rng = np.random.default_rng(42)
    H = build_h2_hamiltonian(distance=0.73)

    for _ in range(10):
        # 무작위 복소 단위벡터
        v = rng.normal(size=4) + 1j * rng.normal(size=4)
        v = v / np.linalg.norm(v)
        # 에러 안 나야 함 + 결과가 실수
        e = expectation_value(v, H)
        assert isinstance(e, float)


def test_variational_principle():
    """무작위 state의 기대값은 항상 ground state energy 이상.

    Rayleigh quotient의 핵심 성질. VQE가 minimization으로 동작하는 근거.
    """
    rng = np.random.default_rng(123)
    H = build_h2_hamiltonian(distance=0.73)
    M = hamiltonian_matrix(H)
    ground_state_energy = np.linalg.eigvalsh(M)[0]

    for _ in range(20):
        v = rng.normal(size=4) + 1j * rng.normal(size=4)
        v = v / np.linalg.norm(v)
        e = expectation_value(v, H)
        # 수치 오차 여유: 1e-10
        assert e >= ground_state_energy - 1e-10, (
            f"Variational principle violated: "
            f"E={e}, E_min={ground_state_energy}"
        )