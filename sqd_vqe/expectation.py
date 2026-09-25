"""Pauli Hamiltonian의 기대값 계산.

이 모듈은 PauliHamiltonian과 state vector를 받아 ⟨ψ|H|ψ⟩를 정확히 계산한다.
시뮬레이션의 핵심 reference implementation. 7주차에 C++ 시뮬레이터의 동일
기능과 cross-validation할 기준이 된다.

Pauli string("XY"같은 라벨)을 d×d Hermitian 행렬로 한 번 변환해서 캐싱한 뒤,
이후 호출은 행렬-벡터 곱만 수행한다.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from sqd_vqe.hamiltonian import PauliHamiltonian


# 단일 큐비트 Pauli matrices
_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


@lru_cache(maxsize=256)
def pauli_string_matrix(label: str) -> NDArray[np.complex128]:
    """Pauli string 라벨(예: "IZ", "XX")을 d×d Hermitian 행렬로 변환.

    Qiskit의 little-endian 컨벤션을 따른다: 라벨의 가장 왼쪽 글자가
    가장 높은 큐비트 인덱스에 대응. Qiskit의 SparsePauliOp.to_matrix()와
    동일한 결과를 낸다.

    캐싱: 같은 라벨은 한 번만 계산. VQE iteration 중 반복 호출에 효율적.
    """
    if not label:
        raise ValueError("Empty Pauli string")
    if any(ch not in _PAULI for ch in label):
        raise ValueError(f"Invalid Pauli label: {label!r}")

    # little-endian: 라벨 가장 왼쪽이 highest qubit
    # 텐서곱 순서는 left-to-right = highest-to-lowest qubit
    result = _PAULI[label[0]]
    for ch in label[1:]:
        result = np.kron(result, _PAULI[ch])
    return result


def hamiltonian_matrix(H: PauliHamiltonian) -> NDArray[np.complex128]:
    """PauliHamiltonian의 가중합을 d×d 행렬로 빌드.

    주로 검증/디버깅 용. 정확한 lowest eigenvalue 계산(np.linalg.eigvalsh)과
    같은 작업에 사용. VQE 메인 루프는 expectation_value()를 직접 호출.
    """
    if not H.terms:
        raise ValueError("Empty Hamiltonian")
    d = pauli_string_matrix(H.terms[0][0]).shape[0]
    M = np.zeros((d, d), dtype=np.complex128)
    for label, weight in H.terms:
        M += weight * pauli_string_matrix(label)
    return M


def expectation_value(
    state: NDArray[np.complex128],
    H: PauliHamiltonian,
) -> float:
    """⟨state|H|state⟩를 계산. 결과는 실수.

    H가 Hermitian이고 state가 정규화되었으면 결과는 실수임이 보장된다.
    수치 오차로 발생하는 미세한 허수부는 버린다 (단, 임계 초과 시 에러).
    """
    energy = 0.0 + 0.0j
    for label, weight in H.terms:
        P = pauli_string_matrix(label)
        energy += weight * (state.conj() @ P @ state)

    if abs(energy.imag) > 1e-9:
        raise ValueError(
            f"Non-real expectation value: {energy}. "
            f"State or Hamiltonian may not be valid."
        )
    return float(energy.real)