"""Pauli Hamiltonian의 기대값 계산.

이 모듈은 PauliHamiltonian과 state vector를 받아 ⟨ψ|H|ψ⟩를 정확히 계산한다.
시뮬레이션의 핵심 reference implementation이며, C++ 시뮬레이터의 동일 기능과
cross-validation할 때의 기준이 된다.

Pauli string("XY" 같은 라벨)을 d×d Hermitian 행렬로 한 번 변환해 캐싱한 뒤,
이후 호출은 행렬-벡터 곱만 수행한다.

성능에 관한 주의: VQE 메인 루프는 이 모듈의 expectation_value()를 쓰지 않고,
hamiltonian_matrix()로 행렬을 한 번 만들어 재사용한다 (vqe.py 참조). LiH의
100개 항을 매번 순회하는 것보다 약 100배 빠르기 때문이다. 두 경로는
수학적으로 같고 부동소수점 합산 순서만 다르며, 실측 차이는 |E|≈7.88에서
약 10 ULP였다.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from sqd_vqe.hamiltonian import PauliHamiltonian


# 단일 큐비트 Pauli matrices. Y만 허수 성분을 가진다 — 이 사실이 분자
# Hamiltonian이 실수 행렬이 되는 이유와 직결된다 (Y가 짝수 개면 허수부가
# 상쇄된다).
_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


@lru_cache(maxsize=256)
def pauli_string_matrix(label: str) -> NDArray[np.complex128]:
    """Pauli string 라벨(예: "IZ", "XX")을 d×d Hermitian 행렬로 변환.

    Qiskit의 little-endian 컨벤션을 따른다: 라벨의 가장 왼쪽 글자가
    가장 높은 큐비트 인덱스에 대응한다. 즉 텐서곱을 왼쪽에서 오른쪽으로
    누적하면 Qiskit의 SparsePauliOp.to_matrix()와 비트 단위로 같아진다.
    (tools/dump_pauli.py가 7개 라벨에 대해 이를 확인했다.)

    캐싱: 같은 라벨은 한 번만 계산한다. R 스윕에서 같은 Pauli string이
    거리마다 반복 등장하므로 효과가 있다.
    """
    if not label:
        raise ValueError("Empty Pauli string")
    if any(ch not in _PAULI for ch in label):
        raise ValueError(f"Invalid Pauli label: {label!r}")

    # little-endian: 라벨 가장 왼쪽이 highest qubit.
    # 텐서곱 순서는 left-to-right = highest-to-lowest qubit.
    # 첫 글자로 시작해 오른쪽으로 가며 kron을 누적한다.
    result = _PAULI[label[0]]
    for ch in label[1:]:
        result = np.kron(result, _PAULI[ch])
    return result


def hamiltonian_matrix(H: PauliHamiltonian) -> NDArray[np.complex128]:
    """PauliHamiltonian의 가중합을 d×d 행렬로 빌드.

    두 가지 용도가 있다.
      1. 검증/디버깅: np.linalg.eigvalsh()로 정확한 최소 고윳값 계산
      2. VQE 메인 루프: 행렬을 한 번 만들어 재사용 (vqe.py)

    항을 더하는 순서는 H.terms의 순서를 그대로 따른다. 부동소수점 덧셈은
    결합법칙이 성립하지 않으므로 순서가 결과를 ULP 수준에서 바꾼다. C++
    쪽과 비교할 때 양쪽이 같은 순서로 누적해야 하는 이유다.
    """
    if not H.terms:
        raise ValueError("Empty Hamiltonian")
    # 첫 항의 라벨 길이로 차원을 정한다. 라벨 길이가 섞여 있으면 아래
    # 덧셈에서 shape 불일치로 바로 터진다 — 조용히 틀리는 것보다 낫다.
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

    H가 Hermitian이고 state가 정규화되었으면 결과가 실수임이 수학적으로
    보장된다. 따라서 허수부가 임계를 넘으면 입력이 잘못된 것이고, 조용히
    실수부만 취하는 대신 즉시 에러를 낸다. 수치 오차로 생기는 미세한
    허수부(1e-9 미만)는 버린다.

    항별로 순회하며 Σ Wₘ⟨Pₘ⟩를 누적한다. 행렬을 만들지 않으므로 한 번만
    호출할 때는 이쪽이 싸지만, VQE처럼 수천 번 부를 때는 행렬 재사용이
    훨씬 빠르다.
    """
    energy = 0.0 + 0.0j
    for label, weight in H.terms:
        P = pauli_string_matrix(label)
        # state.conj()가 왼쪽에 와야 한다. 켤레를 반대쪽에 걸면 분자
        # Hamiltonian(실수 대칭)에서는 같은 값이 나와 버그가 드러나지 않지만,
        # Y가 홀수인 항이 있으면 값이 갈린다.
        energy += weight * (state.conj() @ P @ state)

    if abs(energy.imag) > 1e-9:
        raise ValueError(
            f"Non-real expectation value: {energy}. "
            f"State or Hamiltonian may not be valid."
        )
    return float(energy.real)