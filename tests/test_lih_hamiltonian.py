"""LiH Hamiltonian이 논문 Table S2와 일치하는지 검증.

Table S2: LiH-qiskit, R = 1.55 Å, 4-qubit, 100 Pauli strings.
"""

import math

import pytest

from sqd_vqe.hamiltonian import build_lih_hamiltonian


# 논문 Table S2 전체 (R = 1.55 Å, 100개 중 대표 샘플)
# 각 commuting group에서 골고루 추출
PAPER_S2 = {
    "IIII": -6.996701,
    "IIIZ": -0.092811,
    "IZXZ": -0.012782,
    "IZXI": -0.012782,
    "IIXZ":  0.019380,
    "IIXI":  0.019380,
    "IZII":  0.092811,
    "ZZIZ":  0.056269,
    "ZZII": -0.211881,
    "ZZXZ": -0.009002,
    "ZIII": -0.364913,
    "IZIZ": -0.122697,
    "ZIIZ":  0.113930,
    "IIZI":  0.364916,
    "IZZI":  0.113930,
    "ZIZI": -0.113827,
    "IIZZ": -0.211881,
    "ZZZZ":  0.084595,
    "XXXX":  0.030852,
    "YYYY":  0.030852,
    "XXYY": -0.030852,
    "YYXX": -0.030852,
}

ATOL = 1e-4


def test_lih_has_four_qubits():
    """모든 Pauli string이 4글자 (4-qubit)."""
    H = build_lih_hamiltonian(distance=1.55)
    lengths = {len(label) for label, _ in H.terms}
    assert lengths == {4}, f"Pauli string 길이가 섞임: {lengths}"


def test_lih_has_100_pauli_terms():
    """논문과 동일하게 100개 Pauli string."""
    H = build_lih_hamiltonian(distance=1.55)
    assert len(H.terms) == 100


def test_lih_coefficients_match_paper_table_s2():
    """Table S2 대표 계수가 일치."""
    H = build_lih_hamiltonian(distance=1.55)
    coeffs = H.as_dict()

    mismatches = []
    for pauli, expected in PAPER_S2.items():
        actual = coeffs.get(pauli)
        if actual is None:
            mismatches.append((pauli, "missing", expected))
            continue
        if not math.isclose(actual, expected, abs_tol=ATOL):
            mismatches.append((pauli, actual, expected))

    assert not mismatches, (
        "Table S2 (R=1.55 Å) 불일치:\n"
        + "\n".join(
            f"  {p}: qiskit={a}, paper={e}" for p, a, e in mismatches
        )
    )


def test_lih_all_paper_strings_present():
    """논문에 있는 Pauli string이 전부 존재."""
    H = build_lih_hamiltonian(distance=1.55)
    qiskit_strings = set(H.as_dict().keys())
    missing = set(PAPER_S2.keys()) - qiskit_strings
    assert not missing, f"논문에 있는데 qiskit에 없는 항: {missing}"