"""H2 Hamiltonian이 논문 Table S1과 일치하는지 검증.

논문: Kim et al., Sci. Adv. 10, eado3472 (2024), Table S1.
검증 범위: R = 0.73 Å (bonding length) + 여러 R 값에서 spot-check.
"""

import math

import pytest

from sqd_vqe.hamiltonian import build_h2_hamiltonian


# 논문 Table S1, R = 0.73 Å (bonding length)
PAPER_S1_073 = {
    "II": -0.326386,
    "IZ":  0.401061,
    "ZI": -0.401061,
    "ZZ": -0.011314,
    "XX":  0.180653,
}

# Table S1의 일부 R 값. (논문 소수 6자리)
PAPER_S1_SPOT_CHECK = {
    0.1: {"II":  4.756650, "XX": 0.156170, "IZ":  1.027315,
          "ZI": -1.027315, "ZZ": -0.013867},
    0.5: {"II":  0.110647, "XX": 0.168870, "IZ":  0.583080,
          "ZI": -0.583080, "ZZ": -0.012516},
    1.0: {"II": -0.540066, "XX": 0.196791, "IZ":  0.267529,
          "ZI": -0.267529, "ZZ": -0.009015},
    2.0: {"II": -0.663968, "XX": 0.259138, "IZ":  0.060628,
          "ZI": -0.060628, "ZZ": -0.001431},
    3.0: {"II": -0.633651, "XX": 0.299212, "IZ":  0.011235,
          "ZI": -0.011235, "ZZ": -0.000074},
}

# 논문이 소수 6자리만 제공 → 절대허용오차도 그 수준
ATOL = 1e-4


def test_h2_pauli_strings_match_paper():
    """Pauli string 집합이 논문 Table S1과 정확히 일치한다."""
    H = build_h2_hamiltonian(distance=0.73)
    qiskit_strings = set(H.as_dict().keys())
    paper_strings = set(PAPER_S1_073.keys())

    missing = paper_strings - qiskit_strings
    extra = qiskit_strings - paper_strings

    assert not missing, f"논문에 있는데 qiskit에 없는 항: {missing}"
    assert not extra, f"논문에 없는데 qiskit에 있는 항: {extra}"


def test_h2_pauli_coefficients_at_bonding_length():
    """R = 0.73 Å에서 모든 Pauli 계수가 논문 값과 일치한다."""
    H = build_h2_hamiltonian(distance=0.73)
    coeffs = H.as_dict()

    mismatches = []
    for pauli, expected in PAPER_S1_073.items():
        actual = coeffs[pauli]
        if not math.isclose(actual, expected, abs_tol=ATOL):
            mismatches.append((pauli, actual, expected))

    assert not mismatches, (
        "Table S1 (R=0.73 Å) 불일치:\n"
        + "\n".join(
            f"  {p}: qiskit={a:+.6f}, paper={e:+.6f}, "
            f"diff={a-e:+.2e}"
            for p, a, e in mismatches
        )
    )


@pytest.mark.parametrize("R, expected_coeffs", PAPER_S1_SPOT_CHECK.items())
def test_h2_pauli_coefficients_spot_check(R, expected_coeffs):
    """여러 R 값에서 Table S1 일치성 spot-check.

    bonding length 외의 R에서도 동일 메커니즘이 동작하는지 확인.
    이게 통과해야 Fig. 3 (potential energy curve) 재현 시
    매 R마다 hamiltonian을 신뢰할 수 있음.
    """
    H = build_h2_hamiltonian(distance=R)
    coeffs = H.as_dict()

    mismatches = []
    for pauli, expected in expected_coeffs.items():
        actual = coeffs[pauli]
        if not math.isclose(actual, expected, abs_tol=ATOL):
            mismatches.append((pauli, actual, expected))

    assert not mismatches, (
        f"Table S1 (R={R} Å) 불일치:\n"
        + "\n".join(
            f"  {p}: qiskit={a:+.6f}, paper={e:+.6f}, "
            f"diff={a-e:+.2e}"
            for p, a, e in mismatches
        )
    )


def test_nuclear_repulsion_is_inverse_distance():
    """Nuclear repulsion 값이 1/R (in Bohr) 공식과 일치한다.

    H2는 핵 한 쌍만 있으므로 NR = Z_H * Z_H / R = 1/R.
    이 검증은 Qiskit이 단위 변환(Å → Bohr)을 제대로 했는지 확인.
    """
    ANGSTROM_TO_BOHR = 1.8897261254535

    for R_angstrom in [0.5, 0.73, 1.0, 2.0]:
        H = build_h2_hamiltonian(distance=R_angstrom)
        R_bohr = R_angstrom * ANGSTROM_TO_BOHR
        expected_nr = 1.0 / R_bohr
        assert math.isclose(
            H.nuclear_repulsion, expected_nr, rel_tol=1e-6
        ), f"R={R_angstrom}: NR={H.nuclear_repulsion}, expected={expected_nr}"