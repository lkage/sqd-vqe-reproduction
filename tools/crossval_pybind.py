"""pybind11 모듈을 Python의 두 평가 경로와 대조 검증한다.

각 열이 비교하는 쌍:

    termwise   C++ 항별 순회   vs  파일의 기준값 (Python expectation_value)
    dense      C++ 조밀 행렬   vs  파일의 기준값 (Python expectation_value)
    vs_matvec  C++ 조밀 행렬   vs  Python `state.conj() @ M @ state`

세 번째 열이 VQE 루프에서 실제로 의미를 갖는 쌍이다. run_vqe는
hamiltonian_matrix()로 M을 한 번 만들어 수축시키므로, C++ 커널이 대체하는
것은 바로 그 식이다. 앞의 두 열은 파일 기준값이 항별 순회로 만들어졌으니
termwise끼리 맞는지 보는 확인용이다.

hamiltonian_matrix()는 PauliHamiltonian을 받는데, 그 객체는 분자 행에만
존재한다. ycomplex/random 행의 행렬은 C++이 읽는 바로 그 텍스트 파일에서
여기서 직접 조립한다. 이 대체를 정당화하기 위해, 분자 행마다 직접 조립한
행렬을 hamiltonian_matrix() 결과와 대조해 **완전 일치**를 요구한다.

항 순서가 중요하다. 텍스트 파일은 dump_crossval_states.py가 쓴 순서를
따르고, PauliHamiltonian.terms는 _to_pauli_hamiltonian이 라벨 기준으로
정렬한 순서다. 둘이 어긋나면 두 조밀 행렬이 ULP 수준에서 달라지고,
vs_matvec 열은 엉뚱한 것을 재게 된다.

사용법:
    PYTHONPATH=<cpp_repo>/build-py/python uv run python tools/crossval_pybind.py
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

import qudit_simulator as qs
from sqd_vqe.expectation import expectation_value, hamiltonian_matrix
from sqd_vqe.hamiltonian import get_h2_hamiltonian, get_lih_hamiltonian

STATES_PATH = Path("results/crossval_states.txt")
HAM_DIR = Path("results/crossval_hamiltonians")
TOLERANCE = 1e-12

PATHS = ("termwise", "dense", "vs_matvec")

# 진단력이 높은 순서로 둔다. ycomplex는 Y가 홀수인 항을 포함하므로 켤레
# 방향이 뒤집히면 즉시 드러난다. random은 일반적인 복소 진폭 처리를
# 확인한다. 분자 행은 실수 ground state라 주로 정밀도만 본다.
GROUP_ORDER = [
    "ycomplex_d4", "ycomplex_d16",
    "random_d4", "random_d16",
    "h2", "lih",
]

PAULI_2X2 = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

# 분자 행은 PauliHamiltonian으로 되짚을 수 있지만, 합성 Hamiltonian은
# 텍스트로만 존재하므로 불가능하다.
REAL_HAMILTONIAN_LOADERS = {
    "h2": get_h2_hamiltonian,
    "lih": get_lih_hamiltonian,
}


def read_terms(molecule: str, r_text: str, expected_dim: int):
    """TERMS 파일을 (계수, 라벨) 쌍으로 파싱한다. 파일에 적힌 순서 그대로."""
    path = HAM_DIR / f"{molecule}_R{r_text}.txt"
    tokens = path.read_text().split()

    if tokens[0] != "TERMS":
        raise ValueError(f"{path}: expected TERMS header, got {tokens[0]!r}")
    n_terms = int(tokens[1])

    terms = []
    for i in range(n_terms):
        coeff = float(tokens[2 + 2 * i])
        label = tokens[3 + 2 * i]
        if 2 ** len(label) != expected_dim:
            raise ValueError(
                f"{path}: label {label!r} implies dimension "
                f"{2 ** len(label)}, row says {expected_dim}"
            )
        terms.append((coeff, label))

    if len(terms) != n_terms:
        raise ValueError(f"{path}: expected {n_terms} terms, read {len(terms)}")
    return terms


def pauli_matrix(label: str) -> np.ndarray:
    m = PAULI_2X2[label[0]]
    for c in label[1:]:
        m = np.kron(m, PAULI_2X2[c])
    return m


def assemble_matrix(terms) -> np.ndarray:
    """파일 순서대로 가중합. C++ 쪽 누적 순서와 맞춘다."""
    dim = 2 ** len(terms[0][1])
    m = np.zeros((dim, dim), dtype=np.complex128)
    for coeff, label in terms:
        m += coeff * pauli_matrix(label)
    return m


def matvec_energy(psi: np.ndarray, m: np.ndarray) -> float:
    """run_vqe의 목적 함수가 평가하는 바로 그 식."""
    energy = complex(psi.conj() @ m @ psi)
    if abs(energy.imag) > 1e-9:
        raise ValueError(f"Non-real expectation value: {energy}")
    return float(energy.real)


class Case:
    """하나의 (molecule, R) 쌍을 평가하는 데 필요한 모든 것."""

    def __init__(self, molecule: str, r_text: str, dim: int):
        terms = read_terms(molecule, r_text, dim)
        self.cpp = qs.Hamiltonian(terms)
        self.matrix = assemble_matrix(terms)

        # 분자 행에만 존재한다. 파일의 기준값을 실제 Python 코드 경로로
        # 다시 유도해 보기 위한 것.
        self.pauli_hamiltonian = None
        self.matrix_verified = False

        loader = REAL_HAMILTONIAN_LOADERS.get(molecule)
        if loader is None:
            return

        self.pauli_hamiltonian = loader(distance=float(r_text))
        reference = hamiltonian_matrix(self.pauli_hamiltonian)
        if not np.array_equal(reference, self.matrix):
            spread = float(np.max(np.abs(reference - self.matrix)))
            raise SystemExit(
                f"{molecule} R={r_text}: locally assembled matrix differs "
                f"from hamiltonian_matrix() by {spread:.3e}.\n"
                f"The text file's term order likely differs from "
                f"PauliHamiltonian.terms, which _to_pauli_hamiltonian sorts "
                f"by label. Reconcile the two before trusting the vs_matvec "
                f"column."
            )
        self.matrix_verified = True


def new_group() -> dict:
    return {
        "rows": 0,
        **{p: {"fail": 0, "max": 0.0, "worst": ""} for p in PATHS},
    }


def main() -> int:
    if not STATES_PATH.exists():
        print(f"missing {STATES_PATH}; run tools/dump_crossval_states.py first")
        return 1

    cache: dict[tuple[str, str], Case] = {}
    stats: dict[str, dict] = OrderedDict()

    max_internal_spread = 0.0
    internal_worst = ""

    # 기준 파일 자체에 대한 점검이다. PauliHamiltonian을 구할 수 있는 행에서
    # 저장된 에너지를 expectation_value()로 다시 계산해 파일과 정확히 같은지
    # 확인한다. 여기서 0이 아닌 값이 나오면, 파일이 헤더가 주장하는 경로로
    # 작성되지 않았다는 뜻이다.
    max_reference_drift = 0.0
    reference_checked = 0

    verified_cases = 0

    for line_no, line in enumerate(STATES_PATH.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        fields = line.split()
        molecule, r_text, _kind = fields[0], fields[1], fields[2]
        dim = int(fields[3])

        expected_fields = 4 + 2 * dim + 1
        if len(fields) != expected_fields:
            print(f"line {line_no}: expected {expected_fields} fields, "
                  f"got {len(fields)}")
            return 1

        flat = np.array([float(x) for x in fields[4:4 + 2 * dim]])
        psi = flat[0::2] + 1j * flat[1::2]
        reference = float(fields[-1])

        key = (molecule, r_text)
        if key not in cache:
            cache[key] = Case(molecule, r_text, dim)
            if cache[key].matrix_verified:
                verified_cases += 1
        case = cache[key]

        if case.pauli_hamiltonian is not None:
            drift = abs(expectation_value(psi, case.pauli_hamiltonian)
                        - reference)
            max_reference_drift = max(max_reference_drift, drift)
            reference_checked += 1

        cpp_termwise = case.cpp.expectation(psi)
        cpp_dense = case.cpp.expectation_dense(psi)
        py_matvec = matvec_energy(psi, case.matrix)

        spread = abs(cpp_termwise - cpp_dense)
        if spread > max_internal_spread:
            max_internal_spread = spread
            internal_worst = f"{molecule} R={r_text}"

        diffs = {
            "termwise": abs(cpp_termwise - reference),
            "dense": abs(cpp_dense - reference),
            "vs_matvec": abs(cpp_dense - py_matvec),
        }

        g = stats.setdefault(molecule, new_group())
        g["rows"] += 1

        for path, diff in diffs.items():
            p = g[path]
            if diff > p["max"]:
                p["max"] = diff
                p["worst"] = r_text
            if diff > TOLERANCE:
                p["fail"] += 1
                print(f"FAIL  {path:<10s} {molecule:<14s} R={r_text:<8s}  "
                      f"diff={diff:.3e}")

    names = GROUP_ORDER + [n for n in stats if n not in GROUP_ORDER]

    print(f"\n{'group':<14s} {'rows':>5s}  "
          f"{'termwise':>11s} {'dense':>11s} {'vs_matvec':>11s}")
    print("-" * 56)

    total_rows = 0
    totals = {p: {"fail": 0, "max": 0.0} for p in PATHS}

    for name in names:
        g = stats.get(name)
        if g is None:
            continue
        print(f"{name:<14s} {g['rows']:>5d}  "
              f"{g['termwise']['max']:>11.3e} {g['dense']['max']:>11.3e} "
              f"{g['vs_matvec']['max']:>11.3e}")
        total_rows += g["rows"]
        for path in PATHS:
            totals[path]["fail"] += g[path]["fail"]
            totals[path]["max"] = max(totals[path]["max"], g[path]["max"])

    print("-" * 56)
    print(f"{'total':<14s} {total_rows:>5d}  "
          f"{totals['termwise']['max']:>11.3e} {totals['dense']['max']:>11.3e} "
          f"{totals['vs_matvec']['max']:>11.3e}")

    print(f"\nfailures (> {TOLERANCE:.1e}):  "
          + ", ".join(f"{p} {totals[p]['fail']}" for p in PATHS))
    print(f"max |termwise - dense| within C++: {max_internal_spread:.3e}"
          f"  ({internal_worst})")
    print(f"matrices checked against hamiltonian_matrix(): "
          f"{verified_cases} of {len(cache)}")
    print(f"reference file re-derived on {reference_checked} rows, "
          f"max drift {max_reference_drift:.3e}")

    all_pass = all(totals[p]["fail"] == 0 for p in PATHS)
    print(f"result: {'PASS' if all_pass else 'FAIL'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())