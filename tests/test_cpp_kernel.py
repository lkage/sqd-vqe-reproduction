"""C++ 커널 경로 검증.

qudit_simulator 모듈이 없는 환경에서는 전체가 skip된다. 이 레포는 C++
모듈 없이도 완전히 동작해야 하므로, 모듈 유무가 테스트 통과 여부를
바꾸면 안 된다.

검증 범위:
  1. 바인딩의 입력 검증 (빈 항, 라벨 길이, state 차원, 정규화)
  2. C++ 내부 두 경로(termwise / dense)의 일치
  3. C++과 NumPy의 일치 — 특히 켤레 방향
  4. run_vqe_cpp의 수렴과 재현성
  5. 모듈이 없을 때의 에러 메시지

더 광범위한 교차 검증은 tools/crossval_pybind.py가 55개 state로 수행한다.
여기서는 CI에서 빠르게 돌 수 있는 것만 둔다.
"""

import math
import sys

import numpy as np
import pytest

# 모듈이 없으면 이 파일 전체를 skip한다. collect 단계에서 처리되므로
# 아래 import들이 실패할 일은 없다.
qs = pytest.importorskip(
    "qudit_simulator",
    reason="C++ kernel not built; see README for build instructions",
)

from sqd_vqe.ansatz import H2_NUM_PARAMS, h2_ansatz_state  # noqa: E402
from sqd_vqe.expectation import hamiltonian_matrix  # noqa: E402
from sqd_vqe.hamiltonian import (  # noqa: E402
    PauliHamiltonian,
    get_h2_hamiltonian,
    get_lih_hamiltonian,
)
from sqd_vqe.vqe import (  # noqa: E402
    _make_cpp_evaluator,
    _make_numpy_evaluator,
    run_vqe,
    run_vqe_cpp,
)


CHEMICAL_ACCURACY = 1.6e-3

# Y가 홀수인 항. 이것이 있어야 Hamiltonian 행렬이 복소수가 되고, 켤레
# 방향이 뒤집혔을 때 값이 갈린다. 분자 Hamiltonian은 Y가 항상 짝수라
# 실수 대칭 행렬이고, 실수 H에서는 켤레를 어느 쪽에 걸든 결과가 같다.
ODD_Y_TERMS = [("IY", 0.137), ("XY", -0.091), ("YZ", 0.223)]


def to_cpp(H: PauliHamiltonian):
    """PauliHamiltonian을 C++ Hamiltonian으로.

    Python은 (label, coeff), C++ 바인딩은 (coeff, label) 순서를 받는다.
    항 순서는 그대로 둔다 — 양쪽이 같은 순서로 누적해야 조밀 행렬이
    비트 단위로 같아진다.
    """
    return qs.Hamiltonian([(c, label) for label, c in H.terms])


def with_odd_y(H: PauliHamiltonian) -> PauliHamiltonian:
    """H2 Hamiltonian에 Y 홀수 항을 더해 복소 행렬로 만든다.

    Pauli string은 Hermitian이고 계수가 실수이므로 결과도 Hermitian이다.
    달라지는 것은 행렬이 더 이상 실수가 아니라는 점뿐이다.
    """
    merged = dict(H.terms)
    for label, coeff in ODD_Y_TERMS:
        assert label.count("Y") % 2 == 1
        merged[label] = coeff
    return PauliHamiltonian(
        terms=sorted(merged.items(), key=lambda t: t[0]),
        nuclear_repulsion=H.nuclear_repulsion,
    )


def random_states(dim: int, count: int, seed: int) -> list[np.ndarray]:
    """Haar-uniform 복소 단위 벡터. 복소 정규분포를 뽑아 정규화한다."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        out.append(v / np.linalg.norm(v))
    return out


@pytest.fixture(scope="module")
def h2():
    return get_h2_hamiltonian(distance=0.73)


@pytest.fixture(scope="module")
def lih():
    return get_lih_hamiltonian(distance=1.55)


# --------------------------------------------------------------------------
# 1. 바인딩의 입력 검증
# --------------------------------------------------------------------------

def test_dimension_and_num_terms(h2, lih):
    """차원과 항 개수를 올바로 보고한다."""
    assert to_cpp(h2).dimension == 4
    assert to_cpp(h2).num_terms == 5
    assert to_cpp(lih).dimension == 16
    assert to_cpp(lih).num_terms == 100


def test_rejects_empty_terms():
    with pytest.raises(ValueError):
        qs.Hamiltonian([])


def test_rejects_mixed_label_lengths():
    """라벨 길이가 섞이면 차원이 모순된다.

    이걸 거부하지 않으면 첫 라벨이 차원을 정해 버리고 나머지는 조용히
    틀린 자리에 더해진다.
    """
    with pytest.raises(ValueError):
        qs.Hamiltonian([(1.0, "II"), (0.5, "IZZ")])


def test_rejects_wrong_state_dimension(h2):
    """state 차원이 안 맞으면 ValueError.

    이 검사가 없으면 Eigen이 Release 빌드에서 메모리를 넘어 읽는다.
    """
    H_cpp = to_cpp(h2)
    wrong = np.ones(8, dtype=np.complex128) / np.sqrt(8)
    with pytest.raises(ValueError):
        H_cpp.expectation_dense(wrong)
    with pytest.raises(ValueError):
        H_cpp.expectation(wrong)


def test_rejects_unnormalized_state(h2):
    """정규화되지 않은 state는 C++이 거부한다.

    NumPy 경로는 이 검사를 하지 않는다 — 두 경로의 의도된 비대칭이다.
    VQE 루프에서는 ansatz가 항상 정규화된 state를 주므로 실제로
    문제가 되지 않지만, 경로를 바꿔 끼울 때 알고 있어야 한다.
    """
    H_cpp = to_cpp(h2)
    bad = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.complex128)  # norm = sqrt(2)

    with pytest.raises(Exception):
        H_cpp.expectation_dense(bad)

    # NumPy 경로는 같은 입력에 대해 조용히 값을 돌려준다. 이것이 버그는
    # 아니지만, 두 경로가 다르게 행동한다는 사실 자체를 고정해 둔다.
    numpy_eval = _make_numpy_evaluator(h2)
    assert isinstance(numpy_eval(bad), float)


def test_cannot_build_non_hermitian():
    """이 API로는 비Hermitian Hamiltonian을 만들 수 없다.

    계수가 실수이고 Pauli string은 항상 Hermitian이므로 가중합도 항상
    Hermitian이다. 설계상의 성질이라 못으로 박아 둔다 — 나중에 복소
    계수를 받도록 바꾸면 이 테스트가 깨지면서 허수부 검사의 의미가
    달라졌음을 알려 준다.
    """
    H_cpp = qs.Hamiltonian([(1.0, "XY"), (-0.5, "IZ"), (0.3, "YY")])
    for psi in random_states(4, 5, seed=1):
        # 허수부 예외 없이 실수가 나와야 한다.
        assert isinstance(H_cpp.expectation_dense(psi), float)


# --------------------------------------------------------------------------
# 2. C++ 내부 두 경로의 일치
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 7, 42])
def test_termwise_matches_dense(h2, lih, seed):
    """항별 순회와 조밀 행렬 수축이 같은 값을 낸다.

    수학적으로 같고 부동소수점 합산 순서만 다르다. 실측 차이는 LiH에서
    최대 8 ULP였으므로 1e-12면 충분히 여유 있는 기준이다.
    """
    for H in (h2, lih):
        H_cpp = to_cpp(H)
        for psi in random_states(H.dimension, 3, seed=seed):
            assert abs(H_cpp.expectation(psi)
                       - H_cpp.expectation_dense(psi)) < 1e-12


# --------------------------------------------------------------------------
# 3. C++과 NumPy의 일치
# --------------------------------------------------------------------------

def test_matches_numpy_on_ground_state(h2):
    """ground state에서 두 경로가 같은 에너지를 낸다."""
    M = hamiltonian_matrix(h2)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    psi = eigenvectors[:, 0]

    cpp = to_cpp(h2).expectation_dense(psi)
    numpy = _make_numpy_evaluator(h2)(psi)

    assert abs(cpp - numpy) < 1e-12
    # eigenstate이므로 eigenvalue 자체와도 같아야 한다.
    assert abs(cpp - eigenvalues[0]) < 1e-12


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_matches_numpy_on_random_states(h2, lih, seed):
    """무작위 복소 state에서도 두 경로가 일치한다.

    VQE 루프가 비교하는 짝은 (C++ dense, NumPy matvec)이다. 실측 차이는
    55개 검증 state에서 최대 4 ULP였다.
    """
    for H in (h2, lih):
        H_cpp = to_cpp(H)
        numpy_eval = _make_numpy_evaluator(H)
        for psi in random_states(H.dimension, 3, seed=seed):
            assert abs(H_cpp.expectation_dense(psi) - numpy_eval(psi)) < 1e-12


def test_conjugation_direction(h2):
    """켤레 방향이 올바른지 확인한다.

    분자 Hamiltonian만으로는 이것을 검증할 수 없다. 실수 대칭 행렬에서는
    psi.conj() @ M @ psi 와 psi @ M @ psi.conj() 가 복소 psi에 대해서도
    항등적으로 같기 때문이다. Y가 홀수인 항을 넣어 행렬을 복소수로 만들어야
    두 값이 갈린다.
    """
    H = with_odd_y(h2)
    M = hamiltonian_matrix(H)

    # 전제 확인: 행렬이 실제로 복소수여야 이 테스트에 의미가 있다.
    assert not np.allclose(M.imag, 0)
    assert np.allclose(M, M.conj().T)  # 여전히 Hermitian

    H_cpp = to_cpp(H)
    for psi in random_states(4, 5, seed=99):
        correct = float(np.real(psi.conj() @ M @ psi))
        flipped = float(np.real(psi @ M @ psi.conj()))

        # 두 값이 실제로 갈리는지. 갈리지 않으면 이 테스트는 아무것도
        # 검증하지 못하므로, 그 사실 자체를 먼저 확인한다.
        assert not math.isclose(correct, flipped, abs_tol=1e-9)

        # C++은 올바른 쪽과 일치해야 한다.
        assert abs(H_cpp.expectation_dense(psi) - correct) < 1e-12
        assert abs(H_cpp.expectation(psi) - correct) < 1e-12


# --------------------------------------------------------------------------
# 4. run_vqe_cpp
# --------------------------------------------------------------------------

def test_run_vqe_cpp_converges(h2):
    """C++ 커널로도 chemical accuracy에 도달한다."""
    exact = float(np.linalg.eigvalsh(hamiltonian_matrix(h2))[0])
    result = run_vqe_cpp(h2, ansatz=h2_ansatz_state, seed=42,
                         n_params=H2_NUM_PARAMS)
    assert abs(result.energy - exact) < CHEMICAL_ACCURACY


def test_run_vqe_cpp_respects_variational_principle(h2):
    """결과가 정확한 ground state energy보다 작을 수 없다."""
    exact = float(np.linalg.eigvalsh(hamiltonian_matrix(h2))[0])
    result = run_vqe_cpp(h2, ansatz=h2_ansatz_state, seed=42,
                         n_params=H2_NUM_PARAMS)
    assert result.energy >= exact - 1e-9


def test_run_vqe_cpp_is_reproducible(h2):
    """같은 seed → 같은 결과."""
    kw = dict(ansatz=h2_ansatz_state, seed=42, n_params=H2_NUM_PARAMS)
    r1 = run_vqe_cpp(h2, **kw)
    r2 = run_vqe_cpp(h2, **kw)
    assert r1.energy == r2.energy
    assert np.array_equal(r1.params, r2.params)
    assert r1.n_iterations == r2.n_iterations


def test_both_kernels_reach_same_accuracy(h2):
    """두 커널이 비슷한 정확도에 도달한다.

    궤적이 비트 단위로 같으리라 기대하면 안 된다. 목적 함수가 몇 ULP
    다르고 COBYLA는 값을 비교해 움직이므로, 차이가 그 폭 안으로 좁혀지는
    지점에서 비교가 뒤집히면 이후 경로가 갈린다. 최종 결과가 둘 다
    chemical accuracy 안에 들어오는지만 본다.
    """
    exact = float(np.linalg.eigvalsh(hamiltonian_matrix(h2))[0])
    kw = dict(ansatz=h2_ansatz_state, seed=42, n_params=H2_NUM_PARAMS)

    a = run_vqe(h2, **kw)
    b = run_vqe_cpp(h2, **kw)

    assert abs(a.energy - exact) < CHEMICAL_ACCURACY
    assert abs(b.energy - exact) < CHEMICAL_ACCURACY


# --------------------------------------------------------------------------
# 5. 모듈이 없을 때
# --------------------------------------------------------------------------

def test_missing_module_gives_actionable_error(h2, monkeypatch):
    """모듈이 없으면 빌드 방법을 알려주는 ImportError가 올라온다.

    sys.modules에 None을 넣으면 import가 ImportError로 실패한다. 실제로
    모듈이 없는 상황을 흉내 내는 표준적인 방법이다.

    메시지가 중요한 이유: 그냥 ModuleNotFoundError가 올라가면 사용자가
    빌드 플래그를 추측해야 한다.
    """
    monkeypatch.setitem(sys.modules, "qudit_simulator", None)
    with pytest.raises(ImportError, match="QUDIT_BUILD_PYTHON"):
        _make_cpp_evaluator(h2)