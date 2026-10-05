"""VQE (Variational Quantum Eigensolver) 메인 루프.

논문 Kim et al., Sci. Adv. 10, eado3472 (2024)의 VQE 알고리즘을
SciPy COBYLA로 구현한다.

흐름:
    1. 초기 각도 파라미터 α_0
    2. ansatz(α_n) → state vector
    3. ⟨ψ|H|ψ⟩ → 에너지
    4. COBYLA로 α_{n+1} 갱신
    5. |α_{n+1} − α_n| < 0.01(논문 명시값) 수렴 시 종료

ansatz와 Hamiltonian이 분리되어 있어 H2(4D)와 LiH(16D)를 같은 루프로 다룬다.

에너지 평가 경로가 둘이다. run_vqe/run_vqe_multistart는 NumPy로 계산하고,
run_vqe_cpp/run_vqe_multistart_cpp는 C++ 커널(qudit_simulator)에 위임한다.
최적화 루프 자체는 _run_vqe_with_evaluator 하나를 공유하므로, 루프 쪽 변경은
두 경로에 동시에 반영된다. 두 경로를 비교할 때 "루프가 달라서 갈렸나"를
의심할 필요가 없다는 뜻이기도 하다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from sqd_vqe.ansatz import H2_NUM_PARAMS, h2_ansatz_state
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import PauliHamiltonian


AnsatzFn = Callable[[NDArray[np.float64]], NDArray[np.complex128]]

#: state vector → 실수 에너지. 최적화 루프에 주입되는 평가 함수.
EnergyFn = Callable[[NDArray[np.complex128]], float]


@dataclass
class VQEResult:
    """VQE 최적화 결과.

    energy: 최종 ground state energy 추정값 (평가된 값 중 최소)
    params: 그 지점의 각도 파라미터
    energy_history: 함수 평가별 에너지 (수렴 곡선용)
    param_history: 함수 평가별 파라미터 (Fig. 3A/4A 재현용)
    n_iterations: 총 함수 평가 횟수. SciPy COBYLA의 maxiter는 실제로
        함수 평가 횟수이며, 논문의 실험 iteration(H2 48±6.8,
        LiH 242±29.3)과는 다른 단위다
    n_restarts: multi-start 시도 횟수 (단일 실행이면 1)
    best_restart_seed: 최선의 결과를 낸 restart의 시드

    frozen이 아닌 이유: _run_multistart가 n_restarts와 best_restart_seed를
    사후에 채워 넣는다. 루프 본체는 단일 실행만 알면 되도록 두기 위한 선택.
    """
    energy: float
    params: NDArray[np.float64]
    energy_history: list[float]
    param_history: list[NDArray[np.float64]]
    n_iterations: int
    n_restarts: int = 1
    best_restart_seed: int | None = None


# --------------------------------------------------------------------------
# 공통 최적화 루프
# --------------------------------------------------------------------------

def _run_vqe_with_evaluator(
    evaluate: EnergyFn,
    ansatz: AnsatzFn,
    initial_params: NDArray[np.float64] | None,
    seed: int,
    n_params: int,
    tolerance: float,
    max_iter: int,
) -> VQEResult:
    """COBYLA 루프 본체. 에너지 평가만 주입받는다.

    evaluate: state vector → 실수 에너지. 허수부 검사는 평가 함수 책임.
    나머지 인자는 run_vqe와 같은 의미.
    """
    if initial_params is None:
        # 각도 파라미터이므로 [0, 2π)에서 균등하게 뽑는다. default_rng에
        # seed를 주면 같은 seed가 항상 같은 초기값을 낸다 — NumPy 경로와
        # C++ 경로를 같은 지점에서 출발시킬 수 있는 근거.
        rng = np.random.default_rng(seed)
        initial_params = rng.uniform(0, 2 * np.pi, size=n_params)

    energy_history: list[float] = []
    param_history: list[NDArray[np.float64]] = []

    def objective(params: NDArray[np.float64]) -> float:
        state = ansatz(params)
        e = evaluate(state)
        # 매 평가를 기록한다. Fig. 3A/4A가 이 history를 그대로 그리고,
        # 두 커널의 궤적이 어디서 갈렸는지 찾을 때도 쓴다.
        energy_history.append(e)
        # SciPy가 넘겨주는 배열을 재사용할 수 있으므로 복사해 둔다.
        # 복사하지 않으면 history 전체가 같은 객체를 가리킬 위험이 있다.
        param_history.append(params.copy())
        return e

    result = minimize(
        objective,
        x0=initial_params,
        method="COBYLA",
        options={
            # rhobeg는 초기 trust region 크기다. SciPy 기본값 1.0을 쓴다.
            # 작게 주면(예: 0.01) trust region이 충분히 줄어들지 못해
            # 종료 조건이 걸리지 않고 max_iter까지 간다 — 실제로 겪은 문제.
            "rhobeg": 1.0,
            # tol은 COBYLA의 rhoend에 대응한다. 논문의 종료 기준
            # |α_{n+1} − α_n| < 0.01이 바로 이 값이다.
            "tol": tolerance,
            "maxiter": max_iter,
            "disp": False,
        },
    )

    return VQEResult(
        # result.fun은 평가된 값 중 최소다. energy_history[-1]과 다를 수 있다 —
        # COBYLA가 simplex를 갱신하려고 일부러 나쁜 점도 평가하기 때문이다.
        energy=float(result.fun),
        params=np.asarray(result.x),
        energy_history=energy_history,
        param_history=param_history,
        n_iterations=len(energy_history),
    )


def _run_multistart(
    evaluate: EnergyFn,
    ansatz: AnsatzFn,
    n_restarts: int,
    seed: int,
    n_params: int,
    tolerance: float,
    max_iter: int,
) -> VQEResult:
    """서로 다른 초기값으로 여러 번 돌리고 최선을 고른다.

    평가 함수는 restart 사이에 재사용한다. Hamiltonian은 최적화 동안 고정이므로
    restart마다 다시 만들 이유가 없다.
    """
    best: VQEResult | None = None
    best_seed: int | None = None

    for i in range(n_restarts):
        result = _run_vqe_with_evaluator(
            evaluate=evaluate,
            ansatz=ansatz,
            initial_params=None,
            # seed, seed+1, seed+2, ... 로 가므로 재현 가능하면서도 서로 다른
            # 초기값이 나온다.
            seed=seed + i,
            n_params=n_params,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        # 변분 원리상 에너지는 낮을수록 좋다. 더 낮은 값을 찾으면 교체한다.
        if best is None or result.energy < best.energy:
            best = result
            best_seed = seed + i

    assert best is not None
    # 반환되는 history는 '최선이었던 그 시도'의 것이다. 여러 시도를 섞지
    # 않으므로 Fig. 4A처럼 수렴 곡선을 그릴 때 의미가 유지된다.
    best.n_restarts = n_restarts
    best.best_restart_seed = best_seed
    return best


# --------------------------------------------------------------------------
# 평가 함수 생성
# --------------------------------------------------------------------------

def _make_numpy_evaluator(hamiltonian: PauliHamiltonian) -> EnergyFn:
    """Hamiltonian을 d×d 행렬로 한 번 빌드해 재사용하는 평가 함수.

    Pauli string을 매번 순회하는 것보다 LiH(100 terms)에서 약 100배 빠르며,
    결과는 수치적으로 동일하다 (test_expectation.py가 두 경로의 일치를 검증).
    """
    # 클로저에 M을 가둬 둔다. 수천 번의 평가 동안 한 번만 만든다.
    M = hamiltonian_matrix(hamiltonian)

    def evaluate(state: NDArray[np.complex128]) -> float:
        energy = complex(state.conj() @ M @ state)
        # Hermitian H와 정규화된 state라면 결과는 실수여야 한다. 허수부가
        # 남으면 입력이 잘못된 것이므로 조용히 실수부만 취하지 않는다.
        if abs(energy.imag) > 1e-9:
            raise ValueError(f"Non-real expectation value: {energy}")
        return float(energy.real)

    return evaluate


def _make_cpp_evaluator(hamiltonian: PauliHamiltonian) -> EnergyFn:
    """C++ 커널에 위임하는 평가 함수.

    expectation_dense를 쓴다. NumPy 경로가 행렬을 한 번 만들어
    state.conj() @ M @ state를 계산하므로, 대응되는 것은 항별 순회가 아니라
    조밀 행렬 수축이다. 두 경로의 실측 차이는 55개 검증 state에서 최대
    4 ULP (3.553e-15) 였다.

    PauliHamiltonian.terms는 (label, coeff) 순서이고 C++ 바인딩은
    (coeff, label)을 받으므로 뒤집는다. 항 순서는 그대로 두어야 한다 —
    양쪽이 같은 순서로 누적해야 조밀 행렬이 비트 단위로 같아진다.

    import를 함수 안에서 하는 이유: C++ 모듈은 선택적 의존성이고, 없는
    환경에서 vqe 모듈 자체의 import가 실패하면 안 된다.
    """
    try:
        import qudit_simulator as qs
    except ImportError as exc:
        # 원인과 해결 방법을 함께 알려준다. 단순히 ModuleNotFoundError가
        # 올라가면 사용자는 빌드 플래그를 추측해야 한다.
        raise ImportError(
            "qudit_simulator module not found. Build it in the "
            "qudit-simulator-cpp repo with -DQUDIT_BUILD_PYTHON=ON and put "
            "build-py/python on PYTHONPATH."
        ) from exc

    # 생성자 안에서 Pauli 행렬 빌드와 조밀 행렬 누적이 일어난다. 여기서
    # 한 번 만들고 아래에서 메서드만 반환하므로, 평가 때마다 재구성되지 않는다.
    H_cpp = qs.Hamiltonian(
        [(coeff, label) for label, coeff in hamiltonian.terms]
    )
    return H_cpp.expectation_dense


# --------------------------------------------------------------------------
# 공개 API — NumPy 경로
# --------------------------------------------------------------------------

def run_vqe(
    hamiltonian: PauliHamiltonian,
    ansatz: AnsatzFn = h2_ansatz_state,
    initial_params: NDArray[np.float64] | None = None,
    seed: int = 42,
    n_params: int = H2_NUM_PARAMS,
    tolerance: float = 0.01,
    max_iter: int = 3000,
) -> VQEResult:
    """단일 초기값에서 VQE를 실행한다.

    hamiltonian: PauliHamiltonian
    ansatz: params → state vector. 기본은 H2 ansatz (식 5)
    initial_params: 초기값. None이면 seed 기반 난수
    seed: 난수 시드 (재현 가능성)
    tolerance: COBYLA 종료 기준(= rhoend). 논문 명시값 0.01.
        rhobeg는 SciPy 기본값 1.0을 쓴다 — 작게 주면 trust region이
        충분히 줄어들지 못해 max_iter까지 종료되지 않는다
    max_iter: 함수 평가 상한. 안전장치
    """
    return _run_vqe_with_evaluator(
        evaluate=_make_numpy_evaluator(hamiltonian),
        ansatz=ansatz,
        initial_params=initial_params,
        seed=seed,
        n_params=n_params,
        tolerance=tolerance,
        max_iter=max_iter,
    )


def run_vqe_multistart(
    hamiltonian: PauliHamiltonian,
    ansatz: AnsatzFn = h2_ansatz_state,
    n_restarts: int = 3,
    seed: int = 42,
    n_params: int = H2_NUM_PARAMS,
    tolerance: float = 0.01,
    max_iter: int = 3000,
) -> VQEResult:
    """여러 무작위 초기값에서 VQE를 실행하고 최선의 결과를 반환한다.

    고차원 ansatz에서 COBYLA는 local minimum에 자주 갇힌다. LiH(30차원)의
    단일 실행 성공률은 약 50%이며, 5회 restart면 실측상 항상 chemical
    accuracy에 도달한다. 논문도 각 interatomic distance에서 실험을 여러 번
    수행하고 에너지 차이가 최소인 결과를 채택했다 (Figs. 3B, 4B 캡션).

    n_restarts: 서로 다른 초기값으로 시도할 횟수
    seed: 각 restart의 시드는 seed, seed+1, ... (재현 가능)

    반환: 가장 낮은 에너지를 얻은 시도의 VQEResult. history도 그 시도의 것.
    """
    return _run_multistart(
        evaluate=_make_numpy_evaluator(hamiltonian),
        ansatz=ansatz,
        n_restarts=n_restarts,
        seed=seed,
        n_params=n_params,
        tolerance=tolerance,
        max_iter=max_iter,
    )


# --------------------------------------------------------------------------
# 공개 API — C++ 커널 경로
# --------------------------------------------------------------------------

def run_vqe_cpp(
    hamiltonian: PauliHamiltonian,
    ansatz: AnsatzFn = h2_ansatz_state,
    initial_params: NDArray[np.float64] | None = None,
    seed: int = 42,
    n_params: int = H2_NUM_PARAMS,
    tolerance: float = 0.01,
    max_iter: int = 3000,
) -> VQEResult:
    """run_vqe와 동일하되 에너지 평가를 C++ 커널에 위임한다.

    인자와 반환값은 run_vqe와 같다. 같은 seed를 주면 초기값도 같으므로 두
    함수의 결과를 직접 비교할 수 있다.

    목적 함수가 4 ULP 수준으로 다르므로 궤적이 비트 단위로 같으리라 기대하면
    안 된다. COBYLA는 결정적이지만, 두 후보의 에너지 차가 그 정도로 좁혀지는
    지점에서 비교가 뒤집히면 이후 경로가 갈린다. 비교는 fidelity
    |⟨ψ_py|ψ_cpp⟩|²와 최종 에너지로 하는 것이 맞다. 최종 파라미터 비교는
    적절하지 않다 — 이 파라미터화에는 중복이 있어(θ=0이면 그 아래 ω가
    무의미) 같은 state가 다른 파라미터로 나올 수 있다.
    """
    return _run_vqe_with_evaluator(
        evaluate=_make_cpp_evaluator(hamiltonian),
        ansatz=ansatz,
        initial_params=initial_params,
        seed=seed,
        n_params=n_params,
        tolerance=tolerance,
        max_iter=max_iter,
    )


def run_vqe_multistart_cpp(
    hamiltonian: PauliHamiltonian,
    ansatz: AnsatzFn = h2_ansatz_state,
    n_restarts: int = 3,
    seed: int = 42,
    n_params: int = H2_NUM_PARAMS,
    tolerance: float = 0.01,
    max_iter: int = 3000,
) -> VQEResult:
    """run_vqe_multistart의 C++ 커널 버전.

    C++ Hamiltonian은 한 번만 만들어 restart 전체에서 재사용한다. Pauli 행렬
    빌드와 조밀 행렬 누적이 생성자에서 일어나므로, LiH 100항을 restart 횟수만큼
    반복할 이유가 없다.
    """
    return _run_multistart(
        evaluate=_make_cpp_evaluator(hamiltonian),
        ansatz=ansatz,
        n_restarts=n_restarts,
        seed=seed,
        n_params=n_params,
        tolerance=tolerance,
        max_iter=max_iter,
    )