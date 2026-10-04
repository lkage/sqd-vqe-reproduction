"""분자 Hamiltonian을 Qiskit Nature로 생성해 Pauli string 가중합으로 반환.

논문 Kim et al., Sci. Adv. 10, eado3472 (2024)의 Table S1(H2), Table S2(LiH)와
비교 가능한 형태로 만든다.

두 분자 모두 축소된 qubit 수로 매핑된다:
  H2  : 4 spin-orbital → ParityMapper 2-qubit reduction → 2 qubit (4D)
  LiH : 12 spin-orbital → FreezeCore + orbital 제거 → 6 spin-orbital
        → ParityMapper 2-qubit reduction → 4 qubit (16D)

JordanWignerMapper만으로는 축소가 일어나지 않는다. H2의 경우 4 qubit,
15개 Pauli term이 나와 논문의 2 qubit, 5 term과 형태가 다르다. 축소는
ParityMapper에 num_particles를 넘길 때 자동으로 적용된다 — 입자 수 보존과
스핀 대칭을 이용해 두 큐비트를 제거한다.

재현성: Qiskit의 fermion→Pauli 변환은 같은 Pauli string끼리 계수를 합산할 때
내부 순회 순서가 프로세스마다 달라, 계수가 최대 ~150 ULP(상대오차 2e-14) 변한다.
물리적으로는 무해하고 Table S1/S2와의 일치에도 영향이 없지만, 고차원 COBYLA가
이를 증폭시켜 다른 local minimum으로 수렴하게 만든다. get_*_hamiltonian()은
한 번 생성한 결과를 JSON으로 고정한다. (추적 과정은 examples/diagnostics/ 참조)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer


# 고정된 Hamiltonian 데이터 위치. 패키지 내부에 두어 레포를 clone하면
# 바로 따라오게 한다 — C++ cross-validation도 이 파일들을 읽는다.
HAMILTONIAN_DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class PauliHamiltonian:
    """Pauli string의 가중합으로 표현된 Hamiltonian.

    terms: [("II", -0.326386), ("IZ", 0.401061), ...] 형태.
        모든 상수항(nuclear repulsion, frozen core energy 등)이
        identity string 계수에 이미 합산되어 있다. 논문 Table S1/S2 컨벤션.
    nuclear_repulsion: 참조용으로 별도 보존. 검증/디버깅에 사용.

    frozen=True인 이유: Hamiltonian은 한 번 만들어지면 바뀌지 않는 값이다.
    VQE가 수천 번 참조하는 동안 누가 수정할 여지를 아예 없앤다.
    """
    terms: list[tuple[str, float]]
    nuclear_repulsion: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.terms)

    @property
    def num_qubits(self) -> int:
        # 모든 라벨의 길이가 같다는 전제. _to_pauli_hamiltonian이 하나의
        # qubit_op에서 만들므로 보장된다.
        return len(self.terms[0][0])

    @property
    def dimension(self) -> int:
        """Hilbert 공간 차원 (2^num_qubits). H2: 4, LiH: 16."""
        return 2 ** self.num_qubits


# --------------------------------------------------------------------------
# Hamiltonian 생성 (Qiskit Nature 직접 호출)
# --------------------------------------------------------------------------

def _to_pauli_hamiltonian(problem) -> PauliHamiltonian:
    """ElectronicStructureProblem을 PauliHamiltonian으로 변환.

    모든 상수항을 identity Pauli string 계수에 합산한다. LiH처럼 active space
    축소를 하면 상수가 둘(nuclear repulsion + frozen core energy)이며, 둘 다
    더해야 논문 Table S2의 IIII 계수와 일치한다. H2는 상수가 하나뿐이라
    이 함정이 드러나지 않는다 — LiH로 넘어갈 때 처음 발견된 문제다.
    """
    fermionic_op = problem.hamiltonian.second_q_op()

    # constants는 dict다. H2는 {nuclear_repulsion_energy: ...} 하나뿐이고,
    # LiH는 {nuclear_repulsion_energy: +1.02, FreezeCoreTransformer: -7.82}로
    # 둘이다. sum()으로 받으면 양쪽을 같은 코드로 처리할 수 있다.
    constants = problem.hamiltonian.constants
    total_constant = float(sum(constants.values()))

    # num_particles를 넘기는 것이 핵심. 이게 있어야 two-qubit reduction이
    # 적용되어 논문과 같은 qubit 수가 나온다.
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(fermionic_op)

    terms: list[tuple[str, float]] = []
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        # Hermitian Hamiltonian이면 계수는 실수여야 한다. 허수부가 남아 있으면
        # 앞 단계에서 뭔가 잘못된 것이므로 조용히 버리지 않고 터뜨린다.
        if abs(coeff.imag) > 1e-10:
            raise ValueError(
                f"Non-real coefficient in Hermitian Hamiltonian: "
                f"{pauli} -> {coeff}"
            )
        label = str(pauli)
        c = float(coeff.real)
        # identity string("II", "IIII")을 찾는다. 라벨 길이가 분자마다 다르므로
        # 문자열 비교 대신 집합으로 판정한다.
        if set(label) == {"I"}:
            c += total_constant
        terms.append((label, c))

    # 라벨 기준 정렬. 출력 순서를 고정해 두면 사람이 읽기 쉽고, 두 번 생성한
    # 결과를 비교할 때 항 순서 차이와 계수 차이를 구분할 수 있다.
    # (다만 이 정렬만으로는 위에서 말한 ULP 비결정성이 해결되지 않는다 —
    #  문제는 최종 나열 순서가 아니라 Qiskit 내부의 중간 합산 단계에 있다.)
    terms.sort(key=lambda t: t[0])

    return PauliHamiltonian(
        terms=terms,
        nuclear_repulsion=float(
            constants.get("nuclear_repulsion_energy", 0.0)
        ),
    )


def build_h2_hamiltonian(
    distance: float = 0.73,
    basis: str = "sto3g",
) -> PauliHamiltonian:
    """H2 Hamiltonian을 2-qubit(4D) reduced 형태로 생성.

    논문 Table S1과 일치 (R = 0.1 ~ 3.0 Å에서 6자리까지 검증됨).

    distance: 단위 Å. H2의 bonding length는 0.73 Å.
    basis: 기저함수. 논문은 STO-3G.
    """
    # 원자를 z축 위에 놓는다. 이 배치는 LiH에서 중요해진다 — 결합 방향이
    # z축이어야 2px, 2py orbital을 제거하는 것이 정당해진다.
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis=basis,
    )
    return _to_pauli_hamiltonian(driver.run())


def build_lih_hamiltonian(
    distance: float = 1.55,
    basis: str = "sto3g",
) -> PauliHamiltonian:
    """LiH Hamiltonian을 4-qubit(16D) reduced 형태로 생성.

    논문 Table S2 (LiH-qiskit, R = 1.55 Å, 100 Pauli strings)와 일치.

    축소 경로:
        LiH/STO-3G: 6 spatial orbital, 4 electron (12 spin-orbital)
        → FreezeCoreTransformer: Li 1s core 동결 (전자 2개 제외)
          + orbital 3, 4 제거 (2px, 2py — z축 결합에 비기여)
          → 3 spatial orbital, 2 active electron (6 spin-orbital)
        → ParityMapper 2-qubit reduction → 4 qubit (16D)

    distance: 단위 Å. 논문 Table S2의 bonding length는 1.55 Å.
        (Table S4는 openfermion 기준 1.51 Å로 다른 데이터이므로 혼동 주의)
    """
    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {distance}",
        basis=basis,
    )
    problem = driver.run()

    # freeze_core=True가 Li 1s를 얼리고, remove_orbitals=[3,4]가 2px, 2py를
    # 뺀다. 두 orbital은 z축 결합에 기여하지 않으므로 ground state 기술에
    # 필요 없다. ActiveSpaceTransformer(4, 3)은 틀린 설정이다 — core를
    # 얼리지 않아 활성 전자가 4개가 되고 물리가 달라진다.
    transformer = FreezeCoreTransformer(
        freeze_core=True,
        remove_orbitals=[3, 4],
    )
    return _to_pauli_hamiltonian(transformer.transform(problem))


# --------------------------------------------------------------------------
# 캐시 (재현성 및 C++ cross-validation용)
# --------------------------------------------------------------------------

def save_hamiltonian(H: PauliHamiltonian, path: Path) -> None:
    """Hamiltonian을 JSON으로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 튜플은 JSON에 없으므로 리스트로 적는다. 읽을 때 다시 튜플로 돌린다.
    payload = {
        "terms": [[label, coeff] for label, coeff in H.terms],
        "nuclear_repulsion": H.nuclear_repulsion,
    }
    path.write_text(json.dumps(payload, indent=1))


def load_hamiltonian(path: Path) -> PauliHamiltonian:
    """저장된 Hamiltonian을 읽어온다."""
    payload = json.loads(path.read_text())
    return PauliHamiltonian(
        terms=[(label, float(c)) for label, c in payload["terms"]],
        nuclear_repulsion=float(payload["nuclear_repulsion"]),
    )


def _get_cached(
    molecule: str,
    builder,
    distance: float,
    basis: str,
    use_cache: bool,
) -> PauliHamiltonian:
    """캐시가 있으면 읽고, 없으면 생성 후 저장."""
    # 파일명에 R을 %.4f로 적는다. 이 표기가 곧 조회 키가 되므로, 다른
    # 스크립트(tools/)도 같은 포맷을 써야 파일을 찾을 수 있다.
    cache_path = (
        HAMILTONIAN_DATA_DIR / f"{molecule}_{basis}_R{distance:.4f}.json"
    )
    if use_cache and cache_path.exists():
        return load_hamiltonian(cache_path)

    H = builder(distance=distance, basis=basis)
    if use_cache:
        save_hamiltonian(H, cache_path)
    return H


def get_h2_hamiltonian(
    distance: float = 0.73,
    basis: str = "sto3g",
    use_cache: bool = True,
) -> PauliHamiltonian:
    """H2 Hamiltonian을 캐시에서 가져온다 (없으면 생성 후 저장).

    H2는 Pauli string이 5개뿐이라 ULP 비결정성이 실질적으로 나타나지 않지만,
    C++ cross-validation에서 양쪽이 동일한 입력을 읽어야 하므로 캐시 경로를
    제공한다. LiH와 인터페이스를 맞추는 효과도 있다.

    use_cache=False면 항상 새로 생성 (Qiskit 재현성 조사용).
    """
    return _get_cached(
        "h2", build_h2_hamiltonian, distance, basis, use_cache
    )


def get_lih_hamiltonian(
    distance: float = 1.55,
    basis: str = "sto3g",
    use_cache: bool = True,
) -> PauliHamiltonian:
    """LiH Hamiltonian을 캐시에서 가져온다 (없으면 생성 후 저장).

    LiH는 100개 Pauli string 중 16개가 프로세스마다 최대 152 ULP 다르게
    나온다. 30차원 COBYLA가 이를 증폭시키므로 캐시 사용이 사실상 필수다.
    캐시 없이 돌리면 같은 seed로도 매번 다른 local minimum에 안착한다.

    use_cache=False면 항상 새로 생성 (Qiskit 재현성 조사용).
    """
    return _get_cached(
        "lih", build_lih_hamiltonian, distance, basis, use_cache
    )