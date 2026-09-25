"""H2 분자 Hamiltonian을 Qiskit Nature(PySCF + Parity mapping)로 생성.

논문 Kim et al., Sci. Adv. 10, eado3472 (2024) Table S1과 비교 가능한
2-qubit reduced 형태로 반환한다.

핵심 결정사항 (3주차 Day 2 검증 완료):
- ParityMapper(num_particles=...) 사용 → 2-qubit Hamiltonian
- 'II' 계수는 nuclear repulsion을 포함한 형태로 반환
  (논문 Table S1이 이 컨벤션)
- 계수 컨벤션: little-endian (Qiskit 기본). 논문과 일치 확인됨
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

@dataclass(frozen=True)
class PauliHamiltonian:
    """Pauli string의 가중합으로 표현된 Hamiltonian.

    terms: [("II", -0.326386), ("IZ", 0.401061), ...] 형태.
        nuclear repulsion이 이미 II 계수에 포함됨.
    nuclear_repulsion: 참조용으로 별도 보존. 검증/디버깅에 사용.
    """
    terms: list[tuple[str, float]]
    nuclear_repulsion: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.terms)


def build_h2_hamiltonian(
    distance: float = 0.73,
    basis: str = "sto3g",
) -> PauliHamiltonian:
    """H2 Hamiltonian을 주어진 interatomic distance로 생성.

    distance: 단위 Å
    basis: 기저함수. 논문은 STO-3G 사용.

    반환: PauliHamiltonian. 'II' 계수에 nuclear repulsion 포함.
    """
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis=basis,
    )
    problem = driver.run()
    return _to_pauli_hamiltonian(problem)


def build_lih_hamiltonian(
    distance: float = 1.55,
    basis: str = "sto3g",
) -> PauliHamiltonian:
    """LiH Hamiltonian을 4-qubit(16D) reduced 형태로 생성.

    논문 Table S2 (LiH-qiskit, R=1.55 Å, 100 Pauli strings)와 일치.

    축소 경로:
        LiH/STO-3G: 6 spatial orbital, 4 electron (12 spin-orbital)
        → FreezeCoreTransformer: Li 1s core 동결 (전자 2개 제외)
          + orbital 3, 4 제거 (2px, 2py — z축 결합 비기여)
          → 3 spatial orbital, 2 active electron (6 spin-orbital)
        → ParityMapper 2-qubit reduction → 4 qubit (16D)

    상수항 주의: frozen core energy가 nuclear repulsion보다 크다
    (R=1.55 Å에서 -7.82 vs +1.02). 둘 다 identity 항에 합산해야
    Table S2와 일치한다.
    """
    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {distance}",
        basis=basis,
    )
    problem = driver.run()

    transformer = FreezeCoreTransformer(
        freeze_core=True,
        remove_orbitals=[3, 4],
    )
    reduced = transformer.transform(problem)

    return _to_pauli_hamiltonian(reduced)


def _to_pauli_hamiltonian(problem) -> PauliHamiltonian:
    """ElectronicStructureProblem을 PauliHamiltonian으로 변환.

    모든 상수항(nuclear repulsion + frozen core energy 등)을
    identity Pauli string 계수에 합산한다. 논문 Table S1/S2 컨벤션.
    """
    fermionic_op = problem.hamiltonian.second_q_op()
    constants = problem.hamiltonian.constants
    total_constant = float(sum(constants.values()))

    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(fermionic_op)

    terms: list[tuple[str, float]] = []
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        if abs(coeff.imag) > 1e-10:
            raise ValueError(
                f"Non-real coefficient in Hermitian Hamiltonian: "
                f"{pauli} -> {coeff}"
            )
        label = str(pauli)
        c = float(coeff.real)
        if set(label) == {"I"}:  # identity string (II, IIII, ...)
            c += total_constant
        terms.append((label, c))

    return PauliHamiltonian(
        terms=terms,
        nuclear_repulsion=float(
            constants.get("nuclear_repulsion_energy", 0.0)
        ),
    )

import json
from pathlib import Path


# 고정된 Hamiltonian 데이터 위치
HAMILTONIAN_DATA_DIR = Path(__file__).parent / "data"


def save_hamiltonian(H: PauliHamiltonian, path: Path) -> None:
    """Hamiltonian을 JSON으로 저장.

    Qiskit의 fermion→Pauli 변환은 내부 합산 순서가 프로세스마다 달라져
    계수가 최대 ~150 ULP(상대오차 2e-14) 변한다. 물리적으로는 무해하지만,
    30차원 COBYLA는 이 차이를 수천 iteration에 걸쳐 증폭시켜 다른 local
    minimum으로 수렴한다. 재현 가능한 결과와 C++ cross-validation을 위해
    Hamiltonian을 한 번 생성해 고정한다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
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


def get_lih_hamiltonian(
    distance: float = 1.55,
    basis: str = "sto3g",
    use_cache: bool = True,
) -> PauliHamiltonian:
    """LiH Hamiltonian을 가져온다. 캐시가 있으면 읽고, 없으면 생성 후 저장.

    use_cache=False면 항상 새로 생성 (Qiskit 재현성 조사용).
    """
    cache_path = (
        HAMILTONIAN_DATA_DIR / f"lih_{basis}_R{distance:.4f}.json"
    )
    if use_cache and cache_path.exists():
        return load_hamiltonian(cache_path)

    H = build_lih_hamiltonian(distance=distance, basis=basis)
    if use_cache:
        save_hamiltonian(H, cache_path)
    return H