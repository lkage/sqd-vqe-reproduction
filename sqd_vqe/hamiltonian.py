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

    fermionic_op = problem.hamiltonian.second_q_op()
    nuclear_repulsion = float(problem.hamiltonian.nuclear_repulsion_energy)

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
        if label == "II":
            c += nuclear_repulsion
        terms.append((label, c))

    return PauliHamiltonian(
        terms=terms,
        nuclear_repulsion=nuclear_repulsion,
    )