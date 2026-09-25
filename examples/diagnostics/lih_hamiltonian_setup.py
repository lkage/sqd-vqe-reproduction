"""[5주차 진단] LiH Hamiltonian 생성 설정을 탐색하고 Table S2와 비교.

LiH를 논문과 같은 4-qubit(16D), 100 Pauli string 형태로 만드는 설정을
찾기 위해 사용한 스크립트.

결론: FreezeCoreTransformer(freeze_core=True, remove_orbitals=[3, 4])
      + ParityMapper(num_particles=...) + 모든 상수항 합산

핵심 함정: active space를 쓰면 상수가 둘이다. nuclear_repulsion(+1.02)과
FreezeCoreTransformer(-7.82). H2 코드처럼 nuclear repulsion만 더하면
IIII가 약 7.8 Hartree 어긋난다.

Table S2 규격:
  LiH-qiskit, R = 1.55 Å, 4-qubit, 100 Pauli strings
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer


# 논문 Table S2 대표 계수 (R = 1.55 Å)
PAPER_S2_SAMPLE = {
    "IIII": -6.996701,
    "IIZI":  0.364916,
    "ZIII": -0.364913,
    "ZZII": -0.211881,
    "IIZZ": -0.211881,
    "IZZI":  0.113930,
    "ZIIZ":  0.113930,
    "ZZZZ":  0.084595,
}


def inspect(label, problem):
    """problem의 구조를 출력하고 Table S2와 비교."""
    print(f"\n{'='*60}")
    print(f"설정: {label}")
    print(f"{'='*60}")
    print(f"spin orbitals: {problem.num_spin_orbitals}")
    print(f"particles: {problem.num_particles}")

    constants = problem.hamiltonian.constants
    print("\nconstants dict:")
    for k, v in constants.items():
        print(f"  {k}: {v:+.6f}")
    print(f"  --- sum: {sum(constants.values()):+.6f}")

    fermionic_op = problem.hamiltonian.second_q_op()
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(fermionic_op)

    print(f"\nqubits: {qubit_op.num_qubits}")
    print(f"Pauli terms: {len(qubit_op)}  (논문: 100)")

    total_constant = sum(constants.values())
    coeffs = {}
    for pauli, c in zip(qubit_op.paulis, qubit_op.coeffs):
        lbl = str(pauli)
        val = float(c.real)
        if set(lbl) == {"I"}:
            val += total_constant
        coeffs[lbl] = val

    print("\nTable S2 대표 계수 비교:")
    print(f"  {'Pauli':<8} {'qiskit':>12} {'paper':>12} {'diff':>12}")
    for lbl, expected in PAPER_S2_SAMPLE.items():
        actual = coeffs.get(lbl)
        if actual is None:
            print(f"  {lbl:<8} {'(없음)':>12} {expected:>12.6f}")
        else:
            print(f"  {lbl:<8} {actual:>12.6f} {expected:>12.6f} "
                  f"{actual-expected:>12.2e}")

    return coeffs


def main():
    R = 1.55  # Table S2의 bond distance

    driver = PySCFDriver(atom=f"Li 0 0 0; H 0 0 {R}", basis="sto3g")
    problem = driver.run()

    print(f"[변환 전] spin orbitals: {problem.num_spin_orbitals}, "
          f"particles: {problem.num_particles}")

    transformer = FreezeCoreTransformer(
        freeze_core=True,
        remove_orbitals=[3, 4],
    )
    reduced = transformer.transform(problem)
    inspect("FreezeCore + remove_orbitals=[3,4]", reduced)


if __name__ == "__main__":
    main()