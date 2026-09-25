"""LiH Hamiltonian 생성 설정을 탐색하고 Table S2와 비교.

Table S2 규격:
  LiH-qiskit, R = 1.55 Å, 4-qubit, 100 Pauli strings
  IIII = -6.996701, IIZI = 0.364916, ZIII = -0.364913,
  ZZII = -0.211881, IIZZ = -0.211881
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer


# Table S2 대표 계수 (R = 1.55 Å)
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
    """problem의 구조를 출력."""
    print(f"\n{'='*60}")
    print(f"설정: {label}")
    print(f"{'='*60}")
    print(f"spin orbitals: {problem.num_spin_orbitals}")
    print(f"particles: {problem.num_particles}")

    # 상수항 전부 확인 (핵심!)
    constants = problem.hamiltonian.constants
    print(f"\nconstants dict:")
    for k, v in constants.items():
        print(f"  {k}: {v:+.6f}")
    print(f"  --- sum: {sum(constants.values()):+.6f}")

    fermionic_op = problem.hamiltonian.second_q_op()
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(fermionic_op)

    print(f"\nqubits: {qubit_op.num_qubits}")
    print(f"Pauli terms: {len(qubit_op)}  (논문: 100)")

    # 상수 전부 더한 버전으로 비교
    total_constant = sum(constants.values())
    coeffs = {}
    for pauli, c in zip(qubit_op.paulis, qubit_op.coeffs):
        lbl = str(pauli)
        val = float(c.real)
        if set(lbl) == {"I"}:  # identity string
            val += total_constant
        coeffs[lbl] = val

    print(f"\nTable S2 대표 계수 비교:")
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

    # 설정 A: 표준 Qiskit LiH 레시피
    #   freeze_core로 Li 1s 제거 + 2px, 2py 제거 (orbital index 3, 4)
    driver = PySCFDriver(atom=f"Li 0 0 0; H 0 0 {R}", basis="sto3g")
    problem = driver.run()

    print(f"\n[변환 전] spin orbitals: {problem.num_spin_orbitals}, "
          f"particles: {problem.num_particles}")

    transformer = FreezeCoreTransformer(
        freeze_core=True,
        remove_orbitals=[3, 4],
    )
    reduced = transformer.transform(problem)
    inspect("FreezeCore + remove_orbitals=[3,4]", reduced)


if __name__ == "__main__":
    main()