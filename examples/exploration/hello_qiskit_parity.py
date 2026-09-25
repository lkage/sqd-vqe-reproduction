"""[3주차 학습 흔적] ParityMapper + 2-qubit reduction으로 H2 Hamiltonian 확인.

JordanWigner(hello_qiskit.py)와 달리 2-qubit, 5 Pauli term이 나와
논문 Table S1과 형태가 일치한다.

상수항을 identity에 더한 버전도 함께 출력해서, 논문 Table S1의 II 계수가
nuclear repulsion을 포함한 값임을 확인한다.
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper


# 논문 Table S1, R = 0.73 Å
PAPER_S1_073 = {
    "II": -0.326386,
    "IZ":  0.401061,
    "ZI": -0.401061,
    "ZZ": -0.011314,
    "XX":  0.180653,
}


def main():
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.73",
        basis="sto3g",
    )
    problem = driver.run()

    fermionic_op = problem.hamiltonian.second_q_op()
    nuclear_repulsion = problem.hamiltonian.nuclear_repulsion_energy

    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(fermionic_op)

    print("=== H2 Hamiltonian (Parity + 2-qubit reduction) ===")
    print(f"Number of qubits: {qubit_op.num_qubits}")
    print(f"Number of Pauli terms: {len(qubit_op)}")
    print(f"Nuclear repulsion (별도): {nuclear_repulsion:.6f} Hartree")

    print("\nPauli decomposition (electronic part only):")
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        print(f"  {coeff.real:+.6f}  {pauli}")

    print("\nWith nuclear repulsion added to II coefficient:")
    print(f"  {'Pauli':<6} {'qiskit':>12} {'paper':>12}")
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        c = coeff.real
        label = str(pauli)
        if label == "II":
            c += nuclear_repulsion
        expected = PAPER_S1_073.get(label)
        exp_str = f"{expected:>12.6f}" if expected is not None else " " * 12
        print(f"  {label:<6} {c:>12.6f} {exp_str}")


if __name__ == "__main__":
    main()