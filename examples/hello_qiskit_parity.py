"""H2 Hamiltonian을 ParityMapper + 2-qubit reduction으로 생성.

JordanWigner 버전(hello_qiskit.py)과 비교하여 qubit 수와
Pauli term 구성이 어떻게 달라지는지 확인.
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper


def main():
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.73",
        basis="sto3g",
    )
    problem = driver.run()

    fermionic_op = problem.hamiltonian.second_q_op()
    nuclear_repulsion = problem.hamiltonian.nuclear_repulsion_energy

    # ParityMapper에 num_particles를 전달하면 2-qubit reduction이 자동 적용됨
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(fermionic_op)

    print("=== H2 Hamiltonian (Parity + 2-qubit reduction) ===")
    print(f"Number of qubits: {qubit_op.num_qubits}")
    print(f"Number of Pauli terms: {len(qubit_op)}")
    print(f"Nuclear repulsion (별도): {nuclear_repulsion:.6f} Hartree")

    print("\nPauli decomposition (electronic part only):")
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        print(f"  {coeff.real:+.6f}  {pauli}")

    # II 항에 nuclear repulsion 더한 'total H' 형태도 같이 출력
    print("\nWith nuclear repulsion added to II coefficient:")
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        c = coeff.real
        label = str(pauli)
        if label == "II":
            c += nuclear_repulsion
        print(f"  {c:+.6f}  {label}")


if __name__ == "__main__":
    main()
