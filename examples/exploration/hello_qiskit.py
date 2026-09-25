"""[3주차 학습 흔적] JordanWigner 매핑으로 H2 Hamiltonian 확인.

qubit reduction 없이 JW만 쓰면 4-qubit, 15 Pauli term이 나온다.
논문 Table S1(2-qubit, 5 term)과 형태가 다르다는 것을 확인하기 위한 스크립트.
비교: hello_qiskit_parity.py
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper


def main():
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.73",
        basis="sto3g",
    )
    problem = driver.run()

    print("=== H2 molecule ===")
    print(f"Number of spin-orbitals: {problem.num_spin_orbitals}")
    print(f"Number of electrons: {problem.num_particles}")
    print(f"Nuclear repulsion energy: "
          f"{problem.hamiltonian.nuclear_repulsion_energy:.6f} Hartree")

    fermionic_op = problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)

    print("\n=== Qubit Hamiltonian (Jordan-Wigner) ===")
    print(f"Number of qubits: {qubit_op.num_qubits}")
    print(f"Number of Pauli terms: {len(qubit_op)}")
    print("\nPauli decomposition:")
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        print(f"  {coeff.real:+.6f}  {pauli}")


if __name__ == "__main__":
    main()