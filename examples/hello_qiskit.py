"""환경 확인용 스크립트.

Qiskit Nature가 동작하고 H2 분자의 정보를 가져올 수 있는지 확인.
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper


def main():
    # H2 분자, R = 0.73 Å (bonding length)
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.73",
        basis="sto3g",
    )
    problem = driver.run()
    
    print("=== H2 molecule ===")
    print(f"Number of spin-orbitals: {problem.num_spin_orbitals}")
    print(f"Number of electrons: {problem.num_particles}")
    print(f"Nuclear repulsion energy: {problem.nuclear_repulsion_energy:.6f} Hartree")
    
    # Fermionic Hamiltonian → Qubit Hamiltonian
    fermionic_op = problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)
    
    print(f"\n=== Qubit Hamiltonian (Jordan-Wigner) ===")
    print(f"Number of qubits: {qubit_op.num_qubits}")
    print(f"Number of Pauli terms: {len(qubit_op)}")
    print(f"\nPauli decomposition:")
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        print(f"  {coeff.real:+.6f}  {pauli}")


if __name__ == "__main__":
    main()
