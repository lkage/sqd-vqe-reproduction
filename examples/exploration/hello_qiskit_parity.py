"""[3주차 학습 흔적] ParityMapper + 2-qubit reduction으로 H2 Hamiltonian 확인.

JordanWigner(hello_qiskit.py)와 달리 2-qubit, 5 Pauli term이 나와
논문 Table S1과 형태가 일치한다.

차이를 만든 것은 한 줄이다:
    JordanWignerMapper()
    ParityMapper(num_particles=problem.num_particles)   ← 이쪽

num_particles를 넘기면 Qiskit이 입자 수 보존과 스핀 대칭을 이용해 두
큐비트를 제거한다. 4 qubit → 2 qubit, 15 term → 5 term.

상수항을 identity에 더한 버전도 함께 출력한다. 논문 Table S1의 II 계수가
nuclear repulsion을 포함한 값인지 아닌지가 처음에는 불분명했는데, 두 버전을
나란히 찍어 보고 "포함"임을 확정했다.

판정 근거였던 관찰: Table S1에서 R이 커져도 II가 0으로 수렴하지 않는다
(R=3에서 -0.633651). 핵 반발 1/R은 R이 커지면 0으로 가므로, II가 순수
전자 항이라면 R이 클 때 전자 에너지만 남아야 한다. 실제로는 더해진 쪽이
맞았다.
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper


# 논문 Table S1, R = 0.73 Å.
# 아래 출력과 나란히 찍어 눈으로 대조하기 위한 값이다. 자동 검증은
# tests/test_h2_hamiltonian.py가 담당한다.
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
    # H2는 상수가 이것 하나뿐이다. LiH는 active space 축소 때문에
    # frozen core energy가 추가로 생겨 둘이 된다 — 그 함정은 5주차에
    # 드러났다. (examples/diagnostics/lih_hamiltonian_setup.py 참조)
    nuclear_repulsion = problem.hamiltonian.nuclear_repulsion_energy

    # num_particles가 핵심이다. 이 인자가 two-qubit reduction을 발동시킨다.
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(fermionic_op)

    print("=== H2 Hamiltonian (Parity + 2-qubit reduction) ===")
    # 2가 나와야 한다. JW 버전은 4였다.
    print(f"Number of qubits: {qubit_op.num_qubits}")
    # 5가 나와야 한다. JW 버전은 15였다.
    print(f"Number of Pauli terms: {len(qubit_op)}")
    print(f"Nuclear repulsion (별도): {nuclear_repulsion:.6f} Hartree")

    # --- 버전 1: 전자 부분만 ---------------------------------------------
    # II가 -1.051287로 나온다. 논문의 -0.326386과 다르다.
    print("\nPauli decomposition (electronic part only):")
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        print(f"  {coeff.real:+.6f}  {pauli}")

    # --- 버전 2: II에 nuclear repulsion 합산 ------------------------------
    # -1.051287 + 0.724900 = -0.326386. 논문과 일치한다.
    # 두 버전을 나란히 보는 것이 이 스크립트의 핵심이다.
    print("\nWith nuclear repulsion added to II coefficient:")
    print(f"  {'Pauli':<6} {'qiskit':>12} {'paper':>12}")
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        c = coeff.real
        label = str(pauli)
        if label == "II":
            c += nuclear_repulsion
        expected = PAPER_S1_073.get(label)
        # 논문에 없는 항이 나오면 빈칸으로 둔다. 실제로는 5개가 정확히
        # 일치하지만, 설정을 바꿔가며 돌릴 때 이 칸이 비면 바로 눈에 띈다.
        exp_str = f"{expected:>12.6f}" if expected is not None else " " * 12
        print(f"  {label:<6} {c:>12.6f} {exp_str}")


if __name__ == "__main__":
    main()