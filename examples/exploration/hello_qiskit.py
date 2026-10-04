"""[3주차 학습 흔적] JordanWigner 매핑으로 H2 Hamiltonian 확인.

qubit reduction 없이 JW만 쓰면 4-qubit, 15 Pauli term이 나온다.
논문 Table S1(2-qubit, 5 term)과 형태가 다르다는 것을 확인하기 위한
스크립트다. 비교: hello_qiskit_parity.py

이 스크립트의 역할은 "실패를 확인하는 것"이다. 3주차에 Python 환경을 잡고
처음 돌린 코드이고, 여기서 형태가 안 맞는다는 사실이 드러나야 다음 단계
(ParityMapper)로 가는 이유가 생긴다. 결과만 놓고 보면 쓸모없는 코드지만,
어떻게 답에 도달했는지를 보여주는 기록으로 남겨 둔다.

출력에서 주목할 것:
  - Number of qubits: 4
      H2/STO-3G는 spatial orbital 2개 = spin-orbital 4개다. JW는 spin-orbital
      하나를 qubit 하나에 그대로 대응시키므로 4 qubit이 된다.
  - Number of Pauli terms: 15
      논문의 5개보다 훨씬 많다. 축소 전이라 그렇다.
  - XXXX, YYYY, XXYY, YYXX의 계수가 모두 같다
      spin-up 섹터와 spin-down 섹터의 대칭성에서 온다. 이 대칭성이 바로
      two-qubit reduction이 이용하는 구조다.
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper


def main():
    # R=0.73 Å는 H2의 결합 길이. 논문 Table S1의 기준 거리이기도 하다.
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.73",
        basis="sto3g",
    )
    problem = driver.run()

    print("=== H2 molecule ===")
    print(f"Number of spin-orbitals: {problem.num_spin_orbitals}")
    # (1, 1)로 나온다 — alpha 1개, beta 1개. 전자 2개짜리 분자다.
    print(f"Number of electrons: {problem.num_particles}")
    # 1/R (Bohr 단위)에 해당한다. R=0.73 Å = 1.3795 Bohr이므로 약 0.7249.
    print(f"Nuclear repulsion energy: "
          f"{problem.hamiltonian.nuclear_repulsion_energy:.6f} Hartree")

    # second_q_op()는 생성/소멸 연산자로 쓰인 fermionic Hamiltonian을 준다.
    # 이것을 qubit 연산자로 바꾸는 것이 mapper의 일이다.
    fermionic_op = problem.hamiltonian.second_q_op()
    # JordanWignerMapper는 인자를 받지 않는다. 축소할 여지를 알려줄 통로가
    # 없으므로 축소도 일어나지 않는다. ParityMapper는 num_particles를
    # 받는데, 그 정보가 two-qubit reduction을 가능하게 한다.
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)

    print("\n=== Qubit Hamiltonian (Jordan-Wigner) ===")
    print(f"Number of qubits: {qubit_op.num_qubits}")
    print(f"Number of Pauli terms: {len(qubit_op)}")
    print("\nPauli decomposition:")
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        # Hermitian이므로 계수는 실수다. .real을 그냥 찍어도 안전하다.
        # 여기서 상수항(nuclear repulsion)은 더하지 않았다는 점에 주의 —
        # 논문 Table S1의 II 계수와 직접 비교하려면 더해야 한다.
        print(f"  {coeff.real:+.6f}  {pauli}")


if __name__ == "__main__":
    main()