"""[5주차 진단] LiH Hamiltonian 생성 설정을 탐색하고 Table S2와 비교.

LiH를 논문과 같은 4-qubit(16D), 100 Pauli string 형태로 만드는 설정을
찾기 위해 사용한 스크립트.

출발점의 문제: 아무 변환 없이 JW 매핑하면 12-qubit, Pauli string 수백 개가
나온다. 논문은 4-qubit, 100개다. 두 단계 축소가 필요하다.

결론: FreezeCoreTransformer(freeze_core=True, remove_orbitals=[3, 4])
      + ParityMapper(num_particles=...) + 모든 상수항 합산

왜 이 설정인가:
  LiH/STO-3G는 spatial orbital 6개(Li의 1s, 2s, 2px, 2py, 2pz + H의 1s),
  전자 4개다. 결합이 z축 방향이므로 2px와 2py는 ground state 기술에
  기여하지 않는다. Li 1s는 core라 동결해도 된다.
    6 orbital - (1s 동결) - (2px, 2py 제거) = 3 spatial orbital
    = 6 spin-orbital → ParityMapper 2-qubit reduction → 4 qubit (16D)

틀린 설정: ActiveSpaceTransformer(4, 3)
  "4 전자, 3 orbital"이라 qubit 수는 맞게 나오지만 core를 얼리지 않아
  활성 전자가 4개가 된다. 물리가 다르고 계수가 안 맞는다.
  올바른 쪽은 활성 전자 2개다 — 출력의 particles가 (1, 1)인지 확인할 것.

핵심 함정: active space를 쓰면 상수가 둘이다.
  nuclear_repulsion(+1.02)과 FreezeCoreTransformer(-7.82).
  H2 코드처럼 nuclear repulsion만 더하면 IIII가 약 7.8 Hartree 어긋난다.
  H2는 상수가 하나뿐이라 이 함정이 드러나지 않았다.

Table S2 규격:
  LiH-qiskit, R = 1.55 Å, 4-qubit, 100 Pauli strings
  (Table S4는 openfermion 기준 1.51 Å로 다른 데이터다. 혼동 주의)
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer


# 논문 Table S2 대표 계수 (R = 1.55 Å).
# 100개를 다 적을 필요는 없다. 크기가 다양한 항을 골라 두면 설정이 맞는지
# 판정하기에 충분하다 — 설정이 틀리면 전부 틀리지 일부만 틀리지 않는다.
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
    """problem의 구조를 출력하고 Table S2와 비교.

    설정을 바꿔가며 여러 번 호출할 수 있도록 함수로 분리했다. 당시에는
    ActiveSpaceTransformer 등 여러 조합을 나란히 찍어 보며 좁혔다.
    """
    print(f"\n{'='*60}")
    print(f"설정: {label}")
    print(f"{'='*60}")
    # 6이어야 한다(= 3 spatial orbital). 12면 변환이 적용되지 않은 것.
    print(f"spin orbitals: {problem.num_spin_orbitals}")
    # (1, 1)이어야 한다 — 활성 전자가 알파 1개, 베타 1개.
    # (2, 2)면 core를 얼리지 않은 것이다.
    print(f"particles: {problem.num_particles}")

    # 여기가 진단의 핵심이다. 키가 몇 개인지 반드시 눈으로 확인할 것.
    constants = problem.hamiltonian.constants
    print("\nconstants dict:")
    for k, v in constants.items():
        print(f"  {k}: {v:+.6f}")
    print(f"  --- sum: {sum(constants.values()):+.6f}")

    fermionic_op = problem.hamiltonian.second_q_op()
    # num_particles를 넘겨야 two-qubit reduction이 적용된다. 빼먹으면
    # 6 qubit이 나온다.
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(fermionic_op)

    print(f"\nqubits: {qubit_op.num_qubits}")
    print(f"Pauli terms: {len(qubit_op)}  (논문: 100)")

    total_constant = sum(constants.values())
    coeffs = {}
    for pauli, c in zip(qubit_op.paulis, qubit_op.coeffs):
        lbl = str(pauli)
        val = float(c.real)
        # identity string에 모든 상수를 합산한다. 라벨 길이가 분자마다
        # 다르므로 "IIII" 문자열 비교 대신 집합으로 판정한다.
        if set(lbl) == {"I"}:
            val += total_constant
        coeffs[lbl] = val

    print("\nTable S2 대표 계수 비교:")
    print(f"  {'Pauli':<8} {'qiskit':>12} {'paper':>12} {'diff':>12}")
    for lbl, expected in PAPER_S2_SAMPLE.items():
        actual = coeffs.get(lbl)
        if actual is None:
            # 항이 아예 없으면 축소 설정이 틀린 것이다. 계수 차이와는
            # 다른 종류의 실패이므로 구분해서 표시한다.
            print(f"  {lbl:<8} {'(없음)':>12} {expected:>12.6f}")
        else:
            # 논문이 소수 6자리까지만 주므로 diff가 1e-6 수준이면 일치다.
            print(f"  {lbl:<8} {actual:>12.6f} {expected:>12.6f} "
                  f"{actual-expected:>12.2e}")

    return coeffs


def main():
    R = 1.55  # Table S2의 bond distance. 1.51(Table S4)과 혼동하지 말 것

    driver = PySCFDriver(atom=f"Li 0 0 0; H 0 0 {R}", basis="sto3g")
    problem = driver.run()

    # 변환 전: 12 spin-orbital, (2, 2) particles. 이 상태로 매핑하면
    # 12 qubit이 나와 논문과 비교가 불가능하다.
    print(f"[변환 전] spin orbitals: {problem.num_spin_orbitals}, "
          f"particles: {problem.num_particles}")

    # remove_orbitals의 인덱스 [3, 4]는 PySCF의 orbital 정렬 순서 기준이다.
    # 다른 분자나 기저로 바꾸면 이 인덱스가 달라질 수 있으므로, 그때는
    # 출력의 qubit 수와 term 수로 다시 확인해야 한다.
    transformer = FreezeCoreTransformer(
        freeze_core=True,
        remove_orbitals=[3, 4],
    )
    reduced = transformer.transform(problem)
    inspect("FreezeCore + remove_orbitals=[3,4]", reduced)


if __name__ == "__main__":
    main()