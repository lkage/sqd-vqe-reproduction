"""Qiskit SparsePauliOp(label).to_matrix()를 덤프한다.

C++ pauli_string() 빌더가 Qiskit과 같은 endian 컨벤션을 쓰는지 확인하기
위한 기준 데이터를 만든다. 손계산이 아니라 실제 Qiskit 출력이어야 한다.

라벨마다 두 가지 구조적 성질을 검증한다:
  1. 비영 원소 개수 == d  (d = 2^len(label)) — 행마다, 열마다 정확히 하나
  2. 모든 원소가 정확히 0, ±1, ±i — 반올림 오차 없음. 따라서 C++ 쪽도
     허용오차가 아니라 == 비교를 쓸 수 있다

확인된 컨벤션:
    to_matrix() == kron(P[0], P[1], ..., P[n-1])   — 왼쪽에서 오른쪽
    즉 라벨의 가장 왼쪽 문자가 가장 바깥 kron 인자이고, 가장 오른쪽 문자가
    행/열 인덱스의 최하위 비트에 작용한다.

주의: XX는 좌우 대칭이라 두 컨벤션 모두에서 같은 행렬이 나온다. endian
테스트의 판별력은 IZ와 ZI에서 나온다 — IZ는 diag(1,-1,1,-1)로 낮은 비트에서
부호가 바뀌고, ZI는 diag(1,1,-1,-1)로 높은 비트에서 바뀐다. XX만으로
테스트를 짜면 컨벤션이 뒤집혀도 통과한다.
"""

import qiskit
from qiskit.quantum_info import SparsePauliOp

LABELS = ["IZ", "ZI", "XX", "IX", "XY", "IZXY", "XYZI"]

# 허용되는 정확한 값
ALLOWED = {(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)}


def fmt(x: float) -> str:
    """float을 그대로 포맷. 여기 값들은 전부 정수여야 한다."""
    return f"{x:.1f}"


def main():
    print(f"// Generated with Qiskit {qiskit.__version__}")
    print("// SparsePauliOp(label).to_matrix(), little-endian:")
    print("//   leftmost character = highest qubit index")
    print("//   M = kron(P[0], P[1], ..., P[n-1]) left to right")
    print()

    all_ok = True

    for label in LABELS:
        M = SparsePauliOp.from_list([(label, 1.0)]).to_matrix()
        d = 2 ** len(label)

        entries = []
        exact = True
        for r in range(d):
            for c in range(d):
                v = M[r, c]
                if v == 0:
                    continue
                pair = (float(v.real), float(v.imag))
                if pair not in ALLOWED:
                    exact = False
                entries.append((r, c, pair[0], pair[1]))

        count_ok = len(entries) == d
        all_ok = all_ok and count_ok and exact

        print(f"// {label}: dim={d}x{d}, nonzeros={len(entries)} "
              f"(expected {d}) {'OK' if count_ok else 'MISMATCH'}, "
              f"exact={'yes' if exact else 'NO'}")
        print(f"const std::vector<PauliEntry> kPauli_{label} = {{")
        for r, c, re, im in entries:
            print(f"    {{{r:2d}, {c:2d}, {fmt(re):>5s}, {fmt(im):>5s}}},")
        print("};")
        print()

    print(f"// ALL CHECKS: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()