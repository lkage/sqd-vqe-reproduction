"""[5주차 진단] 두 실행에서 나온 LiH 계수를 비트 단위로 비교.

determinism_check.py를 여러 번 실행해 results/determinism/에 두 종류
이상의 JSON이 모인 뒤 사용한다. 어느 Pauli 항이 몇 ULP 다른지 보여준다.

ULP(unit in the last place)로 재는 이유: 절대 차이는 값의 크기에 묻힌다.
IIII처럼 -7 근처인 항과 IIXX처럼 7.67e-04인 항은 같은 1 ULP라도 절대
차이가 10^4배 난다. ULP는 "마지막 비트 몇 칸 차이인가"를 직접 세므로
크기에 무관하게 비교할 수 있다.

실측 결과 (5주차):
  100개 중 16개 항이 다름. 최대 152 ULP (IIXX, IIYY).
  전부 두 큐비트에만 작용하는 항이었다 — 이 패턴이 원인을 지목했다.
  이런 항은 가장 많은 fermionic 항으로부터 기여를 받으므로 합산 횟수가
  많고, 따라서 순회 순서 변동의 효과가 가장 많이 누적된다.

또 하나 눈에 띈 것: 값이 쌍으로 움직였다. IIXX와 IIYY가 부호만 다르고
크기가 같으며 ULP도 똑같이 152였고, XXII와 YYII도 마찬가지로 24였다.
같은 fermionic 항에서 함께 생성되는 Pauli 쌍이라는 뜻이다.
"""

import json
from pathlib import Path

import numpy as np


def ulp_distance(a: float, b: float) -> int:
    """두 float 사이의 ULP 거리.

    IEEE 754 double의 비트 패턴을 int64로 재해석하면, 같은 부호 영역
    안에서는 비트 패턴의 차이가 그대로 '사이에 끼어 있는 표현 가능한 값의
    개수'가 된다. 지수부가 상위 비트에 있고 가수부가 하위 비트에 있는
    배치 덕분이다.

    부호가 다른 경우에는 이 계산이 의미를 잃지만, 여기서 비교하는 두 값은
    같은 계산의 미세한 변동이라 부호가 갈릴 일이 없다.
    """
    if a == b:
        return 0
    ia = np.frombuffer(np.float64(a).tobytes(), dtype=np.int64)[0]
    ib = np.frombuffer(np.float64(b).tobytes(), dtype=np.int64)[0]
    return int(abs(ia - ib))


def main():
    files = sorted(Path("results/determinism").glob("lih_*.json"))
    if len(files) < 2:
        # 비결정성이 매번 나타나지는 않는다. 실측상 8회 중 2종류 정도였다.
        # 그래서 여러 번 돌려야 비교할 재료가 모인다.
        print(f"비교하려면 2개 이상 필요. 현재 {len(files)}개.")
        print("determinism_check.py를 여러 번 실행하세요:")
        print("  for i in $(seq 1 10); do "
              "uv run python examples/diagnostics/determinism_check.py "
              "> /dev/null; done")
        return

    print(f"비교: {files[0].name}  vs  {files[1].name}\n")

    a = json.loads(files[0].read_text())
    b = json.loads(files[1].read_text())

    # 라벨 집합이 다르면 항이 생기거나 사라진 것이고, 그건 계수 변동과는
    # 전혀 다른 문제다(절단 임계 근처 항 등). 먼저 배제하고 들어간다.
    assert set(a) == set(b), "label 집합이 다름"

    diffs = []
    for label in sorted(a):
        # == 로 비교한다. np.isclose 같은 허용오차 비교를 쓰면 바로 이
        # 스크립트가 찾으려는 미세 차이를 놓친다.
        if a[label] != b[label]:
            diffs.append((
                label, a[label], b[label],
                ulp_distance(a[label], b[label]),
            ))

    print(f"전체 {len(a)}개 중 {len(diffs)}개 항이 다름\n")
    if diffs:
        print(f"{'label':<8} {'run A':>22} {'run B':>22} {'ULP':>6}")
        print("-" * 62)
        for label, va, vb, ulp in diffs:
            # %.17e로 찍는다. double의 왕복 변환을 보장하는 자릿수라
            # 마지막 비트까지 눈으로 확인할 수 있다.
            print(f"{label:<8} {va:>22.17e} {vb:>22.17e} {ulp:>6}")


if __name__ == "__main__":
    main()