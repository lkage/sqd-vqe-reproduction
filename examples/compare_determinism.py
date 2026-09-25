"""두 실행에서 나온 LiH 계수를 비트 단위로 비교.

어느 Pauli 항이 몇 ULP 다른지 찾는다.
"""

import json
import math
from pathlib import Path

import numpy as np


def ulp_distance(a: float, b: float) -> int:
    """두 float 사이의 ULP 거리."""
    if a == b:
        return 0
    ia = np.frombuffer(np.float64(a).tobytes(), dtype=np.int64)[0]
    ib = np.frombuffer(np.float64(b).tobytes(), dtype=np.int64)[0]
    return int(abs(ia - ib))


def main():
    files = sorted(Path("results/determinism").glob("lih_*.json"))
    if len(files) < 2:
        print(f"비교하려면 2개 이상 필요. 현재 {len(files)}개.")
        return

    print(f"비교: {files[0].name}  vs  {files[1].name}\n")

    a = json.loads(files[0].read_text())
    b = json.loads(files[1].read_text())

    assert set(a) == set(b), "label 집합이 다름"

    diffs = []
    for label in sorted(a):
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
            print(f"{label:<8} {va:>22.17e} {vb:>22.17e} {ulp:>6}")


if __name__ == "__main__":
    main()