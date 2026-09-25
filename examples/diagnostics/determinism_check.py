"""[5주차 진단] Qiskit Hamiltonian 생성의 비결정성 추적.

같은 seed인데 LiH VQE 결과가 실행마다 달라서 원인을 찾은 스크립트.

결론:
  - PySCF는 완전히 결정적 (HF energy, 1e/2e 적분 비트 단위 일치)
  - Qiskit의 fermion→Pauli 변환에서 같은 Pauli string끼리 계수를 합산할 때
    내부 순회 순서가 프로세스마다 달라짐
  - 100개 중 16개 항이 최대 152 ULP(상대오차 2e-14) 다르게 나옴
  - 다른 항은 전부 두 큐비트에만 작용하는 항(IIXX, XXII 등). 이런 항이
    가장 많은 fermionic 항으로부터 기여를 받아 합산 횟수가 많기 때문
  - 물리적으로 무해하지만 30차원 COBYLA가 증폭시켜 다른 local minimum으로 감
  - 대응: get_lih_hamiltonian()의 JSON 캐시

사용법: 여러 번 실행해 canonical hash가 몇 종류 나오는지 확인
    for i in $(seq 1 8); do uv run python examples/diagnostics/determinism_check.py | tail -6; echo "---"; done

계수 덤프를 results/determinism/에 저장하므로, 두 종류 이상 모이면
compare_determinism.py로 어느 항이 몇 ULP 다른지 볼 수 있다.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver

from sqd_vqe.hamiltonian import build_lih_hamiltonian


def hash_floats(values) -> str:
    """부동소수점 배열의 비트 단위 해시."""
    arr = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main():
    R = 1.55

    # 단계별 지문 — 어디서 갈라지는지 확인
    driver = PySCFDriver(atom=f"Li 0 0 0; H 0 0 {R}", basis="sto3g")
    problem = driver.run()

    print("=== 단계별 지문 ===")
    print(f"1. nuclear repulsion: "
          f"{problem.hamiltonian.nuclear_repulsion_energy:.15f}")
    print(f"2. HF (reference) energy: {problem.reference_energy:.15f}")

    # 최종 Pauli 계수
    H = build_lih_hamiltonian(distance=R)

    raw_coeffs = [c for _, c in H.terms]
    labels_in_order = "".join(label for label, _ in H.terms)
    canonical = ";".join(
        f"{label}:{coeff:.17e}" for label, coeff in H.terms
    )
    canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    print(f"\n3. Pauli coeff hash: {hash_floats(raw_coeffs)}")
    print(f"4. Pauli label 순서 hash: "
          f"{hashlib.sha256(labels_in_order.encode()).hexdigest()[:16]}")
    print(f"5. canonical hash: {canonical_hash}")

    print(f"\n항 개수: {len(H.terms)}")
    abs_coeffs = sorted(abs(c) for _, c in H.terms)
    print(f"가장 작은 계수 5개: {[f'{c:.3e}' for c in abs_coeffs[:5]]}")
    print(f"IIII coefficient: {H.as_dict()['IIII']:.17f}")

    # 계수 덤프 저장 (compare_determinism.py가 사용)
    out_dir = Path("results/determinism")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / f"lih_{canonical_hash}.json"

    if not dump_path.exists():
        payload = {label: float(coeff) for label, coeff in H.terms}
        dump_path.write_text(json.dumps(payload, indent=1))
        print(f"\n저장: {dump_path}")
    else:
        print(f"\n이미 존재: {dump_path}")


if __name__ == "__main__":
    main()