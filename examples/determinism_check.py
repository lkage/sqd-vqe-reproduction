"""PySCF Hamiltonian 생성의 비결정성 원인 추적.

프로세스 간 Hamiltonian hash가 달라지는 지점을 좁힌다.
"""

import hashlib

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver

from sqd_vqe.hamiltonian import build_lih_hamiltonian


def hash_floats(values) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main():
    R = 1.55

    # A. SCF 단계까지의 출력을 직접 확인
    driver = PySCFDriver(atom=f"Li 0 0 0; H 0 0 {R}", basis="sto3g")
    problem = driver.run()

    print("=== PySCF driver 출력 ===")
    print(f"HF energy:              "
          f"{problem.reference_energy:.15f}")
    print(f"nuclear repulsion:      "
          f"{problem.hamiltonian.nuclear_repulsion_energy:.15f}")

    # one/two-electron 적분 해시
    integrals = problem.hamiltonian.electronic_integrals
    alpha_1body = np.asarray(integrals.alpha["+-"])
    print(f"1-electron integral hash: {hash_floats(alpha_1body.ravel())}")

    if "++--" in integrals.alpha:
        alpha_2body = np.asarray(integrals.alpha["++--"])
        print(f"2-electron integral hash: {hash_floats(alpha_2body.ravel())}")

    # 최종 Pauli 계수 — 정렬 전/후를 모두 확인
    H = build_lih_hamiltonian(distance=R)

    raw_coeffs = [c for _, c in H.terms]
    sorted_terms = sorted(H.terms, key=lambda t: t[0])
    sorted_coeffs = [c for _, c in sorted_terms]
    labels_in_order = "".join(label for label, _ in H.terms)

    print(f"\nPauli coeff hash (원래 순서):  {hash_floats(raw_coeffs)}")
    print(f"Pauli coeff hash (정렬 후):    {hash_floats(sorted_coeffs)}")
    print(f"Pauli label 순서 hash:         "
          f"{hashlib.sha256(labels_in_order.encode()).hexdigest()[:16]}")
    print(f"IIII coefficient: {H.as_dict()['IIII']:.17f}")

        # 항 개수와 계수 분포 확인
    print(f"\n항 개수: {len(H.terms)}")

    abs_coeffs = sorted(abs(c) for _, c in H.terms)
    print(f"가장 작은 계수 5개: "
          f"{[f'{c:.3e}' for c in abs_coeffs[:5]]}")

    # 정렬된 label 목록의 처음/끝
    labels = sorted(label for label, _ in H.terms)
    print(f"첫 5개 label: {labels[:5]}")
    print(f"끝 5개 label: {labels[-5:]}")

    # 정렬된 (label, coeff) 전체를 한 줄 문자열로 → 완전한 지문
    canonical = ";".join(
        f"{label}:{coeff:.17e}"
        for label, coeff in sorted(H.terms, key=lambda t: t[0])
    )
    print(f"canonical hash: "
          f"{hashlib.sha256(canonical.encode()).hexdigest()[:16]}")
        # 계수를 파일로 저장 — 여러 실행을 비교하기 위해
    import json
    from pathlib import Path

    out_dir = Path("results/determinism")
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    dump_path = out_dir / f"lih_{canonical_hash}.json"

    if not dump_path.exists():
        payload = {
            label: float(coeff)
            for label, coeff in sorted(H.terms, key=lambda t: t[0])
        }
        dump_path.write_text(json.dumps(payload, indent=1))
        print(f"저장: {dump_path}")
    else:
        print(f"이미 존재: {dump_path}")


if __name__ == "__main__":
    main()