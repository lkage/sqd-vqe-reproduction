"""[5주차 진단] Qiskit Hamiltonian 생성의 비결정성 추적.

같은 seed인데 LiH VQE 결과가 실행마다 달라서 원인을 찾은 스크립트.

추적 전략: 파이프라인을 단계로 쪼개고 각 단계의 지문을 찍는다. 어느 단계부터
지문이 갈리는지 보면 원인 구간이 특정된다.

    분자 좌표 → PySCF SCF → 1e/2e 적분 → fermionic op → Pauli 계수

실측 결과:

    단계                      프로세스 간 일치
    nuclear repulsion         비트 단위 일치
    HF energy                 비트 단위 일치
    1e / 2e 적분              비트 단위 일치
    최종 Pauli 계수           16개 항이 최대 152 ULP 다름

즉 PySCF는 완전히 결정적이고, 문제는 Qiskit의 fermion→Pauli 변환에 있다.
같은 Pauli string끼리 계수를 합산할 때 내부 순회 순서가 프로세스마다 달라져
부동소수점 덧셈 순서가 바뀐다.

다른 16개는 전부 두 큐비트에만 작용하는 항(IIXX, XXII 등)이었다. 이런 항이
가장 많은 fermionic 항으로부터 기여를 받아 합산 횟수가 많고, 따라서 재배열
효과가 가장 많이 쌓인다. 4큐비트 전체에 걸친 항(ZZZZ 등)은 기여하는 항이
적어 영향이 없다.

절대 오차는 1e-17 이하이고 Table S2와는 항상 1e-6 이내로 일치한다. 물리적
으로는 완전히 무해하다. 그러나 30차원 COBYLA가 수천 번의 평가에 걸쳐 이를
증폭시켜 전혀 다른 local minimum으로 수렴하게 만든다.

대응: get_lih_hamiltonian()의 JSON 캐시. 한 번 생성해 고정한다.

주의 — 폐기된 가설 두 개를 기록해 둔다. 같은 길을 다시 가지 않기 위함이다.
  (a) "멀티스레드 BLAS의 합산 순서 때문" → 적분이 비트 단위로 같으므로 기각.
      BLAS는 적분 계산에 쓰이고 Pauli 변환에는 쓰이지 않는다.
  (b) "최종 항 나열 순서만 다른 것이니 terms.sort()로 해결" → 정렬 후에도
      계수 자체가 달랐으므로 기각. 문제는 나열 순서가 아니라 중간 합산 단계다.

사용법: 여러 번 실행해 canonical hash가 몇 종류 나오는지 확인

    for i in $(seq 1 8); do \
      uv run python examples/diagnostics/determinism_check.py | tail -6; \
      echo "---"; \
    done

계수 덤프를 results/determinism/에 저장하므로, 두 종류 이상 모이면
compare_determinism.py로 어느 항이 몇 ULP 다른지 볼 수 있다.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver

# 캐시된 get_lih_hamiltonian이 아니라 build_lih_hamiltonian을 쓴다.
# 캐시를 읽으면 당연히 매번 같은 값이 나와서 조사 자체가 성립하지 않는다.
from sqd_vqe.hamiltonian import build_lih_hamiltonian


def hash_floats(values) -> str:
    """부동소수점 배열의 비트 단위 해시.

    tobytes()는 IEEE 754 비트 패턴을 그대로 준다. 따라서 1 ULP 차이도
    완전히 다른 해시를 만든다 — 미세한 차이를 놓치지 않기 위한 선택.
    """
    arr = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main():
    R = 1.55

    # --- 단계별 지문 ---------------------------------------------------
    # 여기서 갈리면 PySCF 문제, 안 갈리면 그 아래 단계 문제로 범위가 좁혀진다.
    driver = PySCFDriver(atom=f"Li 0 0 0; H 0 0 {R}", basis="sto3g")
    problem = driver.run()

    print("=== 단계별 지문 ===")
    # 좌표 파싱과 단위 변환만 거치는 값. 여기가 갈리면 입력 자체가 다른 것.
    print(f"1. nuclear repulsion: "
          f"{problem.hamiltonian.nuclear_repulsion_energy:.15f}")
    # SCF 수렴 결과. DIIS 가속이 부동소수점 누적에 민감해 의심했던 지점이지만
    # 실제로는 비트 단위로 같았다.
    print(f"2. HF (reference) energy: {problem.reference_energy:.15f}")

    # --- 최종 Pauli 계수 ------------------------------------------------
    H = build_lih_hamiltonian(distance=R)

    raw_coeffs = [c for _, c in H.terms]
    labels_in_order = "".join(label for label, _ in H.terms)
    # canonical: 라벨과 계수를 함께, %.17g로. 라벨 순서 차이와 계수 값 차이를
    # 모두 잡는 지문이다. 셋을 따로 찍는 이유는 어느 쪽이 원인인지 구분하기
    # 위해서다 — 계수 해시만 보면 "순서가 다른 건지 값이 다른 건지" 알 수 없다.
    canonical = ";".join(
        f"{label}:{coeff:.17e}" for label, coeff in H.terms
    )
    canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    print(f"\n3. Pauli coeff hash: {hash_floats(raw_coeffs)}")
    print(f"4. Pauli label 순서 hash: "
          f"{hashlib.sha256(labels_in_order.encode()).hexdigest()[:16]}")
    print(f"5. canonical hash: {canonical_hash}")

    # 항 개수와 최소 계수 크기. 한때 "임계값 근처 항이 절단되었다 사라졌다
    # 하는 것"을 의심했으나, 항 개수가 항상 100이고 최소 계수가 7.67e-04로
    # 절단 임계(보통 1e-8)와 한참 멀어서 기각됐다.
    print(f"\n항 개수: {len(H.terms)}")
    abs_coeffs = sorted(abs(c) for _, c in H.terms)
    print(f"가장 작은 계수 5개: {[f'{c:.3e}' for c in abs_coeffs[:5]]}")
    print(f"IIII coefficient: {H.as_dict()['IIII']:.17f}")

    # --- 계수 덤프 저장 -------------------------------------------------
    # 파일명에 canonical hash를 넣으므로, 같은 결과는 덮어쓰지 않고 다른
    # 결과만 새 파일이 된다. 여러 번 돌리면 서로 다른 버전만 쌓인다.
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