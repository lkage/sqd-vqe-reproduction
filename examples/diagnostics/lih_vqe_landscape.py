"""[5주차 진단] LiH 30차원 VQE의 수렴 거동 관찰.

30개 파라미터 공간에서 COBYLA가 global minimum을 얼마나 자주 찾는지,
multi-start가 몇 회 필요한지 실측한 스크립트.

왜 측정이 필요했나: H2(6차원)는 단일 실행으로 거의 항상 수렴했다. LiH로
넘어가면서 같은 코드가 같은 seed로도 들쭉날쭉한 값을 내기 시작했는데,
그것이 (a) 구현 버그인지 (b) 고차원 비볼록 최적화의 정상적인 난이도인지
구분해야 했다. 20개 seed를 돌려 분포를 보는 것이 가장 빠른 판별법이었다.

결론:
  - 단일 실행 chemical accuracy 성공률 ~50%
  - 실패는 두 무리로 갈린다: -7.8632 근처(명확한 local minimum),
    -7.879 근처(얕게 갇혔거나 수렴이 덜 된 것)
  - gap이 0.118 Ha로 크다. 즉 실패한 지점이 excited state가 아니라
    파라미터 공간 안의 진짜 local minimum이다 — 버그가 아니라는 근거
  - multi-start 3회는 간혹 실패, 5회는 항상 성공 → 5회 채택

논문도 각 interatomic distance에서 실험을 여러 번 수행하고 에너지 차이가
최소인 결과를 채택했다 (Fig. 4B 캡션). 같은 전략이다.

논문의 LiH 평균 242 ± 29.3 iteration과 여기 출력된 evaluation 수(평균
1700 정도)는 직접 비교할 수 없다. 논문은 물리적 측정 사이클 단위이고
SciPy COBYLA의 maxiter는 함수 평가 횟수다.
"""

import numpy as np

from sqd_vqe.ansatz import LIH_NUM_PARAMS, lih_ansatz_state
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import get_lih_hamiltonian
from sqd_vqe.vqe import run_vqe, run_vqe_multistart


CHEMICAL_ACCURACY = 1.6e-3


def main():
    R = 1.55
    # 캐시된 Hamiltonian을 쓴다. 캐시 없이 돌리면 Qiskit의 ULP 비결정성
    # 때문에 실행마다 다른 결과가 나와서, local minimum 통계와 비결정성이
    # 섞여 버린다. 이 스크립트는 전자만 재야 한다.
    H = get_lih_hamiltonian(distance=R)
    M = hamiltonian_matrix(H)
    eigenvalues = np.linalg.eigvalsh(M)
    exact = float(eigenvalues[0])

    print(f"LiH at R = {R} Å")
    print(f"  Hamiltonian: {len(H.terms)} Pauli terms, "
          f"{M.shape[0]}x{M.shape[0]}")
    print(f"  Exact ground state energy: {exact:.6f} Hartree")
    # gap을 함께 찍는 이유: 실패 지점이 첫 excited state 근처라면 ansatz가
    # 엉뚱한 고유상태로 수렴한 것이고, gap보다 한참 아래라면 고유상태가
    # 아닌 파라미터 공간의 local minimum이다. 둘은 원인이 완전히 다르다.
    print(f"  Next eigenvalue (gap):     {eigenvalues[1]:.6f} "
          f"(gap = {eigenvalues[1]-exact:.4f})")
    print(f"  Ansatz params: {LIH_NUM_PARAMS}")

    print(f"\n{'seed':>5} {'energy':>13} {'error':>11} {'evals':>7}  status")
    print("-" * 55)

    results = []
    for seed in range(20):
        r = run_vqe(
            H, ansatz=lih_ansatz_state, seed=seed,
            n_params=LIH_NUM_PARAMS, max_iter=3000,
        )
        err = abs(r.energy - exact)
        status = "OK" if err < CHEMICAL_ACCURACY else "local min?"
        # max_iter에 닿은 실행은 "수렴해서 멈춘" 것이 아니라 "시간이 다 돼
        # 멈춘" 것이다. 성공/실패와 별개의 정보이므로 따로 표시한다.
        if r.n_iterations >= 3000:
            status = "hit max_iter"
        print(f"{seed:>5} {r.energy:>13.6f} {err:>11.2e} "
              f"{r.n_iterations:>7}  {status}")
        results.append((seed, r.energy, err, r.n_iterations))

    errors = [e for _, _, e, _ in results]
    evals = [i for _, _, _, i in results]
    n_ok = sum(e < CHEMICAL_ACCURACY for e in errors)

    print("-" * 55)
    print(f"Chemical accuracy 달성: {n_ok}/20")
    # best와 worst를 함께 본다. best가 충분히 정확하면 ansatz와 Hamiltonian은
    # 옳고 문제는 최적화뿐이라는 뜻이다. best마저 나쁘면 구현을 의심해야 한다.
    print(f"Best error:  {min(errors):.2e}")
    print(f"Worst error: {max(errors):.2e}")
    print(f"Evaluations: mean={np.mean(evals):.0f}, "
          f"min={min(evals)}, max={max(evals)}")

    # --- multi-start 효과 측정 -------------------------------------------
    # restart 횟수를 몇으로 할지 감이 아니라 측정으로 정하기 위한 부분이다.
    # trial_seed를 바꿔 여러 번 보는 이유: restart 3회가 한 번 성공했다고
    # 3회면 충분하다고 결론 내릴 수 없다. 서로 다른 시작점 묶음에서
    # 반복 확인해야 한다.
    print(f"\n{'='*55}")
    print("Multi-start 효과 측정")
    print(f"{'='*55}")
    print(f"{'restarts':>9} {'energy':>13} {'error':>11}  status")
    print("-" * 55)

    for n_restarts in [3, 5, 10]:
        for trial_seed in [0, 100, 200]:
            r = run_vqe_multistart(
                H, ansatz=lih_ansatz_state, n_restarts=n_restarts,
                seed=trial_seed, n_params=LIH_NUM_PARAMS, max_iter=3000,
            )
            err = abs(r.energy - exact)
            status = "OK" if err < CHEMICAL_ACCURACY else "FAIL"
            # best seed를 찍어 두면 어느 시작점이 성공했는지 추적할 수 있다.
            # 10회에서만 좋은 값이 나왔다면 그 seed가 3, 5회 범위 밖이라는
            # 뜻이고, 그것이 곧 restart를 늘린 효과다.
            print(f"{n_restarts:>9} {r.energy:>13.6f} {err:>11.2e}  "
                  f"{status} (best seed={r.best_restart_seed})")


if __name__ == "__main__":
    main()