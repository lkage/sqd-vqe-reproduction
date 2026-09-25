"""[5주차 진단] LiH 30차원 VQE의 수렴 거동 관찰.

30개 파라미터 공간에서 COBYLA가 global minimum을 얼마나 자주 찾는지,
multi-start가 몇 회 필요한지 실측한 스크립트.

결론:
  - 단일 실행 chemical accuracy 성공률 ~50%
  - 실패는 두 무리: -7.8632 근처(명확한 local minimum),
    -7.879 근처(얕은 갇힘)
  - multi-start 3회는 간혹 실패, 5회는 항상 성공 → 5회 채택

논문: LiH 실험 평균 242 ± 29.3 iteration (물리적 측정 사이클 단위,
      우리의 함수 평가 횟수와는 다른 단위)
"""

import numpy as np

from sqd_vqe.ansatz import LIH_NUM_PARAMS, lih_ansatz_state
from sqd_vqe.expectation import hamiltonian_matrix
from sqd_vqe.hamiltonian import get_lih_hamiltonian
from sqd_vqe.vqe import run_vqe, run_vqe_multistart


CHEMICAL_ACCURACY = 1.6e-3


def main():
    R = 1.55
    H = get_lih_hamiltonian(distance=R)
    M = hamiltonian_matrix(H)
    eigenvalues = np.linalg.eigvalsh(M)
    exact = float(eigenvalues[0])

    print(f"LiH at R = {R} Å")
    print(f"  Hamiltonian: {len(H.terms)} Pauli terms, "
          f"{M.shape[0]}x{M.shape[0]}")
    print(f"  Exact ground state energy: {exact:.6f} Hartree")
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
    print(f"Best error:  {min(errors):.2e}")
    print(f"Worst error: {max(errors):.2e}")
    print(f"Evaluations: mean={np.mean(evals):.0f}, "
          f"min={min(evals)}, max={max(evals)}")

    # multi-start 효과 측정
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
            print(f"{n_restarts:>9} {r.energy:>13.6f} {err:>11.2e}  "
                  f"{status} (best seed={r.best_restart_seed})")


if __name__ == "__main__":
    main()