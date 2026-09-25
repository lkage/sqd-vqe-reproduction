"""SQD-VQE reproduction package.

BLAS 스레드를 1개로 고정한다. 이유:
  1. 재현성: 멀티스레드 BLAS는 부동소수점 합산 순서가 실행마다 달라져
     1e-15 수준 차이를 만든다. 30차원 COBYLA는 2000 iteration에 걸쳐
     이 차이를 증폭시켜 같은 seed에서도 다른 결과를 낸다.
  2. 속도: 이 프로젝트의 행렬은 최대 16x16이라 스레드 오버헤드가
     계산 자체보다 크다.

numpy import 전에 설정해야 효과가 있으므로 패키지 최상단에 둔다.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")