"""SQD-VQE reproduction package.

BLAS 스레드를 1개로 고정한다. 이유:

  1. 재현성: 멀티스레드 BLAS는 작업을 스레드에 쪼개 나눈 뒤 합치므로
     부동소수점 합산 순서가 실행마다 달라진다. 1e-15 수준의 차이지만,
     30차원 COBYLA는 수천 번의 평가에 걸쳐 이를 증폭시켜 같은 seed에서도
     다른 local minimum으로 수렴하게 만든다.

  2. 속도: 이 프로젝트의 행렬은 최대 16x16이다. 스레드를 띄우고 결과를
     모으는 오버헤드가 계산 자체보다 크다. 고정하는 편이 오히려 빠르다.

numpy import 전에 설정해야 효과가 있다. BLAS 라이브러리는 로드 시점에
환경변수를 읽고 그 뒤로는 무시하기 때문이다. 그래서 패키지 최상단에 둔다 —
sqd_vqe의 어떤 모듈을 import하든 이 파일이 먼저 실행된다.

setdefault를 쓰는 이유: 사용자가 이미 값을 지정했다면 존중한다. 벤치마크
등에서 의도적으로 스레드를 늘리고 싶을 수 있다.
"""

import os

# 주요 BLAS 구현체마다 변수 이름이 다르다. 어느 것이 깔려 있을지 모르므로
# 전부 설정한다.
#   OMP_NUM_THREADS        - OpenMP 일반
#   OPENBLAS_NUM_THREADS   - OpenBLAS
#   MKL_NUM_THREADS        - Intel MKL
#   NUMEXPR_NUM_THREADS    - numexpr
#   VECLIB_MAXIMUM_THREADS - macOS Accelerate
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")