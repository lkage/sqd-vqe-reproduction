"""R 스윕 기능 검증.

전체 21개 R 스윕은 examples/h2_potential_curve.py에서 실행.
여기서는 sweep 인프라가 올바르게 동작하는지만 검증.
"""

import numpy as np

from sqd_vqe.sweep import PAPER_DISTANCES, sweep_distances


CHEMICAL_ACCURACY = 1.6e-3


def test_sweep_returns_one_point_per_distance():
    distances = np.array([0.5, 1.0, 1.5])
    results = sweep_distances(distances, seed=42)
    assert len(results) == 3
    assert [p.distance for p in results] == [0.5, 1.0, 1.5]


def test_sweep_small_range_achieves_chemical_accuracy():
    """3개 R 지점에서 모두 chemical accuracy 이내.

    21개 전체는 examples/에서 검증. 여기는 인프라 정상성 확인.
    """
    distances = np.array([0.5, 0.73, 1.5])
    results = sweep_distances(distances, seed=42)

    failed = [p for p in results if p.error >= CHEMICAL_ACCURACY]
    assert not failed, "\n".join(
        f"R={p.distance}: err={p.error:.4e} Hartree" for p in failed
    )


def test_paper_distances_has_21_points():
    """논문 Table S1과 동일한 21개 R 값."""
    assert len(PAPER_DISTANCES) == 21
    assert PAPER_DISTANCES[0] == 0.1
    assert PAPER_DISTANCES[-1] == 3.0
    assert 0.73 in PAPER_DISTANCES  # H2 bonding length