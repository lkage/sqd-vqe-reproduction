"""R 스윕 기능 검증.

전체 스윕은 examples/figures/에서 실행. 여기서는 인프라 정상성만 확인.
"""

import numpy as np

from sqd_vqe.sweep import (
    PAPER_H2_DISTANCES,
    PAPER_LIH_DISTANCES,
    sweep_h2_distances,
)


CHEMICAL_ACCURACY = 1.6e-3


def test_paper_h2_distances():
    """논문 Figure S10의 21개 R 값."""
    assert len(PAPER_H2_DISTANCES) == 21
    assert PAPER_H2_DISTANCES[0] == 0.1
    assert PAPER_H2_DISTANCES[-1] == 3.0
    assert 0.73 in PAPER_H2_DISTANCES  # bonding length


def test_paper_lih_distances():
    """논문 Figure S11의 14개 R 값."""
    assert len(PAPER_LIH_DISTANCES) == 14
    assert PAPER_LIH_DISTANCES[0] == 0.1
    assert PAPER_LIH_DISTANCES[-1] == 3.4
    assert 1.55 in PAPER_LIH_DISTANCES  # bonding length


def test_sweep_returns_one_point_per_distance():
    distances = np.array([0.5, 1.0, 1.5])
    results = sweep_h2_distances(distances, n_restarts=2, seed=42)
    assert len(results) == 3
    assert [p.distance for p in results] == [0.5, 1.0, 1.5]


def test_sweep_achieves_chemical_accuracy():
    """3개 R 지점에서 모두 chemical accuracy 이내."""
    distances = np.array([0.5, 0.73, 1.5])
    results = sweep_h2_distances(distances, n_restarts=3, seed=42)

    failed = [p for p in results if p.error >= CHEMICAL_ACCURACY]
    assert not failed, "\n".join(
        f"R={p.distance}: err={p.error:.4e} Hartree" for p in failed
    )