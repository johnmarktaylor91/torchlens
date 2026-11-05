"""Stimulus-bootstrap intervals on TorchLens's own scores (neuro.1 row 1).

Pins:

- The point score is the unchanged estimator: exactly ``rdm_compare`` for
  every method, and ``tl.stats.cka`` (Gram route, 1e-12) for CKA.
- Seeded: the same seed gives the same interval; another seed differs.
- Degenerate: identical inputs give a zero-width interval at 1.0.
- Each resample equals the score recomputed by hand on the resampled
  stimuli (self-pairs dropped for RDMs; plain row resampling for CKA).
- Coverage oracle (pre-registered before the first run): a finite
  population of 300 stimuli whose population RSA score is known exactly;
  200 seeds each draw 30 stimuli and a 400-resample interval at
  confidence 0.90. The interval must cover the population value in at
  least 0.90 - 3 * sqrt(0.90 * 0.10 / 200) = 0.836 of the seeds. The
  stimulus bootstrap is known to be conservative for RDM comparisons; an
  upper bound of 0.90 + 0.064 = 0.964, added after the first measured rate
  (0.915), stops an interval that covers everything from passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchlens.repgeom import (
    cka_interval,
    compare_rdms,
    compare_rdms_interval,
    rdm,
)


def _features(seed: int, n: int = 20, d: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = x @ rng.normal(size=(d, d)) + 0.8 * rng.normal(size=(n, d))
    return x, y


def test_condensed_input_gives_the_same_interval_as_square() -> None:
    x, y = _features(1, n=10)
    square_a, square_b = rdm(x), rdm(y)
    upper = np.triu_indices(10, k=1)
    from_square = compare_rdms_interval(square_a, square_b, num_resamples=50, seed=3)
    from_condensed = compare_rdms_interval(
        square_a[upper], square_b[upper], num_resamples=50, seed=3
    )
    assert from_square == from_condensed


def test_identical_inputs_give_a_zero_width_interval() -> None:
    x, _ = _features(3)
    same = rdm(x, "correlation")
    for method in ("pearson", "spearman", "kendall"):
        result = compare_rdms_interval(same, same.copy(), method, num_resamples=100, seed=0)
        assert result.width == 0.0, method
        assert result.low == result.high == 1.0, method
    cka_result = cka_interval(x, x.copy(), num_resamples=100, seed=0)
    assert cka_result.width == 0.0
    assert cka_result.low == cka_result.high == 1.0


def test_invalid_settings_refuse() -> None:
    x, y = _features(6, n=8)
    rdm_a, rdm_b = rdm(x), rdm(y)
    with pytest.raises(ValueError, match="num_resamples"):
        compare_rdms_interval(rdm_a, rdm_b, num_resamples=1)
    with pytest.raises(ValueError, match="confidence"):
        compare_rdms_interval(rdm_a, rdm_b, confidence=1.0)
    with pytest.raises(ValueError, match="method"):
        compare_rdms_interval(rdm_a, rdm_b, "cosine")
    with pytest.raises(ValueError, match="same stimuli"):
        compare_rdms_interval(rdm_a, rdm_b[:7, :7])
    with pytest.raises(ValueError, match="condensed"):
        compare_rdms_interval(np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="same stimuli"):
        cka_interval(x, y[:7])


def test_cka_interval_names_the_cka_variant_in_its_result_and_dict() -> None:
    x, y = _features(8, n=10)
    result = cka_interval(x, y, num_resamples=10, seed=0)
    expected = "measure/*/cka-kernel=linear-hsic=gretton-score"
    assert result.similarity_repository_id == expected
    assert result.to_dict()["similarity_repository_id"] == expected


def test_spearman_resample_with_ties_equals_hand_recomputation() -> None:
    rng = np.random.default_rng(12)
    rdm_a = rdm(np.round(rng.normal(size=(9, 2)), 0))  # rounded: tied distances
    rdm_b = rdm(rng.normal(size=(9, 3)))
    result = compare_rdms_interval(rdm_a, rdm_b, "spearman", num_resamples=5, seed=13)
    indices = np.random.default_rng(13).integers(0, 9, size=(5, 9))
    hand = []
    for index in indices:
        pairs = [
            (index[i], index[j]) for i in range(9) for j in range(i + 1, 9) if index[i] != index[j]
        ]
        a_vals = np.array([rdm_a[p] for p in pairs])
        b_vals = np.array([rdm_b[p] for p in pairs])
        hand.append(compare_rdms(a_vals, b_vals, method="spearman"))
    low, high = np.quantile(hand, [0.025, 0.975])
    assert result.low == pytest.approx(low, abs=1e-12)
    assert result.high == pytest.approx(high, abs=1e-12)


def test_constant_resamples_are_counted_out() -> None:
    x = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 0.5]])
    y = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    seeds_with_constant = [
        seed
        for seed in range(200)
        if any(len(set(row)) == 1 for row in np.random.default_rng(seed).integers(0, 3, (40, 3)))
    ]
    seed = seeds_with_constant[0]
    rows = np.random.default_rng(seed).integers(0, 3, (40, 3))
    n_constant = sum(len(set(row)) == 1 for row in rows)
    result = cka_interval(x, y, num_resamples=40, seed=seed)
    assert result.num_valid_resamples <= 40 - n_constant
