"""RDM widening, rdm_compare, and CKA placement oracles (F20 D-5/D-6/D-25).

D-5: the existing ``rdm`` name widens with keyword-only options; the
no-new-keyword call stays bit-identical. D-6: ``rdm_compare`` is the
descriptive rank correlation over aligned upper triangles, diagonal
excluded, with the silent hand-rolled errors prevented. D-25: CKA gains
``device=``/``dtype=`` through the one placement seam.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchlens as tl
from torchlens import repgeom
from torchlens.stats import CKA, cka


def _acts(n: int = 10, d: int = 24) -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(n, d, dtype=torch.float64)


def test_no_new_keyword_call_is_bit_identical() -> None:
    """D-5's compatibility law: plain rdm() moves no caller's numbers."""

    acts = _acts()
    assert (repgeom.rdm(acts) == repgeom.activation_distance_matrix(acts)).all()


def test_manhattan_metric_and_row_blocking() -> None:
    """Manhattan ships; blocked computation equals one-shot computation."""

    acts = _acts()
    one_shot = repgeom.rdm(acts, metric="manhattan")
    blocked = repgeom.rdm(acts, metric="manhattan", row_chunk_size=3)
    assert np.allclose(one_shot, blocked)
    # Manhattan >= euclidean pairwise, elementwise (norm inequality).
    euclid = repgeom.rdm(acts)
    assert (one_shot >= euclid - 1e-12).all()


def test_blocked_euclidean_matches_legacy() -> None:
    """Row blocking is a memory shape, never a numeric change."""

    acts = _acts()
    assert np.allclose(repgeom.rdm(acts, row_chunk_size=4), repgeom.rdm(acts))


def test_condensed_output_is_strict_upper_triangle() -> None:
    """output='condensed' returns the row-major strict upper triangle."""

    acts = _acts(8)
    square = repgeom.rdm(acts)
    condensed = repgeom.rdm(acts, output="condensed")
    assert condensed.shape == (8 * 7 // 2,)
    assert np.allclose(condensed, square[np.triu_indices(8, k=1)])


def test_batched_input_kind_is_explicit() -> None:
    """[B, N, D] means one RDM per batch element ONLY under input_kind."""

    batch = torch.randn(3, 6, 5, dtype=torch.float64)
    batched = repgeom.rdm(batch, input_kind="batched")
    assert batched.shape == (3, 6, 6)
    for b in range(3):
        assert np.allclose(batched[b], repgeom.rdm(batch[b]))
    # The default reading (3-D flattened per stimulus) is unchanged.
    flat = repgeom.rdm(batch)
    assert flat.shape == (3, 3)
    with pytest.raises(ValueError, match="input_kind"):
        repgeom.rdm(batch, input_kind="stacked")


def test_rdm_compare_descriptive_correlations() -> None:
    """D-6: aligned upper triangles, diagonal excluded, three methods."""

    acts = _acts()
    a = repgeom.rdm(acts)
    b = repgeom.rdm(acts, metric="manhattan")
    assert abs(repgeom.rdm_compare(a, a) - 1.0) < 1e-12
    spearman = repgeom.rdm_compare(a, b)
    pearson = repgeom.rdm_compare(a, b, method="pearson")
    kendall = repgeom.rdm_compare(a, b, method="kendall")
    assert -1.0 <= spearman <= 1.0
    assert -1.0 <= pearson <= 1.0
    assert -1.0 <= kendall <= 1.0
    # Condensed and square spellings agree (the diagonal never leaks in).
    condensed = repgeom.rdm(acts, output="condensed")
    assert repgeom.rdm_compare(condensed, b) == pytest.approx(spearman)


def test_rdm_compare_refusals_teach() -> None:
    """Shape mismatch and unknown methods refuse; inference points away."""

    a = repgeom.rdm(_acts(8))
    b = repgeom.rdm(_acts(9))
    with pytest.raises(ValueError, match="same stimuli"):
        repgeom.rdm_compare(a, b)
    with pytest.raises(ValueError, match="rsatoolbox"):
        repgeom.rdm_compare(a, a, method="pvalue")


def test_cka_gains_placement_through_one_seam() -> None:
    """D-25: cka()/CKA accept device=/dtype=; defaults stay bit-identical."""

    torch.manual_seed(0)
    a, b = torch.randn(24, 12), torch.randn(24, 12)
    legacy = cka(a, b)
    f32 = cka(a, b, dtype=torch.float32)
    assert legacy == pytest.approx(f32, abs=1e-5)
    acc = CKA(dtype=torch.float32)
    acc.update(a[:12], b[:12])
    acc.update(a[12:], b[12:])
    assert acc.result() == pytest.approx(legacy, abs=1e-4)
    with pytest.raises(ValueError, match="floating"):
        cka(a, b, dtype=torch.int64)


def test_tl_repgeom_rdm_compare_is_reachable() -> None:
    """The facade re-exports the new descriptive comparator."""

    assert tl.repgeom.rdm_compare is repgeom.rdm_compare
