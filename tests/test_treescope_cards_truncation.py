"""F16 B3: the ported truncation protocol (treescope memo section 4).

Pinned regressions from the memo's test plan: port equivalence against
upstream on the measured shapes AND a non-default ``doubling_bonus``
(dropping the fifth parameter silently changes which cells are visible);
truncation-budget provenance (OUR 4,000/128/5 numbers, not render_array's
10,000/512/5); the truncation-mask law (a band travels as ``mask=False``,
never a data zero); and the adapter transfer bound on a gpt2-shaped
payload.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchlens.notebook._truncation import (
    CELL_BUDGET_DEFAULT,
    EDGE_ITEMS_DEFAULT,
    PER_AXIS_BUDGET_DEFAULT,
    compute_truncated_shape,
    infer_balanced_truncation,
    truncate_tensor_with_mask,
)

#: The memo's measured fixture shapes plus edge shapes.
SHAPES = [
    (2, 50257),
    (128, 768),
    (8, 12, 64, 64),
    (3, 224, 224),
    (7,),
    (16, 16),
]


def test_budget_constants_are_the_autovisualizer_numbers() -> None:
    """Provenance pin: our budgets are ArrayAutovisualizer's, NOT render_array's."""

    assert (CELL_BUDGET_DEFAULT, PER_AXIS_BUDGET_DEFAULT, EDGE_ITEMS_DEFAULT) == (4_000, 128, 5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bonus", [10.0, 3.0])
def test_port_matches_upstream_arithmetic(shape: tuple[int, ...], bonus: float) -> None:
    """The ported arithmetic equals upstream cell-for-cell, incl. non-default bonus."""

    upstream = pytest.importorskip("treescope._internal.arrayviz_impl")
    ours = infer_balanced_truncation(shape, 4_000, 128, 5, doubling_bonus=bonus)
    theirs = upstream.infer_balanced_truncation(shape, 4_000, 128, 5, doubling_bonus=bonus)
    assert ours == theirs
    assert compute_truncated_shape(shape, ours) == upstream.compute_truncated_shape(shape, ours)


@pytest.mark.parametrize("shape", [(128, 768), (8, 12, 16, 16), (7,), (33,)])
def test_port_matches_upstream_cells_and_mask(shape: tuple[int, ...]) -> None:
    """Slice-then-convert fetch: identical visible cells AND validity mask."""

    torch_support = pytest.importorskip("treescope.external.torch_support")
    adapter = torch_support.TorchTensorAdapter()
    tensor = torch.randn(*shape)
    edges = infer_balanced_truncation(shape, 4_000, 128, 5)
    theirs_values, theirs_mask = adapter.get_array_data_with_truncation(tensor, None, edges)
    ours_values, ours_mask = truncate_tensor_with_mask(tensor, edges)
    assert np.array_equal(theirs_values, ours_values)
    assert np.array_equal(theirs_mask, ours_mask)


def test_bf16_widens_like_upstream() -> None:
    """16-bit floats widen to float32 numpy exactly like the upstream adapter."""

    torch_support = pytest.importorskip("treescope.external.torch_support")
    adapter = torch_support.TorchTensorAdapter()
    tensor = torch.randn(300, 300, dtype=torch.bfloat16)
    edges = infer_balanced_truncation((300, 300), 4_000, 128, 5)
    theirs_values, theirs_mask = adapter.get_array_data_with_truncation(tensor, None, edges)
    ours_values, ours_mask = truncate_tensor_with_mask(tensor, edges)
    assert ours_values.dtype == theirs_values.dtype == np.float32
    assert np.array_equal(theirs_values, ours_values)
    assert np.array_equal(theirs_mask, ours_mask)


def test_mask_band_is_false_never_a_data_zero() -> None:
    """The truncation band is mask=False; visible cells carry real values."""

    tensor = torch.arange(1.0, 34.0)  # 33 elements, all nonzero
    edges = infer_balanced_truncation((33,), 20, 10, 3)
    values, mask = truncate_tensor_with_mask(tensor, edges)
    assert mask.dtype == np.bool_
    assert not mask.all() and mask.any()
    # Every UNMASKED (band) cell is the fill value, every masked-True cell a
    # real source value -- the band never launders as data.
    assert (values[~mask] == 0).all()
    assert (values[mask] != 0).all()


def test_untruncated_shapes_pass_through_whole() -> None:
    """Small shapes return every cell with an all-True mask."""

    tensor = torch.randn(4, 5)
    edges = infer_balanced_truncation((4, 5), 4_000, 128, 5)
    assert edges == (None, None)
    values, mask = truncate_tensor_with_mask(tensor, edges)
    assert values.shape == (4, 5) and mask.all()


def test_transfer_bound_on_gpt2_shaped_logits() -> None:
    """The visible-cell fraction stays under 2 percent at the pinned scale."""

    shape = (128, 50257)
    edges = infer_balanced_truncation(shape, 4_000, 128, 5)
    truncated = compute_truncated_shape(shape, edges)
    visible = int(np.prod(truncated))
    total = int(np.prod(shape))
    assert visible / total < 0.02
