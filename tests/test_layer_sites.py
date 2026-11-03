"""Layer site accessors: ``site_key`` (typed ambiguity, I-S3'),
``site_peers`` (live index, typed legacy refusal), and ``shape_summary``
(derived across-pass data string with its pinned character class).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes._layer_sites import (
    SHAPE_SUMMARY_CHARACTER_CLASS,
    layer_shape_summary,
    layer_site_key,
)
from torchlens._errors import InvalidArgumentError


class _Tied(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.act(self.lin(x))
        return x


class _ResBlock(nn.Module):
    """Within-call recurrence fixture: a residual line run twice by a loop
    inside ONE block call is pass 1/2 of one layer per op (grouping rule rows
    2 and 11) whose passes sit at two positions of that call -- a
    site-SPANNING group. (Two residual lines written out are two layers,
    row 10, so they no longer span.)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            x = x + torch.tanh(x)
        return x


class _BranchAct(nn.Module):
    """Applies a relu written on a different line for each branch."""

    def forward(self, x: torch.Tensor, branch: int) -> torch.Tensor:
        if branch == 0:
            return torch.relu(x)
        if branch == 1:
            return torch.relu(x)
        return torch.relu(x)


class _Shrink(nn.Module):
    """Shared-site cohort fixture: one module called three times, each call
    taking a different branch line, mints three single-pass relu layers
    (grouping rule rows 41-42: per-line layers) that SHARE one site key, since
    each is the first relu of its own call. (A reused ``nn.ReLU`` instance no
    longer forms a cohort: its calls are one layer's passes, row 28.)"""

    def __init__(self) -> None:
        super().__init__()
        self.act = _BranchAct()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for branch, k in enumerate((8, 4, 2)):
            x = self.act(x[:, :k], branch)
        return x


# ---------------------------------------------------------------------------
# layer_site_key (the Layer-level site-key accessor)
# ---------------------------------------------------------------------------


def test_site_key_refuses_typed_when_keys_absent() -> None:
    # Legacy-artifact shape: no op carries a key -> typed refusal, never a
    # None read (consumer-matrix row).
    log = tl.trace(_Tied(), torch.randn(2, 8))
    layer = log["linear_1_1"]
    for _, op in layer.ops.items():
        op.site_key = None
    with pytest.raises(InvalidArgumentError) as excinfo:
        _ = layer_site_key(layer)
    assert excinfo.value.fields["code"] == "site_key_unavailable"


# ---------------------------------------------------------------------------
# Layer.site_peers
# ---------------------------------------------------------------------------


def test_site_peers_refuses_typed_without_keys() -> None:
    # Never a None-key peer-of-everything collapse.
    log = tl.trace(_Shrink(), torch.randn(2, 16))
    relu_labels = [label for label in log.layer_labels if "relu" in label]
    layer = log[relu_labels[0]]
    for _, op in layer.ops.items():
        op.site_key = None
    with pytest.raises(InvalidArgumentError) as excinfo:
        _ = layer.site_peers
    assert excinfo.value.fields["code"] == "site_key_unavailable"
    # Other layers with keys never match the stripped layer either way.
    other = log[relu_labels[1]]
    assert relu_labels[0] not in {peer.layer_label for peer in other.site_peers}


# ---------------------------------------------------------------------------
# Layer.shape_summary (derived data string; rendering is L5's)
# ---------------------------------------------------------------------------


class _StubOps(dict):
    def items(self):  # noqa: D102 - dict passthrough for the accessor shape
        return super().items()


class _StubOp:
    def __init__(self, shape: tuple[int, ...] | None) -> None:
        self.shape = shape


class _StubLayer:
    layer_label = "stub_1_1"

    def __init__(self, shapes: list[tuple[int, ...] | None]) -> None:
        self.ops = _StubOps(
            {pass_index: _StubOp(shape) for pass_index, shape in enumerate(shapes, start=1)}
        )


def test_shape_summary_format_matrix() -> None:
    # Single varying axis, monotone (asc / desc): first->last.
    assert layer_shape_summary(_StubLayer([(2, 2, 8), (2, 3, 8), (2, 4, 8)])) == "2->4"
    assert layer_shape_summary(_StubLayer([(2, 8), (2, 4), (2, 2)])) == "8->2"
    # Single varying axis, non-monotone: min-max.
    assert layer_shape_summary(_StubLayer([(2, 2), (2, 4), (2, 3)])) == "2-4"
    # Multiple varying axes: first-to-last full shapes.
    assert layer_shape_summary(_StubLayer([(2, 64, 8, 8), (2, 512, 4, 4)])) == "2x64x8x8->2x512x4x4"
    # Rank-varying: first-to-last full shapes.
    assert layer_shape_summary(_StubLayer([(2, 8), (2, 8, 1)])) == "2x8->2x8x1"
    # None / uniform / single-pass: no summary.
    assert layer_shape_summary(_StubLayer([(2, 8), None])) is None
    assert layer_shape_summary(_StubLayer([(2, 8), (2, 8)])) is None
    assert layer_shape_summary(_StubLayer([(2, 8)])) is None
    # Character-class pin holds on every emitted form.
    for shapes in ([(2, 2), (2, 4), (2, 3)], [(2, 64, 8, 8), (2, 512, 4, 4)]):
        summary = layer_shape_summary(_StubLayer(shapes))
        assert summary is not None and set(summary) <= SHAPE_SUMMARY_CHARACTER_CLASS
