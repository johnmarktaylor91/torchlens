"""Layer site accessors: ``site_key`` (typed ambiguity, I-S3'),
``site_peers`` (live index, typed legacy refusal), and ``shape_summary``
(derived across-pass data string with its pinned character class).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.data_classes._layer_sites import (
    SHAPE_SUMMARY_CHARACTER_CLASS,
    layer_shape_summary,
)


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
    """Within-call recurrence fixture: the grouper pairs the two residual
    adds (and tanhs) of ONE block call as pass 1/2 of one layer -- a
    site-SPANNING group (the transformer residual-add pattern in miniature).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.tanh(x)
        x = x + torch.tanh(x)
        return x


class _ResModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blk = _ResBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blk(x)


class _Shrink(nn.Module):
    """Reused-relu cohort fixture: one ReLU module applied to three
    different shapes mints three single-pass layers SHARING one site key
    (the ResNet reused-relu pattern in miniature)."""

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for k in (8, 4, 2):
            x = self.act(x[:, :k])
        return x


class _Grow(nn.Module):
    """Variable-shape param loop: one Linear reused on a growing slice
    mints a multi-pass layer whose passes vary in exactly one axis."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x[:, :2, :]
        for k in (2, 3, 4):
            y = self.lin(x[:, :k, :]) + 0.0 * y.sum()
        return y


# ---------------------------------------------------------------------------
# Layer.site_key
# ---------------------------------------------------------------------------


def test_site_key_single_on_uniform_layer() -> None:
    log = tl.trace(_Tied(), torch.randn(2, 8))
    assert log["linear_1_1"].site_key == "s1|lin|linear||1"
    assert log["relu_1_2"].site_key == "s1|act|relu||1"


def test_site_key_refuses_typed_on_spanning_layer() -> None:
    # I-S3': a silent single-key read on a site-spanning layer is the
    # failure this tripwire exists to catch.
    log = tl.trace(_ResModel(), torch.randn(2, 8))
    layer = log["tanh_1_1"]
    member_keys = [op.site_key for _, op in sorted(layer.ops.items())]
    assert member_keys == ["s1|blk|tanh||1", "s1|blk|tanh||2"]
    with pytest.raises(InvalidArgumentError) as excinfo:
        _ = layer.site_key
    assert excinfo.value.fields["code"] == "layer_site_ambiguous"
    assert ".ops[k].site_key" in str(excinfo.value)


def test_site_key_refuses_typed_when_keys_absent() -> None:
    # Legacy-artifact shape: no op carries a key -> typed refusal, never a
    # None read (consumer-matrix row).
    log = tl.trace(_Tied(), torch.randn(2, 8))
    layer = log["linear_1_1"]
    for _, op in layer.ops.items():
        op.site_key = None
    with pytest.raises(InvalidArgumentError) as excinfo:
        _ = layer.site_key
    assert excinfo.value.fields["code"] == "site_key_unavailable"


# ---------------------------------------------------------------------------
# Layer.site_peers
# ---------------------------------------------------------------------------


def test_site_peers_finds_reused_site_cohort() -> None:
    log = tl.trace(_Shrink(), torch.randn(2, 16))
    relu_labels = [label for label in log.layer_labels if "relu" in label]
    assert len(relu_labels) == 3
    first = log[relu_labels[0]]
    assert [peer.layer_label for peer in first.site_peers] == relu_labels[1:]
    # Symmetric membership: every cohort member sees the other two.
    for label in relu_labels:
        peer_labels = {peer.layer_label for peer in log[label].site_peers}
        assert peer_labels == set(relu_labels) - {label}


def test_site_peers_empty_when_site_unshared() -> None:
    log = tl.trace(_Tied(), torch.randn(2, 8))
    assert log["linear_1_1"].site_peers == ()


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


def test_shape_summary_none_for_single_pass_and_uniform() -> None:
    log = tl.trace(_Tied(), torch.randn(2, 8))
    assert log["linear_1_1"].shape_summary is None  # uniform multi-pass
    assert log["input_1"].shape_summary is None  # single-pass


def test_shape_summary_single_axis_monotone_on_real_capture() -> None:
    log = tl.trace(_Grow(), torch.randn(2, 4, 8))
    multi_pass = [log[label] for label in log.layer_labels if log[label].num_passes > 1]
    linear = next(layer for layer in multi_pass if "linear" in layer.layer_label)
    assert linear.shape_summary == "2->4"
    assert set(linear.shape_summary) <= SHAPE_SUMMARY_CHARACTER_CLASS


@pytest.mark.smoke
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


def test_shape_summary_may_contain_arrow_never_markup() -> None:
    # The S5 handoff fact: the ratified format CONTAINS ">" ("64->512"), so
    # render-side escaping is mandatory and assert-absence is impossible.
    summary = layer_shape_summary(_StubLayer([(2, 64), (2, 512)]))
    assert summary == "64->512"
    assert ">" in summary and "<" not in summary and "&" not in summary
