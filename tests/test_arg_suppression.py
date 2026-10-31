"""L5 M4 pins: checked suppression of redundant constructor-arg rows (DEFAULT-ON).

Covers the design-memo sec-5 contract: the closed candidate table (both
directions per row — suppresses on PROVEN equality, survives on mismatch,
including a deliberately-lying func_config), the never-candidate set, the
input-side single-edge rule, grouped convs, the deliberate rolled-vs-
unrolled divergence, the detached-record degrade rule, and the
``show_redundant_args=`` opt-out.

HONESTY: the equality check is the LICENSE — a mismatch is exactly the
interesting case and must stay VISIBLE. The rule can only reveal more,
never hide a discrepancy. Never weaken it to make a label shorter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._arg_suppression import (
    SUPPRESSION_CANDIDATE_TABLE,
    compute_suppressed_arg_keys,
    suppressed_arg_keys_for_record,
)
from torchlens.visualization._label_format import format_module_kwargs
from torchlens.visualization._render_nodes import compute_default_node_lines

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MixedNet(nn.Module):
    """One member per major candidate family."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.ln = nn.LayerNorm(8)
        self.conv = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.emb = nn.Embedding(10, 5)

    def forward(self, x: torch.Tensor, img: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.ln(self.fc(x)).sum() + self.conv(img).sum() + self.emb(idx).sum()


class GroupedConv(nn.Module):
    """Grouped conv: in_channels equals the ACTUAL input channel dim."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, groups=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VaryingRecurrent(nn.Module):
    """Variable-length recurrence: rolled aggregate has no single shape."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n in (4, 3, 2):
            x = torch.relu(self.fc(x[:n]))
        return x.sum()


@pytest.fixture(scope="module")
def mixed_log() -> Any:
    log = tl.trace(
        MixedNet(),
        (torch.randn(3, 4), torch.randn(2, 3, 8, 8), torch.randint(0, 10, (2, 3))),
    )
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def varying_log() -> Any:
    log = tl.trace(VaryingRecurrent(), torch.randn(4, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _draw(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        vis_save_only=True,
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Table rows: suppresses on equality, per family
# ---------------------------------------------------------------------------


def test_default_draw_suppresses_proven_redundant_args(mixed_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(mixed_log, tmp_path)
    for redundant in (
        "in_features",
        "out_features",
        "in_channels",
        "out_channels",
        "normalized_shape",
        "embedding_dim",
    ):
        assert redundant not in dot, redundant


def test_never_candidates_stay_visible(mixed_log: tl.Trace, tmp_path: Path) -> None:
    """kernel_size/stride/padding/num_embeddings are not recoverable from
    I/O shapes and are NEVER suppression candidates."""

    dot = _draw(mixed_log, tmp_path)
    for kept in ("kernel_size", "stride", "padding", "num_embeddings"):
        assert kept in dot, kept


def test_show_redundant_args_shows_everything(mixed_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(mixed_log, tmp_path, show_redundant_args=True)
    for arg in (
        "in_features",
        "out_features",
        "in_channels",
        "out_channels",
        "normalized_shape",
        "embedding_dim",
        "kernel_size",
        "num_embeddings",
    ):
        assert arg in dot, arg


def test_grouped_conv_license_is_self_honest(tmp_path: Path) -> None:
    """Grouped conv: the equality check compares against the ACTUAL captured
    input channel dim, never a param shape.

    Live capture derives ``in_channels`` from the weight (in/groups = 2
    here), which does NOT equal the actual input channel dim (4) — so the
    license correctly REFUSES and the arg stays visible (mismatch is the
    interesting case). ``out_channels`` matches the output channel dim and
    suppresses. If capture ever records the constructor value instead, the
    equality check flips to suppression automatically — either way, no
    param shape is consulted and nothing is hidden unprovably.
    """

    log = tl.trace(GroupedConv(), torch.randn(2, 4, 8, 8))
    try:
        dot = _draw(log, tmp_path)
        assert "in_channels=2" in dot  # weight-derived claim != input dim 4
        assert "out_channels" not in dot
        assert "groups=2" in dot
    finally:
        log.cleanup()


# ---------------------------------------------------------------------------
# The check is the license: mismatch/unavailable stays VISIBLE
# ---------------------------------------------------------------------------


def test_lying_func_config_stays_visible(tmp_path: Path) -> None:
    """A func_config whose claim does NOT match the captured shape keeps the
    arg visible — the mismatch case is exactly the interesting one."""

    log = tl.trace(nn.Sequential(nn.Linear(4, 8)), torch.randn(3, 4))
    try:
        record = log["linear_1_1"]
        record.func_config["out_features"] = 999
        record.func_config["in_features"] = 999
        dot = _draw(log, tmp_path)
        assert "out_features=999" in dot
        assert "in_features=999" in dot
    finally:
        log.cleanup()


def test_multi_parent_input_side_stays_visible() -> None:
    """Input-side rows check ONLY via exactly one incoming data edge."""

    class Stub:
        layer_type = "linear"
        func_config = {"in_features": 4, "out_features": 8}
        parents = ("a", "b")
        shape = (3, 8)

    class StubTrace:
        backend = "torch"
        layer_logs: dict[str, Any] = {}

    keys = suppressed_arg_keys_for_record(StubTrace(), Stub(), {})
    assert keys == frozenset({"out_features"})


def test_non_torch_backend_never_candidates() -> None:
    class StubTrace:
        backend = "tf"
        layer_list = ()

    class StubUniverse:
        units = ()

    assert compute_suppressed_arg_keys(StubTrace(), StubUniverse()) == {}


def test_unrecognized_layer_type_not_candidate() -> None:
    class Stub:
        layer_type = "mycustommodule"
        func_config = {"in_features": 4}
        parents = ("a",)
        shape = (3, 4)

    class StubTrace:
        backend = "torch"
        layer_logs: dict[str, Any] = {}

    assert suppressed_arg_keys_for_record(StubTrace(), Stub(), {}) == frozenset()


# ---------------------------------------------------------------------------
# Rolled-vs-unrolled divergence (deliberate, pinned in BOTH modes)
# ---------------------------------------------------------------------------


def test_varying_recurrence_unrolled_suppresses(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Per-pass nodes have their own captured shapes: the equality check
    licenses suppression pass by pass."""

    dot = _draw(varying_log, tmp_path, vis_mode="unrolled")
    assert "out_features" not in dot


def test_varying_recurrence_rolled_keeps_args_visible(
    varying_log: tl.Trace, tmp_path: Path
) -> None:
    """The rolled aggregate has NO single output shape (variation marker):
    the license cannot be checked, so the arg stays VISIBLE. An honest,
    DELIBERATE cross-mode difference — do not 'fix' it."""

    dot = _draw(varying_log, tmp_path, vis_mode="rolled")
    assert "out_features=4" in dot


# ---------------------------------------------------------------------------
# Detached-record degrade rule (pinned, not inferred)
# ---------------------------------------------------------------------------


def test_detached_record_renders_all_args(mixed_log: tl.Trace) -> None:
    """Row builds without the trace-bearing prepass show every arg and never
    crash (no source_trace weakref read exists in the label path)."""

    record = mixed_log["linear_1_1"]
    kwargs_line = format_module_kwargs(record)
    assert kwargs_line is not None
    assert "in_features" in kwargs_line and "out_features" in kwargs_line
    lines = compute_default_node_lines(record)
    assert any("in_features" in line for line in lines)


def test_format_module_kwargs_suppressed_keys_param() -> None:
    class Stub:
        layer_type = "linear"
        func_config = {"in_features": 4, "out_features": 8, "bias": True}

    line = format_module_kwargs(Stub(), suppressed_keys=frozenset({"in_features"}))
    assert line is not None
    assert "in_features" not in line and "out_features=8" in line and "bias=True" in line


# ---------------------------------------------------------------------------
# Table shape pins
# ---------------------------------------------------------------------------


def test_never_candidate_args_absent_from_table() -> None:
    forbidden = {
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "num_embeddings",
        "num_heads",
        "p",
        "dropout",
    }
    for rows in SUPPRESSION_CANDIDATE_TABLE.values():
        assert not (set(rows) & forbidden)


def test_table_sides_and_axes_are_closed_vocabulary() -> None:
    for rows in SUPPRESSION_CANDIDATE_TABLE.values():
        for side, axis in rows.values():
            assert side in {"input", "output"}
            assert axis in {"last", "channel", "trailing"}
