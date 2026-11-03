"""Encoding on collapsed module boxes and segments, and label-aware size.

Covers the draw-time contract for ``color_by`` / ``size_by`` when the graph is
collapsed (``depth=N`` / ``collapse=...``): a module box or segment capsule is
encoded with the EXACT sum of its member ops for additive sources (time,
bytes, flops), is left unencoded with a legend note for sources that have no
exact sum, and ``size_by`` accepts the same builtin tokens as ``color_by``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._encoding import resolve_color_by, resolve_size_by
from torchlens._visualization._encoding_aggregate import (
    LINE_SUM,
    NOTE_INCOMPLETE,
    AggregateTarget,
    aggregate_field,
    sum_member_ops,
    unaggregatable_note,
)
from torchlens._visualization._encoding_size_layout import (
    DEFAULT_FONT_PT,
    SIZE_BY_MAX_AREA_MULT,
    estimate_label_box,
    scaled_size,
)


class Block(nn.Module):
    """Two-op block (linear + relu)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


class Nested(nn.Module):
    """A stem, a small block stack and a wide head: boxes differ in bytes."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 8)
        self.blocks = nn.Sequential(Block(8), Block(8))
        self.head = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


@pytest.fixture(scope="module")
def nested_log() -> Any:
    log = tl.trace(Nested(), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _draw(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(file_format="svg", path=str(tmp_path / "g"), **kwargs).dot


def _box_targets(log: tl.Trace) -> dict[str, AggregateTarget]:
    state = log._last_encoding_state
    assert state.aggregates is not None
    return {target.name: target for target in state.aggregates.by_unit.values()}


# ---------------------------------------------------------------------------
# size_by builtin tokens (same set and meaning as color_by)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["time", "bytes", "flops", "magnitude", "grad_norm"])
def test_size_by_accepts_color_by_builtins(token: str) -> None:
    size_spec = resolve_size_by(token)
    color_spec = resolve_color_by(token)
    assert size_spec is not None and color_spec is not None
    assert size_spec.source_kind == color_spec.source_kind == "builtin"
    assert size_spec.source == color_spec.source


def test_size_by_bytes_matches_activation_memory_field(
    nested_log: tl.Trace, tmp_path: Path
) -> None:
    _draw(nested_log, tmp_path, size_by="bytes")
    by_token = dict(nested_log._last_encoding_state.size_values)
    _draw(nested_log, tmp_path, size_by="activation_memory")
    by_field = dict(nested_log._last_encoding_state.size_values)
    assert by_token and by_token == by_field


def test_size_by_bytes_legend_uses_byte_units(nested_log: tl.Trace, tmp_path: Path) -> None:
    svg = _draw(nested_log, tmp_path, size_by="bytes")
    assert "size_by: bytes" in svg
    assert " B" in svg or "KiB" in svg or "kB" in svg or "KB" in svg


# ---------------------------------------------------------------------------
# Exact aggregates on collapsed module boxes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,field", [("bytes", "activation_memory"), ("flops", "flops_forward")]
)
def test_box_value_is_exact_member_sum(
    nested_log: tl.Trace, tmp_path: Path, token: str, field: str
) -> None:
    _draw(nested_log, tmp_path, depth=1, color_by=token, size_by=token)
    state = nested_log._last_encoding_state
    targets = _box_targets(nested_log)
    assert targets, "depth=1 drew no module boxes"
    for target in targets.values():
        assert target.op_labels
        expected = sum(float(getattr(nested_log.ops[label], field)) for label in target.op_labels)
        assert state.values[target.key] == pytest.approx(expected)
        assert state.size_values[target.key] == pytest.approx(expected)
    assert LINE_SUM in state.aggregation_lines
    assert LINE_SUM in state.size_aggregation_lines


def test_box_scope_covers_every_member_op(nested_log: tl.Trace, tmp_path: Path) -> None:
    _draw(nested_log, tmp_path, depth=1, color_by="bytes")
    targets = _box_targets(nested_log)
    blocks = [target for name, target in targets.items() if name.startswith("blocks")]
    assert len(blocks) == 1
    # Two blocks of (linear, relu): four member ops, none shared with another box.
    assert len(blocks[0].op_labels) == 4
    seen: set[str] = set()
    for target in targets.values():
        assert not seen & set(target.op_labels)
        seen |= set(target.op_labels)


def test_boxes_get_fill_and_size_in_dot(nested_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(nested_log, tmp_path, depth=1, color_by="bytes", size_by="bytes")
    state = nested_log._last_encoding_state
    for target in _box_targets(nested_log).values():
        assert target.key in state.colors
        assert target.key in state.final_sizes
        width = state.final_sizes[target.key][0]
        assert f"width={width}" in dot or f'width="{width}"' in dot


def test_unaggregatable_source_leaves_boxes_unencoded_with_note(
    nested_log: tl.Trace, tmp_path: Path
) -> None:
    _draw(nested_log, tmp_path, depth=1, color_by="magnitude")
    state = nested_log._last_encoding_state
    spec = resolve_color_by("magnitude")
    for target in _box_targets(nested_log).values():
        assert target.key not in state.values
        assert target.key not in state.colors
    assert unaggregatable_note(spec) in state.notes


def test_aggregate_field_allowlist() -> None:
    assert aggregate_field(resolve_color_by("time")) == "func_duration"
    assert aggregate_field(resolve_color_by("bytes")) == "activation_memory"
    assert aggregate_field(resolve_color_by("flops")) == "flops_forward"
    assert aggregate_field(resolve_color_by("magnitude")) is None
    assert aggregate_field(resolve_size_by("dims")) is None


def test_sum_member_ops_refuses_incomplete(nested_log: tl.Trace) -> None:
    labels = [op.label for op in nested_log.ops][:3]
    assert sum_member_ops(nested_log, (), "activation_memory") is None
    assert sum_member_ops(nested_log, (*labels, "no_such_op"), "activation_memory") is None
    assert NOTE_INCOMPLETE.endswith("unencoded")


# ---------------------------------------------------------------------------
# Label-aware size geometry
# ---------------------------------------------------------------------------


def test_scaled_size_area_spans_max_mult() -> None:
    reference = (2.0, 1.0)
    small = scaled_size(reference, 0.0, SIZE_BY_MAX_AREA_MULT)
    large = scaled_size(reference, 1.0, SIZE_BY_MAX_AREA_MULT)
    assert small == reference
    assert large[0] * large[1] == pytest.approx(
        SIZE_BY_MAX_AREA_MULT * reference[0] * reference[1], rel=1e-3
    )


def test_final_sizes_fit_labels_and_scale_from_a_shared_base(
    nested_log: tl.Trace, tmp_path: Path
) -> None:
    _draw(nested_log, tmp_path, size_by="bytes")
    state = nested_log._last_encoding_state
    finals = state.final_sizes
    assert finals and set(finals) <= set(state.size_fractions)
    floor = estimate_label_box(["x"], DEFAULT_FONT_PT)
    for size in finals.values():
        assert size[0] >= floor[0] and size[1] >= floor[1]
    # Area / base area is exactly 1 + f * (M - 1) for every node; nodes that share
    # the median base are therefore ordered by value.
    bases = {}
    for key, size in finals.items():
        factor = math.sqrt(1.0 + state.size_fractions[key] * (SIZE_BY_MAX_AREA_MULT - 1.0))
        bases[key] = (round(size[0] / factor, 2), round(size[1] / factor, 2))
    shared = max(set(bases.values()), key=list(bases.values()).count)
    order = sorted(
        (key for key in finals if bases[key] == shared), key=lambda key: state.size_fractions[key]
    )
    areas = [finals[key][0] * finals[key][1] for key in order]
    assert len(order) >= 2
    assert all(b >= a - 1e-6 for a, b in zip(areas, areas[1:], strict=False))


def test_one_long_label_does_not_inflate_every_node() -> None:
    from types import SimpleNamespace

    from torchlens._visualization._encoding_size_layout import layout_encoded_sizes

    short = ("op_1", "(2, 8)")
    long_row = ("op_9", "x" * 120)
    state = SimpleNamespace(
        size_fractions={"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.5},
        sizes={},
        unit_keys={},
        final_sizes={},
        aggregates=SimpleNamespace(
            segments=(
                SimpleNamespace(key="a", label_lines=short),
                SimpleNamespace(key="b", label_lines=short),
                SimpleNamespace(key="c", label_lines=short),
                SimpleNamespace(key="d", label_lines=long_row),
            )
        ),
    )
    layout_encoded_sizes((), state, SimpleNamespace(font_size=None))
    finals = state.final_sizes
    short_box = estimate_label_box(short, DEFAULT_FONT_PT)
    long_box = estimate_label_box(long_row, DEFAULT_FONT_PT)
    assert finals["a"][0] == pytest.approx(short_box[0], abs=1e-3)
    assert finals["c"][0] * finals["c"][1] == pytest.approx(
        SIZE_BY_MAX_AREA_MULT * finals["a"][0] * finals["a"][1], rel=1e-2
    )
    assert finals["d"][0] >= long_box[0]


def test_label_box_ignores_markup() -> None:
    assert estimate_label_box(["<b>ab</b>"], DEFAULT_FONT_PT) == estimate_label_box(
        ["ab"], DEFAULT_FONT_PT
    )
    assert estimate_label_box(["a&amp;b"], DEFAULT_FONT_PT) == estimate_label_box(
        ["a&b"], DEFAULT_FONT_PT
    )


def test_user_override_of_size_survives_layout(nested_log: tl.Trace, tmp_path: Path) -> None:
    def node_spec_fn(layer_log: Any, default_spec: Any) -> Any:
        return default_spec.replace(width=9.5, height=4.25)

    dot = _draw(nested_log, tmp_path, size_by="bytes", node_spec_fn=node_spec_fn)
    assert "width=9.5" in dot or 'width="9.5"' in dot
