"""L5 M1 pins: encoding-channel core + color_by.

Covers the design-memo 2.5 test list: channel resolution (both phases), the
prepass callable-once counting pin, conflict-matrix pins, legend content +
visibility rule, the collapse-plan-hash pin, the engine-resolution matrix,
the 2.3b rolled-aggregate allowlist color pins (alias rows, mirrored per-call
numerics, unclassified sources, classification completeness), and the 2.4(i)
record-derived image exclusion pins.

HONESTY: several pins here are tripwires — an encoding must never imply
uniformity it cannot prove. Never weaken them to make a render pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._layer_spec import _LAYER_MIRROR_SPEC, _LAYER_STATE_ORDER
from torchlens._data_classes.layer import Layer
from torchlens._record_type import record_type
from torchlens._visualization._encoding import (
    DARK_RAMP,
    LAYER_SOURCE_ROWS,
    LIGHT_RAMP,
    MIRRORED_PER_CALL_NUMERIC_FIELDS,
    NOTE_CONSTANT,
    NOTE_FIRST_PASS_ONLY,
    NOTE_NA_UNENCODED,
    NOTE_VARIES,
    ROW_MIRRORED_PER_CALL,
    interpolate_hex,
    resolve_color_by,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SmallMLP(nn.Module):
    """Plain single-pass model."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class VaryingRecurrent(nn.Module):
    """Variable-length recurrence: the same Linear over a shrinking batch.

    Produces a rolled multi-pass Layer whose ``varying_across_passes`` marker
    names shape / activation_memory / flops_* — the exact population the
    rolled-aggregate allowlist governs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n in (4, 3, 2):
            x = torch.relu(self.fc(x[:n]))
        return x.sum()


class UniformRecurrent(nn.Module):
    """Fixed-shape recurrence: multi-pass with NO varying marker."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.relu(self.fc(x))
        return x.sum()


@pytest.fixture(scope="module")
def mlp_log() -> Any:
    log = tl.trace(SmallMLP(), torch.randn(3, 4))
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


@pytest.fixture(scope="module")
def uniform_log() -> Any:
    log = tl.trace(UniformRecurrent(), torch.randn(4, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _draw(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    ).dot


def _state(log: tl.Trace) -> Any:
    """Return the last draw's encoding state (session diagnostic)."""

    return log._last_encoding_state


def _varying_layer(log: tl.Trace) -> Layer:
    # grouping rule rows 6 and 49: the loop's ``x[:n]`` getitem is now also one
    # multi-pass Layer, but it does no FLOPs; the varying population is fc's linear.
    return next(
        layer
        for layer in log._layer_logs.values()
        if len(layer.ops) > 1 and layer.layer_label.startswith("linear_")
    )


# ---------------------------------------------------------------------------
# Channel resolution (option validation)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_non_string_non_callable_refuses() -> None:
    with pytest.raises(Exception) as excinfo:
        resolve_color_by(42)
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_nan_overlay_is_not_a_scalar_color_source() -> None:
    """'nan' is a boolean overlay, not a scalar channel source."""

    with pytest.raises(Exception) as excinfo:
        resolve_color_by("nan")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


# ---------------------------------------------------------------------------
# Phase A/B end-to-end: encoded fills, callable-once pin, opt-in no-churn
# ---------------------------------------------------------------------------


def test_collapse_plan_identical_with_channel_active(tmp_path: Path) -> None:
    """CollapsePlan is byte-identical with the channel on vs off."""

    model = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), nn.Linear(4, 2))
    log = tl.trace(model, torch.randn(2, 4))
    plan_off = log.collapse_plan(mode="auto")
    _draw(log, tmp_path, color_by="bytes", collapse="auto")
    plan_on = log.collapse_plan(mode="auto")
    assert plan_off == plan_on


# ---------------------------------------------------------------------------
# Value/type rules (closed table, memo 2.2)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_bool_value_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, color_by=lambda node: True)
    assert excinfo.value.fields["code"] == "encoding_value_invalid"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_non_scalar_tensor_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, color_by=lambda node: torch.zeros(3))
    assert excinfo.value.fields["code"] == "encoding_value_invalid"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_one_element_tensor_accepted(mlp_log: tl.Trace, tmp_path: Path) -> None:
    _draw(mlp_log, tmp_path, color_by=lambda node: torch.tensor([2.5]))
    state = _state(mlp_log)
    # Degenerate domain (min == max) renders UNENCODED with the note, never
    # mid-ramp (themes memo build item 11): a uniform paint would claim a
    # difference from unencoded nodes the data cannot support.
    assert not state.colors
    assert state.values  # the scalar coercion itself succeeded
    assert NOTE_CONSTANT in state.notes


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_non_finite_degrades_to_unencoded(mlp_log: tl.Trace, tmp_path: Path) -> None:
    def source(node: Any) -> float:
        return float("nan") if record_type(node) == "relu" else float(node.raw_index)

    _draw(mlp_log, tmp_path, color_by=source)
    state = _state(mlp_log)
    assert NOTE_NA_UNENCODED in state.notes
    assert not any(key.startswith("relu") for key in state.colors)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py, _visualization/overlays.py.
def test_absent_field_degrades_to_unencoded(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """grad is absent (no backward ran): every node stays unencoded, no raise."""

    _draw(mlp_log, tmp_path, color_by="grad_norm")
    state = _state(mlp_log)
    assert state.colors == {}
    assert NOTE_NA_UNENCODED in state.notes


# ---------------------------------------------------------------------------
# Engine-resolution fence matrix (memo 2.1)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py, _visualization/_render_dot.py.
def test_explicit_rank_with_channel_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, color_by="bytes", layout="rank")
    assert excinfo.value.fields["code"] == "encoding_requires_dot_layout"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_rank_layout_internal/layout.py, _visualization/_rank_segments.py, _visualization/_render_dot.py.
def test_explicit_rank_without_channel_still_works(mlp_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(mlp_log, tmp_path, layout="rank")
    assert dot is not None


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py, _visualization/_rank_layout_internal/layout.py.
def test_auto_large_graph_forces_dot_with_notice(
    mlp_log: tl.Trace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUTO + cost>threshold + channel: dot forced, notice names channel+cost."""

    from torchlens._visualization._rank_layout_internal import layout as rank_layout

    monkeypatch.setattr(rank_layout, "RANK_LAYOUT_COST_THRESHOLD", 0)
    with pytest.warns(UserWarning, match="color_by"):
        _draw(mlp_log, tmp_path, color_by="bytes", layout="auto")
    assert _state(mlp_log).colors  # rendered on dot: channel applied


# ---------------------------------------------------------------------------
# Legend visibility rule + content (memo 2.3)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_header_block.py, _visualization/_legend.py, _visualization/_render_dot.py ....
def test_legend_true_without_channel_keeps_todays_meaning(
    mlp_log: tl.Trace, tmp_path: Path
) -> None:
    dot = _draw(mlp_log, tmp_path, show_legend=True)
    assert 'ID="tl_key"' in dot
    assert "TorchLens encoding" not in dot


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_draw_validation.py, _visualization/_render_dot.py.
def test_show_legend_rejects_non_tri_state(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, show_legend="yes")
    assert excinfo.value.fields["code"] == "visualization_bool_option_invalid"


# ---------------------------------------------------------------------------
# Conflict matrix (memo 2.4)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_edit_marks.py, _visualization/_encoding.py.
def test_intervention_border_composes_with_channel_fill(tmp_path: Path) -> None:
    """Intervention site styling (border) + channel fill land on one node."""

    model = SmallMLP()
    log = tl.trace(
        model,
        torch.randn(3, 4),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    dot = _draw(log, tmp_path, color_by=lambda node: float(node.raw_index))
    state = _state(log)
    relu_key = next(key for key in state.colors if key.startswith("relu"))
    fill = state.colors[relu_key]
    site_line = next(line for line in dot.splitlines() if "relu" in line and "fillcolor" in line)
    assert fill in site_line
    assert "#7B1FA2" in site_line  # edited node ring colour


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py, _visualization/_render_edges.py, _visualization/_render_leaf.py ....
def test_collapsed_module_boxes_not_encoded(tmp_path: Path) -> None:
    """Collapsed boxes keep their trainability fill semantics (2.4)."""

    model = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), nn.Linear(4, 2))
    log = tl.trace(model, torch.randn(2, 4))
    _draw(log, tmp_path, color_by="bytes", collapse_fn=lambda module: module.address == "0")
    state = _state(log)
    # Only raw_op record nodes carry channel values; hidden members of the
    # collapsed module contribute nothing.
    for key in state.colors:
        assert not key.startswith("0:"), key


# ---------------------------------------------------------------------------
# 2.3b rolled-aggregate allowlist pins (the honesty tripwire)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_uniform_multipass_field_encodes(uniform_log: tl.Trace, tmp_path: Path) -> None:
    """No marker -> all passes agree -> exact value encodes normally."""

    layer = _varying_layer(uniform_log)
    assert "varying_across_passes" not in layer.annotations
    _draw(uniform_log, tmp_path, color_by="activation_memory", rolled=True)
    state = _state(uniform_log)
    assert layer.layer_label in state.colors
    assert NOTE_VARIES not in state.notes


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py, _visualization/overlays.py.
def test_per_pass_builtin_degrades_on_rolled(varying_log: tl.Trace, tmp_path: Path) -> None:
    """'time' (func_duration) is per-pass: rolled nodes stay unencoded."""

    layer = _varying_layer(varying_log)
    _draw(varying_log, tmp_path, color_by="time", rolled=True)
    state = _state(varying_log)
    assert layer.layer_label not in state.colors
    # The per-pass row carries its OWN note now (themes memo item 11): the
    # old pass-1 read painted a uniform mid-ramp from a min:1/max:1 domain.
    from torchlens._visualization._encoding import NOTE_PER_PASS_ROLLED

    assert NOTE_PER_PASS_ROLLED in state.notes


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_callable_per_pass_read_on_rolled_chains_tripwire(
    varying_log: tl.Trace, tmp_path: Path
) -> None:
    """A per-pass read inside a callable surfaces as encoding_callable_error."""

    with pytest.raises(Exception) as excinfo:
        _draw(
            varying_log,
            tmp_path,
            color_by=lambda node: float(node.func_duration),
            rolled=True,
        )
    assert excinfo.value.fields["code"] == "encoding_callable_error"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/layer.py, _visualization/_encoding.py.
def test_alias_rows_flops(varying_log: tl.Trace, tmp_path: Path) -> None:
    """ALIAS PIN: flops_forward unencodes; total_flops_forward encodes.

    Different quantities carry different verdicts DELIBERATELY: the per-pass
    value has no single value on a varying recurrence, while the trace total
    is exact — both disclosed, neither silent. A regression toward either
    silent encoding or blanket refusal fails here.
    """

    layer = _varying_layer(varying_log)
    _draw(varying_log, tmp_path, color_by="flops_forward", rolled=True)
    per_pass_state = _state(varying_log)
    assert layer.layer_label not in per_pass_state.colors
    assert NOTE_VARIES in per_pass_state.notes

    dot = _draw(varying_log, tmp_path, color_by="total_flops_forward", rolled=True)
    total_state = _state(varying_log)
    assert layer.layer_label in total_state.colors
    assert any("total_flops_forward" in line for line in total_state.aggregation_lines)
    assert "total across passes" in dot  # mandatory aggregation legend line


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_mirrored_per_call_numeric_unencoded(varying_log: tl.Trace, tmp_path: Path) -> None:
    """THE RESIDUAL PIN (sol r4 MAJOR-1): a varying mirrored numeric can never
    reach a uniform encoding.

    ``raw_index`` is mirror-projected from pass 1 (here 3, against per-pass
    truth [3, 6, 9]) with no variation marker — encoding it as if uniform
    would be a dishonest visual. The mirrored-per-call row unencodes it with
    a legend note; the same holds for every member of the enumerated set.
    """

    layer = _varying_layer(varying_log)
    per_pass_truth = [op.raw_index for op in layer.ops.values()]
    assert len(set(per_pass_truth)) > 1, "fixture no longer varies raw_index"
    assert layer.raw_index == per_pass_truth[0]  # the mirror IS a pass-1 projection

    for source in ("raw_index", "step_index", "ordinal_index", "transformed_gradient_memory"):
        dot = _draw(varying_log, tmp_path, color_by=source, rolled=True)
        state = _state(varying_log)
        assert layer.layer_label not in state.colors, source
        assert NOTE_FIRST_PASS_ONLY in state.notes, source
        assert "first-pass-only field" in dot, source


def test_mirrored_per_call_enumeration_matches_live_mirror_spec() -> None:
    """The enumerated mirrored-numeric set stays honest against the live spec.

    Every enumerated STORED name must actually be a first-pass mirror (in
    ``_LAYER_MIRROR_SPEC``) and NOT be reconciled/overwritten elsewhere; and
    the memo's original single-member claim stays corrected (the set is
    strictly larger than {transformed_gradient_memory}).
    """

    stored_members = MIRRORED_PER_CALL_NUMERIC_FIELDS & set(_LAYER_MIRROR_SPEC)
    assert "transformed_gradient_memory" in stored_members
    assert {"raw_index", "step_index", "ordinal_index", "grad_fn_object_id", "buffer_pass"} <= (
        stored_members
    ), "the sol r4 MAJOR-1 correction regressed"
    from torchlens._postprocess.finalization import (
        _MULTIPASS_BYTES_FIELDS,
        _MULTIPASS_FLOPS_FIELDS,
        _MULTIPASS_SHAPE_FIELDS,
    )

    reconciled = (
        set(_MULTIPASS_BYTES_FIELDS) | set(_MULTIPASS_FLOPS_FIELDS) | set(_MULTIPASS_SHAPE_FIELDS)
    )
    assert not (MIRRORED_PER_CALL_NUMERIC_FIELDS & reconciled)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py.
def test_unclassified_source_on_rolled_refuses(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Allowlist default: an Op-only field with no rolled row refuses typed."""

    # 'num_inputs' is a NUMERIC Op-schema field (legal at option validation,
    # encodable on single-pass nodes) with no declared Layer rolled row.
    from torchlens._constants import LAYER_PASS_LOG_FIELD_ORDER

    assert "num_inputs" in LAYER_PASS_LOG_FIELD_ORDER
    assert "num_inputs" not in LAYER_SOURCE_ROWS
    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, color_by="num_inputs", rolled=True)
    assert excinfo.value.fields["code"] == "encoding_source_invalid"
    assert "rolled" in str(excinfo.value)


def test_classification_completeness_pin() -> None:
    """Every readable Layer name is classified into exactly one 2.3b row.

    Walks _LAYER_STATE_ORDER + _LAYER_MIRROR_SPEC keys + the public Layer
    properties and asserts set equality with the classification table. A new
    field/property FAILS here until a row classifies it — and until then it
    refuses at runtime by the allowlist default.
    """

    properties = {
        name
        for name, value in vars(Layer).items()
        if isinstance(value, property) and not name.startswith("_")
    }
    walked = set(_LAYER_STATE_ORDER) | set(_LAYER_MIRROR_SPEC) | properties
    classified = set(LAYER_SOURCE_ROWS)
    missing = sorted(walked - classified)
    stale = sorted(classified - walked)
    assert missing == [], f"unclassified Layer names (add a 2.3b row): {missing}"
    assert stale == [], f"stale classification rows (field/property gone): {stale}"


def test_mirrored_row_members_are_declared_mirrored() -> None:
    for name in MIRRORED_PER_CALL_NUMERIC_FIELDS:
        assert LAYER_SOURCE_ROWS[name] == ROW_MIRRORED_PER_CALL


# ---------------------------------------------------------------------------
# 2.4(i) record-derived image exclusion pins
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_annotations.py, _data_classes/trace.py, _visualization/_encoding.py ....
def test_annotation_image_node_excluded_from_channel_and_domain(tmp_path: Path) -> None:
    """An annotation-image node contributes NO channel attrs and NO domain values."""

    model = SmallMLP()
    log = tl.trace(model, torch.randn(3, 4))
    image_path = tmp_path / "node.png"
    from PIL import Image

    Image.new("RGB", (8, 8), "red").save(image_path)
    target = next(layer for layer in log._layer_list if layer.type == "relu")
    log.annotate(target.layer_label, image=str(image_path))

    def source(node: Any) -> float:
        # An extreme value: if the image node leaked into the domain, every
        # other node's normalized color would move.
        return 1e9 if record_type(node) == "relu" else float(node.raw_index)

    _draw(log, tmp_path, color_by=source)
    state = _state(log)
    assert not any(key.startswith("relu") for key in state.colors)
    assert state.domain is not None
    assert state.domain[1] < 1e9  # the image node's value never entered the domain


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py, _visualization/_render_nodes.py.
def test_visualizer_path_node_excluded(tmp_path: Path) -> None:
    model = SmallMLP()
    log = tl.trace(model, torch.randn(3, 4))
    image_path = tmp_path / "vp.png"
    from PIL import Image

    Image.new("RGB", (8, 8), "blue").save(image_path)
    target = next(layer for layer in log._layer_list if layer.type == "relu")
    target.visualizer_path = str(image_path)

    _draw(log, tmp_path, color_by=lambda node: float(node.raw_index))
    state = _state(log)
    assert not any(key.startswith("relu") for key in state.colors)


# ---------------------------------------------------------------------------
# Colormap + theme
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/bundle_diff.py.
def test_bundle_diff_consumes_shared_interpolation() -> None:
    from torchlens._visualization.bundle_diff import _delta_color

    assert _delta_color(0.0, 1.0) == "#FFFFFF"
    assert _delta_color(1.0, 1.0) == interpolate_hex("#FFFFFF", "#C93F3F", 1.0)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py, _visualization/themes.py.
def test_dark_theme_uses_dark_ramp(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """Ramps resolve through the SKIN's 3-anchor record (N4); the dark low
    anchor is deliberately off-ground (never equal to the default fill --
    the white-on-white absence defect's dark analog)."""

    from torchlens._visualization.themes import resolve_theme

    _draw(mlp_log, tmp_path, color_by="bytes", theme="dark")
    state = _state(mlp_log)
    dark = resolve_theme("dark")
    assert state.ramp == (dark.ramp[0], dark.ramp[2])
    assert state.ramp[1] == DARK_RAMP[1]  # the high anchor is unchanged
    assert state.ramp[0] != dark.default_fill
    _draw(mlp_log, tmp_path, color_by="bytes")
    light = resolve_theme("torchlens")
    assert _state(mlp_log).ramp == (light.ramp[0], light.ramp[2])
    assert LIGHT_RAMP[1] == light.ramp[2]  # endpoint continuity with the module ramp


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-q"])


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_encoding.py, _visualization/_render_dot.py.
def test_dagua_renderer_with_channel_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """The alternate label path never silently drops an active channel."""

    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, color_by="bytes", renderer="dagua")
    assert excinfo.value.fields["code"] == "encoding_requires_dot_layout"
