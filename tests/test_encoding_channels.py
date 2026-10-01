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
from torchlens.data_classes._layer_spec import _LAYER_MIRROR_SPEC, _LAYER_STATE_ORDER
from torchlens.data_classes.layer import Layer
from torchlens.visualization._encoding import (
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
from torchlens.visualization.node_spec import NodeSpec

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
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def _state(log: tl.Trace) -> Any:
    """Return the last draw's encoding state (session diagnostic)."""

    return log._last_encoding_state


def _varying_layer(log: tl.Trace) -> Layer:
    return next(layer for layer in log.layer_logs.values() if len(layer.ops) > 1)


# ---------------------------------------------------------------------------
# Channel resolution (option validation)
# ---------------------------------------------------------------------------


def test_resolve_none_is_inactive() -> None:
    assert resolve_color_by(None) is None


def test_resolve_builtin_and_field_and_callable() -> None:
    assert resolve_color_by("time").source_kind == "builtin"
    assert resolve_color_by("flops_forward").source_kind == "field"
    assert resolve_color_by(lambda node: 1.0).source_kind == "callable"


def test_unknown_source_refuses_at_option_validation() -> None:
    with pytest.raises(Exception) as excinfo:
        resolve_color_by("no_such_field_anywhere")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_non_string_non_callable_refuses() -> None:
    with pytest.raises(Exception) as excinfo:
        resolve_color_by(42)
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_nan_overlay_is_not_a_scalar_color_source() -> None:
    """'nan' is a boolean overlay, not a scalar channel source."""

    with pytest.raises(Exception) as excinfo:
        resolve_color_by("nan")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


# ---------------------------------------------------------------------------
# Phase A/B end-to-end: encoded fills, callable-once pin, opt-in no-churn
# ---------------------------------------------------------------------------


def test_color_by_encodes_fillcolors(mlp_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(mlp_log, tmp_path, color_by="bytes")
    state = _state(mlp_log)
    assert state is not None and state.populated
    assert state.colors, "no node was encoded"
    for fill in state.colors.values():
        assert fill in dot


def test_callable_invoked_exactly_once_per_visible_eligible_node(
    mlp_log: tl.Trace, tmp_path: Path
) -> None:
    """The prepass counting pin (memo 2.3 Phase A)."""

    calls: dict[str, int] = {}

    def source(node: Any) -> float:
        key = str(node.layer_label)
        calls[key] = calls.get(key, 0) + 1
        return float(node.raw_index)

    _draw(mlp_log, tmp_path, color_by=source)
    state = _state(mlp_log)
    assert calls, "callable never invoked"
    assert all(count == 1 for count in calls.values()), calls
    assert len(calls) == state.eligible_count


def test_plain_draw_untouched_by_channel_machinery(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """Opt-in -> no churn: plain draw carries no channel artifacts."""

    dot = _draw(mlp_log, tmp_path)
    assert _state(mlp_log) is None
    assert "cluster_torchlens_encoding_legend" not in dot
    assert "cluster_torchlens_legend" not in dot  # AUTO + no channel = no legend


def test_request_hash_ignores_channel_fields() -> None:
    """Channels are presentation-only: the collapse-planning hash is untouched."""

    from torchlens.visualization.request import ResolvedRenderRequest

    base = ResolvedRenderRequest(vis_mode="rolled")
    with_channel = ResolvedRenderRequest(vis_mode="rolled", color_by="bytes", encoding=object())
    assert hash(base) == hash(with_channel)


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


def test_bool_value_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, color_by=lambda node: True)
    assert excinfo.value.fields["code"] == "encoding_value_invalid"


def test_non_scalar_tensor_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, color_by=lambda node: torch.zeros(3))
    assert excinfo.value.fields["code"] == "encoding_value_invalid"


def test_one_element_tensor_accepted(mlp_log: tl.Trace, tmp_path: Path) -> None:
    _draw(mlp_log, tmp_path, color_by=lambda node: torch.tensor([2.5]))
    state = _state(mlp_log)
    assert state.colors
    assert NOTE_CONSTANT in state.notes  # constant domain -> midpoint + note


def test_callable_raise_chains_typed(mlp_log: tl.Trace, tmp_path: Path) -> None:
    class Boom(RuntimeError):
        pass

    def source(node: Any) -> float:
        raise Boom("user bug")

    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, color_by=source)
    assert excinfo.value.fields["code"] == "encoding_callable_error"
    assert isinstance(excinfo.value.__cause__, Boom)


def test_non_finite_degrades_to_unencoded(mlp_log: tl.Trace, tmp_path: Path) -> None:
    def source(node: Any) -> float:
        return float("nan") if node.layer_type == "relu" else float(node.raw_index)

    _draw(mlp_log, tmp_path, color_by=source)
    state = _state(mlp_log)
    assert NOTE_NA_UNENCODED in state.notes
    assert not any(key.startswith("relu") for key in state.colors)


def test_absent_field_degrades_to_unencoded(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """grad is absent (no backward ran): every node stays unencoded, no raise."""

    _draw(mlp_log, tmp_path, color_by="grad_norm")
    state = _state(mlp_log)
    assert state.colors == {}
    assert NOTE_NA_UNENCODED in state.notes


# ---------------------------------------------------------------------------
# Engine-resolution fence matrix (memo 2.1)
# ---------------------------------------------------------------------------


def test_explicit_rank_with_channel_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, color_by="bytes", layout="rank")
    assert excinfo.value.fields["code"] == "encoding_requires_dot_layout"


def test_explicit_rank_without_channel_still_works(mlp_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(mlp_log, tmp_path, layout="rank")
    assert dot is not None


def test_explicit_dot_with_channel_works(mlp_log: tl.Trace, tmp_path: Path) -> None:
    _draw(mlp_log, tmp_path, color_by="bytes", layout="dot")
    assert _state(mlp_log).colors


def test_auto_small_graph_uses_dot_no_notice(
    mlp_log: tl.Trace, tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    _draw(mlp_log, tmp_path, color_by="bytes", layout="auto")
    assert _state(mlp_log).colors
    assert not [w for w in recwarn if "encoding channel" in str(w.message)]


def test_auto_large_graph_forces_dot_with_notice(
    mlp_log: tl.Trace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUTO + cost>threshold + channel: dot forced, notice names channel+cost."""

    from torchlens.visualization._rank_layout_internal import layout as rank_layout

    monkeypatch.setattr(rank_layout, "RANK_LAYOUT_COST_THRESHOLD", 0)
    with pytest.warns(UserWarning, match="color_by"):
        _draw(mlp_log, tmp_path, color_by="bytes", layout="auto")
    assert _state(mlp_log).colors  # rendered on dot: channel applied


# ---------------------------------------------------------------------------
# Legend visibility rule + content (memo 2.3)
# ---------------------------------------------------------------------------


def test_legend_auto_with_channel_emits_channel_only(mlp_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(mlp_log, tmp_path, color_by="bytes")
    assert "cluster_torchlens_encoding_legend" in dot
    assert "cluster_torchlens_legend" not in dot  # channel-only, not the role legend
    assert "linear min-max" in dot
    assert "color_by: bytes" in dot


def test_legend_true_with_channel_emits_both(mlp_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(mlp_log, tmp_path, color_by="bytes", show_legend=True)
    assert "cluster_torchlens_legend" in dot
    assert "cluster_torchlens_encoding_legend" in dot


def test_legend_false_honored_even_with_channel(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """Explicit False is a deliberate act: no legend, encoding undisclosed."""

    dot = _draw(mlp_log, tmp_path, color_by="bytes", show_legend=False)
    assert "cluster_torchlens_encoding_legend" not in dot
    assert "cluster_torchlens_legend" not in dot


def test_legend_true_without_channel_keeps_todays_meaning(
    mlp_log: tl.Trace, tmp_path: Path
) -> None:
    dot = _draw(mlp_log, tmp_path, show_legend=True)
    assert "cluster_torchlens_legend" in dot
    assert "cluster_torchlens_encoding_legend" not in dot


def test_show_legend_rejects_non_tri_state(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, show_legend="yes")
    assert excinfo.value.fields["code"] == "visualization_bool_option_invalid"


def test_channel_legend_text_is_escaped(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """E1-corpus reuse: a '<lambda>' display name escapes at the choke point."""

    dot = _draw(mlp_log, tmp_path, color_by=lambda node: float(node.raw_index))
    assert "callable &lt;lambda&gt;" in dot
    assert "callable <lambda>" not in dot


# ---------------------------------------------------------------------------
# Conflict matrix (memo 2.4)
# ---------------------------------------------------------------------------


def test_channel_fill_wins_over_role_fill(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """Encoded input/output record nodes get the channel fill, not role fill."""

    dot = _draw(mlp_log, tmp_path, color_by=lambda node: float(node.raw_index))
    state = _state(mlp_log)
    input_key = next(key for key in state.colors if key.startswith("input"))
    assert state.colors[input_key] in dot


def test_user_node_spec_fn_overrides_channel(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """The user callback runs AFTER the channel and may veto its fill."""

    def node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec | None:
        if layer_log.layer_type == "relu":
            return default_spec.replace(fillcolor="#ABCDEF")
        return None

    dot = _draw(mlp_log, tmp_path, color_by="bytes", node_spec_fn=node_spec_fn)
    assert "#ABCDEF" in dot


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
    assert "#FF00FF" in site_line  # intervention site border color


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


def test_varying_reconciled_field_unencoded_with_note(
    varying_log: tl.Trace, tmp_path: Path
) -> None:
    """Marker-varying bytes on a rolled node: unencoded + legend note."""

    layer = _varying_layer(varying_log)
    assert "activation_memory" in layer.annotations["varying_across_passes"]
    dot = _draw(varying_log, tmp_path, color_by="activation_memory", vis_mode="rolled")
    state = _state(varying_log)
    assert layer.layer_label not in state.colors
    assert NOTE_VARIES in state.notes
    assert "varies across passes" in dot  # disclosed in the drawn legend


def test_uniform_multipass_field_encodes(uniform_log: tl.Trace, tmp_path: Path) -> None:
    """No marker -> all passes agree -> exact value encodes normally."""

    layer = _varying_layer(uniform_log)
    assert "varying_across_passes" not in layer.annotations
    _draw(uniform_log, tmp_path, color_by="activation_memory", vis_mode="rolled")
    state = _state(uniform_log)
    assert layer.layer_label in state.colors
    assert NOTE_VARIES not in state.notes


def test_per_pass_builtin_degrades_on_rolled(varying_log: tl.Trace, tmp_path: Path) -> None:
    """'time' (func_duration) is per-pass: rolled nodes stay unencoded."""

    layer = _varying_layer(varying_log)
    _draw(varying_log, tmp_path, color_by="time", vis_mode="rolled")
    state = _state(varying_log)
    assert layer.layer_label not in state.colors
    assert NOTE_NA_UNENCODED in state.notes


def test_single_pass_layers_unaffected_by_allowlist(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Single-pass layers in the same graph encode normally."""

    _draw(varying_log, tmp_path, color_by="activation_memory", vis_mode="rolled")
    state = _state(varying_log)
    single_pass = [
        layer.layer_label
        for layer in varying_log.layer_logs.values()
        if len(layer.ops) == 1 and layer.activation_memory
    ]
    assert any(label in state.colors for label in single_pass)


def test_callable_on_rolled_bypasses_with_disclosure(varying_log: tl.Trace, tmp_path: Path) -> None:
    """A callable asserts its own aggregate semantics; legend discloses it."""

    layer = _varying_layer(varying_log)
    dot = _draw(
        varying_log,
        tmp_path,
        color_by=lambda node: float(node.total_flops_forward or 0),
        vis_mode="rolled",
    )
    state = _state(varying_log)
    assert layer.layer_label in state.colors
    assert "value from user callable" in dot


def test_callable_per_pass_read_on_rolled_chains_tripwire(
    varying_log: tl.Trace, tmp_path: Path
) -> None:
    """A per-pass read inside a callable surfaces as encoding_callable_error."""

    with pytest.raises(Exception) as excinfo:
        _draw(
            varying_log,
            tmp_path,
            color_by=lambda node: float(node.func_duration),
            vis_mode="rolled",
        )
    assert excinfo.value.fields["code"] == "encoding_callable_error"


def test_alias_rows_flops(varying_log: tl.Trace, tmp_path: Path) -> None:
    """ALIAS PIN: flops_forward unencodes; total_flops_forward encodes.

    Different quantities carry different verdicts DELIBERATELY: the per-pass
    value has no single value on a varying recurrence, while the trace total
    is exact — both disclosed, neither silent. A regression toward either
    silent encoding or blanket refusal fails here.
    """

    layer = _varying_layer(varying_log)
    _draw(varying_log, tmp_path, color_by="flops_forward", vis_mode="rolled")
    per_pass_state = _state(varying_log)
    assert layer.layer_label not in per_pass_state.colors
    assert NOTE_VARIES in per_pass_state.notes

    dot = _draw(varying_log, tmp_path, color_by="total_flops_forward", vis_mode="rolled")
    total_state = _state(varying_log)
    assert layer.layer_label in total_state.colors
    assert any("total_flops_forward" in line for line in total_state.aggregation_lines)
    assert "total across passes" in dot  # mandatory aggregation legend line


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
        dot = _draw(varying_log, tmp_path, color_by=source, vis_mode="rolled")
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
    from torchlens.postprocess.finalization import (
        _MULTIPASS_BYTES_FIELDS,
        _MULTIPASS_FLOPS_FIELDS,
        _MULTIPASS_SHAPE_FIELDS,
    )

    reconciled = (
        set(_MULTIPASS_BYTES_FIELDS) | set(_MULTIPASS_FLOPS_FIELDS) | set(_MULTIPASS_SHAPE_FIELDS)
    )
    assert not (MIRRORED_PER_CALL_NUMERIC_FIELDS & reconciled)


def test_unclassified_source_on_rolled_refuses(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Allowlist default: an Op-only field with no rolled row refuses typed."""

    # 'num_inputs' is a NUMERIC Op-schema field (legal at option validation,
    # encodable on single-pass nodes) with no declared Layer rolled row.
    from torchlens.constants import LAYER_PASS_LOG_FIELD_ORDER

    assert "num_inputs" in LAYER_PASS_LOG_FIELD_ORDER
    assert "num_inputs" not in LAYER_SOURCE_ROWS
    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, color_by="num_inputs", vis_mode="rolled")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"
    assert "rolled" in str(excinfo.value)


def test_unrolled_per_pass_nodes_encode_their_own_pass(
    varying_log: tl.Trace, tmp_path: Path
) -> None:
    """Unrolled mode: per-pass Op nodes of the varying layer encode per-pass."""

    layer = _varying_layer(varying_log)
    _draw(varying_log, tmp_path, color_by="activation_memory", vis_mode="unrolled")
    state = _state(varying_log)
    pass_keys = [key for key in state.values if key.startswith(layer.layer_label + ":")]
    assert len(pass_keys) == len(layer.ops)
    assert len({state.values[key] for key in pass_keys}) > 1  # genuinely per-pass


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


def test_annotation_image_node_excluded_from_channel_and_domain(tmp_path: Path) -> None:
    """An annotation-image node contributes NO channel attrs and NO domain values."""

    model = SmallMLP()
    log = tl.trace(model, torch.randn(3, 4))
    image_path = tmp_path / "node.png"
    from PIL import Image

    Image.new("RGB", (8, 8), "red").save(image_path)
    target = next(layer for layer in log.layer_list if layer.layer_type == "relu")
    log.annotate(target.layer_label, image=str(image_path))

    def source(node: Any) -> float:
        # An extreme value: if the image node leaked into the domain, every
        # other node's normalized color would move.
        return 1e9 if node.layer_type == "relu" else float(node.raw_index)

    _draw(log, tmp_path, color_by=source)
    state = _state(log)
    assert not any(key.startswith("relu") for key in state.colors)
    assert state.domain is not None
    assert state.domain[1] < 1e9  # the image node's value never entered the domain


def test_visualizer_path_node_excluded(tmp_path: Path) -> None:
    model = SmallMLP()
    log = tl.trace(model, torch.randn(3, 4))
    image_path = tmp_path / "vp.png"
    from PIL import Image

    Image.new("RGB", (8, 8), "blue").save(image_path)
    target = next(layer for layer in log.layer_list if layer.layer_type == "relu")
    target.visualizer_path = str(image_path)

    _draw(log, tmp_path, color_by=lambda node: float(node.raw_index))
    state = _state(log)
    assert not any(key.startswith("relu") for key in state.colors)


# ---------------------------------------------------------------------------
# Colormap + theme
# ---------------------------------------------------------------------------


def test_interpolate_hex_endpoints_and_clamp() -> None:
    assert interpolate_hex("#000000", "#FFFFFF", 0.0) == "#000000"
    assert interpolate_hex("#000000", "#FFFFFF", 1.0) == "#FFFFFF"
    assert interpolate_hex("#000000", "#FFFFFF", -1.0) == "#000000"
    assert interpolate_hex("#000000", "#FFFFFF", 2.0) == "#FFFFFF"


def test_bundle_diff_consumes_shared_interpolation() -> None:
    from torchlens.visualization.bundle_diff import _delta_color

    assert _delta_color(0.0, 1.0) == "#FFFFFF"
    assert _delta_color(1.0, 1.0) == interpolate_hex("#FFFFFF", "#C93F3F", 1.0)


def test_dark_theme_uses_dark_ramp(mlp_log: tl.Trace, tmp_path: Path) -> None:
    _draw(mlp_log, tmp_path, color_by="bytes", vis_theme="dark")
    state = _state(mlp_log)
    assert state.ramp == DARK_RAMP
    _draw(mlp_log, tmp_path, color_by="bytes")
    assert _state(mlp_log).ramp == LIGHT_RAMP


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-q"])


def test_dagua_renderer_with_channel_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """The alternate label path never silently drops an active channel."""

    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, color_by="bytes", vis_renderer="dagua")
    assert excinfo.value.fields["code"] == "encoding_requires_dot_layout"
