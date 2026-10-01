"""L5 M2 pins: size encoding channel (``size_by`` + ``scale=``), D4 default-applied.

Covers the design-memo 3.1 test list plus the 2.2 "dims" typed-shape-path
pins, the 2.4(ii) funnel image-drop rule, the C2(b) width/height/fixedsize
writer discipline, and the 2.3b rolled-aggregate size verdicts (size REFUSES
where color degrades).

D4 STATUS: unruled at this merge -- the shipped mapping is the METAPLAN
default (sqrt + conservative area-only C2 mapping + typed refusal on rolled
varying sources), marked default-applied.

HONESTY: the rolled-varying refusal is a tripwire — a size encoding must
never imply a single value it cannot certify. Never weaken it to make a
render pass.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._encoding import (
    DEFAULT_NODE_HEIGHT_IN,
    DEFAULT_NODE_WIDTH_IN,
    LAYER_SOURCE_ROWS,
    NOTE_CALLABLE,
    SIZE_BY_MAX_AREA_MULT,
    _non_batch_numel,
    resolve_size_by,
    resolve_size_scale,
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


class SpatialVarying(nn.Module):
    """Recurrent conv whose NON-batch output shape varies per pass.

    Per-pass conv outputs: (1,8,14,14) / (1,8,10,10) / (1,8,6,6) — distinct
    non-batch numels per pass (unrolled sizing) and a ``shape`` variation
    marker on the rolled aggregate (the 3.1 refusal case).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(())
        for size in (16, 12, 8):
            total = total + self.conv(x[..., :size, :size]).sum()
        return total


class InteriorVarying(nn.Module):
    """First == last pass shape with interior variation (4, 2, 4 rows).

    The any-variation reconciler marker catches this; a first-vs-last
    summary would miss it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum() * 0
        for n in (4, 2, 4):
            total = total + self.fc(x[:n]).sum()
        return total


class UniformRecurrent(nn.Module):
    """Fixed-shape recurrence: multi-pass with NO varying marker."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.relu(self.fc(x))
        return x.sum()


class VaryingRecurrent(nn.Module):
    """Variable-length recurrence (batch shrinks): varying bytes/flops marker."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n in (4, 3, 2):
            x = torch.relu(self.fc(x[:n]))
        return x.sum()


@pytest.fixture(scope="module")
def mlp_log() -> Any:
    log = tl.trace(SmallMLP(), torch.randn(3, 4))
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def spatial_log() -> Any:
    log = tl.trace(SpatialVarying(), torch.randn(1, 3, 16, 16))
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def interior_log() -> Any:
    log = tl.trace(InteriorVarying(), torch.randn(4, 4))
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
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def _state(log: tl.Trace) -> Any:
    return log._last_encoding_state


# ---------------------------------------------------------------------------
# Option validation (resolution + scale vocabulary)
# ---------------------------------------------------------------------------


def test_resolve_none_is_inactive() -> None:
    assert resolve_size_by(None) is None


def test_resolve_dims_field_and_callable_kinds() -> None:
    assert resolve_size_by("dims").source_kind == "dims"
    assert resolve_size_by("activation_memory").source_kind == "field"
    assert resolve_size_by(lambda node: 1.0).source_kind == "callable"


def test_unknown_source_refuses_at_option_validation() -> None:
    with pytest.raises(Exception) as excinfo:
        resolve_size_by("no_such_field_anywhere")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_non_string_non_callable_refuses() -> None:
    with pytest.raises(Exception) as excinfo:
        resolve_size_by(42)
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_scalar_builtins_are_not_size_sources() -> None:
    """Memo 2.2: only color_by accepts the node_overlay scalar builtins."""

    with pytest.raises(Exception) as excinfo:
        resolve_size_by("magnitude")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_scale_defaults_to_sqrt() -> None:
    assert resolve_size_scale(None, size_by_active=True) == "sqrt"
    assert resolve_size_scale(None, size_by_active=False) == "sqrt"


def test_scale_closed_vocabulary() -> None:
    assert resolve_size_scale("linear", size_by_active=True) == "linear"
    with pytest.raises(Exception) as excinfo:
        resolve_size_scale("log", size_by_active=True)
    assert excinfo.value.fields["code"] == "encoding_scale_invalid"


def test_scale_without_size_by_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, scale="sqrt")
    assert excinfo.value.fields["code"] == "scale_requires_size_by"


# ---------------------------------------------------------------------------
# "dims" typed shape path (memo 2.2) + D4 default mapping (C2)
# ---------------------------------------------------------------------------


def test_non_batch_numel_mapping() -> None:
    """C2 mapping pins: torch.Size/tuple parity, empty/rank-0/rank-1 default."""

    assert _non_batch_numel(()) == 1.0
    assert _non_batch_numel((5,)) == 1.0
    assert _non_batch_numel((2, 3)) == 3.0
    assert _non_batch_numel((2, 3, 4)) == 12.0
    assert _non_batch_numel(tuple(torch.Size((2, 3, 4)))) == 12.0


def test_size_by_dims_encodes_minimum_geometry(mlp_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(mlp_log, tmp_path, size_by="dims")
    state = _state(mlp_log)
    assert state.sizes, "no node was size-encoded"
    assert "fixedsize=false" in dot
    assert "width=" in dot and "height=" in dot
    # Fonts never scale: the channel injects no fontsize attribute.
    plain = _draw(mlp_log, tmp_path)
    assert dot.count("fontsize") == plain.count("fontsize")


def test_size_area_clamped_to_max_mult(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """Encoded area spans [1x .. SIZE_BY_MAX_AREA_MULT x] the default area."""

    _draw(mlp_log, tmp_path, size_by="dims")
    default_area = DEFAULT_NODE_WIDTH_IN * DEFAULT_NODE_HEIGHT_IN
    for width, height in _state(mlp_log).sizes.values():
        area = width * height
        assert area >= default_area * 0.99
        assert area <= default_area * SIZE_BY_MAX_AREA_MULT * 1.01


def test_plain_draw_untouched_by_size_machinery(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """STRICTLY OPT-IN: no width/height/fixedsize emissions on plain draw."""

    dot = _draw(mlp_log, tmp_path)
    assert "fixedsize" not in dot
    assert not re.search(r"\bwidth=", dot)


def test_scale_linear_vs_sqrt_changes_geometry(mlp_log: tl.Trace, tmp_path: Path) -> None:
    _draw(mlp_log, tmp_path, size_by="dims", scale="sqrt")
    sqrt_sizes = dict(_state(mlp_log).sizes)
    _draw(mlp_log, tmp_path, size_by="dims", scale="linear")
    linear_sizes = dict(_state(mlp_log).sizes)
    assert sqrt_sizes != linear_sizes
    # Extremes agree (min-max normalization pins both endpoints)...
    sqrt_order = sorted(sqrt_sizes, key=lambda key: sqrt_sizes[key])
    linear_order = sorted(linear_sizes, key=lambda key: linear_sizes[key])
    # ...and the ordering is identical (both transforms are monotone).
    assert sqrt_order == linear_order


def test_per_pass_op_nodes_size_by_their_own_pass(spatial_log: tl.Trace, tmp_path: Path) -> None:
    """Unrolled per-pass nodes of a shape-varying layer size per-pass."""

    _draw(spatial_log, tmp_path, size_by="dims", vis_mode="unrolled")
    state = _state(spatial_log)
    conv_sizes = {key: value for key, value in state.sizes.items() if "conv" in key}
    assert len(conv_sizes) == 3
    assert len(set(conv_sizes.values())) == 3, conv_sizes


def test_callable_shape_return_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """Callables must return scalars; 'dims' is the only shape-valued source."""

    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_by=lambda node: tuple(node.shape))
    assert excinfo.value.fields["code"] == "encoding_value_invalid"
    assert "dims" in str(excinfo.value)


def test_bool_value_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_by=lambda node: bool(node.is_input))
    assert excinfo.value.fields["code"] == "encoding_value_invalid"


def test_callable_raise_chains_typed(mlp_log: tl.Trace, tmp_path: Path) -> None:
    class Boom(RuntimeError):
        pass

    def source(node: Any) -> float:
        raise Boom("bad size")

    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_by=source)
    assert excinfo.value.fields["code"] == "encoding_callable_error"
    assert isinstance(excinfo.value.__cause__, Boom)


def test_negative_value_under_sqrt_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_by=lambda node: -1.0)
    assert excinfo.value.fields["code"] == "encoding_value_invalid"
    # Same values under scale='linear' are legal (min-max handles negatives).
    _draw(mlp_log, tmp_path, size_by=lambda node: -float(node.raw_index), scale="linear")
    assert _state(mlp_log).sizes


def test_absent_field_degrades_to_default_box(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """2.2 gap rule (single-pass): absent/None values leave the node unsized."""

    _draw(mlp_log, tmp_path, size_by=lambda node: None)
    state = _state(mlp_log)
    assert state.sizes == {}


# ---------------------------------------------------------------------------
# Rolled-varying refusal — THE D4 detector (memo 3.1)
# ---------------------------------------------------------------------------


def test_rolled_varying_dims_refuses(spatial_log: tl.Trace, tmp_path: Path) -> None:
    """Refusal fires BEFORE dims resolution (ordering pin: no '3..4' strings)."""

    with pytest.raises(Exception) as excinfo:
        _draw(spatial_log, tmp_path, size_by="dims", vis_mode="rolled")
    assert excinfo.value.fields["code"] == "size_by_rolled_varying"


def test_interior_only_variation_refuses(interior_log: tl.Trace, tmp_path: Path) -> None:
    """first==last with interior variation — the case a first/last summary misses."""

    with pytest.raises(Exception) as excinfo:
        _draw(interior_log, tmp_path, size_by="dims", vis_mode="rolled")
    assert excinfo.value.fields["code"] == "size_by_rolled_varying"


def test_uniform_multipass_dims_no_refusal(uniform_log: tl.Trace, tmp_path: Path) -> None:
    _draw(uniform_log, tmp_path, size_by="dims", vis_mode="rolled")
    assert _state(uniform_log).sizes


def test_single_pass_layers_unaffected(mlp_log: tl.Trace, tmp_path: Path) -> None:
    _draw(mlp_log, tmp_path, size_by="activation_memory", vis_mode="rolled")
    assert _state(mlp_log).sizes


def test_varying_scalar_field_refuses(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Generalized refusal: ANY marker-varying size source, not just shapes."""

    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, size_by="activation_memory", vis_mode="rolled")
    assert excinfo.value.fields["code"] == "size_by_rolled_varying"


def test_uniform_scalar_field_encodes(uniform_log: tl.Trace, tmp_path: Path) -> None:
    _draw(uniform_log, tmp_path, size_by="activation_memory", vis_mode="rolled")
    assert _state(uniform_log).sizes


def test_summed_alias_encodes_with_aggregation_line(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Alias rule: total_activation_memory carries a DIFFERENT verdict than
    activation_memory on the same varying recurrence (different quantities,
    both disclosed)."""

    _draw(varying_log, tmp_path, size_by="total_activation_memory", vis_mode="rolled")
    state = _state(varying_log)
    assert state.sizes
    assert any("total across passes" in line for line in state.size_aggregation_lines)


def test_mirrored_per_call_numeric_refuses(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Widened semantics: an unreconciled first-pass projection refuses."""

    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, size_by="transformed_gradient_memory", vis_mode="rolled")
    assert excinfo.value.fields["code"] == "size_by_rolled_varying"


def test_per_pass_field_refuses_on_rolled(varying_log: tl.Trace, tmp_path: Path) -> None:
    """A per-pass attribute cannot be certified single-valued on a rolled node."""

    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, size_by="func_duration", vis_mode="rolled")
    assert excinfo.value.fields["code"] == "size_by_rolled_varying"


def test_unclassified_source_on_rolled_refuses(varying_log: tl.Trace, tmp_path: Path) -> None:
    from torchlens.constants import LAYER_PASS_LOG_FIELD_ORDER

    assert "num_inputs" in LAYER_PASS_LOG_FIELD_ORDER
    assert "num_inputs" not in LAYER_SOURCE_ROWS
    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, size_by="num_inputs", vis_mode="rolled")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_callable_on_rolled_bypasses_with_disclosure(varying_log: tl.Trace, tmp_path: Path) -> None:
    _draw(varying_log, tmp_path, size_by=lambda node: float(node.num_passes), vis_mode="rolled")
    state = _state(varying_log)
    assert state.sizes
    assert NOTE_CALLABLE in state.size_notes


def test_shape_field_source_names_dims_remedy(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """size_by='shape' hits the wrong-type row and points at 'dims'."""

    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_by="shape")
    assert excinfo.value.fields["code"] == "encoding_value_invalid"
    assert "dims" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Channel composition + engine fence + legend
# ---------------------------------------------------------------------------


def test_color_and_size_compose_freely(mlp_log: tl.Trace, tmp_path: Path) -> None:
    _draw(mlp_log, tmp_path, color_by="bytes", size_by="dims")
    state = _state(mlp_log)
    assert state.colors and state.sizes
    assert state.active_channels() == ("color_by", "size_by")


def test_explicit_rank_with_size_channel_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_by="dims", layout="rank")
    assert excinfo.value.fields["code"] == "encoding_requires_dot_layout"
    assert "size_by" in str(excinfo.value)


def test_dagua_renderer_with_size_channel_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_by="dims", renderer="dagua")
    assert excinfo.value.fields["code"] == "encoding_requires_dot_layout"


def test_legend_states_scale_transform(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """Disclosure contract: every legend drawn states the size transform."""

    dot = _draw(mlp_log, tmp_path, size_by="dims")
    assert "TorchLens encoding" in dot
    assert "sqrt(dims)" in dot
    dot_linear = _draw(mlp_log, tmp_path, size_by="dims", scale="linear")
    assert "linear(dims)" in dot_linear


def test_legend_false_honored_with_size_channel(mlp_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(mlp_log, tmp_path, size_by="dims", show_legend=False)
    assert "cluster_torchlens_encoding_legend" not in dot


def test_request_hash_ignores_size_fields() -> None:
    from torchlens.visualization.request import ResolvedRenderRequest

    base = ResolvedRenderRequest()
    sized = ResolvedRenderRequest(size_by="dims", scale="linear", encoding=object())
    assert hash(base) == hash(sized)


# ---------------------------------------------------------------------------
# 2.4(ii) funnel image-drop rule + extra_attrs power valve (C2(b))
# ---------------------------------------------------------------------------


def test_funnel_drops_size_fields_on_image_specs() -> None:
    from torchlens.visualization._render_leaf import _node_spec_to_graphviz_args

    spec = NodeSpec(lines=["x"], width=2.0, height=1.0, fixedsize="false", image="node.png")
    args = _node_spec_to_graphviz_args(spec)
    assert "width" not in args and "height" not in args
    assert args["image"] == "node.png"


def test_funnel_emits_size_fields_without_image() -> None:
    from torchlens.visualization._render_leaf import _node_spec_to_graphviz_args

    spec = NodeSpec(lines=["x"], width=2.0, height=1.0, fixedsize="false")
    args = _node_spec_to_graphviz_args(spec)
    assert args["width"] == "2.0" and args["height"] == "1.0"
    assert args["fixedsize"] == "false"


def test_extra_attrs_wins_over_nodespec_size_fields() -> None:
    """The power valve wins by merge order — a rule, not an accident."""

    from torchlens.visualization._render_leaf import _node_spec_to_graphviz_args

    spec = NodeSpec(lines=["x"], width=2.0, height=1.0, extra_attrs={"width": "9"})
    args = _node_spec_to_graphviz_args(spec)
    assert args["width"] == "9"
    # And the image-drop rule cannot drop the power valve either.
    spec_image = NodeSpec(lines=["x"], width=2.0, image="node.png", extra_attrs={"width": "9"})
    assert _node_spec_to_graphviz_args(spec_image)["width"] == "9"


def test_user_node_spec_fn_sees_and_can_override_size(mlp_log: tl.Trace, tmp_path: Path) -> None:
    seen_widths: list[Any] = []

    def node_spec_fn(layer_log: Any, default_spec: NodeSpec) -> NodeSpec:
        seen_widths.append(default_spec.width)
        return default_spec.replace(width=None, height=None, fixedsize=None)

    dot = _draw(mlp_log, tmp_path, size_by="dims", node_spec_fn=node_spec_fn)
    assert any(width is not None for width in seen_widths)
    assert "fixedsize" not in dot


def test_annotation_image_node_excluded_from_size_and_domain(tmp_path: Path) -> None:
    """2.4(i) size pin: an annotation-image node contributes NO size attrs and
    NO normalization-domain values."""

    model = SmallMLP()
    log = tl.trace(model, torch.randn(3, 4))
    try:
        image_path = tmp_path / "node.png"
        from PIL import Image

        Image.new("RGB", (8, 8), color=(200, 10, 10)).save(image_path)
        log.annotate("relu_1_2", image=str(image_path))

        def source(node: Any) -> float:
            return 1e9 if "relu" in str(node.layer_label) else float(node.raw_index)

        _draw(log, tmp_path, size_by=source)
        state = _state(log)
        assert not any("relu" in key for key in state.sizes)
        assert state.size_domain is not None and state.size_domain[1] < 1e9
    finally:
        log.cleanup()


# ---------------------------------------------------------------------------
# C2(b) writer discipline: width/height/fixedsize node-arg emission sites
# ---------------------------------------------------------------------------


def test_size_attr_writer_sites_are_the_declared_allowlist() -> None:
    """No TorchLens-internal call site may emit width/height/fixedsize into
    node args on an encodable node (memo C2(b)).

    Grep scope: dict-literal keys and subscript assignments in the graphviz
    node-arg surfaces (visualization/, viz/, repgeom/). The allowlisted
    writers are the record-derived image/montage/helper/container sites (all
    on non-encodable node classes), the four PUBLIC node_spec_fn factories
    (user-slot callbacks, governed by the funnel image-drop rule), and the
    ONE sanctioned NodeSpec funnel. A new writer fails here until it is
    consciously allowlisted.
    """

    package_root = Path(tl.__file__).parent
    scopes = ["visualization", "viz", "repgeom"]
    key_pattern = re.compile(r"[\"'](width|height|fixedsize)[\"']\s*:")
    subscript_pattern = re.compile(r"\[[\"'](width|height|fixedsize)[\"']\]\s*=")
    found: dict[str, int] = {}
    for scope in scopes:
        for path in sorted((package_root / scope).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            count = len(key_pattern.findall(text)) + len(subscript_pattern.findall(text))
            if count:
                found[str(path.relative_to(package_root))] = count
    expected = {
        # Record-derived image branches (2x fixedsize) + input-batch montage
        # (fixedsize/width/height) — non-encodable node classes.
        "visualization/_render_nodes.py": 5,
        # Intervention-hook diamond helper (width/height in extra_attrs).
        "visualization/_render_edges.py": 2,
        # Collapsed container-record node (fixedsize).
        "visualization/_render_flow.py": 1,
        # THE NodeSpec funnel (fixedsize key + width/height emissions).
        "visualization/_render_leaf.py": 3,
        # PUBLIC user-slot node_spec_fn factories (C2(b) allowlist).
        "viz/feature_maps.py": 1,
        # C01 architecture move: the three node-visual writer sites left
        # __init__.py for the dedicated submodule; same sites, same count.
        "repgeom/_node_visuals.py": 3,
        # Stage-0 lens audit reads graphviz plain-output geometry back into a
        # measurement record ("width"/"height" keys) -- a READER of the
        # rendered layout, never a node-arg writer; no node class involved.
        "visualization/lenses/audit/stage0.py": 2,
    }
    assert found == expected, (
        "width/height/fixedsize node-arg writer sites changed; a new internal "
        f"writer on an encodable node class is forbidden (C2(b)). Found: {found}"
    )
