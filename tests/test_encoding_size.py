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

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._encoding import (
    LAYER_SOURCE_ROWS,
    resolve_size_by,
    resolve_size_scale,
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
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    ).dot


def _state(log: tl.Trace) -> Any:
    return log._last_encoding_state


# ---------------------------------------------------------------------------
# Option validation (resolution + scale vocabulary)
# ---------------------------------------------------------------------------


def test_non_string_non_callable_refuses() -> None:
    with pytest.raises(Exception) as excinfo:
        resolve_size_by(42)
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_scalar_builtins_are_not_size_sources() -> None:
    """Memo 2.2: only color_by accepts the node_overlay scalar builtins."""

    with pytest.raises(Exception) as excinfo:
        resolve_size_by("magnitude")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_scale_closed_vocabulary() -> None:
    assert resolve_size_scale("linear", size_by_active=True) == "linear"
    with pytest.raises(Exception) as excinfo:
        resolve_size_scale("log", size_by_active=True)
    assert excinfo.value.fields["code"] == "encoding_scale_invalid"


# ---------------------------------------------------------------------------
# "dims" typed shape path (memo 2.2) + D4 default mapping (C2)
# ---------------------------------------------------------------------------


def test_per_pass_op_nodes_size_by_their_own_pass(spatial_log: tl.Trace, tmp_path: Path) -> None:
    """Unrolled per-pass nodes of a shape-varying layer size per-pass."""

    _draw(spatial_log, tmp_path, size_by="dims", rolled=False)
    state = _state(spatial_log)
    conv_sizes = {key: value for key, value in state.sizes.items() if "conv" in key}
    assert len(conv_sizes) == 3
    assert len(set(conv_sizes.values())) == 3, conv_sizes


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
    _draw(mlp_log, tmp_path, size_by=lambda node: -float(node.raw_index), size_scale="linear")
    assert _state(mlp_log).sizes


def test_absent_field_degrades_to_default_box(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """2.2 gap rule (single-pass): absent/None values leave the node unsized."""

    _draw(mlp_log, tmp_path, size_by=lambda node: None)
    state = _state(mlp_log)
    assert state.sizes == {}


# ---------------------------------------------------------------------------
# Rolled-varying refusal — THE D4 detector (memo 3.1)
# ---------------------------------------------------------------------------


def test_interior_only_variation_refuses(interior_log: tl.Trace, tmp_path: Path) -> None:
    """first==last with interior variation — the case a first/last summary misses."""

    with pytest.raises(Exception) as excinfo:
        _draw(interior_log, tmp_path, size_by="dims", rolled=True)
    assert excinfo.value.fields["code"] == "size_by_rolled_varying"


def test_varying_scalar_field_refuses(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Generalized refusal: ANY marker-varying size source, not just shapes."""

    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, size_by="activation_memory", rolled=True)
    assert excinfo.value.fields["code"] == "size_by_rolled_varying"


def test_uniform_scalar_field_encodes(uniform_log: tl.Trace, tmp_path: Path) -> None:
    _draw(uniform_log, tmp_path, size_by="activation_memory", rolled=True)
    assert _state(uniform_log).sizes


def test_summed_alias_encodes_with_aggregation_line(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Alias rule: total_activation_memory carries a DIFFERENT verdict than
    activation_memory on the same varying recurrence (different quantities,
    both disclosed)."""

    _draw(varying_log, tmp_path, size_by="total_activation_memory", rolled=True)
    state = _state(varying_log)
    assert state.sizes
    assert any("total across passes" in line for line in state.size_aggregation_lines)


def test_mirrored_per_call_numeric_refuses(varying_log: tl.Trace, tmp_path: Path) -> None:
    """Widened semantics: an unreconciled first-pass projection refuses."""

    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, size_by="transformed_gradient_memory", rolled=True)
    assert excinfo.value.fields["code"] == "size_by_rolled_varying"


def test_per_pass_field_refuses_on_rolled(varying_log: tl.Trace, tmp_path: Path) -> None:
    """A per-pass attribute cannot be certified single-valued on a rolled node."""

    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, size_by="func_duration", rolled=True)
    assert excinfo.value.fields["code"] == "size_by_rolled_varying"


def test_unclassified_source_on_rolled_refuses(varying_log: tl.Trace, tmp_path: Path) -> None:
    from torchlens._constants import LAYER_PASS_LOG_FIELD_ORDER

    assert "num_inputs" in LAYER_PASS_LOG_FIELD_ORDER
    assert "num_inputs" not in LAYER_SOURCE_ROWS
    with pytest.raises(Exception) as excinfo:
        _draw(varying_log, tmp_path, size_by="num_inputs", rolled=True)
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_shape_field_source_names_dims_remedy(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """size_by='shape' hits the wrong-type row and points at 'dims'."""

    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_by="shape")
    assert excinfo.value.fields["code"] == "encoding_value_invalid"
    assert "dims" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Channel composition + engine fence + legend
# ---------------------------------------------------------------------------


def test_dagua_renderer_with_size_channel_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_by="dims", renderer="dagua")
    assert excinfo.value.fields["code"] == "encoding_requires_dot_layout"


# ---------------------------------------------------------------------------
# 2.4(ii) funnel image-drop rule + extra_attrs power valve (C2(b))
# ---------------------------------------------------------------------------


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


def test_scale_without_size_by_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, size_scale="sqrt")
    assert excinfo.value.fields["code"] == "scale_requires_size_by"
