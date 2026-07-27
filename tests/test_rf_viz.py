"""Smoke coverage for receptive-field visualization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import pytest
import torch
from torch import nn
from PIL import Image

import torchlens as tl
from torchlens.receptive_field._errors import AmbiguousInputError, ReceptiveFieldError
from torchlens.receptive_field._types import (
    GradientReceptiveField,
    GridLayout,
    ReceptiveField,
    ReceptiveFieldAxis,
    ReceptiveFieldAlignment,
    ReceptiveFieldBox,
    ReceptiveFieldBoxAxis,
    ReceptiveFieldStatus,
)
from torchlens.receptive_field._viz import show
from torchlens.receptive_field import _rules, node_spec
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult


def _descriptor(
    rank: int, status: ReceptiveFieldStatus = ReceptiveFieldStatus.EXACT
) -> ReceptiveField:
    """Create a compact grid descriptor with the requested spatial rank."""

    shape = (1, 1, *(8 for _ in range(rank)))
    axes = tuple(
        ReceptiveFieldAxis(
            input_axis=index,
            output_axis=index,
            kind="windowed" if index >= 2 else "pointwise",
            input_extent=extent,
            size=3 if index >= 2 else None,
            jump=Fraction(1) if index >= 2 else None,
            center0=Fraction(1) if index >= 2 else None,
            exact=status is ReceptiveFieldStatus.EXACT,
            aligned=True,
            sparse_possible=False,
        )
        for index, extent in enumerate(shape)
    )
    return ReceptiveField(
        op_label="target",
        io_role="input_1",
        input_op_label="input_1_raw",
        input_shape=shape,
        axes=axes,
        layout=GridLayout(
            tuple(axis.kind for axis in axes),
            tuple(range(2, rank + 2)),
            "derived",
            ("test",) * len(shape),
        ),
        status=status,
        alignment=ReceptiveFieldAlignment.ALIGNED,
        batch_coupled=False,
        rule="test",
        notes=(),
    )


def _box(descriptor: ReceptiveField) -> ReceptiveFieldBox:
    """Create a finite centered test box matching one descriptor."""

    return ReceptiveFieldBox(
        op_label="target",
        io_role="input_1",
        unit=(0, 0, *(3 for _ in descriptor.layout.windowed_axes)),
        axes=tuple(
            ReceptiveFieldBoxAxis(
                input_axis=index,
                kind=axis.kind,
                theoretical_start=Fraction(2) if index >= 2 else Fraction(0),
                theoretical_stop=Fraction(5) if index >= 2 else Fraction(extent),
                index_start=2 if index >= 2 else 0,
                index_stop=5 if index >= 2 else extent,
                clipped_start=2 if index >= 2 else 0,
                clipped_stop=5 if index >= 2 else extent,
            )
            for index, (axis, extent) in enumerate(
                zip(descriptor.axes or (), descriptor.input_shape)
            )
        ),
        input_shape=descriptor.input_shape,
        status=descriptor.status,
        exact=descriptor.status is ReceptiveFieldStatus.EXACT,
        clipped=True,
        empty=False,
        covers_input=False,
    )


class _View:
    """Minimal T6-compatible view used to exercise the standalone adapter."""

    def __init__(
        self, descriptor: ReceptiveField, gradient_result: GradientReceptiveField | None = None
    ) -> None:
        """Store one descriptor, box, and optional empirical result."""

        self.per_input: Mapping[str, ReceptiveField] = {descriptor.io_role: descriptor}
        self._box = _box(descriptor)
        self._gradient = gradient_result

    def at(self, unit: Sequence[int], *, input: Any | None = None) -> ReceptiveFieldBox:
        """Return the configured geometric box."""

        _ = unit, input
        return self._box

    def gradient(self, unit: Sequence[int], *, input: Any | None = None) -> GradientReceptiveField:
        """Return the configured empirical result."""

        _ = unit, input
        assert self._gradient is not None
        return self._gradient


def _gradient(descriptor: ReceptiveField) -> GradientReceptiveField:
    """Create a nonuniform full-input gradient payload."""

    values = torch.arange(int(torch.tensor(descriptor.input_shape).prod())).reshape(
        descriptor.input_shape
    )
    return GradientReceptiveField(
        op_label="target",
        io_role=descriptor.io_role,
        unit=(0,) * len(descriptor.input_shape),
        grad=values.float(),
        support_mask=values.bool(),
        support_ranges=tuple((0, extent) for extent in descriptor.input_shape),
        spatial_support_mask=None,
        batch_support=(0,),
        cross_batch_influence=False,
        atol=0.0,
        rtol=0.0,
        nonfinite_count=0,
        warnings=(),
    )


@pytest.mark.smoke
def test_show_1d_strip_and_status_styling() -> None:
    """Render 1-D boxes as strips, with dashed upper bounds distinct from exact boxes."""

    exact = _descriptor(1)
    upper = _descriptor(1, ReceptiveFieldStatus.UPPER_BOUND)
    base = Image.new("RGB", (80, 20), "white")
    exact_image = show(_View(exact), (0, 0, 3), image=base)
    upper_image = show(_View(upper), (0, 0, 3), image=base)
    assert exact_image.size == (80, 20)
    assert exact_image.tobytes() != upper_image.tobytes()


@pytest.mark.smoke
def test_show_2d_overlay_with_gradient_and_unknown_honesty() -> None:
    """Blend a 2-D gradient heatmap and never draw an unknown geometric box."""

    descriptor = _descriptor(2)
    base = Image.new("RGB", (80, 60), "white")
    rendered = show(
        _View(descriptor, _gradient(descriptor)), (0, 0, 3, 3), image=base, gradient=True
    )
    assert rendered.tobytes() != base.tobytes()

    unknown = _descriptor(2, ReceptiveFieldStatus.UNKNOWN)
    assert show(_View(unknown), (0, 0, 3, 3), image=base).tobytes() == base.tobytes()


@pytest.mark.smoke
def test_show_3d_requires_explicit_slice() -> None:
    """Require a plane selection before rendering a 3-D spatial input."""

    descriptor = _descriptor(3)
    with pytest.raises(
        ReceptiveFieldError,
        match=r"3-D receptive-field visualization requires slice=\(axis, index\)\.",
    ):
        show(_View(descriptor), (0, 0, 3, 3, 3), image=Image.new("RGB", (80, 60)))


@pytest.mark.smoke
def test_show_rejects_ambiguous_input() -> None:
    """Reject multi-input display requests unless the IO role is selected."""

    view = _View(_descriptor(2))
    view.per_input = {"input_1": _descriptor(2), "input_2": _descriptor(2)}
    with pytest.raises(AmbiguousInputError, match="Select one reachable input"):
        show(view, (0, 0, 3, 3), image=Image.new("RGB", (80, 60)))


@pytest.mark.smoke
def test_node_spec_draws_ancestor_cone_with_tooltips(tmp_path: Any) -> None:
    """Render the selected input-to-target cone through the standard node-spec hook."""

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    try:

        @_rules.register_rf_rule("conv2d", replace="conv2d" in _rules._RF_RULES)
        def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
            """Emit an exact two-dimensional convolution window."""

            return context.window(kernel=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1))

        trace = tl.trace(nn.Conv2d(1, 1, 3), torch.ones(1, 1, 8, 8))
        target = next(op for op in trace.layer_list if op.func_name == "conv2d")
        output_path = tmp_path / "rf_cone.svg"
        trace.draw(
            node_spec_fn=node_spec(target),
            vis_fileformat="svg",
            vis_save_only=True,
            vis_outpath=str(output_path),
        )
        svg = output_path.read_text(encoding="utf-8")
        assert "RF " in svg
        assert "#ffd8a8" in svg.lower()
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(saved_rules)
        _rules._RF_RULES_EPOCH = saved_epoch
