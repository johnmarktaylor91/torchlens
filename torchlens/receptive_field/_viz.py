"""PIL and graph visualization helpers for receptive-field queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING, cast

import torch
from PIL import Image, ImageDraw

from ._errors import AmbiguousInputError, ReceptiveFieldError
from ._types import (
    GradientReceptiveField,
    ReceptiveField,
    ReceptiveFieldBox,
    ReceptiveFieldDirection,
    ReceptiveFieldStatus,
)
from ..viz.node_plots import render_heatmap
from ..visualization.node_spec import NodeSpec, NodeSpecFn

if TYPE_CHECKING:
    from ._types import ReceptiveFieldView
    from ..data_classes.op import Op


_CONE_FILL = "#FFD8A8"
_DIM_FILL = "#E4E7EB"


def show(
    view: "ReceptiveFieldView",
    unit: Sequence[int] | None = None,
    *,
    input: Any | None = None,
    direction: ReceptiveFieldDirection = ReceptiveFieldDirection.RECEPTIVE,
    target: Any | None = None,
    image: Image.Image | None = None,
    gradient: bool = False,
    slice: tuple[int, int] | None = None,
    box_color: str = "#FF3B30",
    alpha: float = 0.6,
    cmap: str = "magma",
) -> Image.Image:
    """Render an honest input-space receptive-field visualization.

    Parameters
    ----------
    view:
        Receptive-field view for the target operation.
    unit:
        Complete output-element index, required when ``gradient`` is true.
    input:
        Input operation handle or exact IO role.
    image:
        Optional base image overriding the captured raw stimulus.
    gradient:
        Whether to alpha-blend the empirical gradient magnitude.
    slice:
        Required ``(input_axis, index)`` plane selection for three spatial axes.
    box_color:
        RGB/CSS color for the geometric-box outline.
    alpha:
        Gradient heatmap opacity in ``[0, 1]``.
    cmap:
        Colormap passed to :func:`torchlens.viz.render_heatmap`.

    Returns
    -------
    PIL.Image.Image
        A standalone RGB visualization.

    Raises
    ------
    ReceptiveFieldError
        If the requested data cannot honestly be rendered as a spatial view.
    ValueError
        If display arguments are invalid.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    selected = target if direction is ReceptiveFieldDirection.PROJECTIVE else input
    descriptor = _select_descriptor(view, selected)
    if gradient and unit is None:
        raise ReceptiveFieldError("gradient=True requires an explicit complete output unit.")
    box = None if unit is None else _view_box(view, unit, descriptor, selected, direction)
    gradient_result = _view_gradient(view, unit, selected, direction) if gradient else None
    spatial_axes = _spatial_axes(descriptor, gradient_result)
    rendered_axes = _rendered_axes(spatial_axes, slice)
    base = _base_image(view, descriptor, image, rendered_axes)
    if gradient_result is not None:
        heatmap = _gradient_image(gradient_result, rendered_axes, slice, base.size, cmap)
        base = Image.blend(base, heatmap, alpha)
    if box is not None and box.status in {
        ReceptiveFieldStatus.EXACT,
        ReceptiveFieldStatus.WHOLE_INPUT,
        ReceptiveFieldStatus.UPPER_BOUND,
    }:
        _draw_box(
            base,
            box,
            rendered_axes,
            dashed=box.status is ReceptiveFieldStatus.UPPER_BOUND,
            color=box_color,
        )
    return base


def node_spec(
    op: "Op",
    *,
    unit: Sequence[int] | None = None,
    input: Any | None = None,
    direction: ReceptiveFieldDirection | str = ReceptiveFieldDirection.RECEPTIVE,
    target: Any | None = None,
    cone: bool = True,
    overlay: bool = True,
    gradient: bool = False,
    dim_others: bool = True,
    box_color: str = "#FF3B30",
    alpha: float = 0.6,
    cmap: str = "magma",
) -> NodeSpecFn:
    """Create a ``Trace.draw`` callback highlighting an RF ancestor cone.

    Parameters
    ----------
    op:
        Target operation whose receptive-field ancestry is shown.
    unit, input, cone, overlay, gradient, dim_others, box_color, alpha, cmap:
        Draw-time RF options. ``unit`` is forwarded only when an input-node
        overlay is requested.

    Returns
    -------
    NodeSpecFn
        Callback suitable for ``Trace.draw(node_spec_fn=...)``.

    Raises
    ------
    AmbiguousInputError
        If more than one model input reaches ``op`` and ``input`` is omitted.
    """

    _ = overlay, gradient, box_color, alpha, cmap
    resolved_direction = ReceptiveFieldDirection(direction)
    selected = target if resolved_direction is ReceptiveFieldDirection.PROJECTIVE else input
    descriptor = _descriptor_for_op(op, selected, resolved_direction)
    trace = op.source_trace
    cone_labels = (
        _projective_cone(op, descriptor.input_op_label)
        if cone and resolved_direction is ReceptiveFieldDirection.PROJECTIVE
        else _ancestor_cone(op, descriptor.io_role)
        if cone
        else frozenset()
    )

    def node_spec_fn(layer: Any, spec: NodeSpec) -> NodeSpec | None:
        """Style one rendered layer according to on-demand cone membership."""

        labels = _rendered_labels(layer)
        in_cone = bool(labels & cone_labels)
        if not in_cone:
            if not dim_others:
                return None
            return spec.replace(fillcolor=_DIM_FILL, fontcolor="#667085", tooltip=spec.tooltip)
        node_descriptor = _descriptor_for_labels(trace, labels, descriptor.io_role)
        tooltip = _tooltip(node_descriptor) if node_descriptor is not None else "RF ancestor cone"
        return spec.replace(fillcolor=_CONE_FILL, color="#D97706", penwidth=2.0, tooltip=tooltip)

    return node_spec_fn


def _select_descriptor(view: "ReceptiveFieldView", selected: Any | None) -> ReceptiveField:
    """Resolve one descriptor from a view without guessing a multi-input role."""

    per_input = view.per_input
    if selected is None:
        if len(per_input) != 1:
            raise AmbiguousInputError(
                "Select one reachable input; available IO roles: " + ", ".join(per_input)
            )
        return next(iter(per_input.values()))
    role = selected if isinstance(selected, str) else getattr(selected, "io_role", None)
    if role not in per_input:
        raise ReceptiveFieldError(f"Input {role!r} is not reachable from this target.")
    return per_input[cast(str, role)]


def _view_box(
    view: "ReceptiveFieldView",
    unit: Sequence[int],
    descriptor: ReceptiveField,
    selected: Any | None,
    direction: ReceptiveFieldDirection,
) -> ReceptiveFieldBox:
    """Project a complete output unit to ``at()`` windowed coordinates."""

    if descriptor.axes is None:
        raise ReceptiveFieldError("Per-unit geometry is unavailable; use .gradient() instead.")
    windowed_axes = tuple(axis.output_axis for axis in descriptor.axes if axis.kind == "windowed")
    if any(axis is None for axis in windowed_axes):
        raise ReceptiveFieldError("The derived layout is ambiguous; use .gradient() instead.")
    coordinates = tuple(unit[cast(int, axis)] for axis in windowed_axes)
    try:
        if direction is ReceptiveFieldDirection.RECEPTIVE:
            return cast(ReceptiveFieldBox, view.at(coordinates, input=selected))
        if direction is ReceptiveFieldDirection.PROJECTIVE:
            return cast(
                ReceptiveFieldBox, view.at(coordinates, direction=direction, target=selected)
            )
        return cast(ReceptiveFieldBox, view.at(coordinates, direction=direction, input=selected))
    except TypeError as exc:
        raise ReceptiveFieldError(
            "ReceptiveFieldView.at() must support the complete-unit show() contract."
        ) from exc


def _view_gradient(
    view: "ReceptiveFieldView",
    unit: Sequence[int] | None,
    selected: Any | None,
    direction: ReceptiveFieldDirection,
) -> GradientReceptiveField:
    """Obtain and disambiguate one empirical gradient result."""

    assert unit is not None
    if direction is ReceptiveFieldDirection.RECEPTIVE:
        result = view.gradient(tuple(unit), input=selected)
    else:
        result = view.gradient(tuple(unit), direction=direction, target=selected)
    if isinstance(result, Mapping):
        if len(result) != 1:
            raise AmbiguousInputError("Select one reachable input before rendering a gradient.")
        return next(iter(result.values()))
    return cast(GradientReceptiveField, result)


def _spatial_axes(
    descriptor: ReceptiveField, gradient: GradientReceptiveField | None
) -> tuple[int, ...]:
    """Return layout-derived spatial axes, falling back to a gradient mask only."""

    if descriptor.axes is not None:
        axes = tuple(axis.input_axis for axis in descriptor.axes if axis.kind == "windowed")
        if axes:
            return axes
    if gradient is not None and gradient.spatial_support_mask is not None:
        return tuple(range(gradient.spatial_support_mask.ndim))
    raise ReceptiveFieldError(
        "Input layout has no renderable spatial grid; use the gradient support mask or indices()."
    )


def _rendered_axes(
    spatial_axes: tuple[int, ...], slice_spec: tuple[int, int] | None
) -> tuple[int, ...]:
    """Validate rank-specific visualization selection and return visible axes."""

    rank = len(spatial_axes)
    if rank == 1:
        return spatial_axes
    if rank == 2:
        return spatial_axes
    if rank == 3:
        if slice_spec is None:
            raise ReceptiveFieldError(
                "3-D receptive-field visualization requires slice=(axis, index)."
            )
        axis, _ = slice_spec
        if axis not in spatial_axes:
            raise ReceptiveFieldError("slice axis must be one of the receptive-field spatial axes.")
        return tuple(item for item in spatial_axes if item != axis)
    raise ReceptiveFieldError(
        f"Input has {rank} spatial axes; this visualization supports only 1-D, 2-D, or sliced 3-D."
    )


def _base_image(
    view: "ReceptiveFieldView",
    descriptor: ReceptiveField,
    image: Image.Image | None,
    rendered_axes: tuple[int, ...],
) -> Image.Image:
    """Resolve an RGB base image from an override, raw PIL stimulus, or tensor payload."""

    if image is not None:
        return image.convert("RGB")
    op = getattr(view, "op", getattr(view, "_op", None))
    trace = getattr(op, "source_trace", None)
    if trace is not None:
        from ..repgeom import _matching_pil_image_batch

        batch = _matching_pil_image_batch(
            getattr(trace, "raw_input", None), descriptor.input_shape[0]
        )
        if batch:
            return batch[0].convert("RGB")
        try:
            input_op = trace[descriptor.input_op_label]
            tensor = input_op.out
            if isinstance(tensor, torch.Tensor):
                return _tensor_image(tensor, descriptor, rendered_axes)
        except (KeyError, AttributeError):
            pass
    width = max(1, int(descriptor.input_shape[rendered_axes[-1]]))
    height = (
        32 if len(rendered_axes) == 1 else max(1, int(descriptor.input_shape[rendered_axes[0]]))
    )
    return Image.new("RGB", (width, height), "white")


def _tensor_image(
    tensor: torch.Tensor, descriptor: ReceptiveField, axes: tuple[int, ...]
) -> Image.Image:
    """Render a captured tensor by averaging all non-spatial axes."""

    data = tensor.detach().float().cpu().abs()
    reduce_axes = tuple(axis for axis in range(data.ndim) if axis not in axes)
    for axis in sorted(reduce_axes, reverse=True):
        data = data.mean(dim=axis)
    if data.ndim == 1:
        data = data.unsqueeze(0)
    return render_heatmap(
        data.numpy(),
        width=max(1, int(descriptor.input_shape[axes[-1]])),
        height=32 if len(axes) == 1 else max(1, int(descriptor.input_shape[axes[0]])),
    )


def _gradient_image(
    result: GradientReceptiveField,
    axes: tuple[int, ...],
    slice_spec: tuple[int, int] | None,
    size: tuple[int, int],
    cmap: str,
) -> Image.Image:
    """Render gradient magnitude over selected spatial axes as an RGB heatmap."""

    data = result.grad.detach().abs().float().cpu()
    if slice_spec is not None:
        sliced_axis, index = slice_spec
        if index < 0 or index >= data.shape[sliced_axis]:
            raise ReceptiveFieldError("slice index is out of bounds for the selected input axis.")
        data = data.select(sliced_axis, index)
        source_axes = tuple(axis for axis in range(result.grad.ndim) if axis != sliced_axis)
    else:
        source_axes = tuple(range(result.grad.ndim))
    positions = tuple(source_axes.index(axis) for axis in axes)
    reduce_axes = tuple(axis for axis in range(data.ndim) if axis not in positions)
    for axis in sorted(reduce_axes, reverse=True):
        data = data.mean(dim=axis)
    if data.ndim == 1:
        data = data.unsqueeze(0)
    return render_heatmap(data.numpy(), width=size[0], height=size[1], cmap=cmap)


def _draw_box(
    image: Image.Image, box: ReceptiveFieldBox, axes: tuple[int, ...], *, dashed: bool, color: str
) -> None:
    """Draw a status-styled RF region scaled from tensor extents to image pixels."""

    bounds = [_axis_bounds(box, axis) for axis in axes]
    if any(item is None for item in bounds):
        return
    draw = ImageDraw.Draw(image)
    if len(axes) == 1:
        start, stop, extent = cast(tuple[int, int, int], bounds[0])
        rectangle = (
            start * image.width / extent,
            0.0,
            stop * image.width / extent,
            float(image.height - 1),
        )
    else:
        (top, bottom, height), (left, right, width) = cast(
            tuple[tuple[int, int, int], tuple[int, int, int]], tuple(bounds)
        )
        rectangle = (
            left * image.width / width,
            top * image.height / height,
            right * image.width / width,
            bottom * image.height / height,
        )
    if dashed:
        _dashed_rectangle(draw, rectangle, color)
    else:
        draw.rectangle(rectangle, outline=color, width=2)


def _axis_bounds(box: ReceptiveFieldBox, axis: int) -> tuple[int, int, int] | None:
    """Return clipped bounds and extent for one box axis, if finite."""

    item = box.axes[axis]
    if item.clipped_start is None or item.clipped_stop is None:
        return None
    return item.clipped_start, item.clipped_stop, box.input_shape[axis]


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw, rectangle: tuple[float, float, float, float], color: str
) -> None:
    """Draw a simple dashed rectangle without relying on Pillow version-specific APIs."""

    left, top, right, bottom = (round(value) for value in rectangle)
    for start, stop, fixed, horizontal in (
        (left, right, top, True),
        (left, right, bottom, True),
        (top, bottom, left, False),
        (top, bottom, right, False),
    ):
        for offset in range(start, stop + 1, 8):
            end = min(offset + 4, stop)
            draw.line(
                (offset, fixed, end, fixed) if horizontal else (fixed, offset, fixed, end),
                fill=color,
                width=2,
            )


def _descriptor_for_op(
    op: "Op", selected: Any | None, direction: ReceptiveFieldDirection
) -> ReceptiveField:
    """Resolve an RF descriptor for one target operation."""

    if direction is ReceptiveFieldDirection.PROJECTIVE:
        view = op.projective_field
        return _select_descriptor(view, selected)

    from ._engine import lookup

    descriptors = lookup(op.source_trace, op)
    if selected is None:
        if len(descriptors) != 1:
            raise AmbiguousInputError(
                "Select one reachable input; available IO roles: " + ", ".join(descriptors)
            )
        return next(iter(descriptors.values()))
    role = selected if isinstance(selected, str) else getattr(selected, "io_role", None)
    if role not in descriptors:
        raise ReceptiveFieldError(f"Input {role!r} is not reachable from this target.")
    return descriptors[cast(str, role)]


def _projective_cone(op: "Op", target_label: str) -> frozenset[str]:
    """Return the source-to-target path slice for a projective graph cone."""

    from ._path import between_labels

    return between_labels(op.source_trace, op, target_label)


def _ancestor_cone(op: "Op", io_role: str) -> frozenset[str]:
    """Reverse-walk target parents to one selected model input on demand."""

    trace = op.source_trace
    by_reference = {
        reference: item
        for item in trace.layer_list
        for reference in (item.label, item.layer_label, item._layer_label_raw)
    }
    input_op = next((item for item in trace.layer_list if item.io_role == io_role), None)
    if input_op is None:
        return frozenset()
    labels: set[str] = set()
    stack = [op]
    while stack:
        current = stack.pop()
        if current.label in labels:
            continue
        labels.add(current.label)
        if current.label == input_op.label:
            continue
        stack.extend(by_reference[parent] for parent in current.parents if parent in by_reference)
    return frozenset(labels)


def _rendered_labels(layer: Any) -> frozenset[str]:
    """Return every op label represented by a callback layer argument."""

    values = set(getattr(layer, "op_labels", ()))
    for name in ("label", "layer_label", "_layer_label_raw"):
        value = getattr(layer, name, None)
        if isinstance(value, str):
            values.add(value)
    return frozenset(values)


def _descriptor_for_labels(trace: Any, labels: frozenset[str], role: str) -> ReceptiveField | None:
    """Find a selected-role descriptor for a rendered layer's represented operation."""

    from ._engine import lookup

    for label in labels:
        descriptors = lookup(trace, label)
        if role in descriptors:
            return descriptors[role]
    return None


def _tooltip(descriptor: ReceptiveField) -> str:
    """Format a concise honest RF descriptor tooltip."""

    if descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT:
        return f"{_whole_endpoint_caption(descriptor)} (status=WHOLE_INPUT)"
    if descriptor.axes is None:
        return descriptor.status.value.upper()
    sizes = "x".join(str(axis.size) for axis in descriptor.axes if axis.kind == "windowed")
    jumps = "x".join(str(axis.jump) for axis in descriptor.axes if axis.kind == "windowed")
    return f"RF {sizes or 'non-grid'}, jump {jumps or 'n/a'}, {descriptor.status.value.upper()}"


def _whole_endpoint_caption(descriptor: ReceptiveField) -> str:
    """Return the direction-aware caption for a whole far-end result grid."""

    if descriptor.direction is ReceptiveFieldDirection.RECEPTIVE:
        return "whole input"
    if descriptor.io_role.startswith("output"):
        return "whole output"
    return f"whole {descriptor.input_op_label}"


__all__ = ["node_spec", "show"]
