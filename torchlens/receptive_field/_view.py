"""Entity-level receptive-field query view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, cast

from ._errors import (
    AmbiguousInputError,
    AmbiguousTargetError,
    ReceptiveFieldError,
    ReceptiveFieldUnavailableError,
)
from ._types import (
    GradientReceptiveField,
    GridLayout,
    ReceptiveField,
    ReceptiveFieldAxis,
    ReceptiveFieldBox,
    ReceptiveFieldDirection,
    ReceptiveFieldStatus,
    ReceptiveFieldValidation,
)

if TYPE_CHECKING:
    from PIL import Image

    from ..data_classes.op import Op
    from ._engine import _ReceptiveFieldSolution
    from ._engine_forward import _ProjectiveFieldSolution


def _optional_callable(module_name: str, function_name: str, feature: str) -> Any:
    """Load an optional receptive-field callable.

    Parameters
    ----------
    module_name:
        Receptive-field module containing the callable.
    function_name:
        Callable name within the module.
    feature:
        Human-readable capability name used in the unavailable diagnostic.

    Returns
    -------
    Any
        Imported callable.

    Raises
    ------
    ReceptiveFieldUnavailableError
        If the backing capability is not importable in this installation.
    """

    try:
        module = import_module(module_name, package=__package__)
        return getattr(module, function_name)
    except (AttributeError, ImportError) as exc:
        raise ReceptiveFieldUnavailableError(
            f"Receptive-field {feature} is not available in this installation."
        ) from exc


class ReceptiveFieldView:
    """Lazy query surface for one operation's per-input receptive fields."""

    __slots__ = ("_op", "_solution", "_direction", "per_input")

    def __init__(
        self,
        op: Op,
        solution: _ReceptiveFieldSolution | _ProjectiveFieldSolution,
        direction: ReceptiveFieldDirection = ReceptiveFieldDirection.RECEPTIVE,
    ) -> None:
        """Initialize an operation view from one trace-owned solution.

        Parameters
        ----------
        op:
            Target operation.
        solution:
            Current whole-trace geometric solution.
        """

        self._op = op
        self._solution = solution
        self._direction = direction
        self.per_input: Mapping[str, ReceptiveField] = solution.per_op.get(op.label, {})

    @classmethod
    def projective(cls, op: Op) -> ReceptiveFieldView:
        """Build the source-anchored projective sibling view.

        Parameters
        ----------
        op:
            Source operation whose output grid owns ``unit`` coordinates.

        Returns
        -------
        ReceptiveFieldView
            Projective view targeting all reachable model outputs by default.
        """

        from ._engine_forward import solve_projective

        return cls(
            op,
            solve_projective(op.source_trace, op.source_trace.output_ops),
            ReceptiveFieldDirection.PROJECTIVE,
        )

    def _direction_for(
        self, direction: ReceptiveFieldDirection | str | None
    ) -> ReceptiveFieldDirection:
        """Resolve a per-call direction without changing the view's binding."""

        return self._direction if direction is None else ReceptiveFieldDirection(direction)

    def _projective_solution(self, target: object | None) -> _ProjectiveFieldSolution:
        """Return a target-anchored solution for a projective query."""

        from ._engine_forward import solve_projective
        from ._path import resolve_graph_point

        if target is None:
            return (
                cast("_ProjectiveFieldSolution", self._solution)
                if self._direction is ReceptiveFieldDirection.PROJECTIVE
                else solve_projective(self._op.source_trace, self._op.source_trace.output_ops)
            )
        return solve_projective(
            self._op.source_trace, (resolve_graph_point(self._op.source_trace, target),)
        )

    def __getitem__(self, key: str | Op) -> ReceptiveField:
        """Return a descriptor by exact IO role or model-input operation.

        Parameters
        ----------
        key:
            Exact ``per_input`` key or graph-native input operation.

        Returns
        -------
        ReceptiveField
            Descriptor for the selected reachable input.

        Raises
        ------
        KeyError
            If the input is not reachable from the target operation.
        """

        if isinstance(key, str):
            return self.per_input[key]
        for descriptor in self.per_input.values():
            if descriptor.input_op_label == key.label:
                return descriptor
        raise KeyError(key)

    def _descriptor(self, input: Op | str | None = None) -> ReceptiveField:
        """Resolve one descriptor and reject ambiguous convenience access.

        Parameters
        ----------
        input:
            Optional exact IO role or graph endpoint operation.

        Returns
        -------
        ReceptiveField
            Selected descriptor.

        Raises
        ------
        AmbiguousInputError
            If this is a receptive view, no input is selected, and several
            are reachable.
        AmbiguousTargetError
            If this is a projective view, no target is selected, and several
            are reachable.
        ReceptiveFieldError
            If no endpoint is reachable or the requested one is not reachable.
        """

        projective = self._direction is ReceptiveFieldDirection.PROJECTIVE
        if input is None:
            if len(self.per_input) > 1:
                roles = ", ".join(self.per_input)
                if projective:
                    raise AmbiguousTargetError(
                        f"Source {self._op.label!r} has multiple reachable targets: "
                        f"{roles}. Select one with view[target_op] or target=<io_role>."
                    )
                raise AmbiguousInputError(
                    f"Target {self._op.label!r} has multiple reachable inputs: {roles}. "
                    "Select one with view[input_op] or input=<io_role>."
                )
            if not self.per_input:
                if projective:
                    raise ReceptiveFieldError(
                        f"Source {self._op.label!r} has no reachable model output."
                    )
                raise ReceptiveFieldError(
                    f"Target {self._op.label!r} has no reachable model input."
                )
            return next(iter(self.per_input.values()))
        try:
            return self[input]
        except KeyError as exc:
            identity = input if isinstance(input, str) else input.label
            if projective:
                raise ReceptiveFieldError(
                    f"Target {identity!r} is not reachable from source {self._op.label!r}."
                ) from exc
            raise ReceptiveFieldError(
                f"Input {identity!r} is not reachable from target {self._op.label!r}."
            ) from exc

    @property
    def status(self) -> ReceptiveFieldStatus:
        """Return status for the only reachable input."""

        return self._descriptor().status

    @property
    def axes(self) -> tuple[ReceptiveFieldAxis, ...] | None:
        """Return axes for the only reachable input."""

        return self._descriptor().axes

    @property
    def size(self) -> tuple[int, ...]:
        """Return window sizes for the only reachable input."""

        return self._descriptor().size

    @property
    def jump(self) -> tuple[Fraction, ...]:
        """Return effective jumps for the only reachable input."""

        return self._descriptor().jump

    @property
    def center0(self) -> tuple[Fraction, ...]:
        """Return output-zero centers for the only reachable input."""

        return self._descriptor().center0

    @property
    def layout(self) -> GridLayout:
        """Return the derived layout for the only reachable input."""

        return self._descriptor().layout

    def _center_unit(self, descriptor: ReceptiveField) -> tuple[int, ...]:
        """Return windowed-axis midpoint coordinates for the ``"center"`` selector.

        The midpoint is taken in this view operation's own output grid, which is
        the coordinate space that ``unit`` addresses for both receptive and
        projective queries. Shared by both directions so ``"center"`` is honored
        identically wherever ``at`` accepts it.
        """

        if descriptor.axes is None:
            raise ReceptiveFieldError("Geometric axes are unavailable; use .gradient() instead.")
        output_axes = sorted(
            cast(int, axis.output_axis) for axis in descriptor.axes if axis.kind == "windowed"
        )
        return tuple(int(self._op.shape[axis]) // 2 for axis in output_axes)

    def at(
        self,
        unit: tuple[int, ...] | Literal["center"],
        *,
        input: Op | str | None = None,
        source: object | None = None,
        direction: ReceptiveFieldDirection | str | None = None,
        target: object | None = None,
        clip: bool = True,
    ) -> ReceptiveFieldBox:
        """Return geometric source bounds for one windowed-grid output unit.

        Parameters
        ----------
        unit:
            Windowed-axis coordinates, or ``"center"`` for their midpoint.
            Coordinates must be non-negative and in range; unlike ``.gradient()``
            and ``.check()``, negative indices are rejected rather than wrapped.
        input:
            Optional exact IO role or model-input operation.
        source:
            Optional ancestor graph point. When given, the returned box is in the
            SOURCE operation's output-grid coordinate space (not this operation's).
        direction:
            Optional per-call receptive/projective direction override. ``None``
            uses the view's bound direction.
        target:
            Optional descendant graph point selecting the far endpoint of a
            projective query. Not permitted together with ``source``.
        clip:
            Whether bounds are clipped to captured input extents.

        Returns
        -------
        ReceptiveFieldBox
            Per-unit theoretical and clipped receptive-field bounds.
        """

        resolved_direction = self._direction_for(direction)
        if input is not None and source is not None:
            raise TypeError("input and source cannot be supplied together.")
        if resolved_direction is ReceptiveFieldDirection.PROJECTIVE:
            if input is not None or source is not None:
                raise TypeError("Projective queries select their far endpoint with target=.")
            from ._forward_query import _select_target_descriptor, box_for_source_unit

            projective_solution = self._projective_solution(target)
            source_unit: Sequence[int]
            if unit == "center":
                descriptors = projective_solution.per_op.get(self._op.label)
                if not descriptors:
                    raise ReceptiveFieldError(
                        f"No projective-field solution is available from source {self._op.label!r}."
                    )
                proj_descriptor = _select_target_descriptor(
                    descriptors, cast("Op | str | None", target)
                )
                source_unit = self._center_unit(proj_descriptor)
            else:
                source_unit = cast(Sequence[int], unit)
            return box_for_source_unit(
                projective_solution,
                self._op,
                source_unit,
                target=cast("Op | str | None", target),
                clip=clip,
            )
        if target is not None:
            if source is not None:
                raise TypeError("source and target cannot be supplied together.")
            source = target

        from . import _engine
        from ._path import require_path, resolve_graph_point
        from ._query import box_for_unit

        source_op: Op | None = None
        solution = self._solution
        selected_input = input
        if source is not None:
            source_op = resolve_graph_point(self._op.source_trace, source)
            require_path(source_op, self._op, "receptive")
            solution = _engine.solve_from(self._op.source_trace, source_op)
            selected_input = None

        selected: tuple[int, ...]
        if unit == "center":
            descriptors = solution.per_op.get(self._op.label, {})
            descriptor = (
                self._descriptor(selected_input)
                if source_op is None
                else next(
                    (
                        item
                        for item in descriptors.values()
                        if item.input_op_label == source_op.label
                    ),
                    None,
                )
            )
            if descriptor is None:
                if source_op is None:
                    raise ReceptiveFieldError(
                        f"No receptive-field descriptor is available for target {self._op.label!r}."
                    )
                raise ReceptiveFieldError(
                    f"Source {source_op.label!r} is not reachable from target {self._op.label!r}."
                )
            selected = self._center_unit(descriptor)
        else:
            selected = unit
        return box_for_unit(
            cast("_ReceptiveFieldSolution", solution),
            self._op,
            selected,
            input=selected_input,
            source=source_op,
            clip=clip,
        )

    def gradient(
        self,
        unit: tuple[int, ...],
        *,
        input: Op | str | None = None,
        source: object | None = None,
        direction: ReceptiveFieldDirection | str | None = None,
        target: object | None = None,
        atol: float = 0.0,
        rtol: float = 0.0,
        retain_graph: bool = False,
    ) -> GradientReceptiveField | Mapping[str, GradientReceptiveField]:
        """Probe empirical influence for one complete output-element index.

        Parameters
        ----------
        unit:
            Complete target output-element index.
        input:
            Optional exact IO role or model-input operation.
        source:
            Reserved ancestor graph point for layer-to-layer receptive probes.
        direction:
            Optional per-call receptive/projective direction override. ``None``
            uses the view's bound direction.
        target:
            Optional descendant graph point selecting the far endpoint of a
            projective query. Not permitted together with ``source``.
        atol, rtol:
            Non-negative gradient support thresholds.
        retain_graph:
            Whether autograd should retain saved graph buffers.

        Returns
        -------
        GradientReceptiveField or Mapping[str, GradientReceptiveField]
            One selected result, or the mapping over all reachable inputs.
        """

        resolved_direction = self._direction_for(direction)

        if input is not None and source is not None:
            raise TypeError("input and source cannot be supplied together.")
        if resolved_direction is ReceptiveFieldDirection.PROJECTIVE:
            if input is not None or source is not None:
                raise TypeError("Projective queries select their far endpoint with target=.")
            projective_gradient = _optional_callable(
                "._gradient_forward", "projective_gradient_for_unit", "projective gradient support"
            )
            return cast(
                GradientReceptiveField | Mapping[str, GradientReceptiveField],
                projective_gradient(
                    self._op, unit, target=target, atol=atol, rtol=rtol, retain_graph=retain_graph
                ),
            )
        from ._gradient import gradient_for_unit

        if target is not None:
            if source is not None:
                raise TypeError("source and target cannot be supplied together.")
            source = target
        if source is not None:
            return cast(
                GradientReceptiveField | Mapping[str, GradientReceptiveField],
                cast(Any, gradient_for_unit)(
                    self._op,
                    unit,
                    source=source,
                    atol=atol,
                    rtol=rtol,
                    retain_graph=retain_graph,
                ),
            )
        return gradient_for_unit(
            self._op,
            unit,
            input=input,
            atol=atol,
            rtol=rtol,
            retain_graph=retain_graph,
        )

    def check(
        self,
        unit: tuple[int, ...],
        *,
        input: Op | str | None = None,
        source: object | None = None,
        direction: ReceptiveFieldDirection | str | None = None,
        target: object | None = None,
        atol: float = 0.0,
        rtol: float = 0.0,
        retain_graph: bool = False,
    ) -> ReceptiveFieldValidation:
        """Cross-check geometry against gradients for one complete output index.

        Parameters
        ----------
        unit:
            Complete target output-element index.
        input:
            Optional exact IO role or model-input operation.
        source:
            Optional ancestor graph point for layer-to-layer receptive validation.
        direction:
            Receptive or projective containment direction.
        target:
            Optional descendant graph point for projective validation.
        atol, rtol:
            Accepted for compatibility and ignored. Validation always uses exact
            finite nonzero gradient support.
        retain_graph:
            Keep the armed autograd graph alive after the check. The default
            ``False`` FREES the graph, exactly like ``gradient()``'s default:
            call with ``retain_graph=True`` (or re-capture) when a later
            ``gradient(..., retain_graph=True)`` or second check must run on
            the same armed capture.

        Returns
        -------
        ReceptiveFieldValidation
            Tri-state containment result.
        """

        resolved_direction = self._direction_for(direction)
        if input is not None and source is not None:
            raise TypeError("input and source cannot be supplied together.")
        check_for_unit = _optional_callable("._validation", "check_for_unit", "validation")
        return cast(
            ReceptiveFieldValidation,
            check_for_unit(
                self._op,
                unit,
                input=input,
                source=source,
                direction=resolved_direction,
                target=target,
                atol=atol,
                rtol=rtol,
                retain_graph=retain_graph,
            ),
        )

    def center_unit(
        self,
        *,
        batch_index: int | None = None,
        input: Op | str | None = None,
    ) -> tuple[int, ...]:
        """Resolve the complete centered output index without guessing a sample.

        Parameters
        ----------
        batch_index:
            Required index for the derived batch-like output axis.
        input:
            Optional exact IO role or model-input operation.

        Returns
        -------
        tuple[int, ...]
            Complete output-element index suitable for gradient-based methods.

        Raises
        ------
        ReceptiveFieldError
            If batch semantics are unknown or ``batch_index`` is invalid or omitted.
        """

        descriptor = self._descriptor(input)
        state = self._solution.states.get((self._op.label, descriptor.io_role))
        if state is None or state.axes is None or state.batch_axis is None:
            raise ReceptiveFieldError(
                f"The derived layout for {self._op.label!r} cannot establish a batch-like axis."
            )
        batch_state = state.axes[state.batch_axis]
        batch_output_axis = batch_state.output_axis
        if batch_output_axis is None:
            raise ReceptiveFieldError(
                f"The derived layout for {self._op.label!r} cannot map its batch-like axis "
                "to the output."
            )
        if batch_index is None:
            raise ReceptiveFieldError(
                f"batch_index is required for batched target {self._op.label!r}."
            )
        batch_extent = int(self._op.shape[batch_output_axis])
        if isinstance(batch_index, bool) or not isinstance(batch_index, int):
            raise ReceptiveFieldError("batch_index must be an integer.")
        if batch_index < 0 or batch_index >= batch_extent:
            raise ReceptiveFieldError(
                f"batch_index {batch_index} is out of bounds for extent {batch_extent}."
            )
        result = [int(extent) // 2 for extent in self._op.shape]
        result[batch_output_axis] = batch_index
        return tuple(result)

    def show(
        self,
        unit: tuple[int, ...] | None = None,
        *,
        input: Op | str | None = None,
        direction: ReceptiveFieldDirection | str | None = None,
        target: object | None = None,
        image: object | None = None,
        gradient: bool = False,
        retain_graph: bool = False,
        plane: object | None = None,
        box_color: str = "#FF3B30",
        alpha: float = 0.6,
        cmap: str = "magma",
    ) -> Image.Image:
        """Render an input-space receptive-field overlay.

        Parameters
        ----------
        unit:
            Optional complete target output-element index.
        input:
            Optional exact IO role or model-input operation.
        direction:
            Optional per-call receptive/projective direction override. ``None``
            uses the view's bound direction.
        target:
            Optional descendant graph point selecting the far endpoint of a
            projective overlay.
        image:
            Optional source image override.
        gradient:
            Whether to include empirical gradient magnitude.
        retain_graph:
            Whether the empirical gradient probe retains autograd buffers, so
            repeated ``show(gradient=True)`` or later ``gradient()`` calls over
            the same captured graph stay possible.
        slice:
            Required plane selector for three-dimensional inputs.
        box_color:
            Geometric-box overlay color.
        alpha:
            Overlay opacity.
        cmap:
            Gradient heatmap colormap.

        Returns
        -------
        PIL.Image.Image
            Rendered input-space overlay.
        """

        show = _optional_callable("._viz", "show", "visualization")

        return cast(
            "Image.Image",
            show(
                self,
                unit,
                input=input,
                direction=self._direction_for(direction),
                target=target,
                image=cast("Image.Image | None", image),
                gradient=gradient,
                retain_graph=retain_graph,
                slice=cast("tuple[int, int] | None", plane),
                box_color=box_color,
                alpha=alpha,
                cmap=cmap,
            ),
        )


__all__ = ["ReceptiveFieldView"]
