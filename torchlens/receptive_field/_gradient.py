"""Autograd-backed receptive-field probes with capture-side-effect suppression."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast
import warnings

import torch

from .._io import FieldPolicy
from ..backends import BackendUnsupportedError, get_backend_spec
from . import _engine
from ._errors import ReceptiveFieldError, ReceptiveFieldUnavailableError
from ._types import GradientReceptiveField, GridLayout


if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from ._engine_geometry import _InputState
    from ._types import ReceptiveField


_RECAPTURE_RECIPE = "tl.trace(model, x, backward_ready=True, save=...)"


@dataclass(frozen=True)
class _ProbeSnapshot:
    """Trace mutation counters guarded by the RF probe tripwire."""

    num_backward_passes: int
    has_gradients: bool
    backward_event_count: int
    active_index_present: bool
    active_index: object


class _GradientReceptiveFieldResult(GradientReceptiveField):
    """Gradient result with deterministic influence-set convenience methods."""

    def indices(self) -> tuple[torch.Tensor, ...]:
        """Return exact supported indices, one tensor per input dimension.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Coordinates from the first-class support mask.
        """

        return torch.nonzero(self.support_mask, as_tuple=True)

    def effective(self, mass: float = 0.95) -> torch.Tensor:
        """Return the deterministic smallest magnitude-mass influence set.

        Entries are ordered by descending finite gradient magnitude, with flat
        input index as the tie-break. Only entries in ``support_mask`` contribute.

        Parameters
        ----------
        mass:
            Fraction of total supported finite gradient magnitude to retain.

        Returns
        -------
        torch.Tensor
            Boolean mask with the same full input shape as ``grad``.

        Raises
        ------
        ValueError
            If ``mass`` is outside ``(0, 1]``.
        """

        if not 0.0 < mass <= 1.0:
            raise ValueError("mass must be in the interval (0, 1].")
        flat_magnitude = self.grad.reshape(-1)
        flat_support = self.support_mask.reshape(-1)
        supported_indices = torch.nonzero(flat_support, as_tuple=False).reshape(-1)
        result = torch.zeros_like(flat_support, dtype=torch.bool)
        if supported_indices.numel() == 0:
            return result.reshape(self.grad.shape)
        magnitudes = flat_magnitude[supported_indices]
        finite = torch.isfinite(magnitudes)
        supported_indices = supported_indices[finite]
        magnitudes = magnitudes[finite]
        if supported_indices.numel() == 0:
            return result.reshape(self.grad.shape)
        order = torch.argsort(magnitudes, descending=True, stable=True)
        ordered_indices = supported_indices[order]
        ordered_magnitudes = magnitudes[order]
        total = ordered_magnitudes.sum()
        if total <= 0:
            return result.reshape(self.grad.shape)
        cutoff = total * mass
        count = int(torch.searchsorted(ordered_magnitudes.cumsum(0), cutoff).item()) + 1
        result[ordered_indices[:count]] = True
        return result.reshape(self.grad.shape)

    def effective_ranges(self, mass: float = 0.95) -> tuple[tuple[int, int], ...] | None:
        """Return the tight half-open box around the effective influence set.

        Parameters
        ----------
        mass:
            Fraction passed to :meth:`effective`.

        Returns
        -------
        tuple[tuple[int, int], ...] or None
            Derived bounding ranges, or ``None`` for empty effective support.
        """

        return _support_ranges(self.effective(mass))


def _backward_event_count(trace: Trace) -> int:
    """Return the trace's current number of backward capture events.

    Parameters
    ----------
    trace:
        Trace whose sealed event lane is inspected.

    Returns
    -------
    int
        Number of captured backward events.
    """

    event_stream = getattr(trace, "event_stream", None)
    if event_stream is None:
        return 0
    return len(event_stream.backward_events)


def _snapshot_probe_state(trace: Trace) -> _ProbeSnapshot:
    """Snapshot every trace counter protected by the probe tripwire.

    Parameters
    ----------
    trace:
        Trace about to be probed.

    Returns
    -------
    _ProbeSnapshot
        Immutable pre-probe state.
    """

    trace_dict = trace.__dict__
    return _ProbeSnapshot(
        num_backward_passes=int(getattr(trace, "num_backward_passes", 0)),
        has_gradients=bool(getattr(trace, "has_gradients", False)),
        backward_event_count=_backward_event_count(trace),
        active_index_present="_active_backward_pass_index" in trace_dict,
        active_index=trace_dict.get("_active_backward_pass_index"),
    )


def _assert_probe_state_unchanged(trace: Trace, before: _ProbeSnapshot) -> None:
    """Raise if an RF probe leaked into TorchLens backward capture state.

    Parameters
    ----------
    trace:
        Trace after a probe attempt.
    before:
        State captured immediately before suppression began.

    Raises
    ------
    ReceptiveFieldError
        If any protected capture counter changed.
    """

    after = _snapshot_probe_state(trace)
    if after != before:
        raise ReceptiveFieldError(
            "Receptive-field probe suppression leaked into backward capture state: "
            f"before={before!r}, after={after!r}."
        )


@contextmanager
def _probe_suppressed(trace: Trace) -> Iterator[None]:
    """Suppress TorchLens backward wrappers and tensor hooks for one RF probe.

    The runtime-only flag and its prior presence/value are restored even when
    autograd or user code raises. A post-condition tripwire rejects any mutation
    of TorchLens backward-pass state.

    Parameters
    ----------
    trace:
        Trace whose saved autograd graph is being probed.

    Yields
    ------
    None
        Control while both backward-capture gates are suppressed.
    """

    type(trace).PORTABLE_STATE_SPEC.setdefault("_tl_rf_probe_active", FieldPolicy.DROP)
    trace_dict = trace.__dict__
    flag_was_present = "_tl_rf_probe_active" in trace_dict
    previous_flag = trace_dict.get("_tl_rf_probe_active")
    before = _snapshot_probe_state(trace)
    trace._tl_rf_probe_active = True
    try:
        yield
    finally:
        if flag_was_present:
            trace._tl_rf_probe_active = previous_flag
        else:
            trace_dict.pop("_tl_rf_probe_active", None)
        _assert_probe_state_unchanged(trace, before)


def _normalize_unit(unit: Sequence[int], shape: tuple[int, ...], op_label: str) -> tuple[int, ...]:
    """Validate and normalize a complete output-element index.

    Parameters
    ----------
    unit:
        Complete output index supplied by the caller.
    shape:
        Captured target output shape.
    op_label:
        Target label used in diagnostics.

    Returns
    -------
    tuple[int, ...]
        Non-negative normalized index.

    Raises
    ------
    ReceptiveFieldError
        If rank, types, or bounds do not identify exactly one output element.
    """

    raw = tuple(unit)
    if len(raw) != len(shape):
        raise ReceptiveFieldError(
            f"unit for {op_label!r} must contain one index for every output axis "
            f"({len(shape)} required, {len(raw)} provided)."
        )
    normalized: list[int] = []
    for axis, (index, extent) in enumerate(zip(raw, shape)):
        if not isinstance(index, int) or isinstance(index, bool):
            raise ReceptiveFieldError(f"unit axis {axis} must be an integer index.")
        resolved = index + extent if index < 0 else index
        if resolved < 0 or resolved >= extent:
            raise ReceptiveFieldError(
                f"unit axis {axis} index {index} is out of bounds for extent {extent}."
            )
        normalized.append(resolved)
    return tuple(normalized)


def _input_ops_by_role(trace: Trace) -> dict[str, Op]:
    """Return model-input operations keyed by their exact IO roles.

    Parameters
    ----------
    trace:
        Trace whose input boundary is inspected.

    Returns
    -------
    dict[str, Op]
        Input operations in trace order.
    """

    return {
        str(op.io_role or op.label): op
        for op in trace.layer_list
        if bool(getattr(op, "is_input", False))
    }


def _select_inputs(
    trace: Trace,
    descriptors: Mapping[str, ReceptiveField],
    input: Op | str | None,
) -> tuple[tuple[str, Op, ReceptiveField | None], ...]:
    """Resolve requested input operations and their reachability descriptors.

    Parameters
    ----------
    trace:
        Owning trace.
    descriptors:
        Geometrically reachable inputs for the target operation.
    input:
        Optional exact IO role or input-operation handle.

    Returns
    -------
    tuple[tuple[str, Op, ReceptiveField or None], ...]
        Selected input roles, operations, and optional reachable descriptors.

    Raises
    ------
    ReceptiveFieldError
        If the requested input is not a model-input operation on this trace.
    """

    inputs_by_role = _input_ops_by_role(trace)
    if input is None:
        selected: list[tuple[str, Op, ReceptiveField | None]] = []
        for role, descriptor in descriptors.items():
            op = next(
                (
                    candidate
                    for candidate in inputs_by_role.values()
                    if candidate.label == descriptor.input_op_label
                ),
                None,
            )
            if op is None:
                raise ReceptiveFieldUnavailableError(
                    f"Reachable input {descriptor.input_op_label!r} has no live saved operation. "
                    f"Recapture with {_RECAPTURE_RECIPE}."
                )
            selected.append((role, op, descriptor))
        return tuple(selected)

    if isinstance(input, str):
        role = input
        op = inputs_by_role.get(role)
    else:
        op = input
        role = str(getattr(op, "io_role", None) or getattr(op, "label", ""))
    if op is None or inputs_by_role.get(role) is not op:
        raise ReceptiveFieldError(
            "input must be an input Op from this trace or its exact io_role mapping key."
        )
    return ((role, op, descriptors.get(role)),)


def _saved_tensor(op: Op, purpose: str) -> torch.Tensor:
    """Return one live saved tensor or raise the typed recapture diagnostic.

    Parameters
    ----------
    op:
        Operation whose payload is required.
    purpose:
        Human-readable purpose used in the error.

    Returns
    -------
    torch.Tensor
        Live saved tensor payload.

    Raises
    ------
    ReceptiveFieldUnavailableError
        If no graph-connected tensor payload is available.
    """

    try:
        value = op.out
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise ReceptiveFieldUnavailableError(
            f"The {purpose} op {op.label!r} has no saved payload. Recapture with "
            f"{_RECAPTURE_RECIPE}."
        ) from exc
    if not isinstance(value, torch.Tensor):
        raise ReceptiveFieldUnavailableError(
            f"The {purpose} op {op.label!r} does not expose a tensor payload. Recapture with "
            f"{_RECAPTURE_RECIPE}."
        )
    return value


def _support_ranges(mask: torch.Tensor) -> tuple[tuple[int, int], ...] | None:
    """Compute a tight half-open bounding box around a Boolean support mask.

    Parameters
    ----------
    mask:
        Boolean tensor of arbitrary rank.

    Returns
    -------
    tuple[tuple[int, int], ...] or None
        One half-open range per axis, or ``None`` when support is empty.
    """

    coordinates = torch.nonzero(mask, as_tuple=False)
    if coordinates.numel() == 0:
        return None
    if mask.ndim == 0:
        return ()
    return tuple(
        (int(coordinates[:, axis].min().item()), int(coordinates[:, axis].max().item()) + 1)
        for axis in range(mask.ndim)
    )


def _spatial_support(mask: torch.Tensor, layout: GridLayout | None) -> torch.Tensor | None:
    """Reduce support over derived non-windowed axes when layout is known.

    Parameters
    ----------
    mask:
        Full input support mask.
    layout:
        Derived layout, or ``None`` for a disconnected requested input.

    Returns
    -------
    torch.Tensor or None
        Windowed-axis support mask, or ``None`` when layout is unknown.
    """

    if layout is None or "unknown" in layout.axis_kinds:
        return None
    reduce_axes = tuple(axis for axis in range(mask.ndim) if axis not in layout.windowed_axes)
    if not reduce_axes:
        return mask.clone()
    return mask.any(dim=reduce_axes)


def _batch_semantics(
    mask: torch.Tensor,
    state: _InputState | None,
    unit: tuple[int, ...],
) -> tuple[tuple[int, ...] | None, bool]:
    """Derive supported samples and cross-sample influence from engine layout state.

    Parameters
    ----------
    mask:
        Full input support mask.
    state:
        Reachable engine state carrying the derived batch-like input axis.
    unit:
        Complete normalized target output index.

    Returns
    -------
    tuple[tuple[int, ...] or None, bool]
        Supported batch indices and whether any differs from the seeded sample.
    """

    if state is None or state.batch_axis is None:
        return None, False
    batch_axis = state.batch_axis
    reduce_axes = tuple(axis for axis in range(mask.ndim) if axis != batch_axis)
    per_batch = mask if not reduce_axes else mask.any(dim=reduce_axes)
    batch_support = tuple(int(index) for index in torch.nonzero(per_batch).reshape(-1).tolist())
    seeded_batch = unit[batch_axis] if batch_axis < len(unit) else None
    return batch_support, seeded_batch is not None and any(
        index != seeded_batch for index in batch_support
    )


def _build_result(
    *,
    target: Op,
    role: str,
    unit: tuple[int, ...],
    grad: torch.Tensor,
    descriptor: ReceptiveField | None,
    state: _InputState | None,
    atol: float,
    rtol: float,
) -> GradientReceptiveField:
    """Build the immutable empirical RF result for one full input gradient.

    Parameters
    ----------
    target:
        Probed target operation.
    role:
        Exact model-input IO role.
    unit:
        Complete normalized target index.
    grad:
        Full unsliced input gradient.
    descriptor:
        Reachable geometric descriptor, when one exists.
    state:
        Internal engine state used only for derived batch semantics.
    atol, rtol:
        Support thresholds.

    Returns
    -------
    GradientReceptiveField
        Full empirical influence-set result.
    """

    magnitude = grad.detach().abs()
    finite = torch.isfinite(magnitude)
    nonfinite_count = int((~finite).sum().item())
    finite_values = magnitude[finite]
    maximum = finite_values.max() if finite_values.numel() else magnitude.new_tensor(0.0)
    support = finite & (magnitude > atol + rtol * maximum)
    result_warnings: tuple[str, ...] = ()
    if nonfinite_count:
        message = (
            f"Gradient receptive field for {target.label!r} contains {nonfinite_count} "
            "NaN or Inf entries; validation is indeterminate."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        result_warnings = (message,)
    batch_support, cross_batch = _batch_semantics(support, state, unit)
    return _GradientReceptiveFieldResult(
        op_label=target.label,
        io_role=role,
        unit=unit,
        grad=magnitude,
        support_mask=support,
        support_ranges=_support_ranges(support),
        spatial_support_mask=_spatial_support(
            support, None if descriptor is None else descriptor.layout
        ),
        batch_support=batch_support,
        cross_batch_influence=cross_batch,
        atol=atol,
        rtol=rtol,
        nonfinite_count=nonfinite_count,
        warnings=result_warnings,
    )


def gradient_for_unit(
    target: Op,
    unit: Sequence[int],
    *,
    input: Op | str | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    retain_graph: bool = False,
) -> GradientReceptiveField | Mapping[str, GradientReceptiveField]:
    """Probe exactly one target output element against selected model inputs.

    Parameters
    ----------
    target:
        Target operation with a live saved output tensor.
    unit:
        Complete output-element index, including every target axis.
    input:
        Optional model-input Op or exact IO role. ``None`` probes every
        geometrically reachable model input in one autograd call.
    atol, rtol:
        Non-negative support thresholds. Defaults select exact finite nonzero entries.
    retain_graph:
        Whether PyTorch should retain saved autograd buffers after the probe.

    Returns
    -------
    GradientReceptiveField or Mapping[str, GradientReceptiveField]
        One result for an explicit input, otherwise an insertion-ordered mapping.

    Raises
    ------
    BackendUnsupportedError
        If the trace backend does not support true backward capture.
    ReceptiveFieldUnavailableError
        If capture state is not graph-connected or a reachable input yields ``None``.
    ReceptiveFieldError
        If the unit or requested input is invalid, or suppression leaks state.
    """

    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative.")
    trace = target.source_trace
    spec = get_backend_spec(str(getattr(trace, "backend", "torch")))
    if not spec.capabilities.backward_capture:
        raise BackendUnsupportedError(
            f"Backend {spec.name!r} does not support gradient receptive-field probes."
        )
    if not getattr(trace, "backward_ready", False):
        raise ReceptiveFieldUnavailableError(
            "Gradient receptive fields require backward_ready=True and saved payloads. "
            f"Recapture with {_RECAPTURE_RECIPE}."
        )

    target_tensor = _saved_tensor(target, "target")
    normalized_unit = _normalize_unit(unit, tuple(target_tensor.shape), target.label)
    if not (torch.is_floating_point(target_tensor) or torch.is_complex(target_tensor)):
        raise ReceptiveFieldUnavailableError(
            f"Target op {target.label!r} has non-differentiable dtype {target_tensor.dtype}."
        )
    seed = target_tensor[normalized_unit]
    if seed.ndim != 0 or not seed.requires_grad:
        raise ReceptiveFieldUnavailableError(
            f"Target op {target.label!r} is not graph-connected. Recapture with "
            f"{_RECAPTURE_RECIPE}."
        )

    solution = _engine.solve(trace)
    descriptors = _engine.lookup(trace, target)
    selected = _select_inputs(trace, descriptors, input)
    if not selected:
        raise ReceptiveFieldUnavailableError(
            f"Target op {target.label!r} has no geometrically reachable model input."
        )
    input_tensors = tuple(_saved_tensor(op, "input") for _, op, _ in selected)
    if any(not tensor.requires_grad for tensor in input_tensors):
        raise ReceptiveFieldUnavailableError(
            "Selected model-input tensors do not require gradients. Pass input tensors with "
            f"requires_grad=True and recapture with {_RECAPTURE_RECIPE}."
        )
    with _probe_suppressed(trace):
        grads = torch.autograd.grad(
            seed,
            input_tensors,
            retain_graph=retain_graph,
            allow_unused=True,
        )

    results: dict[str, GradientReceptiveField] = {}
    for (role, input_op, descriptor), grad in zip(selected, grads):
        state = solution.states.get((target.label, role))
        if grad is None and descriptor is not None:
            raise ReceptiveFieldUnavailableError(
                f"Input {role!r} is reachable from {target.label!r} in the captured DAG, but "
                "autograd returned no gradient. The saved tensor identity may be stale or the "
                "path may have been detached; recapture and inspect the path."
            )
        if grad is None:
            grad = torch.zeros_like(_saved_tensor(input_op, "input"))
        results[role] = _build_result(
            target=target,
            role=role,
            unit=normalized_unit,
            grad=cast(torch.Tensor, grad),
            descriptor=descriptor,
            state=state,
            atol=atol,
            rtol=rtol,
        )

    if input is not None:
        return next(iter(results.values()))
    return MappingProxyType(results)


__all__: list[str] = []
