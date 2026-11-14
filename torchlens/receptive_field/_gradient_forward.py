"""Saved-graph double-VJP probes for projective receptive fields."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from .._record_type import record_type
from ..backends._backend_names import TORCH_BACKEND_NAME
from ._engine_forward import solve_projective
from ._errors import ReceptiveFieldUnavailableError
from ._gradient import (
    _batch_semantics,
    _GradientReceptiveFieldResult,
    _normalize_unit,
    _probe_suppressed,
    _saved_tensor,
    _spatial_support,
    _support_ranges,
)
from ._gradient_replay_hook import replay_probe
from ._path import descendant_labels, require_path, resolve_graph_point
from ._types import GradReceptiveField, ReceptiveFieldDirection

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from ._engine_forward import _ProjectiveFieldSolution


_RECAPTURE_RECIPE = (
    "tl.trace(model, x, "
    'capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"))'
)


def _no_projective_vjp_reason(source: Op, targets: Sequence[Op], trace: Trace) -> str:
    """Return the honest reason a projective VJP came back empty.

    The message is cause-specific: it never prescribes ``save_mode="reference"``
    on a trace already captured in that mode (a remedy that would send a user in
    a circle), and it does not blame a "stale saved tensor identity" where
    nothing is stale. When the only target is a structural
    OUTPUT MARKER whose sole parent is the source, both payloads are independent
    clones of the SAME captured value, so there is no autograd edge between them
    for any VJP to traverse.

    Parameters
    ----------
    source:
        Source operation whose unit was probed.
    targets:
        Selected projective targets.
    trace:
        Trace both payloads were captured into.

    Returns
    -------
    str
        Cause-specific message; the ``save_mode`` prescription appears only when
        that mode is NOT already in force.
    """

    marker_targets = [
        target
        for target in targets
        if record_type(target, None) == "output"
        and set(getattr(target, "parents", ()) or ()) == {source.layer_label}
    ]
    if marker_targets and len(marker_targets) == len(targets):
        names = ", ".join(repr(target.label) for target in marker_targets)
        return (
            f"Projective validation is not applicable from {source.label!r} to {names}: the "
            "target is a structural output marker whose only parent is the source, so both "
            "saved payloads are independent clones of ONE captured value and no autograd "
            "edge exists between them to differentiate. The projective mapping here is the "
            "identity; probe a downstream COMPUTATIONAL op for a non-trivial projective "
            "field, or read the receptive-direction result, which is unaffected."
        )
    reason = (
        f"Source {source.label!r} is structurally reachable from the selected targets, "
        "but autograd returned no VJP, so the saved tensor identity is not the one in the "
        "captured graph."
    )
    if str(trace._options_save_mode) != "reference":
        return reason + ' Recapture with save_mode="reference".'
    return (
        reason + ' This trace already used save_mode="reference", so the cause is not the '
        "save mode: probe in the same process and before any in-place mutation of the saved "
        f"payloads, recapturing with {_RECAPTURE_RECIPE} if the graph was released."
    )


def _target_key(target: Op) -> str:
    """Return the canonical result key for a projective target.

    Parameters
    ----------
    target:
        Resolved target operation.

    Returns
    -------
    str
        Exact IO role for a model output, otherwise the pass-qualified label.
    """

    return str(target.graph.io_role or target.label)


def _select_targets(source: Op, target: object | None) -> tuple[tuple[Op, ...], bool]:
    """Resolve one explicit target or all reachable model outputs.

    Parameters
    ----------
    source:
        Source operation whose element will be seeded.
    target:
        Optional descendant graph-point handle.

    Returns
    -------
    tuple[tuple[Op, ...], bool]
        Trace-ordered targets and whether the caller selected one explicitly.

    Raises
    ------
    ReceptiveFieldUnavailableError
        If no reachable model output exists for the default target selection.
    NoInfluencePathError
        If an explicit target is not a descendant of ``source``.
    """

    trace = source.source_trace
    if target is not None:
        target_op = resolve_graph_point(trace, target)
        require_path(source, target_op, ReceptiveFieldDirection.PROJECTIVE)
        return (target_op,), True

    reachable = descendant_labels(trace, source)
    targets = tuple(
        resolve_graph_point(trace, output)
        for output in trace.output_ops
        if output.label in reachable
    )
    if not targets:
        raise ReceptiveFieldUnavailableError(
            f"Source op {source.label!r} has no reachable model output to probe."
        )
    return targets, False


def _projective_recapture_recipe(source: Op, targets: Sequence[Op]) -> str:
    """Build a capture recipe retaining the projective probe endpoints.

    Parameters
    ----------
    source:
        Seed operation.
    targets:
        Selected result operations.

    Returns
    -------
    str
        Selector-based recapture guidance.
    """

    labels = (source.short_layer_label, *(target.short_layer_label for target in targets))
    selectors = " | ".join(f"tl.label({label!r})" for label in labels)
    return (
        "tl.trace(model, x, "
        'capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"), '
        f"save={selectors})"
    )


def _source_tensor(source: Op, targets: Sequence[Op]) -> torch.Tensor:
    """Return a real, graph-connected source tensor for double-VJP probing.

    Parameters
    ----------
    source:
        Resolved source operation.
    targets:
        Selected result operations, used in recapture guidance.

    Returns
    -------
    torch.Tensor
        Live real floating-point source payload.

    Raises
    ------
    ReceptiveFieldUnavailableError
        If the payload is absent, complex, non-differentiable, or disconnected.
    """

    recipe = _projective_recapture_recipe(source, targets)
    try:
        tensor = _saved_tensor(source, "source")
    except ReceptiveFieldUnavailableError as exc:
        raise ReceptiveFieldUnavailableError(
            f"Source op {source.label!r} has no retained activation payload. "
            f"Recapture with {recipe}."
        ) from exc
    if torch.is_complex(tensor):
        raise ReceptiveFieldUnavailableError(
            "projective probing of complex sources requires two-basis support; not yet implemented."
        )
    if not torch.is_floating_point(tensor):
        raise ReceptiveFieldUnavailableError(
            f"Source op {source.label!r} has non-differentiable dtype {tensor.dtype}."
        )
    if not tensor.requires_grad:
        raise ReceptiveFieldUnavailableError(
            f"Source op {source.label!r} is not graph-connected. Recapture with {recipe}."
        )
    return tensor


def _target_tensors(targets: Sequence[Op]) -> tuple[torch.Tensor, ...]:
    """Return saved differentiable tensors for all selected targets.

    Parameters
    ----------
    targets:
        Resolved reachable targets.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Graph-connected target payloads in target order.

    Raises
    ------
    ReceptiveFieldUnavailableError
        If any selected target is not a differentiable saved tensor.
    """

    tensors: list[torch.Tensor] = []
    for target in targets:
        tensor = _saved_tensor(target, "target")
        if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
            raise ReceptiveFieldUnavailableError(
                f"Target op {target.label!r} has non-differentiable dtype {tensor.dtype}."
            )
        if not tensor.requires_grad:
            raise ReceptiveFieldUnavailableError(
                f"Target op {target.label!r} is not graph-connected. Recapture with "
                f"{_RECAPTURE_RECIPE}."
            )
        tensors.append(tensor)
    return tuple(tensors)


def _build_projective_result(
    *,
    source: Op,
    target: Op,
    unit: tuple[int, ...],
    column: torch.Tensor,
    solution: _ProjectiveFieldSolution,
    atol: float,
    rtol: float,
) -> GradReceptiveField:
    """Build one target-space empirical projective-field result.

    Parameters
    ----------
    source:
        Seed operation.
    target:
        Result operation.
    unit:
        Complete normalized source-element index.
    column:
        Jacobian column over the target tensor.
    solution:
        Target-anchored geometric solution for derived result metadata.
    atol, rtol:
        Non-negative support thresholds.

    Returns
    -------
    GradReceptiveField
        Immutable target-space influence set.
    """

    magnitude = column.detach().abs()
    finite = torch.isfinite(magnitude)
    num_nonfinite = int((~finite).sum().item())
    finite_values = magnitude[finite]
    maximum = finite_values.max() if finite_values.numel() else magnitude.new_tensor(0.0)
    support = finite & (magnitude > atol + rtol * maximum)
    result_warnings: tuple[str, ...] = ()
    if num_nonfinite:
        message = (
            f"Projective gradient from {source.label!r} to {target.label!r} contains "
            f"{num_nonfinite} NaN or Inf entries; validation is indeterminate."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        result_warnings = (message,)

    key = _target_key(target)
    descriptor = solution.per_op.get(source.label, {}).get(key)
    state = solution.states.get((source.label, key))
    batch_support, cross_batch = _batch_semantics(support, state, unit, projective=True)
    return _GradientReceptiveFieldResult(
        op_label=source.label,
        io_role=key,
        unit=unit,
        grad=magnitude,
        signed_grad=column.detach(),
        support_mask=support,
        support_ranges=_support_ranges(support),
        spatial_support_mask=_spatial_support(
            support, None if descriptor is None else descriptor.axis_roles
        ),
        batch_support=batch_support,
        cross_batch_influence=cross_batch,
        atol=atol,
        rtol=rtol,
        num_nonfinite=num_nonfinite,
        warnings=result_warnings,
        direction=ReceptiveFieldDirection.PROJECTIVE,
        unit_shape=tuple(source.shape),
    )


def projective_supports_by_row_sweep(
    source: Op, units: Sequence[Sequence[int]], target: Op
) -> tuple[GradReceptiveField, ...]:
    """Probe several source units' projective fields one target element at a time.

    The fallback for a captured graph that cannot be differentiated twice (a
    fused kernel with no second derivative, such as fused scaled-dot-product
    attention): row ``t`` of the Jacobian is one ordinary VJP seeded at target
    element ``t``, and each unit's column is read off the rows. The cost is one
    backward pass per target element, however many source units are probed.

    Parameters
    ----------
    source:
        Operation whose output elements seed the projective fields.
    units:
        Complete source indices, one per probe.
    target:
        The one operation whose output space the fields live in.

    Returns
    -------
    tuple[GradReceptiveField, ...]
        One exact (zero-threshold) projective field per unit, in ``units`` order.
    """

    trace: Trace = source.source_trace
    targets, _ = _select_targets(source, target)
    source_tensor = _source_tensor(source, targets)
    (target_tensor,) = _target_tensors(targets)
    solution = solve_projective(trace, targets)
    normalized = [_normalize_unit(unit, tuple(source_tensor.shape), source.label) for unit in units]
    unit_index = tuple(torch.tensor(axis) for axis in zip(*normalized, strict=True))
    flat_target = target_tensor.reshape(-1)
    columns = torch.zeros((len(normalized), flat_target.numel()), dtype=source_tensor.dtype)
    with _probe_suppressed(trace):
        for element in range(flat_target.numel()):
            (row,) = torch.autograd.grad(
                flat_target[element], source_tensor, retain_graph=True, allow_unused=True
            )
            if row is not None:
                columns[:, element] = row.detach()[unit_index]
    return tuple(
        _build_projective_result(
            source=source,
            target=targets[0],
            unit=unit,
            column=columns[position].reshape(target_tensor.shape),
            solution=solution,
            atol=0.0,
            rtol=0.0,
        )
        for position, unit in enumerate(normalized)
    )


def _projective_columns(
    source: Op,
    targets: Sequence[Op],
    source_tensor: torch.Tensor,
    target_tensors: Sequence[torch.Tensor],
    seed: torch.Tensor,
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    """Linearize the target cotangents and pull one projective column per target."""

    trace: Trace = source.source_trace
    try:
        with _probe_suppressed(trace):
            cotangents = tuple(
                torch.zeros_like(tensor, requires_grad=True) for tensor in target_tensors
            )
            vjp = torch.autograd.grad(
                target_tensors,
                source_tensor,
                grad_outputs=cotangents,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if vjp is None:
                raise ReceptiveFieldUnavailableError(
                    _no_projective_vjp_reason(source, targets, trace)
                )
            columns = torch.autograd.grad(
                vjp,
                cotangents,
                grad_outputs=seed,
                retain_graph=retain_graph,
                allow_unused=True,
            )
    except ReceptiveFieldUnavailableError:
        raise
    except RuntimeError as exc:
        raise ReceptiveFieldUnavailableError(
            "Projective gradient cotangent linearization unavailable at the captured "
            f"autograd node: {exc}. Use a receptive-probe sweep over candidate target "
            "units as the O(units) alternative (a field view's .of(method='gradient') "
            "runs that sweep itself when the target site has at most probe_budget "
            "elements)."
        ) from exc
    return columns


def projective_gradient_for_unit(
    source: Op,
    unit: Sequence[int],
    *,
    target: object | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    retain_graph: bool = False,
) -> GradReceptiveField | Mapping[str, GradReceptiveField]:
    """Probe the target-space influence of exactly one source element.

    The probe extracts a Jacobian column from the captured autograd graph using
    two VJP calls. Compute and temporary memory therefore scale with the number
    and sizes of selected targets even though the call count remains exactly two.

    Parameters
    ----------
    source:
        Source operation with a live saved output tensor.
    unit:
        Complete source-element index, including every source axis.
    target:
        Optional descendant graph point. ``None`` probes every reachable model
        output in one double-VJP operation.
    atol, rtol:
        Non-negative support thresholds.
    retain_graph:
        Whether PyTorch should retain saved autograd buffers after the probe.

    Returns
    -------
    GradReceptiveField or Mapping[str, GradReceptiveField]
        One result for an explicit target, otherwise an insertion-ordered mapping.

    Raises
    ------
    BackendUnsupportedError
        If the backend supports neither backward capture nor cone replay.
    ReceptiveFieldUnavailableError
        If saved-graph probing or cotangent linearization is unavailable.
    ReceptiveFieldError
        If the source index is invalid or suppression leaks trace state.
    """

    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative.")
    trace: Trace = source.source_trace
    if str(trace._provenance_backend) != TORCH_BACKEND_NAME:  # autograd path is torch-only
        probe = replay_probe("projective")
        return probe(source, unit, target=target, atol=atol, rtol=rtol)
    if not trace._options_backward_ready:
        raise ReceptiveFieldUnavailableError(
            "Projective gradient fields require backward_ready=True and saved payloads. "
            f"Recapture with {_RECAPTURE_RECIPE}."
        )

    targets, explicit_target = _select_targets(source, target)
    source_tensor = _source_tensor(source, targets)
    normalized_unit = _normalize_unit(unit, tuple(source_tensor.shape), source.label)
    target_tensors = _target_tensors(targets)
    solution = solve_projective(trace, targets)

    seed = torch.zeros_like(source_tensor)
    seed[normalized_unit] = 1
    columns = _projective_columns(
        source, targets, source_tensor, target_tensors, seed, retain_graph=retain_graph
    )

    results: dict[str, GradReceptiveField] = {}
    for target_op, column in zip(targets, columns, strict=True):
        if column is None:
            raise ReceptiveFieldUnavailableError(
                f"Cotangent linearization returned no projective column for target "
                f"{target_op.label!r}."
            )
        results[_target_key(target_op)] = _build_projective_result(
            source=source,
            target=target_op,
            unit=normalized_unit,
            column=column,
            solution=solution,
            atol=atol,
            rtol=rtol,
        )
    if explicit_target:
        return next(iter(results.values()))
    return MappingProxyType(results)
