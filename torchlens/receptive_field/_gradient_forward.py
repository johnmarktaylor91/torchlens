"""Saved-graph double-VJP probes for projective receptive fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING
import warnings

import torch

from ..backends import BackendUnsupportedError, get_backend_spec
from ._engine_forward import solve_projective
from ._errors import ReceptiveFieldUnavailableError
from ._gradient import (
    _GradientReceptiveFieldResult,
    _batch_semantics,
    _normalize_unit,
    _probe_suppressed,
    _saved_tensor,
    _spatial_support,
    _support_ranges,
)
from ._path import descendant_labels, require_path, resolve_graph_point
from ._types import GradientReceptiveField, ReceptiveFieldDirection


if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from ._engine_forward import _ProjectiveFieldSolution


_RECAPTURE_RECIPE = (
    "tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True), save=...)"
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

    return str(target.io_role or target.label)


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

    labels = (source.layer_label_short, *(target.layer_label_short for target in targets))
    selectors = " | ".join(f"tl.label({label!r})" for label in labels)
    return f"tl.trace(model, x, backward_ready=True, save={selectors})"


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
) -> GradientReceptiveField:
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
    GradientReceptiveField
        Immutable target-space influence set.
    """

    magnitude = column.detach().abs()
    finite = torch.isfinite(magnitude)
    nonfinite_count = int((~finite).sum().item())
    finite_values = magnitude[finite]
    maximum = finite_values.max() if finite_values.numel() else magnitude.new_tensor(0.0)
    support = finite & (magnitude > atol + rtol * maximum)
    result_warnings: tuple[str, ...] = ()
    if nonfinite_count:
        message = (
            f"Projective gradient from {source.label!r} to {target.label!r} contains "
            f"{nonfinite_count} NaN or Inf entries; validation is indeterminate."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        result_warnings = (message,)

    key = _target_key(target)
    descriptor = solution.per_op.get(source.label, {}).get(key)
    state = solution.states.get((source.label, key))
    batch_support, cross_batch = _batch_semantics(support, state, unit)
    return _GradientReceptiveFieldResult(
        op_label=source.label,
        io_role=key,
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
        direction=ReceptiveFieldDirection.PROJECTIVE,
        unit_shape=tuple(source.shape),
    )


def projective_gradient_for_unit(
    source: Op,
    unit: Sequence[int],
    *,
    target: object | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    retain_graph: bool = False,
) -> GradientReceptiveField | Mapping[str, GradientReceptiveField]:
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
    GradientReceptiveField or Mapping[str, GradientReceptiveField]
        One result for an explicit target, otherwise an insertion-ordered mapping.

    Raises
    ------
    BackendUnsupportedError
        If the trace backend does not support true backward capture.
    ReceptiveFieldUnavailableError
        If saved-graph probing or cotangent linearization is unavailable.
    ReceptiveFieldError
        If the source index is invalid or suppression leaks trace state.
    """

    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative.")
    trace: Trace = source.source_trace
    spec = get_backend_spec(str(getattr(trace, "backend", "torch")))
    if not spec.capabilities.backward_capture:
        raise BackendUnsupportedError(
            f"Backend {spec.name!r} does not support projective gradient probes."
        )
    if not getattr(trace, "backward_ready", False):
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
                    f"Source {source.label!r} is structurally reachable from the selected "
                    "targets, but autograd returned no VJP. The saved tensor identity may be "
                    "stale or the path may have been detached."
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
            "units as the O(units) alternative."
        ) from exc

    results: dict[str, GradientReceptiveField] = {}
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
