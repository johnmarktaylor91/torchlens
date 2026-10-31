"""Correlate TorchDynamo graph breaks with eager TorchLens operations."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from .._debug_report import Report, rows_of
from .._errors import _actionable_message, _ActionableErrorMixin
from .._input_coerce import _coerce_input_args
from .._utils import _torch_compat
from .._utils._dynamo_explain_shapes import (
    _DynamoExplainOutputError,
    normalize_dynamo_explain_output,
)
from .._utils.arg_handling import normalize_input_args
from ..errors._base import CompatibilityError


class GraphBreaksUnavailableError(_ActionableErrorMixin, CompatibilityError, RuntimeError):
    """Raised when the running torch does not expose Dynamo explain.

    Keeps its historical ``RuntimeError`` base while joining the taxonomy
    with a stable code and a default remedy, so single-message raise sites
    and existing ``except RuntimeError`` callers stay valid (R64).
    """

    code: str = "graph_breaks_unavailable"
    default_remedy: str = (
        "upgrade to a torch build that exposes torch._dynamo.explain, or skip "
        "the graph_breaks() correlation on this runtime"
    )

    def __init__(
        self,
        problem: str,
        *,
        remedy: str | None = None,
        **context: object,
    ) -> None:
        """Initialize an actionable Dynamo-explain capability refusal.

        Parameters
        ----------
        problem:
            Description of the missing Dynamo explain capability.
        remedy:
            Concrete caller action. The class default is used when omitted.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        resolved_remedy = remedy or type(self).default_remedy
        super().__init__(
            _actionable_message(problem, resolved_remedy),
            code=type(self).code,
            remedy=resolved_remedy,
            **cast(dict[str, Any], context),
        )


class GraphBreaksNormalizationError(_ActionableErrorMixin, CompatibilityError, RuntimeError):
    """Raised when a Dynamo explain result has an unsupported shape.

    Keeps its historical ``RuntimeError`` base while joining the taxonomy
    with a stable code and a default remedy, so single-message raise sites
    and existing ``except RuntimeError`` callers stay valid (R64).
    """

    code: str = "graph_breaks_normalization_failed"
    default_remedy: str = (
        "report the unrecognized torch._dynamo.explain result shape to TorchLens, "
        "or pin a torch version whose explain output is recognized"
    )

    def __init__(
        self,
        problem: str,
        *,
        remedy: str | None = None,
        **context: object,
    ) -> None:
        """Initialize an actionable explain-normalization refusal.

        Parameters
        ----------
        problem:
            Description of the unrecognized explain-result shape.
        remedy:
            Concrete caller action. The class default is used when omitted.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        resolved_remedy = remedy or type(self).default_remedy
        super().__init__(
            _actionable_message(problem, resolved_remedy),
            code=type(self).code,
            remedy=resolved_remedy,
            **cast(dict[str, Any], context),
        )


@dataclass(frozen=True)
class GraphBreak:
    """One Dynamo graph break and its nearest eager operation sites.

    Attributes
    ----------
    reason:
        Reason string supplied by Dynamo.
    source_file:
        User source file supplied by Dynamo, if available.
    line_number:
        One-indexed source line supplied by Dynamo, if available.
    matched_op_labels:
        Eager TorchLens operation labels at the smallest same-file line distance.
    unmatched_reason:
        Evidence-backed explanation when no eager operation could be correlated.
    """

    reason: str
    source_file: str | None
    line_number: int | None
    matched_op_labels: tuple[str, ...]
    unmatched_reason: str | None


@dataclass(frozen=True)
class GraphBreakReport(Report):
    """Structured Dynamo graph-break correlation result.

    Attributes
    ----------
    breaks:
        Graph breaks in Dynamo's reported order.
    """

    breaks: tuple[GraphBreak, ...]

    _subject_text = "compiled model"

    def _report_status(self) -> str:
        """The verdict token on the card's lead line."""

        return f"{len(self.breaks)} breaks" if self.breaks else "no_breaks"

    def _report_findings(self) -> list[str]:
        """One-line findings for the card, most severe first."""

        return [
            f"{item.reason} at {item.source_file}:{item.line_number}"
            if item.source_file is not None
            else item.reason
            for item in self.breaks
        ]

    def _report_rows(self) -> list[dict[str, Any]] | None:
        """The natural table rows behind ``to_pandas()``."""

        return rows_of(self.breaks)

    def __len__(self) -> int:
        """Return the number of graph breaks.

        Returns
        -------
        int
            Number of normalized break records.
        """

        return len(self.breaks)


def _snapshot_model_state(model: nn.Module) -> tuple[dict[str, torch.Tensor], Any, Any]:
    """Capture model parameters, buffers, and global RNG for later restoration.

    Parameters
    ----------
    model:
        Model whose forward mutates buffers or global RNG.

    Returns
    -------
    tuple[dict[str, torch.Tensor], Any, Any]
        Cloned ``state_dict`` tensors and CPU/CUDA RNG states.
    """

    with torch.no_grad():
        params = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    # Initialized-CUDA-only, latch-guarded (R36-4): the bare is_available()
    # gate allocated a CUDA context per visible device on CPU-only runs.
    from .._utils.rng import _snapshot_cuda_rng_states

    cuda_rng = _snapshot_cuda_rng_states() or None
    return params, cpu_rng, cuda_rng


def _restore_model_state(
    model: nn.Module,
    snapshot: tuple[dict[str, torch.Tensor], Any, Any],
) -> None:
    """Restore a snapshot captured by :func:`_snapshot_model_state`.

    Parameters
    ----------
    model:
        Model to restore in place (preserving tied-parameter identity).
    snapshot:
        Snapshot from :func:`_snapshot_model_state`.
    """

    params, cpu_rng, cuda_rng = snapshot
    with torch.no_grad():
        current = model.state_dict()
        for name, tensor in params.items():
            if name in current:
                current[name].copy_(tensor)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)


@contextlib.contextmanager
def _clean_torch() -> Iterator[None]:
    """Temporarily restore original torch callables for a Dynamo probe.

    TorchLens installs persistent wrappers on the first capture and leaves them
    installed process-wide. Dynamo cannot inline a wrapped torch function, so it
    reports every wrapped call as an ``Attempted to inline function marked as
    skipped`` graph break -- misattributing TorchLens instrumentation as a MODEL
    graph break. Restoring the originals for the probe measures the real model.

    Yields
    ------
    None
        Control while torch is unwrapped; wrappers are re-installed on exit if
        they were installed on entry.
    """

    from .. import _state
    from ..backends.torch.wrappers import unwrap_torch, wrap_torch

    was_decorated = bool(getattr(_state, "_is_decorated", False))
    if was_decorated:
        unwrap_torch()
    try:
        yield
    finally:
        if was_decorated:
            wrap_torch()


def _model_call_args(
    model: nn.Module,
    x: Any,
    trace_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Resolve model call arguments consistently with eager tracing.

    Parameters
    ----------
    model:
        Model whose forward signature resolves input-container ambiguity.
    x:
        Public TorchLens input value.
    trace_kwargs:
        Trace keyword arguments, including optional ``input_kwargs``.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Positional and keyword arguments for Dynamo explain.
    """

    coerced = _coerce_input_args(model, x)
    args = tuple(normalize_input_args(coerced, model))
    input_kwargs = trace_kwargs.get("input_kwargs")
    kwargs = dict(input_kwargs) if isinstance(input_kwargs, dict) else {}
    return args, kwargs


def _op_locations(op: Any) -> tuple[tuple[str, int], ...]:
    """Return recorded source locations for one eager operation.

    Parameters
    ----------
    op:
        TorchLens operation record.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Absolute source filenames and one-indexed line numbers.
    """

    locations: list[tuple[str, int]] = []
    for frame in getattr(op, "code_context", ()) or ():
        source_file = getattr(frame, "file", None)
        line_number = getattr(frame, "line_number", None)
        if source_file is None or not isinstance(line_number, int):
            continue
        locations.append((os.path.abspath(str(source_file)), line_number))
    return tuple(locations)


def _correlate_break(trace: Any, record: Any) -> GraphBreak:
    """Map one normalized Dynamo break to nearest same-file eager ops.

    Parameters
    ----------
    trace:
        Completed eager TorchLens trace.
    record:
        Version-neutral break record from the compatibility layer.

    Returns
    -------
    GraphBreak
        Public correlation record.
    """

    if record.source_file is None or record.line_number is None:
        return GraphBreak(
            record.reason,
            record.source_file,
            record.line_number,
            (),
            "Dynamo did not provide a source file and line number.",
        )
    source_file = os.path.abspath(record.source_file)
    distances: list[tuple[int, int, str]] = []
    for op in trace._layer_list:
        label = str(getattr(op, "label", getattr(op, "layer_label", "<unknown>")))
        same_file_lines = [line for filename, line in _op_locations(op) if filename == source_file]
        if same_file_lines:
            distances.append(
                (
                    min(abs(line - record.line_number) for line in same_file_lines),
                    int(getattr(op, "ordinal_index", 0)),
                    label,
                )
            )
    if not distances:
        return GraphBreak(
            record.reason,
            record.source_file,
            record.line_number,
            (),
            "No eager operation recorded a source location in the Dynamo break file.",
        )
    nearest_distance = min(distance for distance, _, _ in distances)
    labels = tuple(
        label
        for distance, _, label in sorted(distances, key=lambda item: (item[0], item[1]))
        if distance == nearest_distance
    )
    return GraphBreak(record.reason, record.source_file, record.line_number, labels, None)


def graph_breaks(model: nn.Module, x: Any, **trace_kwargs: Any) -> GraphBreakReport:
    """Report Dynamo graph breaks and correlate them to eager trace operations.

    The Dynamo probe runs against ORIGINAL (unwrapped) torch so TorchLens's
    persistent wrappers are not misreported as model graph breaks, and the
    model's parameters, buffers, and global RNG are snapshotted and restored so
    neither the probe nor the eager correlation trace mutates the caller's
    model. Both runs observe identical state, so a stateful branching model
    takes the same path in the probe and the correlation trace. The Dynamo
    result is normalized through ``torchlens._utils._dynamo_explain_shapes`` without
    torch-version parsing. Correlation uses recorded :class:`FuncCallLocation`
    file/line evidence; empty matches state why no correlation was possible.

    Parameters
    ----------
    model:
        Eager PyTorch module to inspect.
    x:
        Model input accepted by :func:`torchlens.trace`.
    **trace_kwargs:
        Additional keyword arguments forwarded to :func:`torchlens.trace`.

    Returns
    -------
    GraphBreakReport
        Normalized breaks and nearest eager operation labels.

    Raises
    ------
    GraphBreaksUnavailableError
        If this torch runtime lacks ``torch._dynamo.explain``.
    GraphBreaksNormalizationError
        If the runtime returns an unrecognized explain-result shape.
    """

    if not _torch_compat.HAS_DYNAMO_EXPLAIN:
        raise GraphBreaksUnavailableError(
            "torch._dynamo.explain is unavailable in this torch runtime"
        )
    args, kwargs = _model_call_args(model, x, trace_kwargs)
    snapshot = _snapshot_model_state(model)
    try:
        with _clean_torch():
            try:
                raw_explanation = _torch_compat.run_dynamo_explain(model, args, kwargs)
            except RuntimeError as exc:
                if _torch_compat.get_dynamo_explain() is None:
                    raise GraphBreaksUnavailableError(str(exc)) from exc
                raise
        # Reset so the eager correlation trace observes the SAME state the
        # probe did (a stateful branching model must take the same path in both
        # runs for correlation to be valid), not the probe's mutated state.
        _restore_model_state(model, snapshot)
        try:
            normalized = normalize_dynamo_explain_output(raw_explanation)
        except _DynamoExplainOutputError as exc:
            raise GraphBreaksNormalizationError(str(exc)) from exc

        from .. import trace as capture_trace

        eager_trace = capture_trace(model, x, **trace_kwargs)
        return GraphBreakReport(
            tuple(_correlate_break(eager_trace, record) for record in normalized)
        )
    finally:
        # Neither the probe nor the eager trace should leave the caller's model
        # mutated (buffers or global RNG).
        _restore_model_state(model, snapshot)


__all__ = [
    "GraphBreak",
    "GraphBreakReport",
    "GraphBreaksNormalizationError",
    "GraphBreaksUnavailableError",
    "graph_breaks",
]
