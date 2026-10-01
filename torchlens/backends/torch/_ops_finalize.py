"""Predicate context evaluation and layer-entry finalization."""

import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch

from ..._state import pause_logging
from ...capture.predicates import (
    _evaluate_keep_op,
)
from ...capture.projections import (
    commit_op,
)
from ...capture.session import capture_session_for
from ...capture.stop import stop_directive_for_trace
from ...data_classes._nonfinite import record_op_nonfinite
from ...data_classes.op import (
    _recursive_safe_copy,
)
from ...fastlog.types import (
    CaptureSpec,
    RecordContext,
)
from ...intervention.selectors import (
    BaseSelector,
)
from ...ir.events import (
    FunctionCallRef,
    ModuleFrame,
)
from ...ir.predicate import RetroactiveCaptureDecision
from ...utils.display import _timed_phase
from ...utils.tensor_utils import (
    fp8_widen_for_numeric_ops,
    safe_copy,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

if TYPE_CHECKING:
    from .ops import (
        ExhaustiveOpDraft,
        _admit_saved_args_budget,
        _append_trace_predicate_context,
        _apply_retroactive_decision,
        _build_trace_predicate_context,
        _commit_saved_args_budget,
        _retain_lookback_candidate,
        _save_activation_fields,
        _save_predicate_activation_fields,
    )

__all__ = (
    "_trace_predicate_context_key",
    "_cache_trace_predicate_context",
    "_pop_trace_predicate_context",
    "_evaluate_trace_save_predicate",
    "_module_filter_namespace",
    "_note_module_filter_suppression",
    "_make_layer_log_entry",
    "_raise_if_nonfinite_requested",
    "_record_nonfinite_if_requested",
)


def _trace_predicate_context_key(
    raw_label: str,
    pass_index: int,
    container_path: tuple[Any, ...],
) -> tuple[str, int, tuple[Any, ...]]:
    """Return the runtime key for cached trace predicate contexts."""

    return (raw_label, pass_index, container_path)


def _cache_trace_predicate_context(
    trace: "Trace",
    ctx: RecordContext,
    container_path: tuple[Any, ...],
) -> None:
    """Cache a pre-save predicate context for the current op."""

    contexts = getattr(trace, "_predicate_current_contexts", None)
    if contexts is None:
        contexts = {}
        trace._predicate_current_contexts = contexts
    contexts[
        _trace_predicate_context_key(
            ctx.raw_label or ctx.label,
            ctx.pass_index,
            container_path,
        )
    ] = ctx


def _pop_trace_predicate_context(
    trace: "Trace",
    fields_dict: dict[str, Any],
) -> RecordContext | None:
    """Pop a cached predicate context for an op, if one exists."""

    contexts = getattr(trace, "_predicate_current_contexts", None)
    if contexts is None:
        return None
    key = _trace_predicate_context_key(
        str(fields_dict["_label_raw"]),
        int(fields_dict.get("pass_index", 0)),
        tuple(fields_dict.get("container_path", ())),
    )
    return contexts.pop(key, None)


def _evaluate_trace_save_predicate(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
) -> tuple[CaptureSpec | None, RecordContext | None]:
    """Evaluate ``trace(save=...)`` for one exhaustive op, if configured.

    Parameters
    ----------
    trace:
        Active trace.
    fields_dict:
        Live exhaustive op fields for the tensor.
    tensor:
        Output tensor being considered.

    Returns
    -------
    CaptureSpec | None
        Capture decision when selective predicate save is active, otherwise ``None``.
    """

    options = getattr(trace, "_predicate_save_options", None)
    if options is None:
        return None, None
    if options.keep_op is None:
        return None, None
    ctx = _pop_trace_predicate_context(trace, fields_dict)
    if ctx is None:
        ctx = _build_trace_predicate_context(trace, fields_dict, tensor)
    raw_label = str(fields_dict["_label_raw"])
    try:
        decision = _evaluate_keep_op(ctx, options)
    finally:
        _append_trace_predicate_context(trace, ctx)
    if isinstance(decision, RetroactiveCaptureDecision):
        _apply_retroactive_decision(trace, decision)
        spec = CaptureSpec(save_out=False, save_metadata=True)
    else:
        spec = decision

    if isinstance(options.keep_op, BaseSelector) and (spec.save_out or spec.save_metadata):
        trace._tl_save_selector_fire_count = (
            int(getattr(trace, "_tl_save_selector_fire_count", 0)) + 1
        )
    decisions = getattr(trace, "_predicate_save_decisions", None)
    if decisions is None:
        decisions = {}
        trace._predicate_save_decisions = decisions
    decisions[
        (
            raw_label,
            int(fields_dict.get("pass_index", 0)),
            tuple(fields_dict.get("container_path", ())),
        )
    ] = spec
    return spec, ctx


def _module_filter_namespace(fields_dict: dict[str, Any]) -> SimpleNamespace:
    """Build the legacy-shaped compatibility namespace ``module_filter`` sees.

    The documented compatibility surface (producer unification 2.6): the
    filter runs ONLY inside the exhaustive commit pipeline, and the namespace
    carries the legacy key names WITHOUT ``fire_results`` (popped before the
    filter today — preserved exactly). The decomposed producer builds this
    same namespace lazily from its draft.
    """

    return SimpleNamespace(**fields_dict)


def _note_module_filter_suppression(
    self: "Trace", keep_by_predicate: bool, save_this_activation: bool
) -> None:
    """Count a payload the save selection picked but ``module_filter`` alone dropped.

    Disclosure counter for the third save gate: a capture whose EVERY selected
    payload was suppressed here warns at the trace entry
    (``module_filter_zero_saved``) instead of returning a silently
    payload-free trace.
    """

    if save_this_activation and not keep_by_predicate:
        self.__dict__["_tl_module_filter_suppressed"] = (
            int(self.__dict__.get("_tl_module_filter_suppressed", 0)) + 1
        )


def _make_layer_log_entry(
    self: "Trace",
    t: torch.Tensor,
    fields_dict: dict[str, Any],
    t_args: tuple[Any, ...] | None = None,
    t_kwargs: dict[str, Any] | None = None,
    activation_transform: Callable[..., Any] | None = None,
    event_module_stack: tuple[ModuleFrame, ...] | None = None,
    call_ref_box: list[FunctionCallRef] | None = None,
) -> Any:
    """Create a Op (or Buffer) entry and register it in Trace.

    Instantiates the appropriate log class from ``fields_dict``, conditionally
    saves out data (if this layer is in ``_layer_nums_to_save``), and
    appends the entry to ``_raw_layer_dict`` and ``_raw_layer_labels_list``.

    Args:
        t: The tensor to log.
        fields_dict: Complete field dictionary (~80 fields) for the log entry.
        t_args: Positional arguments to the function that created the tensor.
        t_kwargs: Keyword arguments to the function that created the tensor.
        activation_transform: Optional transform applied to outs before saving.
        event_module_stack: Optional precomputed immutable module frames.
    """
    if t_args is None:
        t_args = ()
    if t_kwargs is None:
        t_kwargs = {}

    fire_results = tuple(fields_dict.pop("fire_results", ()))

    keep_by_predicate = True
    module_filter = getattr(self, "module_filter", None)
    if module_filter is not None:
        keep_by_predicate = bool(module_filter(_module_filter_namespace(fields_dict)))
    layer_nums_to_save = cast(Any, self._layer_nums_to_save)
    raw_index = cast(int, fields_dict["raw_index"])
    predicate_spec, predicate_ctx = _evaluate_trace_save_predicate(self, fields_dict, t)
    if predicate_spec is None:
        save_this_activation = (layer_nums_to_save == "all") or (raw_index in layer_nums_to_save)
    else:
        save_this_activation = predicate_spec.save_out
    capture_session = capture_session_for(self)
    if capture_session is not None:
        capture_session.escrow_candidate(
            raw_index,
            t,
            retain_activation=not (keep_by_predicate and save_this_activation),
        )
    _note_module_filter_suppression(self, keep_by_predicate, save_this_activation)
    if keep_by_predicate and save_this_activation:
        if predicate_spec is None or predicate_ctx is None:
            with _timed_phase(self, "clone_save:activation_fields"):
                _save_activation_fields(
                    self,
                    fields_dict,
                    t,
                    t_args,
                    t_kwargs,
                    activation_transform,
                )
        else:
            with _timed_phase(self, "clone_save:activation_fields"):
                _save_predicate_activation_fields(
                    self,
                    fields_dict,
                    t,
                    predicate_spec,
                    predicate_ctx,
                    activation_transform,
                )
    if predicate_spec is not None and self.save_arg_values and not fields_dict["has_saved_args"]:
        # Admit BEFORE the clones allocate (r8 R34, sol 2).
        arg_reservations = _admit_saved_args_budget(self, fields_dict, t_args, t_kwargs)
        fields_dict["has_saved_args"] = True
        fields_dict["saved_args"] = [_recursive_safe_copy(arg) for arg in t_args]
        fields_dict["saved_kwargs"] = {
            key: _recursive_safe_copy(value) for key, value in t_kwargs.items()
        }
        _commit_saved_args_budget(self, fields_dict, arg_reservations)
    # r29 F3b: seal this record's capture-time (slot -> producer) truth on the
    # Trace, keyed by raw label. The witness (``dropped_edge_tensor_args``) is
    # stamped at CAPTURE, so an edge dropped DOWNSTREAM of it -- anywhere in the
    # 20-step postprocess pipeline or later -- was invisible whenever the
    # payload was trivial. The post-pipeline metadata invariant
    # (``validation/invariants.py::_check_capture_edges_survive_postprocess``)
    # reconciles the final graph against this sealed truth. Runtime-only
    # bookkeeping (same class as ``_validation_orphan_candidate_index``): never
    # persisted, absent on loaded traces.
    _capture_edge_truth = self.__dict__.setdefault("_capture_parent_edge_truth", {})
    _positions = fields_dict["parent_arg_positions"]
    _capture_edge_truth[fields_dict["_label_raw"]] = (
        tuple(("args", key, label) for key, label in _positions["args"].items())
        + tuple(("kwargs", key, label) for key, label in _positions["kwargs"].items())
        + tuple(
            ("parent", None, label)
            for label in fields_dict["parents"]
            if label not in _positions["args"].values()
            and label not in _positions["kwargs"].values()
        )
    )
    # The ONE commit tail (freeze -> atomic append) plus the two declared
    # exhaustive-only post-tail stages (grad-handle index, LiveOpView) —
    # see capture.projections.COMMIT_STAGE_MATRIX.
    new_entry = commit_op(
        self,
        ExhaustiveOpDraft(
            trace=self,
            fields_dict=fields_dict,
            tensor=t,
            fire_results=fire_results,
            module_stack=event_module_stack,
            call_ref_box=call_ref_box,
        ),
    )
    if predicate_ctx is not None and not save_this_activation:
        _retain_lookback_candidate(self, predicate_ctx, fields_dict, t)
    _record_nonfinite_if_requested(self, t, new_entry)
    _raise_if_nonfinite_requested(self, t, new_entry)

    return new_entry


def _record_nonfinite_if_requested(self: Any, tensor: torch.Tensor, entry: Any) -> None:
    """Record this op output's finiteness when ``track_nonfinite`` is enabled.

    Independent of ``raise_on_nan`` (which stays a stop-and-throw and is
    untouched): recording never changes control flow, only lands a per-op
    verdict in the trace-side runtime store served by
    ``Trace.nonfinite_ops`` / ``Trace.nonfinite_coverage``. Device flags are
    deferred, never read here, so the forward is never synchronized per op.

    Parameters
    ----------
    self:
        Active ``Trace`` instance.
    tensor:
        Tensor output produced by the just-logged operation.
    entry:
        Newly registered layer pass log for ``tensor``.
    """

    if not getattr(self, "track_nonfinite", False):
        return
    raw_label = getattr(entry, "_label_raw", getattr(entry, "_layer_label_raw", None))
    if raw_label is None:
        return
    record_op_nonfinite(self, tensor, str(raw_label))


def _raise_if_nonfinite_requested(self: Any, tensor: torch.Tensor, entry: Any) -> None:
    """Raise a structured capture error if ``raise_on_nan`` finds a non-finite tensor.

    Parameters
    ----------
    self:
        Active ``Trace`` instance.
    tensor:
        Tensor output produced by the just-logged operation.
    entry:
        Newly registered layer pass log for ``tensor``.

    Raises
    ------
    CaptureError
        If ``self.raise_on_nan`` is enabled and ``tensor`` contains NaN or Inf.
    """

    if not getattr(self, "raise_on_nan", False):
        return
    try:
        with pause_logging():
            # The ``numel()`` read MUST also sit under pause_logging: ``numel``
            # is a wrapped call, and running it bare on a just-committed BUFFER
            # source tensor (not yet registered as logged) re-entered buffer
            # source logging and recursed without bound -- raise_on_nan=True
            # crashed with RecursionError on ANY BatchNorm-bearing model. The
            # check semantics are byte-identical (empty tensors still return,
            # the same kernel decides, the same abort fires).
            if tensor.numel() == 0:
                return
            # fp8 has no ``isfinite`` kernel, and ``NotImplementedError`` is a
            # ``RuntimeError`` subclass -- so without the exact float32 widening this
            # tripwire SILENTLY declined to check every fp8 activation. Widening keeps
            # the verdict identical (see fp8_widen_for_numeric_ops).
            has_nonfinite = bool(
                (~torch.isfinite(fp8_widen_for_numeric_ops(safe_copy(tensor, detach_tensor=True))))
                .any()
                .item()
            )
    except (RuntimeError, TypeError) as exc:
        # An unrunnable check is NOT a clean tensor. fp8 was the known real case and
        # is handled above, but any dtype/layout without an ``isfinite`` kernel lands
        # here -- and the user explicitly asked for NaN checking, so silence would let
        # them read an unchecked forward as a checked one. Warn once per capture
        # (naming the first skipped op) and keep going: an opt-in diagnostic must not
        # convert an exotic dtype into a failed capture.
        if "nonfinite_check_unavailable" not in self._warned_once:
            self._warned_once.add("nonfinite_check_unavailable")
            warnings.warn(
                "raise_on_nan could not check at least one activation: "
                f"{type(exc).__name__}: {exc}. First skipped op "
                f"{getattr(entry, 'func_name', 'unknown')!r} has dtype {tensor.dtype} on "
                f"{tensor.device}. Those activations are UNCHECKED for NaN/Inf; a clean "
                "capture does not mean they were finite.",
                UserWarning,
                stacklevel=2,
            )
        return
    if not has_nonfinite:
        return

    raw_label = getattr(entry, "_label_raw", getattr(entry, "_layer_label_raw", "unknown"))
    func_name = getattr(entry, "func_name", "unknown")
    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    parents = list(getattr(entry, "parents", []) or [])
    stop_directive_for_trace(self).raise_nonfinite(
        raw_label=raw_label,
        func_name=func_name,
        shape=shape,
        dtype=dtype,
        parents=parents,
    )
