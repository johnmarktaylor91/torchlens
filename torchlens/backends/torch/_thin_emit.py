"""The thin emit: one plain admitted call, its observations, its tensor effects, one append.

A call the fast lane admitted whose raw output is one exact tensor in a plain
session (:func:`._thin_settle.plain_session`: no device-memory tracking, no
predicate save options, no aten values, no nonfinite tracking, no echo session,
no alias-mutation contract) is logged here instead of through
``log_function_output_tensors``. It runs the same helpers the exhaustive emit runs
for that call, in the same order: the shared fields, the output entries, the
per-output fields, the bare-output alias check, the live fire results, the save,
the parent edge truth, the commit (:func:`._thin_record.make_commit`), the
internal-source classification and the tensor tagging. It does not run the
producer dispatch, the generic per-output loop's branches, the draft, the
``commit_op`` freeze, the fields copy or the live view.

This module and :mod:`._thin_record` import nothing from the capture import cycle:
the cycle side (``_ops_exhaustive.bound_thin_emit``) passes the helpers in once
(:class:`ThinHelpers`), and ``_ops_emission.commit_admitted`` passes the bound
emit to ``_call_plan.commit_fast_call``.

Every other call takes today's emit unchanged; today's emit is the reference.
``TORCHLENS_THIN_EMIT=0`` in the environment forces the thin emit off for the
process (the two-path field differential).
"""

from __future__ import annotations

import os
import time
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from ..._utils._internal_read import internal_scalar_read
from ..._utils.display import _record_phase_timing
from ._ops_freeze_refs import _module_frames_from_fields
from ._thin_record import make_commit
from ._thin_settle import plain_session
from .aliasing import plain_output_alias_semantics

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ..._ir.events import ModuleFrame

__all__ = ("ThinHelpers", "make_thin_emit", "thin_emit_enabled", "set_thin_emit_enabled")

_ENABLED = [os.environ.get("TORCHLENS_THIN_EMIT", "1") != "0"]
"""Whether the thin emit serves admitted plain calls (one cell, so tests can flip it)."""

_TENSOR = torch.Tensor
_PHASE = "ctx_build_shared_fields"
_FRAME_CACHE_LIMIT = 4096


class ThinHelpers(NamedTuple):
    """The capture-cycle helpers the thin path calls, passed in once by the cycle side."""

    build_shared_fields: Callable[..., Any]
    output_entries: Callable[..., Any]
    log_output_info: Callable[..., Any]
    pop_fire_results: Callable[..., Any]
    tag_tensor: Callable[..., Any]
    exhaustive_emit: Callable[..., Any]
    keep_alias_contract: Callable[[Any], bool]
    save_activation_fields: Callable[..., Any]
    seal_parent_edge_truth: Callable[..., Any]
    build_edge_use_records: Callable[..., Any]
    capture_session_type: type
    op_store_layout: Any


def thin_emit_enabled() -> bool:
    """Return whether the thin emit serves admitted plain calls in this process."""

    return _ENABLED[0]


def set_thin_emit_enabled(enabled: bool) -> bool:
    """Turn the thin emit on or off for this process; return the previous setting."""

    previous = _ENABLED[0]
    _ENABLED[0] = bool(enabled)
    return previous


def make_thin_emit(helpers: ThinHelpers) -> Callable[[Any, tuple[Any, ...], int], bool]:
    """Return the thin emit bound to ``helpers``: it logs one admitted call or declines it."""

    commit = make_commit(helpers)
    frames_of = _frame_binder()
    (build, entries_of, log_info, pop_fire, tag, exhaustive_emit, keep_alias) = helpers[:7]
    enabled = _ENABLED
    perf = time.perf_counter

    def thin_emit(trace: Trace, call: tuple[Any, ...], func_call_id: int) -> bool:
        """Log one admitted bottom-level call through the thin path, or return False untouched.

        ``call`` is ``(func, func_name, args, kwargs, arg_copies, kwarg_copies, out,
        exec_ctx)``. Returns False, with no side effect, when the call is not a plain
        single output in a plain exhaustive session; the caller then runs today's emit.
        The fast-lane gate already paid for the rest of today's plain-output check: an
        admitted call is exhaustive and not structure-only (``admit_call``), and an
        exact ``torch.Tensor`` output without TorchLens metadata (``output_admitted``,
        or the same-object commit's fresh copy) is loggable and never a lazy parameter.
        """

        out = call[6]
        if type(out) is not _TENSOR or not enabled[0]:
            return False
        policy = trace.__dict__.get("_capture_producer_policy")
        if policy is None or policy.emit is not exhaustive_emit:
            return False
        if not plain_session(trace, keep_alias):
            return False
        func, func_name, args, kwargs, arg_copies, kwarg_copies, _, exec_ctx = call
        started = perf()
        try:
            fields, _, _, parent_param_ops = build(
                trace, func, func_name, args, kwargs, out, exec_ctx, func_call_id
            )
        finally:
            _record_phase(trace, perf() - started)
        entries = entries_of(trace, out)
        entry = entries[0]
        if len(entries) != 1 or entry.container_spec is not None:
            raise RuntimeError("thin emit: a bare tensor output gave a container entry")
        module_stack = frames_of(trace, fields["modules"])
        fields["container_path"] = entry.container_path
        log_info(trace, out, 0, args, kwargs, parent_param_ops, fields, entry.autograd_stats, False)
        # The exhaustive emit's internal reads (``requires_grad``, the alias storage reads).
        with internal_scalar_read():
            requires_grad = out.requires_grad
            fields["backend_semantics"] = plain_output_alias_semantics(args, kwargs, out, fields)
        fire_results = pop_fire(out)
        if fire_results:
            fields["fire_results"] = fire_results
            fields["is_edited"] = any(result.replaced for result in fire_results)
        if entry.autograd_band is not None:
            trace.__dict__.setdefault("_autograd_saved_bands", {})[fields["_label_raw"]] = (
                entry.autograd_band
            )
        handle = commit(trace, out, fields, (arg_copies, kwarg_copies), module_stack, requires_grad)
        if fields["is_internal_source"]:  # ``_classify_new_tensor_in_trace``
            trace.internal_source_ops.append(fields["_label_raw"])
        tag(trace, out, handle, fields, arg_copies, kwarg_copies, args, kwargs)
        return True

    return thin_emit


def _record_phase(trace: Trace, elapsed: float) -> None:
    """Add one ``ctx_build_shared_fields`` sample, as ``_timed_phase`` records it."""

    try:
        stats = trace.__dict__["_phase_timings"][_PHASE]
    except KeyError:
        _record_phase_timing(trace, _PHASE, elapsed)
        return
    stats["total_s"] += elapsed
    stats["count"] += 1


def _frame_binder() -> Callable[[Any, list[tuple[str, int]]], tuple[ModuleFrame, ...]]:
    """Return a per-trace builder of the op's module frames (``_module_frames_from_fields``).

    The frames of one ``(address, pass index)`` stack are built once per trace and
    the frozen tuple is shared by every op of the trace with that stack; a new trace
    starts a new table.
    """

    bound: list[Any] = [None, {}]

    def frames_of(trace: Any, modules: list[tuple[str, int]]) -> tuple[ModuleFrame, ...]:
        ref = bound[0]
        if ref is None or ref() is not trace:
            bound[0], bound[1] = weakref.ref(trace), {}
        table: dict[tuple[tuple[str, int], ...], tuple[ModuleFrame, ...]] = bound[1]
        key = tuple(modules)
        frames = table.get(key)
        if frames is None:
            if len(table) >= _FRAME_CACHE_LIMIT:
                table.clear()
            frames = table[key] = _module_frames_from_fields({"modules": key})
        return frames

    return frames_of
