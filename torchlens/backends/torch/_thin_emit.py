"""The thin emit: one plain admitted call, its observations, its tensor effects, one append.

Stage 1 of the thin record path. A call the fast lane admitted whose raw output is
one exact tensor in a plain session (``plain_single_output``: no device-memory
tracking, no predicate save options, no aten values, no nonfinite tracking, no
echo session, no alias-mutation contract) is logged here instead of through
``log_function_output_tensors``. It runs the same helpers the exhaustive emit runs
for that call, in the same order: the shared fields, the output entries, the
per-output fields, the bare-output alias check, the live fire results, the save,
the parent edge truth, the commit (:func:`._thin_record.commit_thin`), the
internal-source classification and the tensor tagging. It does not run the
producer dispatch, the generic per-output loop's branches, the draft, the
``commit_op`` freeze, the fields copy or the live view.

Every other call takes today's emit unchanged; today's emit is the reference.
``TORCHLENS_THIN_EMIT=0`` in the environment forces the thin emit off for the
process (the two-path field differential).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

from ..._capture_state_helpers import _is_uninitialized_param
from ..._utils.display import _timed_phase
from ._completeness_origins import internal_scalar_read
from ._ops_freeze_refs import _module_frames_from_fields
from ._thin_record import commit_thin
from .aliasing import plain_output_alias_semantics

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = ("thin_emit", "thin_emit_enabled", "set_thin_emit_enabled")

_ENABLED = [os.environ.get("TORCHLENS_THIN_EMIT", "1") != "0"]
"""Whether the thin emit serves admitted plain calls (one cell, so tests can flip it)."""

_BARE_ENTRIES = [SimpleNamespace(container_spec=None)]
"""The output-entry shape ``plain_single_output`` accepts: one entry, no container."""

_TENSOR = torch.Tensor


def thin_emit_enabled() -> bool:
    """Return whether the thin emit serves admitted plain calls in this process."""

    return _ENABLED[0]


def set_thin_emit_enabled(enabled: bool) -> bool:
    """Turn the thin emit on or off for this process; return the previous setting."""

    previous = _ENABLED[0]
    _ENABLED[0] = bool(enabled)
    return previous


def thin_emit(trace: Trace, call: tuple[Any, ...], func_call_id: int) -> bool:
    """Log one admitted bottom-level call through the thin path, or return False untouched.

    ``call`` is ``(func, func_name, args, kwargs, arg_copies, kwarg_copies, out,
    exec_ctx)``. Returns False, with no side effect, when the call is not a plain
    single output in a plain exhaustive session; the caller then runs today's emit.
    """

    out = call[6]
    if type(out) is not _TENSOR or not _ENABLED[0] or not _admitted(trace, out):
        return False
    func, func_name, args, kwargs, arg_copies, kwarg_copies, _, exec_ctx = call
    with _timed_phase(trace, "ctx_build_shared_fields"):
        fields, _, _, parent_param_ops = _build_shared_fields_dict(
            trace, func, func_name, args, kwargs, out, exec_ctx, func_call_id
        )
    entries = _output_entries_with_autograd_stats(trace, out)
    entry = entries[0]
    if len(entries) != 1 or entry.container_spec is not None:
        raise RuntimeError("thin emit: a bare tensor output gave a container entry")
    module_stack = _module_frames_from_fields(fields)
    fields["container_path"] = entry.container_path
    fields["container_spec"] = None
    _log_output_tensor_info(
        trace, out, 0, args, kwargs, parent_param_ops, fields, entry.autograd_stats, False
    )
    # The exhaustive emit's internal reads (``requires_grad``, the alias storage reads).
    with internal_scalar_read():
        requires_grad = out.requires_grad
        fields["backend_semantics"] = plain_output_alias_semantics(args, kwargs, out, fields)
    fire_results = _pop_tensor_live_fire_results(out)
    if fire_results:
        fields["fire_results"] = fire_results
        fields["is_edited"] = any(result.replaced for result in fire_results)
    if entry.autograd_band is not None:
        trace.__dict__.setdefault("_autograd_saved_bands", {})[fields["_label_raw"]] = (
            entry.autograd_band
        )
    handle = commit_thin(
        trace, out, fields, (arg_copies, kwarg_copies), module_stack, requires_grad
    )
    if fields["is_internal_source"]:  # ``_classify_new_tensor_in_trace``
        trace.internal_source_ops.append(fields["_label_raw"])
    _tag_tensor_and_track_variations(
        trace, out, handle, fields, arg_copies, kwarg_copies, args, kwargs
    )
    return True


def _admitted(trace: Trace, out: torch.Tensor) -> bool:
    """Return whether today's emit would take its plain single-output path for ``out``.

    The dispatch facts ``log_function_output_tensors`` reads before its producer
    (an uninitialized lazy buffer, a structure-only capture, the exhaustive
    producer), then ``plain_single_output``'s session facts and loggability. A
    trace whose producer policy is not resolved yet takes today's emit, which
    resolves it.
    """

    policy = trace.__dict__.get("_capture_producer_policy")
    if policy is None or policy.emit is not _emit_exhaustive_operation_events:
        return False
    if trace._options_structure_only or _is_uninitialized_param(out):
        return False
    return plain_single_output(trace, out, _BARE_ENTRIES, True)


# Late imports: the split modules of this package import one another (cycles), so this
# module imports them once every definition above exists.
from ._ops_activations import _log_output_tensor_info  # noqa: E402
from ._ops_autograd import _output_entries_with_autograd_stats  # noqa: E402
from ._ops_exhaustive import (  # noqa: E402
    _emit_exhaustive_operation_events,
    plain_single_output,
)
from ._ops_interventions import _pop_tensor_live_fire_results  # noqa: E402
from ._ops_shared_fields import (  # noqa: E402
    _build_shared_fields_dict,
    _tag_tensor_and_track_variations,
)
