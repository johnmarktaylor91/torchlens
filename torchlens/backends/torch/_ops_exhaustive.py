"""Exhaustive event emission and autograd candidate collection."""

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ..._capture.session import CaptureSession
from ..._capture.stop import evaluate_halt_stop
from ..._data_classes.internal_types import FuncExecutionContext
from ..._data_classes.op import (
    _OP_STORE_LAYOUT,
    Op,
)
from ..._ir.events import (
    FunctionCallRef,
)
from ..._ir.intervention import FunctionEventInput
from ..._utils._tensor_label import (
    get_tensor_label,
    set_tensor_label,
)
from ..._utils._torch_compat import (
    saved_tensors_default_hooks_active,
)
from ..._utils.collections import index_nested
from ..._utils.display import _timed_phase
from ..._utils.tensor_utils import (
    safe_copy,
)
from ._completeness_origins import internal_scalar_read
from ._ops_descriptive_fields import _set_descriptive_fields
from ._ops_freeze_refs import _module_frames_from_fields
from ._ops_row_journal import _OUTPUT_REQUIRES_GRAD
from ._thin_emit import ThinHelpers, make_thin_emit
from ._thin_settle import plain_session
from ._user_grad_fn import stamp_user_grad_fn
from .aliasing import (
    detect_torch_alias_contract,
    detect_torch_output_alias_contract,
    plain_output_alias_semantics,
)
from .tensor_tracking import (
    _add_tensor_backward_hook,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = (
    "_emit_exhaustive_operation_events",
    "_project_foreach_member_parent_fields",
    "_get_parent_contents",
    "_output_should_be_logged",
    "_check_if_tensor_arg",
    "_iter_autograd_saved_candidates",
    "_collect_tensor_values",
)


def _emit_exhaustive_operation_events(
    self: "Trace",
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> None:
    """Full metadata logging for each output tensor of a function call.

    For each loggable output tensor:
      1. Build per-tensor fields (label, shape, equivalence type, FLOPs).
      2. Create a Op entry and optionally save out data.
      3. Update bidirectional family links (parent→child, sibling, spouse).
      4. Tag the output tensor with ``_tl.label_raw`` (downstream ops find parents).
      5. Track parent content variations (for in-place mutation detection).

    Args:
        func: The original (unwrapped) function that was called.
        args, kwargs: Positional and keyword arguments to the function.
        arg_copies, kwarg_copies: Pre-call copies of args and kwargs.
        out_orig: Original output from the function (may be tensor, tuple, etc.).
        exec_ctx: Timing, RNG, and autocast state captured around the function call.
        is_bottom_level_func: True for a leaf of the decoration nesting.
    """
    if not is_bottom_level_func and _nested_call_logged_output(out_orig):
        # The skipped body's one func_call_id consumer: the id sequence stays the same.
        _output_entries_with_autograd_stats(self, out_orig)
        return
    with _timed_phase(self, "ctx_build_shared_fields"):
        (
            fields_dict,
            parent_layer_entries,
            arg_tensors,
            parent_param_ops,
        ) = _build_shared_fields_dict(
            self,
            func,
            func_name,
            args,
            kwargs,
            out_orig,
            exec_ctx,
            func_call_id,
        )

    output_entries = _output_entries_with_autograd_stats(self, out_orig)
    if plain_single_output(self, out_orig, output_entries, is_bottom_level_func):
        call = (args, kwargs, arg_copies, kwarg_copies)  # the loop below, specialized
        emit_plain_output(self, call, output_entries[0], fields_dict, parent_param_ops)
        return
    _register_call_output_container_snapshot(
        self,
        out_orig,
        output_entries=output_entries,
        func_call_id=func_call_id,
        event_index=int(fields_dict.get("raw_index") or func_call_id),
    )
    expected_output_count = len(output_entries)
    loggable_output_count, single_verdict = _loggable_outputs(output_entries, is_bottom_level_func)
    use_single_output_fields = (
        loggable_output_count == 1
        and len(output_entries) == 1
        and output_entries[0].container_spec is None
    )
    event_module_stack = _module_frames_from_fields(fields_dict)
    shared_func_event_input = FunctionEventInput(
        func=func,
        func_name=func_name,
        func_qualname=getattr(func, "__qualname__", None),
        args=args,
        kwargs=kwargs,
        raw_output=out_orig,
        arg_copies=arg_copies,
        kwarg_copies=kwarg_copies,
        module_stack=event_module_stack,
        is_bottom_level_func=is_bottom_level_func,
        func_call_id=func_call_id,
        expected_output_count=expected_output_count,
    )

    # Container entries that ARE an input object (broadcast_tensors(x, x) with
    # conforming shapes, atleast_1d/2d/3d, ...) are pure pass-throughs: torch
    # hands back the caller's own tensor inside the returned tuple. Labeling
    # the live input directly would steal its label (last-wins for duplicated
    # entries), leave phantom dead-end siblings, and reroute downstream direct
    # consumers of the input through an op whose result the user may never use.
    # Mirror the scalar pass-through machinery: log each such
    # entry against a minted safe copy so the live input keeps its label.
    # out= destinations are excluded -- the op genuinely WROTE into them, so
    # the live destination must keep advancing to this op's label.
    out_destination_ids = _out_destination_ids(kwargs) if kwargs else _NO_IDS

    # ``torch._foreach_*`` semantics are ZIPPED: output ``i`` is computed from
    # member ``i`` of each tensor-list argument (plus any whole-call scalar or
    # single-tensor operand). Sharing the call-level parent set across every
    # output would invent all-to-all dependencies, corrupting
    # influence geometry for every consumer.
    is_foreach_call = func_name.startswith("_foreach_")

    # List-returning in-place ops (``torch._foreach_add_`` and family) return a
    # NEW list whose members ARE the mutated receiver tensors from ``args[0]``.
    # Whole-return identity (``id(out) == id(args[0])``) never fires for them,
    # so the live members kept their pre-mutation labels and every downstream
    # consumer bypassed the mutation node (round-31 H1). Detect the receiver
    # members here and thread each member's freshly minted label back onto the
    # live tensor after logging, exactly like the scalar in-place path.
    is_inplace_list_return = (
        func_name.endswith("_")
        and not func_name.startswith("__")
        and bool(args)
        and isinstance(args[0], (list, tuple))
    )
    receiver_member_ids: frozenset[int] = (
        frozenset(id(item) for item in args[0] if isinstance(item, torch.Tensor))
        if is_inplace_list_return
        else frozenset()
    )

    # One shared FunctionCallRef per wrapped call: the first logged
    # output parks the frozen ref here and every sibling reuses it.
    call_ref_box: list[FunctionCallRef] = []

    keep_alias_contract = _should_keep_alias_mutation_contract(self)
    for i, output_entry in enumerate(output_entries):
        out = output_entry.value
        if single_verdict is None:
            if not _output_should_be_logged(out, is_bottom_level_func):
                continue
        elif not single_verdict:
            continue
        out_tensor: torch.Tensor = out
        if (
            output_entry.container_spec is not None
            and id(out_tensor) not in out_destination_ids
            and any(out_tensor is arg_tensor for arg_tensor in arg_tensors)
        ):
            # Preserve the live member's real autograd node; the
            # minted copy's ``CloneBackward`` is bookkeeping, not op metadata.
            live_member_grad_fn = out_tensor.grad_fn
            out_tensor = safe_copy(out_tensor)
            stamp_user_grad_fn(out_tensor, live_member_grad_fn)

        # per-output isolation is a plain dict copy. Every per-output
        # writer REASSIGNS its fields (``_log_output_tensor_info``,
        # ``_build_graph_relationship_fields``, the foreach projection, and
        # this loop all bind fresh containers; the single-output path has
        # always shared ``fields_dict`` itself, so in-place mutation of a
        # call-shared container would already be a bug there). The frozen
        # ``OpEvent`` isolates at construction (tuple/deepcopy), so sibling
        # dicts sharing the call-level container objects is unobservable.
        fields_dict_onetensor = fields_dict if use_single_output_fields else dict(fields_dict)
        fields_dict_onetensor["container_path"] = output_entry.container_path
        fields_dict_onetensor["container_spec"] = output_entry.container_spec
        if output_entry.container_spec is not None:
            fields_dict_onetensor["in_multi_output"] = True
        if is_foreach_call and output_entry.container_spec is not None:
            _project_foreach_member_parent_fields(
                self,
                fields_dict_onetensor,
                output_entry,
                args,
                kwargs,
                out_orig,
            )
        _log_output_tensor_info(
            self,
            out_tensor,
            i,
            args,
            kwargs,
            parent_param_ops,
            fields_dict_onetensor,
            output_entry.autograd_stats,
            keep_alias_contract,
        )
        if i == 0 and getattr(self, "track_device_memory", False):
            _project_call_device_memory(self, func_call_id, fields_dict_onetensor)
        detect_backend_semantics = (
            detect_torch_alias_contract
            if keep_alias_contract
            else detect_torch_output_alias_contract
        )
        # Mutation/alias detection (``tensor_nanequal`` -> ``torch.equal`` /
        # ``torch.allclose``) and the commit's output ``requires_grad`` are internal
        # reads; mark them so the completeness witness records no user host escape.
        with internal_scalar_read():
            _OUTPUT_REQUIRES_GRAD[:] = (weakref.ref(out_tensor), out_tensor.requires_grad)
            fields_dict_onetensor["backend_semantics"] = detect_backend_semantics(
                shared_func_event_input,
                backend_grad_handle=fields_dict_onetensor["grad_fn_handle"],
                grad_fn_class_name=fields_dict_onetensor["grad_fn_class_name"],
                autograd_memory=fields_dict_onetensor["autograd_memory"],
                num_autograd_tensors=fields_dict_onetensor["num_autograd_tensors"],
                bytes_delta_at_call=fields_dict_onetensor["bytes_delta_at_call"],
                bytes_peak_at_call=fields_dict_onetensor["bytes_peak_at_call"],
            )
        fire_results = _pop_tensor_live_fire_results(out_tensor)
        if fire_results:
            # The record's ``edits`` derive from ``fire_results`` at scatter.
            fields_dict_onetensor["fire_results"] = fire_results
            fields_dict_onetensor["is_edited"] = any(result.replaced for result in fire_results)
        if getattr(self, "is_replayable", False):
            fields_dict_onetensor["_edge_uses"] = _build_edge_use_records(
                self,
                fields_dict_onetensor["parent_arg_positions"],
                fields_dict_onetensor["_label_raw"],
                func_call_id,
                func_name=func_name,
            )
        if output_entry.autograd_band is not None:
            # Observe items 7-8: the saved-band decomposition keyed by raw
            # label (session-time bookkeeping, same class as
            # ``_capture_parent_edge_truth``; never persisted). Consumers join
            # to public labels through the step-8 identity map.
            self.__dict__.setdefault("_autograd_saved_bands", {})[
                fields_dict_onetensor["_label_raw"]
            ] = output_entry.autograd_band
        new_layer_entry: Op = _make_layer_log_entry(
            self,
            out_tensor,
            fields_dict=fields_dict_onetensor,
            t_args=arg_copies,
            t_kwargs=kwarg_copies,
            activation_transform=self._options_activation_transform,
            event_module_stack=event_module_stack,
            call_ref_box=call_ref_box,
        )
        # The committed record backs the returned view; its reads go to the record directly.
        new_record = object.__getattribute__(new_layer_entry, "_record")
        new_tensor_label = new_record.label_raw

        _classify_new_tensor_in_trace(self, fields_dict, new_tensor_label)
        _tag_tensor_and_track_variations(
            self,
            out_tensor,
            new_record,
            fields_dict_onetensor,
            arg_copies,
            kwarg_copies,
            args,
            kwargs,
        )
        # Round-31 H1: the member entry was logged against a minted safe copy, but for a
        # list-returning in-place op the LIVE member is the mutated receiver the caller keeps
        # using. Advance its label to this op, hook its gradient and propagate to overlapping
        # storage aliases (the scalar in-place repair), so consumers bind to the mutation.
        live_member = output_entry.value
        if (
            id(live_member) in receiver_member_ids
            and live_member is not out_tensor
            and isinstance(live_member, torch.Tensor)
            and not isinstance(live_member, torch.nn.Parameter)
        ):
            from .wrappers import _propagate_mutation_label_to_storage_aliases

            set_tensor_label(live_member, new_tensor_label)
            # The live member is what downstream ops consume, so it takes
            # gradient ownership of the label (the logged safe copy's hook
            # stops emitting) -- the same transfer the scalar same-object
            # in-place path performs in _register_inplace_live_grad_hook.
            _add_tensor_backward_hook(self, live_member, new_tensor_label, take_ownership=True)
            _propagate_mutation_label_to_storage_aliases(self, live_member, new_tensor_label)
        options = self.__dict__.get("_predicate_save_options")
        if options is not None and options.halt is not None:
            halt_ctx = _build_trace_predicate_context(self, fields_dict_onetensor, out_tensor)
            evaluate_halt_stop(self, halt_ctx, options, frontier_output=out_tensor)


# The exhaustive emit of one bare tensor output in a plain session: a call whose raw
# output is one exact tensor (one entry, no container) in a session asking for none of the
# per-op extras runs the generic per-output loop's steps for it, in order and through the
# same helpers, without the branches that cannot apply (safe copy, foreach projection,
# edge uses, receiver relabel, halt); the output alias check is the bare-output one.


def plain_single_output(
    trace: "Trace", out_orig: Any, output_entries: list[Any], is_bottom_level_func: bool
) -> bool:
    """Return whether the call has one loggable bare tensor output and a plain session.

    The output is one exact ``torch.Tensor`` with one entry and no container,
    and ``_output_should_be_logged`` accepts it (an output it rejects takes the
    generic loop). The session facts are read per call, as the generic emit reads
    them, so a session that switches one on mid-forward takes the generic loop
    from that call.
    """

    if (
        type(out_orig) is not torch.Tensor
        or len(output_entries) != 1
        or output_entries[0].container_spec is not None
    ):
        return False
    return plain_session(trace, _should_keep_alias_mutation_contract) and _output_should_be_logged(
        out_orig, is_bottom_level_func
    )


_BOUND_THIN: dict[str, Any] = {}
"""The thin emit bound to this package's helpers (key ``"thin"``), built on first use."""


def bound_thin_emit() -> Callable[[Any, tuple[Any, ...], int], bool]:
    """Return the thin emit bound to the capture helpers (``_thin_emit.make_thin_emit``).

    The cycle side binds the thin path's helpers, so the ``_thin_*`` modules import
    nothing from the capture import cycle; ``commit_admitted`` passes the result to
    ``_call_plan.commit_fast_call``.
    """

    thin = _BOUND_THIN.get("thin")
    if thin is None:
        thin = _BOUND_THIN["thin"] = make_thin_emit(
            ThinHelpers(
                _build_shared_fields_dict,
                _output_entries_with_autograd_stats,
                _log_output_tensor_info,
                _pop_tensor_live_fire_results,
                _tag_tensor_and_track_variations,
                _emit_exhaustive_operation_events,
                _should_keep_alias_mutation_contract,
                _save_activation_fields,
                _seal_parent_edge_truth,
                _build_edge_use_records,
                CaptureSession,
                _OP_STORE_LAYOUT,
                _set_descriptive_fields,
            )
        )
    return thin


def emit_plain_output(
    trace: "Trace",
    call: tuple[tuple[Any, ...], dict[str, Any], tuple[Any, ...], dict[str, Any]],
    output_entry: Any,
    fields_dict: dict[str, Any],
    parent_param_ops: dict[str, int],
) -> None:
    """Log the call's one bare tensor output, as the generic per-output loop logs it.

    ``call`` is ``(args, kwargs, arg_copies, kwarg_copies)``; the dispatch has
    already checked the output is loggable.
    """

    args, kwargs, arg_copies, kwarg_copies = call
    out_tensor = output_entry.value
    event_module_stack = _module_frames_from_fields(fields_dict)
    fields_dict["container_path"] = output_entry.container_path
    fields_dict["container_spec"] = None
    _log_output_tensor_info(
        trace,
        out_tensor,
        0,
        args,
        kwargs,
        parent_param_ops,
        fields_dict,
        output_entry.autograd_stats,
        False,
    )
    # The generic loop's internal reads (``requires_grad``, the alias storage reads).
    with internal_scalar_read():
        _OUTPUT_REQUIRES_GRAD[:] = (weakref.ref(out_tensor), out_tensor.requires_grad)
        fields_dict["backend_semantics"] = plain_output_alias_semantics(
            args, kwargs, out_tensor, fields_dict
        )
    fire_results = _pop_tensor_live_fire_results(out_tensor)
    if fire_results:
        fields_dict["fire_results"] = fire_results
        fields_dict["is_edited"] = any(result.replaced for result in fire_results)
    if output_entry.autograd_band is not None:
        trace.__dict__.setdefault("_autograd_saved_bands", {})[fields_dict["_label_raw"]] = (
            output_entry.autograd_band
        )
    entry = _make_layer_log_entry(
        trace,
        out_tensor,
        fields_dict=fields_dict,
        t_args=arg_copies,
        t_kwargs=kwarg_copies,
        activation_transform=trace._options_activation_transform,
        event_module_stack=event_module_stack,
        call_ref_box=[],
    )
    new_record = object.__getattribute__(entry, "_record")
    if fields_dict["is_internal_source"]:  # ``_classify_new_tensor_in_trace``
        trace.internal_source_ops.append(new_record.label_raw)
    _tag_tensor_and_track_variations(
        trace, out_tensor, new_record, fields_dict, arg_copies, kwarg_copies, args, kwargs
    )


def _project_call_device_memory(
    self: "Trace", func_call_id: int, fields_dict: dict[str, Any]
) -> None:
    """Re-key the call's device-memory sample to the op's raw label and project it.

    The call-level sample (observe item 15) moves to the committed raw label and
    projects onto the legacy flat fields: ``bytes_delta_at_call`` is the
    single-device allocated delta; ``bytes_peak_at_call`` is the absolute
    high-water after the call only when this call observed an advance.
    """

    samples_store = self.__dict__.get("_device_memory_samples")
    if isinstance(samples_store, dict):
        call_samples = samples_store.pop(f"call:{func_call_id}", None)
        if call_samples is not None:  # deferred import: cycle _observe
            from ..._observe._device_memory import flat_field_projection

            samples_store[fields_dict["_label_raw"]] = call_samples
            delta_value, peak_value = flat_field_projection(call_samples)
            fields_dict["bytes_delta_at_call"] = delta_value
            fields_dict["bytes_peak_at_call"] = peak_value


def _loggable_outputs(output_entries: Any, is_bottom_level_func: bool) -> tuple[int, bool | None]:
    """Return the loggable output count and, for a single output, its verdict.

    The emit loop reuses a single output's verdict (nothing labels a tensor between the
    count and the loop). With several outputs the verdict is ``None`` and the loop asks
    again per entry, since logging an earlier entry labels a duplicated tensor.
    """

    if len(output_entries) == 1:
        verdict = _output_should_be_logged(output_entries[0].value, is_bottom_level_func)
        return int(verdict), verdict
    count = sum(
        1 for entry in output_entries if _output_should_be_logged(entry.value, is_bottom_level_func)
    )
    return count, None


_NO_IDS: frozenset[int] = frozenset()
"""The ``out=`` destination ids of a call without keyword arguments."""


def _out_destination_ids(kwargs: Any) -> frozenset[int]:
    """Return the ids of the call's ``out=`` destination tensors (empty without ``out=``)."""

    out_kwarg_value = kwargs.get("out") if isinstance(kwargs, dict) else None
    if isinstance(out_kwarg_value, torch.Tensor):
        return frozenset((id(out_kwarg_value),))
    if isinstance(out_kwarg_value, (list, tuple)):
        return frozenset(id(item) for item in out_kwarg_value if isinstance(item, torch.Tensor))
    return frozenset()


def _project_foreach_member_parent_fields(
    self: "Trace",
    fields_dict_onetensor: dict[str, Any],
    output_entry: "_OutputTensorEntry",
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
) -> None:
    """Restrict a foreach member's parent fields to its zipped operands.

    ``torch._foreach_*`` applies the op element-wise across its tensor-list
    arguments: output ``i`` depends on member ``i`` of each list operand and
    on every whole-call operand (a single tensor or scalar applied to all
    members). The shared call-level relationship fields parent EVERY output on
    EVERY list member; this projection keeps, for the current member, exactly
    the parents recorded at a zipped ``(arg, i)`` position plus every parent
    at a non-zipped position, then rebuilds the derived relationship fields
    (ancestors, internal-source flags, argument positions) for that subset.
    Parents that never resolved to an argument position are kept for every
    member -- dropping an edge on uncertainty is never acceptable.

    Parameters
    ----------
    self:
        Active capture Trace.
    fields_dict_onetensor:
        Per-output copy of the shared fields, mutated in place.
    output_entry:
        Output partition entry for the current member.
    args:
        Positional arguments of the foreach call.
    kwargs:
        Keyword arguments of the foreach call.
    out_orig:
        Full original output container.
    """

    container_path = output_entry.container_path
    if not container_path:
        return
    first_step = container_path[0]
    member_index = first_step if isinstance(first_step, int) else getattr(first_step, "index", None)
    if not isinstance(member_index, int):
        return
    positions = fields_dict_onetensor.get("parent_arg_positions") or {}
    positioned_labels: set[str] = set()
    kept_positioned_labels: set[str] = set()
    for domain in ("args", "kwargs"):
        for key, label in (positions.get(domain) or {}).items():
            positioned_labels.add(label)
            zipped_member_slot = (
                isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], int)
            )
            if not zipped_member_slot or key[1] == member_index:
                kept_positioned_labels.add(label)
    original_parents = list(fields_dict_onetensor.get("parents") or ())
    kept_labels: list[str] = []
    seen: set[str] = set()
    for label in original_parents:
        if label in seen:
            continue
        if label in positioned_labels and label not in kept_positioned_labels:
            continue
        seen.add(label)
        kept_labels.append(label)
    if kept_labels == original_parents:
        return
    require_event = self.capture_events.live_index.require_event
    kept_entries = [require_event(label) for label in kept_labels]
    _build_graph_relationship_fields(
        self, fields_dict_onetensor, kept_labels, kept_entries, args, kwargs, out_orig
    )


def _get_parent_contents(
    parent_label: str,
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    parent_arg_positions: dict[str, dict[Any, str]],
) -> Any:
    """Retrieve a parent tensor's pre-call value from the saved argument copies.

    Used for child tensor variation tracking: if a parent's value in arg_copies
    differs from its currently saved out, the parent was mutated
    in-place between operations, and the variation is recorded.
    """
    for pos, label in parent_arg_positions["args"].items():
        if label == parent_label:
            return index_nested(arg_copies, pos)
    for argname, label in parent_arg_positions["kwargs"].items():
        if label == parent_label:
            return index_nested(kwarg_copies, argname)
    raise ValueError("Parent layer not found in function arguments.")


def _nested_call_logged_output(out: Any) -> bool:
    """Return whether a higher-level call's output is one tensor a nested call already logged.

    ``nn.ReLU.forward`` calls ``F.relu``, which calls ``torch.relu``: the inner,
    bottom-level call logs the op and labels its output, so the outer call
    would log nothing (:func:`_output_should_be_logged` is False for its only
    output, and a bare tensor registers no output container). Its emission
    exits before the shared fields are built.
    """

    return type(out) is torch.Tensor and get_tensor_label(out) is not None


def _output_should_be_logged(out: Any, is_bottom_level_func: bool) -> bool:
    """Determine whether an output value should be logged as a new graph node.

    Two conditions must hold:
      1. ``out`` must be a torch.Tensor INSTANCE — including user Tensor
         SUBCLASSES (tv_tensors, ``__torch_function__`` wrappers, ...), whose
         ops dispatch through the same wrapped functions and are just as real.
         The exact-type spelling ``type(out) is not torch.Tensor`` silently
         dropped every op in a subclass region (missing ops that still
         validated True). ``nn.Parameter`` stays excluded: parameters
         are SOURCE tensors, never op outputs.
      2. Either the tensor is genuinely new (no ``_tl.label_raw`` value),
         OR this is a bottom-level function.  Bottom-level functions are leaf
         operations in the decoration nesting — even if they return an already-
         labeled tensor (in-place ops), we log them to capture the operation.
         Non-bottom-level functions returning an already-labeled tensor are
         higher-level wrappers whose sub-operations were already logged.

    Returns:
        True if the output should be logged, False otherwise.
    """
    if not isinstance(out, torch.Tensor) or isinstance(out, torch.nn.Parameter):
        return False

    return bool(get_tensor_label(out) is None or is_bottom_level_func)


def _check_if_tensor_arg(arg: Any) -> bool:
    """Helper function to check if an argument either is a tensor or is a list/tuple containing a tensor.

    Args:
        arg: argument

    Returns:
        True if arg is or contains a tensor, false otherwise
    """
    if issubclass(type(arg), torch.Tensor):
        return True
    elif type(arg) in [list, tuple]:
        return any(issubclass(type(elt), torch.Tensor) for elt in arg)
    elif type(arg) is dict:
        return any(issubclass(type(val), torch.Tensor) for val in arg.values())
    else:
        return False


def _iter_autograd_saved_candidates(grad_fn_handle: Any) -> list[Any]:
    """Return accessible autograd-saved values from a grad_fn_handle object.

    Parameters
    ----------
    grad_fn_handle
        PyTorch autograd function object to inspect.

    Returns
    -------
    list
        Values exposed through ``saved_tensors`` and ``_saved_*`` attributes.
        Attribute access failures are ignored because PyTorch may release or
        guard some saved values.

    Notes
    -----
    When default saved-tensors hooks are installed (a non-reentrant
    ``torch.utils.checkpoint`` region, a user offload context), the grad_fn's
    saved values are HOOK-PACKED: reading them runs the user's unpack hook --
    for checkpoint, a full RECOMPUTE of the checkpointed region inside the
    traced forward, which both records phantom ops in the captured graph and
    overcounts memory the checkpoint deliberately does not retain. Packed
    values are therefore skipped. When the runtime cannot answer the
    hooks-installed question, the reads run under ``pause_logging`` so a
    triggered recompute can never corrupt the captured graph.
    """
    hooks_active = saved_tensors_default_hooks_active()
    if hooks_active:
        return []

    saved_values: list[Any] = []

    def _read_saved_values() -> None:
        """Append the grad_fn's saved tensors and ``_saved_*`` attributes to ``saved_values``.

        Every read is individually best-effort: an attribute the autograd node
        refuses to produce is skipped, so a partial list is a normal outcome.
        The caller decides whether this runs under ``pause_logging``.
        """

        try:
            saved_values.extend(getattr(grad_fn_handle, "saved_tensors", ()))
        except Exception:
            pass

        for attr_name in grad_fn_handle.__class__.__dict__:
            if not attr_name.startswith(_AUTOGRAD_SAVED_ATTR_PREFIX):
                continue
            try:
                saved_values.append(getattr(grad_fn_handle, attr_name))
            except Exception:
                continue

    if hooks_active is None:
        from ... import _state

        with _state.pause_logging():
            _read_saved_values()
    else:
        _read_saved_values()
    return saved_values


def _collect_tensor_values(value: Any) -> list[torch.Tensor]:
    """Collect tensor values from a shallow autograd-saved object.

    Parameters
    ----------
    value
        Value read from a grad_fn_handle saved-tensor API.

    Returns
    -------
    list of torch.Tensor
        Tensor instances found in the value.
    """
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, torch.Tensor)]
    return []


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._ops_activations import (  # noqa: E402
    _log_output_tensor_info,
    _save_activation_fields,
)
from ._ops_arguments import (  # noqa: E402
    _build_edge_use_records,
)
from ._ops_autograd import (  # noqa: E402
    _output_entries_with_autograd_stats,
    _register_call_output_container_snapshot,
)
from ._ops_capture_records import (  # noqa: E402
    _should_keep_alias_mutation_contract,
)
from ._ops_finalize import (  # noqa: E402
    _make_layer_log_entry,
    _seal_parent_edge_truth,
)
from ._ops_interventions import (  # noqa: E402
    _pop_tensor_live_fire_results,
)
from ._ops_predicate_events import (  # noqa: E402
    _build_graph_relationship_fields,
)
from ._ops_retention import (  # noqa: E402
    _build_trace_predicate_context,
)
from ._ops_shared_fields import (  # noqa: E402
    _build_shared_fields_dict,
    _classify_new_tensor_in_trace,
    _tag_tensor_and_track_variations,
)
from .ops import (  # noqa: E402
    _AUTOGRAD_SAVED_ATTR_PREFIX,
    _OutputTensorEntry,
)
