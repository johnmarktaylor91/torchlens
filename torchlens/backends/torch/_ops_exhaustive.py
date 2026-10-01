"""Exhaustive event emission and autograd candidate collection."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from ...capture.projections import LiveOpView
from ...capture.stop import evaluate_halt_stop
from ...data_classes.internal_types import FuncExecutionContext
from ...data_classes.op import (
    Op,
)
from ...ir.events import (
    FunctionCallRef,
)
from ...ir.intervention import FunctionEventInput
from ...utils._torch_compat import (
    saved_tensors_default_hooks_active,
)
from ...utils.collections import index_nested
from ...utils.display import _timed_phase
from ...utils.tensor_utils import (
    safe_copy,
)
from ._tl import (
    get_tensor_label,
    set_tensor_label,
)
from .aliasing import (
    detect_torch_alias_contract,
    detect_torch_output_alias_contract,
)
from .completeness_witness import internal_scalar_read
from .tensor_tracking import (
    _add_tensor_backward_hook,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

if TYPE_CHECKING:
    from .ops import (
        _AUTOGRAD_SAVED_ATTR_PREFIX,
        _build_edge_use_records,
        _build_graph_relationship_fields,
        _build_shared_fields_dict,
        _build_trace_predicate_context,
        _classify_new_tensor_in_trace,
        _log_output_tensor_info,
        _make_layer_log_entry,
        _module_frames_from_fields,
        _OutputTensorEntry,
        _partition_output_entries_with_autograd_stats,
        _pop_tensor_live_fire_results,
        _register_call_output_container_snapshot,
        _should_keep_alias_mutation_contract,
        _tag_tensor_and_track_variations,
    )

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
      4. Tag the output tensor with ``_tl.label_raw`` so downstream
         operations can identify it as a parent.
      5. Track parent content variations (for in-place mutation detection).

    Args:
        func: The original (unwrapped) function that was called.
        args: Positional arguments to the function.
        kwargs: Keyword arguments to the function.
        arg_copies: Pre-call copies of args (for child tensor variation tracking).
        kwarg_copies: Pre-call copies of kwargs.
        out_orig: Original output from the function (may be tensor, tuple, etc.).
        exec_ctx: Timing, RNG, and autocast state captured around the function call.
        is_bottom_level_func: True if this function was not called by another
            decorated function (i.e., it's a leaf in the decoration nesting).
    """
    with _timed_phase(self, "ctx_build:shared_fields"):
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

    output_entries = _partition_output_entries_with_autograd_stats(self, out_orig)
    _register_call_output_container_snapshot(
        self,
        out_orig,
        output_entries=output_entries,
        func_call_id=func_call_id,
        event_index=int(fields_dict.get("raw_index") or func_call_id),
    )
    expected_output_count = len(output_entries)
    loggable_output_count = sum(
        1
        for output_entry in output_entries
        if _output_should_be_logged(output_entry.value, is_bottom_level_func)
    )
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
    # consumers of the input through an op whose result the user may never use
    # (W3 audit F4). Mirror the scalar pass-through machinery: log each such
    # entry against a minted safe copy so the live input keeps its label.
    # out= destinations are excluded -- the op genuinely WROTE into them, so
    # the live destination must keep advancing to this op's label.
    out_kwarg_value = kwargs.get("out") if isinstance(kwargs, dict) else None
    if isinstance(out_kwarg_value, torch.Tensor):
        out_destination_ids: frozenset[int] = frozenset((id(out_kwarg_value),))
    elif isinstance(out_kwarg_value, (list, tuple)):
        out_destination_ids = frozenset(
            id(item) for item in out_kwarg_value if isinstance(item, torch.Tensor)
        )
    else:
        out_destination_ids = frozenset()

    # ``torch._foreach_*`` semantics are ZIPPED: output ``i`` is computed from
    # member ``i`` of each tensor-list argument (plus any whole-call scalar or
    # single-tensor operand). Sharing the call-level parent set across every
    # output invented all-to-all dependencies (round-31 M3), corrupting
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

    # One shared FunctionCallRef per wrapped call (M7): the first logged
    # output parks the frozen ref here and every sibling reuses it.
    call_ref_box: list[FunctionCallRef] = []

    for i, output_entry in enumerate(output_entries):
        out = output_entry.value
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue
        out_tensor = cast(torch.Tensor, out)
        if (
            output_entry.container_spec is not None
            and id(out_tensor) not in out_destination_ids
            and any(out_tensor is arg_tensor for arg_tensor in arg_tensors)
        ):
            # Round-31 M5: preserve the live member's real autograd node; the
            # minted copy's ``CloneBackward`` is bookkeeping, not op metadata.
            live_member_grad_fn = out_tensor.grad_fn
            out_tensor = safe_copy(out_tensor)
            if live_member_grad_fn is not None:
                try:
                    setattr(out_tensor, "tl_user_grad_fn", live_member_grad_fn)
                except AttributeError:
                    pass

        # M7: per-output isolation is a plain dict copy. Every per-output
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
        )
        if i == 0 and getattr(self, "track_device_memory", False):
            # Re-key the call-level device-memory sample (observe item 15) to
            # the committed raw label and project it onto the legacy flat
            # fields: bytes_delta_at_call = the single-device allocated
            # delta; bytes_peak_at_call = the absolute high-water after the
            # call ONLY when this call observed an advance.
            samples_store = self.__dict__.get("_device_memory_samples")
            if isinstance(samples_store, dict):
                call_samples = samples_store.pop(f"call:{func_call_id}", None)
                if call_samples is not None:
                    from ...observe._device_memory import flat_field_projection

                    samples_store[fields_dict_onetensor["_label_raw"]] = call_samples
                    delta_value, peak_value = flat_field_projection(call_samples)
                    fields_dict_onetensor["bytes_delta_at_call"] = delta_value
                    fields_dict_onetensor["bytes_peak_at_call"] = peak_value
        detect_backend_semantics = (
            detect_torch_alias_contract
            if _should_keep_alias_mutation_contract(self)
            else detect_torch_output_alias_contract
        )
        # Mutation/alias detection compares pre-call input copies against post-call
        # values (``tensor_nanequal`` -> ``torch.equal`` / ``torch.allclose``), a
        # capture-internal read that returns a Python ``bool``; mark it so the
        # completeness witness does not record it as a user host escape.
        with internal_scalar_read():
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
            fields_dict_onetensor["fire_results"] = fire_results
            fields_dict_onetensor["interventions"] = [
                result.fire_record for result in fire_results if result.fire_record is not None
            ]
            fields_dict_onetensor["intervention_replaced"] = any(
                result.replaced for result in fire_results
            )
        if getattr(self, "intervention_ready", False):
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
        new_layer_entry = cast(
            Op,
            _make_layer_log_entry(
                self,
                out_tensor,
                fields_dict=fields_dict_onetensor,
                t_args=arg_copies,
                t_kwargs=kwarg_copies,
                activation_transform=self.activation_transform,
                event_module_stack=event_module_stack,
                call_ref_box=call_ref_box,
            ),
        )
        new_tensor_label = new_layer_entry._label_raw

        _classify_new_tensor_in_trace(self, fields_dict, new_tensor_label)
        _tag_tensor_and_track_variations(
            self,
            out_tensor,
            new_layer_entry,
            fields_dict_onetensor,
            arg_copies,
            kwarg_copies,
        )
        # Round-31 H1: the member entry was logged against a minted safe copy
        # (pass-through protection), but for a list-returning in-place op the
        # LIVE member is the mutated receiver the caller keeps using. Advance
        # its label to this mutation op, hook its gradient, and propagate to
        # overlapping storage aliases -- the exact same repair the scalar
        # same-object in-place path performs -- so consumers bind to the
        # mutation instead of the stale pre-mutation producer.
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
        options = getattr(self, "_predicate_save_options", None)
        if options is not None and options.halt is not None:
            halt_ctx = _build_trace_predicate_context(self, fields_dict_onetensor, out_tensor)
            evaluate_halt_stop(self, halt_ctx, options, frontier_output=out_tensor)


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
    kept_entries = [
        cast(Op, LiveOpView(self, self.capture_events.live_index.require_event(label)))
        for label in kept_labels
    ]
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


def _output_should_be_logged(out: Any, is_bottom_level_func: bool) -> bool:
    """Determine whether an output value should be logged as a new graph node.

    Two conditions must hold:
      1. ``out`` must be a torch.Tensor INSTANCE — including user Tensor
         SUBCLASSES (tv_tensors, ``__torch_function__`` wrappers, ...), whose
         ops dispatch through the same wrapped functions and are just as real.
         The exact-type spelling ``type(out) is not torch.Tensor`` silently
         dropped every op in a subclass region (missing ops that still
         validated True — W3 F5). ``nn.Parameter`` stays excluded: parameters
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
    values are therefore skipped (r33 F-2). When the runtime cannot answer the
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
