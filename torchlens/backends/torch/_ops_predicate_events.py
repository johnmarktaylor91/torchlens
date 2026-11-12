"""Predicate event records and graph relationship fields."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state as _st
from ..._capture._lazy_contexts import (
    LazyContextHistory,
    lazy_contexts_eligible,
    tensor_fields,
)
from ..._capture.arg_positions import (
    DYNAMIC_SPEC_UNCACHEABLE,
    FUNC_ARG_SPECS,
    VARIADIC_TENSOR_ARG_FUNCS,
    ArgSpec,
    _cache_dynamic_spec,
    _normalize_func_name,
    dynamic_spec_covers_call,
    extract_tensors_and_params,
)
from ..._capture.plan import EnrichmentLevel
from ..._capture.predicates import (
    _evaluate_keep_op,
    _is_halt_only_capture,
    _module_capture_spec,
    _plain_module_union,
    build_op_record_context,
)
from ..._capture.projections import (
    _build_record_context,
    active_recording_state_or_none,
    append_projected_event,
    get_active_recording_state,
)
from ..._capture.session import capture_session_for
from ..._capture.stop import evaluate_halt_stop
from ..._capture_state_helpers import _is_uninitialized_param
from ..._data_classes.internal_types import FuncExecutionContext
from ..._errors import UserTransformError
from ..._ir.events import (
    FunctionCallRef,
)
from ..._ir.predicate import RecordContext, RetroactiveCaptureDecision
from ..._ir.refs import ParamRef
from ..._ir.selector_eval import module_union_matches
from ..._training_validation import TrainingModeConfigError
from ..._utils._call_site import call_site_class, call_site_scope
from ..._utils._tensor_label import (
    get_label_list,
    get_live_tensor_label,
    get_param_meta,
    promote_mutated_parameters,
    set_tensor_label,
)
from ..._utils.display import _timed_phase
from ..._utils.introspection import (
    _get_tensors_and_params_from_obj,
)
from ...recording._exceptions import PredicateError
from ...recording._halt import StopSignal
from ...recording._types import (
    CaptureSpec,
)
from ._ops_arg_slots import current_arg_slots, slot_positions_by_label
from ._relations_resolve import defer_ancestry
from .tensor_tracking import (
    _locate_parent_tensors_in_args,
    _process_parent_param_ops,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = (
    "_emit_predicate_operation_events",
    "_build_graph_relationship_fields",
    "_extract_arg_tensors_and_params",
    "_build_param_fields",
    "_emit_lazy_predicate_operation_events",
    "install_lazy_record_contexts",
)


def _demanded_level(spec: CaptureSpec) -> EnrichmentLevel:
    """Map a capture decision to the enrichment level it demands."""

    if spec.save_out:
        return EnrichmentLevel.PAYLOAD
    if spec.save_metadata:
        return EnrichmentLevel.METADATA
    return EnrichmentLevel.SHELL


def _emit_predicate_operation_events(
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
    """Predicate-mode logging for decorated torch function outputs."""

    del exec_ctx
    state = get_active_recording_state()
    layer_type = _normalize_func_name(func_name)
    arg_tensors, arg_parameters = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    parent_labels = tuple(get_label_list(arg_tensors))
    out_iter = list(_iter_loggable_live_outputs(out_orig, is_bottom_level_func))
    expected_output_count = len(out_iter)
    identity = PredicateOpIdentity.for_call(
        self,
        state,
        args,
        kwargs,
        arg_parameters,
        func_name,
        layer_type,
        expected_output_count,
        _is_inplace_augmented_assignment_dunder,
    )
    function_ref: FunctionCallRef | None = None

    for output_index, (out, container_path, _container_spec) in enumerate(out_iter):
        self._raw_graph_ws.layer_counter += 1
        self._raw_graph_ws.raw_layer_type_counter[layer_type] += 1
        state.op_counts[layer_type] = state.op_counts.get(layer_type, 0) + 1
        state.step_index += 1
        state.event_index += 1
        raw_index = self._raw_graph_ws.layer_counter
        type_index = self._raw_graph_ws.raw_layer_type_counter[layer_type]
        _label_raw = f"{layer_type}_{type_index}_{raw_index}_raw"
        set_tensor_label(out, _label_raw)
        module_frame = state.module_stack[-1] if state.module_stack else None
        identity.register_call_site(_label_raw, output_index)
        ctx = _predicate_output_context(
            self,
            state,
            out,
            label_raw=_label_raw,
            raw_index=raw_index,
            type_index=type_index,
            layer_type=layer_type,
            func=func,
            func_name=func_name,
            parent_labels=parent_labels,
            container_path=container_path,
            output_index=output_index,
            is_bottom_level_func=is_bottom_level_func,
            module_frame=module_frame,
        )
        selected = True  # a halt or a predicate error before the decision keeps the context
        try:
            halt_only = _is_halt_only_capture(state.options)
            if halt_only:
                evaluate_halt_stop(self, ctx, state.options, frontier_output=out)
                continue
            if out.grad_fn is not None:
                state.grad_fn_to_context[out.grad_fn] = ctx
            if function_ref is None:
                function_ref = _predicate_function_ref(func, func_name, args, kwargs, func_call_id)
            # One straight-line commit per observed op: select -> demanded
            # enrichment -> payload disposition -> atomic append -> halt.
            spec = _evaluate_keep_op(ctx, state.options)
            if isinstance(spec, RetroactiveCaptureDecision):
                raise PredicateError(
                    "tl.followed_by(...) retroactive save is only supported by trace. "
                    "Remedy: use tl.trace(save=...) for followed_by retroactive capture.",
                    code="followed_by_unsupported",
                )
            demanded = _demanded_level(spec)
            selected = demanded is not EnrichmentLevel.SHELL
            bulk_default_ram = bool(
                demanded is EnrichmentLevel.PAYLOAD
                and capture_session_for(self) is not None
                and _is_default_ram_payload(state, spec)
            )
            if demanded is not EnrichmentLevel.SHELL:
                backend_semantics = _predicate_backend_semantics(
                    self,
                    out,
                    func,
                    func_name,
                    args,
                    kwargs,
                    out_orig,
                    arg_copies,
                    kwarg_copies,
                    is_bottom_level_func,
                    func_call_id,
                    expected_output_count,
                    bulk_default_ram=bulk_default_ram,
                )
            else:
                backend_semantics = None
            if demanded is EnrichmentLevel.PAYLOAD:
                ram_payload, transformed_ram_payload = _record_predicate_output(ctx, out, spec)
            else:
                ram_payload = transformed_ram_payload = None
            if not (state.options.lean and demanded is EnrichmentLevel.SHELL):  # lean: no shell
                append_projected_event(
                    self,
                    ctx,
                    spec,
                    tensor=out,
                    ram_payload=ram_payload,
                    transformed_ram_payload=transformed_ram_payload,
                    predicate_matched=spec.save_out or spec.save_metadata,
                    backend_semantics=backend_semantics,
                    function=function_ref,
                    container_path=container_path,
                    equivalence_class=identity.equivalence_class(ctx, output_index),
                    params=identity.param_refs,
                )
            evaluate_halt_stop(self, ctx, state.options, frontier_output=out)
        except StopSignal:
            raise
        except (UserTransformError, TrainingModeConfigError):
            # Postfunc + train-mode validation errors are storage failures and
            # must surface directly to the caller, not be aggregated through
            # the predicate-failure pipeline. The orchestrator's outer
            # exception handler aborts disk storage before the raise reaches
            # the caller.
            raise
        except Warning:
            # A TorchLens warning escalated to an error by the caller's filter
            # (e.g. warnings-as-errors on the reference-save-mode caveat) is not a
            # user-predicate failure. Surface it directly with its true type instead
            # of swallowing it into the predicate-failure pipeline, where it would
            # resurface as a misleading PredicateError.
            raise
        except Exception as exc:
            state.handle_predicate_exception(ctx, exc)
        finally:
            if not halt_only:
                capture_events = getattr(self, "capture_events", None)
                idx = None if capture_events is None else capture_events.op_event_by_label_raw
                if not state.options.lean and (idx is None or _label_raw not in idx):
                    append_projected_event(
                        self,
                        ctx,
                        CaptureSpec(save_out=False, save_metadata=False),
                        tensor=out,
                        predicate_matched=False,
                    )
                # Echo narrator slot: exactly once per committed event, after the
                # append, outside the predicate-failure pipeline (a narrator failure
                # warns-and-disables inside the session; only the typed stats-budget refusal
                # propagates). Duck-typed session read: the hot path imports nothing.
                echo_session = self.__dict__.get("_echo_session")
                if echo_session is not None:
                    committed_event = (
                        capture_events.op_event_by_label_raw.get(_label_raw)
                        if capture_events is not None
                        else None
                    )
                    echo_session.emit_op(
                        ctx,
                        tensor=out,
                        trace=self,
                        intervened=bool(getattr(committed_event, "is_edited", False)),
                    )
                state.append_context(ctx, selected=selected)


def _predicate_output_context(
    trace: "Trace",
    state: Any,
    out: Any,
    *,
    label_raw: str,
    raw_index: int,
    type_index: int,
    layer_type: str,
    func: Callable[..., Any],
    func_name: str,
    parent_labels: tuple[str, ...],
    container_path: tuple[Any, ...],
    output_index: int,
    is_bottom_level_func: bool,
    module_frame: Any,
) -> RecordContext:
    """Build the predicate record context of one logged output."""

    with _timed_phase(trace, "ctx_build_record_context"):
        return build_op_record_context(
            kind="op",
            label=label_raw,
            raw_label=label_raw,
            raw_index=raw_index,
            layer_type=layer_type,
            type_index=type_index,
            func_name=func_name,
            parent_labels=parent_labels,
            tensor=out,
            output_index=_live_output_index(container_path) or output_index,
            is_bottom_level_func=is_bottom_level_func,
            module_stack=state.module_stack,
            history=tuple(state.history),
            op_counts=state.op_counts,
            pass_index=state.pass_index,
            event_index=state.event_index,
            step_index=state.step_index,
            capture_start_time=trace._provenance_start_time,
            include_source_events=state.options.include_source_events,
            sample_id=state.sample_id,
            address=module_frame.address if module_frame else None,
            module_type=module_frame.module_type if module_frame else None,
            module_pass_number=module_frame.pass_number if module_frame else None,
            is_transform=bool(getattr(func, "__tl_is_transform_boundary__", False))
            or func_name in TRANSFORM_FUNC_NAMES,
            transform_kind=getattr(func, "__tl_transform_kind__", None)
            or (func_name if func_name in TRANSFORM_FUNC_NAMES else None),
        )


def _build_graph_relationship_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    parent_layer_labels: list[str],
    parent_layer_entries: list[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
) -> None:
    """Populate graph structure fields from the parents' indexed journal records (no views)."""
    # ``out=`` destinations are pre-allocated tensors the op writes into: their
    # producers (``empty``/``empty_like``/...) are genuine parents of this op.
    # Tuple/list destinations (``torch.sort(x, out=(v, i))``, topk, kthvalue,
    # cummax, ...) carry the SAME contract per element as the single-tensor
    # spelling; handling only ``isinstance(out_kwarg, torch.Tensor)`` dropped
    # every tuple-destination edge AND let the pre-allocated producer op get
    # orphan-pruned — an executed op vanished silently.
    out_kwarg = kwargs.get("out")
    if isinstance(out_kwarg, torch.Tensor):
        out_destinations: tuple[torch.Tensor, ...] = (out_kwarg,)
    elif isinstance(out_kwarg, (list, tuple)):
        out_destinations = tuple(item for item in out_kwarg if isinstance(item, torch.Tensor))
    else:
        out_destinations = ()
    for out_destination in out_destinations:
        out_kwarg_label = get_live_tensor_label(
            out_destination, self.capture_events.live_index.by_raw_label
        )
        if out_kwarg_label is not None and out_kwarg_label not in parent_layer_labels:
            parent_layer_labels = [*parent_layer_labels, out_kwarg_label]
            parent_layer_entries = [
                *parent_layer_entries,
                self.capture_events.live_index.require_event(out_kwarg_label),
            ]
    parent_arg_positions = _parent_arg_positions(self, parent_layer_entries, args, kwargs)

    # Only the keys the freeze (``_op_record_from_log``) reads are written. The
    # graph placeholders postprocess sets (children, output and distance flags,
    # buffer and conditional fields, ``internal_source_parents``) come from the
    # records and the op defaults, never from this dict.
    fields_dict["parents"] = parent_layer_labels
    fields_dict["parent_arg_positions"] = parent_arg_positions
    fields_dict["_edge_uses"] = []
    fields_dict["is_input"] = False
    fields_dict["input_was_parameter"] = False
    fields_dict["is_output_parent"] = False
    fields_dict["is_buffer"] = False
    fields_dict["is_internal_source"] = len(parent_layer_labels) == 0
    if parent_layer_entries:
        # The sets are the union of these parents' and resolve on first read (FR).
        defer_ancestry(fields_dict, parent_layer_entries)
    else:
        fields_dict["root_ancestors"] = set()
        fields_dict["input_ancestors"] = set()
        fields_dict["has_internal_source_ancestor"] = False
        fields_dict["internal_source_ancestors"] = set()

    in_multi_output = issubclass(type(out_orig), (list, tuple, dict, set))
    fields_dict["in_multi_output"] = in_multi_output


def _parent_arg_positions(
    self: "Trace", parent_records: list[Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, dict[Any, str]]:
    """Return the parents' argument positions.

    ``parent_records`` are the parents' indexed journal records. When the op's argument
    slots were snapshotted, the positions come from the slots, the same values
    ``_locate_parent_tensors_in_args`` returns; otherwise that derivation runs.
    """

    slots = current_arg_slots(self, args, kwargs) if parent_records else None
    if slots is None:
        return _locate_parent_tensors_in_args(self, parent_records, args, kwargs)
    return slot_positions_by_label(slots, [record.label_raw for record in parent_records])


def _extract_through_dynamic_spec(
    normalized_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[list[torch.Tensor], list[torch.nn.Parameter]] | None:
    """Extract one entry per slot through the Tier-2 cached spec; ``None`` if it misses the call."""
    cached = _st._dynamic_arg_specs.get(normalized_name)
    if isinstance(cached, ArgSpec) and dynamic_spec_covers_call(cached, args, kwargs):
        return promote_mutated_parameters(*extract_tensors_and_params(cached, args, kwargs))
    return None


def _extract_arg_tensors_and_params(
    normalized_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[list[torch.Tensor], list[torch.nn.Parameter]]:
    """O(1) tensor/param extraction via lookup table, with BFS fallback.

    Variadic transform boundary ops (``vmap``/``grad``/``autograd.functional.*``)
    always take the fresh Tier-3 crawl and never touch the name-keyed dynamic
    cache: their tensor-operand arity is call-dependent, so a first-call spec
    would drop later operands (a capture gap; see ``VARIADIC_TENSOR_ARG_FUNCS``).

    Tier-2 dynamic-cache specs are OBSERVED, not authoritative, so they are
    trusted only when they cover every shallow tensor in the live call
    (``dynamic_spec_covers_call``); otherwise the call re-crawls and the cache
    union-merges, so a scalar-RHS first observation can never freeze away the
    tensor-RHS parent of a later call. Names whose crawled tensors
    exceed ArgSpec's representable shapes are marked ``DYNAMIC_SPEC_UNCACHEABLE``
    and re-crawl every call.

    Tier-1 STATIC specs get the same runtime coverage guard (round-31 H2):
    PyTorch accepts scalar tensors in many schema-level ``int``/``int[]``/
    ``Scalar`` control slots (``roll(x, shifts=t)``, ``softmax(x, dim=t)``,
    ``arange(t)``, factory size dims), and a runtime tensor's VALUE there is a
    real data dependency the static spec deliberately does not enumerate. When
    the live call carries a shallow tensor at any slot the static spec does not
    extract, fall through to the BFS crawl so that operand becomes a parent
    instead of a silently dropped edge. Static-spec names never populate the
    Tier-2 cache: the static entry stays authoritative for covered calls.
    """
    is_variadic_transform = normalized_name in VARIADIC_TENSOR_ARG_FUNCS
    spec = None
    if not is_variadic_transform:
        spec = FUNC_ARG_SPECS.get(normalized_name)
        if spec is not None and dynamic_spec_covers_call(spec, args, kwargs):
            return promote_mutated_parameters(*extract_tensors_and_params(spec, args, kwargs))
        if (
            spec is None
            and (cached := _extract_through_dynamic_spec(normalized_name, args, kwargs)) is not None
        ):
            return cached

    # Tier 3 fallback: BFS crawl. Cache/union-merge the derived spec only for
    # fixed-arity functions with no static entry; variadic transform ops,
    # static-spec fall-throughs, and uncacheable names must re-crawl every call.
    all_args = list(args) + list(kwargs.values())
    arg_tensors, arg_parameters = _get_tensors_and_params_from_obj(all_args)
    if (
        not is_variadic_transform
        and spec is None
        and _st._dynamic_arg_specs.get(normalized_name) is not DYNAMIC_SPEC_UNCACHEABLE
    ):
        _cache_dynamic_spec(normalized_name, args, kwargs, arg_tensors, arg_parameters)
        # The crawl dedups by id; extract per slot as later calls do, so a retrace matches.
        if (cached := _extract_through_dynamic_spec(normalized_name, args, kwargs)) is not None:
            return cached
    # A Parameter mutated in place earlier in this pass binds as a graph parent of
    # its mutation op (``promote_mutated_parameters``); the cached spec above keeps
    # recording slot positions only, so it is unaffected.
    return promote_mutated_parameters(arg_tensors, arg_parameters)


def _build_param_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    arg_parameters: list[torch.nn.Parameter],
) -> dict[str, int]:
    """Populate parameter-involvement fields. Returns parent_param_ops dict.

    Session-leak defense: a parameter resolves by address ONLY when the address maps in
    THIS session AND the recorded log's live object IS the value (exact identity, never
    ``==``). Without the check, a foreign parameter carrying a stale leaked stamp whose
    address collided with one of THIS model's own parameters resolved into this trace's
    ``param_logs`` with no object check, so the save bound the call's tensor slot to the
    model's own (different-valued) state and replay staged the WRONG values as VERIFIED.
    Unresolved parameters fall through unprovenanced (the honest path: break marker via
    ``_tensor_has_known_provenance``).
    """
    resolved_parameters: list[torch.nn.Parameter] = []
    validated = _validated_params(self) if arg_parameters else None
    for param in arg_parameters:
        if _is_uninitialized_param(param):
            # A still-uninitialized lazy parameter reachable at op time can
            # only be initialization machinery (e.g. a sibling param during a
            # lazy module's materialize sweep); it has no shape/storage yet
            # and cannot be a real compute input.
            continue
        if validated is not None and validated.get(id(param)) is param:
            resolved_parameters.append(param)  # the checks below, made once per capture
            continue
        param_meta = get_param_meta(param)
        addr = None if param_meta is None else param_meta.param_address
        if addr is None:
            continue
        # An exact address key is what the accessor's lookup returns first; anything
        # else (short names, pass notation) keeps the accessor's own resolution.
        param_log = self._param_logs._dict.get(addr)
        if param_log is None and addr in self._param_logs:
            param_log = self._param_logs[addr]
        if param_log is not None:
            if getattr(param_log, "_param_ref", None) is not param:
                # Offload rebind rung: accelerate offload hooks
                # replace the prep-stamped meta parameter with a fresh
                # real-valued object each forward; the offload shim registers
                # exactly that materialization in the session-scoped
                # weak-valued registry. Exact identity against it carries the
                # same per-object belt as the ``_param_ref`` check; anything
                # else still falls through unprovenanced (the honest path).
                rebinds = getattr(self, "_offload_param_rebinds", None)
                if rebinds is None or rebinds.get(addr) is not param:
                    continue
            resolved_parameters.append(param)

    # Runs for its side effect too: assigns each parameter's barcode on first sight
    # and advances its per-parameter call index. The parameter totals, memory,
    # barcode list and parent-param map are not stored here: materialize
    # recomputes them from the record.
    parent_param_ops = _process_parent_param_ops(resolved_parameters)

    fields_dict["parent_params"] = resolved_parameters
    fields_dict["param_shapes"] = [tuple(param.shape) for param in resolved_parameters]
    return parent_param_ops


def _validated_params(trace: "Trace") -> dict[int, Any] | None:
    """Return the capture's validated-parameter memo, or ``None`` outside a capture.

    A parameter in it was validated by ``_session_validated_parameter``: its prep
    address maps, by exact key, to a log of this session whose ``_param_ref`` is that
    very object, which is what ``_build_param_fields`` checks before it resolves one.
    """

    try:
        return trace._raw_graph_ws._validated_params
    except AttributeError:
        return None


@dataclass(frozen=True, slots=True)
class PredicateOpIdentity:
    """What one torch call contributes to its outputs' grouping identity.

    Exhaustive capture gives every op a parameter identity, a call-site class
    and an equivalence class at capture time; predicate recording mints the
    same three here, from the same builders, so ``Recording.to_trace()``
    resolves parameters and groups recurrent layers exactly as ``tl.trace``.
    """

    state: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    func_name: str
    layer_type: str
    output_count: int
    param_refs: tuple[ParamRef, ...]
    call_site_chain: tuple[Any, ...]
    is_inplace_dunder: Callable[[str], bool]
    #: The innermost stateful module on the stack (``call_site_scope``), as exhaustive capture keys it.
    scope: str

    @classmethod
    def for_call(
        cls,
        trace: "Trace",
        state: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        arg_parameters: list[Any],
        func_name: str,
        layer_type: str,
        output_count: int,
        is_inplace_dunder: Callable[[str], bool],
    ) -> "PredicateOpIdentity":
        """Resolve the call's prepared parameters and walk its call site once."""

        return cls(
            state,
            args,
            kwargs,
            func_name,
            layer_type,
            output_count,
            _prepared_param_refs(state, arg_parameters),
            _predicate_call_site_chain(trace),
            is_inplace_dunder,
            call_site_scope(
                (frame.address for frame in state.module_stack),
                trace._module_capture_ws.stateful_module_addresses,
            ),
        )

    def register_call_site(self, label_raw: str, output_index: int) -> None:
        """Record one output's call-site class when the walk saw the call site."""

        if not self.call_site_chain:
            return
        slot = output_index if self.output_count > 1 else None
        key = (self.scope, self.call_site_chain, self.func_name, slot)
        self.state.call_site_classes[label_raw] = call_site_class(
            self.state.call_site_index, key, self.layer_type, slot
        )

    def equivalence_class(self, ctx: RecordContext, output_index: int) -> str:
        """The loop-grouping fingerprint of one output."""

        return _predicate_equivalence_class(
            self.state,
            ctx,
            self.args,
            self.kwargs,
            self.param_refs,
            func_name=self.func_name,
            layer_type=self.layer_type,
            output_index=output_index,
            in_multi_output=self.output_count > 1,
            is_inplace_dunder=self.is_inplace_dunder,
        )


def install_lazy_record_contexts(trace: "Trace") -> None:
    """Defer this pass's record contexts when its lean module read allows it.

    Runs once per pass where the producer policy is set. An eligible pass (see
    ``lazy_contexts_eligible``; the save selector must also be a plain
    ``tl.module`` union) gets a ``LazyContextHistory`` and the deferring op
    producer; any other pass is left exactly as it was.

    Parameters
    ----------
    trace
        Predicate-mode trace about to run its forward pass.
    """

    state = active_recording_state_or_none()
    if state is None or state.event_index != 0:  # only at a pass start, before any event
        return
    options = state.options
    addresses = _plain_module_union(options.keep_op)
    if addresses is None or not lazy_contexts_eligible(
        options, _module_capture_spec(options), state.no_tensor_capture
    ):
        return
    state.history = LazyContextHistory(
        state,
        build=_build_record_context,
        window=options.history_size,
        module_addresses=addresses,
        matches=module_union_matches,
    )
    trace._capture_producer_policy = CaptureProducerPolicy(
        "predicate", _emit_lazy_predicate_operation_events
    )


def _emit_lazy_predicate_operation_events(
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
    """Label a lean module read's op outputs and defer their record contexts.

    Matches ``_emit_predicate_operation_events`` on a pass that
    ``install_lazy_record_contexts`` admitted: the plain ``tl.module`` union
    answers False for every op context (none has a module-output candidate),
    halt is off and a lean journal appends no shell event for an unkept op, so
    the eager emitter's only lasting effects are the counters, the tensor
    labels and the appended context, which this one defers with the same
    inputs.
    """

    del arg_copies, kwarg_copies, exec_ctx, func_call_id
    state = get_active_recording_state()
    layer_type = _normalize_func_name(func_name)
    arg_tensors, _ = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    call_fields = {
        "type": layer_type,
        "func_name": func_name,
        "parent_labels": tuple(get_label_list(arg_tensors)),
        "is_bottom_level_func": is_bottom_level_func,
        "is_transform": bool(getattr(func, "__tl_is_transform_boundary__", False))
        or func_name in TRANSFORM_FUNC_NAMES,
        "transform_kind": getattr(func, "__tl_transform_kind__", None)
        or (func_name if func_name in TRANSFORM_FUNC_NAMES else None),
    }
    out_iter = list(_iter_loggable_live_outputs(out_orig, is_bottom_level_func))
    for position, (out, container_path, _container_spec) in enumerate(out_iter):
        output_index = _live_output_index(container_path) or position
        _defer_lazy_op_output(self, state, out, output_index, call_fields)


def _defer_lazy_op_output(
    trace: "Trace", state: Any, out: Any, output_index: int, call_fields: dict[str, Any]
) -> None:
    """Count, label and defer one op output (the per-output body of the lazy emitter)."""

    graph_ws = trace._raw_graph_ws
    layer_type = call_fields["type"]
    graph_ws.layer_counter += 1
    graph_ws.raw_layer_type_counter[layer_type] += 1
    state.op_counts[layer_type] = state.op_counts.get(layer_type, 0) + 1
    state.step_index += 1
    state.event_index += 1
    raw_index = graph_ws.layer_counter
    type_index = graph_ws.raw_layer_type_counter[layer_type]
    label_raw = f"{layer_type}_{type_index}_{raw_index}_raw"
    set_tensor_label(out, label_raw)
    frame = state.module_stack[-1] if state.module_stack else None
    data = {
        **call_fields,
        "label": label_raw,
        "raw_label": label_raw,
        "_label_raw": label_raw,
        "raw_index": raw_index,
        "type_index": type_index,
        "output_index": output_index,
        "address": frame.address if frame else None,
        "module_type": frame.module_type if frame else None,
        "module_pass_number": frame.pass_number if frame else None,
        **tensor_fields(out),
    }
    elapsed = time.time() - trace._provenance_start_time
    history = cast(LazyContextHistory, state.history)
    history.defer("op", data, state.event_index, state.step_index, elapsed)


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._ops_call_site_fields import (  # noqa: E402
    _predicate_call_site_chain,
)
from ._ops_capture_records import (  # noqa: E402
    _is_inplace_augmented_assignment_dunder,
)
from ._ops_interventions import (  # noqa: E402
    _iter_loggable_live_outputs,
    _live_output_index,
)
from ._ops_predicates import (  # noqa: E402
    _is_default_ram_payload,
    _predicate_backend_semantics,
    _predicate_function_ref,
    _record_predicate_output,
)
from ._ops_shared_fields import (  # noqa: E402
    _predicate_equivalence_class,
    _prepared_param_refs,
)
from .ops import (  # noqa: E402
    TRANSFORM_FUNC_NAMES,
    CaptureProducerPolicy,
)
