"""Predicate event records and graph relationship fields."""

from collections import defaultdict
from collections.abc import Callable
from math import prod
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state as _st
from ..._capture_state_helpers import _is_uninitialized_param
from ..._errors import TorchLensPostfuncError
from ..._state import pause_logging
from ..._training_validation import TrainingModeConfigError
from ...capture.arg_positions import (
    DYNAMIC_SPEC_UNCACHEABLE,
    FUNC_ARG_SPECS,
    VARIADIC_TENSOR_ARG_FUNCS,
    ArgSpec,
    _cache_dynamic_spec,
    _normalize_func_name,
    dynamic_spec_covers_call,
    extract_tensors_and_params,
)
from ...capture.plan import EnrichmentLevel
from ...capture.predicates import (
    _evaluate_keep_op,
    _is_halt_only_capture,
    build_op_record_context,
)
from ...capture.projections import (
    LiveOpView,
    append_projected_event,
    get_active_recording_state,
)
from ...capture.session import capture_session_for
from ...capture.stop import evaluate_halt_stop
from ...data_classes.internal_types import FuncExecutionContext
from ...data_classes.op import (
    Op,
)
from ...fastlog._halt import HaltSignal
from ...fastlog.exceptions import PredicateError
from ...fastlog.types import (
    CaptureSpec,
)
from ...ir.events import (
    FunctionCallRef,
)
from ...ir.predicate import RetroactiveCaptureDecision
from ...utils.display import _timed_phase
from ...utils.introspection import (
    _get_tensors_and_params_from_obj,
)
from ._tl import (
    get_label_list,
    get_live_tensor_label,
    get_param_meta,
    promote_mutated_parameters,
    set_tensor_label,
)
from .tensor_tracking import (
    _get_ancestors_from_parents,
    _locate_parent_tensors_in_args,
    _process_parent_param_ops,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

if TYPE_CHECKING:
    from .ops import (
        TRANSFORM_FUNC_NAMES,
        _is_default_ram_payload,
        _iter_loggable_live_outputs,
        _live_output_index,
        _predicate_backend_semantics,
        _predicate_function_ref,
        _record_predicate_output,
        _snapshot_exhaustive_module_stack,
    )

__all__ = (
    "_emit_predicate_operation_events",
    "_build_graph_relationship_fields",
    "_extract_arg_tensors_and_params",
    "_build_param_fields",
    "_build_module_context_fields",
)


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
    arg_tensors, _ = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    parent_labels = tuple(get_label_list(arg_tensors))
    out_iter = list(_iter_loggable_live_outputs(out_orig, is_bottom_level_func))
    expected_output_count = len(out_iter)
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
        with _timed_phase(self, "ctx_build:record_context"):
            ctx = build_op_record_context(
                kind="op",
                label=_label_raw,
                raw_label=_label_raw,
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
                capture_start_time=self.capture_start_time,
                include_source_events=state.options.include_source_events,
                sample_id=state.sample_id,
                address=module_frame.address if module_frame else None,
                module_type=module_frame.module_type if module_frame else None,
                module_pass_index=module_frame.pass_index if module_frame else None,
                is_transform=bool(getattr(func, "__tl_is_transform_boundary__", False))
                or func_name in TRANSFORM_FUNC_NAMES,
                transform_kind=getattr(func, "__tl_transform_kind__", None)
                or (func_name if func_name in TRANSFORM_FUNC_NAMES else None),
            )
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
            if spec.save_out:
                demanded = EnrichmentLevel.PAYLOAD
            elif spec.save_metadata:
                demanded = EnrichmentLevel.METADATA
            else:
                demanded = EnrichmentLevel.SHELL
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
                ram_payload = None
                transformed_ram_payload = None
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
            )
            evaluate_halt_stop(self, ctx, state.options, frontier_output=out)
        except HaltSignal:
            raise
        except (TorchLensPostfuncError, TrainingModeConfigError):
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
                if capture_events is None or _label_raw not in capture_events.op_event_by_label_raw:
                    append_projected_event(
                        self,
                        ctx,
                        CaptureSpec(save_out=False, save_metadata=False),
                        tensor=out,
                        predicate_matched=False,
                    )
                # Echo narrator slot (snoop D1): exactly once per committed
                # event, after the append, outside the predicate-failure
                # pipeline (a narrator failure warns-and-disables inside the
                # session; only the typed stats-budget refusal propagates).
                # Duck-typed session read: the hot path imports nothing.
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
                        intervened=bool(getattr(committed_event, "intervention_replaced", False)),
                    )
                state.append_context(ctx)


def _build_graph_relationship_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    parent_layer_labels: list[str],
    parent_layer_entries: list[Op],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
) -> None:
    """Populate graph structure fields: parents, children, ancestors, buffer/IO flags."""
    # ``out=`` destinations are pre-allocated tensors the op writes into: their
    # producers (``empty``/``empty_like``/...) are genuine parents of this op.
    # Tuple/list destinations (``torch.sort(x, out=(v, i))``, topk, kthvalue,
    # cummax, ...) carry the SAME contract per element as the single-tensor
    # spelling; handling only ``isinstance(out_kwarg, torch.Tensor)`` dropped
    # every tuple-destination edge AND let the pre-allocated producer op get
    # orphan-pruned — an executed op vanished silently (W3 audit F3).
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
                cast(
                    Op,
                    LiveOpView(self, self.capture_events.live_index.require_event(out_kwarg_label)),
                ),
            ]
    parent_arg_positions = _locate_parent_tensors_in_args(self, parent_layer_entries, args, kwargs)
    input_ancestors, internal_source_ancestors = _get_ancestors_from_parents(parent_layer_entries)
    internal_parent_layer_labels = [
        label
        for label in parent_layer_labels
        if self.capture_events.live_index.require_event(label).has_internal_source_ancestor
    ]

    fields_dict["parents"] = parent_layer_labels
    fields_dict["parent_arg_positions"] = parent_arg_positions
    fields_dict["_edge_uses"] = []
    fields_dict["root_ancestors"] = input_ancestors.union(internal_source_ancestors)
    fields_dict["children"] = []
    fields_dict["has_children"] = False
    fields_dict["is_input"] = False
    fields_dict["input_was_parameter"] = False
    fields_dict["has_input_ancestor"] = len(input_ancestors) > 0
    fields_dict["input_ancestors"] = input_ancestors
    fields_dict["min_distance_from_input"] = None
    fields_dict["max_distance_from_input"] = None
    fields_dict["is_output"] = False
    fields_dict["is_output_parent"] = False
    fields_dict["is_final_output"] = False
    fields_dict["has_output_descendant"] = False
    fields_dict["output_descendants"] = set()
    fields_dict["is_orphan"] = False
    fields_dict["min_distance_to_output"] = None
    fields_dict["max_distance_to_output"] = None
    fields_dict["io_role"] = None
    fields_dict["is_buffer"] = False
    fields_dict["address"] = None
    fields_dict["buffer_pass"] = None
    fields_dict["buffer_source"] = None
    fields_dict["buffer_write_kind"] = None
    fields_dict["buffer_value_changed"] = None
    fields_dict["buffer_replay_validated"] = None
    fields_dict["buffer_source_func_name"] = None
    fields_dict["is_internal_source"] = len(parent_layer_labels) == 0
    fields_dict["has_internal_source_ancestor"] = len(internal_source_ancestors) > 0
    fields_dict["internal_source_parents"] = internal_parent_layer_labels
    fields_dict["internal_source_ancestors"] = internal_source_ancestors
    fields_dict["is_internal_sink"] = False
    fields_dict["is_terminal_bool"] = False
    fields_dict["is_terminal_conditional_bool"] = False
    fields_dict["conditional_context_kind"] = None
    fields_dict["conditional_wrapper_kind"] = None
    fields_dict["terminal_conditional_id"] = None
    fields_dict["in_conditionals"] = []
    fields_dict["terminal_bool_for"] = None
    fields_dict["is_in_conditional_body"] = False
    fields_dict["conditional_branch_stack"] = []
    fields_dict["conditional_branch_depth"] = 0
    fields_dict["conditional_entry_children"] = []
    fields_dict["conditional_then_children"] = []
    fields_dict["conditional_elif_children"] = {}
    fields_dict["conditional_else_children"] = []
    fields_dict["conditional_arm_children"] = {}

    in_multi_output = any(issubclass(type(out_orig), cls) for cls in [list, tuple, dict, set])
    fields_dict["in_multi_output"] = in_multi_output


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
    tensor-RHS parent of a later call (round-22 F3). Names whose crawled tensors
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
        if spec is None:
            cached = _st._dynamic_arg_specs.get(normalized_name)
            if isinstance(cached, ArgSpec) and dynamic_spec_covers_call(cached, args, kwargs):
                return promote_mutated_parameters(*extract_tensors_and_params(cached, args, kwargs))

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

    r79 session-leak defense (r78 wrong-bind closure): a parameter resolves by
    address ONLY when the address maps in THIS session AND the recorded log's
    live object IS the value (exact identity, never ``==``). Pre-fix, a foreign
    parameter carrying a stale leaked stamp whose address collided with one of
    THIS model's own parameters resolved into this trace's ``param_logs`` with
    no object check, so the save bound the call's tensor slot to the model's own
    (different-valued) state and replay staged the WRONG values as VERIFIED.
    Unresolved parameters fall through unprovenanced (the honest path: break
    marker via ``_tensor_has_known_provenance``).
    """
    _param_logs: list[Any] = []
    resolved_parameters: list[torch.nn.Parameter] = []
    for param in arg_parameters:
        if _is_uninitialized_param(param):
            # A still-uninitialized lazy parameter reachable at op time can
            # only be initialization machinery (e.g. a sibling param during a
            # lazy module's materialize sweep); it has no shape/storage yet
            # and cannot be a real compute input.
            continue
        param_meta = get_param_meta(param)
        addr = None if param_meta is None else param_meta.param_address
        if addr is not None and addr in self.param_logs:
            param_log = self.param_logs[addr]
            if getattr(param_log, "_param_ref", None) is not param:
                # Offload rebind rung (lane F37): accelerate offload hooks
                # replace the prep-stamped meta parameter with a fresh
                # real-valued object each forward; the offload shim registers
                # exactly that materialization in the session-scoped
                # weak-valued registry. Exact identity against it carries the
                # same per-object belt as the ``_param_ref`` check; anything
                # else still falls through unprovenanced (the honest path).
                rebinds = getattr(self, "_offload_param_rebinds", None)
                if rebinds is None or rebinds.get(addr) is not param:
                    continue
            _param_logs.append(param_log)
            resolved_parameters.append(param)

    parent_param_ops = _process_parent_param_ops(resolved_parameters)
    indiv_param_barcodes = list(parent_param_ops.keys())

    fields_dict["parent_params"] = resolved_parameters
    fields_dict["_param_barcodes"] = indiv_param_barcodes
    fields_dict["parent_param_ops"] = parent_param_ops
    fields_dict["_param_logs"] = _param_logs
    fields_dict["param_shapes"] = [tuple(param.shape) for param in resolved_parameters]
    fields_dict["num_params"] = sum(prod(shape) for shape in fields_dict["param_shapes"])
    fields_dict["num_params_trainable"] = sum(
        pl.num_params for pl in _param_logs if pl.is_trainable
    )
    fields_dict["num_params_frozen"] = sum(
        pl.num_params for pl in _param_logs if not pl.is_trainable
    )
    with pause_logging():
        fields_dict["param_memory"] = sum(
            p.nelement() * p.element_size() for p in resolved_parameters
        )
    return parent_param_ops


def _build_module_context_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    arg_tensors: list[torch.Tensor],
    parent_layer_entries: list[Op],
) -> None:
    """Populate module nesting, address, and input/output status fields."""
    modules = _snapshot_exhaustive_module_stack(self)
    module = modules[-1] if modules else None

    fields_dict["module"] = module
    fields_dict["modules"] = modules
    # B3R7-R05-1: same containment fact as ``modules`` (relabeled together at
    # step 11), never the fed-call stack -- that is ``input_to_module_calls``.
    fields_dict["module_call_stack"] = list(modules)
    fields_dict["module_entry_arg_keys"] = defaultdict(list)
    fields_dict["input_to_module_calls"] = []
    fields_dict["output_of_modules"] = []
    fields_dict["output_of_module_calls"] = []
    fields_dict["is_module_output"] = False
    fields_dict["is_atomic_module"] = False
    fields_dict["atomic_module_call"] = None
