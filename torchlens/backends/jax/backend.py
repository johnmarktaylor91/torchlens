"""Jaxpr-first JAX backend."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Final, NamedTuple, cast

from ..._capture.outcome import stamp_backend_finalized
from ..._data_classes._compaction import compact_op_metadata
from ..._data_classes.derived_grad import (
    DerivedGradAccessor,
    DerivedGradRecord,
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
)
from ..._data_classes.layer import Layer
from ..._data_classes.param import Param, ParamAccessor
from ..._data_classes.trace import Trace
from ..._deprecations import MISSING, MissingType
from ..._ir.capture_events import CaptureEvents
from ..._ir.container import ContainerSpec, DictKey, OutputPathComponent, TupleIndex
from ..._ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    OpEvent,
    OutputRef,
    ParentEdge,
)
from ..._ir.op_record import amend_preview_output_parent_rebind
from ..._ir.predicate import RecordContext
from ..._ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ..._ir.semantics import BackendSemantics, CapturePolicy
from ..._ir.workspaces import _init_module_hierarchy_data
from ..._output_slot import returned
from ..._postprocess._grouping_stamp import build_grouping_policy_stamp
from ..._postprocess._materialize import materialize_from_events
from ..._postprocess._selective_save import (
    apply_static_label_save_policy,
    pop_static_label_save_predicate,
)
from ..._postprocess.finalization import _build_module_logs
from ..._postprocess.loop_grouping_adapter import (
    RecurrenceAssignment,
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)
from ..._quantities import Bytes, Duration
from ..._trace_core.relation_views import freeze_trace_relation_views
from ..._utils._record_group_table import record_member
from ..._validation_status import (
    REGION_REPLAY_CLASS,
    REGION_REPLAY_CLASS_KEY,
    REGION_REPLAY_IMPORTER_PROVENANCE,
    REGION_REPLAY_PROVENANCE_KEY,
    ValidationReplaySource,
    ValidationStatus,
    count_importer_region_annotations,
)
from ...backends._finalize import (
    attach_function_root_module,
    mirror_param_derived_grads,
    new_preview_function_trace,
    normalize_op_module_calls,
    numel_from_shape as _numel,
    session_callable_identity as _callable_identity,
    stamp_preview_graph_shape_hash,
    value_nbytes as _nbytes,
)
from ...backends.registry import BackendName, BackendUnsupportedError, get_backend_spec
from ...options import CaptureOptions, merge_capture_options
from ...recording._types import CaptureSpec
from .._entry_refusals import refuse_if_bad_jax_control_flow, refuse_if_bad_max_intermediate_grads
from .._facet_recipes import attach_facet_recipes, pop_facet_recipes
from .._finalize import _attach_param_usage
from .._finalize_labels import apply_final_label, torch_rule_final_labels
from .._finalize_totals import _stamp_param_totals
from .._invariants import run_metadata_invariants
from .._options import (
    JAX_EXTRA_KWARG_POLICY,
    JAX_PREVIEW_TRACE_OPTION_POLICY,
    default_if_missing as _default_if_missing,
    is_missing as _is_missing,
    reject_extra_trace_kwargs,
    reject_unsupported_trace_options,
    resolve_optional_capture_field as _resolve_optional_capture_field,
)
from .._root_frames import refuse_rootless_stack, strip_op_root_frames
from .._validation_shared import float_replay_tolerances
from . import _module_return_probe as return_probe
from ._capture_scopes import forward_scopes, retain_backward_state
from ._closure_consts import _readable_closure_consts
from ._converted_grads import DerivedGradCapture, attach_jax_derived_grads
from ._edits import append_session_event, open_equation_session, pop_edit_kwargs, replay_with_edits
from ._host_state import _reject_closed_over_host_state
from ._input_kwargs import bind_input_kwargs
from ._module_frames import jax_event_module_frames
from ._perturbation_probes import _perturb_candidates
from ._pytree_param_barcodes import stamp_pytree_param_barcodes
from ._raw_relabel import (
    _is_jax_recurrence_source,
    _jax_equivalent_labels_by_key,
    _relabel_jax_lineage_fields,
)
from ._site_dialect import jax_site_keys
from ._trace_graph_helpers import (
    _control_parent_labels,
    _data_parent_arg_positions,
    _data_parent_labels,
    _jax_dict_keys,
    _ordered_jax_data_parent_labels,
    _parent_perturbations_change_output,
    _path_to_components,
    _path_to_string,
    _relabel_jax_edge_use,
    _relabel_jax_parent_arg_positions,
    _saved_op_outputs,
    _saved_parent_input,
    _saved_region_boundary_outputs,
    _values_close,
)
from ._validation_log import (
    _JaxValidationLog,
    _JaxVerdict,
    _record_insensitive_outcome,
    _record_jax_constant_leaves,
    _record_jax_decision,
)
from .jaxpr import (
    ALL_JAX_EQUATION_KINDS,
    JaxCaptureResult,
    JaxEquationCapture,
    JaxRegionCapture,
    JaxRuntimeCapture,
    derive_closed_jaxpr,
    flatten_dynamic_args,
    interpret_closed_jaxpr_with_inlining,
    jax_equivalence_key,
    reject_attributed_module_strict_control_flow,
    reject_undeclared_consts,
    replay_equation,
)
from .modules import (
    EquinoxModuleTree,
    NnxModuleTree,
    discover_equinox_module_tree,
    discover_nnx_module_tree,
    equinox_param_logs,
    nnx_param_logs,
    scoped_equinox_module_calls,
    scoped_nnx_module_calls,
)


@dataclass(frozen=True)
class GradOptions:
    """JAX derived-gradient options.

    Parameters
    ----------
    params
        Parameter pytree passed as positional argument 0 to ``fn(params, *inputs)``.
    loss_fn
        Optional callable mapping raw function output to a scalar loss. Required
        unless the raw traced output is already scalar.
    input_grad_argnums
        Input-relative positional argument indexes to differentiate in addition
        to params. ``0`` refers to the first item after params, and translates to
        JAX argnum ``1``.
    intermediate_grads
        Whether to run the opt-in zero-tap AD replay for exact op-level
        intermediate derived gradients.
    max_intermediate_grads
        Hard cap on saved op boundaries processed by the intermediate-gradient
        producer and oracle.
    """

    params: Any
    loss_fn: Callable[[Any], Any] | None = None
    input_grad_argnums: tuple[int, ...] = ()
    intermediate_grads: bool = False
    # DELIBERATELY lower than the 64 default of the other four previews: the
    # JAX intermediate producer runs a zero-tap AD replay whose cost scales
    # with the boundary cap, and control-flow unrolling multiplies JAX
    # boundary counts. Raise explicitly per call when more are needed.
    max_intermediate_grads: int = 8

    def __init__(
        self,
        *,
        params: Any,
        loss_fn: Callable[[Any], Any] | None = None,
        input_grad_argnums: Sequence[int] = (),
        intermediate_grads: bool = False,
        max_intermediate_grads: int = 8,
    ) -> None:
        """Initialize JAX derived-gradient options.

        Parameters
        ----------
        params
            Parameter pytree passed as positional argument 0.
        loss_fn
            Callable mapping raw output to scalar loss, or ``None`` for scalar
            raw outputs.
        input_grad_argnums
            Input-relative argnums to differentiate.
        intermediate_grads
            Whether to expose exact op-level intermediate derived gradients.
        max_intermediate_grads
            Hard cap on saved op boundaries for the zero-tap producer and oracle.
        """

        refuse_if_bad_max_intermediate_grads(max_intermediate_grads)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "loss_fn", loss_fn)
        object.__setattr__(self, "input_grad_argnums", tuple(input_grad_argnums))
        object.__setattr__(self, "intermediate_grads", bool(intermediate_grads))
        object.__setattr__(self, "max_intermediate_grads", max_intermediate_grads)


@dataclass(frozen=True)
class _ExperimentalBoundaryVJPRecord:
    """Private JAX per-boundary VJP oracle record for tests.

    Parameters
    ----------
    label
        Caller-owned boundary label.
    grad
        Cotangent of the scalar loss with respect to the boundary value.
    provenance
        Private helper metadata.
    """

    label: str
    grad: Any
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class _ExperimentalBoundaryVJPResult:
    """Private JAX per-boundary VJP oracle result for tests.

    Parameters
    ----------
    records
        Exact boundary cotangents keyed by caller-owned label.
    skipped
        Skip reasons keyed by caller-owned label.
    """

    records: Mapping[str, _ExperimentalBoundaryVJPRecord]
    skipped: Mapping[str, str]


@dataclass(frozen=True)
class _JaxIntermediateTapSpec:
    """One JAX zero-tap boundary selected for an op output.

    Parameters
    ----------
    op_label
        Final TorchLens op label that owns the boundary.
    layer_label
        Final TorchLens layer label that owns the op.
    capture_index
        Flattened JAX capture index from the shipped label index.
    output_index
        Output position within the JAX equation capture.
    value
        Saved boundary value used to shape the zero tap.
    aval
        Human-readable abstract value.
    dtype_ref
        Backend-neutral dtype reference for the value.
    """

    op_label: str
    layer_label: str
    capture_index: int
    output_index: int
    value: Any
    aval: str
    dtype_ref: DtypeRef | None


_UNSUPPORTED_CONTAINER_SPEC: Final = object()


def _stamp_jaxpr_results(trace: Trace, closed_jaxpr: Any, result: Any, static_argnums: Any) -> None:
    """Record the interpreted jaxpr and its captures on the trace."""

    trace._provenance_jax_closed_jaxpr = closed_jaxpr
    trace.jax_equation_captures = result.equations
    trace.jax_ordered_captures = result.captures
    trace.jax_region_captures = tuple(
        capture for capture in result.captures if isinstance(capture, JaxRegionCapture)
    )
    trace._provenance_jax_inlined_call_primitives = result.inlined_call_primitives
    trace._options_jax_static_argnums = static_argnums


def _jax_pre_replay_failure_status(
    trace: Trace, log: _JaxValidationLog, exc: BaseException
) -> ValidationStatus:
    """The failed status recorded when validation raised before any replay verdict."""

    _record_jax_decision(
        log,
        "<trace>",
        "",
        _JaxVerdict("metadata", "failed", "metadata_invariant_exception"),
    )
    status_result = ValidationStatus.result(
        passed=False,
        backend="jax",
        source="loaded" if getattr(trace, "_loaded_from_bundle", False) else "live",
        payload_load_status=getattr(trace, "payload_load_status", None),
        num_failed_ops=1,
        decisions=tuple(log.decisions),
    )
    return replace(
        status_result,
        message=f"Replay validation failed before replay: {type(exc).__name__}: {exc}",
    )


def _jax_captures_consistent(
    captures: tuple[Any, ...], equation_ops: Sequence[Any], log: _JaxValidationLog
) -> bool:
    """Equation captures and equation ops agree in count and in kind multiset."""

    if len(captures) != len(equation_ops):
        _record_jax_decision(
            log,
            "<trace>",
            "",
            _JaxVerdict("metadata", "failed", "capture_inconsistent"),
            f"{len(captures)} equation captures for {len(equation_ops)} equation ops",
        )
        return False
    capture_kind_counts = Counter(capture.kind for capture in captures)
    op_kind_counts = Counter(_jax_op_capture_kind(op) for op in equation_ops)
    if capture_kind_counts != op_kind_counts:
        _record_jax_decision(
            log,
            "<trace>",
            "",
            _JaxVerdict("metadata", "failed", "capture_inconsistent"),
            "equation capture kinds differ from op capture kinds",
        )
        return False
    return True


def _jax_ops_by_label(trace: Trace) -> dict[str, Any]:
    """Index ops by raw, final and (for single-pass ops) bare labels."""

    # TORCH PARITY: ``parent_arg_positions`` (and ``parents``/``children``)
    # resolve through the CONDITIONAL label map -- a single-pass op's bare
    # ``layer_label``, a multi-pass op's pass-qualified ``label`` -- so a
    # lookup keyed only by raw/final-qualified labels misses every
    # single-pass parent reference. Index the bare label too (safe: a
    # single-pass op's bare label is unambiguous).
    ops_by_label: dict[str, Any] = {}
    for op in trace._layer_list:
        labels = [getattr(op, "_label_raw", None), getattr(op, "label", None)]
        if getattr(op, "num_passes", 1) == 1:
            labels.append(getattr(op, "layer_label", None))
        for label in labels:
            if isinstance(label, str):
                ops_by_label[label] = op
    return ops_by_label


def _validate_one_jax_equation(
    jax: Any,
    capture: Any,
    op: Any,
    ops_by_label: dict[str, Any],
    hidden_outputs_by_label: dict[str, tuple[Any, ...]],
    log: _JaxValidationLog,
) -> bool:
    """Replay one equation against its saved outputs and record the verdict."""

    op_label = str(getattr(op, "label", getattr(op, "_label_raw", "")))
    func_name = str(getattr(op, "func_name", capture.primitive))
    if capture.kind != _jax_op_capture_kind(op):
        _record_jax_decision(
            log,
            op_label,
            func_name,
            _JaxVerdict("metadata", "failed", "capture_inconsistent"),
        )
        return False
    try:
        inputs = _inputs_from_trace_graph(capture, op, ops_by_label, hidden_outputs_by_label)
        replayed, raw = replay_with_edits(capture, op, replay_equation(capture, inputs))
        saved_outputs = _saved_op_outputs(
            op,
            len(replayed),
            hidden_outputs_by_label=hidden_outputs_by_label,
        )
    except Exception as exc:  # noqa: BLE001 - name the op, then fail closed
        _record_jax_decision(
            log,
            op_label,
            func_name,
            _JaxVerdict("replay", "failed", "replay_mismatch"),
            f"{type(exc).__name__}: {exc}",
        )
        return False
    if not jax.tree.all(jax.tree.map(_values_close, replayed, saved_outputs)):
        _record_jax_decision(
            log, op_label, func_name, _JaxVerdict("replay", "failed", "replay_mismatch")
        )
        return False
    reference = saved_outputs if raw is None else raw  # edits: sensitivity of the raw replay
    if not _parent_perturbations_change_output(capture, op, inputs, reference):
        return _record_insensitive_outcome(
            log, (op_label, func_name), capture, op, (inputs, reference)
        )
    _record_jax_decision(
        log, op_label, func_name, _JaxVerdict("replay", "validated", "replay_match")
    )
    return True


class JAXBackend:
    """JAX adapter that captures forward graphs from closed jaxprs."""

    name = "jax"

    def capture_trace(
        self,
        model: Callable[..., Any],
        input_args: object,
        input_kwargs: dict[Any, Any] | None = None,
        *,
        save: Any = None,
        keep_orphans: bool | MissingType = MISSING,
        output_device: str | MissingType = MISSING,
        activation_transform: object | None = None,
        save_raw_activations: bool | MissingType = MISSING,
        detach_saved_activations: bool | MissingType = MISSING,
        save_grads: bool | str | list[Any] | object | None = None,
        random_seed: int | None = None,
        num_context_lines: int | MissingType = MISSING,
        save_arg_values: bool | MissingType = MISSING,
        save_code_context: bool | MissingType = MISSING,
        save_rng_states: bool | MissingType = MISSING,
        recurrence_detection: bool | MissingType = MISSING,
        compute_input_output_distances: bool | MissingType = MISSING,
        verbose: bool | MissingType = MISSING,
        backward_ready: bool | MissingType = MISSING,
        name: str | None | MissingType = MISSING,
        transform: object | None = None,
        raw_input: object | None = None,
        save_raw_input: str | bool | MissingType = MISSING,
        batch_render: str | MissingType = MISSING,
        output_transform: object | None = None,
        save_transformed_output: str | bool | MissingType = MISSING,
        layer_visualizers: dict[Any, Any] | None = None,
        save_visualizations: bool | MissingType = MISSING,
        lookback: int = 0,
        lookback_payload_policy: str = "metadata_only",
        jax_static_argnums: int | Sequence[int] | MissingType = MISSING,
        jax_control_flow: str | MissingType = MISSING,
        jax_max_control_flow_unroll: int | MissingType = MISSING,
        module_identity_mode: str | None | MissingType = MISSING,
        grad_options: GradOptions | None | MissingType = MISSING,
        capture: CaptureOptions | None = None,
        **kwargs: Any,
    ) -> Trace:
        """Capture a JAX raw-function forward pass into a TorchLens ``Trace``.

        Parameters
        ----------
        model
            JAX-compatible callable, expected to follow ``fn(params, *inputs)``.
        input_args
            Positional arguments for the callable.
        input_kwargs
            Keyword arguments: array values trace as pytree leaves, others bind statically.
        save
            ``"all"``, omitted, or a static ``tl.trace(save=)`` selector of ops to keep.
        keep_orphans
            Whether orphan ops are retained.
        output_device
            Must be ``"same"``.
        activation_transform
            Unsupported for JAX.
        save_raw_activations
            Must be true.
        detach_saved_activations
            Must be false.
        save_grads
            Unsupported for JAX.
        random_seed
            Unsupported for JAX; stochasticity must be explicit
            through params/input PRNG key leaves.
        num_context_lines
            Stored on the returned trace.
        save_arg_values
            Must be false.
        save_code_context
            Must be false.
        save_rng_states
            Must be false.
        recurrence_detection
            Stored on the returned trace.
        verbose
            Stored on the returned trace.
        backward_ready
            Unsupported for JAX.
        name
            Trace label.
        transform
            Unsupported for JAX.
        raw_input
            Original user input.
        save_raw_input
            Raw-input save policy.
        batch_render
            Raw-input render policy.
        output_transform
            Optional metadata transform for final output.
        save_transformed_output
            Transformed-output save policy.
        layer_visualizers
            Unsupported for JAX.
        save_visualizations
            Unsupported for JAX.
        lookback
            Predicate lookback window. Only the default ``0`` is supported.
        lookback_payload_policy
            Predicate lookback payload policy. Only the default is supported.
        jax_static_argnums
            Positional callable argument indexes to declare static for
            ``jax.make_jaxpr``. Static values are not interpreted as dynamic
            jaxpr leaves.
        jax_control_flow
            JAX nested control-flow policy. ``"unroll"`` expands supported
            control-flow primitives into graph-visible equations and falls
            back to regions for over-cap scan/while; ``"region"`` imports
            supported scan/while/custom-VJP forward boundaries as unverified
            regions; ``"reject"`` preserves the earlier nested-primitive
            rejection behavior.
        jax_max_control_flow_unroll
            Maximum static unroll length for one JAX control-flow primitive.
        module_identity_mode
            Optional JAX module mode. Raw callables use ``"function_root"``;
            Equinox module roots default to ``"pytree_module"``.
        grad_options
            Optional JAX derived-gradient configuration: a second pure functional
            ``jax.value_and_grad`` pass populates ``trace.autograd.derived_grads``.
        capture
            Grouped ``CaptureOptions``, the sole spelling for the flat capture kwargs
            above; an explicit field is merged in (``merge_capture_options``) first.
        **kwargs
            Extra public trace kwargs rejected by this backend.

        Returns
        -------
        Trace
            Captured JAX trace.
        """

        save_predicate = pop_static_label_save_predicate({"save": save}, backend_name="JAX")
        facet_recipes = pop_facet_recipes(kwargs)
        intervene = pop_edit_kwargs(kwargs)
        self._reject_extra_kwargs(kwargs)
        _reject_transformed_callable(model)
        # N5: ``trace()`` passes the grouped ``capture=CaptureOptions(...)`` only,
        # so an explicit ``capture=`` value must reach these fields, never be
        # dropped. MISSING-typed flat kwargs merge directly; fields defaulting to
        # a concrete ``None`` merge below via ``_resolve_optional_capture_field``.
        capture_options = merge_capture_options(
            capture=capture,
            keep_orphans=keep_orphans,
            output_device=output_device,
            detach_saved_activations=detach_saved_activations,
            save_args=save_arg_values,
            save_source_context=save_code_context,
            save_rng=save_rng_states,
            recurrence_detection=recurrence_detection,
            compute_input_output_distances=compute_input_output_distances,
            verbose=verbose,
            backward_ready=backward_ready,
            name=name,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            save_transformed_output=save_transformed_output,
            save_visualizations=save_visualizations,
            jax_control_flow=jax_control_flow,
            jax_max_control_flow_unroll=jax_max_control_flow_unroll,
            module_identity_mode=module_identity_mode,
        )
        layers_to_save = capture_options._layers_to_save
        keep_orphans = capture_options.keep_orphans
        output_device = capture_options.output_device
        detach_saved_activations = capture_options.detach_saved_activations
        save_arg_values = capture_options.save_args
        save_code_context = capture_options.save_source_context
        save_rng_states = capture_options.save_rng
        recurrence_detection = capture_options.recurrence_detection
        compute_input_output_distances = capture_options.compute_input_output_distances
        verbose = capture_options.verbose
        backward_ready = capture_options.backward_ready
        name = capture_options.name
        save_raw_input = capture_options.save_raw_input
        batch_render = capture_options.batch_render
        save_transformed_output = capture_options.save_transformed_output
        save_visualizations = capture_options.save_visualizations
        jax_control_flow = capture_options.jax_control_flow
        jax_max_control_flow_unroll = capture_options.jax_max_control_flow_unroll
        module_identity_mode = capture_options.module_identity_mode
        save_grads = _resolve_optional_capture_field(capture, "save_grads", save_grads)
        transform = _resolve_optional_capture_field(capture, "transform", transform)
        output_transform = _resolve_optional_capture_field(
            capture, "output_transform", output_transform
        )
        layer_visualizers = _resolve_optional_capture_field(
            capture, "layer_visualizers", layer_visualizers
        )
        random_seed = _resolve_optional_capture_field(capture, "seed", random_seed)
        layers_to_save = _default_if_missing(layers_to_save, "all")
        keep_orphans = _default_if_missing(keep_orphans, False)
        output_device = _default_if_missing(output_device, "same")
        save_raw_activations = _default_if_missing(save_raw_activations, True)
        detach_saved_activations = _default_if_missing(detach_saved_activations, False)
        num_context_lines = _default_if_missing(num_context_lines, 7)
        save_arg_values = _default_if_missing(save_arg_values, False)
        save_code_context = _default_if_missing(save_code_context, False)
        save_rng_states = _default_if_missing(save_rng_states, False)
        recurrence_detection = _default_if_missing(recurrence_detection, True)
        # Torch-parity default: the depth flood runs unless explicitly disabled.
        compute_input_output_distances = _default_if_missing(compute_input_output_distances, True)
        verbose = _default_if_missing(verbose, False)
        backward_ready = _default_if_missing(backward_ready, False)
        name = _default_if_missing(name, None)
        save_raw_input = _default_if_missing(save_raw_input, "small")
        batch_render = _default_if_missing(batch_render, "auto")
        save_transformed_output = _default_if_missing(save_transformed_output, "small")
        save_visualizations = _default_if_missing(save_visualizations, False)
        activation_transform = None if _is_missing(activation_transform) else activation_transform
        save_grads = None if _is_missing(save_grads) else save_grads
        transform = None if _is_missing(transform) else transform
        output_transform = None if _is_missing(output_transform) else output_transform
        layer_visualizers = None if _is_missing(layer_visualizers) else layer_visualizers
        jax_control_flow = _default_if_missing(jax_control_flow, "unroll")
        jax_max_control_flow_unroll = _default_if_missing(jax_max_control_flow_unroll, 64)
        module_identity_mode = _default_if_missing(module_identity_mode, None)
        grad_options = None if _is_missing(grad_options) else grad_options
        args = self._normalize_input_args(input_args)
        _reject_tracer_inputs(args)
        refuse_if_bad_jax_control_flow(jax_control_flow, jax_max_control_flow_unroll)
        eqx_tree = discover_equinox_module_tree(model)
        nnx_tree = discover_nnx_module_tree(model) if eqx_tree is None else None
        module_tree = eqx_tree or nnx_tree
        use_pytree_module = _resolve_jax_module_identity_mode(
            module_identity_mode,
            module_tree,
        )
        call_fn, args = bind_input_kwargs(model, args, input_kwargs)
        if not _is_missing(random_seed) and random_seed is not None:
            raise BackendUnsupportedError(
                "The JAX backend requires explicit PRNG keys as params/input leaves; "
                "random_seed and torch-style RNG replay are unsupported."
            )
        static_argnums = _normalize_static_argnums(jax_static_argnums, len(args))
        self._reject_unsupported_options(
            save=layers_to_save,
            input_kwargs=input_kwargs,
            output_device=output_device,
            activation_transform=activation_transform,
            save_raw_activations=save_raw_activations,
            detach_saved_activations=detach_saved_activations,
            save_grads=save_grads,
            save_args=save_arg_values,
            save_source_context=save_code_context,
            save_rng=save_rng_states,
            backward_ready=backward_ready,
            transform=transform,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,
        )
        trace = self._new_trace(
            model=model,
            keep_orphans=cast(bool, keep_orphans),
            num_context_lines=cast(int, num_context_lines),
            recurrence_detection=cast(bool, recurrence_detection),
            compute_input_output_distances=cast(bool, compute_input_output_distances),
            verbose=cast(bool, verbose),
            name=cast(str | None, name),
            raw_input=raw_input,
            save_raw_input=cast(str | bool, save_raw_input),
            batch_render=cast(str, batch_render),
            output_transform=output_transform,
            save_transformed_output=cast(str | bool, save_transformed_output),
        )
        trace.capture_events = CaptureEvents()
        trace._provenance_start_time = time.time()
        if use_pytree_module and module_tree is None:
            raise AssertionError("pytree_module mode requires JAX module metadata.")
        tree = module_tree if use_pytree_module else None
        closed_jaxpr, closure_consts, output = _derive_and_run(call_fn, args, static_argnums, tree)
        flat_args, _args_treedef = flatten_dynamic_args(args, static_argnums)
        session_capture = (args, closure_consts, grad_options, save_code_context)
        session = open_equation_session(self, trace, intervene, session_capture)
        with forward_scopes(session, backward_ready=cast(bool, backward_ready)) as hooks:
            result = interpret_closed_jaxpr_with_inlining(
                return_probe.jaxpr_to_interpret(tree, closed_jaxpr),
                flat_args,
                jax_control_flow=cast(str, jax_control_flow),
                jax_max_control_flow_unroll=cast(int, jax_max_control_flow_unroll),
                equation_hook=session.hook,
                hooks=hooks,
            )
        result, output = session.finish(return_probe.split_outputs(result, tree), output)
        output_leaf_paths = self._output_leaf_paths(output)
        output_container_spec = self._builtin_output_container_spec(output)
        trace._cost_forward_duration = Duration(time.time() - trace._provenance_start_time)

        trace._output_slot = returned(output)
        trace.transformed_output = output_transform(output) if callable(output_transform) else None
        self._emit_arg_sources(trace, args)
        self._emit_const_sources(trace, closure_consts)
        self._emit_equations(trace, result)
        trace.jax_outvar_key_to_capture_index = dict(result.outvar_key_to_capture_index)
        self._mark_output_events(trace, result.outputs, output_leaf_paths, output_container_spec)
        return_probe.note_return_labels(trace, tree)
        events = trace.capture_events
        materialize_from_events(trace, events)
        delattr(trace, "capture_events")
        if tree is not None:
            self._attach_pytree_params(trace, tree)
            stamp_pytree_param_barcodes(trace, tree, result.captures)
        else:
            self._attach_params(trace, args[0] if args else None)
        self._finish_trace(trace, tree, result.equations)
        _stamp_jaxpr_results(trace, closed_jaxpr, result, static_argnums)
        retain_backward_state(trace, session, events, backward_ready=cast(bool, backward_ready))
        apply_static_label_save_policy(trace, save_predicate, backend_name="JAX")
        if grad_options is not None:
            captured = (model, call_fn, args, static_argnums, closed_jaxpr, tree, output)
            flow = (cast(str, jax_control_flow), cast(int, jax_max_control_flow_unroll))
            attach_jax_derived_grads(
                self,
                trace=trace,
                capture=DerivedGradCapture(*captured, grad_options, flow),
                derive=derive_closed_jaxpr,
            )
        attach_facet_recipes(trace, facet_recipes, backend_name="JAX")
        freeze_trace_relation_views(trace)
        stamp_backend_finalized(trace)
        return trace

    def validate_trace(
        self,
        trace: Trace,
        *_args: Any,
        **kwargs: Any,
    ) -> bool | ValidationStatus:
        """Validate a JAX trace with replay, perturbation, and invariants.

        Parameters
        ----------
        trace
            Trace produced by this backend.
        *_args
            Ignored compatibility arguments.
        **kwargs
            Compatibility keyword arguments. ``validate_metadata`` controls
            whether backend-neutral invariant checks run.

        Returns
        -------
        bool or ValidationStatus
            True when saved graph payloads replay and parent edges perturb.
            Loaded traces whose runtime capture was stripped return an explicit
            unavailable status.
        """

        status = trace.validation_replay_status
        if not status.is_available:
            trace._validation_replay_status = status
            return status
        status_result = self.validation_status(
            trace, validate_metadata=bool(kwargs.get("validate_metadata", True))
        )
        return status_result if status_result.state == "unverified" else status_result.passed

    def validation_status(
        self, trace: Trace, *, validate_metadata: bool = True
    ) -> ValidationStatus:
        """The protocol's ``validate``: always a ``ValidationStatus``, never a bool.

        Parameters
        ----------
        trace
            Trace produced by this backend.
        validate_metadata
            Whether backend-neutral invariant checks run first.

        Returns
        -------
        ValidationStatus
            ``unavailable`` when runtime captures were stripped; otherwise the
            replay result with per-op ``decisions`` and, on failure, a message
            naming the first offending op.
        """

        status = trace.validation_replay_status
        if not status.is_available:
            trace._validation_replay_status = status
            return status
        log = _JaxValidationLog()
        try:
            if validate_metadata:
                run_metadata_invariants(trace)
            passed = self._validate_jax_equations(trace, log)
            _record_jax_constant_leaves(trace, log)
            if passed and not self._validate_jax_regions(trace):
                passed = False
                _record_jax_decision(
                    log,
                    "<trace>",
                    "",
                    _JaxVerdict("metadata", "failed", "capture_inconsistent"),
                    "importer-owned region boundary or seam check failed",
                )
            status_result = self._validation_status_from_counts(trace, passed=passed, log=log)
        except Exception as exc:  # noqa: BLE001 - the tripwire reports, never raises
            status_result = _jax_pre_replay_failure_status(trace, log, exc)
        trace._validation_replay_status = status_result
        return status_result

    def _validation_status_from_counts(
        self,
        trace: Trace,
        *,
        passed: bool,
        log: _JaxValidationLog | None = None,
    ) -> ValidationStatus:
        """Build aggregate replay status from JAX replay and region counts.

        Parameters
        ----------
        trace
            Trace produced by this backend.
        passed
            Whether per-equation replay validation passed.

        Returns
        -------
        ValidationStatus
            Failed status when replay failed, unverified status when all
            replayed ops passed but importer-owned regions are present, and
            passed status otherwise.
        """

        source: ValidationReplaySource = (
            "loaded" if getattr(trace, "_loaded_from_bundle", False) else "live"
        )
        payload_load_status = getattr(trace, "payload_load_status", None)
        replayed_node_count = len(_jax_equation_ops(trace)) if passed else 0
        unverified_node_count = count_importer_region_annotations(trace) if passed else 0
        decisions = tuple(log.decisions) if log is not None else ()
        status = ValidationStatus.from_replay_counts(
            backend="jax",
            source=source,
            num_replayed_ops=replayed_node_count,
            num_unverified_ops=unverified_node_count,
            num_failed_ops=0 if passed else 1,
            payload_load_status=payload_load_status,
            unverified_reason_counts=(
                {"effectful_region": unverified_node_count} if unverified_node_count else {}
            ),
            exempted_reason_counts=dict(log.exempted_reason_counts) if log is not None else {},
            decisions=decisions,
        )
        first_failure = log.first_failure if log is not None else None
        if status.state == "failed" and first_failure is not None:
            label, func_name, reason = first_failure
            status = replace(
                status,
                message=(
                    f"Replay validation failed at {label} ({func_name}): {reason}; "
                    f"{len(_jax_equation_ops(trace))} equation op(s) in the trace."
                ),
            )
        return status

    def _validate_jax_equations(self, trace: Trace, log: _JaxValidationLog | None = None) -> bool:
        """Validate JAX equation captures against materialized trace payloads.

        Parameters
        ----------
        trace
            Trace produced by this backend.

        Returns
        -------
        bool
            True when equation replay and parent perturbation checks pass.
        """

        import jax

        if log is None:
            log = _JaxValidationLog()
        captures = tuple(getattr(trace, "jax_equation_captures", ()))
        equation_ops = _jax_equation_ops(trace)
        if not _jax_captures_consistent(captures, equation_ops, log):
            return False
        ops_by_label = _jax_ops_by_label(trace)
        hidden_outputs_by_label: dict[str, tuple[Any, ...]] = {
            label: value if isinstance(value, tuple) else (value,)
            for label, value in getattr(trace, "_selective_save_hidden_payloads", {}).items()
        }
        # The len(captures) != len(equation_ops) guard above already returned
        # False, so these pair exactly.
        for capture, op in zip(captures, equation_ops, strict=True):
            if not _validate_one_jax_equation(
                jax, capture, op, ops_by_label, hidden_outputs_by_label, log
            ):
                return False
        return True

    def _validate_jax_regions(self, trace: Trace) -> bool:
        """Validate JAX region boundary/projection metadata and seams.

        Parameters
        ----------
        trace
            Trace produced by this backend.

        Returns
        -------
        bool
            True when every region has well-formed provenance, exact boundary
            arity, saved projection payloads, and non-vacuous seam checks for
            real graph-parent positions.
        """

        captures = tuple(
            capture
            for capture in getattr(trace, "jax_region_captures", ())
            if isinstance(capture, JaxRegionCapture)
        )
        if not captures:
            return True
        region_ops = _jax_region_ops(trace)
        if len(captures) != len(region_ops):
            return False
        # TORCH PARITY: ``parent_arg_positions`` (and ``parents``/``children``)
        # resolve through the CONDITIONAL label map -- a single-pass op's bare
        # ``layer_label``, a multi-pass op's pass-qualified ``label`` -- so a
        # lookup keyed only by raw/final-qualified labels misses every
        # single-pass parent reference. Index the bare label too (safe: a
        # single-pass op's bare label is unambiguous).
        ops_by_label: dict[str, Any] = {}
        for op in trace._layer_list:
            labels = [getattr(op, "_label_raw", None), getattr(op, "label", None)]
            if getattr(op, "num_passes", 1) == 1:
                labels.append(getattr(op, "layer_label", None))
            for label in labels:
                if isinstance(label, str):
                    ops_by_label[label] = op
        hidden_outputs_by_label: dict[str, tuple[Any, ...]] = {
            label: value if isinstance(value, tuple) else (value,)
            for label, value in getattr(trace, "_selective_save_hidden_payloads", {}).items()
        }
        region_ops_by_capture_index = {
            capture.index: op for capture, op in zip(captures, region_ops, strict=False)
        }
        boundary_captures = tuple(capture for capture in captures if capture.kind == "region")
        for boundary in boundary_captures:
            boundary_op = region_ops_by_capture_index.get(boundary.index)
            if boundary_op is None:
                return False
            projections = tuple(
                capture
                for capture in captures
                if capture.kind == "region_output" and capture.region_index == boundary.index
            )
            projection_ops = tuple(
                region_ops_by_capture_index.get(projection.index) for projection in projections
            )
            if any(op is None for op in projection_ops):
                return False
            seam = _RegionSeam(
                boundary, boundary_op, projections, cast(tuple[Any, ...], projection_ops)
            )
            if not _validate_region_boundary(
                seam, ops_by_label=ops_by_label, hidden_outputs_by_label=hidden_outputs_by_label
            ):
                return False
        return True

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Capture then validate a JAX forward pass.

        Parameters
        ----------
        *args
            Public validation positional arguments.
        **kwargs
            Public validation keyword arguments.

        Returns
        -------
        bool
            Validation result.
        """

        validate_metadata = bool(kwargs.pop("validate_metadata", True))
        trace = self.capture_trace(*args, **kwargs)
        result = self.validate_trace(trace, validate_metadata=validate_metadata)
        if not isinstance(result, bool):
            raise BackendUnsupportedError(
                "JAX validation entry unexpectedly produced a replay-unavailable status for "
                "a freshly captured live trace."
            )
        return result

    def is_tensor(self, value: object) -> bool:
        """Return whether ``value`` is a JAX array.

        Parameters
        ----------
        value
            Candidate value.

        Returns
        -------
        bool
            True for JAX arrays.
        """

        import jax

        return isinstance(value, jax.Array)

    def _new_trace(
        self,
        *,
        model: Callable[..., Any],
        keep_orphans: bool,
        num_context_lines: int,
        recurrence_detection: bool,
        verbose: bool,
        name: str | None,
        raw_input: object | None,
        save_raw_input: str | bool,
        batch_render: str,
        output_transform: object | None,
        save_transformed_output: str | bool,
        compute_input_output_distances: bool = True,
    ) -> Trace:
        """Construct an empty trace shell via the shared preview constructor."""

        return new_preview_function_trace(
            backend_name=self.name,
            model=model,
            keep_orphans=keep_orphans,
            num_context_lines=num_context_lines,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            name=name,
            raw_input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=output_transform,
            save_transformed_output=save_transformed_output,
            param_source="pytree-derived",
            compute_input_output_distances=compute_input_output_distances,
        )

    def _emit_arg_sources(self, trace: Trace, args: Sequence[Any]) -> None:
        """Emit input source events for dynamic JAX argument leaves.

        Parameters
        ----------
        trace
            Trace receiving events.
        args
            Positional callable arguments.

        Returns
        -------
        None
            Source events are appended to ``trace.capture_events``.
        """

        flat_with_paths = self._tree_leaves_with_paths(tuple(args))
        for path, value in flat_with_paths:
            if not self.is_tensor(value):
                continue
            self._append_event(
                trace=trace,
                kind="source",
                layer_type="input",
                func_name="input",
                output=value,
                parents=(),
                parent_arg_positions={"args": {}, "kwargs": {}},
                container_path=tuple(path.split(".")),
                annotations={"jax_container_path": path},
            )

    def _emit_const_sources(self, trace: Trace, consts: Sequence[Any]) -> None:
        """Emit ``constant`` source events for closed-jaxpr closure constants.

        The shared constants rule (SPEC section 2, rule 1): a closure array
        the forward used is a source op ``constant`` with a saved payload, and
        the equations that read it list it as a parent. Parent resolution is
        by value identity, the same mechanism input leaves use, because the
        interpreter binds the very ``closed_jaxpr.consts`` objects.

        Parameters
        ----------
        trace
            Trace receiving events.
        consts
            Readable closure constants in ``closed_jaxpr.consts`` order.
        """

        for index, value in enumerate(consts):
            self._append_event(
                trace=trace,
                kind="source",
                layer_type="constant",
                func_name="constant",
                output=value,
                parents=(),
                parent_arg_positions={"args": {}, "kwargs": {}},
                container_path=(),
                annotations={"jax_closure_const_index": index},
            )

    def _emit_equations(self, trace: Trace, result: JaxCaptureResult) -> None:
        """Emit operation events for interpreted JAX equations and regions.

        Parameters
        ----------
        trace
            Trace receiving events.
        result
            Interpreted capture result.

        Returns
        -------
        None
            Equation events are appended to ``trace.capture_events``.
        """

        label_by_value_id: dict[int, str] = {
            id(event.output.tensor.payload): event.label_raw
            for event in trace.capture_events.amended_op_records()
            if event.output.tensor.payload is not None
        }
        label_by_capture_index: dict[int, str] = {}
        capture_index_to_raw_op_label: dict[int, str] = {}
        for equation in result.captures:
            if isinstance(equation, JaxRegionCapture):
                self._emit_region_capture(
                    trace=trace,
                    capture=equation,
                    label_by_value_id=label_by_value_id,
                    label_by_capture_index=label_by_capture_index,
                    capture_index_to_raw_op_label=capture_index_to_raw_op_label,
                )
                continue
            data_parents = tuple(
                ParentEdge(parent_label_raw=label, arg_position=index, _edge_use="arg")
                for index, value in enumerate(equation.input_values)
                if (label := label_by_value_id.get(id(value))) is not None
            )
            control_parents = tuple(
                ParentEdge(
                    parent_label_raw=label_by_capture_index[control_index],
                    arg_position=f"control:{control_index}",
                    _edge_use="control",
                )
                for control_index in equation.control_capture_indices
                if control_index in label_by_capture_index
            )
            parents = (*data_parents, *control_parents)
            parent_positions = {
                "args": {edge.arg_position: edge.parent_label_raw for edge in data_parents},
                "kwargs": {},
            }
            output = (
                equation.output_values[0]
                if len(equation.output_values) == 1
                else equation.output_values
            )
            event = self._append_event(
                trace=trace,
                kind="op",
                layer_type=equation.primitive,
                func_name=equation.primitive,
                output=output,
                parents=parents,
                parent_arg_positions=parent_positions,
                container_path=(),
                equivalence_class=jax_equivalence_key(equation),
                module_stack=equation.module_stack,
                module_call_stack=equation.module_call_stack,
                annotations={
                    "jax_params": repr(dict(equation.params)),
                    "jax_capture_kind": equation.kind,
                    "jax_source_path": "/".join(equation.source_path),
                    "jax_invars": equation.invars,
                    "jax_outvars": equation.outvars,
                    "jax_input_avals": equation.input_avals,
                    "jax_output_avals": equation.output_avals,
                    "jax_inlined": equation.inlined,
                    "jax_module_stack": equation.module_stack,
                    "jax_module_call_stack": equation.module_call_stack,
                },
            )
            label_by_capture_index[equation.index] = event.label_raw
            capture_index_to_raw_op_label[equation.index] = event.label_raw
            for value in equation.output_values:
                label_by_value_id[id(value)] = event.label_raw
        trace._jax_capture_index_to_raw_op_label = capture_index_to_raw_op_label

    def _emit_region_capture(
        self,
        *,
        trace: Trace,
        capture: JaxRegionCapture,
        label_by_value_id: dict[int, str],
        label_by_capture_index: dict[int, str],
        capture_index_to_raw_op_label: dict[int, str],
    ) -> None:
        """Emit one JAX region boundary or projection event.

        Parameters
        ----------
        trace
            Trace receiving events.
        capture
            Region capture to emit.
        label_by_value_id
            Mutable map from runtime value identity to raw producer label.
        label_by_capture_index
            Mutable map from capture index to raw producer label.
        capture_index_to_raw_op_label
            Mutable runtime capture-index map for finalized labels.

        Returns
        -------
        None
            Region event is appended to ``trace.capture_events``.
        """

        if capture.kind == "region":
            trace.annotations[REGION_REPLAY_PROVENANCE_KEY] = REGION_REPLAY_IMPORTER_PROVENANCE
            data_parents = tuple(
                ParentEdge(parent_label_raw=label, arg_position=index, _edge_use="arg")
                for index, value in enumerate(capture.input_values)
                if (label := label_by_value_id.get(id(value))) is not None
            )
            parent_positions = {
                "args": {edge.arg_position: edge.parent_label_raw for edge in data_parents},
                "kwargs": {},
            }
            output = tuple(capture.output_values)
            annotations = {
                REGION_REPLAY_CLASS_KEY: REGION_REPLAY_CLASS,
                REGION_REPLAY_PROVENANCE_KEY: REGION_REPLAY_IMPORTER_PROVENANCE,
                "jax_capture_kind": "region",
                "jax_region_primitive": capture.primitive,
                "jax_region_replay_status": "unverified",
                "jax_region_reason": capture.unverified_reason,
                "jax_region_metadata": dict(capture.region_metadata),
                "jax_source_path": "/".join(capture.source_path),
                "jax_invars": capture.invars,
                "jax_outvars": capture.outvars,
                "jax_input_avals": capture.input_avals,
                "jax_output_avals": capture.output_avals,
                "jax_inlined": capture.inlined,
                "jax_module_stack": capture.module_stack,
                "jax_module_call_stack": capture.module_call_stack,
            }
            event = self._append_event(
                trace=trace,
                kind="op",
                layer_type="jax_region",
                func_name=f"region:{capture.primitive}",
                output=output,
                parents=data_parents,
                parent_arg_positions=parent_positions,
                container_path=(),
                equivalence_class=f"jax:region:{capture.primitive}:{capture.index}",
                module_stack=capture.module_stack,
                module_call_stack=capture.module_call_stack,
                annotations=annotations,
            )
        else:
            if capture.region_index is None or capture.output_index is None:
                raise ValueError("JAX region projection capture is missing boundary metadata.")
            boundary_label = label_by_capture_index[capture.region_index]
            parents = (
                ParentEdge(
                    parent_label_raw=boundary_label,
                    arg_position=f"region_output:{capture.output_index}",
                    _edge_use="control",
                ),
            )
            output = capture.output_values[0]
            event = self._append_event(
                trace=trace,
                kind="op",
                layer_type="jax_region_out",
                func_name=f"region_out:{capture.primitive}[{capture.output_index}]",
                output=output,
                parents=parents,
                parent_arg_positions={"args": {}, "kwargs": {}},
                container_path=(),
                equivalence_class=(
                    f"jax:region_out:{capture.primitive}:"
                    f"{capture.region_index}:{capture.output_index}"
                ),
                module_stack=capture.module_stack,
                module_call_stack=capture.module_call_stack,
                annotations={
                    REGION_REPLAY_PROVENANCE_KEY: REGION_REPLAY_IMPORTER_PROVENANCE,
                    "jax_capture_kind": "region_output",
                    "jax_region_primitive": capture.primitive,
                    "jax_region_output_index": capture.output_index,
                    "jax_region_boundary_index": capture.region_index,
                    "jax_region_replay_status": "unverified",
                    "jax_region_reason": capture.unverified_reason,
                    "jax_region_metadata": dict(capture.region_metadata),
                    "jax_source_path": "/".join(capture.source_path),
                    "jax_invars": capture.invars,
                    "jax_outvars": capture.outvars,
                    "jax_input_avals": capture.input_avals,
                    "jax_output_avals": capture.output_avals,
                    "jax_inlined": capture.inlined,
                    "jax_module_stack": capture.module_stack,
                    "jax_module_call_stack": capture.module_call_stack,
                },
            )
            label_by_value_id[id(output)] = event.label_raw
        label_by_capture_index[capture.index] = event.label_raw
        capture_index_to_raw_op_label[capture.index] = event.label_raw

    def _append_event(
        self,
        *,
        trace: Trace,
        kind: str,
        layer_type: str,
        func_name: str,
        output: object,
        parents: tuple[ParentEdge, ...],
        parent_arg_positions: dict[str, dict[Any, str]],
        container_path: tuple[object, ...],
        annotations: Mapping[str, object],
        equivalence_class: str | None = None,
        module_stack: Sequence[str] = (),
        module_call_stack: Sequence[tuple[str, int]] = (),
    ) -> OpEvent:
        """Append one JAX event to the trace event stream.

        Parameters
        ----------
        trace
            Trace receiving the event.
        kind
            Event kind.
        layer_type
            Layer type label.
        func_name
            Function name.
        output
            Event output payload.
        parents
            Parent edges.
        parent_arg_positions
            Parent argument-position metadata.
        container_path
            Pytree container path.
        equivalence_class
            Optional structural equivalence key. Source events receive a
            unique fallback key.
        module_stack
            Decoded module scopes for operation events.
        module_call_stack
            Decoded module calls for operation events.
        annotations
            Extra annotations.

        Returns
        -------
        OpEvent
            Appended event.
        """

        reserved = trace.capture_events.reserve_label(layer_type)
        func_call_id = trace.capture_events.func_call_id_counter + 1
        trace.capture_events.func_call_id_counter = func_call_id
        policy = CapturePolicy(
            save_payload=True,
            save_grad=False,
        )
        tensor_ref = self._tensor_ref(output, reserved.label_raw)
        module_calls, module_frames = jax_event_module_frames(module_stack, module_call_stack)
        input_ancestors = frozenset(
            edge.parent_label_raw for edge in parents if edge.parent_label_raw.startswith("input.")
        )
        event = OpEvent(
            kind=kind,
            label_raw=reserved.label_raw,
            layer_label_raw=reserved.label_raw,
            layer_type=layer_type,
            raw_index=reserved.raw_index,
            type_index=reserved.type_index,
            step_index=reserved.raw_index,
            source_trace=trace,
            source_trace_id=None,
            tracing_finished=False,
            construction_done=True,
            function=FunctionCallRef(
                func=None,
                func_name=func_name,
                func_qualname=func_name,
                func_call_id=func_call_id,
                code_context=(),
                func_duration=None,
                flops_forward=None,
                flops_backward=None,
                func_rng_states=None,
                func_autocast_state=None,
                arg_names=(),
                num_args_total=0,
                num_pos_args=0,
                num_kwargs=0,
                non_tensor_pos_args=(),
                non_tensor_kwargs=(),
                func_non_tensor_args=(),
                is_inplace=False,
                func_config=(),
            ),
            output=OutputRef(
                tensor=tensor_ref,
                transformed_tensor=None,
                has_saved_activation=True,
                output_device="same",
                activation_transform=None,
                detach_saved_activations=False,
                visualizer_path=None,
                multi_output_index=None,
                in_multi_output=False,
                container_path=container_path,
                container_spec=None,
                child_versions=(),
            ),
            templates=ArgTemplateRef(
                saved_args=None,
                saved_kwargs=None,
                args_template=None,
                kwargs_template=None,
                has_saved_args=False,
            ),
            parents=parents,
            parent_arg_positions=parent_arg_positions,
            _edge_uses=tuple(
                (edge.parent_label_raw, edge.arg_position, edge._edge_use) for edge in parents
            ),
            params=(),
            parent_params=(),
            module_stack=module_frames,
            modules=module_calls,
            backend_semantics=BackendSemantics(
                backend_grad_handle=None,
                grad_fn_class_name=None,
                autograd_memory=None,
                num_autograd_tensors=None,
                mutated_input_positions=(),
                aliased_output_inputs=(),
                unknown_aliasing=False,
                bytes_delta_at_call=None,
                bytes_peak_at_call=None,
            ),
            policy=policy,
            predicate_matched=True,
            pass_index=1,
            grad_fn_class_qualname=None,
            grad_fn_handle=None,
            equivalence_class=equivalence_class or f"jax:{kind}:{layer_type}:{reserved.label_raw}",
            is_transform=False,
            transform_kind=None,
            transform_chain=(),
            transform_config={"_tl_annotations": dict(annotations)},
            transform_fn_name=None,
            transform_fn_qualname=None,
            transform_fn_source=None,
            is_output_parent=False,
            has_internal_source_ancestor=kind != "source" and not parents,
            internal_source_ancestors=frozenset(),
            input_ancestors=input_ancestors,
            root_ancestors=input_ancestors or frozenset({reserved.label_raw}),
            func_call_id=func_call_id,
            is_bottom_level=True,
            is_scalar_bool=None,
            bool_value=None,
            intervention_fired=False,
            is_edited=False,
            fire_results=(),
            intervention_template_ref=None,
            record_context=self._record_context(reserved, output, func_name),
            capture_spec=CaptureSpec(save_out=True, save_metadata=True),
        )
        return cast(OpEvent, append_session_event(self, trace, event))

    def _tensor_ref(self, value: object, label_raw: str) -> TensorRef:
        """Build a tensor reference for a JAX payload.

        Parameters
        ----------
        value
            Captured output value.
        label_raw
            Raw TorchLens label.

        Returns
        -------
        TensorRef
            Backend-neutral tensor reference.
        """

        if not self.is_tensor(value):
            return TensorRef(label_raw, None, None, None, None, 0, value, None, str(id(value)))
        return TensorRef(
            label_raw=label_raw,
            shape=tuple(cast(Any, value).shape),
            dtype=str(cast(Any, value).dtype),
            device=str(cast(Any, value).device),
            requires_grad=None,
            memory=_nbytes(value),
            payload=value,
            blob_ref=None,
            backend_handle_id=str(id(value)),
        )

    def _record_context(
        self, reserved: ReservedLabel, output: object, func_name: str
    ) -> RecordContext:
        """Build a lightweight predicate context for a JAX event.

        Parameters
        ----------
        reserved
            Reserved label metadata.
        output
            Event output.
        func_name
            Function name.

        Returns
        -------
        RecordContext
            Predicate context.
        """

        return RecordContext(
            kind="op",
            label=reserved.label,
            raw_label=reserved.label_raw,
            pass_number=1,
            event_index=reserved.raw_index,
            step_index=None,
            layer_type=reserved.layer_type,
            type_index=reserved.type_index,
            raw_index=reserved.raw_index,
            func_name=func_name,
            address=None,
            module_type=None,
            module_pass_number=None,
            module_stack=(),
            recent_events=(),
            recent_ops=(),
            parent_labels=(),
            input_output_address=None,
            shape=tuple(cast(Any, output).shape) if self.is_tensor(output) else None,
            dtype=DtypeRef.from_value(getattr(output, "dtype", None)),
            tensor_device=DeviceRef.from_value(getattr(output, "device", None)),
            tensor_requires_grad=None,
            output_index=None,
            is_bottom_level_func=True,
            time_since_pass_start=0.0,
            sample_id=None,
            label_raw=reserved.label_raw,
            label_prefix=reserved.layer_type,
            func_call_id=reserved.raw_index,
            parent_labels_raw=(),
            is_output_parent=False,
            backend_requires_isolation=False,
            is_scalar_bool=None,
            bool_value=None,
        )

    def _mark_output_events(
        self,
        trace: Trace,
        outputs: Sequence[Any],
        output_leaf_paths: Sequence[tuple[object, ...]],
        output_container_spec: ContainerSpec | None,
    ) -> None:
        """Mark final equation outputs as output parents.

        Parameters
        ----------
        trace
            Trace whose events are updated.
        outputs
            Flat interpreter outputs.
        output_leaf_paths
            Flat direct-output pytree container paths in jaxpr output order.
        output_container_spec
            Full builtin output-container spec when one can be reconstructed
            portably, otherwise ``None`` for path-only degradation.

        Returns
        -------
        None
            Output-parent flags are updated in place.
        """

        output_ids = {id(output) for output in outputs}
        output_metadata_by_id = {
            id(output): (
                index,
                output_leaf_paths[index] if index < len(output_leaf_paths) else (),
            )
            for index, output in enumerate(outputs)
        }
        is_multi_output = len(outputs) > 1
        for event in list(trace.capture_events.amended_op_records()):
            if id(event.output.tensor.payload) not in output_ids:
                continue
            leaf_index, container_path = output_metadata_by_id.get(
                id(event.output.tensor.payload),
                (len(trace.output_layers), ()),
            )
            trace.output_layers.append(event.label_raw)
            updated_output = replace(
                event.output,
                multi_output_index=leaf_index if is_multi_output else None,
                in_multi_output=is_multi_output,
                container_path=container_path,
                container_spec=output_container_spec,
            )
            trace.capture_events.append_amendment(
                amend_preview_output_parent_rebind(
                    event.seq,
                    event.label_raw,
                    is_output_parent=True,
                    output=updated_output,
                )
            )

    def _attach_params(self, trace: Trace, params_tree: object) -> None:
        """Populate ``trace.params`` from first-argument pytree leaves.

        Parameters
        ----------
        trace
            Trace receiving parameter metadata.
        params_tree
            First positional argument, interpreted as params.

        Returns
        -------
        None
            Trace parameter accessors and counters are updated.
        """

        param_logs: dict[str, Param] = {}
        for path, value in self._tree_leaves_with_paths(params_tree):
            if not self.is_tensor(value):
                continue
            shape = tuple(cast(Any, value).shape)
            dtype = str(cast(Any, value).dtype)
            address = path
            param = Param(
                module_address="self",
                name=path,
                shape=shape,
                dtype=cast(Any, dtype),
                num_params=_numel(shape),
                param_memory=_nbytes(value) or 0,
                trainable=True,
                address=address,
                barcode=f"jax:{address}",
                has_optimizer=None,
            )
            param.dtype_ref = DtypeRef(backend="jax", name=dtype)
            param.device_ref = DeviceRef.from_value(getattr(value, "device", None))
            param.backend_address = f"pytree:{address}"
            param.resolver_status = "metadata_only"
            param._param_ref = None
            param.trace = trace
            param_logs[address] = param
        trace._param_logs = ParamAccessor(param_logs)
        _stamp_param_totals(trace, param_logs.values())
        trace._num_layers_with_params = 0
        trace._provenance_param_source = "pytree-derived" if param_logs else "none"

    def _attach_pytree_params(
        self,
        trace: Trace,
        tree: EquinoxModuleTree | NnxModuleTree,
    ) -> None:
        """Populate ``trace.params`` from a JAX module tree.

        Parameters
        ----------
        trace
            Trace receiving parameter metadata.
        tree
            Discovered Equinox or Flax NNX module tree.

        Returns
        -------
        None
            Trace parameter accessors and counters are updated.
        """

        if isinstance(tree, EquinoxModuleTree):
            param_logs = equinox_param_logs(tree, trace)
        else:
            param_logs = nnx_param_logs(tree, trace)
        trace._param_logs = ParamAccessor(param_logs)
        _stamp_param_totals(trace, param_logs.values())
        trace._num_layers_with_params = 0
        trace._provenance_param_source = "pytree-derived" if param_logs else "none"

    def _attach_derived_grads(
        self,
        *,
        trace: Trace,
        model: Callable[..., Any],
        args: Sequence[Any],
        static_argnums: Sequence[int],
        closed_jaxpr: Any,
        captured_output: Any,
        grad_options: GradOptions,
        jax_control_flow: str,
        jax_max_control_flow_unroll: int,
    ) -> None:
        """Compute and attach JAX leaf-level derived gradients.

        Parameters
        ----------
        trace
            Trace receiving derived gradient records.
        model
            Pure JAX callable captured by the trace.
        args
            Normalized positional call arguments.
        static_argnums
            Declared static positional argument indexes.
        closed_jaxpr
            Captured forward closed jaxpr.
        captured_output
            Raw output from the captured forward function.
        grad_options
            Derived-gradient options.
        jax_control_flow
            Control-flow policy used by normal capture.
        jax_max_control_flow_unroll
            Control-flow unroll cap used by normal capture.

        Returns
        -------
        None
            ``trace.autograd.derived_grads`` and unambiguous param gradient slots are populated.
        """

        import jax

        if not args:
            raise ValueError("JAX derived gradients require params as positional argument 0.")
        if 0 in set(static_argnums):
            raise ValueError("JAX derived gradients require params arg 0 to be dynamic.")
        _reject_closed_over_host_state(model)
        _assert_same_treedef(grad_options.params, args[0], label="grad_options.params")
        input_grad_argnums = _normalize_input_grad_argnums(
            grad_options.input_grad_argnums, len(args) - 1
        )
        differentiated_argnums = (0, *(index + 1 for index in input_grad_argnums))
        if set(differentiated_argnums) & set(static_argnums):
            raise ValueError("JAX derived gradients cannot differentiate declared static args.")
        differentiated_paths = _differentiated_leaf_paths(args, differentiated_argnums)
        fingerprint = _gradient_fingerprint(
            closed_jaxpr=closed_jaxpr,
            args=args,
            static_argnums=static_argnums,
            differentiated_paths=differentiated_paths,
            loss_fn=grad_options.loss_fn,
        )
        repeat_closed_jaxpr = derive_closed_jaxpr(model, args, static_argnums)
        reject_undeclared_consts(repeat_closed_jaxpr)
        repeat_fingerprint = _gradient_fingerprint(
            closed_jaxpr=repeat_closed_jaxpr,
            args=args,
            static_argnums=static_argnums,
            differentiated_paths=differentiated_paths,
            loss_fn=grad_options.loss_fn,
        )
        if repeat_fingerprint != fingerprint:
            raise ValueError("JAX derived gradient preflight fingerprint diverged before AD run.")

        def value_fn(*value_args: Any) -> tuple[Any, Any]:
            """Return scalar loss plus raw output aux for ``jax.value_and_grad``.

            Parameters
            ----------
            *value_args
                Positional values passed by JAX AD.

            Returns
            -------
            tuple[Any, Any]
                Scalar loss and raw model output.
            """

            raw_output = model(*value_args)
            if grad_options.loss_fn is None:
                loss = raw_output
            else:
                loss = grad_options.loss_fn(raw_output)
            if not _is_scalar_jax_value(loss):
                raise ValueError(
                    "JAX derived gradients require loss_fn(raw_output) to be scalar unless "
                    "the traced output is already scalar."
                )
            return loss, raw_output

        grad_fn = jax.value_and_grad(value_fn, argnums=differentiated_argnums, has_aux=True)
        (_loss, aux_output), grads = grad_fn(*args)
        aux_output = _block_until_ready_tree(aux_output)
        captured_output = _block_until_ready_tree(captured_output)
        if not jax.tree.all(jax.tree.map(_values_close, aux_output, captured_output)):
            raise ValueError(
                "JAX derived gradient run raw output diverged from captured raw output; "
                "refusing to expose trace.autograd.derived_grads."
            )
        final_closed_jaxpr = derive_closed_jaxpr(model, args, static_argnums)
        reject_undeclared_consts(final_closed_jaxpr)
        final_fingerprint = _gradient_fingerprint(
            closed_jaxpr=final_closed_jaxpr,
            args=args,
            static_argnums=static_argnums,
            differentiated_paths=differentiated_paths,
            loss_fn=grad_options.loss_fn,
        )
        if final_fingerprint != fingerprint:
            raise ValueError("JAX derived gradient fingerprint diverged after AD run.")

        grad_trees = grads
        records: dict[str, DerivedGradRecord] = {}
        for argnum, grad_tree in zip(differentiated_argnums, grad_trees, strict=True):
            records.update(
                _records_for_grad_tree(
                    grad_tree=grad_tree,
                    argnum=argnum,
                    provenance={
                        "backend": "jax",
                        "kind": "derived_gradient",
                        "fingerprint": fingerprint,
                        "loss_fn": _callable_identity(grad_options.loss_fn),
                    },
                )
            )
        trace._autograd_derived_grads = DerivedGradAccessor(records)
        mirror_param_derived_grads(trace, records)
        if grad_options.intermediate_grads:
            try:
                from ..._data_classes.derived_grad import set_intermediate_derived_grads

                set_intermediate_derived_grads(
                    trace,
                    (
                        self._derive_intermediate_grads_zero_tap(
                            trace=trace,
                            closed_jaxpr=closed_jaxpr,
                            args=args,
                            static_argnums=static_argnums,
                            captured_output=captured_output,
                            grad_options=grad_options,
                            fingerprint=fingerprint,
                            jax_control_flow=jax_control_flow,
                            jax_max_control_flow_unroll=jax_max_control_flow_unroll,
                        )
                    ),
                )
            except BackendUnsupportedError:
                raise
            except Exception as exc:
                from ..._data_classes.derived_grad import set_intermediate_derived_grads

                set_intermediate_derived_grads(trace, IntermediateDerivedGradAccessor())
                trace.jax_intermediate_derived_grad_status = {
                    "status": "degraded",
                    "reason": f"producer_error:{type(exc).__name__}",
                }

    def _derive_intermediate_grads_zero_tap(
        self,
        *,
        trace: Trace,
        closed_jaxpr: Any,
        args: Sequence[Any],
        static_argnums: Sequence[int],
        captured_output: Any,
        grad_options: GradOptions,
        fingerprint: str,
        jax_control_flow: str,
        jax_max_control_flow_unroll: int,
    ) -> IntermediateDerivedGradAccessor:
        """Compute exact JAX op-level grads with a separate zero-tap AD replay.

        Parameters
        ----------
        trace
            Finalized trace whose saved ops define the attachment set.
        closed_jaxpr
            Captured forward closed jaxpr.
        args
            Normalized positional call arguments.
        static_argnums
            Declared static positional argument indexes.
        captured_output
            Raw output from normal capture.
        grad_options
            Derived-gradient options.
        fingerprint
            Leaf-gradient fingerprint shared in provenance.
        jax_control_flow
            Control-flow policy used by normal capture.
        jax_max_control_flow_unroll
            Control-flow unroll cap used by normal capture.

        Returns
        -------
        IntermediateDerivedGradAccessor
            Exact op-level records confirmed by producer, VJP, and finite difference.
        """

        import jax
        import jax.numpy as jnp

        specs = _jax_intermediate_tap_specs(trace)
        if len(specs) > grad_options.max_intermediate_grads:
            raise BackendUnsupportedError(
                "JAX intermediate derived gradients are capped at "
                f"{grad_options.max_intermediate_grads} saved boundaries; got {len(specs)}."
            )
        if not specs:
            return IntermediateDerivedGradAccessor()

        flat_args, _args_treedef = flatten_dynamic_args(args, static_argnums)
        _output_leaves, output_treedef = jax.tree.flatten(captured_output)
        zero_taps = tuple(jnp.zeros_like(spec.value) for spec in specs)

        def loss_from_taps(taps: tuple[Any, ...]) -> Any:
            """Return scalar loss from one all-tap replay.

            Parameters
            ----------
            taps
                Flat zero-tap values aligned with ``specs``.

            Returns
            -------
            Any
                Scalar loss.
            """

            tap_values = {
                (spec.capture_index, spec.output_index): tap
                for spec, tap in zip(specs, taps, strict=True)
            }
            result = interpret_closed_jaxpr_with_inlining(
                closed_jaxpr,
                flat_args,
                jax_control_flow=jax_control_flow,
                jax_max_control_flow_unroll=jax_max_control_flow_unroll,
                tap_values_by_capture_output=tap_values,
            )
            raw_output = jax.tree.unflatten(output_treedef, result.outputs)
            loss = raw_output if grad_options.loss_fn is None else grad_options.loss_fn(raw_output)
            if not _is_scalar_jax_value(loss):
                raise ValueError("JAX intermediate derived gradients require a scalar loss.")
            return loss

        def loss_from_replacement(spec: _JaxIntermediateTapSpec, replacement: Any) -> Any:
            """Return scalar loss after replacing one boundary value.

            Parameters
            ----------
            spec
                Boundary to replace.
            replacement
                Replacement value for the boundary output.

            Returns
            -------
            Any
                Scalar loss.
            """

            result = interpret_closed_jaxpr_with_inlining(
                closed_jaxpr,
                flat_args,
                jax_control_flow=jax_control_flow,
                jax_max_control_flow_unroll=jax_max_control_flow_unroll,
                replacement_values_by_capture_output={
                    (spec.capture_index, spec.output_index): replacement
                },
            )
            raw_output = jax.tree.unflatten(output_treedef, result.outputs)
            loss = raw_output if grad_options.loss_fn is None else grad_options.loss_fn(raw_output)
            if not _is_scalar_jax_value(loss):
                raise ValueError("JAX intermediate derived gradients require a scalar loss.")
            return loss

        producer_grads = jax.grad(loss_from_taps)(zero_taps)
        replayed_output = _raw_output_from_replay(
            closed_jaxpr=closed_jaxpr,
            flat_args=flat_args,
            output_treedef=output_treedef,
            jax_control_flow=jax_control_flow,
            jax_max_control_flow_unroll=jax_max_control_flow_unroll,
        )
        if not jax.tree.all(jax.tree.map(_values_close, replayed_output, captured_output)):
            raise ValueError("JAX zero-tap replay raw output diverged from captured output.")

        specs_by_label: dict[str, list[_JaxIntermediateTapSpec]] = defaultdict(list)
        passed_by_label: dict[str, list[tuple[_JaxIntermediateTapSpec, Any]]] = defaultdict(list)
        for tap_index, (spec, grad) in enumerate(zip(specs, producer_grads, strict=True)):
            specs_by_label[spec.op_label].append(spec)
            if not _jax_intermediate_oracle_passes(
                spec=spec,
                tap_index=tap_index,
                producer_grad=grad,
                zero_taps=zero_taps,
                loss_from_replacement=loss_from_replacement,
            ):
                continue
            passed_by_label[spec.op_label].append((spec, grad))

        records: dict[str, IntermediateDerivedGradRecord] = {}
        for op_label, passed_specs in passed_by_label.items():
            if len(passed_specs) != len(specs_by_label[op_label]):
                continue
            first_spec = passed_specs[0][0]
            grad_payload = (
                passed_specs[0][1]
                if len(passed_specs) == 1
                else tuple(grad for _spec, grad in passed_specs)
            )
            records[op_label] = IntermediateDerivedGradRecord(
                op_label=op_label,
                layer_label=first_spec.layer_label,
                aval=(
                    f"ShapedArray({tuple(getattr(grad_payload, 'shape', ()))}, "
                    f"{getattr(grad_payload, 'dtype', None)})"
                ),
                dtype_ref=DtypeRef.from_value(getattr(grad_payload, "dtype", None)),
                grad=grad_payload,
                provenance={
                    "backend": "jax",
                    "kind": "intermediate_derived_gradient",
                    "mechanism": "jax_zero_tap",
                    "fingerprint": fingerprint,
                    "loss_id": _callable_identity(grad_options.loss_fn),
                    "save_predicate_id": "trace.saved_ops",
                    "status": "exact",
                    "oracle": "producer_all_tap+per_tap_vjp+finite_difference",
                    "max_intermediate_grads": grad_options.max_intermediate_grads,
                },
            )
        return IntermediateDerivedGradAccessor(records)

    def _finish_trace(
        self,
        trace: Trace,
        module_tree: EquinoxModuleTree | NnxModuleTree | None = None,
        captures: Sequence[JaxRuntimeCapture] = (),
    ) -> None:
        """Finalize materialized JAX raw logs into public accessors.

        Parameters
        ----------
        trace
            Trace to finalize.
        module_tree
            Optional JAX module tree for pytree-module finalization.
        captures
            Interpreted JAX equation captures in event order.

        Returns
        -------
        None
            Trace accessors are populated.
        """

        assignments = self._jax_recurrence_assignments(trace)
        trace._options_grouping_policy = build_grouping_policy_stamp(
            ran_recurrence_grouping=bool(trace._options_recurrence_detection),
            requested=trace._options_grouping,
        )
        raw_labels = tuple(trace._raw_graph_ws.raw_layer_labels_list)
        raw_ops = trace._raw_graph_ws.raw_layer_dict
        finals = torch_rule_final_labels(
            {label: raw_ops[label] for label in raw_labels}, assignments
        )
        raw_to_final_op_label: dict[str, str] = {}
        # TORCH PARITY: torch (``postprocess/labeling.py``) relabels graph-edge
        # fields (``parents``/``children``/the lineage sets) through a
        # CONDITIONAL mapping -- a referenced op's bare ``layer_label`` when its
        # layer has a single pass, its pass-qualified ``label`` only when the
        # layer is multi-pass -- never the unconditionally-qualified op label.
        # ``raw_to_final_op_label`` stays the always-qualified mapping (used for
        # ``recurrent_ops``/``equivalent_ops``/capture-index bookkeeping below);
        # this sibling mapping feeds ``_relabel_jax_graph_edges`` instead.
        raw_to_final_layer_label: dict[str, str] = {}

        trace._layer_list = []
        trace._layer_dict_main_keys.clear()
        trace._layer_dict_all_keys.clear()
        trace._layer_logs.clear()
        trace.op_labels = []
        trace.layer_labels = []
        trace._layer_num_calls.clear()
        trace._lookup_keys_to_layer_num_dict.clear()
        trace._layer_num_to_lookup_keys_dict.clear()

        for raw_index, label in enumerate(raw_labels):
            op_log = trace._raw_graph_ws.raw_layer_dict[label]
            assignment = assignments.get(
                label,
                RecurrenceAssignment(
                    layer_label=label,
                    recurrent_labels=(label,),
                    pass_index=1,
                    num_passes=1,
                    equivalence_key=op_log.graph.equivalence_class or label,
                ),
            )
            pass_label = apply_final_label(op_log, finals[label])
            layer_label = op_log.layer_label
            op_log._label_raw = label
            op_log._layer_label_raw = assignment.layer_label
            op_log.pass_number = assignment.pass_index
            op_log.num_passes = assignment.num_passes
            op_log._graph_equivalence_class = assignment.equivalence_key
            op_log._graph_site_key = assignment.site_key
            op_log.dtype_ref = DtypeRef.from_value(op_log.dtype)
            op_log.device_ref = DeviceRef.from_value(getattr(op_log.out, "device", None))
            op_log.backend_address = f"jaxpr:{label}"
            op_log.resolver_status = "resolved"
            raw_to_final_op_label[label] = pass_label
            raw_to_final_layer_label[label] = (
                layer_label if assignment.num_passes == 1 else pass_label
            )

        self._relabel_jax_graph_edges(trace, raw_to_final_layer_label)
        raw_by_capture_index = getattr(trace, "_jax_capture_index_to_raw_op_label", {})
        trace.jax_capture_index_to_final_op_label = {
            capture_index: raw_to_final_op_label[raw_label]
            for capture_index, raw_label in raw_by_capture_index.items()
            if raw_label in raw_to_final_op_label
        }
        equivalent_labels_by_key = _jax_equivalent_labels_by_key(trace, raw_labels)

        for raw_index, label in enumerate(raw_labels):
            op_log = trace._raw_graph_ws.raw_layer_dict[label]
            assignment = assignments[label]
            op_log._graph_recurrent_ops = [
                raw_to_final_op_label[member]
                for member in assignment.recurrent_labels
                if member in raw_to_final_op_label
            ]
            op_log._graph_equivalent_ops = equivalent_labels_by_key.get(
                op_log.graph.equivalence_class, set()
            )
            # Torch parity (see the identical fix in
            # ``backends/_finalize.py::_finalize_single_op``): always
            # register the bare layer label, not just for multi-pass groups.
            # For a single-pass op the bare key is unambiguous and resolves
            # exactly that op; without it, ``postprocess/finalization.py``'s
            # ``_build_module_logs`` -- which looks each module's recorded
            # ops up by bare layer label through ``layer_dict_all_keys`` --
            # silently dropped every single-pass op (the common case) from
            # ``Module.layer_labels``, tripping the ``module_attribution``
            # metadata invariant.
            op_log.lookup_keys = [label, op_log.label, op_log.layer_label]
            trace._layer_list.append(op_log)
            trace._layer_dict_main_keys[op_log.label] = op_log
            for lookup_key in op_log.lookup_keys:
                if lookup_key not in trace._layer_dict_all_keys:
                    trace._layer_dict_all_keys[lookup_key] = op_log
                    trace._lookup_keys_to_layer_num_dict[lookup_key] = raw_index
                trace._layer_num_to_lookup_keys_dict[raw_index].append(lookup_key)
            if op_log.num_passes > 1:
                # Incidental raw-index artifact, not a contract: every pass
                # overwrites the bare layer label (last wins, torch parity).
                trace._layer_dict_all_keys[op_log.layer_label] = op_log
                trace._lookup_keys_to_layer_num_dict[op_log.layer_label] = raw_index
            trace.op_labels.append(op_log.label)
            if op_log.layer_label not in trace.layer_labels:
                trace.layer_labels.append(op_log.layer_label)
            trace._layer_num_calls[op_log.layer_label] = op_log.num_passes
            layer_log = trace._layer_logs.get(op_log.layer_label)
            if layer_log is None:
                layer_log = Layer(op_log)
                trace._layer_logs[op_log.layer_label] = layer_log
            layer_log.num_passes = op_log.num_passes
            layer_log.ops[op_log.pass_number] = op_log
            layer_log.call_labels.append(op_log.label)
            layer_log.dtype_ref = op_log.dtype_ref
            layer_log.device_ref = op_log.device_ref
            layer_log.backend_address = op_log.backend_address
            layer_log.resolver_status = op_log.resolver_status

        if module_tree is not None:
            self._attach_pytree_op_params(trace, module_tree, captures)

        trace._graph_op_equivalence_classes.clear()
        trace._graph_op_equivalence_classes.update(equivalent_labels_by_key)
        trace._num_ops = sum(
            1
            for op_log in trace._layer_list
            if not (op_log.is_input or op_log.is_output or op_log.graph.is_buffer)
        )
        if module_tree is None:
            # ``jax_container_path`` is only set on the input-SOURCE event
            # where a param first enters the trace (``_emit_arg_sources``
            # above) -- it is never set on the downstream ops that actually
            # consume the param in a computation (e.g. the matmul/conv that
            # takes the param as an operand). Matching on it directly (as
            # the old code did) attaches params to the input-echo op
            # instead of the op that genuinely uses them, so
            # ``has_params``/``used_by_ops``/``_num_layers_with_params``
            # never reflect real param usage. Mirror
            # ``_attach_pytree_op_params`` above (and the equivalent
            # ``torch``/``mlx``/``paddle`` param attribution): resolve each
            # param's source-event label, then attach the param to every op
            # whose graph ``parents`` include that source label -- i.e. the
            # op that actually took the param value as a computational
            # operand.
            # Keyed by BOTH the pass-qualified ``op_log.label`` and (for a
            # single-pass source, torch parity) the bare ``op_log.layer_label``:
            # ``op_log.parents`` on the consuming op now holds whichever
            # spelling matches this source's own pass count (see
            # ``_relabel_jax_graph_edges``), and a source op is always
            # single-pass, so both keys point at the one param.
            param_log_by_source_label: dict[str, Param] = {}
            for op_log in trace._layer_list:
                path = getattr(op_log, "annotations", {}).get("jax_container_path")
                if not isinstance(path, str) or not path.startswith("0."):
                    continue
                param_address = path.removeprefix("0.")
                if param_address not in trace._param_logs:
                    continue
                param_log_by_source_label[op_log.label] = trace._param_logs[param_address]
                if op_log.num_passes == 1:
                    param_log_by_source_label[op_log.layer_label] = trace._param_logs[param_address]

            if param_log_by_source_label:
                usage_membership: dict[int, tuple[set[str], set[str], set[str]]] = {}
                for op_log in trace._layer_list:
                    seen_barcodes: set[str] = set()
                    consumed_params: list[Param] = []
                    for parent_label in op_log.parents:
                        if not isinstance(parent_label, str):
                            continue
                        param = param_log_by_source_label.get(parent_label)
                        if param is None or param.barcode in seen_barcodes:
                            continue
                        seen_barcodes.add(param.barcode)
                        consumed_params.append(param)
                    if not consumed_params:
                        continue
                    op_log._param_barcodes = [param.barcode for param in consumed_params]
                    op_log._param_logs = consumed_params
                    op_log.param_shapes = [param.shape for param in consumed_params]
                    op_log.num_params = sum(param.num_params for param in consumed_params)
                    op_log.num_params_trainable = sum(
                        param.num_params for param in consumed_params if param.is_trainable
                    )
                    op_log.num_params_frozen = op_log.num_params - op_log.num_params_trainable
                    op_log._cost_param_memory = Bytes(
                        sum(int(param.param_memory) for param in consumed_params)
                    )
                    # Set-memoized usage cross-links (the bare list scans were
                    # O(m^2) for a param consumed by m ops).
                    _attach_param_usage(trace, op_log, usage_membership)

            # Recompute the trace-level aggregate param counters from the
            # (now-correct) per-op attribution, deduplicating by
            # ``layer_label`` -- mirrors ``_attach_pytree_op_params`` above.
            # These were previously set once, early, to the total count of
            # ALL param-pytree leaves regardless of whether the traced
            # function actually consumes them, which both over-reports
            # unused params and fails ``check_metadata_invariants``'
            # ``trace_self_consistency`` check (which independently
            # recomputes the same dedup-by-layer sum from ``layer_list``).
            trace._num_layers_with_params = len(
                {op.layer_label for op in trace._layer_list if op.has_params}
            )
            # Parameter-identity basis, not a per-layer-label dedup sum: a
            # param reused across more than one op/layer (a tied weight, or a
            # module shared across two sibling attributes) is attached to
            # EACH referencing layer, so summing once per unique layer_label
            # still double-counts the param itself and trips the
            # ``trace_self_consistency`` invariant's
            # ``num_param_tensors != len(param_logs)`` check (same fix as
            # ``backends/_finalize.py``'s ``_update_param_totals_from_layers``).
            _stamp_param_totals(trace, trace._param_logs)

        def _final_label(label: str) -> str:
            """Resolve one raw label to its final CONDITIONAL label.

            Torch parity: bare ``layer_label`` for a single-pass op, but the
            pass-qualified ``label`` for a multi-pass one. Torch's own
            ``input_layers``/``output_layers`` rename always uses the bare
            label, but only because torch always wraps real computation in
            single-pass input/output pseudo-ops -- the two mappings coincide
            there. JAX does not always wrap this way (``trace.output_layers``
            can name the real last-pass op of a multi-pass layer directly),
            where a bare label would be ambiguous (multi-pass bare-label
            lookups refuse typed), defeating the whole point of
            ``output_layers`` as "the op that produced this output" (see the
            identical ``backends/_finalize.py`` fix for the shared preview
            path).
            """

            op_log = trace._raw_graph_ws.raw_layer_dict.get(label)
            if op_log is None:
                return label
            return op_log.layer_label if op_log.num_passes == 1 else op_log.label

        trace.output_layers = [_final_label(label) for label in trace.output_layers]
        # ``trace.input_layers`` needs the identical raw -> final relabel:
        # left raw, it seeded ``compute_preview_input_output_distances``'s
        # "input" flood below with RAW labels (``input_1_1_raw``), which adds
        # the raw string straight into every input op's ``input_ancestors``
        # (`getattr(starting_op, "input_ancestors").add(starting_label)`),
        # surviving postprocessing and tripping the ``graph_ordering``
        # invariant on every capture with ``mark_layer_depths`` on (the
        # default) -- i.e. nearly every JAX capture.
        trace.input_layers = [_final_label(label) for label in trace.input_layers]
        trace._layers_logged = True
        trace._layers_saved = True
        trace._autograd_has_backward_pass = False
        trace._provenance_end_time = time.time()
        trace._provenance_backend = cast(BackendName, self.name)
        if module_tree is None:
            trace._options_module_identity_mode = "function_root"
            attach_function_root_module(trace)
        else:
            trace._options_module_identity_mode = "pytree_module"
            self._attach_pytree_module_logs(trace, module_tree)
        trace._tracing_finished = True
        compact_op_metadata(trace)
        stamp_preview_graph_shape_hash(trace)
        # The depth flood deliberately resolves ops through its own explicit
        # label index, NOT Trace.__getitem__ (finished-mode lookup returns
        # Layer objects, not the ops the flood must mutate); running it after
        # the finished flag flips is still required so relabeled child edges
        # are present on the ops the index collects.
        trace._options_mark_layer_depths = bool(trace._options_mark_layer_depths)
        if trace._options_mark_layer_depths:
            from .._finalize_distances import compute_preview_input_output_distances

            compute_preview_input_output_distances(trace)

    def _attach_pytree_op_params(
        self,
        trace: Trace,
        tree: EquinoxModuleTree | NnxModuleTree,
        captures: Sequence[JaxRuntimeCapture],
    ) -> None:
        """Attach JAX module parameters to primitive ops that consume them.

        Parameters
        ----------
        trace
            Trace with finalized JAX op labels.
        tree
            Discovered Equinox or Flax NNX module tree.
        captures
            Interpreted JAX equation captures in event order.

        Returns
        -------
        None
            Op parameter fields and reverse parameter usage lists are updated.
        """

        equation_labels = [
            label
            for label in trace._raw_graph_ws.raw_layer_labels_list
            if not trace._raw_graph_ws.raw_layer_dict[label].is_input
        ]
        usage_membership: dict[int, tuple[set[str], set[str], set[str]]] = {}
        for label, capture in zip(equation_labels, captures, strict=False):
            op_log = trace._raw_graph_ws.raw_layer_dict[label]
            param_addresses: list[str] = []
            for value in capture.input_values:
                address = tree.param_address_by_value_id.get(id(value))
                if address is not None and address not in param_addresses:
                    param_addresses.append(address)
            if not param_addresses:
                continue
            params = [trace._param_logs[address] for address in param_addresses]
            op_log._param_barcodes = [param.barcode for param in params]
            op_log._param_logs = params
            op_log.param_shapes = [param.shape for param in params]
            op_log.num_params = sum(param.num_params for param in params)
            op_log.num_params_trainable = sum(
                param.num_params for param in params if param.is_trainable
            )
            op_log.num_params_frozen = op_log.num_params - op_log.num_params_trainable
            op_log._cost_param_memory = sum(int(param.param_memory) for param in params)
            # Set-memoized usage cross-links (the bare list scans were
            # O(m^2) for a param consumed by m ops).
            _attach_param_usage(trace, op_log, usage_membership)
        trace._num_layers_with_params = len(
            {op.layer_label for op in trace._layer_list if op.has_params}
        )
        # Parameter-identity basis (see the identical note in
        # ``_jax_recurrence_assignments``'s sibling counter block above):
        # a per-layer-label dedup sum still double-counts a param shared
        # across more than one layer.
        _stamp_param_totals(trace, trace._param_logs)

    def _attach_pytree_module_logs(
        self,
        trace: Trace,
        tree: EquinoxModuleTree | NnxModuleTree,
    ) -> None:
        """Build public module logs for a JAX pytree-module trace.

        Parameters
        ----------
        trace
            Trace receiving module accessors.
        tree
            Discovered Equinox or Flax NNX module tree. Module logs are built
            from the decoded op stacks, then the root frame of root-only ops
            leaves the public op fields, as on torch.
        """

        mbd = trace._module_capture_ws.module_build_data = _init_module_hierarchy_data()
        trace._module_capture_ws.module_forward_args = dict(tree.forward_args_by_call)
        trace._module_capture_ws.module_metadata = tree.metadata
        for address, metadata in tree.metadata.items():
            if address not in mbd["addresses"]:
                mbd["addresses"].append(address)
            mbd["module_types"][address] = str(metadata.get("class_name", ""))
            mbd["module_training_modes"][address] = False
            mbd["module_num_calls"][address] = max(1, tree.call_counts.get(address, 1))
            for child_address in metadata.get("address_children", []):
                if child_address not in mbd["module_children"][address]:
                    mbd["module_children"][address].append(child_address)
            if address != "self" and "." not in address:
                mbd["top_level_modules"].append(address)

        for param in trace._param_logs:
            owner = param.module_address
            mbd["module_nparams"][owner] += param.num_params
            if param.is_trainable:
                mbd["module_nparams_trainable"][owner] += param.num_params
            else:
                mbd["module_nparams_frozen"][owner] += param.num_params

        self._populate_pytree_module_build_data(trace)
        return_probe.build_module_logs_with_returns(trace, tree, _build_module_logs)
        strip_op_root_frames(trace._layer_list)

    def _populate_pytree_module_build_data(self, trace: Trace) -> None:
        """Populate module hierarchy side channels from attributed JAX ops.

        Parameters
        ----------
        trace
            Trace whose finalized ops carry module tuples.

        Returns
        -------
        None
            ``trace._module_capture_ws.module_build_data`` is updated in place.
        """

        mbd = trace._module_capture_ws.module_build_data
        seen_layers: dict[str, set[str]] = defaultdict(set)
        seen_pass_layers: dict[str, set[str]] = defaultdict(set)
        seen_module_ops: set[str] = set()
        seen_top_level_ops: set[str] = set()
        seen_pass_children: dict[str, set[str]] = defaultdict(set)
        seen_addresses = set(mbd["addresses"])

        for op_log in trace._layer_list:
            normalized_calls = normalize_op_module_calls(op_log.modules)
            refuse_rootless_stack(op_log, normalized_calls, nested_root_dropped=True)
            op_log.modules = [f"{address}:{call_index}" for address, call_index in normalized_calls]
            op_log.module = op_log.modules[-1] if op_log.modules else None
            # ``module_call_stack`` carries the same containment fact as ``modules`` and
            # gets the same raw-tuple -> "address:N" normalization (torch: ``postprocess/
            # labeling.py``); raw tuples would stringify as ``"('fc1', 1)"`` reprs the
            # ``module_attribution`` invariant cannot resolve. An op decoded with no scope
            # at all is refused (the decoder drops the root only before a submodule).
            module_call_stack_calls = normalize_op_module_calls(op_log.module_context.call_stack)
            op_log._module_context_call_stack = [
                f"{address}:{call_index}" for address, call_index in module_call_stack_calls
            ]
            parent_call_label = "self:1" if "self" in seen_addresses else None
            for module_index, (address, call_index) in enumerate(normalized_calls):
                call_label = f"{address}:{call_index}"
                # Torch parity (same population as
                # ``backends/_finalize.py::populate_object_module_build_data``):
                # each ModuleCall's own ancestor chain, read by
                # ``_build_module_logs`` to build ``ModuleCall.module_call_stack``.
                # Missing here, every JAX ModuleCall's stack stayed empty and
                # tripped the ``module_hierarchy`` invariant's ancestor-prefix
                # check for any module nested more than one level deep.
                if call_label not in mbd["module_call_stacks"]:
                    mbd["module_call_stacks"][call_label] = [
                        f"{ancestor_address}:{ancestor_call_index}"
                        for ancestor_address, ancestor_call_index in normalized_calls[:module_index]
                        if ancestor_address != "self"
                    ]
                if mbd["module_num_calls"][address] < call_index:
                    mbd["module_num_calls"][address] = call_index
                mbd["module_num_tensors"][address] += 1
                mbd["module_call_index_tensors"][call_label] += 1
                if op_log.layer_label not in seen_layers[address]:
                    seen_layers[address].add(op_log.layer_label)
                    mbd["module_layers"][address].append(op_log.layer_label)
                if op_log.label not in seen_pass_layers[call_label]:
                    seen_pass_layers[call_label].add(op_log.label)
                    mbd["module_pass_layers"][call_label].append(op_log.label)
                if address not in seen_addresses:
                    seen_addresses.add(address)
                    mbd["addresses"].append(address)
                if call_label not in seen_module_ops:
                    seen_module_ops.add(call_label)
                    mbd["module_ops"].append(call_label)
                if module_index == 0:
                    if call_label not in seen_top_level_ops:
                        seen_top_level_ops.add(call_label)
                        mbd["top_level_module_ops"].append(call_label)
                    if address != "self" and address not in mbd["top_level_modules"]:
                        mbd["top_level_modules"].append(address)
                if parent_call_label is not None and parent_call_label != call_label:
                    if call_label not in seen_pass_children[parent_call_label]:
                        seen_pass_children[parent_call_label].add(call_label)
                        mbd["module_pass_children"][parent_call_label].append(call_label)
                parent_call_label = call_label

    def _jax_recurrence_assignments(self, trace: Trace) -> dict[str, RecurrenceAssignment]:
        """Return recurrence assignments for a materialized JAX raw trace.

        Parameters
        ----------
        trace
            Trace containing materialized raw JAX ops.

        Returns
        -------
        dict[str, RecurrenceAssignment]
            Recurrence assignments keyed by raw label.
        """

        site_keys = self._jax_site_keys(trace)
        if not trace._options_recurrence_detection:
            return {
                label: self._jax_singleton_assignment(label, op_log, site_keys.get(label))
                for label, op_log in trace._raw_graph_ws.raw_layer_dict.items()
            }
        graph = self._build_jax_recurrence_grouping_graph(trace, site_keys)
        assignments = group_recurrent_nodes(graph)
        return {
            label: assignments.get(
                label, self._jax_singleton_assignment(label, op_log, site_keys.get(label))
            )
            for label, op_log in trace._raw_graph_ws.raw_layer_dict.items()
        }

    def _jax_site_keys(self, trace: Trace) -> dict[str, str]:
        """Mint ``site_key_v1`` per retained op via the JAX site dialect.

        The dialect (iteration-stripped containing source path as the site
        axis, iteration-qualified path as the call instance) lives in
        :mod:`._site_dialect`.
        """

        return jax_site_keys(trace)

    def _build_jax_recurrence_grouping_graph(
        self, trace: Trace, site_keys: Mapping[str, str] | None = None
    ) -> RecurrenceGroupingGraph:
        """Build the neutral recurrence graph from JAX materialized raw logs.

        Parameters
        ----------
        trace
            Trace containing materialized raw JAX ops.
        site_keys
            Pre-minted ``site_key_v1`` strings per retained raw label
            (:meth:`_jax_site_keys`).

        Returns
        -------
        RecurrenceGroupingGraph
            Backend-neutral graph with data edges only.
        """

        nodes: dict[str, RecurrenceNode] = {}
        raw_labels = tuple(trace._raw_graph_ws.raw_layer_labels_list)
        raw_label_set = set(raw_labels)
        data_parents_by_label = {
            label: _ordered_jax_data_parent_labels(op_log, raw_label_set)
            for label, op_log in trace._raw_graph_ws.raw_layer_dict.items()
        }
        data_children_by_label: dict[str, list[str]] = {label: [] for label in raw_labels}
        for label, parents in data_parents_by_label.items():
            for parent in parents:
                data_children_by_label[parent].append(label)

        eligible_labels: list[str] = []
        source_labels: list[str] = []
        for label in raw_labels:
            op_log = trace._raw_graph_ws.raw_layer_dict[label]
            pruned = bool(record_member(op_log, "is_orphan", False))
            retain = not pruned
            if retain:
                eligible_labels.append(label)
            if _is_jax_recurrence_source(op_log, data_parents_by_label[label]):
                source_labels.append(label)
            nodes[label] = RecurrenceNode(
                label=label,
                raw_order=op_log.raw_index,
                equivalence_key=op_log.graph.equivalence_class,
                equivalent_labels=tuple(
                    member for member in op_log.graph.equivalent_ops if member in raw_label_set
                ),
                data_parents=data_parents_by_label[label],
                data_children=tuple(data_children_by_label[label]),
                layer_label=op_log._layer_label_raw,
                recurrent_labels=(label,),
                uses_params=bool(op_log.has_params),
                func_name=op_log.func_name,
                param_barcodes=tuple(op_log._param_barcodes),
                retain=retain,
                pruned=pruned,
                output_slot=(
                    getattr(op_log, "multi_output_index", None)
                    if getattr(op_log, "in_multi_output", False)
                    else None
                ),
                site_key=(site_keys or {}).get(label),
            )

        return RecurrenceGroupingGraph(
            nodes=nodes,
            raw_labels=raw_labels,
            source_labels=tuple(source_labels),
            eligible_labels=tuple(eligible_labels),
        )

    def _jax_singleton_assignment(
        self, label: str, op_log: Any, site_key: str | None = None
    ) -> RecurrenceAssignment:
        """Return a singleton recurrence assignment for one JAX op.

        Parameters
        ----------
        label
            Raw operation label.
        op_log
            Materialized operation log.
        site_key
            Pre-minted ``site_key_v1`` string for this op, when retained.

        Returns
        -------
        RecurrenceAssignment
            Single-pass assignment preserving the op's current key.
        """

        return RecurrenceAssignment(
            layer_label=label,
            recurrent_labels=(label,),
            pass_index=1,
            num_passes=1,
            equivalence_key=op_log.graph.equivalence_class or label,
            site_key=site_key,
        )

    def _relabel_jax_graph_edges(self, trace: Trace, raw_to_final: Mapping[str, str]) -> None:
        """Replace raw graph edge labels with final op labels.

        Parameters
        ----------
        trace
            Trace containing materialized raw JAX ops.
        raw_to_final
            Mapping from raw op labels to final labels -- a referenced
            single-pass layer's bare ``layer_label``, a multi-pass layer's
            pass-qualified ``label`` (torch parity; see ``_finish_trace``).

        Returns
        -------
        None
            Parent, child, lineage-set, parent-position, and edge-use labels
            are updated in place.

        Notes
        -----
        The lineage sets (``input_ancestors``, ``output_descendants``,
        ``root_ancestors``, ``internal_source_ancestors``,
        ``internal_source_parents``) are seeded with raw labels at capture
        time and during the pre-relabel input/output depth flood, exactly
        like ``parents``/``children``. Omitting them here left raw labels
        (e.g. ``input_1_1_raw``) surviving postprocessing and tripping the
        ``graph_ordering`` metadata invariant on every capture whose depth
        flood touched an op's own lineage set (effectively every capture,
        since ``_check_graph_ordering`` runs via ``validate_forward_pass`` ->
        ``check_metadata_invariants``).
        """

        for op_log in trace._raw_graph_ws.raw_layer_dict.values():
            op_log.parents = [
                raw_to_final.get(parent, parent) if isinstance(parent, str) else parent
                for parent in op_log.parents
            ]
            op_log.children = [
                raw_to_final.get(child, child) if isinstance(child, str) else child
                for child in op_log.children
            ]
            _relabel_jax_lineage_fields(op_log, raw_to_final)
            op_log._call_parent_arg_positions = _relabel_jax_parent_arg_positions(
                op_log.call.parent_arg_positions,
                raw_to_final,
            )
            op_log._internal_set(
                "_edge_uses",
                tuple(_relabel_jax_edge_use(edge, raw_to_final) for edge in op_log._edge_uses),
            )

    def _normalize_input_args(self, input_args: object) -> list[Any]:
        """Normalize public input args to a positional list.

        Parameters
        ----------
        input_args
            User-supplied public input args.

        Returns
        -------
        list[Any]
            Positional argument list.
        """

        if isinstance(input_args, list):
            return input_args
        if isinstance(input_args, tuple):
            return list(input_args)
        return [input_args]

    def _tree_leaves_with_paths(self, tree: object) -> list[tuple[str, Any]]:
        """Return pytree leaves with dotted container paths.

        Parameters
        ----------
        tree
            Pytree to flatten.

        Returns
        -------
        list[tuple[str, Any]]
            ``(path, leaf)`` pairs.
        """

        import jax

        leaves_with_paths, _treedef = jax.tree_util.tree_flatten_with_path(tree)
        return [(_path_to_string(path), value) for path, value in leaves_with_paths]

    def _output_leaf_paths(self, output: object) -> tuple[tuple[object, ...], ...]:
        """Return flat output pytree paths.

        Parameters
        ----------
        output
            Direct callable output.

        Returns
        -------
        tuple[tuple[object, ...], ...]
            Flat output container paths.
        """

        import jax

        leaves_with_paths, _treedef = jax.tree_util.tree_flatten_with_path(output)
        return tuple(_path_to_components(path) for path, _value in leaves_with_paths)

    def _builtin_output_container_spec(self, output: object) -> ContainerSpec | None:
        """Return a reconstructable spec for builtin JAX output pytrees.

        Parameters
        ----------
        output
            Direct callable output.

        Returns
        -------
        ContainerSpec | None
            Full spec for builtin tuple/list/dict trees with tensor leaves, or
            ``None`` for scalar outputs, custom pytrees, and mixed non-tensor leaves.
        """

        spec = self._builtin_container_spec(output, is_root=True)
        if spec is _UNSUPPORTED_CONTAINER_SPEC:
            return None
        return cast(ContainerSpec | None, spec)

    def _builtin_container_spec(
        self,
        value: object,
        *,
        is_root: bool = False,
    ) -> ContainerSpec | None | object:
        """Build a spec for builtin tuple/list/dict containers only.

        Parameters
        ----------
        value
            Candidate container or leaf value.
        is_root
            Whether ``value`` is the top-level model output.

        Returns
        -------
        ContainerSpec | None | object
            Container spec, ``None`` for tensor leaves, or ``None`` for the
            root when an unsupported non-tensor leaf/custom node is seen.
        """

        if isinstance(value, tuple) and type(value) is tuple:
            tuple_child_specs: list[tuple[OutputPathComponent, ContainerSpec]] = []
            for index, child in enumerate(value):
                child_spec = self._builtin_container_spec(child)
                if child_spec is _UNSUPPORTED_CONTAINER_SPEC:
                    return _UNSUPPORTED_CONTAINER_SPEC
                if child_spec is not None:
                    tuple_child_specs.append((TupleIndex(index), cast(ContainerSpec, child_spec)))
            return ContainerSpec(
                kind="tuple",
                length=len(value),
                child_specs=tuple(tuple_child_specs),
            )
        if isinstance(value, list):
            list_child_specs: list[tuple[OutputPathComponent, ContainerSpec]] = []
            for index, child in enumerate(value):
                child_spec = self._builtin_container_spec(child)
                if child_spec is _UNSUPPORTED_CONTAINER_SPEC:
                    return _UNSUPPORTED_CONTAINER_SPEC
                if child_spec is not None:
                    list_child_specs.append((TupleIndex(index), cast(ContainerSpec, child_spec)))
            return ContainerSpec(
                kind="list",
                length=len(value),
                child_specs=tuple(list_child_specs),
            )
        if isinstance(value, dict) and type(value) is dict:
            dict_child_specs: list[tuple[OutputPathComponent, ContainerSpec]] = []
            keys = _jax_dict_keys(value)
            for key in keys:
                child_spec = self._builtin_container_spec(value[key])
                if child_spec is _UNSUPPORTED_CONTAINER_SPEC:
                    return _UNSUPPORTED_CONTAINER_SPEC
                if child_spec is not None:
                    dict_child_specs.append((DictKey(key), cast(ContainerSpec, child_spec)))
            return ContainerSpec(kind="dict", keys=keys, child_specs=tuple(dict_child_specs))
        if self.is_tensor(value):
            return None
        if is_root:
            return None
        return _UNSUPPORTED_CONTAINER_SPEC

    def _reject_unsupported_options(self, **options: Any) -> None:
        """Reject public trace options unsupported by the JAX backend.

        Parameters
        ----------
        **options
            Public option values.

        Returns
        -------
        None
            Returns when all options are supported.
        """

        if options["save_rng"]:
            raise BackendUnsupportedError(
                "The JAX backend requires explicit PRNG keys as params/input leaves; "
                "save_rng_states and torch-style RNG replay are unsupported."
            )
        reject_unsupported_trace_options(
            options,
            JAX_PREVIEW_TRACE_OPTION_POLICY,
            spec=get_backend_spec("jax"),
        )

    def _reject_extra_kwargs(self, kwargs: Mapping[str, Any]) -> None:
        """Reject unrecognized kwargs reaching the backend.

        Parameters
        ----------
        kwargs
            Extra keyword arguments.

        Returns
        -------
        None
            Returns when no extras are present.
        """

        reject_extra_trace_kwargs(
            dict(kwargs),
            JAX_EXTRA_KWARG_POLICY,
            spec=get_backend_spec("jax"),
        )


def _normalize_static_argnums(value: object, num_args: int) -> tuple[int, ...]:
    """Normalize declared JAX static positional argument indexes.

    Parameters
    ----------
    value
        Public ``jax_static_argnums`` value.
    num_args
        Number of positional arguments in the normalized call.

    Returns
    -------
    tuple[int, ...]
        Unique static argument indexes in ascending order.

    Raises
    ------
    ValueError
        If any static index is out of range.
    TypeError
        If the value is neither an integer nor a sequence of integers.
    """

    if _is_missing(value) or value is None:
        return ()
    raw_indexes: tuple[int, ...]
    if isinstance(value, int):
        raw_indexes = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        raw_indexes = tuple(value)
        if not all(isinstance(index, int) for index in raw_indexes):
            raise TypeError("jax_static_argnums must contain only integer indexes.")
    else:
        raise TypeError("jax_static_argnums must be an int, sequence of ints, or None.")
    normalized = tuple(sorted({index if index >= 0 else num_args + index for index in raw_indexes}))
    invalid = [index for index in normalized if index < 0 or index >= num_args]
    if invalid:
        raise ValueError(
            f"jax_static_argnums indexes out of range for {num_args} positional args: {invalid}."
        )
    return normalized


def _resolve_jax_module_identity_mode(
    value: object,
    module_tree: EquinoxModuleTree | NnxModuleTree | None,
) -> bool:
    """Return whether this JAX capture should use pytree-module attribution.

    Parameters
    ----------
    value
        Public ``module_identity_mode`` value after missing normalization.
    module_tree
        Discovered JAX module tree, if the root callable is supported.

    Returns
    -------
    bool
        True for ``pytree_module`` captures.

    Raises
    ------
    BackendUnsupportedError
        If ``pytree_module`` is requested for an unsupported root.
    ValueError
        If an unknown mode is requested.
    """

    if value not in {None, "function_root", "pytree_module"}:
        # Typed like the four sibling previews: an unknown identity mode is a
        # capability refusal, not a plain value error.
        raise BackendUnsupportedError(
            "JAX module_identity_mode must be None, 'function_root', or 'pytree_module'."
        )
    if value == "pytree_module" and module_tree is None:
        raise BackendUnsupportedError(
            "JAX module_identity_mode='pytree_module' requires an Equinox eqx.Module "
            "or Flax NNX nnx.Module root. Raw JAX callables use "
            "module_identity_mode='function_root'."
        )
    if value == "function_root":
        return False
    return module_tree is not None


@contextmanager
def _scoped_pytree_module_calls(
    tree: EquinoxModuleTree | NnxModuleTree,
) -> Iterator[None]:
    """Wrap one supported JAX module tree with TorchLens named scopes.

    Parameters
    ----------
    tree
        Discovered Equinox or Flax NNX module tree.

    Yields
    ------
    None
        Control while module-call wrappers are installed.
    """

    if isinstance(tree, EquinoxModuleTree):
        with scoped_equinox_module_calls(tree):
            yield
    else:
        with scoped_nnx_module_calls(tree):
            yield


def _raise_nnx_trace_context_error(
    tree: EquinoxModuleTree | NnxModuleTree,
    exc: Exception,
) -> None:
    """Raise a stable TorchLens error for unsupported NNX trace-context failures.

    Parameters
    ----------
    tree
        Active JAX module tree.
    exc
        Exception raised by Flax/JAX tracing.

    Returns
    -------
    None
        Returns when ``exc`` is not the NNX trace-context failure this helper owns.

    Raises
    ------
    BackendUnsupportedError
        If an NNX module-mode capture hits Flax trace-context mutation limits.
    """

    if not isinstance(tree, NnxModuleTree):
        return
    if type(exc).__name__ != "TraceContextError":
        return
    if not type(exc).__module__.startswith("flax"):
        return
    raise BackendUnsupportedError(
        "JAX pytree_module strict mode does not support Flax NNX state rebinding "
        "or lifted transforms inside attributed modules. Move the mutation/transform "
        "outside the module, capture a raw function_root callable, or wait for widened "
        "NNX attribution support."
    ) from exc


def _derive_and_run(
    model: Callable[..., Any],
    args: Sequence[Any],
    static_argnums: tuple[int, ...],
    module_tree: EquinoxModuleTree | NnxModuleTree | None,
) -> tuple[Any, tuple[Any, ...], Any]:
    """Derive the closed jaxpr and run the callable once.

    Parameters
    ----------
    model
        JAX-compatible callable following ``fn(params, *inputs)``.
    args
        Normalized positional arguments.
    static_argnums
        Positional indexes declared static for ``jax.make_jaxpr``.
    module_tree
        The attributed module tree for ``pytree_module`` capture, or ``None``
        for a raw function root.

    Returns
    -------
    tuple[Any, tuple[Any, ...], Any]
        The closed jaxpr, its readable closure constants (empty under
        ``pytree_module``) and the eager output.
    """

    if module_tree is None:
        closed_jaxpr = derive_closed_jaxpr(model, args, static_argnums)
        closure_consts = _readable_closure_consts(closed_jaxpr)
        return closed_jaxpr, closure_consts, model(*args)
    try:
        with _scoped_pytree_module_calls(module_tree):
            extended = return_probe.extend(model, module_tree)
            closed_jaxpr = return_probe.strip(
                derive_closed_jaxpr(extended, args, static_argnums), module_tree
            )
    except Exception as exc:
        _raise_nnx_trace_context_error(module_tree, exc)
        raise
    reject_attributed_module_strict_control_flow(closed_jaxpr)
    module_tree.call_counts.clear()
    module_tree.forward_args_by_call.clear()
    try:
        with _scoped_pytree_module_calls(module_tree):
            output = model(*args)
    except Exception as exc:
        _raise_nnx_trace_context_error(module_tree, exc)
        raise
    return closed_jaxpr, (), output


def _reject_transformed_callable(model: Callable[..., Any]) -> None:
    """Reject root JAX transformed callables before jaxpr capture.

    Parameters
    ----------
    model
        User-supplied callable passed to ``tl.trace``.

    Returns
    -------
    None
        Returns when the callable looks like a raw function.
    """

    cls_name = type(model).__name__
    cls_module = type(model).__module__
    if cls_name in {"PjitFunction", "PmapFunction"} or cls_module.startswith("jaxlib"):
        raise BackendUnsupportedError(
            "The JAX backend does not accept a transformed callable as the root model "
            "(for example jax.jit). Pass the raw function to tl.trace(..., backend='jax') "
            "and let TorchLens derive its own closed jaxpr."
        )
    try:
        closure = inspect.getclosurevars(model)
    except TypeError:
        return
    wrapped = closure.nonlocals.get("fun")
    wrapped_module = getattr(wrapped, "__module__", "")
    wrapped_qualname = getattr(wrapped, "__qualname__", "")
    if wrapped_module == "jax._src.api" and wrapped_qualname.startswith(
        ("grad.", "vmap.", "jacfwd.", "jacrev.")
    ):
        transform_name = wrapped_qualname.split(".", maxsplit=1)[0]
        raise BackendUnsupportedError(
            "The JAX backend does not accept a transformed callable as the root model "
            f"(detected jax.{transform_name}). Pass the raw function to "
            "tl.trace(..., backend='jax'); nested transform support is a follow-up."
        )


def _reject_tracer_inputs(args: Sequence[Any]) -> None:
    """Reject root capture calls executed inside JAX transforms.

    Parameters
    ----------
    args
        Normalized public positional arguments.

    Returns
    -------
    None
        Returns when no argument leaf is a JAX tracer.
    """

    import jax

    leaves, _treedef = jax.tree.flatten(tuple(args))
    tracer_type = getattr(jax.core, "Tracer", None)
    if tracer_type is None:
        return
    if any(isinstance(leaf, tracer_type) for leaf in leaves):
        raise BackendUnsupportedError(
            "The JAX backend cannot run tl.trace from inside jax.jit, jax.vmap, "
            "or jax.grad. Call tl.trace(..., backend='jax') outside the transform on "
            "concrete params/input leaves."
        )


def _normalize_input_grad_argnums(value: Sequence[int], num_inputs: int) -> tuple[int, ...]:
    """Normalize input-relative JAX gradient argnums.

    Parameters
    ----------
    value
        Input-relative gradient argnums.
    num_inputs
        Number of positional inputs after params.

    Returns
    -------
    tuple[int, ...]
        Unique non-negative input-relative indexes.
    """

    normalized = tuple(sorted({index if index >= 0 else num_inputs + index for index in value}))
    invalid = [index for index in normalized if index < 0 or index >= num_inputs]
    if invalid:
        raise ValueError(
            f"input_grad_argnums indexes out of range for {num_inputs} inputs: {invalid}."
        )
    return normalized


def _assert_same_treedef(left: Any, right: Any, *, label: str) -> None:
    """Raise if two pytrees do not have the same structure.

    Parameters
    ----------
    left
        First pytree.
    right
        Second pytree.
    label
        User-facing label for the first pytree.

    Returns
    -------
    None
        Returns when tree structures match.
    """

    import jax

    if jax.tree.structure(left) != jax.tree.structure(right):
        raise ValueError(f"{label} must have the same pytree structure as positional arg 0.")


def _differentiated_leaf_paths(
    args: Sequence[Any], differentiated_argnums: Sequence[int]
) -> tuple[str, ...]:
    """Return stable differentiated leaf paths.

    Parameters
    ----------
    args
        Positional callable arguments.
    differentiated_argnums
        Backend positional argnums passed to JAX AD.

    Returns
    -------
    tuple[str, ...]
        Stable leaf paths included in the gradient fingerprint.
    """

    import jax

    paths: list[str] = []
    for argnum in differentiated_argnums:
        leaves, _treedef = jax.tree_util.tree_flatten_with_path(args[argnum])
        for path, _value in leaves:
            paths.append(_grad_path_for_argnum(argnum, _path_to_string(path)))
    return tuple(paths)


def _gradient_fingerprint(
    *,
    closed_jaxpr: Any,
    args: Sequence[Any],
    static_argnums: Sequence[int],
    differentiated_paths: Sequence[str],
    loss_fn: Callable[[Any], Any] | None,
) -> str:
    """Build a fail-closed fingerprint for JAX derived-gradient validation.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr for the declared forward call.
    args
        Positional callable arguments.
    static_argnums
        Declared static positional argument indexes.
    differentiated_paths
        Stable differentiated leaf paths.
    loss_fn
        Optional loss function.

    Returns
    -------
    str
        SHA-256 digest of structural gradient-run metadata.
    """

    import jax

    static_values = {str(index): repr(args[index]) for index in static_argnums if index < len(args)}
    dynamic_args = tuple(arg for index, arg in enumerate(args) if index not in set(static_argnums))
    payload = {
        "closed_jaxpr": repr(closed_jaxpr),
        "consts": tuple(repr(value) for value in getattr(closed_jaxpr, "consts", ())),
        "statics": static_values,
        "treedef": repr(jax.tree.structure(dynamic_args)),
        "config": _jax_config_fingerprint(),
        "differentiated_paths": tuple(differentiated_paths),
        "loss_fn": _callable_identity(loss_fn),
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jax_config_fingerprint() -> dict[str, str]:
    """Return JAX configuration fields relevant to derived gradients.

    Returns
    -------
    dict[str, str]
        JSON-ready JAX config snapshot.
    """

    import jax

    names = (
        "jax_enable_x64",
        "jax_numpy_dtype_promotion",
        "jax_default_matmul_precision",
        "jax_default_prng_impl",
    )
    return {name: repr(getattr(jax.config, name, None)) for name in names}


def _is_scalar_jax_value(value: Any) -> bool:
    """Return whether ``value`` is scalar-shaped.

    Parameters
    ----------
    value
        Candidate JAX value.

    Returns
    -------
    bool
        True when the value has shape ``()``.
    """

    import jax.numpy as jnp

    return tuple(jnp.asarray(value).shape) == ()


def _block_until_ready_tree(tree: Any) -> Any:
    """Block on all JAX array leaves and return ``tree``.

    Parameters
    ----------
    tree
        Pytree possibly containing JAX arrays.

    Returns
    -------
    Any
        The same tree after async work is complete.
    """

    import jax

    def block(value: Any) -> Any:
        """Block one JAX value when supported.

        Parameters
        ----------
        value
            Candidate JAX value.

        Returns
        -------
        Any
            Ready value, or the original value when no async barrier exists.
        """

        block_until_ready = getattr(value, "block_until_ready", None)
        if callable(block_until_ready):
            return block_until_ready()
        return value

    return jax.tree.map(block, tree)


def _records_for_grad_tree(
    *,
    grad_tree: Any,
    argnum: int,
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records for one differentiated arg tree.

    Parameters
    ----------
    grad_tree
        Gradient pytree returned by JAX AD.
    argnum
        Backend positional argnum for this gradient tree.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable leaf path.
    """

    import jax

    records: dict[str, DerivedGradRecord] = {}
    leaves, _treedef = jax.tree_util.tree_flatten_with_path(grad_tree)
    for path, grad in leaves:
        local_path = _path_to_string(path)
        full_path = _grad_path_for_argnum(argnum, local_path)
        source = "params" if argnum == 0 else "inputs"
        records[full_path] = DerivedGradRecord(
            path=full_path,
            source=source,
            argnum=argnum,
            input_argnum=None if argnum == 0 else argnum - 1,
            aval=f"ShapedArray({tuple(getattr(grad, 'shape', ()))}, {getattr(grad, 'dtype', None)})",
            dtype_ref=DtypeRef.from_value(getattr(grad, "dtype", None)),
            grad=grad,
            provenance=provenance,
        )
    return records


def _jax_intermediate_tap_specs(trace: Trace) -> tuple[_JaxIntermediateTapSpec, ...]:
    """Return saved, inexact JAX op outputs eligible for zero-tap gradients.

    Parameters
    ----------
    trace
        Finalized JAX trace with capture-index label maps.

    Returns
    -------
    tuple[_JaxIntermediateTapSpec, ...]
        Tap specs in deterministic capture/output order. Non-float/control
        outputs and unsaved ops are skipped.
    """

    specs: list[_JaxIntermediateTapSpec] = []
    capture_to_label = getattr(trace, "jax_capture_index_to_final_op_label", {})
    captures = tuple(getattr(trace, "jax_equation_captures", ()))
    for capture in sorted(captures, key=lambda item: item.index):
        label = capture_to_label.get(capture.index)
        if not isinstance(label, str) or label not in trace.op_labels:
            continue
        op = trace.ops[label]
        if not getattr(op, "has_saved_activation", False):
            continue
        if getattr(op, "is_input", False) or getattr(op, "is_output", False):
            continue
        for output_index, value in enumerate(capture.output_values):
            if not _is_inexact_jax_array(value):
                continue
            specs.append(
                _JaxIntermediateTapSpec(
                    op_label=label,
                    layer_label=op.layer_label,
                    capture_index=capture.index,
                    output_index=output_index,
                    value=value,
                    aval=(
                        f"ShapedArray({tuple(getattr(value, 'shape', ()))}, "
                        f"{getattr(value, 'dtype', None)})"
                    ),
                    dtype_ref=DtypeRef.from_value(getattr(value, "dtype", None)),
                )
            )
    return tuple(specs)


def _is_inexact_jax_array(value: Any) -> bool:
    """Return whether ``value`` is a floating JAX array.

    Parameters
    ----------
    value
        Candidate boundary value.

    Returns
    -------
    bool
        True when the boundary can be differentiated by JAX AD.
    """

    import jax
    import jax.numpy as jnp

    return isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.floating)


def _raw_output_from_replay(
    *,
    closed_jaxpr: Any,
    flat_args: Sequence[Any],
    output_treedef: Any,
    jax_control_flow: str,
    jax_max_control_flow_unroll: int,
) -> Any:
    """Replay the closed jaxpr and rebuild the public raw output tree.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr to replay.
    flat_args
        Dynamic argument leaves.
    output_treedef
        Output pytree definition from the captured raw output.
    jax_control_flow
        Control-flow policy used by normal capture.
    jax_max_control_flow_unroll
        Control-flow unroll cap used by normal capture.

    Returns
    -------
    Any
        Rebuilt raw output tree.
    """

    import jax

    result = interpret_closed_jaxpr_with_inlining(
        closed_jaxpr,
        flat_args,
        jax_control_flow=jax_control_flow,
        jax_max_control_flow_unroll=jax_max_control_flow_unroll,
    )
    return jax.tree.unflatten(output_treedef, result.outputs)


def _jax_intermediate_oracle_passes(
    *,
    spec: _JaxIntermediateTapSpec,
    tap_index: int,
    producer_grad: Any,
    zero_taps: tuple[Any, ...],
    loss_from_replacement: Callable[[_JaxIntermediateTapSpec, Any], Any],
) -> bool:
    """Return whether independent VJP and finite difference confirm a tap grad.

    Parameters
    ----------
    spec
        Boundary tap spec under validation.
    tap_index
        Position of ``spec`` in the flat all-tap tuple.
    producer_grad
        Gradient from the single all-tap producer pass.
    zero_taps
        All-zero tap tuple aligned with the producer specs.
    loss_from_replacement
        Scalar loss function that replaces one boundary value.

    Returns
    -------
    bool
        True only when the per-tap VJP and perturbation checks agree.
    """

    import jax
    import jax.numpy as jnp

    if tap_index < 0 or tap_index >= len(zero_taps):
        return False

    def replacement_loss(replacement: Any) -> Any:
        """Return scalar loss with one boundary output replaced.

        Parameters
        ----------
        replacement
            Candidate boundary value.

        Returns
        -------
        Any
            Scalar loss.
        """

        return loss_from_replacement(spec, replacement)

    loss, pullback = jax.vjp(replacement_loss, spec.value)
    if not _is_scalar_jax_value(loss):
        return False
    (reference_grad,) = pullback(jnp.ones_like(loss))
    if not _values_close(producer_grad, reference_grad):
        return False
    return _finite_difference_directional_check(
        value=spec.value,
        grad=producer_grad,
        scalar_loss=replacement_loss,
    )


def _finite_difference_directional_check(
    *,
    value: Any,
    grad: Any,
    scalar_loss: Callable[[Any], Any],
) -> bool:
    """Check a gradient with a central finite difference along ``sign(grad)``.

    The comparison is between the observed central loss delta and the
    first-order prediction ``sum(grad * perturbation)``, judged against a
    WRITTEN-DOWN error model: the dtype replay band
    (``float_replay_tolerances``) widened by the central-difference
    truncation term (``~step**2`` relative, 8x headroom) plus the
    loss-evaluation roundoff (``8 * eps * max(|L+|, |L-|)`` absolute — the
    two loss evaluations are the quantities actually differenced). The
    former fixed ``rtol=5e-2, atol=5e-3`` pair was dtype-blind and its
    absolute floor blessed any tap whose true directional derivative sat
    below 5e-3.

    Parameters
    ----------
    value
        Boundary tap value.
    grad
        Candidate gradient for ``value``.
    scalar_loss
        Scalar loss as a function of one tap value.

    Returns
    -------
    bool
        True when the finite-difference loss delta agrees with the gradient
        prediction within the derived error model.
    """

    import jax.numpy as jnp

    direction = jnp.sign(grad)
    if not bool(jnp.any(direction)):
        direction = jnp.ones_like(grad)
    # Dtype-derived step: a flat 1e-4 for every non-fp32 dtype would sit BELOW
    # fp16/bf16 spacing at unit scale, so ``value +/- eps`` would round back to
    # ``value`` and the second oracle would be vacuous there. Wide dtypes keep
    # their tuned steps; storage-rounding dtypes take cbrt(eps) (the
    # central-difference optimum: ~0.1 fp16, ~0.2 bf16), which survives their
    # spacing.
    finfo = jnp.finfo(value.dtype)
    eps32 = float(jnp.finfo(jnp.float32).eps)
    if value.dtype == jnp.float32:
        step = 1e-2
    elif float(finfo.eps) <= eps32:
        step = 1e-4
    else:
        step = float(finfo.eps) ** (1.0 / 3.0)
    # Per-element step scaling: float spacing is RELATIVE while a
    # scalar step is ABSOLUTE, so an element of magnitude ~1e6 fp32 has
    # spacing far above the raw step and stays frozen in both probes. Scale
    # each element's step by max(1, |value|) so every intended probe moves.
    one = jnp.asarray(1.0, dtype=value.dtype)
    eps = jnp.asarray(step, dtype=value.dtype) * jnp.maximum(one, jnp.abs(value))
    perturbation = eps * direction
    plus_input = value + perturbation
    minus_input = value - perturbation
    intended = direction != 0
    frozen = (intended & (plus_input == value)) | (intended & (minus_input == value))
    if bool(jnp.any(frozen)):
        # An intended probe never moved its element (spacing underflow even
        # after magnitude scaling — non-finite or extreme values): the
        # observed delta would silently omit that element's contribution
        # while the prediction includes it. Fail closed rather than certify
        # a partially-probed gradient (the former all(...) and all(...)
        # gate passed MIXED freezes).
        return False
    plus = scalar_loss(plus_input)
    minus = scalar_loss(minus_input)
    observed = (plus - minus) / jnp.asarray(2, dtype=value.dtype)
    expected = jnp.sum(grad * perturbation)
    rtol_repl, atol_repl = float_replay_tolerances(finfo)
    rtol_fd = max(rtol_repl, 8.0 * step * step)
    roundoff = 8.0 * float(finfo.eps) * float(jnp.maximum(jnp.abs(plus), jnp.abs(minus)))
    scale = float(jnp.maximum(jnp.abs(observed), jnp.abs(expected)))
    tolerance = rtol_fd * scale + roundoff + atol_repl
    return bool(jnp.abs(observed - expected) <= tolerance)


def _experimental_per_op_boundary_vjp_oracle(
    *,
    boundaries: Mapping[str, tuple[Any, Callable[[Any], Any]]],
    loss_fn: Callable[[Any], Any] | None,
    max_save_set: int = 8,
) -> _ExperimentalBoundaryVJPResult:
    """Run a capped private per-boundary JAX VJP oracle for tests.

    Parameters
    ----------
    boundaries
        Mapping from caller-owned labels to ``(boundary_value, suffix_fn)``
        pairs. ``suffix_fn`` must rebuild the model output from a replacement
        boundary value.
    loss_fn
        Optional scalar loss function applied to each suffix output.
    max_save_set
        Hard upper bound on the number of boundaries to process.

    Returns
    -------
    _ExperimentalBoundaryVJPResult
        Exact records plus skip reasons. This helper is private and does not
        populate any public TorchLens trace surface.
    """

    import jax
    import jax.numpy as jnp

    if max_save_set < 1:
        raise ValueError("max_save_set must be >= 1")
    if len(boundaries) > max_save_set:
        raise ValueError(
            f"JAX private boundary VJP oracle is capped at {max_save_set} save sites; "
            f"got {len(boundaries)}."
        )
    records: dict[str, _ExperimentalBoundaryVJPRecord] = {}
    skipped: dict[str, str] = {}
    for label, (boundary_value, suffix_fn) in boundaries.items():
        try:

            def scalar_loss(replacement: Any, *, suffix_fn: Any = suffix_fn) -> Any:
                """Return scalar loss for one replacement boundary value.

                Parameters
                ----------
                replacement
                    Replacement boundary value.
                suffix_fn
                    Suffix function bound at definition time (loop-safe).

                Returns
                -------
                Any
                    Scalar loss candidate.
                """

                output = suffix_fn(replacement)
                return output if loss_fn is None else loss_fn(output)

            loss, pullback = jax.vjp(scalar_loss, boundary_value)
            if not _is_scalar_jax_value(loss):
                skipped[label] = "non_scalar_loss"
                continue
            (grad,) = pullback(jnp.ones_like(loss))
            records[label] = _ExperimentalBoundaryVJPRecord(
                label=label,
                grad=grad,
                provenance={
                    "backend": "jax",
                    "kind": "private_boundary_vjp_oracle",
                    "mechanism": "per_op_boundary_vjp",
                    "status": "exact",
                    "max_save_set": max_save_set,
                },
            )
        except Exception as exc:  # pragma: no cover - exercised through skip assertions.
            skipped[label] = f"error:{type(exc).__name__}"
    return _ExperimentalBoundaryVJPResult(records=records, skipped=skipped)


def _grad_path_for_argnum(argnum: int, local_path: str) -> str:
    """Return a public derived-gradient path for one leaf.

    Parameters
    ----------
    argnum
        Backend positional argnum.
    local_path
        Dotted pytree path within that argument.

    Returns
    -------
    str
        Public stable path.
    """

    suffix = "" if local_path == "root" else f".{local_path}"
    if argnum == 0:
        return f"params{suffix}"
    return f"inputs.{argnum - 1}{suffix}"


def _jax_equation_ops(trace: Trace) -> tuple[Any, ...]:
    """Return materialized operation logs that correspond to JAX replay captures.

    Parameters
    ----------
    trace
        Trace produced by the JAX backend.

    Returns
    -------
    tuple[Any, ...]
        Operation logs with JAX replay-kind annotations, in execution order.
    """

    return tuple(
        op
        for op in trace._layer_list
        if isinstance(getattr(op, "annotations", None), Mapping)
        and op.annotations.get("jax_capture_kind") in ALL_JAX_EQUATION_KINDS
    )


def _jax_region_ops(trace: Trace) -> tuple[Any, ...]:
    """Return materialized operation logs that correspond to JAX regions.

    Parameters
    ----------
    trace
        Trace produced by the JAX backend.

    Returns
    -------
    tuple[Any, ...]
        Region boundary and projection operation logs in execution order.
    """

    return tuple(
        op
        for op in trace._layer_list
        if isinstance(getattr(op, "annotations", None), Mapping)
        and op.annotations.get("jax_capture_kind") in {"region", "region_output"}
    )


def _jax_op_capture_kind(op: Any) -> str:
    """Return the replay kind annotation from a materialized JAX op.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    str
        JAX replay kind stored on the operation.
    """

    annotations = getattr(op, "annotations", {})
    kind = annotations.get("jax_capture_kind") if isinstance(annotations, Mapping) else None
    if not isinstance(kind, str):
        raise ValueError("JAX equation op is missing a string replay kind.")
    if kind not in ALL_JAX_EQUATION_KINDS:
        raise ValueError(f"JAX equation op has unknown replay kind: {kind!r}.")
    return kind


class _RegionSeam(NamedTuple):
    """One JAX region boundary with its projections, runtime and materialized.

    Parameters
    ----------
    boundary
        Runtime boundary capture.
    boundary_op
        Materialized boundary op.
    projections
        Runtime projection captures for the boundary outputs.
    projection_ops
        Materialized projection ops aligned with ``projections``.
    """

    boundary: JaxRegionCapture
    boundary_op: Any
    projections: tuple[JaxRegionCapture, ...]
    projection_ops: tuple[Any, ...]


def _validate_region_boundary(
    seam: _RegionSeam,
    *,
    ops_by_label: Mapping[str, Any],
    hidden_outputs_by_label: Mapping[str, tuple[Any, ...]],
) -> bool:
    """Validate one JAX region boundary and its projection seams.

    Parameters
    ----------
    seam
        The boundary, its op and the aligned projections.
    ops_by_label
        Materialized operations keyed by raw and final labels.
    hidden_outputs_by_label
        Runtime-only replay outputs keyed by raw and final op labels.

    Returns
    -------
    bool
        True when boundary/projection metadata and seam perturbation pass.
    """

    import jax

    boundary, boundary_op, projections, projection_ops = seam
    if boundary.kind != "region":
        return False
    if _jax_region_capture_kind(boundary_op) != "region":
        return False
    annotations = getattr(boundary_op, "annotations", {})
    if not isinstance(annotations, Mapping):
        return False
    if annotations.get(REGION_REPLAY_CLASS_KEY) != REGION_REPLAY_CLASS:
        return False
    if annotations.get(REGION_REPLAY_PROVENANCE_KEY) != REGION_REPLAY_IMPORTER_PROVENANCE:
        return False
    metadata = annotations.get("jax_region_metadata")
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get("num_inputs") != len(boundary.input_values):
        return False
    if metadata.get("num_outputs") != len(boundary.output_values):
        return False
    if len(projections) != len(boundary.output_values):
        return False
    ordered_projections = tuple(sorted(projections, key=lambda item: item.output_index or -1))
    if tuple(projection.output_index for projection in ordered_projections) != tuple(
        range(len(boundary.output_values))
    ):
        return False
    inputs = _inputs_from_region_trace_graph(
        boundary,
        boundary_op,
        ops_by_label,
        hidden_outputs_by_label,
    )
    replayed_outputs = _bind_region(boundary, inputs)
    if len(replayed_outputs) != len(boundary.output_values):
        return False
    boundary_outputs = _saved_region_boundary_outputs(
        boundary_op,
        len(boundary.output_values),
        hidden_outputs_by_label=hidden_outputs_by_label,
    )
    if not jax.tree.all(jax.tree.map(_values_close, replayed_outputs, boundary_outputs)):
        return False
    saved_projection_outputs: list[Any] = []
    for projection, projection_op in zip(ordered_projections, projection_ops, strict=False):
        if _jax_region_capture_kind(projection_op) != "region_output":
            return False
        if projection.region_index != boundary.index or projection.output_index is None:
            return False
        if not _projection_has_boundary_parent(projection_op, boundary_op):
            return False
        saved_output = _saved_parent_input(projection_op, hidden_outputs_by_label)
        expected_output = replayed_outputs[projection.output_index]
        if not _values_close(saved_output, expected_output):
            return False
        saved_projection_outputs.append(saved_output)
    if not jax.tree.all(
        jax.tree.map(_values_close, tuple(saved_projection_outputs), replayed_outputs)
    ):
        return False
    return _region_seam_perturbation_changes_output(boundary, boundary_op, inputs, replayed_outputs)


def _jax_region_capture_kind(op: Any) -> str:
    """Return the region capture-kind annotation from a materialized op.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    str
        ``"region"`` or ``"region_output"``.
    """

    annotations = getattr(op, "annotations", {})
    kind = annotations.get("jax_capture_kind") if isinstance(annotations, Mapping) else None
    if kind not in {"region", "region_output"}:
        raise ValueError(f"JAX region op has unknown capture kind: {kind!r}.")
    return cast(str, kind)


def _inputs_from_region_trace_graph(
    capture: JaxRegionCapture,
    op: Any,
    ops_by_label: Mapping[str, Any],
    hidden_outputs_by_label: Mapping[str, tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Build region boundary inputs from recorded graph parents.

    Parameters
    ----------
    capture
        Runtime region boundary capture.
    op
        Materialized region boundary operation.
    ops_by_label
        Materialized operations keyed by raw and final op labels.
    hidden_outputs_by_label
        Runtime-only replay outputs keyed by raw and final op labels.

    Returns
    -------
    tuple[Any, ...]
        Boundary inputs with graph-parent positions reconstructed from saved
        parent payloads.
    """

    inputs = list(capture.input_values)
    graph_positions = _data_parent_arg_positions(op)
    parent_labels = _data_parent_labels(op)
    positioned_labels = {label for label in graph_positions.values() if isinstance(label, str)}
    if positioned_labels != parent_labels:
        raise ValueError("JAX region parent labels and parent_arg_positions disagree.")
    for position, parent_label in graph_positions.items():
        if not isinstance(position, int) or position < 0 or position >= len(inputs):
            raise ValueError(f"JAX region parent arg position {position!r} is invalid.")
        parent_op = ops_by_label[parent_label]
        inputs[position] = _saved_parent_input(parent_op, hidden_outputs_by_label, inputs[position])
    return tuple(inputs)


def _bind_region(capture: JaxRegionCapture, inputs: Sequence[Any]) -> tuple[Any, ...]:
    """Execute a JAX region boundary as a black-box primitive.

    Parameters
    ----------
    capture
        Runtime region boundary capture.
    inputs
        Boundary inputs to bind.

    Returns
    -------
    tuple[Any, ...]
        Boundary outputs.
    """

    if capture.primitive == "custom_vjp_call":
        from .jaxpr import _evaluate_closed_jaxpr_no_capture

        return _evaluate_closed_jaxpr_no_capture(capture.params["call_jaxpr"], tuple(inputs))
    result = capture.primitive_obj.bind(*tuple(inputs), **capture.params)
    return tuple(result if capture.primitive_obj.multiple_results else (result,))


def _projection_has_boundary_parent(projection_op: Any, boundary_op: Any) -> bool:
    """Return whether one projection has the boundary as a control parent.

    Parameters
    ----------
    projection_op
        Materialized region projection op.
    boundary_op
        Materialized region boundary op.

    Returns
    -------
    bool
        True when the projection records a control edge from the boundary.
    """

    # TORCH PARITY: ``projection_op``'s control-parent edge resolves through
    # the CONDITIONAL label map -- the boundary's bare ``layer_label`` when
    # it is single-pass (the common case; a region boundary is never
    # recurrence-grouped), its pass-qualified ``label`` only if multi-pass --
    # so matching only raw/always-qualified labels here permanently missed
    # every single-pass boundary and reported no control parent at all.
    boundary_labels = {
        label
        for label in (
            getattr(boundary_op, "_label_raw", None),
            getattr(boundary_op, "label", None),
            getattr(boundary_op, "layer_label", None)
            if getattr(boundary_op, "num_passes", 1) == 1
            else None,
        )
        if isinstance(label, str)
    }
    control_parents = _control_parent_labels(projection_op)
    return bool(boundary_labels & control_parents)


def _region_seam_perturbation_changes_output(
    capture: JaxRegionCapture,
    op: Any,
    inputs: tuple[Any, ...],
    saved_outputs: tuple[Any, ...],
) -> bool:
    """Return whether real region parent perturbations change boundary outputs.

    Parameters
    ----------
    capture
        Runtime region boundary capture.
    op
        Materialized region boundary operation.
    inputs
        Boundary inputs reconstructed from the trace graph.
    saved_outputs
        Saved boundary outputs.

    Returns
    -------
    bool
        True only when a real graph-parent input position can perturb the
        region output, avoiding the normal parent check's no-parent vacuity.
    """

    import jax

    graph_positions = _data_parent_arg_positions(op)
    if not graph_positions:
        return not inputs
    positions_by_parent: dict[str, list[int]] = {}
    for position, parent_label in graph_positions.items():
        positions_by_parent.setdefault(parent_label, []).append(position)
    for positions in positions_by_parent.values():
        first_position = positions[0]
        for candidate in _perturb_candidates(inputs[first_position]):
            perturbed_inputs = list(inputs)
            for position in positions:
                perturbed_inputs[position] = candidate
            perturbed_outputs = _bind_region(capture, perturbed_inputs)
            if not jax.tree.all(jax.tree.map(_values_close, perturbed_outputs, saved_outputs)):
                return True
    return False


def _inputs_from_trace_graph(
    capture: JaxEquationCapture,
    op: Any,
    ops_by_label: Mapping[str, Any],
    hidden_outputs_by_label: Mapping[str, tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Build replay inputs from the trace's recorded parent graph.

    Parameters
    ----------
    capture
        Captured JAX equation metadata.
    op
        Materialized TorchLens operation for the equation.
    ops_by_label
        Materialized operations keyed by raw and final op labels.
    hidden_outputs_by_label
        Runtime-only replay outputs keyed by raw and final op labels.

    Returns
    -------
    tuple[Any, ...]
        Replay inputs where graph parents come from saved parent payloads.
    """

    inputs = list(capture.input_values)
    graph_positions = _data_parent_arg_positions(op)
    parent_labels = _data_parent_labels(op)
    positioned_labels = {label for label in graph_positions.values() if isinstance(label, str)}
    if positioned_labels != parent_labels:
        raise ValueError("JAX trace data parent labels and parent_arg_positions disagree.")
    for position, parent_label in graph_positions.items():
        if not isinstance(position, int) or position < 0 or position >= len(inputs):
            raise ValueError(f"JAX trace parent arg position {position!r} is invalid.")
        parent_op = ops_by_label[parent_label]
        inputs[position] = _saved_parent_input(parent_op, hidden_outputs_by_label, inputs[position])
    return tuple(inputs)
