"""MLX implementation of the capture backend Protocol."""

from __future__ import annotations

import random
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, cast

import numpy as np

from torchlens.backends.registry import BackendUnsupportedError

from ... import _state
from ..._capture._arg_snapshot import split_non_tensor_args
from ..._capture.outcome import StopRequest, stamp_backend_finalized
from ..._data_classes._trace_state_io import fill_unset_declared_fields
from ..._data_classes.derived_grad import (
    DerivedGradAccessor,
    DerivedGradRecord,
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
    set_intermediate_derived_grads,
)
from ..._data_classes.param import Param, ParamAccessor
from ..._data_classes.trace import Trace
from ..._ir.capture_events import CaptureEvents
from ..._ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
    OpEvent,
    OutputRef,
    ParentEdge,
)
from ..._ir.intervention import FireResult, FunctionEventInput
from ..._ir.predicate import _DEFERRED_VALUE, RecordContext
from ..._ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ..._ir.selector_eval import first_selector_kind_outside
from ..._ir.semantics import BackendSemantics, CapturePolicy
from ..._ir.workspaces import RawGraphWorkspace
from ..._output_slot import returned, stopped
from ..._postprocess._materialize import materialize_from_events
from ..._postprocess._selective_save import apply_save_keep_set
from ..._quantities import Duration
from ..._selection.errors import LiveModeLabelError
from ..._trace_core.relation_views import freeze_trace_relation_views
from ..._validation_status import ValidationStatus, count_importer_region_annotations  # noqa: TC001
from ...backends.registry import (
    BackendName,
    get_backend_spec,
    require_capability_implementation,
)
from ...errors import TorchLensWarning
from ...recording._halt import StopSignal
from ...recording._types import CaptureSpec
from .._entry_refusals import refuse_if_bad_max_intermediate_grads
from .._finalize import (
    attach_function_root_module,
    attach_module_owned_op_params,
    attach_object_module_logs_with_returns,
    finalize_single_pass_trace,
    join_module_address as _join_module_address,
    mark_output_label,
    mirror_param_derived_grads,
    nearest_metadata_parent,
    normalize_op_module_calls,
    numel_from_shape as _numel,
    session_callable_identity as _callable_identity,
    value_nbytes as _nbytes,
)
from .._finalize_totals import _stamp_param_totals
from .._invariants import run_metadata_invariants
from .._options import MLX_PREVIEW_TRACE_OPTION_POLICY, reject_unsupported_trace_options
from . import capabilities, wrappers
from ._capture_disclosures import _disclose_compiled_attributes, _disclose_unfired_selectors
from ._capture_hooks import MLXHookSession, forward_hooks, retain_backward_state
from ._edit_replay import check_edit_grad_options, edited_replay_model
from ._grad_oracle import (
    _eval_mlx_tree,
    _has_mlx_array_leaf,
    _is_scalar_mlx_value,
    _mlx_intermediate_oracle_passes,
    _mlx_trace_intermediate_signatures,
    _mlx_trees_close,
    _MLXReplayCall,
    _records_for_mlx_grad_tree,
)
from ._param_tree import (
    _alias_to_primary,
    _iter_parameter_tree,
    _iter_trainable_parameter_tree,
)
from ._replay_status import (
    _mlx_detailed_counts,
    _mlx_metadata_failed_decision,
    _mlx_nothing_replayed_decision,
    _mlx_trace_decision,
    _mlx_unverified_count,
    _mlx_validation_source,
    _name_first_failure,
    kept_or_unavailable_status,
)
from ._retention import LIVE_DECIDABLE_KINDS, RetentionDeps, RetentionRun, start_retention
from ._source_frames import adopt_code_context_records, mlx_code_context
from ._tap_observers import (
    _is_float_mlx_array,
    _MLXIntermediateCandidate,
    _MLXIntermediateSignature,
    _MLXIntermediateTapObserver,
    _replace_mlx_array_leaves,
)
from .external_arrays import external_array_ids
from .interventions import _match_live_sites, selector_matches_capture_context
from .model_prep import (
    MLXModuleTree,
    cleanup_model_session,
    discover_mlx_module_tree,
    prepare_model_once,
    prepare_model_session,
)
from .settings import UNWRAPPED_OP_REASON, get_unwrapped_policy
from .tensor_store import MLXTensorLabelStore
from .validation import MLXOpCapture, build_capture_template
from .wrappers import is_mlx_wrapped, mlx_tap_observer, unwrap_mlx, wrap_mlx


@dataclass(frozen=True)
class MLXParameterCandidate:
    """One flattened MLX parameter candidate.

    Parameters
    ----------
    address
        Dotted parameter address.
    value
        MLX array value.
    owner
        Owning primary module address.
    """

    address: str
    value: Any
    owner: str


@dataclass(frozen=True)
class GradOptions:
    """MLX derived-gradient preview options.

    Parameters
    ----------
    params
        Optional MLX parameter tree to pass as explicit AD argument 0. When
        omitted for ``mlx.nn.Module`` models, ``model.parameters()`` is used.
    loss_fn
        Optional callable mapping raw function output to a scalar loss. Required
        unless the raw traced output is already scalar.
    input_grad_argnums
        Positional input argument indexes to differentiate in addition to params.
    intermediate_grads
        Whether to run the opt-in custom-VJP tap replay for exact op-level
        intermediate derived gradients.
    max_intermediate_grads
        Hard cap on saved op boundaries processed by the intermediate-gradient
        producer and oracle.
    """

    params: Any | None = None
    loss_fn: Callable[[Any], Any] | None = None
    input_grad_argnums: tuple[int, ...] = ()
    intermediate_grads: bool = False
    max_intermediate_grads: int = 64

    def __init__(
        self,
        *,
        params: Any | None = None,
        loss_fn: Callable[[Any], Any] | None = None,
        input_grad_argnums: Sequence[int] = (),
        intermediate_grads: bool = False,
        max_intermediate_grads: int = 64,
    ) -> None:
        """Initialize MLX derived-gradient options.

        Parameters
        ----------
        params
            Explicit MLX parameter tree, or ``None`` to use module parameters
            when the captured model exposes ``parameters()``.
        loss_fn
            Callable mapping raw output to scalar loss, or ``None`` for scalar
            raw outputs.
        input_grad_argnums
            Input-relative argnums to differentiate.
        intermediate_grads
            Whether to expose exact op-level intermediate derived gradients.
        max_intermediate_grads
            Hard cap on saved op boundaries for the tap producer and oracle.
        """

        refuse_if_bad_max_intermediate_grads(max_intermediate_grads)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "loss_fn", loss_fn)
        object.__setattr__(self, "input_grad_argnums", tuple(input_grad_argnums))
        object.__setattr__(self, "intermediate_grads", bool(intermediate_grads))
        object.__setattr__(self, "max_intermediate_grads", max_intermediate_grads)


def _mlx_probe_identity(value: Any) -> Any:
    """Return ``value`` unchanged; probe body for traced-transform detection.

    Parameters
    ----------
    value
        Probe argument.

    Returns
    -------
    Any
        ``value`` unchanged.
    """

    return value


def _mlx_traced_transform_type(mx: Any) -> type | None:
    """Resolve the MLX traced-transform wrapper type from the runtime itself.

    ``mx.compile``, ``mx.grad``, ``mx.value_and_grad``, and ``mx.vmap`` all
    return the same opaque wrapper type, which replays a traced graph and
    therefore bypasses (or worse, tracer-pollutes) the monkeypatched eager
    capture surface. The type is resolved by compiling a trivial probe so the
    authority is the runtime, never a spoofable ``__module__`` string.

    Parameters
    ----------
    mx
        Imported ``mlx.core`` module.

    Returns
    -------
    type | None
        Exact wrapper type, or ``None`` when the runtime exposes no
        ``compile`` entry (detection then degrades gracefully).
    """

    compile_fn = getattr(mx, "compile", None)
    if not callable(compile_fn):
        return None
    try:
        return type(compile_fn(_mlx_probe_identity))
    except Exception:
        return None


def _find_mlx_compiled_attributes(
    model: object,
    transform_type: type | None,
    *,
    max_depth: int = 8,
) -> tuple[str, ...]:
    """Return dotted paths of traced-transform callables reachable from ``model``.

    The scan is bounded and disclosed: instance ``__dict__`` values of the
    model and nested ``mlx.nn.Module`` children, plus one level inside plain
    ``list``/``tuple``/``dict`` containers. Slots-only holders, values created
    inside ``__call__``, and hot global/free compiled callables remain
    documented residuals.

    Parameters
    ----------
    model
        Capture entry object.
    transform_type
        Exact traced-transform wrapper type from
        :func:`_mlx_traced_transform_type`.
    max_depth
        Recursion bound over nested module attributes.

    Returns
    -------
    tuple[str, ...]
        Sorted dotted attribute paths holding traced-transform wrappers.
    """

    if transform_type is None:
        return ()
    try:
        import mlx.nn as mlx_nn
    except ImportError:
        return ()
    found: set[str] = set()
    seen: set[int] = set()

    def _scan_value(path: str, value: Any, depth: int) -> None:
        """Record ``path`` when ``value`` (or a direct list/dict item) is the transform type."""

        if type(value) is transform_type:
            found.add(path)
            return
        if isinstance(value, mlx_nn.Module):
            _scan_module(path, value, depth)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if type(item) is transform_type:
                    found.add(f"{path}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                if type(item) is transform_type:
                    found.add(f"{path}[{key!r}]")

    def _scan_module(prefix: str, module: object, depth: int) -> None:
        """Recurse through one ``mlx.nn.Module``'s attribute and dict surfaces.

        Both surfaces are scanned because ``mlx.nn.Module`` subclasses ``dict``:
        children and arrays live in the dict items while plain Python attributes
        land in ``__dict__``. Bounded by ``max_depth`` and an identity-seen set.
        """

        if depth > max_depth or id(module) in seen:
            return
        seen.add(id(module))
        # mlx.nn.Module subclasses dict: children/arrays live in the dict
        # items while plain Python attributes land in __dict__ — scan both.
        surfaces: list[dict[str, Any]] = []
        attributes = getattr(module, "__dict__", None)
        if isinstance(attributes, dict):
            surfaces.append(attributes)
        if isinstance(module, dict):
            surfaces.append(module)
        for surface in surfaces:
            for name, value in surface.items():
                child_path = f"{prefix}.{name}" if prefix else str(name)
                _scan_value(child_path, value, depth + 1)

    if isinstance(model, mlx_nn.Module):
        _scan_module("", model, 0)
    return tuple(sorted(found))


def _append_replay_inventory(trace: Trace, capture: MLXOpCapture) -> None:
    """Append one op's emit-time fingerprint (from its capture) to the trace's replay inventory."""

    inventory = getattr(trace, "_mlx_replay_inventory", None)
    if inventory is None:
        inventory = []
        trace._mlx_replay_inventory = inventory
    # The 5th element pins emit-time intervention declarations, so a
    # post-hoc claim that a value was (or was not) intervened mismatches
    # the immutable fingerprint instead of steering hook re-application.
    inventory.append(
        (
            capture.op_name,
            capture.labels_raw,
            capture.arg_leaf_labels,
            tuple(sorted((key, labels) for key, labels in capture.kwarg_leaf_labels.items())),
            capture.interventions,
        )
    )


def _drop_forward_only_state(trace: Trace) -> None:
    """Delete the emit-path state the forward read; it must not outlive the model call."""

    # Forward-time-only state: the emit path reads these during the
    # model call; leaving them on the trace breaks the portable-state
    # scrub (they are not in _PORTABLE_STATE_SPEC by design).
    del trace._mlx_intervention_plan
    del trace._mlx_halt_selector
    del trace._mlx_param_ids
    del trace._mlx_external_arrays


def _stamp_param_logs(trace: Trace, module_tree: MLXModuleTree | None) -> None:
    """Record the parameter accessor and counts; empty when no object module was traced."""

    if module_tree is not None:
        trace._param_logs = ParamAccessor(mlx_param_logs(module_tree, trace))
        _stamp_param_totals(trace, trace._param_logs)
        trace._provenance_param_source = "native-module"
    else:
        trace._param_logs = ParamAccessor({})
        _stamp_param_totals(trace, ())
        trace._provenance_param_source = "none"


def _live_save_match(selector: Any, ctx: RecordContext) -> bool:
    """Live ``save=`` match; a finalized label cannot match before labels exist.

    Such a selector answers "not now": the op becomes a witness and the final
    keep set restores its payload from the spill when the trace is settled.
    """

    try:
        return selector_matches_capture_context(selector, ctx)
    except LiveModeLabelError:
        return False


#: The capture-stack callables MLX retention runs use (``_retention`` imports none).
_RETENTION_DEPS = RetentionDeps(
    undecidable=lambda selector: (
        first_selector_kind_outside(selector, allowed=LIVE_DECIDABLE_KINDS) is not None
    ),
    live_match=_live_save_match,
    apply_keep_set=partial(apply_save_keep_set, backend_name="MLX"),
)


class MLXBackend:
    """MLX adapter for the backend-neutral capture Protocol."""

    name = "mlx"
    supports_backward_capture = capabilities.supports_backward_capture

    def __init__(self) -> None:
        """Initialize an MLX backend and verify the optional dependency."""

        self.mx, self.nn = self._import_mlx()
        self.tensor_store = MLXTensorLabelStore()
        self._retention_run: RetentionRun | None = None

    def wrap(self, value: object, module_tree: MLXModuleTree | None = None) -> object:
        """Install MLX wrappers and return ``value`` unchanged."""

        wrap_mlx(self, module_tree=module_tree)
        return value

    def unwrap(self, value: object) -> object:
        """Remove MLX wrappers and return ``value`` unchanged."""

        unwrap_mlx()
        return value

    def is_wrapped(self, value: object) -> bool:
        """Return whether MLX wrappers are installed."""

        return is_mlx_wrapped()

    def start_session(self, options: object) -> object:
        """Start an MLX capture session.

        Parameters
        ----------
        options:
            Trace-like capture options object.

        Returns
        -------
        object
            The unchanged options object.
        """

        return options

    def prepare_model(self, session: object, model: object) -> object:
        """Apply one-time and per-session MLX model preparation."""

        self.prepare_model_once(model)
        self.prepare_model_session(session, model)
        return model

    def prepare_model_once(self, model: object) -> object:
        """Apply one-time MLX model preparation."""

        return prepare_model_once(model)

    def prepare_model_session(self, session: object, model: object) -> object:
        """Apply per-session MLX model preparation and start any retention run."""

        self._retention_run = start_retention(self.mx, _RETENTION_DEPS)
        return prepare_model_session(session, model)

    def cleanup_model_session(self, session: object, prepared_model: object) -> None:
        """Clean up per-session MLX model preparation and end the retention run."""

        self._retention_run = None
        cleanup_model_session(session, prepared_model)

    def cleanup_stopped_forward_session(self, session: object, prepared_model: object) -> None:
        """Clean up MLX state after a halted forward capture."""

        self.cleanup_model_session(session, prepared_model)

    def cleanup_failed_forward_session(
        self,
        session: object,
        prepared_model: object,
        exc: Exception,
    ) -> None:
        """Clean up MLX state after a failed forward capture."""

        del exc
        self.cleanup_model_session(session, prepared_model)

    def cleanup_forward_memory(self, session: object) -> None:
        """Release MLX transient forward-memory caches.

        Parameters
        ----------
        session:
            Active trace session, unused by the current MLX preview.

        Returns
        -------
        None
            No memory cleanup is required.
        """

        del session

    def active_logging(self, session: object) -> AbstractContextManager[None]:
        """Return a context manager that enables MLX logging."""

        return _state.active_logging(cast(Trace, session))

    def pause_logging(self, session: object) -> AbstractContextManager[None]:
        """Return a context manager that pauses MLX logging."""

        return _state.pause_logging()

    def snapshot_rng(self, session: object) -> object:
        """Return the initial MLX RNG snapshot.

        MLX RNG replay is intentionally unsupported in this milestone, per AD-9.
        """

        return None

    def seed_rng(self, session: object, seed: int) -> None:
        """Seed host RNG engines for an MLX shared-capture session.

        Parameters
        ----------
        session:
            Active trace session, unused by MLX RNG seeding.
        seed:
            Integer seed value.

        Returns
        -------
        None
            Python and NumPy RNG engines are seeded in place.
        """

        del session
        random.seed(seed)
        np.random.seed(seed)

    def set_capture_producer_policy(self, session: object, capture_mode: object) -> None:
        """Install MLX producer policy metadata for a shared-capture session.

        Parameters
        ----------
        session:
            Active trace session, unused by MLX.
        capture_mode:
            Capture mode, unused by MLX.

        Returns
        -------
        None
            No producer policy metadata is required for MLX.
        """

        del session, capture_mode

    def restore_rng(self, session: object, rng_state: object) -> None:
        """Restore an MLX RNG snapshot.

        Parameters
        ----------
        session:
            Active trace session, unused because MLX RNG replay is unsupported.
        rng_state:
            Opaque RNG snapshot returned by :meth:`snapshot_rng`.

        Returns
        -------
        None
            No state is restored for the current MLX preview.
        """

        del session, rng_state

    def inference_context(self, session: object) -> AbstractContextManager[None]:
        """Return the MLX inference-only context for this session.

        Parameters
        ----------
        session:
            Active trace session, unused because MLX has no TorchLens no-grad
            context integration in this preview.

        Returns
        -------
        AbstractContextManager[None]
            A null context.
        """

        del session
        return nullcontext()

    def build_record_context(
        self,
        session: object,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> RecordContext:
        """Build the selector predicate context for one MLX output.

        MLX's lazy backend guarantees shape, dtype, and device metadata at call
        time. Value-dependent fields, specifically ``tensor_requires_grad``,
        ``is_scalar_bool``, and ``bool_value``, are represented by the
        ``_DEFERRED_VALUE`` sentinel so predicates cannot accidentally consume
        silently-wrong values or force per-op ``mx.eval``.
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
            func_name=func_event_input.func_name,
            address=None,
            module_type=None,
            module_pass_number=None,
            module_stack=func_event_input.module_stack,
            recent_events=(),
            recent_ops=(),
            parent_labels=(),
            input_output_address=None,
            shape=self._shape(output),
            dtype=DtypeRef.from_value(self._dtype(output)),
            tensor_device=DeviceRef.from_value(self._device(output)),
            tensor_requires_grad=_DEFERRED_VALUE,
            output_index=None,
            is_bottom_level_func=func_event_input.is_bottom_level_func,
            time_since_pass_start=0.0,
            sample_id=None,
            label_raw=reserved.label_raw,
            label_prefix=reserved.layer_type,
            func_call_id=func_event_input.func_call_id,
            parent_labels_raw=(),
            is_output_parent=False,
            backend_requires_isolation=False,
            is_scalar_bool=_DEFERRED_VALUE,
            bool_value=_DEFERRED_VALUE,
        )

    def detect_backend_semantics(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> BackendSemantics:
        """Return MLX backend semantics for one output."""

        return BackendSemantics(
            backend_grad_handle=None,
            grad_fn_class_name=None,
            autograd_memory=None,
            num_autograd_tensors=None,
            mutated_input_positions=(),
            aliased_output_inputs=(),
            unknown_aliasing=False,
            bytes_delta_at_call=None,
            bytes_peak_at_call=None,
        )

    def tensor_ref(
        self,
        session: object,
        value: object,
        payload: object | None,
        policy: CapturePolicy,
    ) -> TensorRef:
        """Build metadata for an MLX array without forcing materialization."""

        if not self.is_tensor(value):
            return TensorRef("", None, None, None, None, None, payload, None, None)
        return TensorRef(
            label_raw=self.tensor_store.get_label(value) or "",
            shape=self._shape(value),
            dtype=self._dtype(value),
            device=self._device(value),
            requires_grad=None,
            memory=self._memory(value),
            payload=payload,
            blob_ref=None,
            backend_handle_id=str(id(value)),
        )

    def set_tensor_label(self, session: object, value: object, label: str) -> None:
        """Set the raw TorchLens label for an MLX array."""

        if self.is_tensor(value):
            self.tensor_store.set_label(value, label)

    def is_tensor(self, value: object) -> bool:
        """Return whether ``value`` is an MLX array."""

        array_type = getattr(self.mx, "array", None)
        return array_type is not None and isinstance(value, array_type)

    def is_parameter(self, value: object) -> bool:
        """Return whether ``value`` is an MLX parameter-like array."""

        return self.is_tensor(value)

    def apply_live_hooks(
        self,
        session: object,
        value: object,
        site: ReservedLabel,
    ) -> tuple[object, tuple[FireResult, ...]]:
        """Return MLX values unchanged because live intervention is out of scope."""

        return value, ()

    def safe_copy(self, session: object, value: object, policy: CapturePolicy) -> object:
        """Return an MLX payload reference for deferred materialization."""

        return value

    def emit_function_outputs(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        isolated_output: object,
        output_sites: tuple[object, ...],
        reserved_block: tuple[ReservedLabel, ...],
        *,
        fire_results_by_site: dict[int, tuple[FireResult, ...]] | None = None,
    ) -> tuple[OpEvent, ...]:
        """Emit topology-complete Protocol operation events for MLX outputs.

        MLX eval timing is deliberately split: shape, dtype, device, parent
        edges, and container paths are populated at call time; saved payloads
        remain lazy and are batch-forced in ``finalize_forward_session`` via
        ``mx.eval``; value-dependent predicate fields are deferred with the
        ``_DEFERRED_VALUE`` sentinel and projected to ``None`` for stored event
        metadata.
        """

        events: list[OpEvent] = []
        policy = self._capture_policy(session)
        output_by_site = tuple(output_sites)
        for site_index, (output, reserved) in enumerate(
            zip(output_by_site, reserved_block, strict=True)
        ):
            if not self.is_tensor(output):
                continue
            self.tensor_store.set_label(output, reserved.label_raw)
            out_policy = self._payload_policy(session, policy, reserved, func_event_input, output)
            parents, parent_arg_positions, edge_uses = self._parent_edges(
                func_event_input.args,
                dict(func_event_input.kwargs),
            )
            fire_results = (fire_results_by_site or {}).get(site_index, ())
            event = self._build_event(
                session=session,
                kind="op",
                reserved=reserved,
                func_event_input=func_event_input,
                output=output,
                parents=parents,
                parent_arg_positions=parent_arg_positions,
                edge_uses=edge_uses,
                policy=out_policy,
                is_input=False,
                fire_results=fire_results,
            )
            events.append(event)
        return tuple(events)

    def finalize_forward_session(
        self,
        session: object,
        trace_state: RawGraphWorkspace,
        *,
        extra_values: Sequence[object] = (),
    ) -> None:
        """Realize every deferred MLX payload and output in one evaluation.

        This is the realize-once rule: the forward ran lazily to its outputs,
        and exactly one ``mx.eval`` over the recorded payloads plus
        ``extra_values`` (the outputs) materializes them together, so RNG and
        fused regions evaluate once, as in one eager forward.
        """

        del trace_state
        payloads = list(getattr(session, "_mlx_saved_payloads", ()))
        seen = {id(value) for value in payloads}
        payloads.extend(value for value in extra_values if id(value) not in seen)
        if self._retention_run is not None:
            self._retention_run.realize(tuple(payloads))
        elif payloads:
            cast(Any, self.mx).eval(*payloads)

    def capture_trace(
        self,
        model: object,
        input_args: object,
        input_kwargs: dict[Any, Any] | None = None,
        *,
        save: str | list[Any] | None = "all",
        keep_orphans: bool = False,
        output_device: str = "same",
        activation_transform: object | None = None,
        save_raw_activations: bool = True,
        detach_saved_activations: bool = False,
        save_grads: bool | str | list[Any] | object | None = None,
        seed: int | None = None,
        num_context_lines: int = 7,
        save_args: bool = False,
        save_source_context: bool = False,
        save_rng: bool = False,
        recurrence_detection: bool = True,
        compute_input_output_distances: bool = True,
        verbose: bool = False,
        backward_ready: bool = False,
        name: str | None = None,
        transform: object | None = None,
        raw_input: object | None = None,
        save_raw_input: str | bool = "small",
        batch_render: str = "auto",
        output_transform: object | None = None,
        save_transformed_output: str | bool = "small",
        layer_visualizers: dict[Any, Any] | None = None,
        save_visualizations: bool = False,
        module_identity_mode: str | None = None,
        grad_options: GradOptions | None = None,
        intervene: object | None = None,
        halt: object | None = None,
    ) -> Trace:
        """Capture an MLX forward pass into a smoke-compatible Trace."""

        reject_unsupported_trace_options(
            {
                "save": save,
                "activation_transform": activation_transform,
                "detach_saved_activations": detach_saved_activations,
                "save_args": save_args,
                "save_grads": save_grads,
                "save_source_context": save_source_context,
                "save_rng": save_rng,
                "backward_ready": backward_ready,
                "transform": transform,
                "output_device": output_device,
                "layer_visualizers": layer_visualizers,
                "save_visualizations": save_visualizations,
            },
            MLX_PREVIEW_TRACE_OPTION_POLICY,
            spec=get_backend_spec("mlx"),
        )
        if seed is not None:
            raise BackendUnsupportedError(
                "The MLX backend does not support random_seed; pass explicit MLX RNG "
                "state through the model/input surface instead.",
                code="explicit_seed_unsupported",
            )
        intervention_plan = None
        halt_selector = None
        if intervene is not None or halt is not None:
            spec = get_backend_spec("mlx")
            if not spec.capabilities.can_intervene:
                raise BackendUnsupportedError(
                    "MLX backend capability table declares can_intervene=False; "
                    "refusing trace(intervene=/stop_after=) instead of silently "
                    "ignoring the requested behavior.",
                    code="interventions_unsupported",
                )
            # The registered binding IS the dispatch surface: resolution and
            # application come from the capability implementation, so a bound-
            # but-unregistered or in-place-flipped table refuses typed.
            resolve = cast(Any, require_capability_implementation(spec, "can_intervene"))
            intervention_plan, halt_selector = resolve(intervene, halt, self.mx)
            check_edit_grad_options(intervention_plan, grad_options)
        transform_type = _mlx_traced_transform_type(self.mx)
        if transform_type is not None and type(model) is transform_type:
            raise BackendUnsupportedError(
                "MLX capture entry is an mx.compile/mx.grad/mx.vmap traced-transform "
                "wrapper. Traced replays bypass the eager capture surface, so logging "
                "them would silently under-capture. Trace the eager mlx.nn.Module or "
                "plain Python callable instead of its compiled/transformed wrapper.",
                code="traced_transform_entry_unsupported",
            )
        compiled_attribute_paths = _find_mlx_compiled_attributes(model, transform_type)
        module_tree = discover_mlx_module_tree(model)
        use_object_module = _resolve_mlx_module_identity_mode(module_identity_mode, module_tree)
        trace = Trace(
            model_class_name=type(model).__name__,
            output_device=output_device,
            activation_transform=cast("Callable[[Any], Any] | None", activation_transform),
            save_raw_activations=save_raw_activations,
            save_raw_grads=True,
            keep_orphans=keep_orphans,
            save_args=save_args,
            save_grads=None,
            detach_saved_activations=detach_saved_activations,
            mark_layer_depths=compute_input_output_distances,
            source_context_lines=num_context_lines,
            save_source_context=save_source_context,
            save_rng=save_rng,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            backward_ready=backward_ready,
            emit_nvtx=False,
            input_transform=cast("Callable[[Any], Any] | None", transform),
            input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=cast("Callable[[Any], Any] | None", output_transform),
            save_transformed_output=save_transformed_output,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
        )
        fill_unset_declared_fields(trace)
        trace.name = name
        trace._provenance_backend = cast(BackendName, self.name)
        # Root identity (C07X item (iv)); the qualname is also the root record's class_qualname.
        if module_tree is not None:
            qualname = f"{type(model).__module__}.{type(model).__qualname__}"
            trace._provenance_root_entry_point = f"module_call:{qualname}.__call__"
        else:
            qualname = getattr(model, "__qualname__", trace._source_class_name)
            trace._provenance_root_entry_point = f"function_call:{qualname}"
        trace._source_class_qualname = qualname
        _disclose_compiled_attributes(trace, compiled_attribute_paths)
        trace.capture_events = CaptureEvents()
        trace._mlx_saved_payloads = []
        trace._mlx_capture_depth = 0
        trace._mlx_module_stack = []
        trace._mlx_intervention_plan = intervention_plan
        trace._mlx_halt_selector = halt_selector
        # The constants rule needs two censuses before the forward runs: the
        # model's parameter identities (a parameter leaf stays a parameter)
        # and every array reachable from the model object (a closure constant).
        # An unlabeled argument in neither came from an unwrapped op.
        trace._mlx_param_ids = (
            frozenset(module_tree.param_address_by_id) if module_tree is not None else frozenset()
        )
        trace._mlx_external_arrays = external_array_ids(model, self.is_tensor)
        trace._mlx_unwrapped_intermediates = 0
        trace._pre_forward_rng_states = None
        trace._options_seed = cast(int, seed) if seed is not None else random.randint(1, 4294967294)
        self.tensor_store.clear()
        # R07: the try owns the wrap call itself -- a raise anywhere between wrapper
        # install and the forward (session prep, input normalization, source labeling)
        # used to strand the process-global MLX wrappers because the unwrap-owning
        # finally had not been entered yet. ``unwrap`` on a partial registry is safe.
        try:
            self.wrap(model, module_tree if use_object_module else None)
            self.prepare_model_session(trace, model)
            args = self._normalize_input_args(input_args)
            kwargs = {} if input_kwargs is None else dict(input_kwargs)
            self._label_source_arrays(trace, args, kwargs)
            trace._provenance_start_time = time.time()
            halt_signal: StopSignal | None = None
            try:
                with self.active_logging(trace), forward_hooks(trace) as hooks:
                    output = cast(Any, model)(*args, **kwargs)
            except StopSignal as signal:
                # Save-then-halt ordering: the frontier op's events were
                # appended before the signal, so the partial graph ends at it.
                halt_signal = signal
                output = signal.frontier_output
                trace.is_stopped = True
                trace.stop_reason = signal.reason
                trace._graph_stop_frontier = signal.reason
            _drop_forward_only_state(trace)
            _disclose_unfired_selectors(
                trace, intervention_plan, halt, halt_selector, halted=halt_signal is not None
            )
            trace._cost_forward_duration = Duration(time.time() - trace._provenance_start_time)

            if halt_signal is not None:
                trace._output_slot = stopped(halt_signal.reason)
                trace.transformed_output = None
            else:
                trace._output_slot = returned(output)
                trace.transformed_output = (
                    output_transform(output) if callable(output_transform) else None
                )
            # Realize-once (SPEC section 3): one evaluation over every
            # recorded value and the outputs, after the forward and before any
            # payload is read; nothing below forces another evaluation.
            self.finalize_forward_session(
                trace, trace._raw_graph_ws, extra_values=tuple(self._iter_arrays(output))
            )
            self._mark_outputs(trace, output)
            materialize_from_events(trace, trace.capture_events)
            delattr(trace, "capture_events")
            _stamp_param_logs(trace, module_tree if use_object_module else None)
            self._finish_trace(trace, module_tree if use_object_module else None)
            if halt_signal is not None:
                self._restrict_halted_param_logs(trace)
            if grad_options is not None:
                self._attach_derived_grads(
                    trace=trace,
                    model=edited_replay_model(self, trace, model, intervention_plan, wrappers),
                    args=args,
                    kwargs=kwargs,
                    captured_output=output,
                    grad_options=grad_options,
                )
            self._settle_capture(trace, hooks)
            if hasattr(trace, "_mlx_module_stack"):
                delattr(trace, "_mlx_module_stack")
            freeze_trace_relation_views(trace)
        finally:
            # Independently-owned resources: a raising session cleanup must
            # not leave the process-global MLX wrappers installed.
            try:
                self.cleanup_model_session(trace, model)
            finally:
                self.unwrap(model)
        # Settlement is the LAST act, after ALL teardown (the path-20 stamp
        # contract): a teardown raise escapes productless -- the object
        # derives UNATTESTED, never carrying a COMPLETE/HALTED stamp.
        stamp_backend_finalized(trace)
        return trace

    def _restrict_halted_param_logs(self, trace: Trace) -> None:
        """Restrict a halted trace's parameter accounting to captured ops.

        A halted partial graph never ran the modules past the frontier, so
        declaring the full model's parameter inventory would break the
        trace/op self-consistency invariant (and overstate what the partial
        capture proved). Only parameters attached to captured ops survive.

        Parameters
        ----------
        trace
            Finalized halted trace.
        """

        attached = {
            barcode
            for op in trace._layer_list
            for barcode in (getattr(op, "_param_barcodes", None) or ())
        }
        kept = {
            address: param for address, param in trace.params.items() if param.barcode in attached
        }
        trace._param_logs = ParamAccessor(kept)
        _stamp_param_totals(trace, kept.values())

    def _attach_derived_grads(
        self,
        *,
        trace: Trace,
        model: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[Any, Any],
        captured_output: Any,
        grad_options: GradOptions,
    ) -> None:
        """Compute and attach MLX leaf-level derived gradients.

        Parameters
        ----------
        trace
            Trace receiving derived gradient records.
        model
            Captured MLX callable.
        args
            Positional call arguments used for capture.
        kwargs
            Keyword call arguments used for capture.
        captured_output
            Raw output from the captured forward call.
        grad_options
            MLX derived-gradient options.

        Returns
        -------
        None
            ``trace.autograd.derived_grads`` and unambiguous param gradient slots are populated.
        """

        params = grad_options.params
        if params is None:
            parameters = getattr(model, "parameters", None)
            if callable(parameters):
                params = parameters()
        has_params = _has_mlx_array_leaf(params)
        input_grad_argnums = _normalize_mlx_input_grad_argnums(
            grad_options.input_grad_argnums,
            len(args),
        )
        if not has_params and not input_grad_argnums:
            raise ValueError(
                "MLX derived gradients require module params or at least one input argnum."
            )

        value_args: tuple[Any, ...]
        differentiated_argnums: tuple[int, ...]
        param_argnum: int | None
        input_argnum_by_value_argnum: dict[int, int]
        if has_params:
            value_args = (params, *args)
            param_argnum = 0
            input_argnum_by_value_argnum = {index + 1: index for index in input_grad_argnums}
            differentiated_argnums = (0, *(index + 1 for index in input_grad_argnums))
        else:
            value_args = tuple(args)
            param_argnum = None
            input_argnum_by_value_argnum = {index: index for index in input_grad_argnums}
            differentiated_argnums = tuple(input_grad_argnums)

        original_params = params if has_params else None
        tap_observer: _MLXIntermediateTapObserver | None = None

        def value_fn(*value_fn_args: Any) -> tuple[Any, Any]:
            """Return scalar loss plus raw output aux for ``mx.value_and_grad``.

            Parameters
            ----------
            *value_fn_args
                Positional values passed by MLX AD.

            Returns
            -------
            tuple[Any, Any]
                Scalar loss and raw model output.
            """

            offset = 0
            if has_params:
                update = getattr(model, "update", None)
                if not callable(update):
                    raise BackendUnsupportedError(
                        "MLX derived gradients require model.update(params) for parameter "
                        "rebinding."
                    )
                update(value_fn_args[0])
                offset = 1
            if tap_observer is not None:
                tap_observer.label_call_inputs(value_fn_args[offset:], kwargs)
            raw_output = model(*value_fn_args[offset:], **dict(kwargs))
            loss = grad_options.loss_fn(raw_output) if grad_options.loss_fn else raw_output
            if not _is_scalar_mlx_value(loss):
                raise ValueError(
                    "MLX derived gradients require loss_fn(raw_output) to be scalar unless "
                    "the traced output is already scalar."
                )
            return loss, raw_output

        grad_fn = cast(Any, self.mx).value_and_grad(
            value_fn,
            argnums=differentiated_argnums[0]
            if len(differentiated_argnums) == 1
            else differentiated_argnums,
        )
        try:
            if grad_options.intermediate_grads:
                tap_observer = _MLXIntermediateTapObserver(self, trace)
                with mlx_tap_observer(tap_observer):
                    (_loss, aux_output), grads = grad_fn(*value_args)
            else:
                (_loss, aux_output), grads = grad_fn(*value_args)
            _eval_mlx_tree(self.mx, aux_output)
            _eval_mlx_tree(self.mx, captured_output)
            if not _mlx_trees_close(aux_output, captured_output):
                raise ValueError(
                    "MLX derived gradient run raw output diverged from captured raw output; "
                    "refusing to expose trace.autograd.derived_grads."
                )
            grad_trees = grads if len(differentiated_argnums) != 1 else (grads,)
            records: dict[str, DerivedGradRecord] = {}
            for value_argnum, grad_tree in zip(differentiated_argnums, grad_trees, strict=True):
                records.update(
                    _records_for_mlx_grad_tree(
                        grad_tree=grad_tree,
                        argnum=value_argnum,
                        param_argnum=param_argnum,
                        input_argnum_by_value_argnum=input_argnum_by_value_argnum,
                        provenance={
                            "backend": "mlx",
                            "kind": "derived_gradient",
                            "mechanism": "mlx_value_and_grad",
                            "loss_fn": _callable_identity(grad_options.loss_fn),
                        },
                    )
                )
            if grad_options.intermediate_grads and tap_observer is not None:
                set_intermediate_derived_grads(
                    trace,
                    (
                        self._records_for_intermediate_mlx_grads(
                            candidates=tap_observer.candidates,
                            replay=_MLXReplayCall(trace, value_args, kwargs, value_fn),
                            grad_options=grad_options,
                        )
                    ),
                )
        finally:
            if has_params:
                update = getattr(model, "update", None)
                if callable(update):
                    update(original_params)
        trace._autograd_derived_grads = DerivedGradAccessor(records)
        mirror_param_derived_grads(trace, records)

    def _records_for_intermediate_mlx_grads(
        self,
        *,
        candidates: Sequence[_MLXIntermediateCandidate],
        replay: _MLXReplayCall,
        grad_options: GradOptions,
    ) -> IntermediateDerivedGradAccessor:
        """Build exact MLX op-level derived gradient records.

        Parameters
        ----------
        candidates
            Tap candidates observed during the auxiliary AD replay.
        replay
            The finalized trace (its saved ops define the public attachment
            set), the ``mx.value_and_grad`` args, the model's original keyword
            arguments and the scalar-loss function of the leaf-gradient replay.
        grad_options
            MLX derived-gradient options.

        Returns
        -------
        IntermediateDerivedGradAccessor
            Exact, oracle-confirmed records keyed by pass-qualified op label.
        """

        trace = replay.trace
        selected_ops = tuple(
            op
            for op in trace._layer_list
            if op.has_saved_activation and not op.is_input and _is_float_mlx_array(op.out)
        )
        if len(selected_ops) > grad_options.max_intermediate_grads:
            raise BackendUnsupportedError(
                "MLX intermediate derived gradients are capped at "
                f"{grad_options.max_intermediate_grads} saved boundaries; got "
                f"{len(selected_ops)}."
            )
        if not selected_ops:
            return IntermediateDerivedGradAccessor()
        _eval_mlx_tree(
            self.mx,
            tuple(candidate.grad for candidate in candidates if candidate.grad is not None),
        )
        trace_groups = _mlx_trace_intermediate_signatures(selected_ops)
        replay_groups: dict[_MLXIntermediateSignature, list[_MLXIntermediateCandidate]] = (
            defaultdict(list)
        )
        for candidate in candidates:
            replay_groups[candidate.signature].append(candidate)

        records: dict[str, IntermediateDerivedGradRecord] = {}
        for signature, ops in trace_groups.items():
            if len(ops) != 1:
                continue
            replay_candidates = replay_groups.get(signature, [])
            if len(replay_candidates) != 1:
                continue
            op = ops[0]
            candidate = replay_candidates[0]
            if candidate.grad is None:
                continue
            if not _mlx_intermediate_oracle_passes(
                backend=self,
                signature=signature,
                value=candidate.value,
                producer_grad=candidate.grad,
                replay=replay,
            ):
                continue
            records[op.label] = IntermediateDerivedGradRecord(
                op_label=op.label,
                layer_label=op.layer_label,
                aval=(
                    f"array(shape={tuple(getattr(candidate.grad, 'shape', ()))}, "
                    f"dtype={getattr(candidate.grad, 'dtype', None)})"
                ),
                dtype_ref=DtypeRef(backend="mlx", name=str(getattr(candidate.grad, "dtype", ""))),
                grad=candidate.grad,
                provenance={
                    "backend": "mlx",
                    "kind": "intermediate_derived_gradient",
                    "mechanism": "mlx_custom_vjp_tap_value_and_grad",
                    "loss_id": _callable_identity(grad_options.loss_fn),
                    "save_predicate_id": "trace.saved_ops",
                    "status": "exact",
                    "oracle": "producer_custom_vjp+boundary_replacement_grad+perturbation",
                    "max_intermediate_grads": grad_options.max_intermediate_grads,
                },
            )
        return IntermediateDerivedGradAccessor(records)

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Capture then validate an MLX forward pass.

        Parameters
        ----------
        *args, **kwargs
            Public validation arguments forwarded to ``capture_trace``.

        Returns
        -------
        bool
            True when live replay validation passes.
        """

        validate_metadata = bool(kwargs.pop("validate_metadata", True))
        trace = self.capture_trace(*args, **kwargs)
        return self.validate_trace(trace, validate_metadata=validate_metadata).passed

    def validate_trace(
        self,
        trace: Trace,
        *_args: Any,
        **kwargs: Any,
    ) -> ValidationStatus:
        """Validate an MLX trace with per-op replay and perturbation.

        Parameters
        ----------
        trace
            MLX trace to validate.
        *_args
            Ignored compatibility arguments.
        **kwargs
            Compatibility keyword arguments. ``validate_metadata`` controls
            whether backend-neutral invariant checks run.

        Returns
        -------
        ValidationStatus
            The replay status, never a bare bool (SPEC 1.1 ``validate``): a
            ``failed`` status names the first offending op and a reason from
            ``VALIDATION_REASONS`` in ``decisions`` and in its message; loaded
            payload-stripped traces return the unavailable status.
        """

        kept = kept_or_unavailable_status(trace, self._stamp_unavailable_status)
        if kept is not None:
            return kept
        replayed_count = 0
        failed_count = 0
        decisions: tuple[dict[str, Any], ...] = ()
        try:
            from .validation import MLX_VALIDATION_REASONS, validate_mlx_captures_detailed

            exempted: dict[str, int] = {}
            unverified_reasons: dict[str, int] = {}
            if kwargs.get("validate_metadata", True) and not run_metadata_invariants(trace):
                failed_count = 1
                decisions = (_mlx_metadata_failed_decision(MLX_VALIDATION_REASONS),)
            else:
                (replayed_count, failed_count, decisions, exempted, unverified_reasons) = (
                    _mlx_detailed_counts(trace, validate_mlx_captures_detailed)
                )
            if failed_count == 0 and replayed_count < 1:
                failed_count = 1
                decisions = (*decisions, _mlx_nothing_replayed_decision(MLX_VALIDATION_REASONS))
            unverified_count = _mlx_unverified_count(
                trace, failed_count, unverified_reasons, count_importer_region_annotations
            )
            status_result = self._counts_status(
                trace,
                (replayed_count, unverified_count, failed_count),
                decisions,
                unverified_reasons=unverified_reasons,
                exempted=exempted,
            )
        except Exception as exc:  # noqa: BLE001 - a raising oracle is a failed verdict
            status_result = self._oracle_raised_status(
                trace, exc, replayed_count, failed_count, decisions
            )
        status_result = _name_first_failure(status_result)
        trace._validation_replay_status = status_result
        return status_result

    def _stamp_unavailable_status(self, trace: Trace) -> ValidationStatus:
        """Record and return the unavailable status for a payload-stripped loaded trace."""

        status_result = ValidationStatus.unavailable_loaded_runtime_stripped(
            backend=self.name,
            payload_load_status=getattr(trace, "payload_load_status", None),
        )
        trace._validation_replay_status = status_result
        return status_result

    def _counts_status(
        self,
        trace: Trace,
        counts: tuple[int, int, int],
        decisions: tuple[dict[str, Any], ...],
        *,
        unverified_reasons: dict[str, int],
        exempted: dict[str, int],
    ) -> ValidationStatus:
        """The status derived from the replay, unverified and failed op ``counts``."""

        replayed_count, unverified_count, failed_count = counts
        return ValidationStatus.from_replay_counts(
            backend=self.name,
            source=_mlx_validation_source(trace),
            num_replayed_ops=replayed_count,
            num_unverified_ops=unverified_count,
            num_failed_ops=failed_count,
            payload_load_status=getattr(trace, "payload_load_status", None),
            unverified_reason_counts=unverified_reasons or None,
            exempted_reason_counts=exempted or None,
            decisions=decisions,
        )

    def _oracle_raised_status(
        self,
        trace: Trace,
        exc: BaseException,
        replayed_count: int,
        failed_count: int,
        decisions: tuple[dict[str, Any], ...],
    ) -> ValidationStatus:
        """The failed status when the replay oracle itself raised."""

        decisions = (
            *decisions,
            _mlx_trace_decision("replay", "no_replayable_function", f"{type(exc).__name__}: {exc}"),
        )
        return ValidationStatus.result(
            passed=False,
            backend=self.name,
            source=_mlx_validation_source(trace),
            payload_load_status=getattr(trace, "payload_load_status", None),
            num_replayed_ops=replayed_count,
            num_failed_ops=max(1, failed_count),
            decisions=decisions,
        )

    def emit_mlx_operation(
        self,
        trace: Trace,
        op_name: str,
        func: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: object,
        *,
        module_stack: tuple[ModuleFrame, ...] | None = None,
    ) -> Any:
        """Append one MLX operation event and return the effective output.

        Returns
        -------
        Any
            The call output the model should consume downstream: the raw
            output, or the tree with intervention-replaced leaves.
        """

        # Container outputs (e.g. mx.split's list of arrays) materialize one
        # op per array leaf; dropping them here would silently disconnect the
        # graph (downstream consumers lose their parents) while keeping the
        # call absent from BOTH sides of the replay inventory, so validation
        # would bless the missing wiring.
        outputs = tuple(self._iter_arrays(output))
        if not outputs:
            return output
        events = getattr(trace, "capture_events", None)
        if events is None:
            events = CaptureEvents()
            trace.capture_events = events
        # Closure constants are recorded first so a constant source op always
        # precedes the op that consumes it in raw order.
        self._record_closure_constants(trace, args, kwargs, events)
        reserved = events.reserve_label_block(op_name, len(outputs))
        func_call_id = events.func_call_id_counter + 1
        events.func_call_id_counter = func_call_id
        func_event_input = FunctionEventInput(
            func=func,
            func_name=op_name,
            func_qualname=getattr(func, "__qualname__", None),
            args=args,
            kwargs=kwargs,
            raw_output=output,
            arg_copies=None,
            kwarg_copies=None,
            module_stack=module_stack or tuple(trace._mlx_module_stack),
            is_bottom_level_func=True,
            func_call_id=func_call_id,
            expected_output_count=len(outputs),
        )
        # Static-label interventions substitute matched leaves BEFORE labeling
        # and payload capture, so the recorded op output IS the value flowing
        # into downstream ops; halt matches after the frontier op is captured
        # (save-then-halt ordering).
        fire_results_by_site: dict[int, tuple[FireResult, ...]] = {}
        intervention_slots: tuple[tuple[int, str], ...] = ()
        intervention_appliers: tuple[tuple[int, Any], ...] = ()
        halt_label: str | None = None
        if trace._mlx_intervention_plan is not None or trace._mlx_halt_selector is not None:
            found = _match_live_sites(self, trace, outputs, reserved, func_event_input)
            fire_results_by_site, halt_label = found.fire_results_by_site, found.halt_label
            if found.replacements:
                output = _replace_mlx_array_leaves(self, output, found.replacements)
                outputs = tuple(self._iter_arrays(output))
                func_event_input = replace(func_event_input, raw_output=output)
                intervention_slots = tuple(found.slots)
                intervention_appliers = tuple(found.appliers)
        op_captures = getattr(trace, "_mlx_op_captures", None)
        if op_captures is None:
            op_captures = []
            trace._mlx_op_captures = op_captures
        labels_raw = tuple(entry.label_raw for entry in reserved)
        # Parent labels are recorded per array leaf BEFORE outputs are labeled,
        # so aliasing outputs cannot shadow their own parents.
        arg_leaf_labels = tuple(
            tuple(self.tensor_store.get_label(leaf) for leaf in self._iter_arrays(value))
            for value in args
        )
        kwarg_leaf_labels = {
            key: tuple(self.tensor_store.get_label(leaf) for leaf in self._iter_arrays(value))
            for key, value in kwargs.items()
        }
        # Template retention (paddle's _template_value model): labeled leaves
        # become REPLAY_SLOT sentinels because replay sources them from saved
        # parent payloads; keeping the raw arrays here pinned every
        # intermediate activation for the trace's lifetime.
        capture = MLXOpCapture(
            labels_raw=labels_raw,
            op_name=op_name,
            func=func,
            args=tuple(
                build_capture_template(
                    value,
                    arg_leaf_labels[index] if index < len(arg_leaf_labels) else (),
                )
                for index, value in enumerate(args)
            ),
            kwargs={
                key: build_capture_template(value, kwarg_leaf_labels.get(key, ()))
                for key, value in kwargs.items()
            },
            arg_leaf_labels=arg_leaf_labels,
            kwarg_leaf_labels=kwarg_leaf_labels,
            interventions=intervention_slots,
            appliers=intervention_appliers,
        )
        op_captures.append(capture)
        # Independent replay inventory: the validation oracle's denominator.
        # Losing a subset of _mlx_op_captures can then never shrink the
        # expected-coverage set along with the evidence. Each record also
        # snapshots the emit-time per-leaf parent labels, so stripping a
        # capture's recorded provenance (to launder emit-time argument values
        # through replay) fails coverage instead of replaying vacuously.
        # Appended to a list — O(1) per op on the capture hot path; wholesale
        # attribute replacement is coherent reauthoring, outside the threat
        # model, so a tuple rebuild bought no protection.
        _append_replay_inventory(trace, capture)
        emitted = self.emit_function_outputs(
            trace,
            func_event_input,
            output,
            outputs,
            reserved,
            fire_results_by_site=fire_results_by_site,
        )
        events.extend(emitted)
        if halt_label is not None:
            # Latch parity with torch's evaluate_halt: the stop request is
            # latched on the trace BEFORE the signal is raised, so a user
            # broad-except that swallows the HaltSignal can never reach the
            # settlement stamp as a blessable COMPLETE -- the one preview
            # stamp checks the latch at the capture boundary.
            trace.__dict__["_stop_requested"] = StopRequest(
                kind="halt",
                reason=halt_label,
                boundary_label=halt_label,
            )
            raise StopSignal(halt_label, frontier_output=output)
        return output

    def _apply_mlx_intervention(self, plan: Any, leaf: Any) -> Any:
        """Apply one resolved intervention to a matched output leaf.

        Parameters
        ----------
        plan
            Resolved ``MLXInterventionPlan``.
        leaf
            Matched MLX output array.

        Returns
        -------
        Any
            Replacement array with identical shape and dtype.

        Raises
        ------
        BackendUnsupportedError
            If the hook result is not an MLX array or changes shape/dtype;
            the MLX preview keeps a strict same-shape/same-dtype contract so
            downstream lazy consumers cannot silently mis-broadcast.
        """

        replacement = plan.applier.apply(leaf)
        if not self.is_tensor(replacement):
            raise BackendUnsupportedError(
                f"MLX intervention {plan.applier.identity!r} returned "
                f"{type(replacement).__name__}; hooks must return an mx.array.",
                code="intervention_result_invalid",
            )
        same_shape = self._shape(replacement) == self._shape(leaf)
        if not same_shape or self._dtype(replacement) != self._dtype(leaf):
            raise BackendUnsupportedError(
                f"MLX intervention {plan.applier.identity!r} changed the output "
                f"from shape={self._shape(leaf)} dtype={self._dtype(leaf)} to "
                f"shape={self._shape(replacement)} dtype={self._dtype(replacement)}; "
                "the MLX backend supports shape- and dtype-preserving interventions only.",
                code="intervention_result_invalid",
            )
        return replacement

    @staticmethod
    def _import_mlx() -> tuple[object, object]:
        """Import MLX lazily.

        Returns
        -------
        tuple[object, object]
            ``mlx.core`` and ``mlx.nn`` modules.
        """

        try:
            import mlx.core as mx
            import mlx.nn as nn
        except ImportError as exc:
            raise ImportError("MLX backend requires the optional 'mlx' package.") from exc
        return mx, nn

    @contextmanager
    def _paused(self) -> Any:
        """Temporarily pause MLX logging."""

        with _state.pause_logging():
            yield

    def _normalize_input_args(self, input_args: object) -> list[Any]:
        """Normalize user MLX input arguments to a positional list."""

        if isinstance(input_args, list):
            return input_args
        if isinstance(input_args, tuple):
            return list(input_args)
        return [input_args]

    def _label_source_arrays(self, trace: Trace, args: list[Any], kwargs: dict[Any, Any]) -> None:
        """Emit resolvable input source events for MLX source arrays."""

        for index, arg in enumerate(args):
            if self.is_tensor(arg):
                label = f"input.arg_{index}"
                self.tensor_store.set_label(arg, label)
                raw_index = trace.capture_events.raw_layer_counter + 1
                trace.capture_events.raw_layer_counter = raw_index
                trace.capture_events.append(
                    self._build_source_event(trace, arg, self._input_reservation(label, raw_index))
                )
        for key, value in kwargs.items():
            if self.is_tensor(value):
                label = f"input.{key}"
                self.tensor_store.set_label(value, label)
                raw_index = trace.capture_events.raw_layer_counter + 1
                trace.capture_events.raw_layer_counter = raw_index
                trace.capture_events.append(
                    self._build_source_event(
                        trace, value, self._input_reservation(label, raw_index)
                    )
                )

    def _capture_policy(self, session: Any) -> CapturePolicy:
        """Return the MLX capture policy for one event (none under structure-only)."""

        run = self._retention_run
        structure_only = run is not None and run.request.structure_only
        return CapturePolicy(
            save_payload=bool(session._options_save_raw_activations) and not structure_only,
            save_grad=False,
        )

    def _payload_policy(
        self,
        session: Any,
        policy: CapturePolicy,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> CapturePolicy:
        """Record one op output's payload and return the policy its event uses.

        A plain capture queues the payload for the single evaluation. Under a
        retention run the live decision keeps the payload public or makes it a
        replay witness, whose event then carries no payload.
        """

        if not policy.save_payload:
            return policy
        run = self._retention_run
        if run is None:
            cast(Any, session)._mlx_saved_payloads.append(output)
            return policy
        context = partial(self.build_record_context, session, reserved, func_event_input, output)
        if run.note(reserved.label_raw, output, context):
            return policy
        return replace(policy, save_payload=False)

    def _settle_capture(self, trace: Trace, hooks: MLXHookSession) -> None:
        """Hand off backward state and apply any retention run's final keep set."""

        retain_backward_state(trace, hooks)
        if self._retention_run is not None:
            self._retention_run.settle(trace)

    @staticmethod
    def _input_reservation(label: str, raw_index: int) -> ReservedLabel:
        """Reserve ``label`` as an input source op at ``raw_index``."""

        return ReservedLabel(
            label=label,
            label_raw=label,
            raw_index=raw_index,
            type_index=raw_index,
            layer_type="input",
            site=label,
        )

    def _build_source_event(
        self,
        trace: Trace,
        output: object,
        reserved: ReservedLabel,
        *,
        func_name: str = "input",
    ) -> OpEvent:
        """Build an MLX source ``OpEvent`` for an input array or a constant.

        Parameters
        ----------
        trace
            The active trace.
        output
            The source array.
        reserved
            The source op's reserved label: an input reservation from
            :meth:`_input_reservation`, or a label reserved through the events
            block (constants). Its ``raw_index`` is the op's raw layer index.
        func_name
            ``"input"`` for an input leaf, ``"constant"`` for a closure
            constant (the shared constants rule, SPEC section 2 rule 1).
        """

        return self._build_event(
            session=trace,
            kind="source",
            reserved=reserved,
            func_event_input=FunctionEventInput(
                func=None,
                func_name=func_name,
                func_qualname=None,
                args=(),
                kwargs={},
                raw_output=output,
                arg_copies=None,
                kwarg_copies=None,
                module_stack=(),
                is_bottom_level_func=True,
                func_call_id=reserved.raw_index,
                expected_output_count=1,
            ),
            output=output,
            parents=(),
            parent_arg_positions={"args": {}, "kwargs": {}},
            edge_uses=(),
            policy=self._capture_policy(trace),
            is_input=func_name == "input",
        )

    def _record_closure_constants(
        self,
        trace: Trace,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        events: CaptureEvents,
    ) -> None:
        """Record closure constants among an op's array arguments (SPEC section 2).

        Every unlabeled array leaf is classified before the op's parent labels
        are read: a parameter of the model tree stays an unlabeled parameter
        leaf; an array reachable from the model object before the forward is a
        closure constant and becomes a ``constant`` source op with a saved
        payload that the consuming op lists as a parent; anything else was
        produced by an op no wrapper saw and is counted for the unwrapped
        policy (it is never laundered into a constant).
        """

        param_ids: frozenset[int] = trace._mlx_param_ids
        external: dict[int, Any] = trace._mlx_external_arrays
        policy = self._capture_policy(trace)
        leaves: list[Any] = []
        for value in args:
            leaves.extend(self._iter_arrays(value))
        for value in kwargs.values():
            leaves.extend(self._iter_arrays(value))
        for leaf in leaves:
            if self.tensor_store.get_label(leaf) is not None or id(leaf) in param_ids:
                continue
            if id(leaf) not in external:
                trace._mlx_unwrapped_intermediates = trace._mlx_unwrapped_intermediates + 1
                continue
            reserved = events.reserve_label_block("constant", 1)[0]
            self.tensor_store.set_label(leaf, reserved.label_raw)
            if policy.save_payload:
                trace._mlx_saved_payloads.append(leaf)
            events.append(self._build_source_event(trace, leaf, reserved, func_name="constant"))
            self._inventory_constant(trace, reserved.label_raw)

    def _inventory_constant(self, trace: Trace, label_raw: str) -> None:
        """Enter a constant leaf in the replay inventory and capture records.

        The constant has no function to replay; validation treats it as a
        leaf whose saved payload must exist (decision reason
        ``constant_producer_leaf``) and replays its consumers from that payload.
        """

        op_captures = getattr(trace, "_mlx_op_captures", None)
        if op_captures is None:
            op_captures = []
            trace._mlx_op_captures = op_captures
        inventory = getattr(trace, "_mlx_replay_inventory", None)
        if inventory is None:
            inventory = []
            trace._mlx_replay_inventory = inventory
        op_captures.append(
            MLXOpCapture(
                labels_raw=(label_raw,),
                op_name="constant",
                func=None,
                args=(),
                kwargs={},
                output=None,
            )
        )
        inventory.append(("constant", (label_raw,), (), (), ()))

    def _build_event(
        self,
        *,
        session: Any,
        kind: str,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
        parents: tuple[ParentEdge, ...],
        parent_arg_positions: dict[str, dict[Any, str]],
        edge_uses: tuple[object, ...],
        policy: CapturePolicy,
        is_input: bool,
        fire_results: tuple[FireResult, ...] = (),
    ) -> OpEvent:
        """Build one topology-complete MLX operation event."""

        tensor_ref = self.tensor_ref(
            session,
            output,
            output if policy.save_payload else None,
            policy,
        )
        input_ancestors = frozenset(
            edge.parent_label_raw for edge in parents if edge.parent_label_raw.startswith("input.")
        )
        pos_args, kw_args = split_non_tensor_args(
            func_event_input.args, func_event_input.kwargs, self.is_tensor
        )
        return OpEvent(
            kind=kind,
            label_raw=reserved.label_raw,
            layer_label_raw=reserved.label_raw,
            layer_type=reserved.layer_type,
            raw_index=reserved.raw_index,
            type_index=reserved.type_index,
            step_index=reserved.raw_index,
            source_trace=session,
            source_trace_id=None,
            tracing_finished=False,
            construction_done=True,
            function=FunctionCallRef(
                func=func_event_input.func,
                func_name=func_event_input.func_name,
                func_qualname=func_event_input.func_qualname,
                func_call_id=func_event_input.func_call_id,
                code_context=mlx_code_context(session),
                func_duration=None,
                flops_forward=None,
                flops_backward=None,
                func_rng_states=None,
                func_autocast_state=None,
                arg_names=(),
                num_args_total=len(func_event_input.args) + len(func_event_input.kwargs),
                num_pos_args=len(func_event_input.args),
                num_kwargs=len(func_event_input.kwargs),
                non_tensor_pos_args=tuple(pos_args),
                non_tensor_kwargs=tuple(kw_args.items()),
                func_non_tensor_args=tuple(pos_args),
                is_inplace=False,
                func_config=(),
            ),
            output=OutputRef(
                tensor=tensor_ref,
                transformed_tensor=None,
                has_saved_activation=policy.save_payload,
                output_device="same",
                activation_transform=session._options_activation_transform,
                detach_saved_activations=bool(session._options_detach_saved_activations),
                visualizer_path=None,
                multi_output_index=None,
                in_multi_output=False,
                container_path=(),
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
            _edge_uses=edge_uses,
            params=(),
            parent_params=(),
            module_stack=func_event_input.module_stack,
            modules=tuple(
                (frame.address, frame.call_index) for frame in func_event_input.module_stack
            ),
            backend_semantics=self.detect_backend_semantics(session, func_event_input, output),
            policy=policy,
            predicate_matched=policy.save_payload,
            pass_index=1,
            grad_fn_class_qualname=None,
            grad_fn_handle=None,
            equivalence_class=reserved.layer_type,
            is_transform=False,
            transform_kind=None,
            transform_chain=(),
            transform_config={},
            transform_fn_name=None,
            transform_fn_qualname=None,
            transform_fn_source=None,
            unattributed_tensor_args=(),
            dropped_edge_tensor_args=(),
            is_output_parent=False,
            has_internal_source_ancestor=not is_input and not parents,
            internal_source_ancestors=frozenset(),
            input_ancestors=input_ancestors,
            root_ancestors=input_ancestors or frozenset({reserved.label_raw}),
            func_call_id=func_event_input.func_call_id,
            is_bottom_level=func_event_input.is_bottom_level_func,
            is_scalar_bool=None,
            bool_value=None,
            intervention_fired=bool(fire_results),
            is_edited=any(result.replaced for result in fire_results),
            fire_results=fire_results,
            intervention_template_ref=None,
            record_context=self.build_record_context(
                session,
                reserved,
                func_event_input,
                output,
            ),
            capture_spec=CaptureSpec(save_out=policy.save_payload, save_metadata=True),
        )

    def _parent_edges(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[ParentEdge, ...], dict[str, dict[Any, str]], tuple[object, ...]]:
        """Return parent edges and arg-position metadata for MLX inputs."""

        edges: list[ParentEdge] = []
        arg_positions: dict[Any, str] = {}
        kwarg_positions: dict[Any, str] = {}
        edge_uses: list[tuple[str, Any, str]] = []

        def _add(label: str, position: Any, use: str) -> None:
            """Append one unique parent edge."""

            if any(edge.parent_label_raw == label for edge in edges):
                return
            edges.append(ParentEdge(parent_label_raw=label, arg_position=position, _edge_use=use))
            edge_uses.append((label, position, use))

        for index, value in enumerate(args):
            for array in self._iter_arrays(value):
                label = self.tensor_store.get_label(array)
                if label is not None:
                    arg_positions[index] = label
                    _add(label, index, "arg")
        for key, value in kwargs.items():
            for array in self._iter_arrays(value):
                label = self.tensor_store.get_label(array)
                if label is not None:
                    kwarg_positions[key] = label
                    _add(label, key, "kwarg")
        return (tuple(edges), {"args": arg_positions, "kwargs": kwarg_positions}, tuple(edge_uses))

    def _mark_outputs(self, trace: Trace, output: object) -> None:
        """Mark final output-parent operations and settle the unwrapped-op gap."""

        unlabeled = 0
        for value in self._iter_arrays(output):
            label = self.tensor_store.get_label(value)
            if label is None:
                unlabeled += 1
                continue
            mark_output_label(trace, label)
        intermediates = int(trace.__dict__.pop("_mlx_unwrapped_intermediates", 0))
        if unlabeled or intermediates:
            self._settle_unwrapped_gap(trace, unlabeled, intermediates)

    def _settle_unwrapped_gap(self, trace: Trace, outputs: int, intermediates: int) -> None:
        """Flag or refuse a capture that used an op no wrapper saw.

        Parameters
        ----------
        trace
            The trace being finalized.
        outputs
            Output arrays no recorded op produced.
        intermediates
            Arguments of recorded ops that were neither inputs, parameters,
            closure constants nor recorded outputs.

        Raises
        ------
        BackendUnsupportedError
            Under the ``"refuse"`` unwrapped policy, code
            ``unwrapped_op_in_capture``.
        """

        message = (
            f"MLX capture: {outputs} output array(s) and {intermediates} intermediate "
            "array(s) were not produced by a wrapped operation (a function outside the "
            "wrapped mx, mx.fast, mx.array and nn surface, or an array built inside the "
            "forward), so at least one op is missing from the trace."
        )
        if get_unwrapped_policy() == "refuse":
            raise BackendUnsupportedError(
                message + " The MLX unwrapped policy is 'refuse'; capture under the default 'flag' "
                "policy to keep the trace marked trace.provenance.verified=False instead",
                code=UNWRAPPED_OP_REASON,
            )

        warnings.warn(
            TorchLensWarning(
                message + " This capture is marked trace.provenance.verified=False. "
                "Remedy: route the computation through the wrapped mx, mx.fast or nn "
                "surface, or set the MLX unwrapped policy to 'refuse' to make the gap an error",
                code="unwrapped_op_in_capture",
            ),
            stacklevel=4,
        )
        trace._provenance_verified = False
        trace._provenance_verification_reason = UNWRAPPED_OP_REASON

    def _finish_trace(self, trace: Trace, module_tree: MLXModuleTree | None = None) -> None:
        """Finalize a manually captured MLX Trace.

        Parameters
        ----------
        trace
            Trace to finalize.
        module_tree
            Discovered object-module tree, if object-module mode is active.

        Returns
        -------
        None
            Trace accessors are populated in place.
        """

        # The MLX validation sidecars (`_mlx_op_captures`, `_mlx_replay_inventory`)
        # speak RAW label space on both the build and consume sides, and raw
        # labels are never rewritten by recurrence grouping (they stay resolvable
        # through `_label_raw` with recurrence-safe key precedence in
        # `_ops_by_label`), so no relabel hook is needed here.
        finalize_single_pass_trace(
            trace,
            backend_name=self.name,
            module_tree=module_tree,
            attach_function_root_module=attach_function_root_module,
            attach_object_module_logs=self._attach_object_module_logs,
            attach_op_params=attach_module_owned_op_params,
            count_layers_with_attached_params=True,
            recurrence_detection=bool(trace._options_recurrence_detection),
        )
        adopt_code_context_records(trace)

    def _attach_object_module_logs(self, trace: Trace, tree: MLXModuleTree) -> None:
        """Build public module logs for an MLX object-module trace.

        Parameters
        ----------
        trace
            Trace receiving module accessors.
        tree
            Discovered MLX object-module tree.

        Returns
        -------
        None
            ``trace.modules`` is populated by the shared module-log builder.
        """

        attach_object_module_logs_with_returns(
            trace,
            tree,
            normalize_module_calls=normalize_op_module_calls,
            metadata_top_level=_mlx_metadata_top_level,
            op_top_level=_mlx_op_top_level,
            training_mode=_mlx_training_mode,
        )

    def _iter_arrays(self, value: object) -> list[object]:
        """Return MLX arrays nested inside ``value``."""

        if self.is_tensor(value):
            return [value]
        if isinstance(value, (list, tuple)):
            arrays: list[object] = []
            for item in value:
                arrays.extend(self._iter_arrays(item))
            return arrays
        if isinstance(value, dict):
            arrays = []
            for item in value.values():
                arrays.extend(self._iter_arrays(item))
            return arrays
        return []

    def _shape(self, value: object) -> tuple[int, ...] | None:
        """Return an MLX array shape without materializing data."""

        return tuple(cast(Any, value).shape) if self.is_tensor(value) else None

    def _dtype(self, value: object) -> str | None:
        """Return an MLX array dtype without materializing data."""

        return str(cast(Any, value).dtype) if self.is_tensor(value) else None

    def _device(self, value: object) -> str | None:
        """Return an MLX array device description without materializing data."""

        if not self.is_tensor(value):
            return None
        device = getattr(value, "device", None)
        return str(device) if device is not None else None

    def _memory(self, value: object) -> int | None:
        """Return MLX array memory in bytes without materializing data."""

        if not self.is_tensor(value):
            return None
        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        size = getattr(value, "size", None)
        itemsize = getattr(value, "itemsize", None)
        if size is not None and itemsize is not None:
            return int(size) * int(itemsize)
        return None


def mlx_param_logs(tree: MLXModuleTree, trace: Trace) -> dict[str, Param]:
    """Build TorchLens parameter logs from an MLX module parameter tree.

    Parameters
    ----------
    tree
        Discovered MLX object-module tree.
    trace
        Trace receiving the parameter logs.

    Returns
    -------
    dict[str, Param]
        Parameter logs keyed by primary parameter address.
    """

    param_logs: dict[str, Param] = {}
    trainable_ids = {id(value) for _address, value in _iter_trainable_parameter_tree(tree.root)}
    for candidate in _iter_mlx_parameter_candidates(tree):
        existing_address = tree.param_address_by_id.get(id(candidate.value), candidate.address)
        if existing_address in param_logs:
            param = param_logs[existing_address]
            if candidate.address not in param.all_addresses:
                param.all_addresses.append(candidate.address)
            if candidate.address not in param.co_parent_params:
                param.co_parent_params.append(candidate.address)
            for alias in tree.metadata.get(candidate.owner, {}).get(
                "all_addresses", [candidate.owner]
            ):
                if alias not in param.all_module_addresses:
                    param.all_module_addresses.append(alias)
            continue
        shape = tuple(getattr(candidate.value, "shape", ()))
        dtype = str(getattr(candidate.value, "dtype", ""))
        param = Param(
            module_address=candidate.owner,
            name=candidate.address.rsplit(".", 1)[-1],
            shape=shape,
            dtype=cast(Any, dtype),
            num_params=_numel(shape),
            param_memory=_nbytes(candidate.value) or 0,
            trainable=id(candidate.value) in trainable_ids,
            address=existing_address,
            barcode=f"mlx:{existing_address}",
            has_optimizer=None,
        )
        param.dtype_ref = DtypeRef(backend="mlx", name=dtype)
        param.device_ref = DeviceRef.from_value(getattr(candidate.value, "device", None))
        param.backend_address = f"object:{existing_address}"
        param.resolver_status = "resolved"
        param._param_ref = cast(Any, candidate.value)
        param.trace = trace
        param.all_module_addresses = list(
            tree.metadata.get(candidate.owner, {}).get("all_addresses", [candidate.owner])
        )
        param_logs[existing_address] = param
    return param_logs


def _iter_mlx_parameter_candidates(tree: MLXModuleTree) -> list[MLXParameterCandidate]:
    """Return flattened MLX parameter candidates with primary owners.

    Parameters
    ----------
    tree
        Discovered MLX object-module tree.

    Returns
    -------
    list[MLXParameterCandidate]
        Parameter candidates.
    """

    alias_to_primary = _alias_to_primary(tree)
    candidates: list[MLXParameterCandidate] = []
    for param_address, value in _iter_parameter_tree(tree.root):
        owner_alias = param_address.rsplit(".", 1)[0] if "." in param_address else "self"
        owner = alias_to_primary.get(owner_alias, owner_alias)
        primary_param_address = _join_module_address(owner, param_address.rsplit(".", 1)[-1])
        candidates.append(
            MLXParameterCandidate(
                address=primary_param_address,
                value=value,
                owner=owner,
            )
        )
    return candidates


def _mlx_metadata_top_level(
    address: str,
    metadata: dict[str, Any],
    metadata_by_address: dict[str, dict[str, Any]],
) -> bool:
    """Return whether an MLX metadata address is top-level.

    Parameters
    ----------
    address:
        Module address from the discovered MLX module tree.
    metadata:
        Metadata for ``address``, unused by MLX.
    metadata_by_address:
        Complete module metadata mapping.

    Returns
    -------
    bool
        True when the nearest discovered parent is ``"self"``.
    """

    del metadata
    return address != "self" and nearest_metadata_parent(address, metadata_by_address) == "self"


def _mlx_op_top_level(address: str) -> bool:
    """Return whether an MLX op module address is top-level.

    Parameters
    ----------
    address:
        Module address observed in an op call stack.

    Returns
    -------
    bool
        True for every non-root first frame, matching the previous MLX loop.
    """

    return address != "self"


def _mlx_training_mode(metadata: dict[str, Any]) -> bool:
    """Return MLX module training state from metadata.

    Parameters
    ----------
    metadata:
        Module metadata from ``MLXModuleTree``.

    Returns
    -------
    bool
        Stored training-state flag, defaulting to ``False``.
    """

    return bool(metadata.get("training", False))


def _resolve_mlx_module_identity_mode(
    value: str | None,
    module_tree: MLXModuleTree | None,
) -> bool:
    """Return whether MLX should use object-module attribution.

    Parameters
    ----------
    value
        Public ``module_identity_mode`` value after missing normalization.
    module_tree
        Discovered MLX module tree, if any.

    Returns
    -------
    bool
        True when object-module mode should be used.
    """

    if value not in {None, "function_root", "object_module"}:
        raise BackendUnsupportedError(
            "MLX module_identity_mode must be None, 'function_root', or 'object_module'."
        )
    if value == "object_module" and module_tree is None:
        raise BackendUnsupportedError(
            "MLX module_identity_mode='object_module' requires an mlx.nn.Module object. "
            "Raw callables use module_identity_mode='function_root'."
        )
    if value == "function_root":
        return False
    return module_tree is not None


def _normalize_mlx_input_grad_argnums(
    input_grad_argnums: Sequence[int],
    num_args: int,
) -> tuple[int, ...]:
    """Validate MLX input-relative gradient argnums.

    Parameters
    ----------
    input_grad_argnums
        User-supplied positional input indexes.
    num_args
        Number of positional model inputs.

    Returns
    -------
    tuple[int, ...]
        Normalized unique input argnums.
    """

    normalized = tuple(int(index) for index in input_grad_argnums)
    if len(set(normalized)) != len(normalized):
        raise ValueError("MLX derived gradient input_grad_argnums must be unique.")
    for index in normalized:
        if index < 0 or index >= num_args:
            raise ValueError("MLX derived gradient input_grad_argnums are out of range.")
    return normalized


__all__ = ["GradOptions", "MLXBackend"]
