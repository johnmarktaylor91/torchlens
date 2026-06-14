"""UOp-snapshot tinygrad backend preview."""

from __future__ import annotations

import time
import inspect
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, cast

from ..._deprecations import MISSING, MissingType
from ...backends import BackendName, BackendUnsupportedError
from ...data_classes.derived_grad import (
    DerivedGradAccessor,
    DerivedGradRecord,
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
)
from ...data_classes.layer import Layer
from ...data_classes.module import ModuleAccessor
from ...data_classes.param import Param, ParamAccessor
from ...data_classes.trace import Trace
from ...data_classes.trace import _init_module_hierarchy_data
from ...fastlog.types import CaptureSpec
from ...ir.buffer import CaptureEvents
from ...ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
    OpEvent,
    OutputRef,
    ParentEdge,
)
from ...ir.predicate import RecordContext
from ...ir.refs import DeviceRef, DtypeRef, ParamRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...postprocess._materialize import materialize_from_events
from ...postprocess.finalization import _build_module_logs
from ...postprocess.finalization import _build_root_module_log
from ...quantities import Duration
from ...validation.status import ValidationReplayStatus
from .._selective_save import apply_static_label_save_policy
from .._selective_save import pop_static_label_save_predicate

_ACTIVE_TINYGRAD_MODULE_STACK: list["TinygradModuleFrame"] = []


@dataclass(frozen=True)
class TinygradUOpCapture:
    """Captured tinygrad UOp metadata for validation.

    Parameters
    ----------
    label_raw
        Raw TorchLens label for the materialized op.
    uop
        Source tinygrad UOp snapshot.
    op_name
        UOp operation name.
    parent_labels
        Raw parent labels captured from UOp source edges.
    parent_arg_positions
        UOp source positions for each raw parent label.
    payload_snapshot
        Realized tinygrad tensor copy saved during capture.
    """

    label_raw: str
    uop: Any
    op_name: str
    parent_labels: tuple[str, ...]
    parent_arg_positions: tuple[tuple[int, str], ...]
    payload_snapshot: Any


@dataclass(frozen=True)
class TinygradIntermediateCandidate:
    """Live retained tinygrad tensor candidate for one op-level gradient.

    Parameters
    ----------
    signature
        Conservative structural signature used for trace-op matching.
    tensor
        Live tinygrad Tensor retained from the no-realize forward pass.
    aval
        Human-readable abstract-value description.
    dtype_ref
        Backend-neutral dtype reference for the tensor.
    """

    signature: tuple[Any, ...]
    tensor: Any
    aval: str
    dtype_ref: DtypeRef | None


@dataclass(frozen=True)
class GradOptions:
    """tinygrad derived-gradient preview options.

    Parameters
    ----------
    loss_fn
        Optional callable mapping raw function output to a scalar tinygrad loss.
        Required unless the raw traced output is already scalar.
    input_grad_argnums
        Positional input argument indexes to differentiate. ``None`` means all
        positional tinygrad tensor input leaves.
    intermediate_grads
        Whether to run the opt-in no-realize pass for op-level intermediate
        derived gradients.
    """

    loss_fn: Callable[[Any], Any] | None = None
    input_grad_argnums: tuple[int, ...] | None = None
    intermediate_grads: bool = False

    def __init__(
        self,
        *,
        loss_fn: Callable[[Any], Any] | None = None,
        input_grad_argnums: Sequence[int] | None = None,
        intermediate_grads: bool = False,
    ) -> None:
        """Initialize tinygrad derived-gradient options.

        Parameters
        ----------
        loss_fn
            Callable mapping raw output to scalar loss, or ``None`` for scalar
            raw outputs.
        input_grad_argnums
            Positional input argnums to differentiate. ``None`` differentiates
            all positional tinygrad tensor input leaves.
        intermediate_grads
            Whether to retain saved op outputs during a separate no-realize
            backward pass and expose ``trace.intermediate_derived_grads``.
        """

        object.__setattr__(self, "loss_fn", loss_fn)
        normalized = None if input_grad_argnums is None else tuple(input_grad_argnums)
        object.__setattr__(self, "input_grad_argnums", normalized)
        object.__setattr__(self, "intermediate_grads", bool(intermediate_grads))


@dataclass(frozen=True)
class TinygradModuleFrame:
    """One observed tinygrad object-module stack frame.

    Parameters
    ----------
    address
        Primary TorchLens module address.
    call_index
        One-based call index for the primary address.
    module_type
        Runtime class name for the module object.
    """

    address: str
    call_index: int
    module_type: str


@dataclass(frozen=True)
class TinygradModuleTree:
    """Discovered tinygrad object-module tree.

    Parameters
    ----------
    root
        Root callable object.
    metadata
        TorchLens module metadata keyed by primary address.
    address_by_id
        Primary module address keyed by object identity.
    modules_by_class
        Module instance addresses grouped by class for scoped ``__call__`` patching.
    param_owner_by_address
        Owning module address keyed by parameter address.
    param_address_by_uop_id
        Parameter address keyed by exact tensor UOp object identity.
    call_counts
        Per-primary-address call counts recorded during capture.
    forward_args_by_call
        Forward args keyed by ``(primary_address, call_index)``.
    """

    root: Any
    metadata: dict[str, dict[str, Any]]
    address_by_id: dict[int, str]
    modules_by_class: dict[type[Any], dict[int, str]]
    param_owner_by_address: dict[str, str]
    param_address_by_uop_id: dict[int, str]
    call_counts: dict[str, int]
    forward_args_by_call: dict[tuple[str, int], tuple[tuple[Any, ...], dict[str, Any]]]


class TinygradBackend:
    """tinygrad adapter that captures forward graphs from pre-realization UOps."""

    name = "tinygrad"

    def capture_trace(
        self,
        model: Callable[..., Any],
        input_args: object,
        input_kwargs: dict[Any, Any] | None = None,
        *,
        layers_to_save: str | list[Any] | None | MissingType = MISSING,
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
        verbose: bool | MissingType = MISSING,
        backward_ready: bool | MissingType = MISSING,
        name: str | None | MissingType = MISSING,
        module_filter: object | None = None,
        transform: object | None = None,
        raw_input: object | None = None,
        save_raw_input: str | bool | MissingType = MISSING,
        batch_render: str | MissingType = MISSING,
        output_transform: object | None = None,
        save_raw_output: str | bool | MissingType = MISSING,
        layer_visualizers: dict[Any, Any] | None = None,
        save_visualizations: bool | MissingType = MISSING,
        lookback: int = 0,
        lookback_payload_policy: str = "metadata_only",
        module_identity_mode: str | None | MissingType = MISSING,
        grad_options: GradOptions | None | MissingType = MISSING,
        **kwargs: Any,
    ) -> Trace:
        """Capture a tinygrad raw-function forward pass into a TorchLens trace.

        Parameters
        ----------
        model
            Callable accepting tinygrad tensors as positional inputs.
        input_args
            Positional arguments for ``model``.
        input_kwargs
            Keyword arguments. Unsupported in this preview.
        layers_to_save
            Must be ``"all"``; tinygrad preview is full-save only for live traces.
        keep_orphans
            Whether orphan ops are retained.
        output_device
            Must be ``"same"``.
        activation_transform
            Unsupported for tinygrad in this preview.
        save_raw_activations
            Must be true.
        detach_saved_activations
            Must be false.
        save_grads
            Unsupported; true backward graph capture is not available.
        random_seed
            Unsupported torch-style RNG surface.
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
            Unsupported for tinygrad in this preview.
        name
            Trace label.
        module_filter
            Unsupported for tinygrad in this preview.
        transform
            Unsupported for tinygrad in this preview.
        raw_input
            Original user input.
        save_raw_input
            Raw-input save policy.
        batch_render
            Raw-input render policy.
        output_transform
            Optional metadata transform for final output.
        save_raw_output
            Raw-output save policy.
        layer_visualizers
            Unsupported for tinygrad in this preview.
        save_visualizations
            Unsupported for tinygrad in this preview.
        lookback
            Predicate lookback window. Only the default ``0`` is supported.
        lookback_payload_policy
            Predicate lookback payload policy. Only the default is supported.
        module_identity_mode
            Optional tinygrad module mode. Raw callables use ``"function_root"``;
            discovered callable object graphs can use ``"object_module"``.
        grad_options
            tinygrad ``GradOptions`` for bracketed leaf-level derived gradients.
        **kwargs
            Extra public trace kwargs rejected by this backend.

        Returns
        -------
        Trace
            Captured tinygrad trace.
        """

        save_predicate = pop_static_label_save_predicate(kwargs, backend_name="tinygrad")
        self._reject_extra_kwargs(kwargs)
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
        verbose = _default_if_missing(verbose, False)
        backward_ready = _default_if_missing(backward_ready, False)
        name = _default_if_missing(name, None)
        save_raw_input = _default_if_missing(save_raw_input, "small")
        batch_render = _default_if_missing(batch_render, "auto")
        save_raw_output = _default_if_missing(save_raw_output, "small")
        save_visualizations = _default_if_missing(save_visualizations, False)
        activation_transform = None if _is_missing(activation_transform) else activation_transform
        save_grads = None if _is_missing(save_grads) else save_grads
        module_filter = None if _is_missing(module_filter) else module_filter
        transform = None if _is_missing(transform) else transform
        output_transform = None if _is_missing(output_transform) else output_transform
        layer_visualizers = None if _is_missing(layer_visualizers) else layer_visualizers
        module_identity_mode = _default_if_missing(module_identity_mode, None)
        grad_options = None if _is_missing(grad_options) else grad_options
        args = self._normalize_input_args(input_args)
        module_tree = discover_tinygrad_module_tree(model)
        use_object_module = _resolve_tinygrad_module_identity_mode(
            cast(str | None, module_identity_mode),
            module_tree,
        )
        self._reject_unsupported_options(
            layers_to_save=layers_to_save,
            input_kwargs=input_kwargs,
            output_device=output_device,
            activation_transform=activation_transform,
            save_raw_activations=save_raw_activations,
            detach_saved_activations=detach_saved_activations,
            save_grads=save_grads,
            save_arg_values=save_arg_values,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            backward_ready=backward_ready,
            module_filter=module_filter,
            transform=transform,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,
        )
        self._assert_runtime_supported()
        self._assert_tinygrad_inputs(args)
        trace = self._new_trace(
            model=model,
            keep_orphans=cast(bool, keep_orphans),
            num_context_lines=cast(int, num_context_lines),
            recurrence_detection=cast(bool, recurrence_detection),
            verbose=cast(bool, verbose),
            name=cast(str | None, name),
            raw_input=raw_input,
            save_raw_input=cast(str | bool, save_raw_input),
            batch_render=cast(str, batch_render),
            output_transform=output_transform,
            save_raw_output=cast(str | bool, save_raw_output),
        )
        trace.capture_events = CaptureEvents()
        trace.capture_start_time = time.time()
        observed_ops: dict[int, list[str]] = {}
        observed_module_stacks: dict[int, tuple[TinygradModuleFrame, ...]] = {}
        input_identities = self._input_identities(args)
        module_call_context = (
            scoped_tinygrad_module_calls(module_tree, observed_module_stacks)
            if use_object_module and module_tree is not None
            else _null_context()
        )
        with module_call_context:
            with (
                _observe_tensor_ops(observed_ops, observed_module_stacks),
                _reject_mid_capture_execution(),
            ):
                output = model(*args)
        if self._input_identities(args) != input_identities:
            raise BackendUnsupportedError(
                "tinygrad backend preview cannot capture Tensor.assign(), Tensor.replace(), "
                "or setitem input mutation inside the traced callable yet; return a pure lazy "
                "tinygrad expression instead."
            )
        outputs = tuple(self._tensor_leaves(output))
        if not outputs:
            raise BackendUnsupportedError(
                "tinygrad backend preview requires at least one tinygrad Tensor output."
            )
        trace.forward_duration = Duration(time.time() - trace.capture_start_time)
        trace.raw_output = output_transform(output) if callable(output_transform) else None
        uop_labels = self._emit_input_sources(trace, args)
        captures = self._emit_uop_graph(
            trace,
            outputs,
            uop_labels,
            observed_ops,
            observed_module_stacks,
            module_tree if use_object_module else None,
        )
        self._mark_output_events(trace, outputs, uop_labels, captures)
        if use_object_module and module_tree is not None:
            trace.param_logs = ParamAccessor(tinygrad_param_logs(module_tree, trace))
            trace.num_param_tensors = len(trace.param_logs)
            trace.num_params = sum(param.num_params for param in trace.param_logs)
            trace.num_params_trainable = sum(
                param.num_params for param in trace.param_logs if param.is_trainable
            )
            trace.num_params_frozen = trace.num_params - trace.num_params_trainable
            trace.param_source = "native-module"
        else:
            trace.param_logs = ParamAccessor({})
            trace.num_param_tensors = 0
            trace.num_params = 0
            trace.num_params_trainable = 0
            trace.num_params_frozen = 0
            trace.param_source = "none"
        materialize_from_events(trace, trace.capture_events)
        delattr(trace, "capture_events")
        trace.num_layers_with_params = 0
        self._finish_trace(trace, module_tree if use_object_module else None)
        trace.tinygrad_uop_captures = captures
        trace.tinygrad_payload_policy = "dev_python_realized_copy"
        apply_static_label_save_policy(trace, save_predicate, backend_name="tinygrad")
        if grad_options is not None:
            self._attach_derived_grads(
                trace=trace,
                model=model,
                args=args,
                captured_output=output,
                grad_options=cast(GradOptions, grad_options),
            )
        return trace

    def validate_trace(
        self,
        trace: Trace,
        *_args: Any,
        **kwargs: Any,
    ) -> bool | ValidationReplayStatus:
        """Validate a tinygrad trace using UOp replay and metadata invariants.

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
        bool or ValidationReplayStatus
            True when replayed UOp payloads match saved live payload copies.
            Loaded traces whose runtime capture was stripped return an explicit
            unavailable status.
        """

        status = trace.validation_replay_status
        if not status.available:
            setattr(trace, "_validation_replay_status", status)
            return status
        try:
            if kwargs.get("validate_metadata", True):
                from ...validation.invariants import check_metadata_invariants

                check_metadata_invariants(trace)
            passed = self._validate_uops(trace)
        except BackendUnsupportedError:
            raise
        except Exception:
            passed = False
        setattr(
            trace,
            "_validation_replay_status",
            ValidationReplayStatus.result(
                passed=passed,
                backend="tinygrad",
                source="loaded" if getattr(trace, "_loaded_from_bundle", False) else "live",
                payload_load_status=getattr(trace, "payload_load_status", None),
            ),
        )
        return passed

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Capture then validate a tinygrad forward pass.

        Parameters
        ----------
        *args, **kwargs
            Public validation arguments.

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
                "tinygrad validation entry unexpectedly produced a replay-unavailable status for "
                "a freshly captured live trace."
            )
        return result

    def is_tensor(self, value: object) -> bool:
        """Return whether ``value`` is a tinygrad tensor.

        Parameters
        ----------
        value
            Candidate value.

        Returns
        -------
        bool
            True for tinygrad tensors.
        """

        try:
            from tinygrad import Tensor
        except ImportError:
            return False
        return isinstance(value, Tensor)

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
        save_raw_output: str | bool,
    ) -> Trace:
        """Construct an empty tinygrad trace.

        Parameters
        ----------
        model
            Captured callable.
        keep_orphans
            Whether orphan ops are retained.
        num_context_lines
            Source context line count.
        recurrence_detection
            Recurrence-detection setting.
        verbose
            Verbose flag.
        name
            Optional trace label.
        raw_input
            Original user input.
        save_raw_input
            Raw-input save policy.
        batch_render
            Raw-input render policy.
        output_transform
            Optional output transform.
        save_raw_output
            Raw-output save policy.

        Returns
        -------
        Trace
            Empty trace initialized for tinygrad.
        """

        trace = Trace(
            model_class_name=getattr(model, "__name__", type(model).__name__),
            output_device="same",
            activation_transform=None,
            grad_transform=None,
            save_raw_activations=True,
            save_raw_gradients=True,
            keep_orphans=keep_orphans,
            save_arg_values=False,
            save_grads=None,
            detach_saved_activations=False,
            mark_layer_depths=False,
            num_context_lines=num_context_lines,
            optimizer=None,
            save_code_context=False,
            save_rng_states=False,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            backward_ready=False,
            module_filter=None,
            emit_nvtx=False,
            transform=None,
            raw_input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=cast("Callable[[Any], Any] | None", output_transform),
            save_raw_output=save_raw_output,
            layer_visualizers=None,
            save_visualizations=False,
        )
        trace.trace_label = name
        trace.backend = cast(BackendName, self.name)
        trace.module_identity_mode = "function_root"
        trace.param_source = "none"
        trace.model_label = trace.model_class_name
        trace.model_class_qualname = getattr(model, "__qualname__", trace.model_class_name)
        trace._pre_forward_rng_states = None
        return trace

    def _emit_input_sources(self, trace: Trace, args: Sequence[Any]) -> dict[int, str]:
        """Emit source events for positional tinygrad tensor inputs.

        Parameters
        ----------
        trace
            Trace receiving events.
        args
            Normalized positional call arguments.

        Returns
        -------
        dict[int, str]
            Mapping from UOp object id to raw source label.
        """

        uop_labels: dict[int, str] = {}
        for path, value in _tree_leaves_with_paths(tuple(args)):
            if not self.is_tensor(value):
                continue
            event = self._append_event(
                trace=trace,
                kind="source",
                layer_type="input",
                func_name="input",
                output=self._realized_copy(value),
                parents=(),
                parent_arg_positions={"args": {}, "kwargs": {}},
                module_stack=(),
                params=(),
                container_path=tuple(path.split(".")),
                annotations={
                    "tinygrad_container_path": path,
                    "tinygrad_identity": _identity(value),
                },
            )
            uop_labels[id(cast(Any, value).uop)] = event.label_raw
        return uop_labels

    def _emit_uop_graph(
        self,
        trace: Trace,
        outputs: Sequence[Any],
        uop_labels: dict[int, str],
        observed_ops: Mapping[int, list[str]],
        observed_module_stacks: Mapping[int, tuple[TinygradModuleFrame, ...]],
        module_tree: TinygradModuleTree | None,
    ) -> tuple[TinygradUOpCapture, ...]:
        """Emit one event for each tensor-shaped UOp reachable from outputs.

        Parameters
        ----------
        trace
            Trace receiving events.
        outputs
            Flat tinygrad output tensors.
        uop_labels
            Existing UOp label mapping seeded with source inputs.
        observed_ops
            Tensor API observations keyed by returned UOp id.
        observed_module_stacks
            First-observed live module stack keyed by returned UOp id.
        module_tree
            Discovered tinygrad module tree for object-module captures, if any.

        Returns
        -------
        tuple[TinygradUOpCapture, ...]
            Captures used by live validation.
        """

        captures: list[TinygradUOpCapture] = []
        uops = _unique_uops(outputs)
        for uop in uops:
            if id(uop) in uop_labels or not _is_materializable_uop(uop):
                continue
            op_name = _uop_name(uop)
            parents = tuple(
                ParentEdge(parent_label_raw=label, arg_position=index, edge_use="arg")
                for index, src in enumerate(getattr(uop, "src", ()) or ())
                if (label := uop_labels.get(id(src))) is not None
            )
            parent_positions = {
                "args": {edge.arg_position: edge.parent_label_raw for edge in parents},
                "kwargs": {},
            }
            module_stack = _module_stack_for_uop(uop, observed_module_stacks, module_tree)
            param_refs = _param_refs_for_uop(uop, module_tree)
            tensor = self._tensor_from_uop(uop)
            payload = self._realized_copy(tensor)
            event = self._append_event(
                trace=trace,
                kind="op",
                layer_type=op_name.lower(),
                func_name=(observed_ops.get(id(uop)) or [op_name.lower()])[-1],
                output=payload,
                parents=parents,
                parent_arg_positions=parent_positions,
                module_stack=module_stack,
                params=param_refs,
                container_path=(),
                annotations={
                    "tinygrad_uop": op_name,
                    "tinygrad_uop_signature": _uop_signature(uop),
                    "tinygrad_observed_tensor_ops": tuple(observed_ops.get(id(uop), ())),
                    "tinygrad_identity": _identity(tensor),
                },
            )
            uop_labels[id(uop)] = event.label_raw
            captures.append(
                TinygradUOpCapture(
                    label_raw=event.label_raw,
                    uop=uop,
                    op_name=op_name,
                    parent_labels=tuple(edge.parent_label_raw for edge in parents),
                    parent_arg_positions=tuple(
                        (cast(int, edge.arg_position), edge.parent_label_raw) for edge in parents
                    ),
                    payload_snapshot=payload,
                )
            )
        return tuple(captures)

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
        module_stack: tuple[TinygradModuleFrame, ...],
        params: tuple[ParamRef, ...],
        container_path: tuple[object, ...],
        annotations: Mapping[str, object],
    ) -> OpEvent:
        """Append one tinygrad event to the trace event stream.

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
        module_stack
            Active tinygrad object-module stack for this UOp.
        params
            Parameter references consumed by this UOp.
        container_path
            Output container path.
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
            must_keep_topology=True,
            save_payload=True,
            requires_isolation=False,
            save_args=False,
            save_code=False,
            save_rng=False,
            save_grad=False,
            stream=False,
        )
        tensor_ref = self._tensor_ref(output, reserved.label_raw)
        event_module_stack = tuple(
            ModuleFrame(
                address=frame.address,
                address_normalized=frame.address,
                module_type=frame.module_type,
                call_index=frame.call_index,
                fx_qualpath=None,
                entry_argnames=(),
            )
            for frame in module_stack
        )
        event_modules = tuple((frame.address, frame.call_index) for frame in module_stack)
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
                (edge.parent_label_raw, edge.arg_position, edge.edge_use) for edge in parents
            ),
            params=params,
            parent_params=(),
            module_stack=event_module_stack,
            modules=event_modules,
            backend_semantics=BackendSemantics(
                backend_grad_handle=None,
                grad_fn_class_name=None,
                autograd_memory=None,
                num_autograd_tensors=None,
                mutated_input_positions=(),
                aliased_output_inputs=(),
                unknown_aliasing=False,
                bytes_delta_at_call=0,
                bytes_peak_at_call=0,
            ),
            policy=policy,
            predicate_matched=True,
            pass_index=1,
            grad_fn_class_qualname=None,
            grad_fn_handle=None,
            equivalence_class=layer_type,
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
            intervention_replaced=False,
            fire_results=(),
            intervention_template_ref=None,
            record_context=self._record_context(reserved, output, func_name),
            capture_spec=CaptureSpec(save_out=True, save_metadata=True),
        )
        trace.capture_events.append(event)
        return event

    def _tensor_ref(self, value: object, label_raw: str) -> TensorRef:
        """Build a tensor reference for a tinygrad payload.

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
            return TensorRef(label_raw, None, None, None, None, None, value, None, str(id(value)))
        tensor = cast(Any, value)
        return TensorRef(
            label_raw=label_raw,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            requires_grad=getattr(tensor, "requires_grad", None),
            memory=_nbytes(tensor),
            payload=value,
            blob_ref=None,
            backend_handle_id=_identity(tensor),
        )

    def _record_context(
        self, reserved: ReservedLabel, output: object, func_name: str
    ) -> RecordContext:
        """Build a lightweight predicate context for a tinygrad event.

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
            pass_index=1,
            event_index=reserved.raw_index,
            step_index=None,
            layer_type=reserved.layer_type,
            type_index=reserved.type_index,
            raw_index=reserved.raw_index,
            func_name=func_name,
            address=None,
            module_type=None,
            module_pass_index=None,
            module_stack=(),
            recent_events=(),
            recent_ops=(),
            parent_labels=(),
            input_output_address=None,
            shape=tuple(cast(Any, output).shape) if self.is_tensor(output) else None,
            dtype=DtypeRef(backend="tinygrad", name=str(getattr(output, "dtype", "")))
            if self.is_tensor(output)
            else None,
            tensor_device=DeviceRef.from_value(getattr(output, "device", None)),
            tensor_requires_grad=getattr(output, "requires_grad", None),
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
        uop_labels: Mapping[int, str],
        captures: Sequence[TinygradUOpCapture] = (),
    ) -> None:
        """Mark final output UOps as output parents.

        Parameters
        ----------
        trace
            Trace whose events are updated.
        outputs
            Flat tinygrad output tensors.
        uop_labels
            Mapping from UOp object id to raw labels.
        captures
            Captured UOps for structural fallback when tinygrad returns
            equivalent UOp objects with different Python identities.

        Returns
        -------
        None
            Output-parent flags are updated in place.
        """

        labels_by_signature = {
            _uop_signature(capture.uop): capture.label_raw for capture in captures
        }
        output_labels: list[str] = []
        for output in outputs:
            output_uop = cast(Any, output).uop
            label = uop_labels.get(id(output_uop))
            if label is None:
                label = labels_by_signature.get(_uop_signature(output_uop))
            if label is None:
                label = _fallback_output_label(output, captures, output_labels)
            if label is not None:
                output_labels.append(label)
        labels = tuple(output_labels)
        is_multi_output = len(labels) > 1
        for leaf_index, label in enumerate(labels):
            event = trace.capture_events.op_event_by_label_raw.get(label)
            if event is None:
                continue
            trace.output_layers.append(label)
            updated_output = replace(
                event.output,
                multi_output_index=leaf_index if is_multi_output else None,
                in_multi_output=is_multi_output,
                container_path=(leaf_index,) if is_multi_output else (),
            )
            updated = replace(event, is_output_parent=True, output=updated_output)
            trace.capture_events.op_event_by_label_raw[label] = updated
            trace.capture_events.live_index.replace(updated)
            for index, candidate in enumerate(trace.capture_events.op_events):
                if candidate.label_raw == label:
                    trace.capture_events.op_events[index] = updated
                    break

    def _finish_trace(self, trace: Trace, module_tree: TinygradModuleTree | None = None) -> None:
        """Finalize materialized tinygrad raw logs into public accessors.

        Parameters
        ----------
        trace
            Trace to finalize.
        module_tree
            Discovered object-module tree for object-module traces, if any.

        Returns
        -------
        None
            Trace accessors are populated.
        """

        for raw_index, (label, op_log) in enumerate(trace._raw_layer_dict.items()):
            pass_label = f"{label}:1"
            op_log._label_raw = label
            op_log._layer_label_raw = label
            op_log.label = pass_label
            op_log.label_short = pass_label
            op_log.layer_label = label
            op_log.layer_label_short = label
            op_log.lookup_keys = [label, pass_label]
            op_log.pass_index = 1
            op_log.num_passes = 1
            op_log.dtype_ref = DtypeRef(backend="tinygrad", name=str(op_log.dtype))
            op_log.device_ref = DeviceRef.from_value(getattr(op_log.out, "device", None))
            op_log.backend_address = f"uop:{label}"
            op_log.resolver_status = "resolved"
            trace.layer_list.append(op_log)
            trace.layer_dict_main_keys[label] = op_log
            trace.layer_dict_all_keys[label] = op_log
            trace.layer_dict_all_keys[pass_label] = op_log
            trace.op_labels.append(pass_label)
            trace.layer_labels.append(label)
            trace.layer_num_calls[label] = 1
            trace._lookup_keys_to_layer_num_dict[label] = raw_index
            trace._layer_num_to_lookup_keys_dict[raw_index].append(label)
            for param in getattr(op_log, "_param_logs", []):
                if op_log.label not in param.used_by_ops:
                    param.used_by_ops.append(op_log.label)
                if op_log.layer_label not in param.used_by_layers:
                    param.used_by_layers.append(op_log.layer_label)
                if op_log.layer_label not in trace.layers_with_params[param.barcode]:
                    trace.layers_with_params[param.barcode].append(op_log.layer_label)
            layer_log = Layer(op_log)
            layer_log.ops[1] = op_log
            layer_log.call_labels.append(pass_label)
            layer_log.dtype_ref = op_log.dtype_ref
            layer_log.device_ref = op_log.device_ref
            layer_log.backend_address = op_log.backend_address
            layer_log.resolver_status = op_log.resolver_status
            trace.layer_logs[label] = layer_log
        trace.num_ops = sum(
            1
            for op_log in trace.layer_list
            if not (op_log.is_input or op_log.is_output or op_log.is_buffer)
        )
        seen_layers: set[str] = set()
        num_param_tensors = 0
        num_params = 0
        num_params_trainable = 0
        for op_log in trace.layer_list:
            if op_log.layer_label in seen_layers:
                continue
            seen_layers.add(op_log.layer_label)
            num_param_tensors += op_log.num_param_tensors
            num_params += op_log.num_params
            num_params_trainable += op_log.num_params_trainable
        if trace.param_source != "none":
            trace.num_param_tensors = num_param_tensors
            trace.num_params = num_params
            trace.num_params_trainable = num_params_trainable
            trace.num_params_frozen = num_params - num_params_trainable
            trace.num_layers_with_params = len(
                {op.layer_label for op in trace.layer_list if op.uses_params}
            )
        trace._layers_logged = True
        trace._layers_saved = True
        trace.has_backward_pass = False
        trace.capture_end_time = time.time()
        trace.backend = cast(BackendName, self.name)
        if module_tree is None:
            trace.module_identity_mode = "function_root"
            self._attach_function_root_module(trace)
        else:
            trace.module_identity_mode = "object_module"
            self._attach_object_module_logs(trace, module_tree)
        trace._tracing_finished = True

    def _attach_object_module_logs(self, trace: Trace, tree: TinygradModuleTree) -> None:
        """Build public module logs for a tinygrad object-module trace.

        Parameters
        ----------
        trace
            Trace receiving module accessors.
        tree
            Discovered tinygrad object-module tree.

        Returns
        -------
        None
            ``trace.modules`` is populated by the shared module-log builder.
        """

        trace._module_build_data = _init_module_hierarchy_data()
        trace._module_forward_args = dict(tree.forward_args_by_call)
        trace._module_metadata = tree.metadata
        mbd = trace._module_build_data
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

        for param in trace.param_logs:
            owner = param.module_address
            mbd["module_nparams"][owner] += param.num_params
            if param.is_trainable:
                mbd["module_nparams_trainable"][owner] += param.num_params
            else:
                mbd["module_nparams_frozen"][owner] += param.num_params

        self._populate_object_module_build_data(trace)
        _build_module_logs(trace)

    def _populate_object_module_build_data(self, trace: Trace) -> None:
        """Populate module hierarchy side channels from attributed tinygrad ops.

        Parameters
        ----------
        trace
            Trace whose finalized ops carry object-module tuples.

        Returns
        -------
        None
            ``trace._module_build_data`` is updated in place.
        """

        mbd = trace._module_build_data
        seen_layers: dict[str, set[str]] = defaultdict(set)
        seen_pass_layers: dict[str, set[str]] = defaultdict(set)
        seen_module_ops: set[str] = set()
        seen_top_level_ops: set[str] = set()
        seen_pass_children: dict[str, set[str]] = defaultdict(set)
        seen_addresses = set(mbd["addresses"])

        for op_log in trace.layer_list:
            normalized_calls = _tinygrad_op_module_calls(op_log.modules)
            op_log.modules = [f"{address}:{call_index}" for address, call_index in normalized_calls]
            op_log.module = op_log.modules[-1] if op_log.modules else None
            parent_call_label: str | None = None
            for module_index, (address, call_index) in enumerate(normalized_calls):
                call_label = f"{address}:{call_index}"
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
                elif parent_call_label is not None:
                    if call_label not in seen_pass_children[parent_call_label]:
                        seen_pass_children[parent_call_label].add(call_label)
                        mbd["module_pass_children"][parent_call_label].append(call_label)
                parent_call_label = call_label

    def _attach_derived_grads(
        self,
        *,
        trace: Trace,
        model: Callable[..., Any],
        args: Sequence[Any],
        captured_output: Any,
        grad_options: GradOptions,
    ) -> None:
        """Compute and attach bracketed tinygrad leaf-level derived gradients.

        Parameters
        ----------
        trace
            Trace receiving derived gradient records.
        model
            Captured tinygrad callable.
        args
            Positional call arguments used for capture.
        captured_output
            Raw output from the captured forward call.
        grad_options
            Derived-gradient options.

        Returns
        -------
        None
            ``trace.derived_grads`` is populated with input leaf gradient records.
        """

        if getattr(trace, "tinygrad_payload_policy", None) != "dev_python_realized_copy":
            raise BackendUnsupportedError(
                "tinygrad derived gradients require live DEV=PYTHON realized-copy payloads; "
                "audit-only traces cannot expose trace.derived_grads."
            )
        input_argnums = _normalize_tinygrad_input_grad_argnums(
            grad_options.input_grad_argnums,
            len(args),
        )
        leaves = _differentiated_input_leaves(self, args, input_argnums)
        if not leaves and not grad_options.intermediate_grads:
            raise ValueError("tinygrad derived gradients require at least one tensor input leaf.")
        snapshots = _snapshot_tinygrad_grads(self, leaves)
        try:
            if grad_options.intermediate_grads:
                trace.intermediate_derived_grads = self._derive_intermediate_grads_no_realize(
                    trace=trace,
                    model=model,
                    args=args,
                    captured_output=captured_output,
                    grad_options=grad_options,
                    snapshots=snapshots,
                )
                _restore_tinygrad_grads(snapshots)
                snapshots = _snapshot_tinygrad_grads(self, leaves)
            derived_output = model(*args)
            loss = grad_options.loss_fn(derived_output) if grad_options.loss_fn else derived_output
            if not self.is_tensor(loss) or not _is_scalar_tinygrad_value(loss):
                raise ValueError(
                    "tinygrad derived gradients require loss_fn(raw_output) to be scalar unless "
                    "the traced output is already scalar."
                )
            if not _tinygrad_outputs_close(self, derived_output, captured_output):
                raise ValueError(
                    "tinygrad derived gradient run raw output diverged from captured raw output; "
                    "refusing to expose trace.derived_grads."
                )
            cast(Any, loss).backward()
            records = _records_for_tinygrad_leaf_grads(
                backend=self,
                leaves=leaves,
                snapshots=snapshots,
                provenance={
                    "backend": "tinygrad",
                    "kind": "derived_gradient",
                    "bracketing": "snapshot_backward_increment_restore",
                    "loss_fn": _callable_identity(grad_options.loss_fn),
                },
            )
        finally:
            _restore_tinygrad_grads(snapshots)
        trace.derived_grads = DerivedGradAccessor(records)
        self._mirror_param_derived_grads(trace, records)

    def _derive_intermediate_grads_no_realize(
        self,
        *,
        trace: Trace,
        model: Callable[..., Any],
        args: Sequence[Any],
        captured_output: Any,
        grad_options: GradOptions,
        snapshots: Mapping[str, tuple[Any, Any | None, Any | None]],
    ) -> IntermediateDerivedGradAccessor:
        """Compute tinygrad op-level grads using a no-realize backward pass.

        Parameters
        ----------
        trace
            Trace whose saved ops define the bounded attachment set.
        model
            Captured tinygrad callable.
        args
            Positional call arguments used for capture.
        captured_output
            Raw output from the captured forward call.
        grad_options
            Derived-gradient options.
        snapshots
            User input grad snapshots restored by the caller.

        Returns
        -------
        IntermediateDerivedGradAccessor
            Exact, unambiguous op-level intermediate gradient records.
        """

        observed_ops: dict[int, list[str]] = {}
        observed_tensors: dict[int, list[Any]] = {}
        with _observe_tensor_ops(observed_ops, observed_tensors=observed_tensors):
            derived_output = model(*args)
        outputs = tuple(self._tensor_leaves(derived_output))
        loss = grad_options.loss_fn(derived_output) if grad_options.loss_fn else derived_output
        if not self.is_tensor(loss) or not _is_scalar_tinygrad_value(loss):
            raise ValueError(
                "tinygrad intermediate derived gradients require loss_fn(raw_output) to be "
                "scalar unless the traced output is already scalar."
            )
        selected_ops = tuple(
            op
            for op in trace.layer_list
            if op.has_saved_activation
            and not op.is_input
            and op.annotations.get("tinygrad_observed_tensor_ops")
        )
        trace_signatures = _tinygrad_trace_op_signatures(
            selected_ops,
            getattr(trace, "tinygrad_uop_captures", ()),
        )
        live_candidates = _tinygrad_live_intermediate_candidates(
            backend=self,
            outputs=outputs,
            observed_tensors=observed_tensors,
        )
        cast(Any, loss).backward()
        if not _tinygrad_outputs_close(self, derived_output, captured_output):
            raise ValueError(
                "tinygrad intermediate derived gradient run raw output diverged from captured "
                "raw output; refusing to expose trace.intermediate_derived_grads."
            )
        records = _records_for_tinygrad_intermediate_grads(
            backend=self,
            trace_signatures=trace_signatures,
            live_candidates=live_candidates,
            snapshots=snapshots,
            provenance={
                "backend": "tinygrad",
                "kind": "intermediate_derived_gradient",
                "mechanism": "tinygrad_retained_tensor_backward_no_realize",
                "loss_id": _callable_identity(grad_options.loss_fn),
                "save_predicate_id": "trace.saved_ops",
                "status": "exact",
            },
        )
        return IntermediateDerivedGradAccessor(records)

    def _mirror_param_derived_grads(
        self, trace: Trace, records: Mapping[str, DerivedGradRecord]
    ) -> None:
        """Mirror unambiguous tinygrad param derived gradients onto param records.

        Parameters
        ----------
        trace
            Trace containing optional parameter metadata.
        records
            Derived gradient records keyed by leaf path.

        Returns
        -------
        None
            Matching ``trace.params`` entries receive the same gradient payload.
        """

        for address, param in trace.params.items():
            record = records.get(f"params.{address}")
            if record is None:
                continue
            param._derived_grad_payload = record.grad
            param._derived_grad_record_path = record.path
            param.has_grad = True
            param.grad_shape = tuple(getattr(record.grad, "shape", ()))
            param.grad_dtype = cast(Any, str(getattr(record.grad, "dtype", "")))
            param.gradient_memory = _nbytes(record.grad) or 0

    def _attach_function_root_module(self, trace: Trace) -> None:
        """Attach a function-root module accessor to ``trace``.

        Parameters
        ----------
        trace
            Trace receiving the root module.

        Returns
        -------
        None
            ``trace.modules`` is populated with ``self``.
        """

        mbd = trace._module_build_data
        mbd["top_level_modules"] = ["self"]
        mbd["top_level_module_ops"] = ["self:1"]
        trace._module_metadata = {
            "self": {
                "cls": None,
                "class_name": trace.model_class_name,
                "class_qualname": trace.model_class_qualname,
                "all_addresses": ["self"],
                "training": False,
            }
        }
        root = _build_root_module_log(trace, {}, mbd)
        trace._module_logs = ModuleAccessor({"self": root})

    def _validate_uops(self, trace: Trace) -> bool:
        """Validate saved tinygrad payloads against replayed UOps.

        Parameters
        ----------
        trace
            Trace produced by this backend.

        Returns
        -------
        bool
            True when every captured UOp replays to its saved payload.
        """

        captures_raw = getattr(trace, "tinygrad_uop_captures", None)
        captures = tuple(captures_raw) if captures_raw is not None else ()
        if not captures:
            raise BackendUnsupportedError(
                "tinygrad validation requires live DEV=PYTHON realized-copy payloads; "
                "audit-only traces cannot be replay validated."
            )
        ops_by_raw_label = {
            getattr(op, "_label_raw", ""): op for op in getattr(trace, "layer_list", ())
        }
        hidden_outputs_by_label = dict(getattr(trace, "_selective_save_hidden_payloads", {}))
        for capture in captures:
            op = ops_by_raw_label.get(capture.label_raw)
            if op is None:
                return False
            saved_output = _saved_single_output(op, hidden_outputs_by_label)
            replayed = self._replay_uop_from_trace_graph(
                capture,
                op,
                ops_by_raw_label,
                hidden_outputs_by_label=hidden_outputs_by_label,
            )
            if not _payloads_close(replayed, saved_output):
                return False
            if not _parent_perturbations_change_output(
                backend=self,
                capture=capture,
                op=op,
                ops_by_raw_label=ops_by_raw_label,
                hidden_outputs_by_label=hidden_outputs_by_label,
                saved_output=saved_output,
            ):
                return False
        return True

    def _replay_uop_from_trace_graph(
        self,
        capture: TinygradUOpCapture,
        op: Any,
        ops_by_raw_label: Mapping[str, Any],
        replacements: Mapping[int, Any] | None = None,
        hidden_outputs_by_label: Mapping[str, Any] | None = None,
    ) -> Any:
        """Replay one captured UOp with inputs from materialized trace parents.

        Parameters
        ----------
        capture
            Captured UOp metadata.
        op
            Materialized TorchLens op corresponding to ``capture``.
        ops_by_raw_label
            Materialized operations keyed by raw label.
        replacements
            Optional UOp source-position replacements used by perturbation.
        hidden_outputs_by_label
            Runtime-only replay payloads keyed by raw op label.

        Returns
        -------
        Any
            Realized tinygrad tensor replay output.
        """

        src = list(getattr(capture.uop, "src", ()) or ())
        graph_positions = getattr(op, "parent_arg_positions", {}).get("args", {})
        parent_labels = tuple(getattr(op, "parents", ()))
        if not graph_positions and not parent_labels:
            return capture.payload_snapshot
        positioned_labels = {label for label in graph_positions.values() if isinstance(label, str)}
        if positioned_labels != set(parent_labels):
            raise ValueError("tinygrad trace parent labels and parent_arg_positions disagree.")
        if tuple(sorted(graph_positions.items())) != tuple(sorted(capture.parent_arg_positions)):
            raise ValueError("tinygrad trace parent_arg_positions changed after capture.")
        for position, parent_label in graph_positions.items():
            if not isinstance(position, int) or position < 0 or position >= len(src):
                raise ValueError(f"tinygrad trace parent arg position {position!r} is invalid.")
            parent_op = ops_by_raw_label[parent_label]
            parent_value = _saved_single_output(parent_op, hidden_outputs_by_label)
            if _source_matches_payload(src[position], parent_value):
                src[position] = parent_value.uop
        for position, replacement in (replacements or {}).items():
            if position < 0 or position >= len(src):
                raise ValueError(f"tinygrad perturbation position {position!r} is invalid.")
            src[position] = replacement.uop
        replay_uop = capture.uop.replace(src=tuple(src))
        return self._realized_copy(self._tensor_from_uop(replay_uop))

    def _input_identities(self, args: Sequence[Any]) -> tuple[str, ...]:
        """Return tinygrad identities for positional tensor inputs.

        Parameters
        ----------
        args
            Normalized positional arguments.

        Returns
        -------
        tuple[str, ...]
            Versioned tinygrad identities for tensor leaves.
        """

        return tuple(
            _identity(leaf)
            for _path, leaf in _tree_leaves_with_paths(tuple(args))
            if self.is_tensor(leaf)
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

    def _tensor_leaves(self, value: object) -> list[Any]:
        """Return tinygrad tensor leaves from a simple Python container.

        Parameters
        ----------
        value
            Candidate tree.

        Returns
        -------
        list[Any]
            Flat tinygrad tensor leaves.
        """

        return [leaf for _path, leaf in _tree_leaves_with_paths(value) if self.is_tensor(leaf)]

    def _tensor_from_uop(self, uop: Any) -> Any:
        """Create a tinygrad tensor from a UOp snapshot.

        Parameters
        ----------
        uop
            tinygrad UOp.

        Returns
        -------
        Any
            tinygrad Tensor wrapping ``uop``.
        """

        from tinygrad import Tensor

        return Tensor(uop)

    def _realized_copy(self, value: Any) -> Any:
        """Return a sanctioned live payload copy for ``DEV=PYTHON`` tinygrad.

        Parameters
        ----------
        value
            tinygrad Tensor to copy.

        Returns
        -------
        Any
            Realized tinygrad Tensor copy detached from the source UOp lineage.
        """

        from tinygrad import Tensor

        return Tensor(value.tolist(), dtype=value.dtype, device=value.device).realize()

    def _assert_runtime_supported(self) -> None:
        """Reject tinygrad runtimes outside the S0.G-proven live-payload envelope.

        Returns
        -------
        None
            Returns when tinygrad 0.13.0 is importable.
        """

        import tinygrad

        version = getattr(tinygrad, "__version__", None)
        if version not in {"0.13.0", None}:
            raise BackendUnsupportedError(
                f"tinygrad backend preview is pinned to tinygrad==0.13.0; found {version!r}."
            )

    def _assert_tinygrad_inputs(self, args: Sequence[Any]) -> None:
        """Reject calls without tinygrad tensor inputs.

        Parameters
        ----------
        args
            Normalized positional call arguments.

        Returns
        -------
        None
            Returns when at least one tinygrad tensor leaf is present.
        """

        if not any(self.is_tensor(leaf) for _path, leaf in _tree_leaves_with_paths(tuple(args))):
            raise BackendUnsupportedError(
                "tinygrad backend preview requires positional tinygrad Tensor inputs."
            )

    def _reject_unsupported_options(self, **options: Any) -> None:
        """Reject public trace options outside the tinygrad preview surface.

        Parameters
        ----------
        **options
            Normalized public trace options.

        Returns
        -------
        None
            Returns when all options are supported.
        """

        if options["input_kwargs"]:
            raise BackendUnsupportedError("tinygrad backend preview supports positional args only.")
        if options["layers_to_save"] not in ("all", None):
            raise BackendUnsupportedError(
                "tinygrad backend preview is full-save only; save shaping is unsupported."
            )
        rejected_true = (
            "activation_transform",
            "detach_saved_activations",
            "save_grads",
            "save_arg_values",
            "save_code_context",
            "save_rng_states",
            "backward_ready",
            "module_filter",
            "transform",
            "layer_visualizers",
            "save_visualizations",
        )
        for name in rejected_true:
            if options[name]:
                raise BackendUnsupportedError(
                    f"tinygrad backend preview does not support {name}; "
                    "full-save forward capture only."
                )
        if options["output_device"] != "same":
            raise BackendUnsupportedError(
                "tinygrad backend preview only supports output_device='same'."
            )
        if not options["save_raw_activations"]:
            raise BackendUnsupportedError(
                "tinygrad backend preview is full-save only; "
                "save_raw_activations=False is unsupported."
            )
        if options["lookback"] != 0 or options["lookback_payload_policy"] != "metadata_only":
            raise BackendUnsupportedError(
                "tinygrad backend preview is full-save only; save-window shaping is unsupported."
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

        rejected = {
            key: value
            for key, value in kwargs.items()
            if value is not None and not _is_missing(value)
        }
        if rejected:
            names = ", ".join(sorted(rejected))
            raise BackendUnsupportedError(
                "tinygrad backend preview does not support runtime-mutation or stop-early "
                f"options: {names}. Static-label save= selectors are supported as "
                "post-finalization payload filters, but trace(intervene=...) and "
                "trace(halt=...) need predicate-time concrete values and a way to replace or "
                "truncate lazy UOp descendants before realize(), which tinygrad does not expose "
                "through a stable TorchLens surface. Use an unfiltered tl.trace(..., "
                "backend='tinygrad') call, static-label save= selectors, or the PyTorch backend "
                "for intervention, halt, streaming, and value-dependent predicates."
            )


def discover_tinygrad_module_tree(model: Any) -> TinygradModuleTree | None:
    """Discover a tinygrad callable object-module hierarchy.

    Parameters
    ----------
    model
        Candidate tinygrad callable root.

    Returns
    -------
    TinygradModuleTree | None
        Discovered module tree, or ``None`` for raw functions/plain callables.
    """

    if inspect.isfunction(model) or inspect.ismethod(model) or not callable(model):
        return None
    if not _is_tinygrad_module_like(model):
        return None

    metadata: dict[str, dict[str, Any]] = {}
    address_by_id: dict[int, str] = {}
    modules_by_class: defaultdict[type[Any], dict[int, str]] = defaultdict(dict)
    param_owner_by_address: dict[str, str] = {}
    param_address_by_uop_id: dict[int, str] = {}
    _walk_tinygrad_modules(
        module=model,
        address="self",
        metadata=metadata,
        address_by_id=address_by_id,
        modules_by_class=modules_by_class,
        param_owner_by_address=param_owner_by_address,
        param_address_by_uop_id=param_address_by_uop_id,
    )
    if len(metadata) <= 1 and not _iter_tinygrad_tensor_attrs(model, "self"):
        return None
    return TinygradModuleTree(
        root=model,
        metadata=metadata,
        address_by_id=address_by_id,
        modules_by_class=dict(modules_by_class),
        param_owner_by_address=param_owner_by_address,
        param_address_by_uop_id=param_address_by_uop_id,
        call_counts={},
        forward_args_by_call={},
    )


@contextmanager
def scoped_tinygrad_module_calls(
    tree: TinygradModuleTree,
    observed_module_stacks: dict[int, tuple[TinygradModuleFrame, ...]],
) -> Iterator[None]:
    """Temporarily wrap discovered tinygrad module ``__call__`` methods.

    Parameters
    ----------
    tree
        Discovered tinygrad object-module tree.
    observed_module_stacks
        Mutable UOp-id mapping receiving first-observed module stacks.

    Yields
    ------
    None
        Control while class-level wrappers are installed.
    """

    del observed_module_stacks
    originals: dict[type[Any], Any] = {}
    for module_class, address_by_instance_id in tree.modules_by_class.items():
        original_call = getattr(module_class, "__call__")
        originals[module_class] = original_call

        def wrapper(
            self: Any,
            *args: Any,
            __address_by_id: dict[int, str] = address_by_instance_id,
            __original: Any = original_call,
            **kwargs: Any,
        ) -> Any:
            """Call the original module while the live stack records this module."""

            address = __address_by_id.get(id(self))
            if address is None:
                return __original(self, *args, **kwargs)
            call_index = tree.call_counts.get(address, 0) + 1
            tree.call_counts[address] = call_index
            tree.forward_args_by_call[(address, call_index)] = (args, kwargs)
            _ACTIVE_TINYGRAD_MODULE_STACK.append(
                TinygradModuleFrame(
                    address=address,
                    call_index=call_index,
                    module_type=type(self).__name__,
                )
            )
            try:
                return __original(self, *args, **kwargs)
            finally:
                _ACTIVE_TINYGRAD_MODULE_STACK.pop()

        setattr(module_class, "__call__", wrapper)
    try:
        yield
    finally:
        for module_class, original_call in originals.items():
            setattr(module_class, "__call__", original_call)


@contextmanager
def _null_context() -> Iterator[None]:
    """Yield a no-op context manager.

    Yields
    ------
    None
        Control without side effects.
    """

    yield


def tinygrad_param_logs(tree: TinygradModuleTree, trace: Trace) -> dict[str, Param]:
    """Build TorchLens parameter logs from tinygrad module tensor attributes.

    Parameters
    ----------
    tree
        Discovered tinygrad object-module tree.
    trace
        Trace receiving the parameter logs.

    Returns
    -------
    dict[str, Param]
        Parameter logs keyed by object address.
    """

    param_logs: dict[str, Param] = {}
    tensor_by_uop_id: dict[int, Any] = {}
    for address, metadata in tree.metadata.items():
        module = metadata.get("_module_object")
        if module is None:
            continue
        for param_address, tensor in _iter_tinygrad_tensor_attrs(module, address):
            existing_address = tree.param_address_by_uop_id.get(id(tensor.uop))
            tensor_by_uop_id.setdefault(id(tensor.uop), tensor)
            if existing_address is not None and existing_address in param_logs:
                param = param_logs[existing_address]
                if param_address not in param.all_addresses:
                    param.all_addresses.append(param_address)
                if param_address not in param.co_parent_params:
                    param.co_parent_params.append(param_address)
                owner = tree.param_owner_by_address.get(param_address, address)
                for alias in tree.metadata.get(owner, {}).get("all_addresses", [owner]):
                    if alias not in param.all_module_addresses:
                        param.all_module_addresses.append(alias)
                continue
            owner = tree.param_owner_by_address.get(param_address, address)
            shape = tuple(getattr(tensor, "shape", ()))
            dtype = str(getattr(tensor, "dtype", ""))
            param = Param(
                module_address=owner,
                name=param_address.rsplit(".", 1)[-1],
                shape=shape,
                dtype=cast(Any, dtype),
                num_params=_numel(shape),
                param_memory=_nbytes(tensor) or 0,
                trainable=bool(getattr(tensor, "requires_grad", False)),
                address=param_address,
                barcode=f"tinygrad:{param_address}",
                has_optimizer=None,
            )
            param.dtype_ref = DtypeRef(backend="tinygrad", name=dtype)
            param.device_ref = DeviceRef.from_value(getattr(tensor, "device", None))
            param.backend_address = f"object:{param_address}"
            param.resolver_status = "resolved"
            param._param_ref = cast(Any, tensor)
            param.source_trace = trace
            param.all_module_addresses = list(
                tree.metadata.get(owner, {}).get("all_addresses", [owner])
            )
            param_logs[param_address] = param
    return param_logs


def _walk_tinygrad_modules(
    *,
    module: Any,
    address: str,
    metadata: dict[str, dict[str, Any]],
    address_by_id: dict[int, str],
    modules_by_class: defaultdict[type[Any], dict[int, str]],
    param_owner_by_address: dict[str, str],
    param_address_by_uop_id: dict[int, str],
) -> None:
    """Walk callable object attributes and populate tinygrad module metadata.

    Parameters
    ----------
    module
        Module-like object instance.
    address
        TorchLens address for ``module``.
    metadata
        Metadata mapping being populated.
    address_by_id
        Primary address mapping being populated.
    modules_by_class
        Class-level wrapper mapping being populated.
    param_owner_by_address
        Parameter owner mapping being populated.
    param_address_by_uop_id
        Parameter UOp identity mapping being populated.

    Returns
    -------
    None
        Mappings are updated in place.
    """

    module_id = id(module)
    primary = address_by_id.get(module_id)
    if primary is not None:
        metadata[primary].setdefault("all_addresses", [primary]).append(address)
        return

    address_by_id[module_id] = address
    modules_by_class[type(module)][module_id] = address
    children = _iter_tinygrad_module_children(module, address)
    metadata[address] = {
        **_module_source_metadata(module),
        "cls": type(module),
        "class_name": type(module).__name__,
        "class_qualname": f"{type(module).__module__}.{type(module).__qualname__}",
        "address_children": [child_address for child_address, _child in children],
        "all_addresses": [address],
        "training": False,
        "forward_pre_hooks": [],
        "forward_hooks": [],
        "backward_pre_hooks": [],
        "backward_hooks": [],
        "full_backward_pre_hooks": [],
        "full_backward_hooks": [],
        "custom_attributes": {},
        "custom_methods": [],
        "_module_object": module,
    }
    for param_address, tensor in _iter_tinygrad_tensor_attrs(module, address):
        param_owner_by_address[param_address] = address
        param_address_by_uop_id.setdefault(id(tensor.uop), param_address)
    for child_address, child_module in children:
        _walk_tinygrad_modules(
            module=child_module,
            address=child_address,
            metadata=metadata,
            address_by_id=address_by_id,
            modules_by_class=modules_by_class,
            param_owner_by_address=param_owner_by_address,
            param_address_by_uop_id=param_address_by_uop_id,
        )


def _iter_tinygrad_module_children(module: Any, address: str) -> list[tuple[str, Any]]:
    """Return direct callable tinygrad module-like children.

    Parameters
    ----------
    module
        Parent module-like object.
    address
        Parent TorchLens address.

    Returns
    -------
    list[tuple[str, Any]]
        Child address and child object pairs.
    """

    children: list[tuple[str, Any]] = []
    for name, value in getattr(module, "__dict__", {}).items():
        if name.startswith("_") or not callable(value):
            continue
        if _is_tinygrad_module_like(value):
            children.append((_join_module_address(address, name), value))
    return children


def _iter_tinygrad_tensor_attrs(module: Any, address: str) -> list[tuple[str, Any]]:
    """Return direct tinygrad tensor attributes for one object.

    Parameters
    ----------
    module
        Candidate module object.
    address
        TorchLens module address.

    Returns
    -------
    list[tuple[str, Any]]
        Parameter address and tensor pairs.
    """

    return [
        (_join_module_address(address, name), value)
        for name, value in getattr(module, "__dict__", {}).items()
        if not name.startswith("_") and _is_tinygrad_tensor(value)
    ]


def _is_tinygrad_module_like(value: Any) -> bool:
    """Return whether ``value`` is a tinygrad module-like callable object.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True when the object matches the tinygrad module discovery heuristic.
    """

    if inspect.isfunction(value) or inspect.ismethod(value) or not callable(value):
        return False
    if _is_known_tinygrad_nn_type(value):
        return True
    if any(_is_tinygrad_tensor(child) for child in getattr(value, "__dict__", {}).values()):
        return True
    return any(
        callable(child) and _is_tinygrad_module_like(child)
        for child in getattr(value, "__dict__", {}).values()
    )


def _is_known_tinygrad_nn_type(value: Any) -> bool:
    """Return whether ``value`` is an instance of a known ``tinygrad.nn`` class.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True for known tinygrad neural-network helper classes.
    """

    try:
        import tinygrad.nn as tinygrad_nn
    except ImportError:
        return False
    known_types = tuple(
        attr for name in dir(tinygrad_nn) if isinstance((attr := getattr(tinygrad_nn, name)), type)
    )
    return isinstance(value, known_types)


def _is_tinygrad_tensor(value: Any) -> bool:
    """Return whether ``value`` is a tinygrad Tensor.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True when ``value`` is a tinygrad ``Tensor``.
    """

    try:
        from tinygrad import Tensor
    except ImportError:
        return False
    return isinstance(value, Tensor)


def _module_source_metadata(module: Any) -> dict[str, Any]:
    """Return best-effort source metadata for a tinygrad module-like object.

    Parameters
    ----------
    module
        Module-like object.

    Returns
    -------
    dict[str, Any]
        Source metadata compatible with TorchLens module logs.
    """

    cls = type(module)
    init = getattr(cls, "__init__", None)
    call = getattr(cls, "__call__", None)
    return {
        "class_source_file": inspect.getsourcefile(cls),
        "class_source_line": _source_line(cls),
        "init_source_file": inspect.getsourcefile(init) if init is not None else None,
        "init_source_line": _source_line(init),
        "forward_source_file": inspect.getsourcefile(call) if call is not None else None,
        "forward_source_line": _source_line(call),
        "class_docstring": inspect.getdoc(cls),
        "init_signature": _signature_string(init),
        "init_docstring": inspect.getdoc(init) if init is not None else None,
        "forward_signature": _signature_string(call),
        "forward_docstring": inspect.getdoc(call) if call is not None else None,
    }


def _source_line(obj: Any) -> int | None:
    """Return the first source line for ``obj`` when inspectable.

    Parameters
    ----------
    obj
        Object to inspect.

    Returns
    -------
    int | None
        First source line, or ``None``.
    """

    if obj is None:
        return None
    try:
        return inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        return None


def _signature_string(obj: Any) -> str | None:
    """Return ``obj``'s signature string when inspectable.

    Parameters
    ----------
    obj
        Object to inspect.

    Returns
    -------
    str | None
        Signature string, or ``None``.
    """

    if obj is None:
        return None
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _join_module_address(parent: str, child_name: str) -> str:
    """Return a TorchLens child module address.

    Parameters
    ----------
    parent
        Parent module address.
    child_name
        Child attribute name.

    Returns
    -------
    str
        Joined module address.
    """

    return child_name if parent == "self" else f"{parent}.{child_name}"


def _module_stack_for_uop(
    uop: Any,
    observed_module_stacks: Mapping[int, tuple[TinygradModuleFrame, ...]],
    module_tree: TinygradModuleTree | None,
) -> tuple[TinygradModuleFrame, ...]:
    """Return the best object-module stack for a tinygrad UOp.

    Parameters
    ----------
    uop
        tinygrad UOp.
    observed_module_stacks
        First-observed live module stacks keyed by UOp id.
    module_tree
        Discovered module tree, if object-module mode is active.

    Returns
    -------
    tuple[TinygradModuleFrame, ...]
        Module stack for the UOp, or empty in function-root mode.
    """

    observed = observed_module_stacks.get(id(uop))
    if observed:
        return observed
    if module_tree is None:
        return ()
    param_address = module_tree.param_address_by_uop_id.get(id(uop))
    if param_address is None:
        return (TinygradModuleFrame("self", 1, type(module_tree.root).__name__),)
    owner = module_tree.param_owner_by_address.get(param_address, "self")
    return _synthetic_stack_for_address(owner, module_tree)


def _param_refs_for_uop(
    uop: Any,
    module_tree: TinygradModuleTree | None,
) -> tuple[ParamRef, ...]:
    """Return parameter refs whose UOps contribute to ``uop``.

    Parameters
    ----------
    uop
        tinygrad UOp.
    module_tree
        Discovered tinygrad module tree, if object-module mode is active.

    Returns
    -------
    tuple[ParamRef, ...]
        Unique parameter references in first-seen topological order.
    """

    if module_tree is None:
        return ()
    refs: list[ParamRef] = []
    seen: set[str] = set()
    for candidate in uop.toposort():
        param_address = module_tree.param_address_by_uop_id.get(id(candidate))
        if param_address is None or param_address in seen:
            continue
        seen.add(param_address)
        owner = module_tree.param_owner_by_address.get(param_address, "self")
        try:
            from tinygrad import Tensor

            tensor = Tensor(candidate)
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)
            trainable = bool(getattr(tensor, "requires_grad", False))
        except Exception:
            shape = None
            dtype = None
            trainable = False
        refs.append(
            ParamRef(
                barcode=f"tinygrad:{param_address}",
                address=param_address,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                module_address=owner,
            )
        )
    return tuple(refs)


def _fallback_output_label(
    output: Any,
    captures: Sequence[TinygradUOpCapture],
    used_labels: Sequence[str],
) -> str | None:
    """Return a best-effort captured label for a returned tinygrad output.

    Parameters
    ----------
    output
        Returned tinygrad output tensor.
    captures
        Captures emitted from output-reachable UOps.
    used_labels
        Output labels already assigned to earlier leaves.

    Returns
    -------
    str | None
        Matching raw label, if one can be inferred.
    """

    output_name = _uop_name(output.uop)
    output_shape = tuple(getattr(output, "shape", ()))
    output_dtype = str(getattr(output, "dtype", ""))
    used = set(used_labels)
    for capture in reversed(captures):
        if capture.label_raw in used or capture.op_name != output_name:
            continue
        payload = capture.payload_snapshot
        if tuple(getattr(payload, "shape", ())) != output_shape:
            continue
        if str(getattr(payload, "dtype", "")) != output_dtype:
            continue
        return capture.label_raw
    return None


def _synthetic_stack_for_address(
    address: str, module_tree: TinygradModuleTree
) -> tuple[TinygradModuleFrame, ...]:
    """Build a first-call stack for a discovered module address.

    Parameters
    ----------
    address
        Module address needing a synthetic stack.
    module_tree
        Discovered tinygrad object-module tree.

    Returns
    -------
    tuple[TinygradModuleFrame, ...]
        Stack from ``self`` to ``address``.
    """

    parts = [] if address == "self" else address.split(".")
    addresses = ["self", *[".".join(parts[: index + 1]) for index in range(len(parts))]]
    frames: list[TinygradModuleFrame] = []
    for current in addresses:
        metadata = module_tree.metadata.get(current, {})
        frames.append(
            TinygradModuleFrame(
                address=current,
                call_index=1,
                module_type=str(metadata.get("class_name", "")),
            )
        )
    return tuple(frames)


def _tinygrad_op_module_calls(value: Sequence[Any]) -> tuple[tuple[str, int], ...]:
    """Normalize an op's raw module tuple list.

    Parameters
    ----------
    value
        Materialized op ``modules`` field.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Normalized ``(address, call_index)`` pairs.
    """

    calls: list[tuple[str, int]] = []
    for item in value:
        if isinstance(item, tuple) and len(item) == 2:
            address, call_index = item
            calls.append((str(address), int(call_index)))
            continue
        text = str(item)
        address, separator, index_text = text.rpartition(":")
        if separator and index_text.isdigit():
            calls.append((address, int(index_text)))
    return tuple(calls)


def _resolve_tinygrad_module_identity_mode(
    value: str | None,
    module_tree: TinygradModuleTree | None,
) -> bool:
    """Return whether tinygrad should use object-module attribution.

    Parameters
    ----------
    value
        Public ``module_identity_mode`` value after missing normalization.
    module_tree
        Discovered tinygrad module tree, if any.

    Returns
    -------
    bool
        True when object-module mode should be used.
    """

    if value not in {None, "function_root", "object_module"}:
        raise BackendUnsupportedError(
            "tinygrad module_identity_mode must be None, 'function_root', or 'object_module'."
        )
    if value == "object_module" and module_tree is None:
        raise BackendUnsupportedError(
            "tinygrad module_identity_mode='object_module' requires a callable object with "
            "discoverable tinygrad module attributes. Raw callables use "
            "module_identity_mode='function_root'."
        )
    if value == "function_root":
        return False
    return module_tree is not None


def _numel(shape: tuple[int, ...]) -> int:
    """Return number of elements for ``shape``.

    Parameters
    ----------
    shape
        Tensor shape.

    Returns
    -------
    int
        Product of dimensions.
    """

    result = 1
    for dim in shape:
        result *= int(dim)
    return result


class _observe_tensor_ops:
    """Context manager observing tinygrad Tensor API UOp results."""

    def __init__(
        self,
        observed_ops: dict[int, list[str]],
        observed_module_stacks: dict[int, tuple[TinygradModuleFrame, ...]] | None = None,
        observed_tensors: dict[int, list[Any]] | None = None,
    ) -> None:
        """Initialize the observation context.

        Parameters
        ----------
        observed_ops
            Mutable mapping receiving Tensor API names by returned UOp id.
        observed_module_stacks
            Optional mutable mapping receiving first-observed module stacks by
            returned UOp id.
        observed_tensors
            Optional mutable mapping receiving live result tensors by returned
            UOp id.
        """

        self.observed_ops = observed_ops
        self.observed_module_stacks = observed_module_stacks
        self.observed_tensors = observed_tensors
        self.original: Any = None

    def __enter__(self) -> "_observe_tensor_ops":
        """Install the Tensor._apply_uop observer.

        Returns
        -------
        _observe_tensor_ops
            This context manager.
        """

        from tinygrad import Tensor

        self.original = Tensor._apply_uop

        def wrapped(tensor: Any, fxn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            """Observe one Tensor API operation returning a UOp-backed Tensor."""

            result = self.original(tensor, fxn, *args, **kwargs)
            name = getattr(fxn, "__name__", _uop_name(result.uop).lower())
            self.observed_ops.setdefault(id(result.uop), []).append(str(name))
            if self.observed_module_stacks is not None:
                self.observed_module_stacks.setdefault(
                    id(result.uop), tuple(_ACTIVE_TINYGRAD_MODULE_STACK)
                )
            if self.observed_tensors is not None:
                self.observed_tensors.setdefault(id(result.uop), []).append(result)
            return result

        Tensor._apply_uop = wrapped
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Restore the original Tensor._apply_uop implementation.

        Parameters
        ----------
        exc_type
            Exception type, if any.
        exc
            Exception value, if any.
        tb
            Traceback, if any.

        Returns
        -------
        None
            The monkeypatch is removed.
        """

        from tinygrad import Tensor

        Tensor._apply_uop = self.original


class _reject_mid_capture_execution:
    """Context manager rejecting tinygrad execution that truncates lazy UOp lineage."""

    def __init__(self) -> None:
        """Initialize the execution guard."""

        self.original_tensor_run_linear: Any = None
        self.original_jit_run_linear: Any = None

    def __enter__(self) -> "_reject_mid_capture_execution":
        """Install guarded tinygrad realization hooks.

        Returns
        -------
        _reject_mid_capture_execution
            This context manager.
        """

        import tinygrad.engine.jit as jit_module
        import tinygrad.tensor as tensor_module

        self.original_tensor_run_linear = tensor_module.run_linear
        self.original_jit_run_linear = jit_module.run_linear

        def rejected_run_linear(*args: Any, **kwargs: Any) -> Any:
            """Reject tinygrad realization or JIT execution during capture."""

            del args, kwargs
            raise BackendUnsupportedError(
                "tinygrad backend preview cannot capture Tensor.realize(), Tensor.assign(), "
                "or TinyJit execution inside the traced callable yet; return a lazy tinygrad "
                "expression instead."
            )

        tensor_module.run_linear = rejected_run_linear
        jit_module.run_linear = rejected_run_linear
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Restore tinygrad realization hooks.

        Parameters
        ----------
        exc_type
            Exception type, if any.
        exc
            Exception value, if any.
        tb
            Traceback, if any.

        Returns
        -------
        None
            The monkeypatches are removed.
        """

        import tinygrad.engine.jit as jit_module
        import tinygrad.tensor as tensor_module

        tensor_module.run_linear = self.original_tensor_run_linear
        jit_module.run_linear = self.original_jit_run_linear


def _is_missing(value: object) -> bool:
    """Return whether ``value`` is the public missing sentinel.

    Parameters
    ----------
    value
        Candidate value.

    Returns
    -------
    bool
        True when ``value`` is ``MISSING``.
    """

    return value is MISSING


def _default_if_missing(value: Any, default: Any) -> Any:
    """Return ``default`` when ``value`` is the public missing sentinel.

    Parameters
    ----------
    value
        Candidate value.
    default
        Replacement for ``MISSING``.

    Returns
    -------
    Any
        ``default`` or ``value``.
    """

    return default if _is_missing(value) else value


def _tree_leaves_with_paths(value: object, prefix: str = "") -> list[tuple[str, Any]]:
    """Return leaves from a simple Python container with dotted paths.

    Parameters
    ----------
    value
        Container or leaf.
    prefix
        Current dotted path prefix.

    Returns
    -------
    list[tuple[str, Any]]
        Flat ``(path, leaf)`` pairs.
    """

    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _tree_leaves_with_paths(child, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, tuple | list):
        return [
            item
            for index, child in enumerate(value)
            for item in _tree_leaves_with_paths(
                child, f"{prefix}.{index}" if prefix else str(index)
            )
        ]
    return [(prefix, value)]


def _unique_uops(outputs: Sequence[Any]) -> tuple[Any, ...]:
    """Return UOps reachable from outputs in topological order.

    Parameters
    ----------
    outputs
        tinygrad output tensors.

    Returns
    -------
    tuple[Any, ...]
        Unique UOps in first-seen topological order.
    """

    seen: set[int] = set()
    ordered: list[Any] = []
    for output in outputs:
        for uop in cast(Any, output).uop.toposort():
            if id(uop) in seen:
                continue
            seen.add(id(uop))
            ordered.append(uop)
    return tuple(ordered)


def _is_materializable_uop(uop: Any) -> bool:
    """Return whether a UOp can be saved as a tinygrad Tensor payload.

    Parameters
    ----------
    uop
        Candidate tinygrad UOp.

    Returns
    -------
    bool
        True when tinygrad can expose shape and host payload for ``uop``.
    """

    try:
        from tinygrad import Tensor

        tensor = Tensor(uop)
        tuple(tensor.shape)
        if getattr(tensor.dtype.base, "fmt", None) is None:
            return False
        tensor.tolist()
    except Exception:
        return False
    return True


def _uop_name(uop: Any) -> str:
    """Return a stable tinygrad UOp name.

    Parameters
    ----------
    uop
        tinygrad UOp.

    Returns
    -------
    str
        Operation name without the ``Ops.`` prefix.
    """

    op = getattr(uop, "op", None)
    return str(getattr(op, "name", op)).removeprefix("Ops.")


def _uop_signature(uop: Any) -> str:
    """Return a structural UOp signature string.

    Parameters
    ----------
    uop
        tinygrad UOp.

    Returns
    -------
    str
        Recursive operation/dtype/arg signature.
    """

    src = getattr(uop, "src", ()) or ()
    children = ",".join(_uop_signature(child) for child in src)
    return f"{_uop_name(uop)}:{getattr(uop, 'dtype', None)}:{getattr(uop, 'arg', None)}[{children}]"


def _tinygrad_signature_key(
    *,
    uop_signature: str,
    ordinal: int,
    parent_signatures: tuple[str, ...],
    shape: tuple[int, ...],
    dtype: str,
) -> tuple[Any, ...]:
    """Return the conservative key used for tinygrad T1 matching.

    Parameters
    ----------
    uop_signature
        Recursive structural UOp signature.
    ordinal
        Topological ordinal among retained non-input op candidates.
    parent_signatures
        Direct parent structural signatures.
    shape
        Tensor shape.
    dtype
        Tensor dtype string.

    Returns
    -------
    tuple[Any, ...]
        Hashable conservative match key.
    """

    return (uop_signature, ordinal, parent_signatures, shape, dtype)


def _tinygrad_trace_op_signatures(
    ops: Sequence[Any],
    captures: Sequence[TinygradUOpCapture],
) -> dict[tuple[Any, ...], list[Any]]:
    """Group saved trace ops by conservative tinygrad T1 signature.

    Parameters
    ----------
    ops
        Saved non-input trace ops eligible for intermediate-derived gradients.
    captures
        Live UOp captures from the original forward pass.

    Returns
    -------
    dict[tuple[Any, ...], list[Any]]
        Trace ops grouped by signature. Duplicate groups are intentionally kept
        so attachment can reject ambiguous mappings.
    """

    uop_by_label = {capture.label_raw: capture.uop for capture in captures}
    grouped: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for ordinal, op in enumerate(ops):
        uop = uop_by_label.get(op._label_raw)
        if uop is None:
            continue
        key = _tinygrad_signature_key(
            uop_signature=_uop_signature(uop),
            ordinal=ordinal,
            parent_signatures=tuple(_uop_signature(src) for src in getattr(uop, "src", ())),
            shape=tuple(getattr(op, "shape", ()) or ()),
            dtype=str(getattr(op, "dtype", "")),
        )
        grouped[key].append(op)
    return grouped


def _tinygrad_live_intermediate_candidates(
    *,
    backend: TinygradBackend,
    outputs: Sequence[Any],
    observed_tensors: Mapping[int, list[Any]],
) -> dict[tuple[Any, ...], list[TinygradIntermediateCandidate]]:
    """Group no-realize live tensors by conservative tinygrad T1 signature.

    Parameters
    ----------
    backend
        tinygrad backend instance used for tensor checks.
    outputs
        Tensor outputs from the no-realize derived pass.
    observed_tensors
        Live tensors retained by the Tensor API observer, keyed by UOp id.

    Returns
    -------
    dict[tuple[Any, ...], list[TinygradIntermediateCandidate]]
        Live candidates grouped by signature. Duplicate groups remain ambiguous
        and are skipped by the attachment step.
    """

    grouped: dict[tuple[Any, ...], list[TinygradIntermediateCandidate]] = defaultdict(list)
    ordinal = 0
    for uop in _unique_uops(outputs):
        tensors = observed_tensors.get(id(uop), ())
        for tensor in tensors:
            if not backend.is_tensor(tensor):
                continue
            shape = tuple(getattr(tensor, "shape", ()) or ())
            dtype = str(getattr(tensor, "dtype", ""))
            key = _tinygrad_signature_key(
                uop_signature=_uop_signature(uop),
                ordinal=ordinal,
                parent_signatures=tuple(_uop_signature(src) for src in getattr(uop, "src", ())),
                shape=shape,
                dtype=dtype,
            )
            grouped[key].append(
                TinygradIntermediateCandidate(
                    signature=key,
                    tensor=tensor,
                    aval=f"Tensor(shape={shape}, dtype={dtype})",
                    dtype_ref=DtypeRef(backend="tinygrad", name=dtype),
                )
            )
        if tensors:
            ordinal += 1
    return grouped


def _identity(tensor: Any) -> str:
    """Return the versioned tinygrad identity string used for audit metadata.

    Parameters
    ----------
    tensor
        tinygrad Tensor.

    Returns
    -------
    str
        Structural identity containing object, UOp, buffer, view, and mutation fields.
    """

    uop = getattr(tensor, "uop", None)
    base = getattr(uop, "base", None)
    view = getattr(uop, "st", None)
    lineage_hash = hash(_uop_signature(uop)) if uop is not None else 0
    return (
        f"obj={id(tensor)};uop={id(uop)};lineage={lineage_hash};"
        f"buffer={id(base)};view={id(view)};mutation=0"
    )


def _nbytes(tensor: Any) -> int | None:
    """Return tinygrad tensor byte size when available.

    Parameters
    ----------
    tensor
        tinygrad Tensor.

    Returns
    -------
    int | None
        Estimated byte size.
    """

    try:
        return int(tensor.nbytes())
    except Exception:
        try:
            return int(tensor.numel() * tensor.dtype.itemsize)
        except Exception:
            return None


def _payload_list(tensor: Any) -> Any:
    """Return a host payload list/scalar for comparison.

    Parameters
    ----------
    tensor
        tinygrad Tensor.

    Returns
    -------
    Any
        Host scalar/list payload.
    """

    return tensor.tolist()


def _saved_single_output(
    op: Any,
    hidden_outputs_by_label: Mapping[str, Any] | None = None,
) -> Any:
    """Return one operation's saved payload, failing when it was dropped.

    Parameters
    ----------
    op
        Materialized TorchLens operation.
    hidden_outputs_by_label
        Runtime-only replay payloads keyed by raw op label.

    Returns
    -------
    Any
        Saved tinygrad tensor payload.
    """

    if not getattr(op, "has_saved_activation", False):
        label_raw = getattr(op, "_label_raw", None)
        if (
            isinstance(label_raw, str)
            and hidden_outputs_by_label
            and label_raw in hidden_outputs_by_label
        ):
            return hidden_outputs_by_label[label_raw]
        raise ValueError("tinygrad validation requires every replay payload to be saved.")
    output = op.out
    if output is None:
        raise ValueError("tinygrad validation found a missing saved payload.")
    return output


def _parent_perturbations_change_output(
    *,
    backend: TinygradBackend,
    capture: TinygradUOpCapture,
    op: Any,
    ops_by_raw_label: Mapping[str, Any],
    hidden_outputs_by_label: Mapping[str, Any],
    saved_output: Any,
) -> bool:
    """Return whether at least one recorded parent perturbation changes child output.

    Parameters
    ----------
    backend
        tinygrad backend instance used for replay helpers.
    capture
        Captured UOp metadata.
    op
        Materialized TorchLens operation for the UOp.
    ops_by_raw_label
        Materialized operations keyed by raw label.
    hidden_outputs_by_label
        Runtime-only replay payloads keyed by raw op label.
    saved_output
        Saved child output payload.

    Returns
    -------
    bool
        True when a value parent perturbation affects replayed child output.
    """

    graph_positions = getattr(op, "parent_arg_positions", {}).get("args", {})
    if not graph_positions:
        return True
    positions_by_parent: dict[str, list[int]] = {}
    for position, parent_label in graph_positions.items():
        positions_by_parent.setdefault(parent_label, []).append(position)
    attempted = False
    for parent_label, positions in positions_by_parent.items():
        parent_op = ops_by_raw_label[parent_label]
        parent_value = _saved_single_output(parent_op, hidden_outputs_by_label)
        value_positions = tuple(
            position
            for position in positions
            if _source_matches_payload(capture.uop.src[position], parent_value)
        )
        if not value_positions:
            continue
        for candidate in _perturb_candidates(parent_value):
            attempted = True
            replacements = {position: candidate for position in value_positions}
            try:
                perturbed_output = backend._replay_uop_from_trace_graph(
                    capture,
                    op,
                    ops_by_raw_label,
                    replacements=replacements,
                    hidden_outputs_by_label=hidden_outputs_by_label,
                )
            except Exception:
                continue
            if not _payloads_close(perturbed_output, saved_output):
                return True
    return not attempted


def _perturb_candidates(value: Any) -> tuple[Any, ...]:
    """Return deterministic perturbation candidates for a tinygrad tensor.

    Parameters
    ----------
    value
        Parent tinygrad payload to perturb.

    Returns
    -------
    tuple[Any, ...]
        Perturbed realized tinygrad tensors with the same dtype and device.
    """

    from tinygrad import Tensor

    payload = _payload_list(value)
    dtype_name = str(getattr(value, "dtype", ""))
    candidates: tuple[Any, ...]
    if "bool" in dtype_name:
        candidates = (_map_payload(payload, lambda item: not bool(item)),)
    elif "int" in dtype_name:
        candidates = (
            _map_payload(payload, lambda item: int(item) + 1),
            _map_payload(payload, lambda item: 0),
        )
    else:
        magnitude = _payload_max_abs(payload) + 1.0
        candidates = (
            _map_payload(payload, lambda item: float(item) + magnitude),
            _map_payload(payload, lambda item: float(item) - magnitude),
            _map_payload(payload, lambda item: 0.0),
        )
    return tuple(
        Tensor(candidate, dtype=value.dtype, device=value.device).realize()
        for candidate in candidates
    )


def _source_matches_payload(source_uop: Any, value: Any) -> bool:
    """Return whether a UOp source can be replaced by a saved tensor payload.

    Parameters
    ----------
    source_uop
        Original UOp source.
    value
        Saved tinygrad tensor payload.

    Returns
    -------
    bool
        True when dtype and shape match the saved payload.
    """

    try:
        from tinygrad import Tensor

        if _uop_name(source_uop) in {"CONST", "STACK"}:
            return False
        source_tensor = Tensor(source_uop)
    except Exception:
        return False
    return tuple(source_tensor.shape) == tuple(value.shape) and str(source_tensor.dtype) == str(
        value.dtype
    )


def _map_payload(value: Any, fn: Callable[[Any], Any]) -> Any:
    """Map a scalar function over a nested payload.

    Parameters
    ----------
    value
        Scalar or nested list payload.
    fn
        Scalar mapper.

    Returns
    -------
    Any
        Payload with ``fn`` applied to every scalar leaf.
    """

    if isinstance(value, list):
        return [_map_payload(item, fn) for item in value]
    return fn(value)


def _payload_max_abs(value: Any) -> float:
    """Return the maximum absolute scalar magnitude in a nested payload.

    Parameters
    ----------
    value
        Scalar or nested list payload.

    Returns
    -------
    float
        Maximum absolute value, or zero for empty lists.
    """

    if isinstance(value, list):
        return max((_payload_max_abs(item) for item in value), default=0.0)
    return abs(float(value))


def _payloads_close(left: Any, right: Any) -> bool:
    """Return whether two tinygrad tensor payloads match within dtype-aware tolerance.

    Parameters
    ----------
    left
        Left tinygrad tensor.
    right
        Right tinygrad tensor.

    Returns
    -------
    bool
        True when shape, dtype, and payload values match.
    """

    if tuple(left.shape) != tuple(right.shape) or str(left.dtype) != str(right.dtype):
        return False
    return _payload_values_close(_payload_list(left), _payload_list(right), str(left.dtype))


def _normalize_tinygrad_input_grad_argnums(
    value: Sequence[int] | None, num_inputs: int
) -> tuple[int, ...]:
    """Normalize tinygrad input gradient argnums.

    Parameters
    ----------
    value
        User-supplied positional input argnums, or ``None`` for all inputs.
    num_inputs
        Number of positional inputs passed to the tinygrad callable.

    Returns
    -------
    tuple[int, ...]
        Unique non-negative positional input indexes.
    """

    if value is None:
        return tuple(range(num_inputs))
    normalized = tuple(sorted({index if index >= 0 else num_inputs + index for index in value}))
    invalid = [index for index in normalized if index < 0 or index >= num_inputs]
    if invalid:
        raise ValueError(
            f"input_grad_argnums indexes out of range for {num_inputs} inputs: {invalid}."
        )
    return normalized


def _differentiated_input_leaves(
    backend: TinygradBackend,
    args: Sequence[Any],
    input_argnums: Sequence[int],
) -> tuple[tuple[str, int, Any], ...]:
    """Return differentiated tinygrad input tensor leaves.

    Parameters
    ----------
    backend
        tinygrad backend instance used for tensor checks.
    args
        Positional callable arguments.
    input_argnums
        Positional input indexes to differentiate.

    Returns
    -------
    tuple[tuple[str, int, Any], ...]
        ``(path, argnum, tensor)`` records for tensor leaves.
    """

    leaves: list[tuple[str, int, Any]] = []
    for argnum in input_argnums:
        for local_path, leaf in _tree_leaves_with_paths(args[argnum]):
            if not backend.is_tensor(leaf):
                continue
            suffix = f".{local_path}" if local_path else ""
            leaves.append((f"inputs.{argnum}{suffix}", argnum, leaf))
    return tuple(leaves)


def _snapshot_tinygrad_grads(
    backend: TinygradBackend,
    leaves: Sequence[tuple[str, int, Any]],
) -> dict[str, tuple[Any, Any | None, Any | None]]:
    """Snapshot pre-existing tinygrad leaf grads before backward.

    Parameters
    ----------
    backend
        tinygrad backend instance used for realized copies.
    leaves
        Differentiated input leaves.

    Returns
    -------
    dict[str, tuple[Any, Any | None, Any | None]]
        Mapping from derived-grad path to ``(leaf, original_grad_object, value_snapshot)``.
    """

    snapshots: dict[str, tuple[Any, Any | None, Any | None]] = {}
    for path, _argnum, tensor in leaves:
        original_grad = getattr(tensor, "grad", None)
        grad_snapshot = None if original_grad is None else backend._realized_copy(original_grad)
        snapshots[path] = (tensor, original_grad, grad_snapshot)
    return snapshots


def _records_for_tinygrad_leaf_grads(
    *,
    backend: TinygradBackend,
    leaves: Sequence[tuple[str, int, Any]],
    snapshots: Mapping[str, tuple[Any, Any | None, Any | None]],
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records from bracketed tinygrad leaf grads.

    Parameters
    ----------
    backend
        tinygrad backend instance used for realized copies.
    leaves
        Differentiated input leaves.
    snapshots
        Pre-backward gradient snapshots.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable input leaf path.
    """

    records: dict[str, DerivedGradRecord] = {}
    for path, argnum, tensor in leaves:
        post_grad = getattr(tensor, "grad", None)
        if post_grad is None:
            raise ValueError(f"tinygrad backward did not populate .grad for leaf {path!r}.")
        _tensor, _original_grad, grad_snapshot = snapshots[path]
        increment = post_grad if grad_snapshot is None else post_grad - grad_snapshot
        grad = backend._realized_copy(increment)
        records[path] = DerivedGradRecord(
            path=path,
            source="inputs",
            argnum=argnum,
            input_argnum=argnum,
            aval=f"Tensor(shape={tuple(getattr(grad, 'shape', ()))}, dtype={getattr(grad, 'dtype', None)})",
            dtype_ref=DtypeRef(backend="tinygrad", name=str(getattr(grad, "dtype", ""))),
            grad=grad,
            provenance=provenance,
        )
    return records


def _records_for_tinygrad_intermediate_grads(
    *,
    backend: TinygradBackend,
    trace_signatures: Mapping[tuple[Any, ...], list[Any]],
    live_candidates: Mapping[tuple[Any, ...], list[TinygradIntermediateCandidate]],
    snapshots: Mapping[str, tuple[Any, Any | None, Any | None]],
    provenance: Mapping[str, Any],
) -> dict[str, IntermediateDerivedGradRecord]:
    """Build exact tinygrad intermediate-derived gradient records.

    Parameters
    ----------
    backend
        tinygrad backend instance used for realized grad copies.
    trace_signatures
        Saved trace ops grouped by conservative signature.
    live_candidates
        Live no-realize tensors grouped by conservative signature.
    snapshots
        User input grad snapshots used only to avoid attaching records for the
        differentiated input leaves themselves.
    provenance
        Shared provenance metadata for exact attached records.

    Returns
    -------
    dict[str, IntermediateDerivedGradRecord]
        Records keyed by pass-qualified op label. Ambiguous or missing matches
        are skipped.
    """

    records: dict[str, IntermediateDerivedGradRecord] = {}
    skipped_input_ids = {id(tensor) for tensor, _original, _snapshot in snapshots.values()}
    for signature, ops in trace_signatures.items():
        candidates = live_candidates.get(signature, ())
        if len(ops) != 1 or len(candidates) != 1:
            continue
        op = ops[0]
        candidate = candidates[0]
        if id(candidate.tensor) in skipped_input_ids:
            continue
        grad = getattr(candidate.tensor, "grad", None)
        if grad is None:
            continue
        copied_grad = backend._realized_copy(grad)
        records[op.label] = IntermediateDerivedGradRecord(
            op_label=op.label,
            layer_label=op.layer_label,
            aval=candidate.aval,
            dtype_ref=candidate.dtype_ref,
            grad=copied_grad,
            provenance=dict(provenance),
        )
    return records


def _restore_tinygrad_grads(snapshots: Mapping[str, tuple[Any, Any | None, Any | None]]) -> None:
    """Restore pre-existing tinygrad leaf grad state after bracketed backward.

    Parameters
    ----------
    snapshots
        Mapping from derived-grad path to ``(leaf, original_grad_object, value_snapshot)``.

    Returns
    -------
    None
        Leaf ``.grad`` values are restored in place where possible.
    """

    for path, (tensor, original_grad, grad_snapshot) in snapshots.items():
        del path
        if original_grad is None:
            tensor.grad = None
            continue
        original_grad.assign(grad_snapshot)
        tensor.grad = original_grad


def _tinygrad_outputs_close(backend: TinygradBackend, left: Any, right: Any) -> bool:
    """Return whether two tinygrad output trees have matching tensor payloads.

    Parameters
    ----------
    backend
        tinygrad backend instance used for tensor checks.
    left
        First output tree.
    right
        Second output tree.

    Returns
    -------
    bool
        True when tensor leaves have matching paths and payloads.
    """

    left_leaves = [
        (path, leaf) for path, leaf in _tree_leaves_with_paths(left) if backend.is_tensor(leaf)
    ]
    right_leaves = [
        (path, leaf) for path, leaf in _tree_leaves_with_paths(right) if backend.is_tensor(leaf)
    ]
    if [path for path, _leaf in left_leaves] != [path for path, _leaf in right_leaves]:
        return False
    return all(
        _payloads_close(backend._realized_copy(left_leaf), backend._realized_copy(right_leaf))
        for (_left_path, left_leaf), (_right_path, right_leaf) in zip(left_leaves, right_leaves)
    )


def _is_scalar_tinygrad_value(value: Any) -> bool:
    """Return whether a tinygrad value is scalar-shaped.

    Parameters
    ----------
    value
        Candidate tinygrad tensor.

    Returns
    -------
    bool
        True when ``value`` has shape ``()``.
    """

    return tuple(getattr(value, "shape", ())) == ()


def _callable_identity(fn: Callable[[Any], Any] | None) -> str | None:
    """Return a stable best-effort callable identity.

    Parameters
    ----------
    fn
        Callable or ``None``.

    Returns
    -------
    str | None
        Identity string used in derived-gradient provenance.
    """

    if fn is None:
        return None
    return f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}:{id(fn)}"


def _payload_values_close(left: Any, right: Any, dtype_name: str) -> bool:
    """Return whether nested payload values are close for a dtype family.

    Parameters
    ----------
    left
        Left scalar or list payload.
    right
        Right scalar or list payload.
    dtype_name
        tinygrad dtype string.

    Returns
    -------
    bool
        True when values match under the dtype family's comparison rule.
    """

    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(
            _payload_values_close(left_item, right_item, dtype_name)
            for left_item, right_item in zip(left, right)
        )
    if "bool" in dtype_name or "int" in dtype_name:
        return left == right
    return abs(float(left) - float(right)) <= 1e-6 + 1e-5 * abs(float(right))
