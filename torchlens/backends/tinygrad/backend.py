"""UOp-snapshot tinygrad backend preview."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from ..._deprecations import MISSING, MissingType
from ...backends import BackendName, BackendUnsupportedError
from ...data_classes.layer import Layer
from ...data_classes.module import ModuleAccessor
from ...data_classes.param import ParamAccessor
from ...data_classes.trace import Trace
from ...fastlog.types import CaptureSpec
from ...ir.buffer import CaptureEvents
from ...ir.events import ArgTemplateRef, FunctionCallRef, OpEvent, OutputRef, ParentEdge
from ...ir.predicate import RecordContext
from ...ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...postprocess._materialize import materialize_from_events
from ...postprocess.finalization import _build_root_module_log
from ...quantities import Duration


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
        **kwargs
            Extra public trace kwargs rejected by this backend.

        Returns
        -------
        Trace
            Captured tinygrad trace.
        """

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
        args = self._normalize_input_args(input_args)
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
        input_identities = self._input_identities(args)
        with _observe_tensor_ops(observed_ops), _reject_mid_capture_execution():
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
        captures = self._emit_uop_graph(trace, outputs, uop_labels, observed_ops)
        self._mark_output_events(trace, outputs, uop_labels)
        materialize_from_events(trace, trace.capture_events)
        delattr(trace, "capture_events")
        trace.param_logs = ParamAccessor({})
        trace.num_param_tensors = 0
        trace.num_params = 0
        trace.num_params_trainable = 0
        trace.num_params_frozen = 0
        trace.num_layers_with_params = 0
        trace.param_source = "none"
        self._finish_trace(trace)
        trace.tinygrad_uop_captures = captures
        trace.tinygrad_payload_policy = "dev_python_realized_copy"
        return trace

    def validate_trace(self, trace: Trace, *_args: Any, **kwargs: Any) -> bool:
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
        bool
            True when replayed UOp payloads match saved live payload copies.
        """

        try:
            if kwargs.get("validate_metadata", True):
                from ...validation.invariants import check_metadata_invariants

                check_metadata_invariants(trace)
            return self._validate_uops(trace)
        except BackendUnsupportedError:
            raise
        except Exception:
            return False

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
        return self.validate_trace(trace, validate_metadata=validate_metadata)

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
            params=(),
            parent_params=(),
            module_stack=(),
            modules=(),
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
        self, trace: Trace, outputs: Sequence[Any], uop_labels: Mapping[int, str]
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

        Returns
        -------
        None
            Output-parent flags are updated in place.
        """

        labels = tuple(
            label
            for output in outputs
            if (label := uop_labels.get(id(cast(Any, output).uop))) is not None
        )
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

    def _finish_trace(self, trace: Trace) -> None:
        """Finalize materialized tinygrad raw logs into public accessors.

        Parameters
        ----------
        trace
            Trace to finalize.

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
        trace._layers_logged = True
        trace._layers_saved = True
        trace.has_backward_pass = False
        trace.capture_end_time = time.time()
        trace.backend = cast(BackendName, self.name)
        trace.module_identity_mode = "function_root"
        self._attach_function_root_module(trace)
        trace._tracing_finished = True

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

        captures = tuple(getattr(trace, "tinygrad_uop_captures", ()))
        if not captures:
            raise BackendUnsupportedError(
                "tinygrad validation requires live DEV=PYTHON realized-copy payloads; "
                "audit-only traces cannot be replay validated."
            )
        ops_by_raw_label = {
            getattr(op, "_label_raw", ""): op for op in getattr(trace, "layer_list", ())
        }
        for capture in captures:
            op = ops_by_raw_label.get(capture.label_raw)
            if op is None:
                return False
            saved_output = _saved_single_output(op)
            replayed = self._replay_uop_from_trace_graph(capture, op, ops_by_raw_label)
            if not _payloads_close(replayed, saved_output):
                return False
            if not _parent_perturbations_change_output(
                backend=self,
                capture=capture,
                op=op,
                ops_by_raw_label=ops_by_raw_label,
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

        Returns
        -------
        Any
            Realized tinygrad tensor replay output.
        """

        src = list(getattr(capture.uop, "src", ()) or ())
        graph_positions = getattr(op, "parent_arg_positions", {}).get("args", {})
        parent_labels = tuple(getattr(op, "parents", ()))
        positioned_labels = {label for label in graph_positions.values() if isinstance(label, str)}
        if positioned_labels != set(parent_labels):
            raise ValueError("tinygrad trace parent labels and parent_arg_positions disagree.")
        if tuple(sorted(graph_positions.items())) != tuple(sorted(capture.parent_arg_positions)):
            raise ValueError("tinygrad trace parent_arg_positions changed after capture.")
        for position, parent_label in graph_positions.items():
            if not isinstance(position, int) or position < 0 or position >= len(src):
                raise ValueError(f"tinygrad trace parent arg position {position!r} is invalid.")
            parent_op = ops_by_raw_label[parent_label]
            parent_value = _saved_single_output(parent_op)
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
                "tinygrad backend preview is full-save only and does not support "
                f"save-shaping or runtime-mutation options: {names}."
            )


class _observe_tensor_ops:
    """Context manager observing tinygrad Tensor API UOp results."""

    def __init__(self, observed_ops: dict[int, list[str]]) -> None:
        """Initialize the observation context.

        Parameters
        ----------
        observed_ops
            Mutable mapping receiving Tensor API names by returned UOp id.
        """

        self.observed_ops = observed_ops
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


def _saved_single_output(op: Any) -> Any:
    """Return one operation's saved payload, failing when it was dropped.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    Any
        Saved tinygrad tensor payload.
    """

    if not getattr(op, "has_saved_activation", False):
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
        parent_value = _saved_single_output(parent_op)
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
