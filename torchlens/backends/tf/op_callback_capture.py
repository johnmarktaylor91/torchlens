"""TensorFlow eager op-callback capture engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

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
from ...ir.refs import DeviceRef, DtypeRef, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ..registry import BackendUnsupportedError
from .modules import TFModuleTree, patched_tf_module_stack

_INIT_OP_TYPES = frozenset(
    {
        "StatelessRandomUniformV2",
        "RandomUniform",
        "Fill",
        "VarHandleOp",
        "AssignVariableOp",
    }
)
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TFSourceRecord:
    """Typed TensorFlow source tensor record.

    Parameters
    ----------
    kind
        Source kind such as ``"input"``, ``"resource"``, or ``"constant/factory"``.
    ref_key
        Stable eager tensor reference key.
    tensor
        Source tensor object.
    label_raw
        Source event label when one is materialized.
    detail
        Optional diagnostic detail.
    """

    kind: str
    ref_key: object
    tensor: Any
    label_raw: str | None
    detail: str | None = None


@dataclass(frozen=True)
class TFUnresolvedProducer:
    """Consumed TensorFlow tensor with no producer and no registered source.

    Parameters
    ----------
    consumer_label_raw
        Raw label of the consuming op.
    consumer_op_type
        Consuming TensorFlow op type.
    input_index
        Positional input index on the callback.
    ref_key
        Tensor reference key.
    """

    consumer_label_raw: str
    consumer_op_type: str
    input_index: int
    ref_key: object


@dataclass(frozen=True)
class TFInputCapture:
    """Recorded TensorFlow callback input slot.

    Parameters
    ----------
    input_index
        Positional callback input index.
    producer_label_raw
        Raw label of a prior callback output, when the input is produced by a
        captured op.
    source_kind
        Typed source kind when the input is a registered source.
    source_label_raw
        Raw source label when a source event was materialized.
    tensor
        Live TensorFlow tensor observed in the callback input slot.
    ref_key
        Stable eager tensor reference key.
    """

    input_index: int
    producer_label_raw: str | None
    source_kind: str | None
    source_label_raw: str | None
    tensor: Any
    ref_key: object | None


@dataclass(frozen=True)
class TFOpCapture:
    """Recorded TensorFlow callback output used by replay validation.

    Parameters
    ----------
    label_raw
        Raw TorchLens label for this callback output.
    op_type
        Normalized TensorFlow op type.
    attrs
        Raw TensorFlow callback attributes normalized into a mapping.
    output_index
        Output index within the callback invocation.
    inputs
        Recorded callback input slots.
    output_tensor
        Live TensorFlow output tensor for this output slot.
    """

    label_raw: str
    op_type: str
    attrs: dict[str, Any]
    output_index: int
    inputs: tuple[TFInputCapture, ...]
    output_tensor: Any


@dataclass(frozen=True)
class TFCaptureResult:
    """Result of one TensorFlow eager capture.

    Parameters
    ----------
    output
        Raw model output from the single captured forward.
    events
        TorchLens capture events emitted during the forward.
    source_records
        Typed source records observed during capture.
    unresolved_producers
        Consumed tensors with missing producer/source lineage.
    init_op_labels
        Captured initializer/variable-creation op labels.
    op_type_counts
        Captured op histogram.
    op_captures
        Per-output callback records used by replay validation.
    """

    output: Any
    events: CaptureEvents
    source_records: tuple[TFSourceRecord, ...]
    unresolved_producers: tuple[TFUnresolvedProducer, ...]
    init_op_labels: tuple[str, ...]
    op_type_counts: dict[str, int]
    op_captures: tuple[TFOpCapture, ...]


class TFEagerCaptureSession:
    """Stateful TensorFlow eager op-callback capture session."""

    def __init__(
        self,
        *,
        tf: Any,
        callable_obj: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        module_tree: TFModuleTree | None,
        save_payloads: bool = True,
        save_predicate: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize a TensorFlow eager capture session.

        Parameters
        ----------
        tf
            Imported TensorFlow module.
        callable_obj
            Callable forward entry.
        args
            Positional call arguments.
        kwargs
            Keyword call arguments.
        module_tree
            Discovered module tree, if object attribution is active.
        save_payloads
            Whether op output values should be snapshotted.
        save_predicate
            Optional static selector used to decide which op payloads are saved.
        """

        self.tf = tf
        self.callable_obj = callable_obj
        self.args = args
        self.kwargs = kwargs
        self.module_tree = module_tree
        self.save_payloads = save_payloads
        self.save_predicate = save_predicate
        self.events = CaptureEvents()
        self.module_stack: list[ModuleFrame] = []
        self.producer_by_ref: dict[object, str] = {}
        self.source_by_ref: dict[object, TFSourceRecord] = {}
        self.source_records: list[TFSourceRecord] = []
        self.unresolved_producers: list[TFUnresolvedProducer] = []
        self.init_op_labels: list[str] = []
        self.op_type_counts: Counter[str] = Counter()
        self.op_captures: list[TFOpCapture] = []
        self._source_label_by_ref: dict[object, str] = {}

    def run(self) -> TFCaptureResult:
        """Run one real eager forward under TensorFlow op callbacks.

        Returns
        -------
        TFCaptureResult
            Captured output, events, and validation side-channel records.
        """

        self._register_input_sources()
        callback_module = self._op_callbacks_module()

        def callback(
            op_type: Any,
            inputs: tuple[Any, ...],
            attrs: tuple[Any, ...] | dict[str, Any],
            outputs: tuple[Any, ...],
            op_name: str | None = None,
            graph: Any | None = None,
        ) -> None:
            self._callback(
                op_type=op_type,
                inputs=inputs,
                attrs=attrs,
                outputs=outputs,
                op_name=op_name,
                graph=graph,
            )

        callback_module.add_op_callback(callback)
        try:
            with patched_tf_module_stack(self.module_tree, self.tf, self.module_stack):
                output = self.callable_obj(*self.args, **self.kwargs)
        finally:
            callback_module.remove_op_callback(callback)
        return TFCaptureResult(
            output=output,
            events=self.events,
            source_records=tuple(self.source_records),
            unresolved_producers=tuple(self.unresolved_producers),
            init_op_labels=tuple(self.init_op_labels),
            op_type_counts=dict(self.op_type_counts),
            op_captures=tuple(self.op_captures),
        )

    def _callback(
        self,
        *,
        op_type: Any,
        inputs: tuple[Any, ...],
        attrs: tuple[Any, ...] | dict[str, Any],
        outputs: tuple[Any, ...],
        op_name: str | None,
        graph: Any | None,
    ) -> None:
        """Convert one TensorFlow callback invocation into one or more events.

        Parameters
        ----------
        op_type
            TensorFlow operation type, sometimes bytes.
        inputs
            Eager tensor inputs.
        attrs
            Raw TensorFlow callback attributes.
        outputs
            Eager tensor outputs.
        op_name
            Optional operation name.
        graph
            Callback graph object. Eager callbacks pass ``None``.

        Returns
        -------
        None
            Mutates capture state.
        """

        if graph is not None:
            return
        op_type_text = _normalize_op_type(op_type)
        self.op_type_counts[op_type_text] += 1
        parents, parent_positions, edge_uses, input_captures = self._parents_for_inputs(
            op_type_text=op_type_text,
            inputs=inputs,
        )
        output_tensors = tuple(output for output in outputs if _is_tf_tensor(output, self.tf))
        if not output_tensors:
            return
        labels = self.events.reserve_label_block(op_type_text.lower(), len(output_tensors))
        for output_index, (label, output) in enumerate(zip(labels, output_tensors, strict=True)):
            record_context = self._record_context(
                label=label,
                op_type_text=op_type_text,
                output=output,
                output_index=output_index,
                parents=parents,
            )
            save_payload = self._should_save_payload(record_context)
            tensor_ref = self._tensor_ref(
                tensor=output,
                label_raw=label.label_raw,
                save_payload=save_payload,
            )
            event = self._event_for_output(
                op_type_text=op_type_text,
                op_name=op_name,
                attrs=attrs,
                output_index=output_index,
                tensor_ref=tensor_ref,
                label=label,
                parents=parents,
                parent_positions=parent_positions,
                edge_uses=edge_uses,
                save_payload=save_payload,
                record_context=record_context,
            )
            self.events.append(event)
            self.op_captures.append(
                TFOpCapture(
                    label_raw=label.label_raw,
                    op_type=op_type_text,
                    attrs=_attrs_to_raw_dict(attrs),
                    output_index=output_index,
                    inputs=input_captures,
                    output_tensor=output,
                )
            )
            ref_key = _tensor_ref_key(output)
            if ref_key is not None:
                self.producer_by_ref[ref_key] = label.label_raw
            if op_type_text in _INIT_OP_TYPES:
                self.init_op_labels.append(label.label_raw)

    def _event_for_output(
        self,
        *,
        op_type_text: str,
        op_name: str | None,
        attrs: tuple[Any, ...] | dict[str, Any],
        output_index: int,
        tensor_ref: TensorRef,
        label: Any,
        parents: tuple[ParentEdge, ...],
        parent_positions: dict[str, dict[Any, str]],
        edge_uses: tuple[object, ...],
        save_payload: bool,
        record_context: RecordContext,
    ) -> OpEvent:
        """Build one TorchLens ``OpEvent`` for a TensorFlow callback output.

        Parameters
        ----------
        op_type_text
            Normalized TensorFlow op type.
        op_name
            Optional TensorFlow op name.
        attrs
            Raw callback attrs.
        output_index
            Index within the callback outputs.
        tensor_ref
            Captured output tensor reference.
        label
            Reserved TorchLens raw label.
        parents
            Parent producer edges.
        parent_positions
            Parent argument-position mapping.
        edge_uses
            Edge-use records.
        save_payload
            Whether this output payload was selected for snapshotting.
        record_context
            Predicate context used for save selection.

        Returns
        -------
        OpEvent
            Event ready for ``materialize_from_events``.
        """

        modules = tuple((frame.address, frame.call_index) for frame in self.module_stack)
        annotations: dict[str, object] = {
            "_tl_annotations": {
                "tf_op_name": op_name,
                "tf_attrs": _attrs_to_public_dict(attrs),
            }
        }
        return OpEvent(
            kind="op",
            label_raw=label.label_raw,
            layer_label_raw=label.label_raw,
            layer_type=label.layer_type,
            raw_index=label.raw_index,
            type_index=label.type_index,
            step_index=label.raw_index,
            source_trace=None,
            source_trace_id=None,
            tracing_finished=False,
            construction_done=True,
            function=FunctionCallRef(
                func=None,
                func_name=op_type_text,
                func_qualname=op_type_text,
                func_call_id=label.raw_index,
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
                func_config=(("tf_op_name", op_name), ("output_index", output_index)),
            ),
            output=OutputRef(
                tensor=tensor_ref,
                transformed_tensor=None,
                has_saved_activation=tensor_ref.payload is not None,
                output_device="same",
                activation_transform=None,
                detach_saved_activations=False,
                visualizer_path=None,
                multi_output_index=output_index if output_index else None,
                in_multi_output=output_index > 0,
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
            parent_arg_positions=parent_positions,
            _edge_uses=edge_uses,
            params=(),
            parent_params=(),
            module_stack=tuple(self.module_stack),
            modules=modules,
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
            policy=CapturePolicy(
                must_keep_topology=True,
                save_payload=save_payload,
                requires_isolation=False,
                save_args=False,
                save_code=False,
                save_rng=False,
                save_grad=False,
                stream=False,
            ),
            predicate_matched=save_payload,
            pass_index=1,
            grad_fn_class_qualname=None,
            grad_fn_handle=None,
            equivalence_class=op_type_text.lower(),
            is_transform=False,
            transform_kind=None,
            transform_chain=(),
            transform_config=annotations,
            transform_fn_name=None,
            transform_fn_qualname=None,
            transform_fn_source=None,
            is_output_parent=False,
            has_internal_source_ancestor=False,
            internal_source_ancestors=frozenset(),
            input_ancestors=frozenset(),
            root_ancestors=frozenset(),
            func_call_id=label.raw_index,
            is_bottom_level=True,
            is_scalar_bool=_is_scalar_bool_tensor_ref(tensor_ref),
            bool_value=_bool_value(tensor_ref.payload),
            intervention_fired=False,
            intervention_replaced=False,
            fire_results=(),
            intervention_template_ref=None,
            record_context=record_context,
            capture_spec=CaptureSpec(save_out=save_payload, save_metadata=True),
        )

    def _record_context(
        self,
        *,
        label: Any,
        op_type_text: str,
        output: Any,
        output_index: int,
        parents: tuple[ParentEdge, ...],
    ) -> RecordContext:
        """Build a static selector context for one TensorFlow output.

        Parameters
        ----------
        label
            Reserved TorchLens label metadata.
        op_type_text
            Normalized TensorFlow op type.
        output
            TensorFlow eager output tensor.
        output_index
            Index within the callback outputs.
        parents
            Parent producer edges.

        Returns
        -------
        RecordContext
            Context accepted by unified ``save=`` selectors.
        """

        module_frames = tuple(
            {
                "address": frame.address,
                "module_type": frame.module_type,
                "pass_index": frame.call_index,
            }
            for frame in self.module_stack
        )
        return RecordContext(
            kind="op",
            label=label.label,
            raw_label=label.label_raw,
            pass_index=1,
            event_index=label.raw_index,
            step_index=label.raw_index,
            layer_type=label.layer_type,
            type_index=label.type_index,
            raw_index=label.raw_index,
            func_name=op_type_text,
            address=None,
            module_type=None,
            module_pass_index=None,
            module_stack=module_frames,
            recent_events=(),
            recent_ops=(),
            parent_labels=tuple(parent.parent_label_raw for parent in parents),
            input_output_address=None,
            shape=_shape_tuple(output),
            dtype=DtypeRef(backend="tf", name=str(getattr(output, "dtype", ""))),
            tensor_device=DeviceRef(backend="tf", name=str(getattr(output, "device", ""))),
            tensor_requires_grad=None,
            output_index=output_index,
            is_bottom_level_func=True,
            time_since_pass_start=0.0,
            sample_id=None,
            label_raw=label.label_raw,
            label_prefix=label.layer_type,
            func_call_id=label.raw_index,
            parent_labels_raw=tuple(parent.parent_label_raw for parent in parents),
            is_output_parent=False,
            backend_requires_isolation=False,
            is_scalar_bool=_shape_tuple(output) == ()
            and "bool" in str(getattr(output, "dtype", "")),
            bool_value=None,
        )

    def _should_save_payload(self, record_context: RecordContext) -> bool:
        """Return whether a TensorFlow op output should be snapshotted.

        Parameters
        ----------
        record_context
            Static selector context for the op output.

        Returns
        -------
        bool
            Whether the callback should retain a NumPy payload.
        """

        if not self.save_payloads:
            return False
        if self.save_predicate is None:
            return True
        return bool(self.save_predicate(record_context))

    def _parents_for_inputs(
        self,
        *,
        op_type_text: str,
        inputs: tuple[Any, ...],
    ) -> tuple[
        tuple[ParentEdge, ...],
        dict[str, dict[Any, str]],
        tuple[object, ...],
        tuple[TFInputCapture, ...],
    ]:
        """Resolve parent edges and source records for callback inputs.

        Parameters
        ----------
        op_type_text
            Consumer op type.
        inputs
            Callback inputs.

        Returns
        -------
        tuple
            Parent edges, parent position maps, edge-use records, and
            per-input replay capture records.
        """

        edges: list[ParentEdge] = []
        arg_positions: dict[Any, str] = {}
        edge_uses: list[object] = []
        input_captures: list[TFInputCapture] = []
        seen: set[tuple[str, int]] = set()
        for index, input_tensor in enumerate(inputs):
            if not _is_tf_tensor(input_tensor, self.tf):
                continue
            ref_key = _tensor_ref_key(input_tensor)
            if ref_key is None:
                continue
            parent_label = self.producer_by_ref.get(ref_key)
            source: TFSourceRecord | None = self.source_by_ref.get(ref_key)
            if (
                source is not None
                and source.label_raw is not None
                and parent_label == source.label_raw
            ):
                parent_label = source.label_raw
            if parent_label is None:
                source = self._source_for_tensor(ref_key, input_tensor, op_type_text)
                parent_label = source.label_raw
            source_label = source.label_raw if source is not None else None
            is_source_input = source is not None and parent_label == source_label
            input_captures.append(
                TFInputCapture(
                    input_index=index,
                    producer_label_raw=None if is_source_input else parent_label,
                    source_kind=None if source is None else source.kind,
                    source_label_raw=source_label if is_source_input else None,
                    tensor=input_tensor,
                    ref_key=ref_key,
                )
            )
            if parent_label is None:
                if source is None:
                    consumer = f"{op_type_text.lower()}_pending"
                    self.unresolved_producers.append(
                        TFUnresolvedProducer(
                            consumer_label_raw=consumer,
                            consumer_op_type=op_type_text,
                            input_index=index,
                            ref_key=ref_key,
                        )
                    )
                continue
            key = (parent_label, index)
            if key in seen:
                continue
            seen.add(key)
            edges.append(ParentEdge(parent_label, index, "arg"))
            arg_positions[index] = parent_label
            edge_uses.append((parent_label, index, "arg"))
        return (
            tuple(edges),
            {"args": arg_positions, "kwargs": {}},
            tuple(edge_uses),
            tuple(input_captures),
        )

    def _source_for_tensor(
        self,
        ref_key: object,
        tensor: Any,
        consumer_op_type: str,
    ) -> TFSourceRecord:
        """Return or create a typed source record for an input tensor.

        Parameters
        ----------
        ref_key
            Tensor reference key.
        tensor
            TensorFlow tensor.
        consumer_op_type
            Consuming op type.

        Returns
        -------
        TFSourceRecord
            Registered source record.
        """

        existing = self.source_by_ref.get(ref_key)
        if existing is not None:
            return existing
        kind = _classify_source_tensor(tensor, consumer_op_type)
        label_raw = self._emit_source_event(kind, tensor) if kind in {"input"} else None
        record = TFSourceRecord(kind=kind, ref_key=ref_key, tensor=tensor, label_raw=label_raw)
        self.source_by_ref[ref_key] = record
        self.source_records.append(record)
        return record

    def _emit_source_event(self, kind: str, tensor: Any) -> str:
        """Emit a source ``OpEvent`` for a TensorFlow input tensor.

        Parameters
        ----------
        kind
            Source kind.
        tensor
            Source tensor.

        Returns
        -------
        str
            Raw source label.
        """

        label = self.events.reserve_label(kind)
        tensor_ref = self._tensor_ref(tensor=tensor, label_raw=label.label_raw, save_payload=True)
        event = OpEvent(
            kind="source",
            label_raw=label.label_raw,
            layer_label_raw=label.label_raw,
            layer_type=kind,
            raw_index=label.raw_index,
            type_index=label.type_index,
            step_index=label.raw_index,
            source_trace=None,
            source_trace_id=None,
            tracing_finished=False,
            construction_done=True,
            function=FunctionCallRef(
                func=None,
                func_name=kind,
                func_qualname=kind,
                func_call_id=label.raw_index,
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
                container_path=(),
                container_spec=None,
                child_versions=(),
            ),
            templates=None,
            parents=(),
            parent_arg_positions={"args": {}, "kwargs": {}},
            _edge_uses=(),
            params=(),
            parent_params=(),
            module_stack=tuple(self.module_stack),
            modules=(),
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
            policy=CapturePolicy(
                must_keep_topology=True,
                save_payload=True,
                requires_isolation=False,
                save_args=False,
                save_code=False,
                save_rng=False,
                save_grad=False,
                stream=False,
            ),
            predicate_matched=True,
            pass_index=1,
            grad_fn_class_qualname=None,
            grad_fn_handle=None,
            equivalence_class=kind,
            is_transform=False,
            transform_kind=None,
            transform_chain=(),
            transform_config={},
            transform_fn_name=None,
            transform_fn_qualname=None,
            transform_fn_source=None,
            is_output_parent=False,
            has_internal_source_ancestor=False,
            internal_source_ancestors=frozenset(),
            input_ancestors=frozenset({label.label_raw}),
            root_ancestors=frozenset({label.label_raw}),
            func_call_id=label.raw_index,
            is_bottom_level=True,
            is_scalar_bool=_is_scalar_bool_tensor_ref(tensor_ref),
            bool_value=_bool_value(tensor_ref.payload),
            intervention_fired=False,
            intervention_replaced=False,
            fire_results=(),
            intervention_template_ref=None,
        )
        self.events.append(event)
        self.producer_by_ref[_tensor_ref_key(tensor)] = label.label_raw
        return label.label_raw

    def _register_input_sources(self) -> None:
        """Register public TensorFlow tensor inputs before capture starts.

        Returns
        -------
        None
            Populates source maps.
        """

        for tensor in _iter_tf_tensors((self.args, self.kwargs), self.tf):
            ref_key = _tensor_ref_key(tensor)
            if ref_key is None or ref_key in self.source_by_ref:
                continue
            label_raw = self._emit_source_event("input", tensor)
            record = TFSourceRecord("input", ref_key, tensor, label_raw)
            self.source_by_ref[ref_key] = record
            self.source_records.append(record)

    def _tensor_ref(self, *, tensor: Any, label_raw: str, save_payload: bool) -> TensorRef:
        """Build a TensorRef for a TensorFlow tensor.

        Parameters
        ----------
        tensor
            TensorFlow tensor.
        label_raw
            Raw TorchLens label.
        save_payload
            Whether to snapshot the tensor value.

        Returns
        -------
        TensorRef
            Backend-neutral tensor metadata.
        """

        payload = _snapshot_tensor(tensor) if save_payload else None
        memory = _nbytes_from_tensor(tensor)
        return TensorRef(
            label_raw=label_raw,
            shape=_shape_tuple(tensor),
            dtype=str(getattr(tensor, "dtype", "")),
            device=str(getattr(tensor, "device", "")),
            requires_grad=None,
            memory=memory,
            payload=payload,
            blob_ref=None,
            backend_handle_id=str(_tensor_ref_key(tensor)),
        )

    def _op_callbacks_module(self) -> Any:
        """Import TensorFlow op callback helpers lazily.

        Returns
        -------
        Any
            TensorFlow private op-callback module.
        """

        try:
            from tensorflow.python.framework import op_callbacks
        except ImportError as exc:
            raise BackendUnsupportedError(
                "TensorFlow eager capture requires tensorflow.python.framework.op_callbacks."
            ) from exc
        return op_callbacks


def warm_up_tf_callable(callable_obj: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Run the build/warm-up forward outside active logging.

    Parameters
    ----------
    callable_obj
        TensorFlow callable.
    args
        Positional warm-up args.
    kwargs
        Keyword warm-up args.

    Returns
    -------
    Any
        Warm-up output.
    """

    return callable_obj(*args, **kwargs)


def _normalize_op_type(op_type: Any) -> str:
    """Normalize TensorFlow callback op type to text.

    Parameters
    ----------
    op_type
        Raw callback op type.

    Returns
    -------
    str
        Normalized operation type.
    """

    if isinstance(op_type, bytes):
        return op_type.decode("utf-8", errors="replace")
    return str(op_type)


def _attrs_to_public_dict(attrs: tuple[Any, ...] | dict[str, Any]) -> dict[str, str]:
    """Convert TensorFlow attrs into a compact serializable diagnostic mapping.

    Parameters
    ----------
    attrs
        Raw callback attrs.

    Returns
    -------
    dict[str, str]
        Stringified attributes.
    """

    if isinstance(attrs, dict):
        return {str(key): repr(value) for key, value in attrs.items()}
    if isinstance(attrs, tuple):
        return {str(index): repr(value) for index, value in enumerate(attrs)}
    return {}


def _attrs_to_raw_dict(attrs: tuple[Any, ...] | dict[str, Any]) -> dict[str, Any]:
    """Convert TensorFlow callback attrs into a raw replay mapping.

    Parameters
    ----------
    attrs
        Raw callback attrs.

    Returns
    -------
    dict[str, Any]
        Attribute mapping preserving runtime values for raw-op replay.
    """

    if isinstance(attrs, dict):
        return {str(key): value for key, value in attrs.items()}
    if isinstance(attrs, tuple):
        result: dict[str, Any] = {}
        iterator = iter(attrs)
        for key in iterator:
            try:
                value = next(iterator)
            except StopIteration:
                break
            result[str(key)] = value
        return result
    return {}


def _snapshot_tensor(tensor: Any) -> object | None:
    """Snapshot a TensorFlow eager tensor without issuing TensorFlow ops.

    Parameters
    ----------
    tensor
        TensorFlow tensor.

    Returns
    -------
    object | None
        NumPy value, or ``None`` when unsupported or over cap.
    """

    nbytes = _nbytes_from_tensor(tensor)
    if nbytes is not None and nbytes > _MAX_SNAPSHOT_BYTES:
        return None
    numpy_method = getattr(tensor, "numpy", None)
    if not callable(numpy_method):
        return None
    try:
        value = numpy_method()
    except Exception:
        return None
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _nbytes_from_tensor(tensor: Any) -> int | None:
    """Return TensorFlow tensor byte size when available.

    Parameters
    ----------
    tensor
        TensorFlow tensor.

    Returns
    -------
    int | None
        Byte size, or ``None``.
    """

    shape = _shape_tuple(tensor)
    dtype = getattr(tensor, "dtype", None)
    size = getattr(dtype, "size", None)
    if shape is None or size is None:
        return None
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * int(size)


def _shape_tuple(tensor: Any) -> tuple[int, ...] | None:
    """Return a TensorFlow tensor shape tuple.

    Parameters
    ----------
    tensor
        TensorFlow tensor.

    Returns
    -------
    tuple[int, ...] | None
        Shape tuple, or ``None``.
    """

    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _tensor_ref_key(tensor: Any) -> object | None:
    """Return TensorFlow eager tensor identity key from ``tensor.ref()``.

    Parameters
    ----------
    tensor
        TensorFlow tensor.

    Returns
    -------
    object | None
        Hashable TensorFlow reference key.
    """

    ref = getattr(tensor, "ref", None)
    if not callable(ref):
        return None
    try:
        return ref()
    except TypeError:
        return None


def _is_tf_tensor(value: Any, tf: Any) -> bool:
    """Return whether ``value`` is a TensorFlow tensor-like object.

    Parameters
    ----------
    value
        Candidate value.
    tf
        Imported TensorFlow module.

    Returns
    -------
    bool
        True for tensors and variables.
    """

    tensor_type = getattr(tf, "Tensor", None)
    variable_type = getattr(tf, "Variable", None)
    return bool(
        (tensor_type is not None and isinstance(value, tensor_type))
        or (variable_type is not None and isinstance(value, variable_type))
    )


def _iter_tf_tensors(value: Any, tf: Any) -> list[Any]:
    """Return TensorFlow tensor leaves nested inside ``value``.

    Parameters
    ----------
    value
        Candidate nested value.
    tf
        Imported TensorFlow module.

    Returns
    -------
    list[Any]
        Tensor leaves.
    """

    if _is_tf_tensor(value, tf):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors: list[Any] = []
        for item in value:
            tensors.extend(_iter_tf_tensors(item, tf))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_iter_tf_tensors(item, tf))
        return tensors
    return []


def _classify_source_tensor(tensor: Any, consumer_op_type: str) -> str:
    """Classify a non-produced TensorFlow callback input source.

    Parameters
    ----------
    tensor
        Source tensor.
    consumer_op_type
        Consuming op type.

    Returns
    -------
    str
        Typed source kind.
    """

    dtype_text = str(getattr(tensor, "dtype", ""))
    if "resource" in dtype_text:
        return "resource"
    if consumer_op_type in {"Reshape", "Shape", "StridedSlice", "Pack", "ConcatV2"}:
        return "shape/axis"
    return "constant/factory"


def _is_scalar_bool_tensor_ref(tensor_ref: TensorRef) -> bool:
    """Return whether a tensor ref describes a scalar bool.

    Parameters
    ----------
    tensor_ref
        Tensor metadata.

    Returns
    -------
    bool
        True for scalar bool tensors.
    """

    return tensor_ref.shape == () and "bool" in str(tensor_ref.dtype)


def _bool_value(payload: object | None) -> bool | None:
    """Return scalar bool payload as Python bool when possible.

    Parameters
    ----------
    payload
        Snapshotted payload.

    Returns
    -------
    bool | None
        Boolean value, or ``None``.
    """

    if isinstance(payload, np.ndarray) and payload.shape == () and payload.dtype == np.bool_:
        return bool(payload.item())
    if isinstance(payload, bool):
        return payload
    return None
