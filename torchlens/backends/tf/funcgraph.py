"""TensorFlow static FuncGraph capture engine."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

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
from ...validation.status import (
    REGION_REPLAY_CLASS,
    REGION_REPLAY_CLASS_KEY,
    REGION_REPLAY_IMPORTER_PROVENANCE,
    REGION_REPLAY_PROVENANCE_KEY,
)
from ..registry import BackendUnsupportedError
from .op_callback_capture import TFCaptureResult, TFInputCapture, TFOpCapture, TFSourceRecord

_CONTROL_FLOW_OP_TYPES = frozenset(
    {"If", "StatelessIf", "While", "StatelessWhile", "Case", "Switch", "Merge"}
    | {"StatefulPartitionedCall"}
)
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TFStaticCaptureResult:
    """Result of one TensorFlow static FuncGraph capture.

    Parameters
    ----------
    output
        Raw structured output from the pruned graph or the concrete function fallback.
    events
        TorchLens capture events emitted from the FuncGraph inventory.
    source_records
        Typed source records for graph feeds.
    unresolved_producers
        Missing producer records. Static capture emits none when graph edges bind.
    init_op_labels
        Initializer contamination labels. Static capture emits none.
    op_type_counts
        Captured graph op histogram.
    op_captures
        Per-output captures for validation replay where applicable.
    output_label_raws
        Raw labels corresponding to structured graph outputs.
    region_captures
        Raw labels for opaque control-flow or graph fallback regions.
    fallback_error
        Private diagnostic for opaque fallback captures.
    """

    output: Any
    events: CaptureEvents
    source_records: tuple[TFSourceRecord, ...]
    unresolved_producers: tuple[Any, ...]
    init_op_labels: tuple[str, ...]
    op_type_counts: dict[str, int]
    op_captures: tuple[TFOpCapture, ...]
    output_label_raws: tuple[str, ...]
    region_captures: tuple[str, ...]
    fallback_error: str | None = None

    def as_eager_shape(self) -> TFCaptureResult:
        """Return the side-channel subset shared with eager capture.

        Returns
        -------
        TFCaptureResult
            Result shaped like the eager capture payload.
        """

        return TFCaptureResult(
            output=self.output,
            events=self.events,
            source_records=self.source_records,
            unresolved_producers=self.unresolved_producers,
            init_op_labels=self.init_op_labels,
            op_type_counts=self.op_type_counts,
            op_captures=self.op_captures,
        )


def capture_static_funcgraph(
    *,
    tf: Any,
    model: Any,
    callable_obj: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    save_predicate: Callable[[Any], Any] | None,
) -> TFStaticCaptureResult:
    """Capture a graph-only TensorFlow callable from its promoted FuncGraph.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    model
        Public model object, used for SavedModel signature selection.
    callable_obj
        Callable selected by backend mode detection.
    args
        Positional capture inputs.
    kwargs
        Keyword capture inputs.
    save_predicate
        Optional static selector controlling which payloads are fetched.

    Returns
    -------
    TFStaticCaptureResult
        Static graph events and validation side channels.
    """

    _prefer_cpu_static_helpers(tf)
    concrete, bound_args, bound_kwargs = _concrete_function_for_call(
        tf=tf,
        model=model,
        callable_obj=callable_obj,
        args=args,
        kwargs=kwargs,
    )
    try:
        return _capture_promoted_graph(
            tf=tf,
            concrete=concrete,
            args=bound_args,
            kwargs=bound_kwargs,
            save_predicate=save_predicate,
        )
    except Exception as exc:
        return _capture_opaque_graph_region(
            tf=tf,
            concrete=concrete,
            args=bound_args,
            kwargs=bound_kwargs,
            reason="tf_static_prune_unavailable",
            error=f"{type(exc).__name__}: {exc}",
        )


def _prefer_cpu_static_helpers(tf: Any) -> None:
    """Hide GPUs from TensorFlow static helper sessions when possible.

    Parameters
    ----------
    tf
        Imported TensorFlow module.

    Returns
    -------
    None
        Mutates TensorFlow runtime device visibility when the runtime allows it.
    """

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        return
    except ValueError:
        return


def _concrete_function_for_call(
    *,
    tf: Any,
    model: Any,
    callable_obj: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    """Return a ConcreteFunction and arguments that bind to it.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    model
        Public model object.
    callable_obj
        Selected callable.
    args
        Public positional inputs.
    kwargs
        Public keyword inputs.

    Returns
    -------
    tuple[Any, tuple[Any, ...], dict[str, Any]]
        ConcreteFunction plus normalized call args and kwargs.
    """

    concrete_type = getattr(tf.types.experimental, "ConcreteFunction", None)
    if concrete_type is not None and isinstance(callable_obj, concrete_type):
        return _bind_concrete_call(callable_obj, args, kwargs)
    signatures = getattr(model, "signatures", None)
    if isinstance(signatures, Mapping) and signatures:
        concrete = signatures.get("serving_default") or next(iter(signatures.values()))
        return _bind_concrete_call(concrete, args, kwargs)
    if hasattr(callable_obj, "get_concrete_function"):
        concrete = callable_obj.get_concrete_function(*args, **kwargs)
        return concrete, args, kwargs
    call_attr = getattr(model, "call", None)
    if call_attr is not None and hasattr(call_attr, "get_concrete_function"):
        concrete = call_attr.get_concrete_function(*args, **kwargs)
        return concrete, args, kwargs
    call_dunder = getattr(model, "__call__", None)
    if call_dunder is not None and hasattr(call_dunder, "get_concrete_function"):
        concrete = call_dunder.get_concrete_function(*args, **kwargs)
        return concrete, args, kwargs
    raise BackendUnsupportedError("TensorFlow static capture requires a ConcreteFunction source.")


def _bind_concrete_call(
    concrete: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    """Return a ConcreteFunction with args/kwargs that bind it.

    Parameters
    ----------
    concrete
        ConcreteFunction being invoked.
    args
        Public positional inputs.
    kwargs
        Public keyword inputs.

    Returns
    -------
    tuple[Any, tuple[Any, ...], dict[str, Any]]
        ConcreteFunction, positional args, and keyword args.
    """

    if kwargs:
        return concrete, args, kwargs
    if len(args) != 1:
        return concrete, args, kwargs
    _, keyword_specs = getattr(concrete, "structured_input_signature", ((), {}))
    if isinstance(keyword_specs, Mapping) and len(keyword_specs) == 1:
        return concrete, (), {str(next(iter(keyword_specs))): args[0]}
    return concrete, args, kwargs


def _capture_promoted_graph(
    *,
    tf: Any,
    concrete: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    save_predicate: Callable[[Any], Any] | None,
) -> TFStaticCaptureResult:
    """Capture topology and selected values from an inlined frozen graph.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    concrete
        ConcreteFunction to promote and import.
    args
        Concrete positional inputs.
    kwargs
        Concrete keyword inputs.
    save_predicate
        Optional static selector controlling payload fetches.

    Returns
    -------
    TFStaticCaptureResult
        Captured static graph.
    """

    wrapped: Any | None = None
    try:
        frozen_func, graph_def = _convert_variables_to_constants(concrete)
        frozen_inputs = tuple(frozen_func.inputs)
        frozen_outputs = tuple(frozen_func.outputs)
    except Exception:
        graph = concrete.graph
        frozen_inputs = tuple(concrete.inputs)
        frozen_outputs = tuple(concrete.outputs)
    else:
        graph = None
    try:
        if graph is None:
            wrapped = _wrap_graph_def(tf, graph_def)
            graph = wrapped.graph
    except Exception:
        graph = _import_graph_def_graph(tf, graph_def)
    feed_tensors = [graph.get_tensor_by_name(tensor.name) for tensor in frozen_inputs]
    feed_values = _feed_values_for_concrete(concrete, frozen_inputs, args, kwargs)
    output_names = tuple(tensor.name for tensor in frozen_outputs)
    fetch_names = _selected_fetch_names(
        graph=graph,
        output_names=output_names,
        save_predicate=save_predicate,
    )
    fetch_tensors = [graph.get_tensor_by_name(name) for name in fetch_names]
    try:
        fetched = _execute_pruned_fetches(
            tf=tf,
            wrapped=wrapped,
            graph=graph,
            feeds=feed_tensors,
            feed_values=feed_values,
            fetches=fetch_tensors,
        )
        fetched_values = _normalize_fetch_result(fetched, len(fetch_names))
        payload_by_name = {
            name: _snapshot_value(value)
            for name, value in zip(fetch_names, fetched_values, strict=True)
        }
    except Exception:
        payload_by_name = _interpret_graph_payloads(
            graph=graph,
            input_names=tuple(tensor.name for tensor in frozen_inputs),
            input_values=tuple(feed_values),
            fetch_names=fetch_names,
        )
    output = _structured_output_from_fetches(output_names, payload_by_name)
    builder = _StaticEventBuilder(
        tf=tf,
        graph=graph,
        input_names=tuple(tensor.name for tensor in frozen_inputs),
        input_values=tuple(feed_values),
        payload_by_name=payload_by_name,
        save_predicate=save_predicate,
    )
    builder.emit()
    return TFStaticCaptureResult(
        output=output,
        events=builder.events,
        source_records=tuple(builder.source_records),
        unresolved_producers=(),
        init_op_labels=(),
        op_type_counts=dict(builder.op_type_counts),
        op_captures=tuple(builder.op_captures),
        output_label_raws=tuple(
            label for name in output_names if (label := builder.producer_by_tensor_name.get(name))
        ),
        region_captures=tuple(builder.region_labels),
    )


def _execute_pruned_fetches(
    *,
    tf: Any,
    wrapped: Any | None,
    graph: Any,
    feeds: Sequence[Any],
    feed_values: Sequence[Any],
    fetches: Sequence[Any],
) -> Any:
    """Execute promoted graph fetches with prune primary and CPU session fallback.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    wrapped
        Wrapped imported graph function, or ``None`` for the session-only fallback.
    graph
        Imported graph.
    feeds
        Feed tensors.
    feed_values
        Concrete feed values.
    fetches
        Fetch tensors.

    Returns
    -------
    Any
        Raw fetch result.
    """

    if wrapped is not None:
        try:
            pruned = wrapped.prune(list(feeds), list(fetches))
            return pruned(*feed_values)
        except Exception:
            pass
    return _execute_session_fetches(
        tf=tf,
        graph=graph,
        feeds=feeds,
        feed_values=feed_values,
        fetches=fetches,
    )


def _execute_session_fetches(
    *,
    tf: Any,
    graph: Any,
    feeds: Sequence[Any],
    feed_values: Sequence[Any],
    fetches: Sequence[Any],
) -> Any:
    """Execute graph fetches in a CPU-only TensorFlow v1 session.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    graph
        Imported graph.
    feeds
        Feed tensors.
    feed_values
        Concrete feed values.
    fetches
        Fetch tensors.

    Returns
    -------
    Any
        Raw session result.
    """

    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        device_count={"GPU": 0},
    )
    config.gpu_options.visible_device_list = ""
    feed_dict = {
        feed: _snapshot_value(value) if hasattr(value, "numpy") else value
        for feed, value in zip(feeds, feed_values, strict=False)
    }
    with tf.compat.v1.Session(graph=graph, config=config) as session:
        return session.run(list(fetches), feed_dict=feed_dict)


def _interpret_graph_payloads(
    *,
    graph: Any,
    input_names: tuple[str, ...],
    input_values: tuple[Any, ...],
    fetch_names: tuple[str, ...],
) -> dict[str, object | None]:
    """Interpret simple promoted graph ops as a session-free value fallback.

    Parameters
    ----------
    graph
        Promoted TensorFlow graph.
    input_names
        Feed tensor names.
    input_values
        Feed values aligned with ``input_names``.
    fetch_names
        Tensor names selected for payload materialization.

    Returns
    -------
    dict[str, object | None]
        Payloads keyed by fetched tensor name.
    """

    env: dict[str, object | None] = {
        name: _snapshot_value(value) for name, value in zip(input_names, input_values, strict=False)
    }
    for op in graph.get_operations():
        if op.type == "Placeholder":
            continue
        values = [env.get(tensor.name) for tensor in op.inputs]
        outputs = _interpret_op(op, values)
        if outputs is None:
            for output in op.outputs:
                env.setdefault(output.name, None)
            continue
        for output, value in zip(op.outputs, outputs, strict=False):
            env[output.name] = _snapshot_value(value)
    return {name: env.get(name) for name in fetch_names}


def _interpret_op(op: Any, values: Sequence[object | None]) -> tuple[object | None, ...] | None:
    """Interpret one simple TensorFlow graph op with NumPy semantics.

    Parameters
    ----------
    op
        TensorFlow graph operation.
    values
        Input payloads.

    Returns
    -------
    tuple[object | None, ...] | None
        Output payloads, or ``None`` when the op is not interpretable.
    """

    if op.type == "Const":
        return (_const_value(op),)
    if any(value is None for value in values):
        return None
    arrays = [np.asarray(value) for value in values]
    if op.type == "Identity":
        return (arrays[0],)
    if op.type == "AddV2":
        return (arrays[0] + arrays[1],)
    if op.type == "Sub":
        return (arrays[0] - arrays[1],)
    if op.type == "Mul":
        return (arrays[0] * arrays[1],)
    if op.type == "MatMul":
        a = arrays[0].T if bool(op.get_attr("transpose_a")) else arrays[0]
        b = arrays[1].T if bool(op.get_attr("transpose_b")) else arrays[1]
        return (np.matmul(a, b),)
    if op.type == "BiasAdd":
        return (arrays[0] + arrays[1],)
    if op.type == "Relu":
        return (np.maximum(arrays[0], 0),)
    if op.type == "Sum":
        return (np.sum(arrays[0], axis=tuple(np.asarray(arrays[1]).tolist())),)
    if op.type == "Greater":
        return (np.greater(arrays[0], arrays[1]),)
    return None


def _const_value(op: Any) -> object | None:
    """Return a NumPy value from a TensorFlow Const NodeDef.

    Parameters
    ----------
    op
        Const operation.

    Returns
    -------
    object | None
        NumPy constant payload.
    """

    try:
        from tensorflow.python.framework import tensor_util

        return tensor_util.MakeNdarray(op.get_attr("value"))
    except Exception:
        return None


def _convert_variables_to_constants(concrete: Any) -> tuple[Any, Any]:
    """Inline variables/functions in a ConcreteFunction graph.

    Parameters
    ----------
    concrete
        ConcreteFunction to freeze.

    Returns
    -------
    tuple[Any, Any]
        Frozen ConcreteFunction and GraphDef.
    """

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    return convert_variables_to_constants_v2_as_graph(concrete, aggressive_inlining=True)


def _wrap_graph_def(tf: Any, graph_def: Any) -> Any:
    """Import a GraphDef into a prunable wrapped function.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    graph_def
        Promoted GraphDef.

    Returns
    -------
    Any
        Wrapped function with a graph suitable for pruning.
    """

    def import_graph_def() -> None:
        """Import the promoted graph into the wrapper graph."""

        tf.graph_util.import_graph_def(graph_def, name="")

    return tf.compat.v1.wrap_function(import_graph_def, [])


def _import_graph_def_graph(tf: Any, graph_def: Any) -> Any:
    """Import a GraphDef into a plain TensorFlow graph.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    graph_def
        Promoted GraphDef.

    Returns
    -------
    Any
        TensorFlow graph containing the promoted definition.
    """

    graph = tf.Graph()
    with graph.as_default():
        tf.graph_util.import_graph_def(graph_def, name="")
    return graph


def _selected_fetch_names(
    *,
    graph: Any,
    output_names: tuple[str, ...],
    save_predicate: Callable[[Any], Any] | None,
) -> tuple[str, ...]:
    """Return tensor names to fetch from the pruned graph.

    Parameters
    ----------
    graph
        Promoted TensorFlow graph.
    output_names
        Structured output tensor names.
    save_predicate
        Optional static selector.

    Returns
    -------
    tuple[str, ...]
        Fetch tensor names in stable graph order.
    """

    names: list[str] = []
    for op in graph.get_operations():
        for output_index, output in enumerate(op.outputs):
            if output.name in output_names or save_predicate is None:
                names.append(output.name)
                continue
            context = _record_context_for_symbolic(
                label_raw=f"{op.type.lower()}_pending",
                raw_index=0,
                type_index=0,
                op_type=op.type,
                op_name=op.name,
                output=output,
                output_index=output_index,
                parents=(),
                modules=_module_frames_from_name(op.name),
            )
            if bool(save_predicate(context)):
                names.append(output.name)
    return tuple(dict.fromkeys((*output_names, *names)))


def _feed_values_for_concrete(
    concrete: Any,
    frozen_inputs: Sequence[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    """Return concrete values in frozen graph input order.

    Parameters
    ----------
    concrete
        Original ConcreteFunction.
    frozen_inputs
        Frozen graph input tensors.
    args
        Bound positional inputs.
    kwargs
        Bound keyword inputs.

    Returns
    -------
    tuple[Any, ...]
        Feed values aligned with ``frozen_inputs``.
    """

    flat_values = list(args)
    if kwargs:
        _, keyword_specs = getattr(concrete, "structured_input_signature", ((), {}))
        if isinstance(keyword_specs, Mapping):
            for name in keyword_specs:
                if name in kwargs:
                    flat_values.append(kwargs[name])
    if len(flat_values) < len(frozen_inputs):
        captures = tuple(getattr(concrete, "captured_inputs", ()))
        flat_values.extend(captures[: len(frozen_inputs) - len(flat_values)])
    return tuple(flat_values[: len(frozen_inputs)])


def _normalize_fetch_result(value: Any, count: int) -> tuple[Any, ...]:
    """Normalize a pruned graph result into a tuple of fetch values.

    Parameters
    ----------
    value
        Raw pruned function output.
    count
        Expected fetch count.

    Returns
    -------
    tuple[Any, ...]
        Fetch values.
    """

    if count == 1:
        return (value,)
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _structured_output_from_fetches(
    output_names: tuple[str, ...],
    payload_by_name: Mapping[str, object | None],
) -> Any:
    """Build a compact raw output object from fetched graph outputs.

    Parameters
    ----------
    output_names
        Graph output tensor names.
    payload_by_name
        Fetched payloads keyed by tensor name.

    Returns
    -------
    Any
        Single payload or tuple of payloads.
    """

    outputs = tuple(payload_by_name.get(name) for name in output_names)
    if len(outputs) == 1:
        return outputs[0]
    return outputs


class _StaticEventBuilder:
    """Build TorchLens events from a promoted TensorFlow graph."""

    def __init__(
        self,
        *,
        tf: Any,
        graph: Any,
        input_names: tuple[str, ...],
        input_values: tuple[Any, ...],
        payload_by_name: Mapping[str, object | None],
        save_predicate: Callable[[Any], Any] | None,
    ) -> None:
        """Initialize a static event builder.

        Parameters
        ----------
        tf
            Imported TensorFlow module.
        graph
            Promoted TensorFlow graph.
        input_names
            Feed tensor names.
        input_values
            Feed values aligned with ``input_names``.
        payload_by_name
            Fetched payloads keyed by tensor name.
        save_predicate
            Optional save selector.
        """

        self.tf = tf
        self.graph = graph
        self.input_names = input_names
        self.input_values = input_values
        self.payload_by_name = payload_by_name
        self.save_predicate = save_predicate
        self.events = CaptureEvents()
        self.source_records: list[TFSourceRecord] = []
        self.op_captures: list[TFOpCapture] = []
        self.op_type_counts: Counter[str] = Counter()
        self.producer_by_tensor_name: dict[str, str] = {}
        self.region_labels: list[str] = []
        self.source_labels: set[str] = set()

    def emit(self) -> None:
        """Emit all graph operations in topological graph order.

        Returns
        -------
        None
            Mutates builder state.
        """

        input_value_by_name = dict(zip(self.input_names, self.input_values, strict=False))
        for op in self.graph.get_operations():
            if op.type == "Placeholder" and op.outputs:
                self._emit_source(op, input_value_by_name.get(op.outputs[0].name))
                continue
            if op.type in _CONTROL_FLOW_OP_TYPES:
                self._emit_region(op)
                continue
            self._emit_op(op)

    def _emit_source(self, op: Any, value: Any) -> None:
        """Emit one graph feed as an input source event.

        Parameters
        ----------
        op
            Placeholder operation.
        value
            Runtime feed value.

        Returns
        -------
        None
            Mutates event state.
        """

        label = self.events.reserve_label("input")
        output = op.outputs[0]
        payload = _snapshot_value(value)
        tensor_ref = _tensor_ref_for_symbolic(output, label.label_raw, payload)
        event = _base_event(
            label=label,
            kind="source",
            layer_type="input",
            func_name="input",
            op_name=op.name,
            output=tensor_ref,
            output_index=0,
            parents=(),
            parent_positions={"args": {}, "kwargs": {}},
            edge_uses=(),
            modules=(),
            module_stack=(),
            save_payload=True,
            annotations={"tf_static_op_name": op.name, "tf_static_op_type": op.type},
            equivalence_class="input",
            is_output_parent=False,
            predicate_matched=True,
            record_context=None,
        )
        self.events.append(event)
        self.producer_by_tensor_name[output.name] = label.label_raw
        self.source_labels.add(label.label_raw)
        record = TFSourceRecord(
            kind="input",
            ref_key=output.name,
            tensor=value,
            label_raw=label.label_raw,
        )
        self.source_records.append(record)

    def _emit_region(self, op: Any) -> None:
        """Emit one opaque unverified control-flow region.

        Parameters
        ----------
        op
            TensorFlow control-flow operation.

        Returns
        -------
        None
            Mutates event state.
        """

        trace_annotations = {
            REGION_REPLAY_CLASS_KEY: REGION_REPLAY_CLASS,
            REGION_REPLAY_PROVENANCE_KEY: REGION_REPLAY_IMPORTER_PROVENANCE,
            "tf_capture_kind": "region",
            "tf_region_op_type": op.type,
            "tf_region_name": op.name,
            "tf_region_replay_status": "unverified",
            "tf_region_reason": "tf_control_flow_region",
        }
        parents, parent_positions, edge_uses, inputs = self._parents_for_op(op)
        labels = self.events.reserve_label_block("tf_region", max(1, len(op.outputs)))
        for output_index, label in enumerate(labels):
            output = op.outputs[output_index] if output_index < len(op.outputs) else None
            tensor_ref = _tensor_ref_for_symbolic(
                output,
                label.label_raw,
                payload=None,
            )
            event = _base_event(
                label=label,
                kind="op",
                layer_type="tf_region",
                func_name=f"region:{op.type}",
                op_name=op.name,
                output=tensor_ref,
                output_index=output_index,
                parents=parents,
                parent_positions=parent_positions,
                edge_uses=edge_uses,
                modules=tuple(
                    (frame.address, frame.call_index) for frame in _module_frames_from_name(op.name)
                ),
                module_stack=_module_frames_from_name(op.name),
                save_payload=False,
                annotations=trace_annotations,
                equivalence_class=f"tf:region:{op.type}:{op.name}",
                is_output_parent=False,
                predicate_matched=False,
                record_context=None,
            )
            self.events.append(event)
            self.op_captures.append(
                TFOpCapture(
                    label_raw=label.label_raw,
                    op_type=op.type,
                    attrs=_node_attrs(op),
                    output_index=output_index,
                    inputs=inputs,
                    output_tensor=None,
                )
            )
            self.region_labels.append(label.label_raw)
            if output is not None:
                self.producer_by_tensor_name[output.name] = label.label_raw
        self.op_type_counts[op.type] += 1

    def _emit_op(self, op: Any) -> None:
        """Emit one regular TensorFlow graph operation.

        Parameters
        ----------
        op
            TensorFlow graph operation.

        Returns
        -------
        None
            Mutates event state.
        """

        parents, parent_positions, edge_uses, inputs = self._parents_for_op(op)
        outputs = tuple(op.outputs)
        if not outputs:
            return
        labels = self.events.reserve_label_block(op.type.lower(), len(outputs))
        module_stack = _module_frames_from_name(op.name)
        modules = tuple((frame.address, frame.call_index) for frame in module_stack)
        self.op_type_counts[op.type] += 1
        for output_index, (label, output) in enumerate(zip(labels, outputs, strict=True)):
            record_context = _record_context_for_symbolic(
                label_raw=label.label_raw,
                raw_index=label.raw_index,
                type_index=label.type_index,
                op_type=op.type,
                op_name=op.name,
                output=output,
                output_index=output_index,
                parents=parents,
                modules=module_stack,
            )
            payload = self.payload_by_name.get(output.name)
            save_payload = payload is not None
            annotations = {
                "tf_static_op_name": op.name,
                "tf_static_op_type": op.type,
                "tf_static_attrs": {key: repr(value) for key, value in _node_attrs(op).items()},
            }
            event = _base_event(
                label=label,
                kind="op",
                layer_type=op.type.lower(),
                func_name=op.type,
                op_name=op.name,
                output=_tensor_ref_for_symbolic(output, label.label_raw, payload),
                output_index=output_index,
                parents=parents,
                parent_positions=parent_positions,
                edge_uses=edge_uses,
                modules=modules,
                module_stack=module_stack,
                save_payload=save_payload,
                annotations=annotations,
                equivalence_class=op.type.lower(),
                is_output_parent=False,
                predicate_matched=save_payload,
                record_context=record_context,
            )
            self.events.append(event)
            self.op_captures.append(
                TFOpCapture(
                    label_raw=label.label_raw,
                    op_type=op.type,
                    attrs=_node_attrs(op),
                    output_index=output_index,
                    inputs=inputs,
                    output_tensor=payload,
                )
            )
            self.producer_by_tensor_name[output.name] = label.label_raw

    def _parents_for_op(
        self,
        op: Any,
    ) -> tuple[
        tuple[ParentEdge, ...],
        dict[str, dict[Any, str]],
        tuple[object, ...],
        tuple[TFInputCapture, ...],
    ]:
        """Resolve graph parent edges for an operation.

        Parameters
        ----------
        op
            TensorFlow graph operation.

        Returns
        -------
        tuple
            Parent edges, position maps, edge-use records, and validation inputs.
        """

        edges: list[ParentEdge] = []
        arg_positions: dict[Any, str] = {}
        edge_uses: list[object] = []
        input_captures: list[TFInputCapture] = []
        for index, input_tensor in enumerate(op.inputs):
            parent_label = self.producer_by_tensor_name.get(input_tensor.name)
            if parent_label is None:
                continue
            is_source_input = parent_label in self.source_labels
            edge = ParentEdge(parent_label, index, "arg")
            edges.append(edge)
            arg_positions[index] = parent_label
            edge_uses.append((parent_label, index, "arg"))
            input_captures.append(
                TFInputCapture(
                    input_index=index,
                    producer_label_raw=None if is_source_input else parent_label,
                    source_kind="input" if is_source_input else None,
                    source_label_raw=parent_label if is_source_input else None,
                    tensor=None,
                    ref_key=input_tensor.name,
                )
            )
        for control_index, control_op in enumerate(op.control_inputs):
            parent_label = next(
                (
                    self.producer_by_tensor_name[output.name]
                    for output in control_op.outputs
                    if output.name in self.producer_by_tensor_name
                ),
                None,
            )
            if parent_label is None:
                continue
            edge = ParentEdge(parent_label, f"control:{control_index}", "control")
            edges.append(edge)
            edge_uses.append((parent_label, f"control:{control_index}", "control"))
        return (
            tuple(edges),
            {"args": arg_positions, "kwargs": {}},
            tuple(edge_uses),
            tuple(input_captures),
        )


def _capture_opaque_graph_region(
    *,
    tf: Any,
    concrete: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    reason: str,
    error: str | None = None,
) -> TFStaticCaptureResult:
    """Capture an unprunable static graph as one unverified region.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    concrete
        ConcreteFunction to execute as a black box.
    args
        Bound positional inputs.
    kwargs
        Bound keyword inputs.
    reason
        Stable unverified reason.
    error
        Optional private diagnostic detail.

    Returns
    -------
    TFStaticCaptureResult
        Single-region capture with no interior payloads.
    """

    events = CaptureEvents()
    output = concrete(*args, **kwargs)
    flat_outputs = tuple(tf.nest.flatten(output))
    label = events.reserve_label("tf_region")
    annotations: dict[str, object] = {
        REGION_REPLAY_CLASS_KEY: REGION_REPLAY_CLASS,
        REGION_REPLAY_PROVENANCE_KEY: REGION_REPLAY_IMPORTER_PROVENANCE,
        "tf_capture_kind": "region",
        "tf_region_op_type": "FuncGraph",
        "tf_region_replay_status": "unverified",
        "tf_region_reason": reason,
        "tf_region_error": error,
    }
    event = _base_event(
        label=label,
        kind="op",
        layer_type="tf_region",
        func_name="region:FuncGraph",
        op_name=getattr(concrete, "name", "FuncGraph"),
        output=TensorRef(
            label_raw=label.label_raw,
            shape=None,
            dtype=None,
            device=None,
            requires_grad=None,
            memory=None,
            payload=None,
            blob_ref=None,
            backend_handle_id=None,
        ),
        output_index=0,
        parents=(),
        parent_positions={"args": {}, "kwargs": {}},
        edge_uses=(),
        modules=(),
        module_stack=(),
        save_payload=False,
        annotations=annotations,
        equivalence_class="tf:region:FuncGraph",
        is_output_parent=bool(flat_outputs),
        predicate_matched=False,
        record_context=None,
    )
    events.append(event)
    return TFStaticCaptureResult(
        output=output,
        events=events,
        source_records=(),
        unresolved_producers=(),
        init_op_labels=(),
        op_type_counts={"FuncGraph": 1},
        op_captures=(),
        output_label_raws=(label.label_raw,),
        region_captures=(label.label_raw,),
        fallback_error=error,
    )


def _base_event(
    *,
    label: Any,
    kind: str,
    layer_type: str,
    func_name: str,
    op_name: str | None,
    output: TensorRef,
    output_index: int,
    parents: tuple[ParentEdge, ...],
    parent_positions: dict[str, dict[Any, str]],
    edge_uses: tuple[object, ...],
    modules: tuple[tuple[str, int], ...],
    module_stack: tuple[ModuleFrame, ...],
    save_payload: bool,
    annotations: dict[str, object],
    equivalence_class: str,
    is_output_parent: bool,
    predicate_matched: bool,
    record_context: RecordContext | None,
) -> OpEvent:
    """Build a TorchLens event for static TensorFlow capture.

    Parameters
    ----------
    label
        Reserved raw label.
    kind
        Event kind.
    layer_type
        TorchLens layer type.
    func_name
        TensorFlow operation/function name.
    op_name
        TensorFlow graph operation name.
    output
        Output tensor metadata.
    output_index
        Output index within the TensorFlow op.
    parents
        Parent edges.
    parent_positions
        Parent position mapping.
    edge_uses
        Edge-use records.
    modules
        Module call tuples.
    module_stack
        Module frames.
    save_payload
        Whether output payload is saved.
    annotations
        Static capture annotations.
    equivalence_class
        Equivalence key.
    is_output_parent
        Whether the op is a trace output parent.
    predicate_matched
        Whether selective save matched this output.
    record_context
        Selector context.

    Returns
    -------
    OpEvent
        Event ready for materialization.
    """

    return OpEvent(
        kind=kind,
        label_raw=label.label_raw,
        layer_label_raw=label.label_raw,
        layer_type=layer_type,
        raw_index=label.raw_index,
        type_index=label.type_index,
        step_index=label.raw_index,
        source_trace=None,
        source_trace_id=None,
        tracing_finished=False,
        construction_done=True,
        function=FunctionCallRef(
            func=None,
            func_name=func_name,
            func_qualname=func_name,
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
            tensor=output,
            transformed_tensor=None,
            has_saved_activation=save_payload,
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
        module_stack=module_stack,
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
        predicate_matched=predicate_matched,
        pass_index=1,
        grad_fn_class_qualname=None,
        grad_fn_handle=None,
        equivalence_class=equivalence_class,
        is_transform=False,
        transform_kind=None,
        transform_chain=(),
        transform_config={"_tl_annotations": annotations},
        transform_fn_name=None,
        transform_fn_qualname=None,
        transform_fn_source=None,
        is_output_parent=is_output_parent,
        has_internal_source_ancestor=False,
        internal_source_ancestors=frozenset(),
        input_ancestors=frozenset(),
        root_ancestors=frozenset(),
        func_call_id=label.raw_index,
        is_bottom_level=True,
        is_scalar_bool=output.shape == () and output.dtype is not None and "bool" in output.dtype,
        bool_value=_bool_value(output.payload),
        intervention_fired=False,
        intervention_replaced=False,
        fire_results=(),
        intervention_template_ref=None,
        record_context=record_context,
        capture_spec=CaptureSpec(save_out=save_payload, save_metadata=True),
    )


def _record_context_for_symbolic(
    *,
    label_raw: str,
    raw_index: int,
    type_index: int,
    op_type: str,
    op_name: str,
    output: Any,
    output_index: int,
    parents: tuple[ParentEdge, ...],
    modules: tuple[ModuleFrame, ...],
) -> RecordContext:
    """Build a static selector context for a graph tensor.

    Parameters
    ----------
    label_raw
        Raw label.
    raw_index
        Raw capture index.
    type_index
        Per-type index.
    op_type
        TensorFlow op type.
    op_name
        TensorFlow op name.
    output
        Symbolic output tensor.
    output_index
        Output index.
    parents
        Parent edges.
    modules
        Module frames.

    Returns
    -------
    RecordContext
        Selector context.
    """

    module_frames = tuple(
        {
            "address": frame.address,
            "module_type": frame.module_type,
            "pass_index": frame.call_index,
        }
        for frame in modules
    )
    return RecordContext(
        kind="op",
        label=label_raw,
        raw_label=label_raw,
        pass_index=1,
        event_index=raw_index,
        step_index=raw_index,
        layer_type=op_type.lower(),
        type_index=type_index,
        raw_index=raw_index,
        func_name=op_type,
        address=modules[-1].address if modules else None,
        module_type=modules[-1].module_type if modules else None,
        module_pass_index=modules[-1].call_index if modules else None,
        module_stack=module_frames,
        recent_events=(),
        recent_ops=(),
        parent_labels=tuple(parent.parent_label_raw for parent in parents),
        input_output_address=None,
        shape=_shape_tuple(output),
        dtype=DtypeRef(backend="tf", name=str(getattr(output, "dtype", ""))),
        tensor_device=DeviceRef(backend="tf", name=""),
        tensor_requires_grad=None,
        output_index=output_index,
        is_bottom_level_func=True,
        time_since_pass_start=0.0,
        sample_id=None,
        label_raw=label_raw,
        label_prefix=op_type.lower(),
        func_call_id=raw_index,
        parent_labels_raw=tuple(parent.parent_label_raw for parent in parents),
        is_output_parent=False,
        backend_requires_isolation=False,
        is_scalar_bool=_shape_tuple(output) == () and "bool" in str(getattr(output, "dtype", "")),
        bool_value=None,
    )


def _module_frames_from_name(op_name: str) -> tuple[ModuleFrame, ...]:
    """Derive best-effort module frames from TensorFlow name scopes.

    Parameters
    ----------
    op_name
        TensorFlow graph operation name.

    Returns
    -------
    tuple[ModuleFrame, ...]
        Name-scope-derived module frames.
    """

    del op_name
    return ()


def _tensor_ref_for_symbolic(output: Any, label_raw: str, payload: object | None) -> TensorRef:
    """Build a TensorRef for a symbolic TensorFlow tensor.

    Parameters
    ----------
    output
        Symbolic graph tensor, or ``None`` for opaque fallback regions.
    label_raw
        Raw TorchLens label.
    payload
        Optional fetched payload.

    Returns
    -------
    TensorRef
        Backend-neutral tensor metadata.
    """

    return TensorRef(
        label_raw=label_raw,
        shape=_shape_tuple(output),
        dtype=str(getattr(output, "dtype", "")) if output is not None else None,
        device=str(getattr(output, "device", "")) if output is not None else None,
        requires_grad=None,
        memory=_nbytes_from_shape_dtype(output, payload),
        payload=payload,
        blob_ref=None,
        backend_handle_id=str(getattr(output, "name", "")) if output is not None else None,
    )


def _node_attrs(op: Any) -> dict[str, Any]:
    """Return raw NodeDef attributes for a TensorFlow operation.

    Parameters
    ----------
    op
        TensorFlow graph operation.

    Returns
    -------
    dict[str, Any]
        Attribute values keyed by name.
    """

    attrs: dict[str, Any] = {}
    for name in getattr(op.node_def, "attr", {}):
        try:
            attrs[str(name)] = op.get_attr(name)
        except (TypeError, ValueError):
            attrs[str(name)] = repr(op.node_def.attr[name])
    return attrs


def _snapshot_value(value: Any) -> object | None:
    """Snapshot a TensorFlow or NumPy value within the static payload cap.

    Parameters
    ----------
    value
        Runtime value.

    Returns
    -------
    object | None
        NumPy payload, or ``None`` when unsupported or too large.
    """

    if value is None:
        return None
    try:
        array = value.numpy()
    except AttributeError:
        array = np.asarray(value)
    except Exception:
        return None
    if array.dtype.kind in {"O", "U", "S"}:
        return None
    if int(array.nbytes) > _MAX_SNAPSHOT_BYTES:
        return None
    return array


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    """Return a static shape tuple when available.

    Parameters
    ----------
    value
        Tensor-like value.

    Returns
    -------
    tuple[int, ...] | None
        Shape tuple or ``None``.
    """

    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(-1 if item is None else int(item) for item in shape)
    except TypeError:
        return None


def _nbytes_from_shape_dtype(value: Any, payload: object | None) -> int | None:
    """Return tensor memory estimate from payload or symbolic metadata.

    Parameters
    ----------
    value
        Tensor-like value.
    payload
        Optional concrete payload.

    Returns
    -------
    int | None
        Memory estimate in bytes.
    """

    if hasattr(payload, "nbytes"):
        return int(payload.nbytes)
    shape = _shape_tuple(value)
    dtype = getattr(value, "dtype", None)
    size = getattr(dtype, "size", None)
    if shape is None or size is None or any(dim < 0 for dim in shape):
        return None
    return int(np.prod(shape, dtype=np.int64)) * int(size)


def _bool_value(payload: object | None) -> bool | None:
    """Return scalar bool payloads as Python bools.

    Parameters
    ----------
    payload
        Optional saved payload.

    Returns
    -------
    bool | None
        Boolean value when scalar, else ``None``.
    """

    if payload is None:
        return None
    array = np.asarray(payload)
    if array.shape == () and array.dtype == np.bool_:
        return bool(array)
    return None
