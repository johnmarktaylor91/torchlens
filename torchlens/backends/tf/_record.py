"""The TensorFlow ``tl.record`` producer: lean per-op events, values only when selected.

The producer satisfies :class:`torchlens.recording._backend_events.BackendRecordProducer`.
Each pass runs the same eager capture session ``tl.trace(..., backend="tf")``
uses, so every event's raw label, type index and raw index are the ones
``tl.trace`` gives that op (same reservation order: inputs, then constants and
op outputs in callback order). The session snapshots no payload and releases
each op's live tensors once its edges are recorded (``_retention``), so an
intermediate is freed when the model drops it. A value reaches the sink only
when the sink requests it. When a decision says stop, the hook unwinds the
forward, so no op after the stop frontier runs.

No ``Trace`` is built: the producer never finalizes, labels or validates a
graph. Graph-only callables (a ``tf.function`` without an eager body) are
refused, because their ops do not run one at a time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from ...errors import BackendUnsupportedError
from ...recording._backend_events import BackendOpEvent, RecordSink
from ...recording._types import ModuleStackFrame
from .backend import TFBackend, _ensure_built_or_warmable, _resolve_tf_module_identity_mode
from .modules import discover_tf_module_tree
from .op_callback_capture import TFEagerCaptureSession, warm_up_tf_callable


class _StopRecording(Exception):
    """Raised inside the op callback to unwind the forward at a stop."""


class TFPayloadCodec:
    """Payload conversions for selected TensorFlow values."""

    def ram_copy(self, value: Any) -> Any:
        """Return an independent NumPy copy, the type ``tl.trace`` payloads have."""

        import numpy as np

        return np.array(value.numpy() if hasattr(value, "numpy") else value, copy=True)

    def to_disk_tensor(self, value: Any) -> torch.Tensor:
        """Return a CPU ``torch.Tensor`` holding the same numbers as ``value``.

        Parameters
        ----------
        value
            One TensorFlow tensor.

        Returns
        -------
        torch.Tensor
            CPU tensor with the matching dtype (bfloat16 round-trips through float32).
        """

        import numpy as np

        host = self.ram_copy(value)
        if str(host.dtype) == "bfloat16":
            return torch.from_numpy(host.astype(np.float32)).to(torch.bfloat16)
        return torch.from_numpy(host)


@dataclass
class _Prepared:
    """A model discovered once per recorder (the module tree needs a built model)."""

    model: Any
    warmed: bool = False
    module_tree: Any = None
    module_ids: dict[str, int] = field(default_factory=dict)


class TFRecordProducer:
    """``tl.record`` producer for eager TensorFlow callables and Keras models."""

    backend = "tf"
    supported_options: frozenset[str] = frozenset()

    def __init__(self) -> None:
        """Set up the codec and the per-recorder counters tests read."""

        self.codec = TFPayloadCodec()
        self.passes = 0
        self.ops_run = 0

    def prepare(self, model: Any) -> _Prepared:
        """Hold ``model``; discovery waits for the first inputs (Keras builds lazily)."""

        return _Prepared(model=model)

    def run_pass(
        self,
        prepared: _Prepared,
        input_args: Any,
        input_kwargs: Mapping[str, Any],
        sink: RecordSink,
    ) -> Any:
        """Run one eager forward, emitting events; return the output, or ``None`` after a stop."""

        self.passes += 1
        tf, plan = _eager_plan(prepared, input_args, input_kwargs)
        session = TFEagerCaptureSession(
            tf=tf,
            callable_obj=plan.callable_obj,
            args=plan.args,
            kwargs=plan.call_kwargs,
            module_tree=prepared.module_tree,
            save_payloads=False,
            save_predicate=None,
        )
        session.retain_live_tensors = False
        emitter = _EventEmitter(sink, session, prepared.module_ids, self)
        session.output_hook = emitter.on_outputs
        try:
            result = session.run()
        except _StopRecording:
            return None
        emitter.emit_sources()
        return result.output


def _eager_plan(prepared: _Prepared, input_args: Any, input_kwargs: Mapping[str, Any]) -> Any:
    """Return TensorFlow and the eager call plan; warm up and discover modules once."""

    backend = TFBackend()
    plan = backend.normalize_call(
        model=prepared.model, input_args=input_args, input_kwargs=dict(input_kwargs) or None
    )
    if plan.mode == "graph_only":
        raise BackendUnsupportedError(
            f"tl.record on the tf backend needs an eager-executable callable ({plan.reason}); "
            "a graph-only tf.function runs its ops as one graph, not one op at a time."
        )
    tf = backend._import_tensorflow()
    if not prepared.warmed:
        tree = discover_tf_module_tree(prepared.model, tf)
        use_object_module = _resolve_tf_module_identity_mode(None, tree)
        _ensure_built_or_warmable(prepared.model)
        warm_up_tf_callable(plan.callable_obj, plan.args, plan.call_kwargs)
        tree = discover_tf_module_tree(prepared.model, tf) if use_object_module else None
        prepared.module_tree = tree
        address_by_id = getattr(tree, "address_by_id", {}) or {}
        prepared.module_ids = {address: ident for ident, address in address_by_id.items()}
        prepared.warmed = True
    return tf, plan


class _EventEmitter:
    """Output hook that turns the session's records into backend events."""

    def __init__(
        self,
        sink: RecordSink,
        session: Any,
        module_ids: Mapping[str, int],
        producer: TFRecordProducer,
    ) -> None:
        """Bind the sink, the running session and the module id lookup."""

        self.sink = sink
        self.session = session
        self.module_ids = module_ids
        self.producer = producer
        self.sources_seen = 0

    def emit_sources(self) -> None:
        """Emit model inputs registered since the last call (constants have no event kind)."""

        records = self.session.source_records
        for record in records[self.sources_seen :]:
            if record.kind == "input" and record.label_raw is not None:
                journal = _journal_event(self.session.events, record.label_raw)
                self._emit(
                    BackendOpEvent(
                        raw_label=record.label_raw,
                        layer_type=journal.layer_type,
                        type_index=journal.type_index,
                        raw_index=journal.raw_index,
                        func_name="input",
                        shape=_shape(record.tensor),
                        dtype=str(getattr(record.tensor, "dtype", "")),
                        device=getattr(record.tensor, "device", None),
                        func_call_id=journal.raw_index,
                        kind="input",
                    ),
                    record.tensor,
                )
        self.sources_seen = len(records)

    def on_outputs(self, recorded: list[tuple[str, Any, Any]]) -> None:
        """Emit one event per output of one op callback, after any new inputs."""

        self.emit_sources()
        self.producer.ops_run += 1
        for _label_raw, ctx, output in recorded:
            self._emit(self._op_event(ctx, output), output)

    def _op_event(self, ctx: Any, output: Any) -> BackendOpEvent:
        """Build the metadata-only event from the session's record context."""

        frames = tuple(
            ModuleStackFrame(
                address=str(frame["address"]),
                module_type=str(frame["module_type"]),
                module_id=int(self.module_ids.get(str(frame["address"]), 0)),
                pass_number=int(frame["pass_number"]),
            )
            for frame in ctx.module_stack
        )
        return BackendOpEvent(
            raw_label=ctx.raw_label,
            layer_type=ctx.layer_type,
            type_index=ctx.type_index,
            raw_index=ctx.raw_index,
            parent_labels=tuple(ctx.parent_labels_raw),
            func_name=ctx.func_name,
            shape=ctx.shape,
            dtype=str(getattr(output, "dtype", "")),
            device=getattr(output, "device", None),
            output_index=ctx.output_index,
            module_stack=frames,
            func_call_id=ctx.func_call_id,
        )

    def _emit(self, event: BackendOpEvent, value: Any) -> None:
        """Emit one event, deliver its value when requested, unwind when told to stop."""

        decision = self.sink.emit(event)
        if decision.request is not None:
            self.sink.deliver(decision.request, value)
        if decision.stop:
            raise _StopRecording


def _journal_event(events: Any, label_raw: str) -> Any:
    """The session's source ``OpEvent`` carrying ``label_raw``."""

    for event in reversed(events.op_events):
        if getattr(event, "label_raw", None) == label_raw:
            return event
    raise LookupError(f"tf record: no capture event for source {label_raw!r}")


def _shape(value: Any) -> tuple[int, ...] | None:
    """Static shape of ``value`` as ints, or ``None`` when unknown."""

    shape = getattr(value, "shape", None)
    try:
        return None if shape is None else tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def tf_record_producer() -> TFRecordProducer:
    """Return a fresh TensorFlow record producer (the ``fastlog`` capability binding)."""

    return TFRecordProducer()


__all__ = ("TFPayloadCodec", "TFRecordProducer", "tf_record_producer")
