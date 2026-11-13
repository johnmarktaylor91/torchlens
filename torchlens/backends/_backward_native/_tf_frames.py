"""Gradient frames of one TensorFlow backward pass and their forward ops.

TensorFlow's eager tape calls one Python gradient rule per recorded forward op
(``backprop._gradient_function`` for registered ops, the ``backward_function``
given to ``record.record_operation`` for ``tf.custom_gradient``). The native
TensorFlow backward module wraps those entry points, so each executed rule is
one :class:`Frame`: the forward tensors the tape kept, the incoming gradient
per forward output (``out_grads``) and the outgoing gradient per forward input
(the rule's results). The raw backward ops the op callback sees while a frame
runs belong to it; the ones outside every frame are the tape's own work
(accumulating gradients that reach one tensor from several consumers).

This module holds no framework code: tensors are read only through their
``_id`` (TensorFlow's per-tensor identity), so the bookkeeping and the link
from each frame to its forward op are testable without TensorFlow.

The link follows the order the prototype measured (proto-bwd RESULT.md,
"Recommended recording mechanism per framework"): the forward output tensors
the tape kept, then the kept inputs with the op type, then backward data flow.
``forward_pass_name_scope`` is not used: it is ``None`` for fast-path ops.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = (
    "BackwardRecorder",
    "ForwardCall",
    "Frame",
    "RawEvent",
    "flowed_ids",
    "frame_consumers",
    "link_frames",
    "tensor_id",
)


def tensor_id(value: Any) -> int | None:
    """TensorFlow's identity of an eager tensor (``_id``), else ``None``."""

    if value is None:
        return None
    ident = getattr(value, "_id", None)
    return ident if isinstance(ident, int) else None


def _as_list(value: Any) -> list[Any]:
    """A rule's result as a list: a lone tensor becomes one item.

    Iterating a lone TensorFlow tensor would slice it (and run ``StridedSlice``
    ops inside the frame), so only real lists and tuples are unpacked.
    """

    return list(value) if isinstance(value, (list, tuple)) else [value]


@dataclass(frozen=True)
class ForwardCall:
    """One captured forward op call: its op type and the tensors of each slot.

    Attributes
    ----------
    index:
        Position among the captured calls, in execution order.
    op_type:
        TensorFlow op type.
    input_ids, output_ids:
        Tensor identity per input and output slot (``None`` when unknown).
    captures:
        The per-output capture records, by output slot.
    """

    index: int
    op_type: str
    input_ids: tuple[int | None, ...]
    output_ids: tuple[int | None, ...]
    captures: tuple[Any, ...]


@dataclass
class Frame:
    """One executed gradient rule.

    Attributes
    ----------
    index:
        Position in the pass, in call order.
    kind:
        ``"gradient_function"`` (a registered rule) or ``"record_operation"``
        (a Python backward function, as ``tf.custom_gradient`` records).
    op_type:
        TensorFlow's name for the rule: the forward op type, or the name the
        Python rule was recorded under.
    input_ids, output_ids:
        Identity of each forward input and output the tape kept (``None``
        where the rule needs no value and the tape kept none).
    out_grads:
        Incoming gradient per forward output, as the tape passed it.
    results:
        Outgoing gradient per forward input, as the rule returned it.
    """

    index: int
    kind: str
    op_type: str
    input_ids: tuple[int | None, ...]
    output_ids: tuple[int | None, ...]
    out_grads: tuple[Any, ...]
    results: tuple[Any, ...] = ()
    timestamp: float = 0.0
    started: float | None = None
    finished: float | None = None

    @property
    def out_grad_ids(self) -> tuple[int | None, ...]:
        """Identity of each incoming gradient."""

        return tuple(tensor_id(item) for item in self.out_grads)

    @property
    def result_ids(self) -> tuple[int | None, ...]:
        """Identity of each outgoing gradient."""

        return tuple(tensor_id(item) for item in self.results)


@dataclass(frozen=True)
class RawEvent:
    """One raw op the callback saw during the backward pass."""

    op_type: str
    input_ids: tuple[int | None, ...]
    output_ids: tuple[int | None, ...]
    frame: int | None


@dataclass
class BackwardRecorder:
    """Collects the frames and raw ops of one backward pass on its own thread.

    The gradient-rule registration is process-wide while the pass runs; frames
    and raw ops from any other thread pass through unrecorded.
    """

    frames: list[Frame] = field(default_factory=list)
    events: list[RawEvent] = field(default_factory=list)
    stack: list[int] = field(default_factory=list)
    owner: int = field(default_factory=threading.get_ident)

    def owns_thread(self) -> bool:
        """Whether the calling thread is the one running this pass."""

        return threading.get_ident() == self.owner

    def callback(
        self,
        op_type: Any,
        inputs: Sequence[Any] | None,
        attrs: Any,
        outputs: Sequence[Any] | None,
        op_name: str | None = None,
        graph: Any | None = None,
    ) -> None:
        """Op callback: note one eager raw op and the frame it ran in."""

        del attrs, op_name
        if graph is not None or not self.owns_thread():
            return None
        name = op_type.decode() if isinstance(op_type, bytes) else str(op_type)
        self.events.append(
            RawEvent(
                op_type=name,
                input_ids=tuple(tensor_id(item) for item in inputs or ()),
                output_ids=tuple(tensor_id(item) for item in outputs or ()),
                frame=self.stack[-1] if self.stack else None,
            )
        )
        return None

    def run_frame(
        self,
        kind: str,
        op_type: Any,
        inputs: Iterable[Any] | None,
        outputs: Iterable[Any] | None,
        out_grads: Iterable[Any],
        call: Callable[[], Any],
    ) -> Any:
        """Run one gradient rule as a frame and return its result unchanged."""

        name = op_type.decode() if isinstance(op_type, bytes) else str(op_type)
        frame = Frame(
            index=len(self.frames),
            kind=kind,
            op_type=name,
            input_ids=tuple(tensor_id(item) for item in inputs or ()),
            output_ids=tuple(tensor_id(item) for item in outputs or ()),
            out_grads=tuple(out_grads),
            timestamp=time.time(),
        )
        self.frames.append(frame)
        self.stack.append(frame.index)
        frame.started = time.monotonic()
        try:
            result = call()
        finally:
            frame.finished = time.monotonic()
            self.stack.pop()
        frame.results = tuple(_as_list(result))
        return result


def _accumulations(events: Sequence[RawEvent]) -> dict[int, tuple[int, ...]]:
    """Inputs of each tensor a raw op outside every frame produced (the tape's sums)."""

    produced: dict[int, tuple[int, ...]] = {}
    for event in events:
        if event.frame is not None:
            continue
        inputs = tuple(item for item in event.input_ids if item is not None)
        for output in event.output_ids:
            if output is not None:
                produced[output] = inputs
    return produced


def _reaching(start: int, accumulations: dict[int, tuple[int, ...]]) -> set[int]:
    """Tensors that reach ``start`` directly or through the tape's sums."""

    seen = {start}
    pending = [start]
    while pending:
        for item in accumulations.get(pending.pop(), ()):
            if item not in seen:
                seen.add(item)
                pending.append(item)
    return seen


def flowed_ids(
    frames: Sequence[Frame], events: Sequence[RawEvent], seed_ids: Iterable[int]
) -> set[int]:
    """Tensors that carry a gradient that flowed in this pass.

    The seed, every rule's result, and every sum the tape made of those. A
    zero gradient the tape filled in for an output nothing consumed is not
    among them, so it is never recorded as that output's gradient.
    """

    flowed = set(seed_ids)
    for frame in frames:
        flowed.update(item for item in frame.result_ids if item is not None)
    for event in events:
        if event.frame is not None:
            continue
        inputs = [item for item in event.input_ids if item is not None]
        if inputs and all(item in flowed for item in inputs):
            flowed.update(item for item in event.output_ids if item is not None)
    return flowed


def frame_consumers(frames: Sequence[Frame], events: Sequence[RawEvent]) -> dict[int, int]:
    """Map each tensor that feeds a frame's incoming gradient to that frame."""

    accumulations = _accumulations(events)
    consumer: dict[int, int] = {}
    for frame in frames:
        for grad_id in frame.out_grad_ids:
            if grad_id is None:
                continue
            for item in _reaching(grad_id, accumulations):
                consumer.setdefault(item, frame.index)
    return consumer


def _link_by_outputs(calls: Sequence[ForwardCall], frames: Sequence[Frame]) -> list[int | None]:
    """Step 1: a frame whose kept outputs all belong to one forward call."""

    owner = {
        tid: call.index for call in calls for tid in call.output_ids if tid is not None
    }  # fmt: skip
    links: list[int | None] = []
    for frame in frames:
        owners = {owner[tid] for tid in frame.output_ids if tid in owner}
        links.append(owners.pop() if len(owners) == 1 else None)
    return links


def _link_by_inputs(
    calls: Sequence[ForwardCall], frames: Sequence[Frame], links: list[int | None]
) -> None:
    """Step 2: a frame whose kept inputs and op type match exactly one unlinked call."""

    for frame in frames:
        if links[frame.index] is not None or not frame.input_ids:
            continue
        if any(item is None for item in frame.input_ids):
            continue
        used = {item for item in links if item is not None}
        candidates = [
            call.index
            for call in calls
            if call.index not in used
            and call.op_type == frame.op_type
            and call.input_ids == frame.input_ids
        ]
        if len(candidates) == 1:
            links[frame.index] = candidates[0]


def _dataflow_candidates(
    frames: Sequence[Frame],
    links: list[int | None],
    call: ForwardCall,
    slot: int,
    result_id: int,
    accumulations: dict[int, tuple[int, ...]],
) -> list[int]:
    """Unlinked frames of ``call``'s op type whose incoming ``slot`` gradient holds ``result_id``."""

    found = []
    for frame in frames:
        if links[frame.index] is not None or frame.op_type != call.op_type:
            continue
        grad_ids = frame.out_grad_ids
        if slot < len(grad_ids) and grad_ids[slot] is not None:
            if result_id in _reaching(grad_ids[slot], accumulations):
                found.append(frame.index)
    return found


def _gradient_pairs(
    calls: Sequence[ForwardCall],
    frames: Sequence[Frame],
    links: list[int | None],
    seeds: Mapping[int, int],
) -> list[tuple[int, int]]:
    """``(forward tensor, gradient sent to it)`` from the frames and the seeds.

    A linked frame's inputs are its call's; an unlinked frame (an op outside
    the trace, such as a loss computed after the forward) names only the
    inputs the tape kept for it.
    """

    pairs = list(seeds.items())
    for frame in frames:
        linked = links[frame.index]
        input_ids = frame.input_ids if linked is None else calls[linked].input_ids
        result_ids = frame.result_ids
        for slot, input_id in enumerate(input_ids):
            if input_id is not None and slot < len(result_ids) and result_ids[slot] is not None:
                pairs.append((input_id, result_ids[slot]))
    return pairs


def _link_by_dataflow_once(
    calls: Sequence[ForwardCall],
    frames: Sequence[Frame],
    links: list[int | None],
    accumulations: dict[int, tuple[int, ...]],
    seeds: Mapping[int, int],
) -> bool:
    """One sweep of step 3; returns whether a new link was made."""

    producer = {
        tid: (call.index, slot)
        for call in calls
        for slot, tid in enumerate(call.output_ids)
        if tid is not None
    }
    changed = False
    for forward_id, grad_id in _gradient_pairs(calls, frames, links, seeds):
        if forward_id not in producer:
            continue
        parent, out_slot = producer[forward_id]
        if parent in links:
            continue
        found = _dataflow_candidates(frames, links, calls[parent], out_slot, grad_id, accumulations)
        if len(found) == 1:
            links[found[0]] = parent
            changed = True
    return changed


def link_frames(
    calls: Sequence[ForwardCall],
    frames: Sequence[Frame],
    events: Sequence[RawEvent],
    seeds: Mapping[int, int] | None = None,
) -> list[int | None]:
    """Name the forward call each frame differentiates (``None`` when no call matches).

    Step 3 (data flow): a gradient sent to a forward tensor (a frame's result
    for its op's input slot ``i``, or a pass seed, keyed by the tensor it
    seeds in ``seeds``) reaches, directly or through the tape's sums,
    incoming gradient ``k`` of the frame of the op that produced that tensor
    as its output ``k``. A link is made only when exactly one frame qualifies,
    and no call gets two frames.
    """

    links = _link_by_outputs(calls, frames)
    _link_by_inputs(calls, frames, links)
    accumulations = _accumulations(events)
    while _link_by_dataflow_once(calls, frames, links, accumulations, dict(seeds or {})):
        pass
    return links
