"""Frame bookkeeping and frame-to-op linking for TensorFlow backward capture.

Pure Python: tensors are stand-ins carrying TensorFlow's ``_id``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import pytest

from torchlens.backends._backward_native._tf_frames import (
    BackwardRecorder,
    ForwardCall,
    flowed_ids,
    frame_consumers,
    link_frames,
)

pytestmark: list[pytest.MarkDecorator] = []


@dataclass(frozen=True)
class T:
    """A tensor stand-in."""

    _id: int


def _call(index: int, op_type: str, inputs: tuple[int, ...], outputs: tuple[int, ...]) -> Any:
    return ForwardCall(index, op_type, inputs, outputs, captures=(object(),) * len(outputs))


def _run(rec: BackwardRecorder, op_type: str, ins: list[int], outs: list[int], grads: list[int],
         results: Any, raw: list[tuple[str, list[int], list[int]]] = ()) -> None:  # fmt: skip
    def body() -> Any:
        for name, i, o in raw:
            rec.callback(name, [T(n) for n in i], None, [T(n) for n in o])
        return results

    tensors = ([T(n) for n in ins], [T(n) for n in outs], [T(n) for n in grads])
    rec.run_frame("gradient_function", op_type, tensors, body)


def test_lone_tensor_result_is_one_item_and_returned_unchanged() -> None:
    rec = BackwardRecorder()
    lone = T(9)
    out = rec.run_frame("gradient_function", b"Relu", ([T(1)], [T(2)], [T(3)]), lambda: lone)
    assert out is lone
    assert rec.frames[0].op_type == "Relu"
    assert rec.frames[0].result_ids == (9,)
    assert rec.frames[0].finished >= rec.frames[0].started


def test_raw_ops_belong_to_the_running_frame_and_other_threads_are_ignored() -> None:
    rec = BackwardRecorder()
    _run(rec, "Tanh", [1], [2], [10], [T(11)], raw=[("TanhGrad", [2, 10], [11])])
    rec.callback("AddN", [T(11), T(12)], None, [T(13)])
    worker = threading.Thread(target=rec.callback, args=("Mul", [T(1)], None, [T(99)]))
    worker.start()
    worker.join()
    rec.callback("Mul", [T(1)], None, [T(98)], graph=object())
    assert [(e.op_type, e.frame) for e in rec.events] == [("TanhGrad", 0), ("AddN", None)]


def test_link_by_outputs_inputs_and_dataflow() -> None:
    # forward: a=Tanh(x:1)->2 ; b=Mul(a,a)->3 ; c=Sin(a)->4 ; d=AddV2(b,c)->5
    calls = [
        _call(0, "Tanh", (1,), (2,)),
        _call(1, "Mul", (2, 2), (3,)),
        _call(2, "Sin", (2,), (4,)),
        _call(3, "AddV2", (3, 4), (5,)),
    ]
    rec = BackwardRecorder()
    # AddV2 keeps no tensors; Sin keeps its input; Mul keeps inputs; Tanh keeps output.
    _run(rec, "AddV2", [], [], [20], [T(21), T(22)])
    _run(rec, "Sin", [2], [], [22], [T(23)])
    _run(rec, "Mul", [2, 2], [], [21], [T(24), T(25)])
    rec.callback("AddN", [T(23), T(24), T(25)], None, [T(26)])
    _run(rec, "Tanh", [], [2], [26], [T(27)])
    assert link_frames(calls, rec.frames, rec.events) == [None, 2, 1, 0]
    links = link_frames(calls, rec.frames, rec.events, {5: 20})
    assert links == [3, 2, 1, 0]
    consumers = frame_consumers(rec.frames, rec.events)
    assert consumers[23] == consumers[24] == consumers[25] == 3
    assert flowed_ids(rec.frames, rec.events, {20}) >= {20, 21, 22, 23, 24, 25, 26, 27}


def test_a_loss_op_outside_the_trace_links_the_last_forward_op() -> None:
    # forward: y=BiasAdd(x:1, b:2)->3 ; loss outside the trace: l=Mul(y, y)->4
    calls = [_call(0, "BiasAdd", (1, 2), (3,))]
    rec = BackwardRecorder()
    _run(rec, "Mul", [3, 3], [], [20], [T(21), T(22)])
    rec.callback("AddN", [T(21), T(22)], None, [T(23)])
    _run(rec, "BiasAdd", [], [], [23], [T(24), T(25)])
    assert link_frames(calls, rec.frames, rec.events, {4: 20}) == [None, 0]


def test_ambiguous_frames_stay_unlinked() -> None:
    calls = [_call(0, "Relu", (1,), (2,)), _call(1, "Relu", (1,), (3,))]
    rec = BackwardRecorder()
    _run(rec, "Relu", [], [], [10], [T(11)])
    _run(rec, "Relu", [], [], [12], [T(13)])
    assert link_frames(calls, rec.frames, rec.events) == [None, None]


def test_unconsumed_zero_fill_is_not_flowed() -> None:
    rec = BackwardRecorder()
    rec.callback("ZerosLike", [T(5)], None, [T(6)])
    _run(rec, "Split", [], [], [20, 6], [T(21)])
    assert 6 not in flowed_ids(rec.frames, rec.events, {20})
