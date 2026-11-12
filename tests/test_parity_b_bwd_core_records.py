"""The neutral backward record builder reproduces torch's records field by field.

The builder in ``torchlens/backends/_backward_neutral.py`` folds the same
neutral backward events every backend journals. Fed torch's own event stream
with torch's payload policy, it must rebuild GradFn, GradFnCall, BackwardPass
and the per-op gradient fields exactly as torch's own fold does.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._constants import LAYER_PASS_LOG_FIELD_ORDER
from torchlens._data_classes._op_save import _dtype_or_none, _memory_or_none, _shape_or_none
from torchlens._data_classes.backward_pass import _BACKWARD_PASS_STORED_FIELDS
from torchlens._data_classes.grad_fn import _GRAD_FN_STORED_FIELDS
from torchlens._data_classes.grad_fn_call import _GRAD_FN_CALL_STORED_FIELDS
from torchlens.backends import _backward_neutral as neutral
from torchlens.backends.torch import backward as torch_backward

TORCH_PAYLOADS = neutral.PayloadPolicy(
    is_payload=lambda value: isinstance(value, torch.Tensor),
    dtype_from_string=torch_backward._torch_dtype_from_string,
    shape_of=_shape_or_none,
    dtype_of=_dtype_or_none,
    memory_of=_memory_or_none,
    retained_refs=torch_backward._retained_grad_payload_refs,
)

_OP_GRAD_FIELDS = tuple(name for name in LAYER_PASS_LOG_FIELD_ORDER if "grad" in name)
_TRACE_SLOTS = (
    "_autograd_grad_fn_order",
    "_num_backward_passes",
    "_autograd_backward_root_grad_fn_object_ids",
    "_saved_grad_labels",
    "_cost_saved_grad_memory",
    "_cost_total_grad_memory",
    "_cost_total_backward_memory",
)


class _Chain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _Diamond(nn.Module):
    """One value read by three consumers, so gradients sum over consumers."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        return (y * y + torch.sin(y)).sum(dim=1, keepdim=True)


def _norm(value: Any) -> Any:
    """A comparable form of one stored field value."""

    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), value.detach().cpu().tolist())
    if isinstance(value, weakref.ref):
        return ("weakref",)
    if hasattr(value, "_FIELD_POLICY") or type(value).__name__ in {"Op", "Param", "Module"}:
        label = getattr(value, "label", None) or getattr(value, "layer_label", None)
        return ("record", type(value).__name__, label)
    if hasattr(value, "_dict") and isinstance(value._dict, dict):
        return ("accessor", tuple((key, _norm(item)) for key, item in value._dict.items()))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__,) + tuple(
            (field.name, _norm(getattr(value, field.name))) for field in dataclasses.fields(value)
        )
    if isinstance(value, dict):
        return ("dict", tuple((repr(key), _norm(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_norm(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(_norm(item)) for item in value)))
    return value


def _fields(record: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: _norm(getattr(record, name, "<missing>")) for name in names}


def _snapshot(trace: tl.Trace) -> dict[str, Any]:
    """Every backward record field the builder owns, in comparable form."""

    grad_fns = {
        key: _fields(record, _GRAD_FN_STORED_FIELDS) for key, record in trace._grad_fn_logs.items()
    }
    calls = {
        (key, call_key): _fields(call, _GRAD_FN_CALL_STORED_FIELDS)
        for key, record in trace._grad_fn_logs.items()
        for call_key, call in getattr(record.calls, "_dict", record.calls).items()
    }
    passes = {
        key: _fields(record, _BACKWARD_PASS_STORED_FIELDS)
        for key, record in trace._backward_pass_logs.items()
    }
    ops = {
        op.layer_label: {
            **_fields(op, _OP_GRAD_FIELDS),
            "_grad_records": _norm(list(op._slot("_grad_records") or ())),
        }
        for op in trace._layer_list
    }
    params = {
        address: _norm(list(param._slot("_grad_records") or ()))
        if hasattr(param, "_slot")
        else None
        for address, param in (trace._param_logs or {}).items()
    }
    return {
        "grad_fns": grad_fns,
        "calls": calls,
        "passes": passes,
        "ops": ops,
        "params": params,
        "trace": _fields(trace, _TRACE_SLOTS),
    }


def _assert_same(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Field-by-field equality with a readable first difference."""

    for section in expected:
        assert expected[section].keys() == actual[section].keys(), section
        for key, want in expected[section].items():
            got = actual[section][key]
            if isinstance(want, dict):
                for name in want:
                    assert want[name] == got[name], f"{section}[{key!r}].{name}"
            else:
                assert want == got, f"{section}[{key!r}]"


def _chain_one_pass() -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(
        _Chain(),
        torch.randn(3, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
        save="all",
    )
    trace.backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _diamond_one_pass() -> tl.Trace:
    torch.manual_seed(1)
    trace = tl.trace(
        _Diamond(),
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
        save="all",
    )
    trace.backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _chain_two_passes() -> tl.Trace:
    torch.manual_seed(2)
    trace = tl.trace(
        _Chain(),
        torch.randn(3, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
        save="all",
    )
    loss = trace[trace.output_layers[0]].out.sum()
    trace.backward(loss, retain_graph=True)
    trace.backward(loss * 2.0, retain_graph=True)
    return trace


_CASES: dict[str, Callable[[], tl.Trace]] = {
    "chain": _chain_one_pass,
    "diamond": _diamond_one_pass,
    "two_passes": _chain_two_passes,
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_builder_matches_torch_records_field_by_field(case: str) -> None:
    """Torch's events through the neutral builder give torch's records."""

    trace = _CASES[case]()
    assert len(trace.autograd.grad_fns) > 0  # forces torch's own fold
    expected = _snapshot(trace)
    old_grad_fns = dict(trace._grad_fn_logs)

    events = list(trace._capture_events.backward_events)
    neutral.build_backward_records(trace, events, payloads=TORCH_PAYLOADS)

    assert all(trace._grad_fn_logs[key] is not old_grad_fns[key] for key in old_grad_fns)
    _assert_same(expected, _snapshot(trace))


def test_builder_never_fills_unobserved_timing_or_engine_fields() -> None:
    """A stream with no start or end events leaves timing and engine fields empty."""

    trace = _chain_one_pass()
    assert len(trace.autograd.grad_fns) > 0
    kept = [
        event
        for event in trace._capture_events.backward_events
        if type(event).__name__ not in {"BackwardPassStart", "BackwardPassEnd"}
    ]
    neutral.build_backward_records(trace, kept, payloads=TORCH_PAYLOADS)
    record = trace._backward_pass_logs[1]
    assert record.duration is None
    assert record.peak_memory is None
    assert record.engine_flags in (None, {}, ())
    assert record.trigger == "implicit"
    assert record.implicit is True


def test_materialize_matches_torch_and_is_revision_cached() -> None:
    """The cached materialize entry point gives the same records as torch's fold."""

    trace = _diamond_one_pass()
    assert len(trace.autograd.grad_fns) > 0
    expected = _snapshot(trace)

    trace._backward_projection_revision = None
    neutral.materialize_neutral_backward(trace, payloads=TORCH_PAYLOADS)
    _assert_same(expected, _snapshot(trace))
    assert trace._backward_projection_revision == trace._capture_events.backward_revision

    built = dict(trace._grad_fn_logs)
    neutral.materialize_neutral_backward(trace, payloads=TORCH_PAYLOADS)
    assert all(trace._grad_fn_logs[key] is built[key] for key in built)
