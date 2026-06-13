"""Phase 0 capture-unification schema and emission tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.postprocess as postprocess_mod
from torchlens.ir import CaptureEvents, DeviceRef, DtypeRef


@dataclass(slots=True)
class CapturedEventSnapshot:
    """Immutable references to emitted Phase 0 events before Step 0 clears them."""

    op_events: tuple[Any, ...]
    module_prep_events: tuple[Any, ...]
    module_enter_events: tuple[Any, ...]
    module_exit_events: tuple[Any, ...]
    events: CaptureEvents


def _trace_and_capture_events(
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[Any, CapturedEventSnapshot]:
    """Run ``tl.trace`` and snapshot the event buffer passed to Step 0.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    model
        Model to trace.
    x
        Input tensor.

    Returns
    -------
    tuple[Any, CapturedEventSnapshot]
        Final trace and the pre-materialization event snapshot.
    """

    snapshots: list[CapturedEventSnapshot] = []
    real_materialize = postprocess_mod.materialize_from_events

    def spy_materialize(trace: Any, events: CaptureEvents) -> None:
        """Snapshot events, then delegate to the real materializer."""

        snapshots.append(
            CapturedEventSnapshot(
                op_events=tuple(events.op_events),
                module_prep_events=tuple(events.module_prep_events),
                module_enter_events=tuple(events.module_enter_events),
                module_exit_events=tuple(events.module_exit_events),
                events=events,
            )
        )
        real_materialize(trace, events)

    monkeypatch.setattr(postprocess_mod, "materialize_from_events", spy_materialize)
    trace = tl.trace(model, x)
    assert snapshots
    return trace, snapshots[0]


class NestedKwargModel(nn.Module):
    """Tiny model that produces nested positional and keyword parent edges."""

    def __init__(self) -> None:
        """Initialize the test model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with nested tuple and kwarg tensor parents."""

        y = self.linear(x)
        z = torch.cat((x, y), dim=0)
        yy = torch.cat((y, y), dim=0)
        return torch.add(z, other=yy)


class TwoModuleModel(nn.Module):
    """Tiny two-module model for module event emission checks."""

    def __init__(self) -> None:
        """Initialize the test model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through two submodules."""

        return self.relu(self.linear(x))


def test_phase0_op_event_fields_are_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert the 9 new fields and 6 ancestor payload fields are populated."""

    model = NestedKwargModel()
    x = torch.randn(2, 3)
    trace, snapshot = _trace_and_capture_events(monkeypatch, model, x)

    cat_event = next(event for event in snapshot.op_events if event.function.func_name == "cat")
    add_event = next(event for event in snapshot.op_events if event.function.func_name == "add")

    assert add_event.source_trace is trace
    assert add_event.source_trace_id == str(id(trace))
    assert add_event.pass_index == 1
    assert add_event.grad_fn_class_qualname is not None
    assert add_event.grad_fn_handle is not None
    assert add_event.parent_params == ()
    assert add_event.equivalence_class
    assert add_event.is_output_parent is True

    assert cat_event.parent_arg_positions["args"]
    assert any(isinstance(position, tuple) for position in cat_event.parent_arg_positions["args"])
    assert "other" in add_event.parent_arg_positions["kwargs"]
    assert cat_event._edge_uses
    assert add_event._edge_uses
    assert {edge.arg_kind for edge in cat_event._edge_uses} == {"positional"}
    assert {edge.arg_kind for edge in add_event._edge_uses} == {"positional", "keyword"}
    assert any(parent.arg_position == (0, 0) for parent in cat_event.parents)
    assert any(
        parent.arg_position == "other" and parent.edge_use == "kwarg"
        for parent in add_event.parents
    )

    assert isinstance(add_event.has_internal_source_ancestor, bool)
    assert isinstance(add_event.internal_source_ancestors, frozenset)
    assert isinstance(add_event.input_ancestors, frozenset)
    assert isinstance(add_event.root_ancestors, frozenset)
    assert add_event.func_call_id == add_event.function.func_call_id
    assert add_event.modules == tuple(
        (frame.address, frame.call_index) for frame in add_event.module_stack
    )


def test_phase0_module_events_are_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert prep, enter, and exit module events are emitted with atomic payloads."""

    model = TwoModuleModel()
    x = torch.randn(2, 3)
    _trace, snapshot = _trace_and_capture_events(monkeypatch, model, x)

    assert {event.address for event in snapshot.module_prep_events} >= {
        "self",
        "linear",
        "relu",
    }
    assert {event.call_label for event in snapshot.module_enter_events} == {"linear:1", "relu:1"}
    assert {event.call_label for event in snapshot.module_exit_events} == {"linear:1", "relu:1"}

    for event in snapshot.module_enter_events:
        assert event.forward_start_time > 0
        assert event.forward_args is not None
        assert event.forward_kwargs is not None
        assert event.layer_argnames

    for event in snapshot.module_exit_events:
        assert event.forward_duration >= 0
        assert event.output_tensor_labels_raw
        assert event.per_output_atomic
        for label_raw, module_stack, is_atomic_module, atomic_call in event.per_output_atomic:
            assert label_raw in event.output_tensor_labels_raw
            assert module_stack
            assert isinstance(is_atomic_module, bool)
            assert atomic_call is None or (
                isinstance(atomic_call, tuple)
                and len(atomic_call) == 2
                and isinstance(atomic_call[1], int)
            )


def test_phase0_dtype_device_refs_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert neutral dtype/device refs round-trip to the final torch Op metadata."""

    model = TwoModuleModel()
    x = torch.randn(2, 3)
    trace, snapshot = _trace_and_capture_events(monkeypatch, model, x)
    output_event = next(event for event in snapshot.op_events if event.is_output_parent)
    output_op = trace[trace.output_layers[0]]

    assert isinstance(DtypeRef.from_value(output_op.dtype), DtypeRef)
    assert isinstance(DeviceRef.from_value(output_op.out.device), DeviceRef)
    assert output_op.dtype_ref == DtypeRef.from_value(output_op.dtype)
    assert output_op.device_ref == DeviceRef.from_value(output_op.out.device)
    assert output_op.backend_address == output_op.address
    assert output_op.resolver_status == "resolved"
    assert trace.layers[output_op.layer_label].dtype_ref == output_op.dtype_ref
    assert trace.layers[output_op.layer_label].device_ref == output_op.device_ref
    assert output_event.output.tensor.dtype == str(output_op.dtype)
    assert output_event.output.tensor.device == str(output_op.out.device)


def test_phase0_trace_param_source_and_neutral_param_refs() -> None:
    """Assert torch traces expose native-module params with neutral mirror fields."""

    model = TwoModuleModel()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x)

    assert trace.backend == "torch"
    assert trace.module_identity_mode == "torch_module"
    assert trace.param_source == "native-module"
    param = trace.params["linear.weight"]
    assert param.dtype_ref == DtypeRef.from_value(param.dtype)
    assert param.device_ref is None
    assert param.backend_address == param.address
    assert param.resolver_status == "resolved"


def test_phase0_neutral_refs_default_fill_legacy_object_state() -> None:
    """Assert object-state defaults preserve neutral metadata for legacy state."""

    model = TwoModuleModel()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x)

    trace_state = trace.__getstate__()
    trace_state.pop("module_identity_mode")
    trace_state.pop("param_source")
    loaded = type(trace).__new__(type(trace))
    loaded.__setstate__(trace_state)

    op = trace.ops[0]
    op_state = op.__getstate__()
    for field_name in ("dtype_ref", "device_ref", "backend_address", "resolver_status"):
        op_state.pop(field_name)
    loaded_op = type(op).__new__(type(op))
    loaded_op.__setstate__(op_state)

    param = trace.params["linear.weight"]
    param_state = param.__getstate__()
    for field_name in ("dtype_ref", "device_ref", "backend_address", "resolver_status"):
        param_state.pop(field_name)
    loaded_param = type(param).__new__(type(param))
    loaded_param.__setstate__(param_state)

    assert loaded.module_identity_mode == "torch_module"
    assert loaded.param_source == "native-module"
    assert loaded_op.dtype_ref == DtypeRef.from_value(loaded_op.dtype)
    assert loaded_op.resolver_status == "resolved"
    assert loaded_param.dtype_ref == DtypeRef.from_value(loaded_param.dtype)
    assert loaded_param.backend_address == loaded_param.address
