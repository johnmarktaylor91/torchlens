"""Phase 8 MLX hardening tests for deferred values and event topology."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import torchlens as tl
from torchlens.capture.projections import _event_from_record
from torchlens.fastlog.storage_disk import _ctx_from_json, _ctx_to_json
from torchlens.fastlog.types import CaptureSpec, RecordContext
from torchlens.ir import (
    CaptureEvents,
    FunctionEventInput,
    MLXValueUnavailableError,
    _DEFERRED_VALUE,
)


def _minimal_context(**updates: Any) -> RecordContext:
    """Return a minimal ``RecordContext`` for projection tests."""

    values: dict[str, Any] = {
        "kind": "op",
        "label": "mlx_add_1_1_raw",
        "raw_label": "mlx_add_1_1_raw",
        "pass_index": 1,
        "event_index": 1,
        "step_index": 1,
        "layer_type": "add",
        "type_index": 1,
        "raw_index": 1,
        "func_name": "add",
        "address": None,
        "module_type": None,
        "module_pass_index": None,
        "module_stack": (),
        "recent_events": (),
        "recent_ops": (),
        "parent_labels": ("input.arg_0",),
        "input_output_address": None,
        "shape": (2, 2),
        "dtype": None,
        "tensor_device": None,
        "tensor_requires_grad": _DEFERRED_VALUE,
        "output_index": None,
        "is_bottom_level_func": True,
        "time_since_pass_start": 0.0,
        "is_scalar_bool": _DEFERRED_VALUE,
        "bool_value": _DEFERRED_VALUE,
    }
    values.update(updates)
    return RecordContext(**values)


def test_deferred_value_raises_on_use_but_json_round_trips() -> None:
    """The MLX sentinel raises on semantic use and serializes as ``None``."""

    ctx = _minimal_context()

    with pytest.raises(MLXValueUnavailableError, match="MLX lazy evaluation"):
        bool(ctx.tensor_requires_grad)
    with pytest.raises(MLXValueUnavailableError, match="MLX lazy evaluation"):
        ctx.is_scalar_bool == "anything"
    with pytest.raises(MLXValueUnavailableError, match="MLX lazy evaluation"):
        hash(ctx.bool_value)

    encoded = _ctx_to_json(ctx)
    decoded = _ctx_from_json(encoded)

    assert encoded["tensor_requires_grad"] is None
    assert encoded["is_scalar_bool"] is None
    assert encoded["bool_value"] is None
    assert decoded.tensor_requires_grad is None
    assert decoded.is_scalar_bool is None
    assert decoded.bool_value is None


def test_internal_projection_coerces_deferred_value_to_none() -> None:
    """RecordContext-to-event projection never stores the sentinel in metadata."""

    event = _event_from_record(
        _minimal_context(),
        CaptureSpec(save_out=False, save_metadata=True),
        predicate_matched=True,
    )

    assert event.output.tensor.requires_grad is None
    assert event.is_scalar_bool is None
    assert event.bool_value is None


def test_mlx_emit_function_outputs_produces_topology_complete_events() -> None:
    """MLX emits parent-linked ``OpEvent`` objects without Op registration."""

    pytest.importorskip("mlx")
    import mlx.core as mx

    from torchlens.backends.mlx import MLXBackend

    backend = MLXBackend()
    events = CaptureEvents()
    session = SimpleNamespace(
        capture_events=events,
        _mlx_saved_payloads=[],
        save_raw_activations=True,
        save_code_context=False,
        detach_saved_activations=False,
        activation_transform=None,
    )
    x = mx.ones((2, 2))
    y = mx.add(x, x)
    backend.tensor_store.set_label(x, "input.arg_0")
    reserved = events.reserve_label_block("add", 1)

    emitted = backend.emit_function_outputs(
        session,
        FunctionEventInput(
            func=mx.add,
            func_name="add",
            func_qualname="add",
            args=(x, x),
            kwargs={},
            raw_output=y,
            arg_copies=None,
            kwarg_copies=None,
            module_stack=(),
            is_bottom_level_func=True,
            func_call_id=1,
            expected_output_count=1,
        ),
        y,
        (y,),
        reserved,
    )

    assert len(emitted) == 1
    event = emitted[0]
    assert event.label_raw == reserved[0].label_raw
    assert event.parents == event.parents
    assert [edge.parent_label_raw for edge in event.parents] == ["input.arg_0"]
    assert event.output.tensor.shape == (2, 2)
    assert event.output.tensor.dtype is not None
    assert event.output.tensor.requires_grad is None
    assert event.record_context is not None

    ctx = event.record_context
    assert isinstance(ctx, RecordContext)
    with pytest.raises(MLXValueUnavailableError):
        bool(ctx.tensor_requires_grad)


def test_mlx_value_dependent_save_predicate_rejected_clearly() -> None:
    """MLX trace rejects value-dependent save predicates before capture."""

    pytest.importorskip("mlx")
    import mlx.core as mx
    import mlx.nn as nn

    class Tiny(nn.Module):
        """Tiny MLX module for predicate rejection coverage."""

        def __init__(self) -> None:
            """Initialize the linear layer."""

            super().__init__()
            self.linear = nn.Linear(2, 2)

        def __call__(self, x: mx.array) -> mx.array:
            """Run the forward pass."""

            return self.linear(x)

    with pytest.raises(NotImplementedError, match="value-dependent.*tensor_requires_grad"):
        tl.trace(Tiny(), mx.ones((1, 2)), save=lambda ctx: bool(ctx.tensor_requires_grad))
