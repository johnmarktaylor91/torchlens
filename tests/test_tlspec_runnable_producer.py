"""Stage 2b sparse runnable ``.tlspec`` producer tripwires."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests._format3_envelope import (
    read_manifest,
    read_record,
    read_runnable_descriptor,
    tensor_references,
)
from torchlens._io.runnable import (
    assert_sparse_core_has_no_tensor_payload,
    build_sparse_run_descriptor,
)
from torchlens._runnable import (
    ControlWitnessKind,
    PathFaithfulness,
    ReadinessStatus,
    RunnableErrorCode,
    StateSlotRole,
    TensorSlotRole,
)
from torchlens.errors import RunnablePreflightError
from torchlens.intervention.types import CapturedArgTemplate, LiteralValue
from torchlens.options import CaptureOptions


class SparseProducerModel(nn.Module):
    """Small stateful graph with repeated callables and non-tensor arguments."""

    def __init__(self) -> None:
        """Initialize named parameters and a persistent buffer."""

        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.ones(2))

    def forward(self, payload: dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a graph covering state, literals, and nested input binding."""

        value = self.linear(payload["features"])
        value = torch.add(value, 2, alpha=3)
        value = torch.add(value, 1, alpha=2)
        return value * self.scale


class UnregisteredTensorConstantModel(nn.Module):
    """Model carrying a tensor constant outside named state."""

    def __init__(self) -> None:
        """Store an intentionally unregistered tensor attribute."""

        super().__init__()
        self.constant = torch.ones(3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Consume the unreproducible tensor constant."""

        return value + self.constant


class ControlWitnessModel(nn.Module):
    """Taken-path model containing both loop and conditional predicates."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Execute a false loop observation followed by a true conditional arm."""

        while value.sum() < 0:
            value = value + 1
        if value.sum() > 0:
            value = value * 2
        return value


class CatContainerModel(nn.Module):
    """Use a list of tensors as a variadic ``torch.cat`` argument."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Concatenate two derived tensors along their final dimension."""

        return torch.cat([value + 1, value * 2], dim=-1)


class MultiOperandEinsumModel(nn.Module):
    """Use a public einsum whose eager wrapper calls an internal builtin."""

    def __init__(self, operand_count: int) -> None:
        """Store the number of vector operands in the outer product.

        Parameters
        ----------
        operand_count:
            Number of input vectors to include in the einsum expression.
        """

        super().__init__()
        self.operand_count = operand_count

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        """Form an outer product over every supplied vector.

        Parameters
        ----------
        *values:
            One-dimensional tensor operands.

        Returns
        -------
        torch.Tensor
            Outer-product result with one output axis per operand.
        """

        assert len(values) == self.operand_count
        symbols = "abcdefghijklmnopqrstuvwxyz"[: self.operand_count]
        equation = f"{','.join(symbols)}->{symbols}"
        operands = tuple(value + index for index, value in enumerate(values, start=1))
        return torch.einsum(equation, *operands)


class _OpaqueContainerMember:
    """Represent an unsupported object embedded beside a tensor reference."""


def _capture(model: nn.Module, value: Any) -> tl.Trace:
    """Capture the metadata gates required by the sparse producer.

    Parameters
    ----------
    model:
        Model to capture.
    value:
        Structured model input.

    Returns
    -------
    tl.Trace
        Cooked intervention-ready Trace with boundary containers enabled.
    """

    return tl.trace(
        model,
        value,
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
        ),
    )


def test_runnable_save_emits_frozen_sparse_descriptor_and_value_free_recipe(
    tmp_path: Path,
) -> None:
    """Emit descriptor, deduplicated calls, DAG slots, literals, and bindings."""

    trace = _capture(
        SparseProducerModel(),
        {"features": torch.arange(3, dtype=torch.float32).reshape(1, 3)},
    )
    gc.collect()
    assert trace._source_model_ref is not None
    assert trace._source_model_ref() is None
    path = tmp_path / "sparse.tlspec"

    trace.save(path, level="runnable")
    tl.validation.validate_tlspec(path)
    manifest = read_manifest(path)
    run = read_runnable_descriptor(path)

    assert manifest["format_version"] == 3
    assert manifest["tlspec_version"] == 12
    assert manifest["save_level"] == "runnable"
    assert run["capability"] == "sparse_recorded_taken_path_v2"
    assert run["call_recipe"] == "non_tensor_args_tensor_slots_context_and_obligations_v3"
    assert run["preflight"] == {"passed": True, "diagnostics": []}
    assert run["unsupported_sites"] == []
    assert run["payload_layers"] == {
        "weights": {"present": False, "schema": "state_dict_v1"},
        "nonpersistent_buffers": {
            "present": False,
            "schema": "runnable_nonpersistent_buffer_v1",
        },
        "activations": {"present": False, "schema": "selected_activation_v2"},
    }

    registry_ids = [entry["registry_id"] for entry in run["callable_registry"]]
    assert len(registry_ids) == len(set(registry_ids))
    assert all(entry["key"]["import_path"] is None for entry in run["callable_registry"])
    assert all(entry["key"]["namespace"] != "custom" for entry in run["callable_registry"])
    add_calls = [
        call
        for call in run["calls"]
        if next(
            entry
            for entry in run["callable_registry"]
            if entry["registry_id"] == call["registry_id"]
        )["key"]["qualname"]
        == "add"
    ]
    assert len(add_calls) == 2
    assert len({call["registry_id"] for call in add_calls}) == 1
    assert all(call["parent_call_ids"] for call in run["calls"][1:])
    assert any(call["literal_arguments"] for call in add_calls)

    roles = {slot["role"] for slot in run["tensor_slots"]}
    assert {
        TensorSlotRole.MODEL_INPUT.value,
        TensorSlotRole.PARAMETER.value,
        TensorSlotRole.BUFFER.value,
        TensorSlotRole.INTERMEDIATE.value,
        TensorSlotRole.OUTPUT.value,
    } <= roles
    input_slot = next(
        slot for slot in run["tensor_slots"] if slot["role"] == TensorSlotRole.MODEL_INPUT.value
    )
    assert input_slot["input_binding"]["container_path"] == ["features"]
    assert input_slot["input_binding"]["container_record_id"] >= 0

    state_slots = [slot for slot in run["tensor_slots"] if slot["state_binding"] is not None]
    state_names = {slot["state_binding"]["state_dict_name"] for slot in state_slots}
    assert {"linear.weight", "linear.bias", "scale"} <= state_names
    assert (
        next(
            slot
            for slot in state_slots
            if slot["state_binding"]["state_dict_name"] == "linear.weight"
        )["state_binding"]["semantic_role"]
        == StateSlotRole.WEIGHT.value
    )
    assert all(slot["shape"] and slot["dtype"] for slot in run["tensor_slots"])
    assert all(
        "slot_id" in argument for call in run["calls"] for argument in call["tensor_arguments"]
    )


def test_runnable_sparse_core_contains_no_tensor_payload_family(tmp_path: Path) -> None:
    """Assert descriptor, JSON records, and restored Trace contain no tensor values."""

    trace = _capture(
        SparseProducerModel(),
        {"features": torch.ones(1, 3, requires_grad=True)},
    )
    trace._layer_list[-2].out.sum().backward()
    path = tmp_path / "no_payload.tlspec"

    trace.save(path, level="runnable")
    manifest = read_manifest(path)
    records = {kind: read_record(path, kind) for kind in manifest["records"]}
    loaded = tl.load(path)

    assert tensor_references(path) == {}
    assert not list(path.rglob("*.safetensors"))
    assert all(
        family not in manifest["runnable"] for family in ("state_dict", "buffers", "activations")
    )
    assert_sparse_core_has_no_tensor_payload(read_runnable_descriptor(path))
    assert_sparse_core_has_no_tensor_payload(records)
    assert_sparse_core_has_no_tensor_payload(loaded.__getstate__())
    assert loaded._buffer_initial_values == {}
    assert all(op.func is None for op in loaded._layer_list)
    for op in loaded._layer_list:
        assert op.out is None
        assert op.transformed_out is None
        assert op.saved_args is None
        assert op.saved_kwargs is None
        assert op.out_versions_by_child in (None, {})
        assert op.grad is None
        assert op.transformed_grad is None


def test_runnable_preflight_rejects_unregistered_tensor_constant(tmp_path: Path) -> None:
    """Reject a literal tensor lacking input, state, or reproducible-source role."""

    trace = _capture(UnregisteredTensorConstantModel(), torch.zeros(3))
    descriptor = build_sparse_run_descriptor(trace)
    path = tmp_path / "rejected.tlspec"

    assert not descriptor.preflight.passed
    assert RunnableErrorCode.UNSUPPORTED_TENSOR_CONSTANT in {
        diagnostic.code for diagnostic in descriptor.preflight.diagnostics
    }
    with pytest.raises(RunnablePreflightError, match="producer preflight failed"):
        trace.save(path, level="runnable")
    assert not path.exists()


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize("operand_count", (5,))
def test_runnable_internal_einsum_identity_save_load_and_run_verified(
    tmp_path: Path, operand_count: int
) -> None:
    """Replay multi-operand einsum through its captured internal builtin."""

    inputs = tuple(torch.arange(2.0) for _ in range(operand_count))
    model = MultiOperandEinsumModel(operand_count)
    trace = _capture(model, inputs)
    descriptor = build_sparse_run_descriptor(trace)
    path = tmp_path / f"einsum_{operand_count}.tlspec"

    assert descriptor.preflight.passed
    einsum_key = next(
        entry.key for entry in descriptor.callable_registry if entry.key.qualname == "einsum"
    )
    assert einsum_key.namespace == "custom"
    assert einsum_key.import_path == "torch._C._VariableFunctionsClass:einsum"

    trace.save(path, level="runnable")
    loaded = tl.load(path)
    assert loaded.readiness.status is ReadinessStatus.READY

    result = loaded.run(inputs=inputs)
    torch.testing.assert_close(result.output, model(*inputs))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_runnable_preflight_recurses_nested_mixed_mapping_tensor_container() -> None:
    """Emit tensor leaves and a literal skeleton for nested mixed containers."""

    trace = _capture(CatContainerModel(), torch.randn(2, 3))
    target = next(op for op in trace._layer_list if op.func_name == "cat")
    original = target.args_template
    assert isinstance(original, CapturedArgTemplate)
    parents = original.args[0]
    assert isinstance(parents, tuple)
    object.__setattr__(
        target,
        "args_template",
        CapturedArgTemplate(
            args=({"nested": [parents[0], LiteralValue(3), (parents[1],)]},),
            kwargs=original.kwargs,
            func_id=original.func_id,
            notes=original.notes,
        ),
    )

    descriptor = build_sparse_run_descriptor(trace)
    call = next(call for call in descriptor.calls if target.label in call.op_labels)

    assert descriptor.preflight.passed
    assert {argument.argument_path for argument in call.tensor_arguments} == {
        ("args", 0, "nested", 0),
        ("args", 0, "nested", 2, 0),
    }
    assert any(argument.argument_path == ("args", 0) for argument in call.literal_arguments)


def test_runnable_preflight_keeps_typed_rejection_for_opaque_tensor_container_member() -> None:
    """Reject opaque members instead of treating tensor containers as literals."""

    trace = _capture(CatContainerModel(), torch.randn(2, 3))
    target = next(op for op in trace._layer_list if op.func_name == "cat")
    original = target.args_template
    assert isinstance(original, CapturedArgTemplate)
    parents = original.args[0]
    assert isinstance(parents, tuple)
    object.__setattr__(
        target,
        "args_template",
        CapturedArgTemplate(
            args=((parents[0], _OpaqueContainerMember()),),
            kwargs=original.kwargs,
            func_id=original.func_id,
            notes=original.notes,
        ),
    )

    descriptor = build_sparse_run_descriptor(trace)

    assert not descriptor.preflight.passed
    assert RunnableErrorCode.CALL_STRUCTURE_MISMATCH in {
        diagnostic.code for diagnostic in descriptor.preflight.diagnostics
    }


def test_runnable_control_flow_emits_bool_loop_and_arm_entry_witnesses(
    tmp_path: Path,
) -> None:
    """Persist observable bool values, ordered loop predicates, and taken arm entries."""

    trace = _capture(ControlWitnessModel(), torch.ones(2))
    path = tmp_path / "control.tlspec"

    trace.save(path, level="runnable")
    run = read_runnable_descriptor(path)
    kinds = {witness["kind"] for witness in run["control_witnesses"]}

    assert run["witness_completeness"] == "complete"
    assert ControlWitnessKind.LOOP_PREDICATE.value in kinds
    assert ControlWitnessKind.SCALAR_BOOL.value in kinds
    assert ControlWitnessKind.CONDITIONAL_ARM_ENTRY.value in kinds
    assert [witness["order"] for witness in run["control_witnesses"]] == list(
        range(len(run["control_witnesses"]))
    )
