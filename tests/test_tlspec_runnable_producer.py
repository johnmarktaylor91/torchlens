"""Stage 2b sparse runnable ``.tlspec`` producer tripwires."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import (
    assert_sparse_core_has_no_tensor_payload,
    build_sparse_run_descriptor,
)
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    ControlWitnessKind,
    RunnableErrorCode,
    StateSlotRole,
    TensorSlotRole,
)


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
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read one emitted manifest JSON object."""

    with (path / "manifest.json").open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _read_sparse_metadata(path: Path) -> dict[str, Any]:
    """Read trusted test-produced portable metadata for invariant inspection."""

    with (path / "metadata.pkl").open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - artifact is created in this test process
    assert isinstance(value, dict)
    return value


@pytest.mark.smoke
def test_runnable_save_emits_frozen_sparse_descriptor_and_value_free_recipe(
    tmp_path: Path,
) -> None:
    """Emit descriptor, deduplicated calls, DAG slots, literals, and bindings."""

    trace = _capture(
        SparseProducerModel(),
        {"features": torch.arange(3, dtype=torch.float32).reshape(1, 3)},
    )
    path = tmp_path / "sparse.tlspec"

    trace.save(path, level="runnable")
    tl.validation.validate_tlspec(path)
    manifest = _read_manifest(path)
    run = manifest["run"]

    assert manifest["schema_version"] == 2
    assert manifest["save_level"] == "runnable"
    assert run["capability"] == "sparse_recorded_taken_path_v1"
    assert run["call_recipe"] == "non_tensor_args_and_tensor_slots_v1"
    assert run["preflight"] == {"passed": True, "diagnostics": []}
    assert run["unsupported_sites"] == []
    assert run["payload_layers"] == {
        "weights": {"present": False, "schema": "state_dict_v1"},
        "activations": {"present": False, "schema": "selected_activation_v1"},
    }

    registry_ids = [entry["registry_id"] for entry in run["callable_registry"]]
    assert len(registry_ids) == len(set(registry_ids))
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


@pytest.mark.smoke
def test_runnable_sparse_core_contains_no_tensor_payload_family(tmp_path: Path) -> None:
    """Assert descriptor, manifest body, and scrubbed Trace contain no tensor values."""

    trace = _capture(
        SparseProducerModel(),
        {"features": torch.ones(1, 3, requires_grad=True)},
    )
    trace.layer_list[-2].out.sum().backward()
    path = tmp_path / "no_payload.tlspec"

    trace.save(path, level="runnable")
    manifest = _read_manifest(path)
    metadata = _read_sparse_metadata(path)

    assert manifest["body_index"] == []
    assert list((path / "blobs").iterdir()) == []
    assert_sparse_core_has_no_tensor_payload(manifest["run"])
    assert_sparse_core_has_no_tensor_payload(metadata)
    assert metadata["_buffer_initial_values"] == {}
    for op in metadata["layer_list"]:
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


@pytest.mark.smoke
def test_runnable_control_flow_emits_bool_loop_and_arm_entry_witnesses(
    tmp_path: Path,
) -> None:
    """Persist observable bool values, ordered loop predicates, and taken arm entries."""

    trace = _capture(ControlWitnessModel(), torch.ones(2))
    path = tmp_path / "control.tlspec"

    trace.save(path, level="runnable")
    run = _read_manifest(path)["run"]
    kinds = {witness["kind"] for witness in run["control_witnesses"]}

    assert run["witness_completeness"] == "complete"
    assert ControlWitnessKind.LOOP_PREDICATE.value in kinds
    assert ControlWitnessKind.SCALAR_BOOL.value in kinds
    assert ControlWitnessKind.CONDITIONAL_ARM_ENTRY.value in kinds
    assert [witness["order"] for witness in run["control_witnesses"]] == list(
        range(len(run["control_witnesses"]))
    )
