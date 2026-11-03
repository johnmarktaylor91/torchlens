"""Stage 2b sparse runnable ``.tlspec`` producer tripwires."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import (
    build_sparse_run_descriptor,
)
from torchlens._runnable import (
    ControlWitnessKind,
    PathFaithfulness,
    ReadinessStatus,
    RunnableErrorCode,
)
from torchlens.errors import RunnablePreflightError
from torchlens.intervention.types import CapturedArgTemplate, LiteralValue
from torchlens.options import CaptureOptions


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


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read one emitted manifest JSON object."""

    with (path / "manifest.json").open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


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
    run = _read_manifest(path)["run"]
    kinds = {witness["kind"] for witness in run["control_witnesses"]}

    assert run["witness_completeness"] == "complete"
    assert ControlWitnessKind.LOOP_PREDICATE.value in kinds
    assert ControlWitnessKind.SCALAR_BOOL.value in kinds
    assert ControlWitnessKind.CONDITIONAL_ARM_ENTRY.value in kinds
    assert [witness["order"] for witness in run["control_witnesses"]] == list(
        range(len(run["control_witnesses"]))
    )
