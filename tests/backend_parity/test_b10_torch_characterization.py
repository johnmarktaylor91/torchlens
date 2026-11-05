"""Characterization net for torch capture behavior before B10 driver reroutes."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens.intervention.types import DictKey, TupleIndex

pytestmark = pytest.mark.backend_parity


class _RaisesMidForward(nn.Module):
    """Model that raises after one logged torch operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one op and then raise a user exception."""

        _ = torch.relu(x)
        raise RuntimeError("boom during forward")


class _FiveOpTraceModel(nn.Module):
    """Small arithmetic model for halt and exception cleanup coverage."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a deterministic chain of tensor operations."""

        x = x + 1
        x = torch.relu(x)
        x = x * 2
        return x - 3


class _SelectiveFastPassModel(nn.Module):
    """Model whose selective layer request triggers ``save_new_outs``."""

    def __init__(self) -> None:
        """Initialize fixed linear layers."""

        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear-ReLU-linear forward pass."""

        return self.fc2(torch.relu(self.fc1(x)))


class _ContainerOutputModel(nn.Module):
    """Model returning tensors inside nested output containers."""

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Return a nested dict, tuple, and list output structure."""

        return {"a": x + 1, "nested": (x * 2, [x - 1])}


class _DataParallelStructuredInputModel(nn.Module):
    """Model accepting structured tensor inputs through DataParallel unwrap."""

    def forward(self, payload: dict[str, Any]) -> torch.Tensor:
        """Consume tensors from nested mutable input containers."""

        return cast(torch.Tensor, payload["left"][0] + payload["right"]["x"])


class _StructuredInputModel(nn.Module):
    """Model accepting nested positional and keyword tensor containers."""

    def forward(self, payload: dict[str, Any], *, bias: dict[str, Any]) -> torch.Tensor:
        """Consume nested positional and keyword tensors."""

        return cast(
            torch.Tensor,
            payload["left"][0] + payload["right"]["x"] + bias["shift"][0],
        )


class _TupleInputModel(nn.Module):
    """Model that records whether its single positional arg stays a tuple."""

    def __init__(self) -> None:
        """Initialize tuple-observation state."""

        super().__init__()
        self.saw_tuple = False

    def forward(self, pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return the sum of a tuple input while recording its type."""

        self.saw_tuple = isinstance(pair, tuple)
        return pair[0] + pair[1]


class _BufferMutationModel(nn.Module):
    """Model that reads and mutates a registered buffer during forward."""

    def __init__(self) -> None:
        """Initialize a mutable buffer."""

        super().__init__()
        self.register_buffer("offset", torch.zeros(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read the buffer, mutate it in place, then read the new value."""

        before = x + self.offset
        self.offset.add_(1)
        return before + self.offset


class _DropoutTwoPassModel(nn.Module):
    """Training-mode dropout model for RNG alignment characterization."""

    def __init__(self) -> None:
        """Initialize dropout."""

        super().__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout and a trailing op."""

        return self.dropout(x) + 1


class _ParamDeviceModel(nn.Module):
    """Model whose device is selected from its first parameter."""

    def __init__(self) -> None:
        """Initialize a parameter."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the parameter in a tensor op."""

        return x + self.weight


class _BufferDeviceModel(nn.Module):
    """Model whose device is selected from its first buffer."""

    def __init__(self) -> None:
        """Initialize a buffer and no parameters."""

        super().__init__()
        self.register_buffer("bias", torch.ones(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the buffer in a tensor op."""

        return x + self.bias


class _ParamlessDeviceModel(nn.Module):
    """Model that pins no device: no parameters, no buffers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a parameterless tensor op."""

        return x + 1


def _dropout_out(trace: Trace) -> torch.Tensor:
    """Return the saved dropout activation from a trace.

    Parameters
    ----------
    trace:
        Trace containing a dropout op.

    Returns
    -------
    torch.Tensor
        Saved dropout output.
    """

    for layer in trace._layer_list:
        if layer.func_name == "dropout" and layer.has_saved_activation:
            return cast(torch.Tensor, layer.out)
    raise AssertionError("dropout output was not saved")


def _input_io_roles(trace: Trace) -> set[str]:
    """Return source-address roles for all input layers.

    Parameters
    ----------
    trace:
        Trace whose input source layers should be inspected.

    Returns
    -------
    set[str]
        Input ``io_role`` values.
    """

    return {cast(str, trace[label].io_role) for label in trace.input_layers}


def test_intervention_ready_walks_nested_output_containers() -> None:
    """Intervention-ready output extraction records stable nested container paths."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trace = tl.trace(
            _ContainerOutputModel(),
            torch.ones(2, 2),
            capture=tl.options.CaptureOptions(replayable=True),
        )

    output_paths = [trace[label].container_path for label in trace.output_layers]
    assert output_paths == [
        (DictKey("a"),),
        (DictKey("nested"), TupleIndex(0)),
        (DictKey("nested"), TupleIndex(1), TupleIndex(0)),
    ]
    assert [trace[label].io_role for label in trace.output_layers] == [
        "output.a",
        "output.nested.0",
        "output.nested.1.0",
    ]


def test_buffer_mutation_reconciliation_orders_initial_and_written_versions() -> None:
    """Buffer reconciliation emits the initial source before the in-place write version."""

    model = _BufferMutationModel()
    trace = tl.trace(model, torch.ones(2, 2), save="all")
    buffer_versions = [layer for layer in trace._layer_list if layer.is_buffer]

    assert len(buffer_versions) == 2
    # grouping rule row 74 (tests/test_grouping_call_sites.py; docs/guides/grouping.md):
    # buffer versions group by the line that wrote them, and a buffer written once
    # keeps its starting value and its new value as two single-pass layers.
    assert [layer.pass_number for layer in buffer_versions] == [1, 1]
    assert len({layer.layer_label for layer in buffer_versions}) == 2
    assert buffer_versions[0].buffer_write_kind is None
    assert buffer_versions[0].buffer_source is None
    assert buffer_versions[1].buffer_write_kind == "inplace"
    assert str(buffer_versions[1].buffer_source).startswith("add_2_")
    assert buffer_versions[1].parents == ("add_2_2",)
    # A default capture is a pure observer: the live buffer is restored.
    assert torch.equal(model.offset, torch.zeros(2, 2))


def test_nested_input_paths_source_labels_and_caller_input_immutability() -> None:
    """Nested positional and kwarg tensors receive source labels without caller mutation."""

    left = torch.ones(2, 2)
    right = torch.full((2, 2), 2.0)
    shift = torch.full((2, 2), 3.0)
    payload = {"left": [left], "right": {"x": right}}
    bias = {"shift": [shift]}
    original_ids = (id(payload["left"][0]), id(payload["right"]["x"]), id(bias["shift"][0]))

    trace = tl.trace(_StructuredInputModel(), payload, input_kwargs={"bias": bias})

    assert _input_io_roles(trace) == {
        "input.payload.left.0",
        "input.payload.right.x",
        "input.bias.shift.0",
    }
    assert (id(payload["left"][0]), id(payload["right"]["x"]), id(bias["shift"][0])) == original_ids
    assert not hasattr(left, "_tl")
    assert not hasattr(right, "_tl")
    assert not hasattr(shift, "_tl")


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only device-preservation pin; skipped without a real CUDA device.",
)
def test_paramless_model_cuda_inputs_stay_on_cuda() -> None:
    """A device-less model's CUDA inputs are computed on CUDA, not the CPU.

    The pre-fix ``"cpu"`` fallback moved every input of a parameter- and
    buffer-free model to the CPU and ran the whole forward there -- silently
    diverging from eager (``model(x)`` computes on ``x``'s device) and, on
    the first real-GPU run, leaving the CUPTI correlation matrix empty
    because the "CUDA" capture launched zero kernels.
    """

    trace = tl.trace(_ParamlessDeviceModel(), torch.ones(2, 2, device="cuda"))

    ops = list(trace)
    assert ops
    for op in ops:
        out = op.out
        if isinstance(out, torch.Tensor):
            assert out.device.type == "cuda", f"{op.layer_label} ran on {out.device}"
