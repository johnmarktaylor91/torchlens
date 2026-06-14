"""Characterization net for torch capture behavior before B10 driver reroutes."""

from __future__ import annotations

import importlib
import warnings
from typing import Any, cast

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._state as torchlens_state
from torchlens.data_classes.trace import Trace
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
    """Model whose input setup falls back to the CPU device string."""

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

    for layer in trace.layer_list:
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


def test_forward_exception_restores_capture_state() -> None:
    """A user exception mid-forward cleans up global logging state."""

    model = _RaisesMidForward()
    x = torch.ones(2, 2)

    with pytest.raises(RuntimeError, match="boom during forward"):
        tl.trace(model, x, layers_to_save="all")

    assert torchlens_state._logging_enabled is False
    assert torchlens_state._active_trace is None
    assert not hasattr(x, "_tl")

    followup = tl.trace(_FiveOpTraceModel(), torch.ones(2, 2), layers_to_save="all")
    assert followup.num_ops == 4


def test_halt_signal_returns_partial_trace_from_predicate_capture() -> None:
    """Predicate halt finalizes the partial graph through the HaltSignal path."""

    trace = tl.trace(
        _FiveOpTraceModel(),
        torch.ones(2, 2),
        save=lambda ctx: ctx.kind == "op",
        halt=lambda ctx: ctx.label.startswith("relu"),
    )

    assert trace.halted is True
    assert trace.halt_reason == "relu_1_3_raw"
    assert trace.halt_frontier == "relu_1_3_raw"
    assert [layer.func_name for layer in trace.layer_list] == ["none", "__add__", "relu", "none"]
    assert torchlens_state._logging_enabled is False
    assert torchlens_state._active_trace is None


def test_selective_layers_to_save_uses_fast_pass_save_new_outs() -> None:
    """Selective layer names run the fast pass and save output-parent payloads."""

    torch.manual_seed(1703)
    trace = tl.trace(
        _SelectiveFastPassModel(),
        torch.randn(2, 3),
        layers_to_save=["relu"],
        random_seed=1703,
    )

    saved = [(layer.func_name, layer.has_saved_activation) for layer in trace.layer_list]
    assert trace.capture_mode == "fast"
    assert trace.num_saved_ops == 3
    assert saved == [
        ("none", False),
        ("linear", False),
        ("relu", True),
        ("linear", True),
        ("none", True),
    ]


def test_intervention_ready_walks_nested_output_containers() -> None:
    """Intervention-ready output extraction records stable nested container paths."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trace = tl.trace(
            _ContainerOutputModel(),
            torch.ones(2, 2),
            intervention_ready=True,
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


def test_dataparallel_unwrap_preserves_structured_input_addresses() -> None:
    """DataParallel unwrap still handles nontrivial mutable input structure."""

    payload = {
        "left": [torch.ones(2, 2)],
        "right": {"x": torch.full((2, 2), 2.0)},
    }
    trace = tl.trace(nn.DataParallel(_DataParallelStructuredInputModel()), payload)

    assert trace.model_class_name == "_DataParallelStructuredInputModel"
    assert _input_io_roles(trace) == {"input.payload.left.0", "input.payload.right.x"}
    assert torch.allclose(
        cast(torch.Tensor, trace[trace.output_layers[0]].out), torch.full((2, 2), 3.0)
    )


def test_buffer_mutation_reconciliation_orders_initial_and_written_versions() -> None:
    """Buffer reconciliation emits the initial source before the in-place write version."""

    model = _BufferMutationModel()
    trace = tl.trace(model, torch.ones(2, 2), layers_to_save="all")
    buffer_versions = [layer for layer in trace.layer_list if layer.is_buffer]

    assert len(buffer_versions) == 2
    assert [layer.pass_index for layer in buffer_versions] == [1, 2]
    assert buffer_versions[0].buffer_write_kind is None
    assert buffer_versions[0].buffer_source is None
    assert buffer_versions[1].buffer_write_kind == "inplace"
    assert str(buffer_versions[1].buffer_source).startswith("add_2_")
    assert buffer_versions[1].parents == ["add_2_2"]
    assert torch.equal(model.offset, torch.ones(2, 2))


def test_dropout_two_pass_rng_alignment_matches_full_trace() -> None:
    """Fast-pass selective saving restores the exhaustive-pass dropout RNG state."""

    x = torch.ones(4, 4)
    full = tl.trace(_DropoutTwoPassModel().train(), x, layers_to_save="all", random_seed=1704)
    selective = tl.trace(
        _DropoutTwoPassModel().train(),
        x,
        layers_to_save=["dropout"],
        random_seed=1704,
    )

    assert selective.capture_mode == "fast"
    assert torch.equal(_dropout_out(selective), _dropout_out(full))


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


def test_top_level_tuple_input_type_is_preserved_after_internal_device_move() -> None:
    """The top-level tuple argument workaround keeps tuple type for model.forward."""

    model = _TupleInputModel()
    trace = tl.trace(model, (torch.ones(2, 2), torch.full((2, 2), 2.0)))

    assert model.saw_tuple is True
    assert _input_io_roles(trace) == {"input.pair.0", "input.pair.1"}


def test_model_device_selection_params_buffers_and_paramless_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input setup selects param, buffer, then string CPU fallback devices."""

    capture_trace = importlib.import_module("torchlens.capture.trace")
    original = capture_trace._fetch_label_move_input_tensors
    seen_devices: list[Any] = []

    def spy_fetch_label_move_input_tensors(
        input_args: list[Any],
        input_arg_names: list[str],
        input_kwargs: dict[Any, Any],
        model_device: str,
    ) -> tuple[list[torch.Tensor], list[str]]:
        """Record the selected model device before delegating."""

        seen_devices.append(model_device)
        return cast(
            tuple[list[torch.Tensor], list[str]],
            original(input_args, input_arg_names, input_kwargs, model_device),
        )

    monkeypatch.setattr(
        capture_trace,
        "_fetch_label_move_input_tensors",
        spy_fetch_label_move_input_tensors,
    )

    tl.trace(_ParamDeviceModel(), torch.ones(2, 2))
    tl.trace(_BufferDeviceModel(), torch.ones(2, 2))
    tl.trace(_ParamlessDeviceModel(), torch.ones(2, 2))

    assert seen_devices == [torch.device("cpu"), torch.device("cpu"), "cpu"]
