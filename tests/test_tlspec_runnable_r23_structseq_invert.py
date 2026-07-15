"""R23 runnable regressions for torch structseq outputs and unary invert."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, SparseRunDescriptor

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
    random_seed=0,
)


class _StructSeqModel(nn.Module):
    """Return one torch structseq output selected by name."""

    def __init__(self, operation: str) -> None:
        """Store the selected operation.

        Parameters
        ----------
        operation:
            One of ``sort``, ``topk``, ``max``, ``min``, or ``median``.
        """

        super().__init__()
        self.operation = operation

    def forward(self, value: torch.Tensor) -> Any:
        """Return a torch ``return_types`` structseq.

        Parameters
        ----------
        value:
            Input matrix.

        Returns
        -------
        Any
            Selected torch structseq result.
        """

        if self.operation == "sort":
            return torch.sort(value, dim=0)
        if self.operation == "topk":
            return torch.topk(value, k=2, dim=0)
        if self.operation == "max":
            return torch.max(value, dim=0)
        if self.operation == "min":
            return torch.min(value, dim=0)
        if self.operation == "median":
            return torch.median(value, dim=0)
        raise AssertionError(self.operation)


class _InvertMaskModel(nn.Module):
    """Use unary ``~`` on a boolean tensor."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a value derived from ``~mask``.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Numeric tensor proving unary invert was replayed.
        """

        mask = value > 0
        return value + (~mask).to(value.dtype)


class _GenuineInplaceModel(nn.Module):
    """Exercise genuine in-place spellings that must stay classified as in-place."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply several in-place operations and return the mutated tensor.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Mutated tensor.
        """

        out = value.clone()
        out.add_(1.0)
        out.relu_()
        out.__iadd__(2.0)
        out.masked_fill_(out > 4.0, 7.0)
        return out


def _save_load(model: nn.Module, value: torch.Tensor, path: Path) -> tl.Trace:
    """Capture, save, and load one runnable trace.

    Parameters
    ----------
    model:
        Model to capture.
    value:
        Capture input tensor.
    path:
        Destination ``.tlspec`` bundle path.

    Returns
    -------
    tl.Trace
        Loaded runnable trace.
    """

    trace = tl.trace(model, value, capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    return tl.load(path)


def _structseq_call_id(descriptor: SparseRunDescriptor) -> str:
    """Return the call id whose outputs are ``values`` and ``indices``.

    Parameters
    ----------
    descriptor:
        Runnable sparse descriptor.

    Returns
    -------
    str
        Matching runnable call id.
    """

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    for call in descriptor.calls:
        paths = tuple(slots[slot_id].output_path for slot_id in call.output_slot_ids)
        if paths == (("values",), ("indices",)):
            return call.call_id
    raise AssertionError("expected a torch structseq call with values/indices outputs")


def _rewrite_structseq_paths_to_positional(trace: tl.Trace) -> str:
    """Rewrite a loaded structseq descriptor to legacy positional output paths.

    Parameters
    ----------
    trace:
        Loaded runnable trace to mutate for the regression fixture.

    Returns
    -------
    str
        Rewritten call id.
    """

    descriptor = trace.__dict__["_runnable_descriptor"]
    call_id = _structseq_call_id(descriptor)
    call = next(item for item in descriptor.calls if item.call_id == call_id)
    positional_slots = {slot_id: (index,) for index, slot_id in enumerate(call.output_slot_ids)}
    tensor_slots = tuple(
        dataclasses.replace(slot, output_path=positional_slots[slot.slot_id])
        if slot.slot_id in positional_slots
        else slot
        for slot in descriptor.tensor_slots
    )
    trace.__dict__["_runnable_descriptor"] = dataclasses.replace(
        descriptor, tensor_slots=tensor_slots
    )
    return call_id


def _plain_tuple_sort(value: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
    """Return sort's tensor leaves in a plain tuple.

    Parameters
    ----------
    value:
        Input tensor.
    *args:
        Positional arguments forwarded to ``torch.sort``.
    **kwargs:
        Keyword arguments forwarded to ``torch.sort``.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Plain tuple with correct tensor values but wrong container type.
    """

    return tuple(torch.sort(value, *args, **kwargs))


@pytest.mark.smoke
@pytest.mark.parametrize("operation", ["sort", "topk", "max", "min", "median"])
def test_structseq_named_runtime_paths_match_positional_saved_paths(
    operation: str, tmp_path: Path
) -> None:
    """Legacy positional structseq paths must verify against named runtime paths."""

    value = torch.tensor([[1.0, 4.0, -2.0], [3.0, 2.0, 5.0]])
    model = _StructSeqModel(operation)
    loaded = _save_load(model, value, tmp_path / f"{operation}.tlspec")
    _rewrite_structseq_paths_to_positional(loaded)

    result = loaded.run(inputs=value.clone(), seed=0)
    live = model(value)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(result.output.values, live.values)
    assert torch.equal(result.output.indices, live.indices)


@pytest.mark.smoke
def test_plain_tuple_is_not_accepted_as_structseq_under_positional_paths(tmp_path: Path) -> None:
    """A plain tuple with right tensors but wrong structseq type must diverge."""

    value = torch.tensor([[1.0, 4.0, -2.0], [3.0, 2.0, 5.0]])
    loaded = _save_load(_StructSeqModel("sort"), value, tmp_path / "wrong_tuple.tlspec")
    call_id = _rewrite_structseq_paths_to_positional(loaded)
    loaded.__dict__["_runnable_callables_by_call_id"][call_id] = _plain_tuple_sort

    with pytest.raises(PathDivergenceError):
        loaded.run(inputs=value.clone(), seed=0)


@pytest.mark.smoke
def test_unary_invert_is_not_classified_as_inplace_and_runs_verified(tmp_path: Path) -> None:
    """Unary ``~tensor`` must replay without an in-place mutation contract."""

    value = torch.tensor([-1.0, 0.5, 2.0])
    model = _InvertMaskModel()
    trace = tl.trace(model, value, capture=_CAPTURE)
    invert_ops = [op for op in trace.ops.values() if op.func_name == "__invert__"]
    assert invert_ops
    assert all(not op.is_inplace for op in invert_ops)

    path = tmp_path / "invert.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=value.clone(), seed=0)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(result.output, model(value))


@pytest.mark.smoke
def test_genuine_inplace_ops_stay_inplace_and_replay_verified(tmp_path: Path) -> None:
    """In-place op names and augmented dunders must remain in-place."""

    value = torch.tensor([-2.0, 1.0, 3.0])
    model = _GenuineInplaceModel()
    trace = tl.trace(model, value, capture=_CAPTURE)
    inplace_by_name = {
        op.func_name: op.is_inplace
        for op in trace.ops.values()
        if op.func_name in {"add_", "relu_", "__iadd__", "masked_fill_"}
    }

    assert inplace_by_name == {
        "add_": True,
        "relu_": True,
        "__iadd__": True,
        "masked_fill_": True,
    }

    path = tmp_path / "inplace.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=value.clone(), seed=0)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(result.output, model(value))
