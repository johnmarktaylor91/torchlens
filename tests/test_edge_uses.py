"""Public ``Op.edge_uses`` accessor tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens.data_classes.op import Op
from torchlens.data_classes.trace import Trace
from torchlens.intervention.types import EdgeUseRecord, TupleIndex


class _RepeatedEdgeModel(torch.nn.Module):
    """Model with repeated and single parent tensor uses."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run operations that expose per-edge multiplicity.

        Parameters
        ----------
        x:
            Input tensor used by repeated and single-use operations.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Results of ``x + x``, ``torch.cat([x, x])``, and ``torch.relu(x)``.
        """

        added = x + x
        concatenated = torch.cat([x, x], dim=0)
        single_use = torch.relu(x)
        return added, concatenated, single_use


def _edge_trace() -> Trace:
    """Capture the repeated-edge test model.

    Returns
    -------
    Trace
        Trace containing repeated parent edge uses.
    """

    return tl.trace(_RepeatedEdgeModel(), torch.randn(2, 3))


def _first_op_by_func_name(trace: Trace, func_name: str) -> Op:
    """Return the first operation with a matching function name.

    Parameters
    ----------
    trace:
        Trace to search.
    func_name:
        Captured function name to match.

    Returns
    -------
    Op
        First matching operation.
    """

    return next(op for op in trace.layer_list if op.func_name == func_name)


def _arg_path_values(edge_uses: tuple[EdgeUseRecord, ...]) -> list[tuple[int, ...]]:
    """Convert ``EdgeUseRecord.arg_path`` components to integer tuples.

    Parameters
    ----------
    edge_uses:
        Edge-use records whose paths should be normalized.

    Returns
    -------
    list[tuple[int, ...]]
        Integer path payloads for comparison.
    """

    values: list[tuple[int, ...]] = []
    for edge_use in edge_uses:
        path_values: list[int] = []
        for component in edge_use.arg_path:
            if isinstance(component, TupleIndex):
                path_values.append(component.index)
            else:
                path_values.append(int(component))
        values.append(tuple(path_values))
    return values


@pytest.mark.smoke
def test_edge_uses_reports_repeated_and_single_parent_uses() -> None:
    """Repeated same-parent tensor uses remain visible through ``edge_uses``."""

    trace = _edge_trace()
    input_op = next(op for op in trace.layer_list if op.is_input)
    add_op = _first_op_by_func_name(trace, "__add__")
    cat_op = _first_op_by_func_name(trace, "cat")
    relu_op = _first_op_by_func_name(trace, "relu")

    assert isinstance(add_op.edge_uses, tuple)
    assert add_op.edge_uses == tuple(add_op._edge_uses)
    assert len(add_op.edge_uses) == 2
    assert {edge_use.parent_label for edge_use in add_op.edge_uses} == {input_op.layer_label}
    assert _arg_path_values(add_op.edge_uses) == [(0,), (1,)]

    assert len(cat_op.edge_uses) == 2
    assert {edge_use.parent_label for edge_use in cat_op.edge_uses} == {input_op.layer_label}
    assert _arg_path_values(cat_op.edge_uses) == [(0, 0), (0, 1)]

    assert len(relu_op.edge_uses) == 1
    assert relu_op.edge_uses[0].parent_label == input_op.layer_label
    assert _arg_path_values(relu_op.edge_uses) == [(0,)]


@pytest.mark.smoke
def test_edge_uses_is_tuple_and_empty_without_recorded_edges() -> None:
    """The public accessor is immutable and empty for ops with no recorded edges."""

    trace = _edge_trace()
    input_op = next(op for op in trace.layer_list if op.is_input)

    assert input_op.edge_uses == ()
    assert isinstance(input_op.edge_uses, tuple)
    with pytest.raises(AttributeError):
        input_op.edge_uses.append(EdgeUseRecord("", "", "positional", (), None, None, 0))


@pytest.mark.smoke
def test_edge_uses_survives_tlspec_roundtrip(tmp_path: Path) -> None:
    """Saved ``_edge_uses`` records remain reachable via ``edge_uses`` after load."""

    trace = _edge_trace()
    path = tmp_path / "edge_uses.tlspec"
    trace.save(path)

    loaded = tl.load(path)
    assert isinstance(loaded, Trace)

    add_op = _first_op_by_func_name(loaded, "__add__")
    cat_op = _first_op_by_func_name(loaded, "cat")
    input_op = next(op for op in loaded.layer_list if op.is_input)

    assert len(add_op.edge_uses) == 2
    assert {edge_use.parent_label for edge_use in add_op.edge_uses} == {input_op.layer_label}
    assert _arg_path_values(add_op.edge_uses) == [(0,), (1,)]
    assert len(cat_op.edge_uses) == 2
    assert _arg_path_values(cat_op.edge_uses) == [(0, 0), (0, 1)]
