"""Column reads in the finishing steps answer as the ``Op`` facade.

``_step_views`` reads a field of every op once, as a column, for the
lookup-key postcondition, tensor undecoration and the graph-shape hash. The
contract is that every value it returns, and every value it writes, is the one
the facade returns or writes. These tests force the column path on short
traces (``MIN_COLUMN_OPS`` set to zero) and compare it with the facade path on
models that exercise multi-pass layers, buffers, a conditional and saved
activations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._postprocess import _executor, _step_views
from torchlens._utils.hashing import compute_graph_shape_hash


class _Chain(nn.Module):
    """A chain long enough to bind a column view at the default threshold."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forty elementwise ops."""

        for _ in range(20):
            x = torch.relu(x) + 1
        return x


class _Recurrent(nn.Module):
    """A shared Linear applied three times (multi-pass layers)."""

    def __init__(self) -> None:
        """Build the shared layer."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared layer three times."""

        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x


class _Buffered(nn.Module):
    """Reads a registered buffer and branches on the data."""

    def __init__(self) -> None:
        """Register the buffer."""

        super().__init__()
        self.register_buffer("offset", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the buffer, then take a data-dependent branch."""

        y = x + self.offset
        if y.sum() > 0:
            return y * 2
        return y - 1


_MODELS: dict[str, Callable[[], nn.Module]] = {
    "chain": _Chain,
    "recurrent": _Recurrent,
    "buffered": _Buffered,
}


def _trace(name: str, **options: Any) -> Any:
    """Trace model ``name`` on a fixed input."""

    torch.manual_seed(0)
    model = _MODELS[name]().eval()
    return tl.trace(model, torch.randn(2, 4), **options)


def _op_state(trace: Any) -> list[tuple[Any, ...]]:
    """Return the fields the finishing steps read or write, per op."""

    return [(op.label, op.layer_label, op._address_normalized) for op in trace._layer_list]


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_column_path_matches_facade_path(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """The fingerprint and the written addresses are the same on both paths."""

    monkeypatch.setattr(_step_views, "MIN_COLUMN_OPS", 10**9)
    facade = _trace(name)
    monkeypatch.setattr(_step_views, "MIN_COLUMN_OPS", 0)
    columns = _trace(name)
    assert columns._provenance_fingerprint == facade._provenance_fingerprint
    assert _op_state(columns) == _op_state(facade)
    assert columns._provenance_fingerprint == compute_graph_shape_hash(
        columns, include_module_address=False
    )


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_row_views_answer_as_getattr(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Every view attribute is the facade's value, by identity; absent stays absent."""

    monkeypatch.setattr(_step_views, "MIN_COLUMN_OPS", 0)
    trace = _trace(name)
    ops = trace._layer_list
    names = (*_executor._HASH_FIELDS, "has_saved_activation", "lookup_keys")
    bound = _step_views.bind_ops(ops)
    assert bound is not None
    for op, view in zip(ops, _step_views.row_views(ops, names, bound), strict=True):
        for field in names:
            expected = getattr(op, field, _step_views.ABSENT)
            got = getattr(view, field, _step_views.ABSENT)
            assert got is expected or got == expected, (op.label, field)


def test_default_threshold_binds_long_traces_only() -> None:
    """A trace under ``MIN_COLUMN_OPS`` ops keeps the facade; a longer one binds."""

    short = _trace("buffered")
    assert len(short._layer_list) < _step_views.MIN_COLUMN_OPS
    assert _step_views.bind_ops(short._layer_list) is None
    long = _trace("chain")
    assert len(long._layer_list) >= _step_views.MIN_COLUMN_OPS
    assert _step_views.bind_ops(long._layer_list) is not None


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_undecorated_saved_tensors_match(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Saved activations carry no capture metadata on either path."""

    from torchlens._utils._tensor_label import is_tracked

    for threshold in (10**9, 0):
        monkeypatch.setattr(_step_views, "MIN_COLUMN_OPS", threshold)
        trace = _trace(name)
        saved = [op for op in trace._layer_list if op.has_saved_activation]
        assert saved, threshold
        for op in saved:
            assert not is_tracked(op.out), (threshold, op.label)
