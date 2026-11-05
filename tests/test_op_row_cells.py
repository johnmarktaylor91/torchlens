"""Shadow parity for the row-direct op writer (capture C2a).

``op_row_cells`` computes an op's record-sourced store cells straight from the
capture draft, without building the journal ``OpRecord``. These tests run it in
shadow at every exhaustive commit, beside the real freeze, and require its cells
to equal ``scatter_record_to_cells`` of the frozen record cell for cell (same
keys, same value types, equal values) at that same moment, on models that
exercise in-place ops, buffer updates, out= writes, ``.data`` rebinds, mutated
list arguments, mid-forward parameter updates, multi-output calls and loops.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._ir.op_record import IngestExtras
from torchlens._ir.op_record_scatter import scatter_record_to_cells
from torchlens._trace_core.op_store import _MISSING
from torchlens.backends.torch import ops as torch_ops
from torchlens.backends.torch._ops_row_cells import op_row_cells, write_op_row


class _InplaceMLP(nn.Module):
    """Residual adds and in-place activations, then mutates a consumed input."""

    def __init__(self) -> None:
        """Build two linear layers."""

        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with in-place ops after consumption."""

        h = self.a(x)
        y = h * 2
        h.relu_()
        h += y
        z = torch.cat([h, y], dim=1)
        y.mul_(3)
        w = self.b(h)
        w.add_(1.0)
        return z.sum() + w.sum()


class _BufferModel(nn.Module):
    """Batch norm running stats plus a registered buffer updated in place."""

    def __init__(self) -> None:
        """Build a conv, a batch norm and a counter buffer."""

        super().__init__()
        self.c = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.register_buffer("acc", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward that reads then mutates its buffer."""

        h = self.bn(self.c(x))
        s = h.mean(dim=(0, 2, 3)) + self.acc
        self.acc.add_(s.detach())
        return s.sum() + self.acc.sum()


class _LoopModel(nn.Module):
    """Shared parameters across loop passes, an out= write and a ``.data`` rebind."""

    def __init__(self) -> None:
        """Build one shared layer."""

        super().__init__()
        self.l = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with recurrence, out= and ``.data`` rebinding."""

        h = x
        for _ in range(3):
            h = torch.tanh(self.l(h))
        buf = torch.empty_like(h)
        torch.add(h.detach(), 1.0, out=buf)
        t = h.clone()
        t.data = buf * 2
        k = t + h
        return k.sum()


class _ShapeArgModel(nn.Module):
    """A shape list mutated after use, and a multi-output call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward reusing and mutating a shape list, then a two-output max."""

        shape = [2, 12]
        r = x.reshape(shape)
        shape[1] = 6
        shape[0] = 4
        r2 = x.reshape(shape)
        values, indices = torch.max(r2, dim=1)
        r3 = torch.roll(r2, shifts=1, dims=0)
        return r.sum() + r3.sum() + values.sum() + indices.sum()


class _ParamMutModel(nn.Module):
    """Uses a parameter, mutates it in place under no_grad, then uses it again."""

    def __init__(self) -> None:
        """Build one layer."""

        super().__init__()
        self.l = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with a mid-pass parameter update."""

        a = self.l(x)
        with torch.no_grad():
            self.l.weight.mul_(0.5)
        b = self.l(x)
        return (a + b).sum()


_SUBJECTS: tuple[tuple[str, type[nn.Module], tuple[int, ...]], ...] = (
    ("inplace_mlp", _InplaceMLP, (2, 8)),
    ("buffer", _BufferModel, (2, 3, 6, 6)),
    ("loop_out_data", _LoopModel, (2, 6)),
    ("shape_args_multi_output", _ShapeArgModel, (4, 6)),
    ("param_mutation", _ParamMutModel, (2, 5)),
)


def _same_cell(left: Any, right: Any) -> bool:
    """Return whether two cell values are the same type and equal (tensors by identity)."""

    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return False
    try:
        return bool(left == right)
    except RuntimeError:
        return False


def _cell_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """List the cell names whose keys, types or values differ."""

    names = sorted(set(expected) | set(actual))
    return [
        name
        for name in names
        if name not in expected
        or name not in actual
        or not _same_cell(expected[name], actual[name])
    ]


@pytest.fixture
def shadow_commits(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, list[str], int]]]:
    """Run ``op_row_cells`` in shadow beside every exhaustive freeze; yield the comparisons."""

    seen: list[tuple[str, list[str], int]] = []
    real_freeze = torch_ops.ExhaustiveOpDraft.freeze

    def shadow_freeze(draft: Any) -> Any:
        """Compute the direct cells first (on a copy of the call box), then freeze and compare."""

        box = None if draft.call_ref_box is None else list(draft.call_ref_box)
        direct = op_row_cells(
            draft.trace,
            draft.fields_dict,
            draft.tensor,
            draft.fire_results,
            module_stack=draft.module_stack,
            call_ref_box=box,
        )
        record = real_freeze(draft)
        expected = scatter_record_to_cells(record, IngestExtras(), draft.trace)
        seen.append((record.label_raw, _cell_mismatches(expected, direct), len(direct)))
        return record

    monkeypatch.setattr(torch_ops.ExhaustiveOpDraft, "freeze", shadow_freeze)
    yield seen


@pytest.mark.parametrize(("name", "model_cls", "shape"), _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_op_row_cells_equal_scatter_of_frozen_record(
    shadow_commits: list[tuple[str, list[str], int]],
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
) -> None:
    """Every committed op's direct cells equal the scatter of its frozen record."""

    torch.manual_seed(0)
    model = model_cls()
    tl.trace(model, torch.randn(*shape))
    assert shadow_commits, f"{name}: no exhaustive commit was observed"
    bad = [(label, names) for label, names, _ in shadow_commits if names]
    assert bad == [], f"{name}: cell mismatches {bad[:5]}"


def test_write_op_row_places_cells_in_layout_order() -> None:
    """``write_op_row`` appends one row with each cell at its layout slot, the rest unset."""

    from torchlens._data_classes.op import _OP_STORE_LAYOUT
    from torchlens._trace_core.op_store import OpRowStore

    store = OpRowStore(_OP_STORE_LAYOUT)
    fid_by_name = _OP_STORE_LAYOUT.fid_by_name
    first, second = sorted(fid_by_name, key=fid_by_name.__getitem__)[:2]
    row = write_op_row(store, {first: "a"})
    row_two = write_op_row(store, {second: "b"})
    assert (row, row_two) == (0, 1)
    assert store._rows[row][fid_by_name[first]] == "a"
    assert store._rows[row][fid_by_name[second]] is _MISSING
    assert store._rows[row_two][fid_by_name[second]] == "b"
    with pytest.raises(KeyError):
        write_op_row(store, {"no_such_field": 1})
