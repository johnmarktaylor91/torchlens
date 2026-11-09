"""Same-backend ``tl.debug.compare`` / ``first_divergence`` survive an inserted or deleted op.

One inserted op (``x * 1.0``) shifts every later label, so the old label join reported the
first op as ``only-a`` and compared almost nothing. When the two graph fingerprints differ,
ops now pair by graph structure (the Bundle topology matcher), never by value; an op without
a structural partner is ``only-a`` / ``only-b``, and ops whose structural twins differ in
number stay unpaired as ``ambiguous``. Equal fingerprints keep the label join, row for row.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.debug._common import _compared_site_ops, _op_label
from torchlens.debug._compare_pairing import AMBIGUOUS_NOTE, STRUCTURE_NOTE, structural_pairs
from torchlens.debug._graph import _compare_rows

pytest.importorskip("pandas")


class _MLP(nn.Module):
    """``fc2(gelu(fc1(x)))``; subclasses change one step of the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Inserted(_MLP):
    """The same model with one value-preserving op inserted before ``fc1``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x * 1.0)))


class _Deleted(_MLP):
    """The same model with the activation deleted."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class _Twins(_MLP):
    """Two identical ``relu`` ops on one parent, each feeding its own module."""

    def __init__(self) -> None:
        super().__init__()
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x * 1.0) if getattr(self, "insert", False) else self.fc1(x)
        return self.fc2(torch.relu(y)) + self.fc3(torch.relu(y))


class _Loop(nn.Module):
    """One cell called three times (multi-pass ops)."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x


def _build(cls: type[nn.Module], reference: nn.Module | None = None) -> nn.Module:
    """An eval-mode instance, sharing ``reference``'s weights when given."""

    torch.manual_seed(0)
    model = cls().eval()
    if reference is not None:
        model.load_state_dict(reference.state_dict(), strict=False)
    return model


_X = torch.linspace(-1.0, 1.0, 8).reshape(2, 4)
_COUNTS = ("matched", "shape_mismatch", "only_a", "only_b", "value_diverged")


def _labels(trace: tl.Trace, keep: Callable[[Any], bool]) -> list[str]:
    """Row labels of the compared ops ``keep`` selects, in execution order."""

    return [label for label, op in _compared_site_ops(trace).items() if keep(op)]


def _module_op(trace: tl.Trace, module: str, func: str) -> str:
    """The single row label of ``func`` run inside ``module``."""

    (label,) = _labels(
        trace, lambda op: str(op.module).split(":")[0] == module and op.func_name == func
    )
    return label


def test_inserted_op_is_reported_and_every_other_op_compared() -> None:
    """Red on the label join: it compared 1 of the ops; now all pair and the mul is only-b."""

    reference = _build(_MLP)
    a, b = tl.trace(reference, _X), tl.trace(_build(_Inserted, reference), _X)
    frame = tl.debug.compare(a, b).to_pandas()
    paired = frame[frame["status"] == "present-in-both"]
    assert list(paired["op"]) == list(_compared_site_ops(a))
    assert paired["allclose"].all()
    assert frame.attrs["matched"] == len(paired)
    (mul,) = _labels(b, lambda op: op.func_name == "mul")
    assert list(frame.loc[frame["status"] == "only-b", "op"]) == [mul]
    row = tl.debug.first_divergence(a, b)
    assert row is not None and (row["op"], row["status"]) == (mul, "only-b")


def test_inserted_op_then_a_real_change_is_placed_on_the_changed_op() -> None:
    """A later bias change is the first divergence, on fc2's op, paired by structure."""

    reference = _build(_MLP)
    changed = _build(_Inserted, reference)
    with torch.no_grad():
        changed.fc2.bias += 1.0
    a, b = tl.trace(reference, _X), tl.trace(changed, _X)
    row = tl.debug.first_divergence(a, b)
    assert row is not None
    assert (row["op"], row["status"], row["allclose"]) == (
        _module_op(a, "fc2", "linear"),
        "present-in-both",
        False,
    )
    assert STRUCTURE_NOTE in row["reason"]


def test_deleted_op_is_only_a_and_its_consumer_still_pairs() -> None:
    """The deleted gelu is the first divergence; fc2 pairs with fc2 and its values differ."""

    reference = _build(_MLP)
    a, b = tl.trace(reference, _X), tl.trace(_build(_Deleted, reference), _X)
    rows = {row["op"]: row for row in _compare_rows(a, b)[0]}
    gelu = _module_op(a, "act", "gelu")
    assert rows[gelu]["status"] == "only-a"
    assert rows[_module_op(a, "fc1", "linear")]["allclose"] is True
    assert rows[_module_op(a, "fc2", "linear")]["allclose"] is False
    pairs, _ = structural_pairs(a, b, _compared_site_ops(a), _compared_site_ops(b))
    assert pairs[_module_op(a, "fc2", "linear")] == _module_op(b, "fc2", "linear")
    assert tl.debug.first_divergence(a, b)["op"] == gelu  # type: ignore[index]


def test_repeated_identical_ops_pair_in_execution_order() -> None:
    """Equal numbers of structural twins pair one to one, in order; nothing is ambiguous."""

    reference = _build(_Twins)
    inserted = _build(_Twins, reference)
    inserted.insert = True  # type: ignore[assignment]
    a, b = tl.trace(reference, _X), tl.trace(inserted, _X)
    ops_a, ops_b = _compared_site_ops(a), _compared_site_ops(b)
    pairs, ambiguous = structural_pairs(a, b, ops_a, ops_b)
    is_relu = lambda op: op.func_name == "relu"  # noqa: E731
    assert [pairs[label] for label in _labels(a, is_relu)] == _labels(b, is_relu)
    assert ambiguous == (set(), set())
    assert set(pairs) == set(ops_a)


class _TwoRelu(nn.Module):
    """``relu(y) + 2 * relu(y)`` with two separate relu ops."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        return torch.relu(y) + torch.relu(y) * 2.0


class _OneRelu(_TwoRelu):
    """The same sum with one relu op deleted (its value reused)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.relu(self.fc1(x))
        return r + r * 2.0


@pytest.mark.parametrize("deleted_side", ["b", "a"])
def test_a_deleted_twin_is_ambiguous_never_guessed(deleted_side: str) -> None:
    """Two twins against one: none pairs, each relu row is an ambiguous coverage gap."""

    reference = _build(_TwoRelu)
    two, one = tl.trace(reference, _X), tl.trace(_build(_OneRelu, reference), _X)
    a, b = (two, one) if deleted_side == "b" else (one, two)
    rows = _compare_rows(a, b)[0]
    relu_ops = set(_labels(a, lambda op: op.func_name == "relu"))
    relu_ops |= set(_labels(b, lambda op: op.func_name == "relu"))
    relu_rows = [row for row in rows if row["op"] in relu_ops]
    assert len(relu_rows) == 3
    assert all(row["status"] in ("only-a", "only-b") for row in relu_rows)
    assert all(row["reason"] == AMBIGUOUS_NOTE for row in relu_rows)
    assert tl.debug.first_divergence(a, b)["reason"] == AMBIGUOUS_NOTE  # type: ignore[index]


def test_an_extra_same_shaped_candidate_is_never_paired() -> None:
    """One relu against two same-shaped relu candidates: no relu pair is guessed."""

    reference = _build(_TwoRelu)
    a, b = tl.trace(_build(_OneRelu, reference), _X), tl.trace(reference, _X)
    pairs, ambiguous = structural_pairs(a, b, _compared_site_ops(a), _compared_site_ops(b))
    (relu_a,) = _labels(a, lambda op: op.func_name == "relu")
    assert relu_a not in pairs
    assert ambiguous[0] == {relu_a}
    assert ambiguous[1] == set(_labels(b, lambda op: op.func_name == "relu"))


@pytest.mark.parametrize("cls", [_MLP, _Twins, _Loop], ids=lambda cls: cls.__name__)
def test_identical_graphs_give_the_label_join_rows_byte_for_byte(cls: type[nn.Module]) -> None:
    """Forcing the structural path on an identical graph reproduces the label join exactly."""

    model = _build(cls)
    a, b = tl.trace(model, _X), tl.trace(model, _X + 0.5)
    assert a.provenance.fingerprint == b.provenance.fingerprint
    rows, attrs = _compare_rows(a, b)
    b._provenance_fingerprint = "forced-different"
    forced_rows, forced_attrs = _compare_rows(a, b)
    assert forced_rows == rows
    assert {k: forced_attrs[k] for k in _COUNTS} == {k: attrs[k] for k in _COUNTS}
    assert [row["op"] for row in rows] == [_op_label(op) for op in _compared_site_ops(a).values()]


def test_structural_pairing_survives_save_and_load(tmp_path: Any) -> None:
    """The inserted-op pair gives the same rows after both traces are saved and loaded."""

    reference = _build(_MLP)
    a, b = tl.trace(reference, _X), tl.trace(_build(_Inserted, reference), _X)
    live = _compare_rows(a, b)[0]
    a.save(str(tmp_path / "a.tlspec"))
    b.save(str(tmp_path / "b.tlspec"))
    loaded = _compare_rows(tl.load(str(tmp_path / "a.tlspec")), tl.load(str(tmp_path / "b.tlspec")))
    strip = lambda rows: [{k: v for k, v in r.items() if k != "source_line"} for r in rows]  # noqa: E731
    assert strip(loaded[0]) == strip(live)
