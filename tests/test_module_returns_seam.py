"""Unit tests for the recorded-module-returns seam (``torchlens.backends._module_returns``).

The seam is backend-agnostic, so these tests drive it with a stand-in leaf type; the
per-backend behaviour (MLX first) lives in ``tests/test_module_return_cell.py``.
"""

from __future__ import annotations

from collections import namedtuple

from torchlens._ir.container import DictKey, NamedField, TupleIndex
from torchlens.backends._module_returns import (
    ModuleReturnLedger,
    ReturnedLeaf,
    iter_returned_leaves,
)


class _Leaf:
    """Stand-in native array carrying the raw label of its producing op."""

    def __init__(self, label: str | None) -> None:
        self.label = label


def _is_leaf(value: object) -> bool:
    """Return whether ``value`` is a stand-in leaf."""

    return isinstance(value, _Leaf)


def _label_of(value: object) -> str | None:
    """Return the stand-in leaf's raw label."""

    return value.label if isinstance(value, _Leaf) else None


_Pair = namedtuple("_Pair", ["first", "second"])


def test_bare_leaf_has_empty_path() -> None:
    leaf = _Leaf("relu_1_raw")
    assert list(iter_returned_leaves(leaf, _is_leaf)) == [((), leaf)]


def test_nested_paths_use_torch_path_components() -> None:
    a, b, c = _Leaf("a"), _Leaf("b"), _Leaf("c")
    value = (a, {"k": [b, 3, "text", None]}, _Pair(first=c, second=1.5))
    assert list(iter_returned_leaves(value, _is_leaf)) == [
        ((TupleIndex(0),), a),
        ((TupleIndex(1), DictKey("k"), TupleIndex(0)), b),
        ((TupleIndex(2), NamedField("first")), c),
    ]


def test_duplicate_leaves_keep_one_path_per_occurrence() -> None:
    x = _Leaf("x")
    ledger = ModuleReturnLedger()
    ledger.record("block", 1, (x, x), is_leaf=_is_leaf, label_of=_label_of)
    assert ledger.calls[("block", 1)] == (
        ReturnedLeaf(path=(TupleIndex(0),), label_raw="x"),
        ReturnedLeaf(path=(TupleIndex(1),), label_raw="x"),
    )


def test_non_array_return_records_no_leaves() -> None:
    ledger = ModuleReturnLedger()
    ledger.record("head", 2, {"loss": 0.5, "name": "x"}, is_leaf=_is_leaf, label_of=_label_of)
    assert ledger.calls[("head", 2)] == ()


def test_unlabeled_leaf_is_recorded_without_label() -> None:
    ledger = ModuleReturnLedger()
    ledger.record("proj", 1, _Leaf(None), is_leaf=_is_leaf, label_of=_label_of)
    assert ledger.calls[("proj", 1)] == (ReturnedLeaf(path=(), label_raw=None),)


def test_root_call_is_never_recorded() -> None:
    ledger = ModuleReturnLedger()
    ledger.record("self", 1, _Leaf("out"), is_leaf=_is_leaf, label_of=_label_of)
    assert ledger.calls == {}


def test_calls_keep_module_exit_order() -> None:
    ledger = ModuleReturnLedger()
    inner, outer = _Leaf("inner"), _Leaf("outer")
    ledger.record("block.inner", 1, inner, is_leaf=_is_leaf, label_of=_label_of)
    ledger.record("block", 1, (outer, inner), is_leaf=_is_leaf, label_of=_label_of)
    ledger.record("block.inner", 2, inner, is_leaf=_is_leaf, label_of=_label_of)
    assert list(ledger.calls) == [("block.inner", 1), ("block", 1), ("block.inner", 2)]
