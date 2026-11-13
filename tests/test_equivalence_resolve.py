"""Equivalence class and call-site class as resolvers over recorded inputs (lane FQ).

The argument record holds immutable values by reference and computes every other
value's tokens at the call, so resolving the record after the call gives the
call-time class even when an argument was mutated in between. The record is also the
digest memo's key, so equal records must never stand for different tokens.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens._utils._call_site import call_site_class
from torchlens._utils.hashing import make_short_barcode_from_input
from torchlens.backends.torch import _arg_hash
from torchlens.backends.torch._equivalence_resolve import (
    observe_call_site_key,
    resolve_equivalence_class,
)
from torchlens.backends.torch.tensor_tracking import _get_equivalence_class


def _reference_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """The argument hash by the generic walk alone, with no record and no memo."""

    tokens: list[Any] = []
    for index, arg in enumerate(args):
        _arg_hash._append_arg_hash(arg, f"pos{index}", tokens)
    for key, arg in kwargs.items():
        _arg_hash._append_arg_hash(arg, f"kw_{key}", tokens)
    return make_short_barcode_from_input(tokens) if tokens else "no_args"


class _Stateful:
    """An argument whose printed form follows its mutable state."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"stateful{self.value}"


_FIELDS: dict[str, Any] = {"in_multi_output": False, "module": ("block.0", 1)}


def test_argument_mutated_after_the_call_resolves_to_the_call_time_class() -> None:
    """A list, dict, set and object argument changed after the call keep their class."""

    shape = [2, 3]
    options = {"mode": "a"}
    members = {1, 2}
    stateful = _Stateful(1)
    args = (torch.ones(2, 3), shape, members, stateful)
    kwargs = {"options": options}
    at_call = _get_equivalence_class(args, kwargs, 0, "view", _FIELDS)
    record = _arg_hash.observe_arg_inputs(args, kwargs)

    shape[0] = 7
    options["mode"] = "b"
    members.add(3)
    stateful.value = 2

    resolved = resolve_equivalence_class("view", record, None, _FIELDS["module"])
    assert resolved == at_call
    assert _get_equivalence_class(args, kwargs, 0, "view", _FIELDS) != at_call


def test_tensor_resized_after_the_call_resolves_to_its_call_time_shape() -> None:
    """The record holds the shape read at the call, not the tensor."""

    x = torch.ones(2, 3)
    at_call = _get_equivalence_class((x, 1), {}, 0, "add", _FIELDS)
    record = _arg_hash.observe_arg_inputs((x, 1), {})
    x.resize_(4, 5)
    assert resolve_equivalence_class("add", record, None, _FIELDS["module"]) == at_call
    assert _get_equivalence_class((x, 1), {}, 0, "add", _FIELDS) != at_call


_VALUES: list[Any] = [
    0,
    1,
    -3,
    True,
    False,
    None,
    0.0,
    -0.0,
    1.0,
    0.5,
    float("nan"),
    float("inf"),
    "text",
    "an object at 0x1234",
    torch.float32,
    torch.int64,
    torch.device("cpu"),
    torch.ones(2, 3),
    torch.zeros(4),
    nn.Parameter(torch.ones(2)),
    [1, torch.zeros(2), (3.0, None)],
    (1, 2),
    (1.0, 2.0),
    {"k": torch.zeros(1), "j": 4},
    {1, 2},
    frozenset({1, 2}),
    torch.Size([2, 3]),
    slice(1, 5),
    object(),
    torch.Generator(),
]


def test_resolved_hash_matches_the_generic_walk_for_every_value_kind() -> None:
    """Positional and keyword arguments of every kind hash as the generic walk does."""

    for value in _VALUES:
        assert _arg_hash._get_hash_from_args((value,), {}) == _reference_hash((value,), {})
        assert _arg_hash._get_hash_from_args((torch.ones(1), value), {"k": value}) == (
            _reference_hash((torch.ones(1), value), {"k": value})
        )
    assert _arg_hash._get_hash_from_args(tuple(_VALUES), {}) == _reference_hash(tuple(_VALUES), {})
    assert _arg_hash._get_hash_from_args((nn.Parameter(torch.ones(1)),), {}) == "no_args"
    assert _arg_hash._get_hash_from_args((), {}) == "no_args"


def test_equal_comparing_values_with_different_tokens_never_share_a_memo_entry() -> None:
    """1, 1.0 and True compare equal, as do 0.0 and -0.0, and (2, 3) and its Size."""

    groups: list[list[Any]] = [
        [1, 1.0, True],
        [0, 0.0, -0.0, False],
        [(2, 3), (2.0, 3.0)],
        [torch.Size([2, 3]), (2.0, 3.0)],
    ]
    for group in groups:
        _arg_hash._ARG_HASH_DIGESTS.clear()
        for _round in range(2):  # the second round answers from the memo
            for value in group:
                assert _arg_hash._get_hash_from_args((value,), {}) == _reference_hash(
                    (value,), {}
                ), value
        assert len({_reference_hash((value,), {}) for value in group}) == len(group)


def test_equivalence_suffixes_follow_the_recorded_slot_and_module() -> None:
    """The output slot and module address are appended as before."""

    args = (torch.ones(2), 2)
    record = _arg_hash.observe_arg_inputs(args, {})
    fields: dict[str, Any] = {"in_multi_output": True, "module": ("enc", 2)}
    expected = f"split_{_reference_hash(args, {})}_outindex1_moduleenc"
    assert _get_equivalence_class(args, {}, 1, "split", fields) == expected
    assert resolve_equivalence_class("split", record, 1, ("enc", 2)) == expected
    plain: dict[str, Any] = {"in_multi_output": False, "module": None}
    assert _get_equivalence_class(args, {}, 1, "split", plain) == (
        f"split_{_reference_hash(args, {})}"
    )


def test_call_site_key_observed_then_resolved_in_op_order() -> None:
    """Observed keys numbered in op order get first-appearance numbers."""

    chain = (("model.py", 10),)
    other = (("model.py", 12),)
    stateful = frozenset({"enc"})
    modules = [("enc", 1), ("enc.act", 1)]
    keys = [
        observe_call_site_key(chain, modules, stateful, "relu", None),
        observe_call_site_key(other, modules, stateful, "relu", None),
        observe_call_site_key(chain, modules, stateful, "relu", 1),
        observe_call_site_key(chain, modules, stateful, "relu", None),
    ]
    assert keys[0] == ("enc", chain, "relu", None)
    assert observe_call_site_key(chain, modules, None, "relu", None) == (
        "enc.act",
        chain,
        "relu",
        None,
    )
    index: dict[Any, int] = {}
    resolved = [call_site_class(index, key, "relu", key[3]) for key in keys if key is not None]
    assert resolved == ["relu@cs1", "relu@cs2", "relu@cs3_outindex1", "relu@cs1"]
    assert observe_call_site_key((), modules, stateful, "relu", None) is None
    assert observe_call_site_key(None, modules, stateful, "relu", None) is None


class _MutatesPadAfterCall(nn.Module):
    """Pads twice with one list, changed between the two calls."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = [1, 1]
        y = F.pad(x, pad)
        pad[0] = 2
        return F.pad(y, pad)


class _LiteralPads(nn.Module):
    """The same two pads with literal lists."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, [1, 1])
        return F.pad(y, [2, 1])


def test_traced_op_with_a_list_mutated_after_its_call_keeps_its_call_time_class() -> None:
    """Each pad's class in a trace is the class of the list it was called with."""

    x = torch.randn(2, 3)
    mutated = [op.graph.equivalence_class for op in tl.trace(_MutatesPadAfterCall(), x).ops]
    literal = [op.graph.equivalence_class for op in tl.trace(_LiteralPads(), x).ops]
    assert mutated == literal
    assert len(set(mutated)) == len(mutated)
