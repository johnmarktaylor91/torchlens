"""The one tensor-leaf walker: traversal order and leaf rules."""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from types import MappingProxyType

import torch

from torchlens._utils.tensor_utils import iter_tensors

_Pair = namedtuple("_Pair", ["left", "right"])


@dataclass
class _Box:
    tensor: torch.Tensor


def _ids(found: list[torch.Tensor]) -> list[int]:
    return [id(tensor) for tensor in found]


def test_nested_containers_keep_insertion_order() -> None:
    a, b, c, d, e = (torch.full((1,), float(i)) for i in range(5))
    nest = {"z": [a, (b, _Pair(c, d))], "a": {"k": e}}
    assert _ids(list(iter_tensors(nest))) == _ids([a, b, c, d, e])


def test_sort_keys_orders_mappings_by_key_repr() -> None:
    a, b, c = (torch.full((1,), float(i)) for i in range(3))
    nest = ({"z": a, "a": b}, OrderedDict([(2, c)]))
    assert _ids(list(iter_tensors(nest, sort_keys=True))) == _ids([b, a, c])


def test_dataclass_and_scalars_are_leaves() -> None:
    hidden = torch.ones(1)
    assert list(iter_tensors([_Box(hidden), 3, "x", None])) == []


def test_mapping_types_widen_the_walk() -> None:
    hidden = torch.ones(1)
    proxy = MappingProxyType({"k": hidden})
    assert list(iter_tensors(proxy)) == []
    from collections.abc import Mapping

    assert _ids(list(iter_tensors(proxy, mapping_types=(Mapping,)))) == [id(hidden)]


def test_max_depth_stops_entering_containers_but_yields_tensors() -> None:
    top, deep = torch.ones(1), torch.zeros(1)
    nest = [top, [[deep]]]
    assert _ids(list(iter_tensors(nest, max_depth=1))) == [id(top)]
    assert _ids(list(iter_tensors(nest, max_depth=3))) == _ids([top, deep])
    assert _ids(list(iter_tensors(top, max_depth=0))) == [id(top)]
