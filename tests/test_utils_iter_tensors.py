"""The one tensor-leaf walker: traversal order and leaf rules."""

from __future__ import annotations

from collections import namedtuple

import torch

from torchlens._utils.tensor_leaves import iter_tensors

_Pair = namedtuple("_Pair", ["left", "right"])


def _ids(found: list[torch.Tensor]) -> list[int]:
    return [id(tensor) for tensor in found]


# Kept by the shrink-r4 coverage proof: sole branch coverage of _utils/tensor_leaves.py.
def test_max_depth_stops_entering_containers_but_yields_tensors() -> None:
    top, deep = torch.ones(1), torch.zeros(1)
    nest = [top, [[deep]]]
    assert _ids(list(iter_tensors(nest, max_depth=1))) == [id(top)]
    assert _ids(list(iter_tensors(nest, max_depth=3))) == _ids([top, deep])
    assert _ids(list(iter_tensors(top, max_depth=0))) == [id(top)]
