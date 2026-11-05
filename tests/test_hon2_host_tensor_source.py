"""The ``host_tensor_source`` advisory and the "not checked" verification wording.

A default capture never arms the completeness witness, so ``capture_verified`` is
``None``: nothing was checked. A tensor built from host data inside ``forward``
(``torch.from_numpy``, ``torch.from_dlpack``, ``torch.frombuffer``, ``torch.as_tensor``
on non-tensor data) is a recorded source op with no tensor parents; the capture then
carries one ``capture_advisories`` row of kind ``host_tensor_source``. The row is
derived from the recorded ops: it never warns, never moves a verdict, and never
changes the captured graph.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import honesty_preamble_lines


def _ones() -> np.ndarray:
    return np.ones((2, 2), dtype=np.float32)


_SOURCES: dict[str, Callable[[], torch.Tensor]] = {
    "from_numpy": lambda: torch.from_numpy(_ones()),
    "from_dlpack": lambda: torch.from_dlpack(_ones()),
    "frombuffer": lambda: torch.frombuffer(bytearray(_ones().tobytes()), dtype=torch.float32).view(
        2, 2
    ),
    "as_tensor": lambda: torch.as_tensor(_ones()),
}


class _HostSource(nn.Module):
    def __init__(self, make: Callable[[], torch.Tensor]) -> None:
        super().__init__()
        self.a = nn.Linear(2, 2)
        self.make = make

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) + self.make()


# Kept by shrink-r4 (per-id subset rule): sole coverage of report/_explain.py.
def test_advisory_renders_in_the_preamble_and_explain() -> None:
    """The shared honesty surfaces render the row next to the not-checked verdict."""

    trace = tl.trace(_HostSource(_SOURCES["from_numpy"]), torch.randn(2, 2))
    preamble = "\n".join(honesty_preamble_lines(trace))
    assert "host_tensor_source" in preamble
    assert "capture verification not checked" in preamble
    explained = trace.explain()
    assert "host_tensor_source" in explained
    assert "Capture verification: not checked" in explained
