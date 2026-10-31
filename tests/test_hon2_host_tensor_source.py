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

import warnings
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import capture_honesty_facts, honesty_preamble_lines


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


class _Plain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x)


def _host_rows(trace: tl.Trace) -> list[dict[str, object]]:
    return [
        row
        for row in trace.annotations.get("capture_advisories", [])
        if row["kind"] == "host_tensor_source"
    ]


@pytest.mark.parametrize("name", sorted(_SOURCES))
def test_each_named_host_source_records_one_advisory_row(name: str) -> None:
    """Each named source records exactly one row, with no warning and no verdict change."""

    torch.manual_seed(0)
    x = torch.randn(2, 2)
    plain = tl.trace(_Plain(), x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(_HostSource(_SOURCES[name]), x)
    rows = _host_rows(trace)
    assert len(rows) == 1
    row = rows[0]
    assert row["count"] == 1
    assert f"torch.{name}" in str(row["message"])
    assert row["first_location"] is None or "test_hon2_host_tensor_source.py" in str(
        row["first_location"]
    )
    assert [op.label for op in trace.ops] == [
        op.label for op in tl.trace(_HostSource(_SOURCES[name]), x).ops
    ]
    assert not [w for w in caught if "host buffer" in str(w.message)]
    facts = capture_honesty_facts(trace)
    plain_facts = capture_honesty_facts(plain)
    assert trace.capture_verified is None
    assert facts["capture_status"] == plain_facts["capture_status"]
    assert facts["capture_verified"] == plain_facts["capture_verified"]
    assert facts["capture_verification_reason"] == plain_facts["capture_verification_reason"]


@pytest.mark.parametrize(
    "make",
    [
        lambda: torch.as_tensor(torch.ones(2, 2)),
        lambda: torch.ones(2, 2),
        lambda: torch.tensor(_ones()),
    ],
    ids=["as_tensor_of_tensor", "factory", "unnamed_source"],
)
def test_non_host_sources_record_nothing(make: Callable[[], torch.Tensor]) -> None:
    """A tensor argument, a factory, and a source outside the named set: no row."""

    trace = tl.trace(_HostSource(make), torch.randn(2, 2))
    assert _host_rows(trace) == []


def test_plain_capture_records_nothing_and_leaves_torch_alone() -> None:
    """A capture with no host source records nothing and patches nothing."""

    before = {name: getattr(torch, name) for name in _SOURCES}
    trace = tl.trace(_Plain(), torch.randn(2, 2))
    assert _host_rows(trace) == []
    assert {name: getattr(torch, name) for name in _SOURCES} == before


def test_host_source_outside_capture_records_nothing() -> None:
    """Calls outside a capture never reach a trace."""

    trace = tl.trace(_Plain(), torch.randn(2, 2))
    torch.from_numpy(_ones())
    assert _host_rows(trace) == []


def test_repeated_sources_aggregate_into_one_row() -> None:
    """Several calls in one forward are one row with their count and names."""

    def make() -> torch.Tensor:
        return torch.from_numpy(_ones()) + torch.as_tensor(_ones()) + torch.from_numpy(_ones())

    trace = tl.trace(_HostSource(make), torch.randn(2, 2))
    rows = _host_rows(trace)
    assert len(rows) == 1
    assert rows[0]["count"] == 3
    assert "torch.from_numpy, torch.as_tensor" in str(rows[0]["message"])


def test_advisory_renders_in_the_preamble_and_explain() -> None:
    """The shared honesty surfaces render the row next to the not-checked verdict."""

    trace = tl.trace(_HostSource(_SOURCES["from_numpy"]), torch.randn(2, 2))
    preamble = "\n".join(honesty_preamble_lines(trace))
    assert "host_tensor_source" in preamble
    assert "capture verification not checked" in preamble
    explained = trace.explain()
    assert "host_tensor_source" in explained
    assert "Capture verification: not checked" in explained
