"""U53.2: ``Trace.profile(group_by="source_line")`` over existing activation bytes.

The grouped view folds op rows into one row per user source line. It adds no
measurement, so every additive column sums to the ungrouped op view.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

pytest.importorskip("pandas")


class _TwoLineModel(nn.Module):
    """A model whose second line launches two ops (mul, add)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.fc(x))
        return y * 2 + 1  # two ops on one line


def _line_of(marker: str) -> int:
    """Return the source line of ``_TwoLineModel.forward`` holding ``marker``."""

    lines, start = inspect.getsourcelines(_TwoLineModel.forward)
    for offset, text in enumerate(lines):
        if marker in text:
            return start + offset
    raise AssertionError(f"marker {marker!r} not in forward")


@pytest.fixture(scope="module")
def two_line_trace() -> Iterator[Any]:
    torch.manual_seed(0)
    captured = tl.trace(_TwoLineModel(), torch.randn(2, 4))
    try:
        yield captured
    finally:
        captured.cleanup()


def test_grouped_activation_bytes_sum_to_the_ungrouped_view(two_line_trace) -> None:
    ungrouped = two_line_trace.profile().to_pandas()
    grouped = two_line_trace.profile(group_by="source_line").to_pandas()

    assert grouped["activation_memory"].sum() == ungrouped["activation_memory"].sum()
    assert grouped["flops"].sum() == ungrouped["flops"].sum()
    assert grouped["op_count"].sum() == len(two_line_trace._layer_list)
    assert len(grouped) < len(ungrouped)


def test_one_line_with_two_ops_is_one_row(two_line_trace) -> None:
    grouped = two_line_trace.profile(group_by="source_line").to_pandas()
    line = _line_of("two ops on one line")
    name = f"{inspect.getsourcefile(_TwoLineModel)}:{line}"

    row = grouped.loc[grouped["name"] == name]
    assert len(row) == 1
    assert int(row["op_count"].iloc[0]) == 2
    assert row["kind"].iloc[0] == "source_line"
    assert row["role"].iloc[0] == "SUBTOTAL"
    boundary = grouped.loc[grouped["kind"] == "boundary"]
    assert list(boundary["name"]) == ["<inputs and outputs>"]


def test_grouped_profile_records_its_key_and_prints(two_line_trace) -> None:
    profile = two_line_trace.profile(group_by="source_line", sort_by="activation_memory")

    assert profile.group_by == "source_line"
    assert profile.level == "op"
    assert two_line_trace.profile().group_by is None
    assert "<inputs and outputs>" in repr(profile)
    totals = profile.to_pandas(include_totals=True)
    assert totals["name"].iloc[-1] == "TOTAL"
    assert totals["time"].iloc[-1] == pytest.approx(
        two_line_trace.profile().to_pandas(include_totals=True)["time"].iloc[-1]
    )


def test_group_by_refuses_other_levels_and_keys(two_line_trace) -> None:
    with pytest.raises(ValueError, match="use level='op'"):
        two_line_trace.profile("module", group_by="source_line")
    with pytest.raises(ValueError, match="group_by must be 'source_line' or None"):
        two_line_trace.profile(group_by="file")  # type: ignore[arg-type]
