"""A selection resolved on a trace carries over to that trace's forks.

``trace.fork().do(trace["layer4"].receptive_field.of(units, within=[...]),
tl.zero_ablate())`` is the announced receptive-field spelling: the selection
is resolved on the parent and applied on a fork. A fork shares its parent's
sites, so ``do()`` binds the selection to the fork by site identity. A
selection resolved on a trace outside the fork's ancestry (an unrelated
capture, a sibling fork, or the fork itself used on its parent) still refuses
with ``selection_trace_mismatch``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import SelectionError

#: Two layer4 units whose fields are traced back into layer2 and layer3.
UNITS = [(0, 1, 3, 3), (0, 2, 3, 3)]
REGIONS = ("layer2", "layer3")


class _ThreeStageCNN(nn.Module):
    """Small CNN with ``layer2``/``layer3``/``layer4`` stages (distinct widths)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(6, 8, 3, padding=1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(8, 5, 3, padding=1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the four stages in order."""

        return self.layer4(self.layer3(self.layer2(self.stem(x))))


def _trace(seed: int) -> tl.Trace:
    """Capture the model on one seeded 8x8 input with default options."""

    torch.manual_seed(0)
    model = _ThreeStageCNN().eval()
    torch.manual_seed(seed)
    return tl.trace(model, torch.randn(1, 3, 8, 8))


@pytest.fixture
def trace():
    """One default capture of the three-stage CNN."""

    captured = _trace(1)
    yield captured
    captured.cleanup()


def _rows_by_region(trace: tl.Trace, selection: tl.Selection) -> dict[str, torch.Tensor]:
    """Map each region to its row's mask (the stage widths make shapes unique)."""

    masks = {tuple(row.shape): row.mask for row in selection}
    return {region: masks[tuple(trace[region].out.shape)] for region in REGIONS}


def test_parent_receptive_field_selection_ablates_on_fork(trace):
    """The announced line runs, edits only the fork, and zeroes the field."""

    parent_out = {name: trace[name].out.clone() for name in (*REGIONS, "layer4")}
    selection = trace["layer4"].receptive_field.of(UNITS, within=["layer2", "layer3"])
    masks = _rows_by_region(trace, selection)

    fork = trace.fork().do(
        trace["layer4"].receptive_field.of(UNITS, within=["layer2", "layer3"]),
        tl.zero_ablate(),
    )

    for region, mask in masks.items():
        assert mask.any(), f"{region}: the field selected nothing"
        assert parent_out[region][mask].ne(0).any(), f"{region}: nothing to ablate"
        assert fork[region].out[mask].eq(0).all(), f"{region}: field not zeroed on the fork"
    assert torch.equal(fork["layer2"].out[~masks["layer2"]], parent_out["layer2"][~masks["layer2"]])
    assert not torch.equal(fork["layer4"].out, parent_out["layer4"])
    for name, before in parent_out.items():
        assert torch.equal(trace[name].out, before), f"{name}: the parent changed"
    # The parent's selection and its rows stay bound to the parent.
    assert selection._trace is trace
    assert all(row._trace_ref() is trace for row in selection)
    fork.cleanup()


def test_selection_from_outside_the_fork_lineage_refuses(trace):
    """Unrelated captures, sibling forks and the reverse direction keep refusing."""

    other = _trace(2)
    try:
        foreign = other["layer4"].receptive_field.of(UNITS, within=["layer2", "layer3"])
        with pytest.raises(SelectionError) as excinfo:
            trace.fork().do(foreign, tl.zero_ablate())
        assert excinfo.value.fields["code"] == "selection_trace_mismatch"
        assert "align_to" in str(excinfo.value)
    finally:
        other.cleanup()

    # A sibling fork is not an ancestor, and binding never runs child-to-parent.
    first, second = trace.fork(), trace.fork()
    grandchild = first.fork()
    on_first = first["layer4"].receptive_field.of(UNITS, within=["layer2", "layer3"])
    on_grandchild = grandchild["layer4"].receptive_field.of(UNITS, within=["layer2", "layer3"])
    for target, selection in ((second, on_first), (first, on_grandchild)):
        with pytest.raises(SelectionError) as excinfo:
            target.do(selection, tl.zero_ablate())
        assert excinfo.value.fields["code"] == "selection_trace_mismatch"
    for fork in (grandchild, first, second):
        fork.cleanup()
