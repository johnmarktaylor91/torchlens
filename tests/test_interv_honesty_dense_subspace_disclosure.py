"""Intervention honesty: a DENSE direction's subspace edit discloses what ran.

``tl.subspace`` resolves to a SET (a dense direction supports the whole bound
axis) while ``do(subspace(...), edit)`` edits the COMPONENT along the basis
span. Both facts ride the audit record: the resolved source keeps the
whole-axis support marker, and every subspace ``do()`` stamps the projection
marker, so a reader can always tell which meaning ran.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._selection_subspace import DENSE_SUPPORT_NOTE
from torchlens.errors._base import TorchLensWarning
from torchlens.intervention._subspace_projection import PROJECTION_NOTE

_D = 8


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(_D, _D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        _Mlp().eval(),
        torch.randn(2, _D),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )


def _sparse() -> torch.Tensor:
    direction = torch.zeros(_D)
    direction[2] = 1.0
    return direction


def _unit_dense() -> torch.Tensor:
    return torch.ones(_D, dtype=torch.float32) / _D**0.5


def test_dense_direction_do_edits_the_component_and_stamps_both_markers() -> None:
    log = _trace()
    fork = log.fork()
    direction = _unit_dense()
    selection = tl.subspace("relu_1_2", direction, origin="dense steering vector", method="manual")
    fork.do(selection, tl.zero_ablate())
    before = log["relu_1_2"].out
    after = fork["relu_1_2"].out
    want = before - (before @ direction)[:, None] * direction[None, :]
    torch.testing.assert_close(after, want, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(after @ direction, torch.zeros(2), rtol=0.0, atol=1e-6)
    assert not bool((after == 0).all()), "a dense direction no longer zeroes the whole axis"
    (site,) = fork.intervention_audit[-1]["sites"]
    assert DENSE_SUPPORT_NOTE in site["source"]
    assert PROJECTION_NOTE in site["source"]


@pytest.mark.smoke
def test_dense_resolution_source_carries_the_marker() -> None:
    log = _trace()
    resolved = tl.subspace("relu_1_2", torch.ones(_D), origin="dense probe").resolve(log)
    assert all(DENSE_SUPPORT_NOTE in entry.provenance.source for entry in resolved)
    assert all(bool(entry.mask.all()) for entry in resolved)


def test_sparse_direction_do_stamps_the_projection_and_edits_only_its_support() -> None:
    log = _trace()
    fork = log.fork()
    selection = tl.subspace("relu_1_2", _sparse(), origin="sparse probe")
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        fork.do(selection, tl.zero_ablate())
    mask = selection.resolve(log)[0].mask
    assert bool((fork["relu_1_2"].out[mask] == 0).all())
    assert torch.equal(fork["relu_1_2"].out[~mask], log["relu_1_2"].out[~mask])
    (site,) = fork.intervention_audit[-1]["sites"]
    assert PROJECTION_NOTE in site["source"]
    assert DENSE_SUPPORT_NOTE not in site["source"]
