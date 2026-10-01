"""Intervention honesty: editing a DENSE direction's support set discloses at
the point of use (list-A row 8).

``tl.subspace`` is a SET producer: a dense direction supports the whole bound
axis, so ``do(subspace(dense), edit)`` performs full-axis ablation -- the
documented semantics, but the producer's own motivating use cases expect a
projection. The adjudicated fix (projection fork rides list D): a dense-basis
resolution stamps an explicit marker into ``provenance.source`` (and thus the
``do()`` audit record), and ``do()`` fires a point-of-use warning.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors._base import TorchLensWarning
from torchlens.selection_subspace import DENSE_SUPPORT_NOTE

pytestmark = pytest.mark.smoke

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


def test_dense_direction_do_warns_and_stamps_the_audit() -> None:
    log = _trace()
    fork = log.fork()
    selection = tl.subspace(
        "relu_1_2", torch.ones(_D), origin="dense steering vector", method="manual"
    )
    with pytest.warns(TorchLensWarning, match="EVERY element") as caught:
        fork.do(selection, tl.zero_ablate())
    codes = {
        getattr(item.message, "fields", {}).get("code")
        for item in caught
        if isinstance(item.message, TorchLensWarning)
    }
    assert "dense_subspace_full_axis_edit" in codes
    remedies = {
        getattr(item.message, "fields", {}).get("remedy")
        for item in caught
        if isinstance(item.message, TorchLensWarning)
    }
    assert any(remedy for remedy in remedies), "the contracted warning carries a remedy"
    assert bool((fork["relu_1_2"].out == 0).all())
    audit = fork.intervention_audit[-1]
    assert DENSE_SUPPORT_NOTE.strip(" []") not in ("",)
    assert any(DENSE_SUPPORT_NOTE in str(site) for site in audit["sites"]) or (
        DENSE_SUPPORT_NOTE in repr(audit)
    )


def test_dense_resolution_source_carries_the_marker() -> None:
    log = _trace()
    resolved = tl.subspace("relu_1_2", torch.ones(_D), origin="dense probe").resolve(log)
    assert all(DENSE_SUPPORT_NOTE in entry.provenance.source for entry in resolved)


def test_sparse_direction_do_stays_silent() -> None:
    import warnings

    log = _trace()
    fork = log.fork()
    selection = tl.subspace("relu_1_2", _sparse(), origin="sparse probe")
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        fork.do(selection, tl.zero_ablate())
    mask = selection.resolve(log)[0].mask
    assert bool((fork["relu_1_2"].out[mask] == 0).all())
    assert torch.equal(fork["relu_1_2"].out[~mask], log["relu_1_2"].out[~mask])
