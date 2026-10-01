"""W051 regression: a re-executed episode product must round-trip save/load.

Audit finding AUD-CODE 1.1: ``run()`` on an episode capture dropped the
ledger to a travel note (capture_kind read ``plain``) but left every
``Op.episode_step`` stamp in place, so ``tl.save`` succeeded and ``tl.load``
refused ``artifact_episode_step_invalid`` -- an unloadable artifact minted
from the user's own product. The fix scrubs the stamps in the travel policy;
the load gate is deliberately UNCHANGED (a stamp without a declaration is
still a forged or drifted artifact).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._annotations_travel import (
    scrub_episode_step_stamps,
    scrub_fresh_execution_annotations,
)
from torchlens.errors import TorchLensError
from torchlens.intervention import at_step
from torchlens.options import EpisodeSpec

pytestmark = pytest.mark.smoke

V = 16
N_STEPS = 3


class _Step(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, V)
        self.out = nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.eye(V))
            self.out.weight.copy_(torch.roll(torch.eye(V), shifts=1, dims=0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids[:, -1]))


class _Greedy(nn.Module):
    def __init__(self, n: int = N_STEPS) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
        return ids[:, -self.n :]


def _stamped(trace: tl.Trace) -> list[str]:
    return [str(op.label) for op in trace.layer_list if op.episode_step is not None]


def _episode(model: _Greedy, **extra) -> tl.Trace:
    return tl.trace(
        model,
        torch.tensor([[1]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
        **extra,
    )


def test_run_product_round_trips_save_load(tmp_path) -> None:
    """The non-fast provider: run -> save -> load must succeed."""

    model = _Greedy()
    trace = _episode(model)
    assert _stamped(trace)  # the source capture IS stamped
    result = trace.run(inputs=torch.tensor([[9]]))
    product = result.trace
    assert product is not trace
    assert product.capture_kind == "plain"
    assert _stamped(product) == []
    note = product.annotations["episode"]
    assert note["code"] == "episode_evidence_dropped_fresh_execution"
    assert "episode_step stamps were cleared" in note["detail"]
    # The SOURCE keeps its own stamps and ledger untouched.
    assert _stamped(trace)
    assert "rows" in trace.annotations["episode"]

    path = tmp_path / "product.tlspec"
    tl.save(product, path)
    loaded = tl.load(path)
    assert loaded.capture_kind == "plain"
    assert _stamped(loaded) == []


def test_guarded_fast_live_product_round_trips_save_load(tmp_path) -> None:
    """The guarded-fast LIVE path refreshes the user's own trace in place."""

    model = _Greedy()
    trace = _episode(model, save=tl.func("cat"))
    assert _stamped(trace)
    result = trace.run(inputs=torch.tensor([[1]]), fast=True)
    assert result.trace is trace
    assert trace.capture_kind == "plain"
    assert _stamped(trace) == []

    path = tmp_path / "fast.tlspec"
    tl.save(trace, path)
    loaded = tl.load(path)
    assert loaded.capture_kind == "plain"
    assert _stamped(loaded) == []


def test_at_step_never_answers_from_stale_stamps() -> None:
    """Post hoc step selection refuses on the fresh product (no declaration)."""

    model = _Greedy()
    trace = _episode(model)
    product = trace.run(inputs=torch.tensor([[9]])).trace
    with pytest.raises(TorchLensError) as info:
        product.resolve_sites(at_step(1))
    assert info.value.fields["code"] == "episode_step_selector_without_episode"


def test_scrub_helper_counts_and_is_idempotent() -> None:
    model = _Greedy()
    trace = _episode(model)
    n_stamped = len(_stamped(trace))
    assert n_stamped > 0
    fork = trace.fork()
    assert scrub_episode_step_stamps(fork) == n_stamped
    assert scrub_episode_step_stamps(fork) == 0
    assert _stamped(fork) == []
    # The policy entry clears stamps even when the ledger key is absent.
    fork2 = trace.fork()
    del fork2.annotations["episode"]
    assert scrub_fresh_execution_annotations(fork2) == ()
    assert _stamped(fork2) == []


def test_load_gate_still_refuses_stamps_without_declaration(tmp_path) -> None:
    """Tripwire law: the fix scrubs stamps; it never relaxes the load gate."""

    plain = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(1, 4))
    plain.layer_list[0].episode_step = 0
    path = tmp_path / "forged.tlspec"
    tl.save(plain, path)
    with pytest.raises(TorchLensError) as info:
        tl.load(path)
    assert info.value.fields["code"] == "artifact_episode_step_invalid"
