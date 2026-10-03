"""W051-STOCH / AUD-CODE 3.8: the live door derives the SAME coordinate as the replay door.

Capture-time ``intervene=`` used the raw ordinal label as the firing
coordinate, so a seeded stochastic edit drew different donors live vs on
replay and the draw shifted whenever an unrelated upstream op was added. The
live door now stamps the L1 site key (and module-call pass) on its site
proxy. The donor-group id also hashes datum CONTENT, never ``repr``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention import OneDatum, reference, sample_from, sampling_records
from torchlens.intervention.stochastic import _mint_donor_group_id


class _TwoBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.b(torch.relu(self.a(x))))


class _TwoBlockWithPrologue(_TwoBlock):
    """Same relu sites, one unrelated upstream op added (the audit's shift scenario)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.tanh(x))


def _coordinates(trace: tl.Trace) -> list[tuple[tuple, list[int]]]:
    return [
        (tuple(row["logical_firing_coordinate"]), row["donor_ids"])
        for row in sampling_records(trace)
        if row["share_draw"] == "per_firing"
    ]


def test_live_door_matches_replay_door() -> None:
    """Same plan, same seed: capture-time intervene= and fork.do() draw identically."""

    torch.manual_seed(0)
    model = _TwoBlock()
    torch.manual_seed(100)
    x = torch.randn(5, 4)
    options = tl.options.CaptureOptions(intervention_ready=True, random_seed=100)
    log = tl.trace(model, x, capture=options)
    baseline = log["relu_1_2"].out
    donors = reference(torch.stack([baseline + k for k in range(1, 6)]), origin="d")
    plan = sample_from(donors, seed=3)

    fork = log.fork()
    # One batch over the two explicit sites: the same fan-out as tl.func("relu")
    # without the multi-match disclosure warning the suite escalates.
    fork.do(
        [
            (log["relu_1_2"].__selection__(), tl.patch_from(plan)),
            (log["relu_2_4"].__selection__(), tl.patch_from(plan)),
        ]
    )
    replay = _coordinates(fork)
    assert len(replay) == 2

    live = tl.trace(
        model, x, capture=options, intervene=tl.when(tl.func("relu"), tl.patch_from(plan))
    )
    assert _coordinates(live) == replay
    for coordinate, _donors in replay:
        site_key, pass_index, _step = coordinate
        assert isinstance(site_key, str) and site_key.startswith("s1|")
        assert pass_index == 1
    assert {live["relu_1_2"].site_key, live["relu_2_4"].site_key} == {
        coordinate[0] for coordinate, _d in replay
    }, "the stamped live key IS the key postprocess mints"


def test_live_coordinate_is_stable_under_unrelated_upstream_op() -> None:
    """Adding a tanh before the blocks moves every raw ordinal; the site keys do not."""

    torch.manual_seed(0)
    plain = _TwoBlock()
    with_prologue = _TwoBlockWithPrologue()
    with_prologue.load_state_dict(plain.state_dict())
    torch.manual_seed(100)
    x = torch.randn(5, 4)
    donors = reference(torch.randn(5, 5, 4), origin="d")
    plan = sample_from(donors, seed=3)
    options = tl.options.CaptureOptions(random_seed=100)
    live_plain = tl.trace(
        plain, x, capture=options, intervene=tl.when(tl.func("relu"), tl.patch_from(plan))
    )
    live_prologue = tl.trace(
        with_prologue,
        x,
        capture=options,
        intervene=tl.when(tl.func("relu"), tl.patch_from(plan)),
    )
    assert _coordinates(live_plain) == _coordinates(live_prologue)
    labels_plain = [op.label for op in live_plain.ops if op.layer_type == "relu"]
    labels_prologue = [op.label for op in live_prologue.ops if op.layer_type == "relu"]
    assert labels_plain != labels_prologue, "the ordinal labels DID shift; the coordinate did not"


@pytest.mark.smoke
def test_donor_group_id_hashes_datum_content_not_repr() -> None:
    """Two tensors with one repr mint two group ids; equal content mints one."""

    close_a = torch.tensor([1.0000001])
    close_b = torch.tensor([1.0000002])
    assert repr(close_a) == repr(close_b) and not torch.equal(close_a, close_b)
    key = lambda datum: datum  # noqa: E731 -- presence marker only
    group_a = _mint_donor_group_id("pop", 1, "per_firing", close_a, key)
    group_b = _mint_donor_group_id("pop", 1, "per_firing", close_b, key)
    assert group_a != group_b
    assert group_a == _mint_donor_group_id("pop", 1, "per_firing", close_a.clone(), key)
    # Wrapped and nested datums hash by content too (OneDatum / dict / tuple).
    wrapped = _mint_donor_group_id("pop", 1, "per_firing", OneDatum({"t": (close_a, "A")}), key)
    assert wrapped == _mint_donor_group_id(
        "pop", 1, "per_firing", OneDatum({"t": (close_a.clone(), "A")}), key
    )
    assert wrapped != _mint_donor_group_id(
        "pop", 1, "per_firing", OneDatum({"t": (close_b, "A")}), key
    )
