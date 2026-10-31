"""Intervention honesty: ``units()`` partial-rank refusal teaches positions and
the unseeded-RNG disclosure stops leaking internal ``_raw`` labels (edits memo
row 0c).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.selection import SelectionError


class _ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c1(x))


def _log(*, intervention_ready: bool = False) -> tl.Trace:
    torch.manual_seed(0)
    capture = tl.options.CaptureOptions(intervention_ready=intervention_ready)
    return tl.trace(_ConvNet().eval(), torch.randn(2, 1, 4, 4), capture=capture)


def test_partial_rank_unit_index_teaches_full_coordinates() -> None:
    """A too-short coordinate refuses with a rank-teaching message, never the
    misleading 'outside the output space' spelling."""

    log = _log()
    with pytest.raises(SelectionError) as excinfo:
        tl.units("relu_1_2", [(1,)]).resolve(log)
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"
    message = str(excinfo.value)
    assert "rank" in message
    assert "one integer per output axis" in message
    assert "bool mask" in message
    assert "outside" not in message
    assert excinfo.value.fields["given_rank"] == 1
    assert excinfo.value.fields["expected_rank"] == 4


def test_full_rank_out_of_range_message_unchanged() -> None:
    """The out-of-range case keeps its own message (a different mistake)."""

    log = _log()
    with pytest.raises(SelectionError) as excinfo:
        tl.units("relu_1_2", [(9, 9, 9, 9)]).resolve(log)
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"
    assert "outside" in str(excinfo.value)


def test_live_unseeded_note_never_leaks_raw_labels() -> None:
    """The live-door disclosure drops the internal ``_raw`` spelling: it names
    the helper, and the FireRecord's attachment op carries the location."""

    torch.manual_seed(0)
    model = _ConvNet().eval()
    log = tl.trace(
        model,
        torch.randn(2, 1, 4, 4),
        intervene=tl.when(tl.func("relu"), tl.intervention.scramble_elements(torch.ones(3))),
    )
    notes = [
        record.determinism_note
        for record in (log["relu_1_2"].interventions or [])
        if record.determinism_note
    ]
    assert notes, "unseeded scramble must disclose nondeterminism"
    for note in notes:
        assert "_raw" not in note
        assert "scramble_elements used unseeded stochastic RNG" in note


def test_note_keeps_public_labels_and_drops_internal_ones() -> None:
    """The disclosure names the site iff the fire-time spelling is public.

    (The replay-door FireRecord builders do not lift notes yet -- the shared
    builder + notes lift is the intervention-substrate lane's row -- so the
    public-label branch is pinned at the unit level here.)
    """

    from torchlens.intervention.helpers import _enqueue_nondeterminism_note
    from torchlens.intervention.hooks import make_hook_context

    public = make_hook_context(
        name="scramble_elements",
        layer_log={"layer_label": "relu_1_2", "label": "relu_1_2:1"},
        run_ctx={},
    )
    _enqueue_nondeterminism_note(public, "scramble_elements")
    assert public.run_ctx["ledger_notes"] == [
        "scramble_elements used unseeded stochastic RNG at relu_1_2:1"
    ]

    raw_only = make_hook_context(
        name="scramble_elements",
        layer_log={"layer_label": "relu_1_3_raw", "label": "relu_1_3_raw"},
        run_ctx={},
    )
    _enqueue_nondeterminism_note(raw_only, "scramble_elements")
    assert raw_only.run_ctx["ledger_notes"] == ["scramble_elements used unseeded stochastic RNG"]
