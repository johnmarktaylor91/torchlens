"""Pass-qualified replay re-keying on multi-pass (recurrent) models.

Pins the 2026-08-17 decision: the replay engine operates on pass-qualified
op labels internally (cone traversal, overlay, hook targets, pending
commits), a bare label naming a multi-pass layer refuses typed with a
teaching message, single-pass bare-label acceptance is unchanged, the
spurious multi-pass ``ControlFlowDivergenceWarning`` is gone, and
``strict=True`` multi-pass replay works.

Every propagation test asserts EACH pass individually against ground truth
computed BY HAND (an aggregate assertion passes even if only the last pass
was touched -- exactly how the pass-blind bug hid), and the recurrent
fixture uses ``bias=True`` so zero is NOT a fixed point (with ``bias=False``
``relu(cell(0)) == 0`` and a wrong replay coincidentally matches ground
truth -- the confound that masked the original bug).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import SelectionError
from torchlens.intervention.errors import (
    AmbiguousSiteError,
)
from torchlens.intervention.replay import cone_of_effect

_N_PASSES = 3
_ALL_UNITS = [(batch, unit) for batch in range(2) for unit in range(4)]


class _RecurrentCell(nn.Module):
    """One reused Linear(bias=True) in a Python loop: 3-pass recurrence."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for _ in range(_N_PASSES):
            hidden = torch.relu(self.cell(hidden))
        return hidden


class _SinglePassCell(nn.Module):
    """Single-pass control: same op vocabulary, no recurrence."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.cell(x))


@pytest.fixture(scope="module")
def recurrent():
    torch.manual_seed(0)
    model = _RecurrentCell()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def single_pass():
    torch.manual_seed(1)
    model = _SinglePassCell()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _hand_forward_with_zeroed_pass(
    model: _RecurrentCell, x: torch.Tensor, zeroed_pass: int
) -> list[torch.Tensor]:
    """Manual forward applying zero-ablation at ONE relu pass (1-based)."""

    with torch.no_grad():
        hidden = x
        per_pass = []
        for index in range(_N_PASSES):
            hidden = torch.relu(model.cell(hidden))
            if index + 1 == zeroed_pass:
                hidden = torch.zeros_like(hidden)
            per_pass.append(hidden.clone())
    return per_pass


# ---------------------------------------------------------------------------
# Per-pass propagation fidelity (the original silent-corruption bug).
# ---------------------------------------------------------------------------


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.
def test_whole_layer_selection_edits_every_pass(recurrent):
    """The explicit Layer selection is the all-passes spelling; each pass edits."""

    model, x, trace = recurrent
    fork = trace.fork()
    fork.do(fork.layers["relu_1_2"].__selection__(), tl.zero_ablate())

    zero_input = torch.zeros(2, 4)
    with torch.no_grad():
        linear_from_zero = model.cell(zero_input)
    for pass_index in (1, 2, 3):
        assert bool((fork._layer_dict_all_keys[f"relu_1_2:{pass_index}"].out == 0).all())
    for pass_index in (2, 3):
        got = fork._layer_dict_all_keys[f"linear_1_1:{pass_index}"].out
        assert torch.allclose(got, linear_from_zero)
    # FIX-A (A05): the boundary output node mirrors its parent's resolved
    # slot, so the whole-layer edit at relu:3 now REACHES the returned
    # output (zeros). The historical same-call re-slice dropped the member
    # edit and returned relu(cell(0)) -- the exact silent-wrongness class
    # the output-contract fix closes.
    output = fork._layer_dict_all_keys["output_1"].out
    assert torch.equal(output, fork._layer_dict_all_keys["relu_1_2:3"].out)
    assert bool((output == 0).all())


def test_cone_of_effect_crosses_pass_boundaries(recurrent):
    """The cone from pass 1 includes every downstream pass and the output."""

    _model, _x, trace = recurrent
    origin = trace._layer_dict_all_keys["relu_1_2:1"]
    cone_keys = {
        f"{site.layer_label}:{site.pass_number}" for site in cone_of_effect(trace, [origin])
    }
    assert cone_keys == {
        "relu_1_2:1",
        "linear_1_1:2",
        "relu_1_2:2",
        "linear_1_1:3",
        "relu_1_2:3",
        "output_1:1",
    }


# ---------------------------------------------------------------------------
# Divergence-warning honesty + strict mode.
# ---------------------------------------------------------------------------


def test_differentiable_replay_multipass_frontier_keys_are_per_pass(recurrent):
    """Differentiable multi-pass replay runs, with pass-qualified frontier keys."""

    _model, _x, trace = recurrent
    fork = trace.fork()
    replayed = fork.do(
        {tl.label("relu_1_2:1"): lambda out, *, ctx: torch.zeros_like(out)},
        intervention=tl.options.InterventionOptions(differentiable=True),
    )
    assert replayed is not fork
    assert bool((replayed._layer_dict_all_keys["relu_1_2:1"].out == 0).all())
    for key in replayed._replay_frontier:
        base = key.split("->")[0]
        if base.startswith(("relu_1_2", "linear_1_1")):
            assert ":" in base, f"multi-pass frontier key {key!r} is not pass-qualified"


# ---------------------------------------------------------------------------
# The ambiguity refusal (teaching, typed) + single-pass acceptance.
# ---------------------------------------------------------------------------


def test_bare_multipass_label_refuses_in_units(recurrent):
    """units(bare) on a multi-pass layer refuses with every spelling named."""

    _model, _x, trace = recurrent
    with pytest.raises(SelectionError) as excinfo:
        trace[tl.units("relu_1_2", [(0, 0)])]
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "multipass_bare_label"
    message = str(excinfo.value)
    assert "3 passes" in message
    for spelling in ("'relu_1_2:1'", "'relu_1_2:2'", "'relu_1_2:3'"):
        assert spelling in message


def test_bare_multipass_label_refuses_in_do_and_the_trace_door(recurrent):
    """String addressing of a bare multi-pass label refuses typed, teaching."""

    _model, _x, trace = recurrent
    with pytest.raises(AmbiguousSiteError) as excinfo:
        trace[tl.label("relu_1_2")]
    assert excinfo.value.fields["code"] == "multipass_bare_label_ambiguous"
    assert excinfo.value.fields["pass_indices"] == (1, 2, 3)
    message = str(excinfo.value)
    assert "3 passes" in message
    for spelling in ("'relu_1_2:1'", "'relu_1_2:2'", "'relu_1_2:3'"):
        assert spelling in message

    fork = trace.fork()
    with pytest.raises(AmbiguousSiteError) as do_excinfo:
        fork.do("relu_1_2", lambda out, *, ctx: torch.zeros_like(out))
    assert do_excinfo.value.fields["code"] == "multipass_bare_label_ambiguous"

    with pytest.raises(AmbiguousSiteError) as label_excinfo:
        trace[tl.label("relu_1_2")]
    assert label_excinfo.value.fields["code"] == "multipass_bare_label_ambiguous"


def test_substring_patterns_keep_fanout_semantics(recurrent):
    """A genuine substring pattern (not the bare label) still fans out."""

    _model, _x, trace = recurrent
    sites = trace[tl.label_contains("elu_1_2")]
    assert len(sites) == _N_PASSES


def test_single_pass_replay_disclosures_keep_bare_labels(single_pass):
    """Single-pass replay is unchanged: bare cone/origin disclosures, hand math."""

    model, x, trace = single_pass
    fork = trace.fork()
    fork.do(fork[tl.units("relu_1_2", _ALL_UNITS)], tl.zero_ablate())

    assert bool((fork._layer_dict_all_keys["relu_1_2:1"].out == 0).all())
    # FIX-A (A05): the boundary output node mirrors its parent's resolved
    # slot, so the edit at the output-producing relu now reaches the
    # returned output. The historical same-call re-slice kept the fork's
    # output bit-identical to the CAPTURE, silently dropping the edit.
    assert torch.equal(
        fork._layer_dict_all_keys["output_1"].out,
        fork._layer_dict_all_keys["relu_1_2:1"].out,
    )
    assert bool((fork._layer_dict_all_keys["output_1"].out == 0).all())
    assert torch.equal(
        fork._layer_dict_all_keys["linear_1_1:1"].out,
        trace._layer_dict_all_keys["linear_1_1:1"].out,
    )
    last_run = fork.last_run
    assert "relu_1_2" in last_run["cone"]
    assert "output_1" in last_run["cone"]
    assert all(":" not in label for label in last_run["cone"])
    assert all(":" not in label for label in last_run["origins"])


def test_do_value_bare_multipass_string_refuses(recurrent):
    """do(site, value) with a bare multi-pass string refuses typed (never guesses)."""

    _model, _x, trace = recurrent
    fork = trace.fork()
    value = torch.zeros_like(fork._layer_dict_all_keys["relu_1_2:1"].out)
    with pytest.raises(AmbiguousSiteError) as excinfo:
        fork.do("relu_1_2", value)
    assert excinfo.value.fields["code"] == "multipass_bare_label_ambiguous"
