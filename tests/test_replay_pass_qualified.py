"""Pass-qualified replay re-keying on multi-pass (recurrent) models.

Pins the JMT 2026-08-17 ruling: the replay engine operates on pass-qualified
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

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.errors import (
    ControlFlowDivergenceWarning,
    SiteAmbiguityError,
)
from torchlens.intervention.replay import cone_of_effect
from torchlens.selection import SelectionError

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
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def single_pass():
    torch.manual_seed(1)
    model = _SinglePassCell()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
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


def test_zero_ablate_pass1_per_pass_ground_truth(recurrent):
    """Zeroing relu pass 1 zeroes THAT pass and recomputes 2/3 correctly."""

    model, x, trace = recurrent
    expected = _hand_forward_with_zeroed_pass(model, x, zeroed_pass=1)
    fork = trace.fork()
    fork.do(tl.units("relu_1_2:1", _ALL_UNITS).resolve(fork), tl.zero_ablate())

    for pass_index in (1, 2, 3):
        got = fork.layer_dict_all_keys[f"relu_1_2:{pass_index}"].out
        captured = trace.layer_dict_all_keys[f"relu_1_2:{pass_index}"].out
        assert torch.allclose(got, expected[pass_index - 1]), f"pass {pass_index} wrong"
        # bias=True makes every recomputed pass differ from the captured one.
        assert not torch.allclose(got, captured), f"pass {pass_index} stale"
    assert bool((fork.layer_dict_all_keys["relu_1_2:1"].out == 0).all())
    output = fork.layer_dict_all_keys["output_1"].out
    assert torch.allclose(output, expected[-1])
    assert not bool((output == 0).all())


def test_zero_ablate_middle_pass_leaves_pass1_untouched(recurrent):
    """Editing pass 2 recomputes 2/3 only; pass 1 stays bit-identical."""

    model, x, trace = recurrent
    expected = _hand_forward_with_zeroed_pass(model, x, zeroed_pass=2)
    fork = trace.fork()
    fork.do(tl.units("relu_1_2:2", _ALL_UNITS).resolve(fork), tl.zero_ablate())

    pass1 = fork.layer_dict_all_keys["relu_1_2:1"].out
    assert torch.equal(pass1, trace.layer_dict_all_keys["relu_1_2:1"].out)
    assert torch.allclose(pass1, expected[0])
    assert bool((fork.layer_dict_all_keys["relu_1_2:2"].out == 0).all())
    pass3 = fork.layer_dict_all_keys["relu_1_2:3"].out
    assert torch.allclose(pass3, expected[2])
    assert not torch.allclose(pass3, trace.layer_dict_all_keys["relu_1_2:3"].out)
    assert torch.allclose(fork.layer_dict_all_keys["output_1"].out, expected[-1])


def test_whole_layer_selection_edits_every_pass(recurrent):
    """The explicit Layer selection is the all-passes spelling; each pass edits."""

    model, x, trace = recurrent
    fork = trace.fork()
    fork.do(fork["relu_1_2"].__selection__(), tl.zero_ablate())

    zero_input = torch.zeros(2, 4)
    with torch.no_grad():
        linear_from_zero = model.cell(zero_input)
    for pass_index in (1, 2, 3):
        assert bool((fork.layer_dict_all_keys[f"relu_1_2:{pass_index}"].out == 0).all())
    for pass_index in (2, 3):
        got = fork.layer_dict_all_keys[f"linear_1_1:{pass_index}"].out
        assert torch.allclose(got, linear_from_zero)
    # The output record shares relu:3's call and re-slices the RAW call
    # output, so it reflects the recomputed upstream (relu:2 zeroed -> the
    # call recomputes relu(cell(0))) but not relu:3's own member hook --
    # pre-existing shipped same-call semantics, identical on single-pass
    # models on main, out of this lane's scope. The per-pass recompute is
    # what this test pins.
    output = fork.layer_dict_all_keys["output_1"].out
    assert torch.allclose(output, torch.relu(linear_from_zero))


def test_hook_fires_only_at_targeted_pass(recurrent):
    """A hook addressed to ONE pass fires exactly once, at that pass."""

    _model, _x, trace = recurrent
    fork = trace.fork()
    fired: list[str] = []

    def spy_hook(out: torch.Tensor, *, hook) -> torch.Tensor:
        fired.append(str(hook.layer_log.get("label")))
        return torch.zeros_like(out)

    fork.do("relu_1_2:1", spy_hook)

    assert fired == ["relu_1_2:1"]
    assert bool((fork.layer_dict_all_keys["relu_1_2:1"].out == 0).all())
    assert not bool((fork.layer_dict_all_keys["relu_1_2:2"].out == 0).all())
    assert fork.layer_dict_all_keys["relu_1_2:1"].interventions
    assert not fork.layer_dict_all_keys["relu_1_2:2"].interventions
    assert not fork.layer_dict_all_keys["relu_1_2:3"].interventions


def test_cone_of_effect_crosses_pass_boundaries(recurrent):
    """The cone from pass 1 includes every downstream pass and the output."""

    _model, _x, trace = recurrent
    origin = trace.layer_dict_all_keys["relu_1_2:1"]
    cone_keys = {
        f"{site.layer_label}:{site.pass_index}" for site in cone_of_effect(trace, [origin])
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


def test_no_spurious_divergence_warning_on_multipass_replay(recurrent):
    """A plain multi-pass replay emits no ControlFlowDivergenceWarning."""

    _model, _x, trace = recurrent
    fork = trace.fork()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fork.do(tl.units("relu_1_2:1", _ALL_UNITS).resolve(fork), tl.zero_ablate())
    divergence = [
        entry for entry in caught if issubclass(entry.category, ControlFlowDivergenceWarning)
    ]
    assert divergence == []


def test_strict_multipass_replay_works(recurrent):
    """strict=True multi-pass replay succeeds and matches hand ground truth."""

    model, x, trace = recurrent
    expected = _hand_forward_with_zeroed_pass(model, x, zeroed_pass=1)
    fork = trace.fork()
    fork.do(
        tl.units("relu_1_2:1", _ALL_UNITS).resolve(fork),
        tl.zero_ablate(),
        intervention=tl.options.InterventionOptions(strict=True),
    )
    for pass_index in (1, 2, 3):
        got = fork.layer_dict_all_keys[f"relu_1_2:{pass_index}"].out
        assert torch.allclose(got, expected[pass_index - 1])
    assert torch.allclose(fork.layer_dict_all_keys["output_1"].out, expected[-1])


def test_differentiable_replay_multipass_frontier_keys_are_per_pass(recurrent):
    """Differentiable multi-pass replay runs, with pass-qualified frontier keys."""

    _model, _x, trace = recurrent
    fork = trace.fork()
    replayed = fork.push(
        replay=tl.options.ReplayOptions(
            differentiable=True,
            hooks={tl.label("relu_1_2:1"): lambda out, *, hook: torch.zeros_like(out)},
        ),
    )
    assert replayed is not fork
    assert bool((replayed.layer_dict_all_keys["relu_1_2:1"].out == 0).all())
    for key in replayed.replay_frontier:
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
        tl.units("relu_1_2", [(0, 0)]).resolve(trace)
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "multipass_bare_label"
    message = str(excinfo.value)
    assert "3 passes" in message
    for spelling in ("'relu_1_2:1'", "'relu_1_2:2'", "'relu_1_2:3'"):
        assert spelling in message


def test_bare_multipass_label_refuses_in_do_and_resolve_sites(recurrent):
    """String addressing of a bare multi-pass label refuses typed, teaching."""

    _model, _x, trace = recurrent
    with pytest.raises(SiteAmbiguityError) as excinfo:
        trace.resolve_sites("relu_1_2", max_fanout=100)
    assert excinfo.value.fields["code"] == "multipass_bare_label_ambiguous"
    assert excinfo.value.fields["pass_indices"] == (1, 2, 3)
    message = str(excinfo.value)
    assert "3 passes" in message
    for spelling in ("'relu_1_2:1'", "'relu_1_2:2'", "'relu_1_2:3'"):
        assert spelling in message

    fork = trace.fork()
    with pytest.raises(SiteAmbiguityError) as do_excinfo:
        fork.do("relu_1_2", lambda out, *, hook: torch.zeros_like(out))
    assert do_excinfo.value.fields["code"] == "multipass_bare_label_ambiguous"

    with pytest.raises(SiteAmbiguityError) as label_excinfo:
        trace.resolve_sites(tl.label("relu_1_2"), max_fanout=100)
    assert label_excinfo.value.fields["code"] == "multipass_bare_label_ambiguous"


def test_substring_patterns_keep_fanout_semantics(recurrent):
    """A genuine substring pattern (not the bare label) still fans out."""

    from torchlens.intervention.errors import MultiMatchWarning

    _model, _x, trace = recurrent
    with pytest.warns(MultiMatchWarning):
        sites = trace.resolve_sites("elu_1_2", max_fanout=100)
    assert len(list(sites)) == _N_PASSES


def test_single_pass_bare_label_still_accepted(single_pass):
    """Bare labels keep working everywhere the layer is single-pass."""

    model, x, trace = single_pass
    resolved = tl.units("relu_1_2", [(0, 0)]).resolve(trace)
    assert len(tuple(resolved)) == 1
    sites = trace.resolve_sites("relu_1_2")
    assert len(list(sites)) == 1

    with torch.no_grad():
        expected = torch.zeros_like(torch.relu(model.cell(x)))
    fork = trace.fork()
    fork.do("relu_1_2", lambda out, *, hook: torch.zeros_like(out))
    assert torch.equal(fork.layer_dict_all_keys["relu_1_2:1"].out, expected)


def test_single_pass_replay_disclosures_keep_bare_labels(single_pass):
    """Single-pass replay is unchanged: bare cone/origin disclosures, hand math."""

    model, x, trace = single_pass
    fork = trace.fork()
    fork.do(tl.units("relu_1_2", _ALL_UNITS).resolve(fork), tl.zero_ablate())

    assert bool((fork.layer_dict_all_keys["relu_1_2:1"].out == 0).all())
    # The output record shares relu's call and re-slices the raw call
    # output (upstream unchanged here), so it stays bit-identical to the
    # capture -- pre-existing shipped same-call semantics, pinned as the
    # single-pass regression proof.
    assert torch.equal(
        fork.layer_dict_all_keys["output_1"].out,
        trace.layer_dict_all_keys["output_1"].out,
    )
    assert torch.equal(
        fork.layer_dict_all_keys["linear_1_1:1"].out,
        trace.layer_dict_all_keys["linear_1_1:1"].out,
    )
    last_run = fork.last_run
    assert "relu_1_2" in last_run["cone"]
    assert "output_1" in last_run["cone"]
    assert all(":" not in label for label in last_run["cone"])
    assert all(":" not in label for label in last_run["origins"])


def test_push_from_bare_multipass_string_refuses(recurrent):
    """push_from with a bare multi-pass string refuses typed (never guesses)."""

    _model, _x, trace = recurrent
    fork = trace.fork()
    with pytest.raises(SiteAmbiguityError) as excinfo:
        fork.push_from("relu_1_2")
    assert excinfo.value.fields["code"] == "multipass_bare_label_ambiguous"


def test_push_from_pass_qualified_op_recomputes_downstream_passes(recurrent):
    """push_from(op) on pass 1 propagates through passes 2/3 with hand math."""

    model, x, trace = recurrent
    expected = _hand_forward_with_zeroed_pass(model, x, zeroed_pass=1)
    fork = trace.fork()
    origin = fork.layer_dict_all_keys["relu_1_2:1"]
    origin._internal_set("out", torch.zeros_like(origin.out))
    fork.push_from(origin)

    for pass_index in (1, 2, 3):
        got = fork.layer_dict_all_keys[f"relu_1_2:{pass_index}"].out
        assert torch.allclose(got, expected[pass_index - 1]), f"pass {pass_index} wrong"
    assert torch.allclose(fork.layer_dict_all_keys["output_1"].out, expected[-1])
