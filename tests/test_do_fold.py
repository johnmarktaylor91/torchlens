"""``do`` as the one editing verb: the fold of push, push_from and attach_hooks.

Pins the contracts the fold introduced rather than the spellings it removed:

* a mapping ``do`` applies every edit in ONE propagation, with a fire receipt
  per edited site;
* ``InterventionOptions(engine="set_only")`` stages without propagating, and a
  later propagating ``do`` carries the staged edit along;
* ``InterventionOptions(differentiable=True)`` replays onto a new fork that
  keeps the autograd graph and leaves the source untouched, and the rerun and
  set-only engines ignore the flag;
* ``ReplayOptions`` no longer takes ``hooks=`` or ``differentiable=``, and the
  new ``InterventionOptions.differentiable`` field rides the flat/grouped merge;
* the region and edge-substitution doors still push downstream through the
  renamed internal ``replay.replay_from_op``.
"""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import ArgumentConflictError
from torchlens.options import (
    _INTERVENTION_FIELDS,
    _INTERVENTION_FLAT_TO_GROUP,
    _REPLAY_FIELDS,
    InterventionOptions,
    ReplayOptions,
    merge_intervention_options,
)

replay_module = importlib.import_module("torchlens.intervention.replay")

_CAPTURE = tl.options.CaptureOptions(replayable=True, save_args=True)
_GRAD_CAPTURE = tl.options.CaptureOptions(save_grads="all", replayable=True, backward_ready=True)
_SET_ONLY = InterventionOptions(engine="set_only")


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh -> fc3, labelled linear_1_1 .. linear_3_5."""

    def __init__(self) -> None:
        """Build the three linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return self.fc3(torch.tanh(self.fc2(torch.relu(self.fc1(x)))))


def _chain(capture: Any = _CAPTURE) -> tuple[_Chain, torch.Tensor, tl.Trace]:
    """Capture the chain on a fixed input."""

    torch.manual_seed(0)
    model = _Chain().eval()
    x = torch.randn(3, 4)
    return model, x, tl.trace(model, x, capture=capture)


def _hand(model: _Chain, x: torch.Tensor, *, relu: float = 1.0, tanh: float = 1.0) -> torch.Tensor:
    """The chain's output with its relu and tanh outputs scaled by hand."""

    with torch.no_grad():
        hidden = torch.tanh(model.fc2(torch.relu(model.fc1(x)) * relu)) * tanh
        return model.fc3(hidden)


def _double(out: torch.Tensor, *, ctx: Any = None) -> torch.Tensor:
    """Double an activation (a plain callable edit)."""

    del ctx
    return out * 2.0


def _events(trace: tl.Trace) -> list[dict[str, Any]]:
    """The persisted ``do`` transaction envelopes, oldest first."""

    return [
        row
        for row in trace.state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event" and row["door"] == "do"
    ]


def _count_calls(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> Counter[str]:
    """Wrap ``owner.name`` so every call is counted; return the live counter."""

    counts: Counter[str] = Counter()
    inner: Callable[..., Any] = getattr(owner, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        counts[name] += 1
        return inner(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return counts


# ---------------------------------------------------------------------------
# 1. a mapping do: every edit, one propagation, a receipt per site
# ---------------------------------------------------------------------------


def test_mapping_do_applies_both_edits_in_one_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``do({a: edit_a, b: edit_b})`` replays once and both edits fire."""

    model, x, trace = _chain()
    fork = trace.fork()
    replays = _count_calls(monkeypatch, replay_module, "_run_replay")
    events_before = len(_events(fork))

    returned = fork.do({"relu_1_2": tl.scale(2.0), "tanh_1_4": tl.scale(0.5)})

    assert returned is fork
    assert replays["_run_replay"] == 1
    assert torch.allclose(fork["linear_3_5"].out, _hand(model, x, relu=2.0, tanh=0.5), atol=1e-6)
    assert torch.allclose(fork["relu_1_2"].out, torch.relu(model.fc1(x)) * 2.0, atol=1e-6)
    # One transaction envelope, two fires, and a FireRecord on each edited site.
    (event,) = _events(fork)[events_before:]
    assert event["status"] == "fired"
    assert event["fire_count"] == 2
    assert list(fork["relu_1_2"].ops[0].edits or ())
    assert list(fork["tanh_1_4"].ops[0].edits or ())
    # The source trace is untouched.
    assert torch.allclose(trace["linear_3_5"].out, _hand(model, x), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. set_only stages; a later propagating do carries the staged edit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("staged_site", "staged_scale", "later_site", "later_scale"),
    [
        pytest.param("tanh_1_4", "tanh", "relu_1_2", "relu", id="staged_downstream"),
        pytest.param("relu_1_2", "relu", "tanh_1_4", "tanh", id="staged_upstream"),
    ],
)
def test_set_only_do_stages_and_a_later_do_propagates_both(
    monkeypatch: pytest.MonkeyPatch,
    staged_site: str,
    staged_scale: str,
    later_site: str,
    later_scale: str,
) -> None:
    """A set-only ``do`` changes no value; the next propagating ``do`` applies both edits."""

    model, x, trace = _chain()
    fork = trace.fork()
    baseline = fork["linear_3_5"].out.clone()
    staged_baseline = fork[staged_site].out.clone()
    replays = _count_calls(monkeypatch, replay_module, "_run_replay")

    returned = fork.do(staged_site, _double, intervention=_SET_ONLY)

    assert returned is fork
    assert replays["_run_replay"] == 0
    assert torch.equal(fork["linear_3_5"].out, baseline)
    assert torch.equal(fork[staged_site].out, staged_baseline)
    assert _events(fork)[-1]["lane"] == "set_only"

    fork.do(later_site, tl.scale(0.5))

    assert replays["_run_replay"] == 1
    scales = {staged_scale: 2.0, later_scale: 0.5}
    assert torch.allclose(fork["linear_3_5"].out, _hand(model, x, **scales), atol=1e-6)
    assert torch.equal(trace["linear_3_5"].out, baseline)


# ---------------------------------------------------------------------------
# 3. differentiable=True forks with grad; rerun and set_only ignore it
# ---------------------------------------------------------------------------


_DIFFERENTIABLE = InterventionOptions(differentiable=True)


def test_differentiable_do_returns_a_grad_carrying_fork_and_leaves_the_source() -> None:
    """The replay engine reads ``differentiable``: a new fork whose outputs keep grad."""

    model, x, trace = _chain(_GRAD_CAPTURE)
    source = trace.fork()
    before = source["linear_3_5"].out.clone()

    result = source.do("relu_1_2", tl.scale(2.0), intervention=_DIFFERENTIABLE)

    assert result is not source
    out = result["linear_3_5"].out
    assert out.requires_grad
    assert torch.allclose(out.detach(), _hand(model, x, relu=2.0), atol=1e-6)
    assert result.last_run["differentiable"] is True
    frontier = result._replay_frontier["relu_1_2"]
    assert frontier.is_leaf and frontier.requires_grad
    out.sum().backward()
    assert frontier.grad is not None
    # The source keeps its captured, grad-free values.
    assert torch.equal(source["linear_3_5"].out, before)
    assert source.autograd.has_backward_pass is False


def test_rerun_engine_ignores_differentiable() -> None:
    """``engine="rerun"`` runs the live model in place; the flag is no error and no fork."""

    model, x, trace = _chain(_GRAD_CAPTURE)
    fork = trace.fork()
    options = InterventionOptions(engine="rerun", differentiable=True)

    result = fork.do("relu_1_2", tl.scale(2.0), model=model, inputs=x, intervention=options)

    assert result is fork
    assert torch.allclose(fork["linear_3_5"].out, _hand(model, x, relu=2.0), atol=1e-6)


def test_set_only_engine_ignores_differentiable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``engine="set_only"`` stages on this trace; the flag is no error and no fork."""

    _model, _x, trace = _chain(_GRAD_CAPTURE)
    fork = trace.fork()
    before = fork["linear_3_5"].out.clone()
    replays = _count_calls(monkeypatch, replay_module, "_run_replay")
    differentiable_replays = _count_calls(monkeypatch, replay_module, "_run_differentiable_replay")
    options = InterventionOptions(engine="set_only", differentiable=True)

    result = fork.do({"relu_1_2": tl.scale(2.0)}, intervention=options)

    assert result is fork
    assert replays["_run_replay"] == 0
    assert differentiable_replays["_run_differentiable_replay"] == 0
    assert torch.equal(fork["linear_3_5"].out, before)
    assert len(fork._intervention_spec.hook_specs) == 1


# ---------------------------------------------------------------------------
# 4. ReplayOptions lost hooks= and differentiable=; InterventionOptions gained it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("removed", ["hooks", "differentiable"])
def test_replay_options_refuse_the_removed_fields(removed: str) -> None:
    """``ReplayOptions(hooks=...)`` / ``(differentiable=...)`` refuse, naming the field."""

    assert removed not in _REPLAY_FIELDS
    assert removed not in ReplayOptions().as_dict()
    with pytest.raises(TypeError, match=removed):
        ReplayOptions(**{removed: True})


def test_intervention_options_differentiable_round_trips_the_merge() -> None:
    """``differentiable`` is a grouped field with a flat twin, an explicit bit and a conflict."""

    assert "differentiable" in _INTERVENTION_FIELDS
    assert _INTERVENTION_FLAT_TO_GROUP["differentiable"] == "differentiable"

    default = InterventionOptions()
    assert default.differentiable is False
    assert not default.is_field_explicit("differentiable")

    grouped = InterventionOptions(differentiable=True)
    assert grouped.as_dict()["differentiable"] is True
    assert grouped.is_field_explicit("differentiable")
    merged = merge_intervention_options(intervention=grouped)
    assert merged == grouped
    assert merged.is_field_explicit("differentiable")
    rebuilt = InterventionOptions.from_values(grouped.as_dict(), frozenset({"differentiable"}))
    assert rebuilt == grouped

    flat = merge_intervention_options(intervention=None, differentiable=True)
    assert flat.differentiable is True
    assert flat.is_field_explicit("differentiable")
    assert flat.engine == "auto"

    with pytest.raises(ArgumentConflictError) as excinfo:
        merge_intervention_options(
            intervention=InterventionOptions(differentiable=False), differentiable=True
        )
    assert excinfo.value.fields["code"] == "option_group_conflict"


# ---------------------------------------------------------------------------
# 6. region and edge-substitution doors push through replay.replay_from_op
# ---------------------------------------------------------------------------


def test_region_do_pushes_downstream_through_replay_from_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A region edit substitutes its exit and pushes the rest of the graph."""

    model, x, trace = _chain()
    fork = trace.fork()
    pushes = _count_calls(monkeypatch, replay_module, "replay_from_op")

    fork.do(fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region(), tl.zero_ablate())

    assert pushes["replay_from_op"] == 1
    with torch.no_grad():
        want = model.fc3(torch.zeros(3, 4))
    assert torch.allclose(fork["linear_3_5"].out, want, atol=1e-6)
    assert torch.allclose(trace["linear_3_5"].out, _hand(model, x), atol=1e-6)


def test_edge_substitution_pushes_downstream_through_replay_from_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An edge edit re-executes the child and pushes downstream of it."""

    model, x, trace = _chain()
    fork = trace.fork()
    edge = next(e for e in fork.graph.edges if e.parent_label == "relu_1_2")
    pushes = _count_calls(monkeypatch, replay_module, "replay_from_op")

    fork.do(edge.__selection__(), tl.zero_ablate())

    assert pushes["replay_from_op"] == 1
    with torch.no_grad():
        child = model.fc2(torch.zeros(3, 4))
        want = model.fc3(torch.tanh(child))
    assert torch.allclose(fork["linear_2_3"].out, child, atol=1e-6)
    assert torch.allclose(fork["linear_3_5"].out, want, atol=1e-6)
    # Producer truth is intact: only the consumer saw the substitution.
    assert torch.allclose(fork["relu_1_2"].out, torch.relu(model.fc1(x)), atol=1e-6)
