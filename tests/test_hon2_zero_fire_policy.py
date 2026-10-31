"""One zero-fire policy on every intervention door, with one ``on_zero_fire`` knob.

An intervention that fires at zero sites is always recorded on the door's own
record. ``on_zero_fire`` then decides, identically at every door: ``"warn"``
(default) emits the door's coded ``TorchLensWarning``, ``"error"`` raises the door's
typed refusal with the same code, ``"disclose"`` records only. Doors: ``spec.bind``,
``binding.generate(...)``, ``tl.trace(intervene=)``, ``Trace.run`` (capture rerun and the
fast engine). The typed rerun refusals for an edit that cannot fire on a rerun stay
ahead of the policy: ``"disclose"`` cannot silence them.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.intervention.errors import (
    BindingRuntimeError,
    LiveModeLabelError,
    SiteResolutionError,
)
from torchlens.options import CaptureOptions

_POLICIES = ("warn", "error", "disclose")


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))

    def generate(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        return self(x)


def _setup() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _Net().eval(), torch.randn(2, 4)


def _never() -> Any:
    """A one-rule spec whose site never occurs in ``_Net`` (it has no sigmoid)."""

    return tl.when(tl.func("sigmoid"), tl.zero_ablate())


@contextmanager
def _caught() -> Iterator[list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _codes(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message.fields.get("code")) for w in caught if isinstance(w.message, TorchLensWarning)
    ]


def _bare_zero_match(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if not isinstance(w.message, TorchLensWarning) and "matched zero sites" in str(w.message)
    ]


# --- bind and bound generate -------------------------------------------------


@pytest.mark.parametrize("policy", _POLICIES)
def test_bind_door_policy(policy: str) -> None:
    model, x = _setup()
    kwargs = {} if policy == "warn" else {"on_zero_fire": policy}
    bound = _never().bind(model, **kwargs)
    with _caught() as caught:
        if policy == "error":
            with pytest.raises(BindingRuntimeError) as refused:
                bound(x)
            assert refused.value.fields["code"] == "bind_zero_fire"
        else:
            assert bound(x).shape == (2, 2)
    assert bound.last_report is not None
    assert bound.last_report.zero_fire_rule_ids == (bound.recipe.rules[0].rule_id,)
    expected = ["bind_zero_fire"] if policy == "warn" else []
    assert [c for c in _codes(caught) if c.endswith("zero_fire")] == expected


@pytest.mark.parametrize("policy", _POLICIES)
def test_bound_generate_door_policy(policy: str) -> None:
    model, x = _setup()
    kwargs = {} if policy == "warn" else {"on_zero_fire": policy}
    bound = _never().bind(model, **kwargs)
    with _caught() as caught:
        if policy == "error":
            with pytest.raises(BindingRuntimeError) as refused:
                bound.generate(x)
            assert refused.value.fields["code"] == "bind_zero_fire"
        else:
            assert bound.generate(x).shape == (2, 2)
        assert bound.last_report is not None
        assert len(bound.last_report.zero_fire_rule_ids) == 1
    expected = ["bind_zero_fire"] if policy == "warn" else []
    assert [c for c in _codes(caught) if c.endswith("zero_fire")] == expected


def test_bind_fired_rule_never_warns() -> None:
    model, x = _setup()
    bound = tl.when(tl.func("relu"), tl.zero_ablate()).bind(model)
    with _caught() as caught:
        bound(x)
    assert bound.last_report.zero_fire_rule_ids == ()
    assert not [c for c in _codes(caught) if c.endswith("zero_fire")]


# --- tl.trace(intervene=) ------------------------------------------------------


def _unmatched_intervene(trace: tl.Trace) -> list[dict[str, str]]:
    return [
        row
        for row in trace.annotations.get("unmatched_capture_selectors", [])
        if row["slot"] == "intervene"
    ]


@pytest.mark.parametrize("policy", _POLICIES)
def test_trace_door_policy(policy: str) -> None:
    model, x = _setup()
    kwargs = {} if policy == "warn" else {"on_zero_fire": policy}
    with _caught() as caught:
        if policy == "error":
            with pytest.raises(SiteResolutionError) as refused:
                tl.trace(model, x, intervene=_never(), **kwargs)
            assert refused.value.fields["code"] == "trace_zero_fire"
            trace = None
        else:
            trace = tl.trace(model, x, intervene=_never(), **kwargs)
    if trace is not None:
        assert len(_unmatched_intervene(trace)) == 1
    expected = ["trace_zero_fire"] if policy == "warn" else []
    assert [c for c in _codes(caught) if c.endswith("zero_fire")] == expected
    # The zero-fire disclosure is the coded warning, never a bare UserWarning.
    assert _bare_zero_match(caught) == []


def test_trace_policy_never_changes_the_capture() -> None:
    """``on_zero_fire`` only settles the disclosure: the captured trace is identical."""

    model, x = _setup()
    with _caught():
        warned = tl.trace(model, x, intervene=_never())
    disclosed = tl.trace(model, x, intervene=_never(), on_zero_fire="disclose")
    assert _unmatched_intervene(warned) == _unmatched_intervene(disclosed)
    assert warned.op_labels == disclosed.op_labels
    for label in warned.op_labels:
        left, right = warned[label].out, disclosed[label].out
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
    fired = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    fired_disclosed = tl.trace(
        model, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()), on_zero_fire="disclose"
    )
    assert torch.equal(
        fired[fired.output_layers[0]].out, fired_disclosed[fired.output_layers[0]].out
    )


def test_trace_fired_spec_never_warns() -> None:
    model, x = _setup()
    with _caught() as caught:
        tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    assert not [c for c in _codes(caught) if c.endswith("zero_fire")]


# --- Trace.run (rerun) -------------------------------------------------------


class _ReluAdd(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) + 1


def _gated_rerun_case() -> tuple[nn.Module, tl.Trace, dict[str, bool]]:
    """A sticky hook resolved on the source trace that stops matching on the rerun."""

    model = _ReluAdd()
    trace = tl.trace(model, torch.randn(2, 3), capture=CaptureOptions(intervention_ready=True))
    gate = {"active": True}
    trace.attach_hooks(
        tl.func("relu") & tl.where(lambda _op: gate["active"], name_hint="runtime_gate"),
        tl.zero_ablate(),
    )
    gate["active"] = False
    return model, trace, gate


@pytest.mark.parametrize("policy", _POLICIES)
def test_rerun_door_policy(policy: str) -> None:
    model, trace, _ = _gated_rerun_case()
    kwargs = {} if policy == "warn" else {"on_zero_fire": policy}
    new_x = torch.randn(2, 3) + 1.0
    with _caught() as caught:
        if policy == "error":
            with pytest.raises(SiteResolutionError) as refused:
                trace.run(model, new_x, **kwargs)
            assert refused.value.fields["code"] == "rerun_zero_fire"
        else:
            trace.run(model, new_x, **kwargs)
            assert trace.last_run["hooks_unfired"] == 1
            assert trace.last_run["hooks_fired"] == 0
    expected = ["rerun_zero_fire"] if policy == "warn" else []
    assert [c for c in _codes(caught) if c.endswith("zero_fire")] == expected


_DIM = 8


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(_DIM, _DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class _Steerable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()
        self.head = nn.Linear(_DIM, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))


def _fast_steered_case() -> tuple[nn.Module, tl.Trace, Callable[[int], torch.Tensor]]:
    """A steered trace for the fast door (the caller keeps the model alive)."""

    torch.manual_seed(0)
    model = _Steerable().eval()

    def x_of(seed: int) -> torch.Tensor:
        return torch.randn(1, 3, _DIM, generator=torch.Generator().manual_seed(seed))

    spec = tl.when(tl.module("block"), tl.steer(torch.randn(_DIM), strength=3.0, dim=-1))
    trace = tl.trace(model, x_of(0), save=tl.module("block") | tl.module("head"), intervene=spec)
    return model, trace, x_of


@pytest.mark.parametrize("policy", _POLICIES)
def test_fast_run_door_policy_knob_is_accepted(policy: str) -> None:
    """The fast door takes the same knob; a fully fired run stays silent at every value."""

    model, trace, x_of = _fast_steered_case()
    with _caught() as caught:
        trace.run(inputs=x_of(1), fast=True, on_zero_fire=policy)
    del model
    assert trace.last_run["hooks_unfired"] == 0
    assert not [c for c in _codes(caught) if c.endswith("zero_fire")]


@pytest.mark.parametrize("door", ["bind", "trace", "run"])
def test_invalid_policy_refuses_identically(door: str) -> None:
    model, x = _setup()
    with pytest.raises(ValueError) as refused:
        if door == "bind":
            _never().bind(model, on_zero_fire="ignore")
        elif door == "trace":
            tl.trace(model, x, intervene=_never(), on_zero_fire="ignore")
        else:
            tl.trace(model, x).run(model, x, on_zero_fire="ignore")
    assert refused.value.fields["code"] == "zero_fire_policy_invalid"  # type: ignore[attr-defined]


# --- the typed rerun refusals stay ahead of the policy ------------------------


@pytest.mark.parametrize("same_inputs", [True, False], ids=["same_inputs", "new_inputs"])
def test_disclose_cannot_silence_the_post_capture_predicate_refusal(same_inputs: bool) -> None:
    """A rerun edit that can never fire live is refused typed, whatever the knob says."""

    torch.manual_seed(0)
    model = _Net().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))
    label = str(trace.find_sites(tl.func("relu"))[0].layer_label)
    trace.attach_hooks(tl.where(lambda p: p.layer_label == label), tl.zero_ablate())
    run_x = x if same_inputs else x + 1.0
    for policy in _POLICIES:
        with pytest.raises(LiveModeLabelError) as refused:
            trace.run(model, run_x, on_zero_fire=policy)
        assert refused.value.fields["code"] == "rerun_selector_unmatchable_live"


@pytest.mark.parametrize("policy", _POLICIES)
def test_fast_engine_refusal_follows_the_knob(policy: str) -> None:
    """A recorded fast run with an unfired entry refuses only under ``"error"``."""

    from torchlens._fast_live_steer import refuse_unfired_fast_run
    from torchlens.intervention._zero_fire import zero_fire_policy_scope

    class _Recorded:
        last_run = {"engine": "guarded_fast", "hooks_unfired": 2}

    with zero_fire_policy_scope(policy):
        if policy == "error":
            with pytest.raises(SiteResolutionError) as refused:
                refuse_unfired_fast_run(_Recorded())
            assert refused.value.fields["code"] == "rerun_zero_fire"
        else:
            refuse_unfired_fast_run(_Recorded())
