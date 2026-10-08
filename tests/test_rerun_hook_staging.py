"""Staged intervention hooks never accumulate across repeated runs.

Regression pin: after ``tl.trace(model, x, intervene=spec)`` every legacy
rerun ``trace.run(model, x)`` merged the hook plan it had normalized FROM the
trace's own spec back INTO that same spec object, so the staged hooks doubled
per rerun (1, 2, 4, 8) and rerun ``n`` applied the steer ``2**(n-1)`` times.
Every staged-spec door is held to the same shape here: repeated runs must
match a fresh steered capture exactly, keep the staged hook count fixed, and
leave no torch hook behind on the model.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

_SITE = "fc1"
_REPEATS = 3


class _MLP(nn.Module):
    """fc1 -> relu -> fc2."""

    def __init__(self) -> None:
        """Build the two linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.fc2(torch.relu(self.fc1(x)))


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    """Return a seeded model and input."""

    torch.manual_seed(0)
    return _MLP().eval(), torch.randn(2, 4)


def _steer() -> Any:
    """Return a deterministic steer helper for the site."""

    direction = torch.linspace(-1.0, 1.0, 8)
    return tl.steer(direction, magnitude=3.0, feature_axis=-1)


def _spec() -> Any:
    """Return a one-clause steer spec at the site."""

    return tl.when(tl.module(_SITE), _steer())


def _save() -> Any:
    """Return the save selector covering the site and the readout."""

    return tl.module(_SITE) | tl.module("fc2")


def _torch_hook_count(model: nn.Module) -> int:
    """Count every torch module hook that can fire on ``model``."""

    from torch.nn.modules import module as torch_module

    count = sum(
        len(getattr(torch_module, name, {}))
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    )
    for submodule in model.modules():
        for name in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        ):
            count += len(getattr(submodule, name, {}))
    return count


def _site_out(trace: Any) -> torch.Tensor:
    """Return the saved site activation."""

    return trace.find_sites(tl.module(_SITE)).first().out


def _readout(trace: Any) -> torch.Tensor:
    """Return the saved readout activation."""

    return trace.find_sites(tl.module("fc2")).first().out


def _staged_hook_count(trace: Any) -> int:
    """Return how many hook entries the trace keeps staged."""

    spec = getattr(trace, "_intervention_spec", None)
    return 0 if spec is None else len(spec.hook_specs)


@pytest.fixture
def steered() -> dict[str, Any]:
    """A fresh steered capture plus its plain counterpart and hook baseline."""

    model, x = _model_and_input()
    baseline = _torch_hook_count(model)
    plain = model(x).detach()
    fresh = tl.trace(model, x, save=_save(), intervene=_spec())
    assert _torch_hook_count(model) == baseline
    reference = {
        "model": model,
        "x": x,
        "baseline": baseline,
        "site": _site_out(fresh).detach().clone(),
        "out": _readout(fresh).detach().clone(),
    }
    # The steer must move the output, or exactness below proves nothing.
    assert not torch.equal(reference["out"], plain)
    return reference


@pytest.mark.smoke
def test_legacy_rerun_after_intervene_capture_does_not_restage_hooks(
    steered: dict[str, Any],
) -> None:
    """``trace.run(model, x)`` after an ``intervene=`` capture applies the steer once."""

    model, x = steered["model"], steered["x"]
    trace = tl.trace(model, x, save=_save(), intervene=_spec())
    staged = _staged_hook_count(trace)
    assert staged == 1
    assert _torch_hook_count(model) == steered["baseline"]
    for _ in range(_REPEATS):
        trace.run(model, x)
        assert _staged_hook_count(trace) == staged
        assert trace.last_run["hooks"] == staged
        assert torch.equal(_site_out(trace), steered["site"])
        assert torch.equal(_readout(trace), steered["out"])
        assert _torch_hook_count(model) == steered["baseline"]


def test_legacy_rerun_after_attach_hooks_does_not_restage_hooks(
    steered: dict[str, Any],
) -> None:
    """Hooks staged through ``attach_hooks`` stay single across reruns."""

    model, x = steered["model"], steered["x"]
    trace = tl.trace(model, x, save=_save())
    trace.attach_hooks(tl.module(_SITE), _steer(), confirm_mutation=True)
    staged = _staged_hook_count(trace)
    assert staged == 1
    for _ in range(_REPEATS):
        trace.run(model, x)
        assert _staged_hook_count(trace) == staged
        assert trace.last_run["hooks"] == staged
        assert torch.equal(_readout(trace), steered["out"])
        assert _torch_hook_count(model) == steered["baseline"]


def test_append_rerun_does_not_restage_hooks(steered: dict[str, Any]) -> None:
    """``run(..., replay=ReplayOptions(append=True))`` keeps the staged plan fixed."""

    model, x = steered["model"], steered["x"]
    trace = tl.trace(model, x, intervene=_spec())
    staged = _staged_hook_count(trace)
    batch = x.shape[0]
    for _ in range(_REPEATS):
        trace.run(model, x, replay=tl.options.ReplayOptions(append=True))
        assert _staged_hook_count(trace) == staged
        assert trace.last_run["hooks"] == staged
        assert torch.equal(_readout(trace)[-batch:], steered["out"])
        assert _torch_hook_count(model) == steered["baseline"]


def test_transactional_run_inputs_refusal_leaves_staging_untouched(
    steered: dict[str, Any],
) -> None:
    """``trace.run(inputs=x)`` refuses a staged spec every time and stages nothing new.

    The transactional provider never applies a staged spec, so it refuses
    rather than return an un-intervened result; repeated refusals must not
    grow the staged plan or leave hooks behind, and the legacy rerun after
    them still applies the steer exactly once.
    """

    from torchlens.intervention.errors import EngineDispatchError

    model, x = steered["model"], steered["x"]
    trace = tl.trace(model, x, save=_save(), intervene=_spec())
    staged = _staged_hook_count(trace)
    for _ in range(_REPEATS):
        with pytest.raises(EngineDispatchError, match="does NOT apply"):
            trace.run(inputs=x)
        assert _staged_hook_count(trace) == staged
        assert _torch_hook_count(model) == steered["baseline"]
    trace.run(model, x)
    assert _staged_hook_count(trace) == staged
    assert torch.equal(_readout(trace), steered["out"])


def test_fork_do_then_rerun_does_not_restage_hooks(steered: dict[str, Any]) -> None:
    """A forked ``do`` edit applies once, and reruns of the fork keep it single."""

    model, x = steered["model"], steered["x"]
    root = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fork = root.fork()
    fork.do(tl.module(_SITE), _steer())
    assert torch.equal(_readout(fork), steered["out"])
    staged = _staged_hook_count(fork)
    for _ in range(_REPEATS):
        fork.run(model, x)
        assert _staged_hook_count(fork) == staged
        assert torch.equal(_readout(fork), steered["out"])
        assert _torch_hook_count(model) == steered["baseline"]


def test_record_with_intervene_is_stable_across_calls(steered: dict[str, Any]) -> None:
    """Repeated ``tl.record(..., intervene=spec)`` with one spec object stay exact."""

    model, x = steered["model"], steered["x"]
    spec = _spec()
    for _ in range(_REPEATS):
        output, _recording = tl.record(model, x, save=_save(), intervene=spec, return_output=True)
        assert torch.equal(output, steered["out"])
        assert _torch_hook_count(model) == steered["baseline"]


def test_repeated_intervene_captures_with_one_spec_are_stable(
    steered: dict[str, Any],
) -> None:
    """Reusing one spec object across captures never grows what it stages."""

    model, x = steered["model"], steered["x"]
    spec = _spec()
    for _ in range(_REPEATS):
        trace = tl.trace(model, x, save=_save(), intervene=spec)
        assert _staged_hook_count(trace) == 1
        assert torch.equal(_readout(trace), steered["out"])
        assert _torch_hook_count(model) == steered["baseline"]


def test_bound_spec_installs_and_removes_hooks_each_call(steered: dict[str, Any]) -> None:
    """``spec.bind(model)`` fires once per call and tears its hooks down after."""

    model, x = steered["model"], steered["x"]
    bound = _spec().bind(model)
    for _ in range(_REPEATS):
        with torch.no_grad():
            output = bound(x)
        assert torch.equal(output, steered["out"])
        assert _torch_hook_count(model) == steered["baseline"]
