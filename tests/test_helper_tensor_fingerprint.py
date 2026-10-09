"""F9: helpers alias the caller's tensor; doors that re-read a staged one refuse a change.

``tl.steer`` and the other tensor-carrying helpers keep the caller's tensor by
reference. Each staged entry records a full-content digest when it is staged; a
rerun and ``save_intervention`` refuse ``helper_tensor_changed_since_capture``
when the tensor moved since then (including ``.data`` writes, which leave the
version counter alone). An unchanged tensor runs silently and exactly, and a
loop that re-stages after each update keeps working. A bound executor is live:
each call reads the tensor, and its report records the version counter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import SpecMutationError

_CODE = "helper_tensor_changed_since_capture"


class _Net(nn.Module):
    """fc1 -> relu -> fc2."""

    def __init__(self) -> None:
        """Build the two linears."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def setup() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """A seeded model, an input and a steering direction for the fc1 output."""

    torch.manual_seed(0)
    return _Net().eval(), torch.randn(4, 8), torch.randn(16)


def _truth(model: nn.Module, x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Plain-hook ground truth for one steer of the fc1 output."""

    handle = model.fc1.register_forward_hook(lambda _m, _a, out: out + direction)
    try:
        with torch.no_grad():
            return model(x)
    finally:
        handle.remove()


def _steer(direction: torch.Tensor) -> Any:
    """Return an additive steer along ``direction``."""

    return tl.steer(direction, magnitude=1.0, feature_axis=-1)


def _staged_trace(model: nn.Module, x: torch.Tensor, direction: torch.Tensor, door: str) -> Any:
    """Return a trace with the steer staged through ``door``."""

    if door == "intervene_module":
        return tl.trace(model, x, intervene=tl.when(tl.module("fc1"), _steer(direction)))
    trace = tl.trace(model, x)
    trace.attach_hooks(tl.module("fc1"), _steer(direction), confirm_mutation=True)
    return trace


_EDITS: dict[str, Callable[[torch.Tensor], None]] = {
    "in_place": lambda d: d.add_(0.5),
    "data_write": lambda d: d.data.__setitem__(0, d.data[0] + 0.5),
}


@pytest.mark.parametrize("edit", sorted(_EDITS))
@pytest.mark.parametrize("door", ["intervene_module", "attach_hooks"])
def test_rerun_refuses_a_changed_helper_tensor(setup, door: str, edit: str) -> None:
    """An edited steer tensor refuses the rerun typed and leaves the trace unchanged."""

    model, x, direction = setup
    trace = _staged_trace(model, x, direction, door)
    trace.run(model, x)
    readout = trace.output_ops[0].out.clone()
    staged = trace._intervention_spec.freeze().hook_specs
    version = direction._version
    _EDITS[edit](direction)
    if edit == "data_write":
        assert direction._version == version, "the .data write must not bump the version"
    with pytest.raises(SpecMutationError) as excinfo:
        trace.run(model, x)
    assert excinfo.value.fields["code"] == _CODE
    assert "steer" in str(excinfo.value)
    assert torch.equal(trace.output_ops[0].out, readout)
    assert trace._intervention_spec.freeze().hook_specs == staged
    with pytest.raises(SpecMutationError):
        trace.run(model, x, replay=tl.options.ReplayOptions(append=True))


def test_func_predicate_rerun_refuses_a_changed_helper_tensor(setup) -> None:
    """Per-op predicate entries are checked before the predicate is re-armed."""

    model, x, _ = setup
    direction = torch.randn(16)
    trace = tl.trace(model, x, intervene=tl.when(tl.func("relu"), _steer(direction)))
    direction.mul_(2.0)
    with pytest.raises(SpecMutationError) as excinfo:
        trace.run(model, x)
    assert excinfo.value.fields["code"] == _CODE


@pytest.mark.parametrize("door", ["intervene_module", "attach_hooks"])
def test_unchanged_helper_tensor_reruns_silently_and_exactly(setup, door: str) -> None:
    """An untouched tensor reruns exactly, every time, under warnings-as-errors."""

    model, x, direction = setup
    trace = _staged_trace(model, x, direction, door)
    want = _truth(model, x, direction)
    for _ in range(3):
        trace.run(model, x)
        torch.testing.assert_close(trace.output_ops[0].out, want)


def test_learning_loop_that_restages_each_step_keeps_working(setup) -> None:
    """Updating the direction and re-staging each step reruns with the new value."""

    model, x, direction = setup
    trace = tl.trace(model, x)
    handle = trace.attach_hooks(tl.module("fc1"), _steer(direction), confirm_mutation=True)
    for _ in range(3):
        direction.add_(0.25)
        handle.remove()
        handle = trace.attach_hooks(tl.module("fc1"), _steer(direction), confirm_mutation=True)
        trace.run(model, x)
        torch.testing.assert_close(trace.output_ops[0].out, _truth(model, x, direction))


def test_bound_executor_reads_an_in_place_update_on_the_next_call(setup) -> None:
    """A bound executor is live: an in-place update applies to the next call, exactly.

    The report records each helper tensor's version counter at call start, so
    the update is visible after the fact; nothing refuses.
    """

    model, x, direction = setup
    spec = tl.when(tl.module("fc1"), _steer(direction))
    bound = spec.bind(model)
    versions = []
    for step in range(3):
        if step:
            direction.add_(0.5)
        # Rule ids follow the helper tensor's current content, and every report
        # ledger keys by the identity the call started with: read it per call.
        rule_id = spec.rules[0].rule_id
        with torch.no_grad():
            out = bound(x)
        assert torch.equal(out, _truth(model, x, direction))
        assert set(bound.last_report.helper_tensor_versions) == {rule_id}
        versions.append(bound.last_report.helper_tensor_versions[rule_id])
    assert versions[1][0] == versions[0][0] + 1
    assert versions[2][0] == versions[1][0] + 1


def test_save_intervention_refuses_a_changed_helper_tensor(setup, tmp_path) -> None:
    """A recipe save never writes a value that did not produce the trace."""

    from torchlens.io import load_intervention_spec

    model, x, direction = setup
    trace = _staged_trace(model, x, direction, "intervene_module")
    trace.save_intervention(tmp_path / "ok.tlspec", level="portable")
    assert len(load_intervention_spec(tmp_path / "ok.tlspec").hook_specs) == 1
    direction.sub_(1.0)
    with pytest.raises(SpecMutationError) as excinfo:
        trace.save_intervention(tmp_path / "changed.tlspec", level="portable")
    assert excinfo.value.fields["code"] == _CODE
    assert not (tmp_path / "changed.tlspec").exists()


@pytest.mark.parametrize("door", ["legacy", "fast"])
def test_guarded_fast_rerun_refuses_a_changed_helper_tensor(setup, door: str) -> None:
    """The guarded fast engine re-reads the staged steer, so it refuses a change too.

    A module save keeps the trace eligible for the fast engine, which applies
    the staged spec through native module hooks. An edited steer tensor must
    refuse typed on both doors that reach that engine (the legacy
    ``run(model, x)`` and the explicit ``run(inputs=..., fast=True)``), exactly
    as the capture rerun does, and leave the trace's saved values unchanged.
    """

    model, x, direction = setup
    trace = tl.trace(
        model, x, save=tl.module("fc2"), intervene=tl.when(tl.module("fc1"), _steer(direction))
    )

    def rerun() -> None:
        if door == "legacy":
            trace.run(model, x)
        else:
            trace.run(inputs=x, fast=True)

    rerun()
    assert trace.last_run["engine"] == "guarded_fast"
    readout = trace.find_sites(tl.module("fc2")).first().out.clone()
    torch.testing.assert_close(readout, _truth(model, x, direction))
    direction.add_(0.5)
    with pytest.raises(SpecMutationError) as excinfo:
        rerun()
    assert excinfo.value.fields["code"] == _CODE
    assert torch.equal(trace.find_sites(tl.module("fc2")).first().out, readout)
