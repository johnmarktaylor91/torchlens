"""F9: helpers alias the caller's tensor; doors that re-read it refuse a changed value.

``tl.steer`` and the other tensor-carrying helpers keep the caller's tensor by
reference. Each staged entry and each bound rule records a full-content digest
when it is staged or bound; a rerun, a bound call and ``save_intervention``
refuse ``helper_tensor_changed_since_capture`` when the tensor moved since then
(including ``.data`` writes, which leave the version counter alone). An
unchanged tensor runs silently and exactly, and a loop that re-stages or rebinds
after each update keeps working.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import BindingRuntimeError, SpecMutationError

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


def test_bound_call_refuses_a_changed_helper_tensor_and_rebind_works(setup) -> None:
    """A bound executor checks its helpers on every call; rebinding takes the new value."""

    model, x, direction = setup
    spec = tl.when(tl.module("fc1"), _steer(direction))
    bound = spec.bind(model)
    for _ in range(2):
        with torch.no_grad():
            torch.testing.assert_close(bound(x), _truth(model, x, direction))
    direction.data[3] += 1.0
    with pytest.raises(BindingRuntimeError) as excinfo, torch.no_grad():
        bound(x)
    assert excinfo.value.fields["code"] == _CODE
    with torch.no_grad():
        torch.testing.assert_close(spec.bind(model)(x), _truth(model, x, direction))


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
