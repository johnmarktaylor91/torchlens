"""The rerun state refresh: what it copies, what it shares, and what it isolates.

A same-shape rerun refreshes the fork's existing ops in place from the fresh
trace. The refresh copies metadata containers but hands tensor payloads over
by identity, and the replay templates keep each parameter literal as the live
parameter object, exactly as the source capture's templates hold it. These
tests pin that sharing with deterministic counters (clone bytes and
``copy.deepcopy`` calls inside the refresh) and pin the isolation guarantees
around it: parameter edits through a fork never write the live parameter, and
nothing done after forking changes the source trace's recorded values.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

import torchlens as tl
from torchlens.data_classes.trace import Trace
from torchlens.intervention.errors import ReplayPreconditionError
from torchlens.intervention.types import (
    CapturedArgTemplate,
    LiteralTensor,
    LiteralValue,
    ParentRef,
    Unsupported,
)
from torchlens.options import CaptureOptions, InterventionOptions

_CLONE_OPS = ("clone", "_to_copy", "copy_", "copy", "detach_copy", "contiguous")
_TEMPLATE_TYPES = (CapturedArgTemplate, LiteralTensor, LiteralValue, ParentRef, Unsupported)
_ATOMIC_TYPES = (type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _CloneBytes(TorchDispatchMode):
    """Count bytes produced by clone/copy-like aten ops."""

    def __init__(self) -> None:
        super().__init__()
        self.nbytes = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        if func.overloadpacket.__name__ in _CLONE_OPS and isinstance(out, torch.Tensor):
            self.nbytes += out.numel() * out.element_size()
        return out


def _capture() -> tuple[_Net, torch.Tensor, Trace]:
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(replayable=True))
    return model, x, trace


def _rerun_fork(trace: Trace, model: nn.Module, x: torch.Tensor) -> Trace:
    return trace.fork().do(
        tl.func("relu"),
        tl.scale(2.0),
        model=model,
        x=x,
        intervention=InterventionOptions(engine="rerun"),
    )


def _label(trace: Trace, fragment: str, index: int = 0) -> str:
    return [op.layer_label for op in trace.layer_list if fragment in op.layer_label][index]


def _push(trace: Trace, label: str, *, allow: bool = False) -> torch.Tensor:
    edited = trace.fork().do(
        label,
        tl.scale(0.5),
        intervention=InterventionOptions(engine="replay", allow_changed_params=allow),
    )
    return edited.output_ops[0].out.detach().clone()


def _param_literals(trace: Trace) -> list[torch.Tensor]:
    literals = []
    for op in trace.layer_list:
        template = getattr(op, "args_template", None)
        if template is None:
            continue
        literals.extend(comp.value for comp in template.args if isinstance(comp, LiteralTensor))
    return literals


def _recorded(trace: Trace) -> dict[str, torch.Tensor]:
    return {
        op.layer_label: op.out.detach().clone() for op in trace.layer_list if op.out is not None
    }


def _same_recorded(trace: Trace, recorded: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(trace[label].out, value) for label, value in recorded.items())


def test_refresh_keeps_template_parameters_by_identity() -> None:
    model, x, trace = _capture()
    fork = _rerun_fork(trace, model, x)
    assert fork.last_run["fast_refresh"] is True
    params = {id(p) for p in model.parameters()}
    source_literals = _param_literals(trace)
    fork_literals = _param_literals(fork)
    assert source_literals and len(fork_literals) == len(source_literals)
    assert all(id(value) in params for value in source_literals)
    # The refreshed fork's templates hold the live parameters, like the source's.
    assert all(id(value) in params for value in fork_literals)


def test_refresh_clones_no_tensor_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declared counter target: zero clone-like bytes inside the refresh."""

    model, x, trace = _capture()
    counted: list[int] = []
    original = Trace._refresh_matching_rerun_state_from

    def counting(self: Trace, new_log: Trace) -> bool:
        with _CloneBytes() as mode:
            result = original(self, new_log)
        counted.append(mode.nbytes)
        return result

    monkeypatch.setattr(Trace, "_refresh_matching_rerun_state_from", counting)
    fork = _rerun_fork(trace, model, x)
    assert fork.last_run["fast_refresh"] is True
    assert counted == [0]


def test_refresh_deepcopies_no_atomic_value_or_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declared counter target: no deepcopy of a tensor, an atomic value or a template."""

    model, x, trace = _capture()
    copied_types: list[type] = []
    original_deepcopy = copy.deepcopy
    original_refresh = Trace._refresh_matching_rerun_state_from

    depth = [0]

    def recording_deepcopy(value: Any, *args: Any, **kwargs: Any) -> Any:
        # Only the refresh's own calls; deepcopy recurses through this name too.
        if depth[0] == 0:
            copied_types.append(type(value))
        depth[0] += 1
        try:
            return original_deepcopy(value, *args, **kwargs)
        finally:
            depth[0] -= 1

    def refreshing(self: Trace, new_log: Trace) -> bool:
        monkeypatch.setattr(copy, "deepcopy", recording_deepcopy)
        try:
            return original_refresh(self, new_log)
        finally:
            monkeypatch.setattr(copy, "deepcopy", original_deepcopy)

    monkeypatch.setattr(Trace, "_refresh_matching_rerun_state_from", refreshing)
    fork = _rerun_fork(trace, model, x)
    assert fork.last_run["fast_refresh"] is True
    offending = [
        t
        for t in copied_types
        if t in _ATOMIC_TYPES or issubclass(t, torch.Tensor) or t in _TEMPLATE_TYPES
    ]
    assert offending == []


def test_refreshed_fork_matches_a_push_of_the_same_edit() -> None:
    """The refresh changes no value: the rerun fork equals the push fork."""

    model, x, trace = _capture()
    fork = _rerun_fork(trace, model, x)
    relu = _label(trace, "relu")
    pushed = trace.fork().do(relu, tl.scale(2.0), intervention=InterventionOptions(engine="replay"))
    for op in pushed.layer_list:
        if op.out is not None:
            assert torch.equal(fork[op.layer_label].out, op.out), op.layer_label
    fc1 = _label(trace, "linear")
    assert torch.equal(_push(fork, fc1), _push(pushed, fc1))


def test_param_edit_through_a_refreshed_fork_writes_nothing_shared() -> None:
    model, x, trace = _capture()
    recorded = _recorded(trace)
    fc1 = _label(trace, "linear")
    source_push = _push(trace, fc1)
    fork = _rerun_fork(trace, model, x)
    fork_recorded = _recorded(fork)
    weight = model.fc2.weight
    before = weight.detach().clone()
    version = weight._version
    edited = fork.fork()
    edited.do(tl.params("fc2.weight"), tl.scale(0.0))
    assert weight._version == version
    assert weight.detach().view(torch.uint8).equal(before.view(torch.uint8))
    assert _same_recorded(trace, recorded)
    assert _same_recorded(fork, fork_recorded)
    assert torch.equal(_push(trace, fc1), source_push)


def test_live_parameter_edit_after_forking_keeps_recorded_values() -> None:
    model, x, trace = _capture()
    recorded = _recorded(trace)
    fork = _rerun_fork(trace, model, x)
    fork_recorded = _recorded(fork)
    saved = model.fc2.weight.detach().clone()
    try:
        with torch.no_grad():
            model.fc2.weight.mul_(3.0)
        assert _same_recorded(trace, recorded)
        assert _same_recorded(fork, fork_recorded)
    finally:
        with torch.no_grad():
            model.fc2.weight.copy_(saved)


def test_every_trace_and_fork_refuses_a_changed_live_parameter() -> None:
    """After an in-place edit of the live parameter, the source, a push fork and a
    rerun fork each refuse replay; with ``allow_changed_params=True`` all three replay
    with the edited parameter and agree with a fresh capture."""

    model, x, trace = _capture()
    relu = _label(trace, "relu")
    fc1 = _label(trace, "linear")
    rerun_fork = _rerun_fork(trace, model, x)
    push_fork = trace.fork().do(
        relu, tl.scale(2.0), intervention=InterventionOptions(engine="replay")
    )
    saved = model.fc2.weight.detach().clone()
    try:
        with torch.no_grad():
            model.fc2.weight.mul_(3.0)
        for stale in (trace, push_fork, rerun_fork):
            with pytest.raises(ReplayPreconditionError, match="changed since capture"):
                _push(stale, fc1)
        fresh = tl.trace(model, x, capture=CaptureOptions(replayable=True))
        fresh_fork = fresh.fork().do(
            relu, tl.scale(2.0), intervention=InterventionOptions(engine="replay")
        )
        assert torch.equal(_push(trace, fc1, allow=True), _push(fresh, fc1))
        assert torch.equal(_push(push_fork, fc1, allow=True), _push(fresh_fork, fc1))
        assert torch.equal(_push(rerun_fork, fc1, allow=True), _push(push_fork, fc1, allow=True))
    finally:
        with torch.no_grad():
            model.fc2.weight.copy_(saved)
