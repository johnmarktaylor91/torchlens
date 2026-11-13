"""Finished-trace edits through a backend replay adapter, end to end on a real ``Trace``.

A real adapter exists per backend (``backends/_replay_adapters/<name>.py``). This
file drives the Trace-facing path with a stand-in: a PyTorch capture whose
backend name is set to ``fake`` and an adapter that re-runs each recorded op
from its parents. The trace then takes the backend engine (not the PyTorch
template engine), so the proofs below hold for the engine every backend shares:
an edited fork equals a live ``intervene=`` capture with the same edit, a no-op
edit reproduces the trace exactly, a donor from another framework is refused,
the evidence is written, and corrupted evidence fails the replay check.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends import _intervention_evidence as evidence
from torchlens.intervention import _backend_adapter_lookup, _backend_replay
from torchlens.intervention._replay import _reconstruct_args_from_template, _template_for_site
from torchlens.intervention._replay_context import _replay_site_key
from torchlens.intervention._replay_sites import _unmet_log_precondition
from torchlens.intervention.errors import HookValueError, ReplayPreconditionError


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(x))
        return torch.tanh(h * h) + h


class _GuardNet(nn.Module):
    """Picks a branch with a recorded zero-dimensional bool (a data-dependent guard)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(x)) + 0.5
        flag = h.sum() > 0
        return torch.where(flag, h, -h)


def _make_adapter() -> ModuleType:
    """A stand-in replay adapter that re-runs recorded torch ops from their parents."""

    adapter = ModuleType("fake_replay_adapter")

    def can_replay(trace: Any, op: Any) -> str | None:
        return None if op.func is not None else "no callable recorded"

    def recipe(trace: Any, op: Any) -> Any:
        return SimpleNamespace(output_slots=(op.label,), parent_labels=tuple(op.parents))

    def forward(trace: Any, op: Any, parent_values: Sequence[Any]) -> tuple[Any, ...]:
        keys = {p: _replay_site_key(trace._layer_dict_all_keys[p]) for p in op.parents}
        overlay = dict(zip(keys.values(), parent_values))
        args, kwargs = _reconstruct_args_from_template(
            _template_for_site(op),
            op,
            trace,
            overlay,
            strict=False,
            differentiable_frontier=None,
            label_keys={label: (key,) for label, key in keys.items()},
        )
        with torch.no_grad():
            return (op.func(*args, **kwargs),)

    adapter.can_replay = can_replay
    adapter.recipe = recipe
    adapter.forward = forward
    adapter.to_host = lambda value: value.detach().cpu().numpy()
    adapter.zeros_like = torch.zeros_like
    adapter.close = lambda expected, actual: bool(torch.equal(expected, actual))
    return adapter


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    adapter = _make_adapter()
    monkeypatch.setattr(
        _backend_adapter_lookup, "_lookup_adapter", lambda name: adapter if name == "fake" else None
    )
    return adapter


def _capture(**kwargs: Any) -> tl.Trace:
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(2, 4)
    return tl.trace(model, x, **kwargs)


def _as_fake(trace: tl.Trace) -> tl.Trace:
    trace._provenance_backend = "fake"
    return trace


def _op(trace: tl.Trace, label: str) -> Any:
    return trace._layer_dict_all_keys[label]


def _first(trace: tl.Trace, prefix: str) -> str:
    return next(op.label for op in trace._layer_list if op.label.startswith(prefix))


def _output(trace: tl.Trace) -> torch.Tensor:
    return next(op for op in trace._layer_list if op.is_output).out


def test_adapter_lifts_the_finished_trace_refusal(fake_backend: ModuleType) -> None:
    preview = SimpleNamespace(
        _tracing_finished=True, is_replayable=False, options=None, _provenance_backend="fake"
    )
    assert _unmet_log_precondition(preview) is None
    preview._provenance_backend = "jax"
    assert "cannot edit or patch a finished trace" in str(_unmet_log_precondition(preview))


def test_edited_fork_equals_live_intervene(fake_backend: ModuleType) -> None:
    live = _capture(intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    fork = _as_fake(_capture()).fork()
    edited = fork.do(tl.func("relu"), tl.zero_ablate())
    assert edited.last_run["backend"] == "fake"
    assert torch.equal(_output(edited), _output(live))


def test_callable_edit_equals_live_intervene(fake_backend: ModuleType) -> None:
    def double(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        return out * 2.0

    live = _capture(intervene=tl.when(tl.func("relu"), double))
    edited = _as_fake(_capture()).fork().do(tl.func("relu"), double)
    assert torch.allclose(_output(edited), _output(live), rtol=0, atol=0)


def test_noop_edit_reproduces_the_trace_exactly(fake_backend: ModuleType) -> None:
    clean = _as_fake(_capture())
    before = {label: _op(clean, label).out.clone() for label in clean.op_labels}
    edited = clean.fork().do(tl.func("relu"), lambda out, *, ctx: out)
    for label, value in before.items():
        assert torch.equal(_op(edited, label).out, value), label


def test_donor_patch_from_same_backend(fake_backend: ModuleType) -> None:
    torch.manual_seed(1)
    donor = _as_fake(tl.trace(_Net(), torch.randn(2, 4)))
    relu = _first(donor, "relu")
    donor_value = _op(donor, relu).out
    edited = _as_fake(_capture()).fork().do(relu, lambda out, *, ctx: donor_value)
    assert torch.equal(_op(edited, relu).out, donor_value)


def test_donor_from_another_framework_is_refused(fake_backend: ModuleType) -> None:
    clean = _as_fake(_capture())
    relu = _first(clean, "relu")
    foreign = np.zeros(tuple(_op(clean, relu).out.shape), dtype=np.float32)
    with pytest.raises(HookValueError) as caught:
        clean.fork().do(relu, lambda out, *, ctx: foreign)
    assert caught.value.fields["code"] == "intervention_replacement_invalid"


def test_differentiable_replay_is_refused(fake_backend: ModuleType) -> None:
    from torchlens.options import InterventionOptions

    with pytest.raises(ReplayPreconditionError) as caught:
        _as_fake(_capture()).fork().do(
            tl.func("relu"),
            tl.zero_ablate(),
            intervention=InterventionOptions(differentiable=True),
        )
    assert caught.value.fields["code"] == "backend_replay_not_differentiable"


def test_changed_guard_refuses_and_leaves_the_fork_unchanged(fake_backend: ModuleType) -> None:
    torch.manual_seed(0)
    clean = _as_fake(tl.trace(_GuardNet(), torch.randn(2, 4)))
    fork = clean.fork()
    before = _output(fork).clone()
    with pytest.raises(ReplayPreconditionError) as caught:
        fork.do(tl.func("relu"), lambda out, *, ctx: out - 100.0)
    assert caught.value.fields["code"] == "backend_replay_guard_changed"
    assert torch.equal(_output(fork), before)


def test_unchanged_guard_replays(fake_backend: ModuleType) -> None:
    torch.manual_seed(0)
    clean = _as_fake(tl.trace(_GuardNet(), torch.randn(2, 4)))
    edited = clean.fork().do(tl.func("relu"), lambda out, *, ctx: out * 2.0)
    assert _backend_replay.verify_backend_replay(edited) == []


def test_edited_fork_passes_the_replay_check(fake_backend: ModuleType) -> None:
    edited = _as_fake(_capture()).fork().do(tl.func("relu"), tl.zero_ablate())
    assert _backend_replay.verify_backend_replay(edited) == []
    assert edited.is_pristine is False


def test_corrupted_payload_fails_the_replay_check(fake_backend: ModuleType) -> None:
    edited = _as_fake(_capture()).fork().do(tl.func("relu"), tl.zero_ablate())
    tanh = _first(edited, "tanh")
    _op(edited, tanh)._internal_set("out", _op(edited, tanh).out + 1.0)
    assert tanh in _backend_replay.verify_backend_replay(edited)


def test_removed_edit_record_fails_the_replay_check(fake_backend: ModuleType) -> None:
    edited = _as_fake(_capture()).fork().do(tl.func("relu"), tl.zero_ablate())
    relu = _first(edited, "relu")
    _op(edited, relu)._internal_set("edits", [])
    _op(edited, relu)._internal_set("is_edited", False)
    assert relu in _backend_replay.verify_backend_replay(edited)


def test_capture_evidence_stamp() -> None:
    plain = _as_fake(_capture())
    assert evidence.trace_records_intervention_evidence(plain) is False
    evidence.stamp_capture_evidence(plain, intervened=False)
    assert evidence.trace_records_intervention_evidence(plain) is True

    edited = _as_fake(_capture())
    relu = _first(edited, "relu")
    evidence.stamp_capture_evidence(edited, intervened=True, edited_labels=[relu])
    assert _op(edited, relu).is_edited is True
    assert edited.is_pristine is False
