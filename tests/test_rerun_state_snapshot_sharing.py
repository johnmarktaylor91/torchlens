"""The rerun's state snapshot shares the source snapshot's unchanged clones.

An intervention-ready capture clones the model state before its forward pass.
A rerun re-executes the same model, so each state entry whose bytes are
unchanged since the source capture shares the source's immutable clone
instead of being cloned again; an entry that changed (an in-place weight
edit, a flipped sign bit) is cloned afresh. Either way the rerun's snapshot
holds exactly the bytes a fresh clone would, so a runnable save of any trace
a rerun produces behaves as it did when every entry was cloned.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.intervention.rerun as rerun_mod
from torchlens._runnable_state import snapshot_capture_state
from torchlens.data_classes.trace import Trace
from torchlens.options import CaptureOptions, InterventionOptions


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
        self.register_buffer("scale", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x))) * self.scale


def _capture() -> tuple[_Net, torch.Tensor, Trace]:
    torch.manual_seed(0)
    model = _Net().eval()
    x = torch.randn(2, 4)
    return model, x, tl.trace(model, x, capture=CaptureOptions(replayable=True))


@pytest.fixture
def fresh_reruns(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Trace]]:
    """Collect every rerun's own fresh trace (the one the replace path adopts)."""

    grabbed: list[Trace] = []
    original = rerun_mod._capture_with_active_spec

    def grab(*args: Any, **kwargs: Any) -> Trace:
        new_log = original(*args, **kwargs)
        grabbed.append(new_log)
        return new_log

    monkeypatch.setattr(rerun_mod, "_capture_with_active_spec", grab)
    yield grabbed


def _rerun(trace: Trace, model: nn.Module, x: torch.Tensor) -> Trace:
    return trace.fork().do(
        tl.func("relu"),
        tl.scale(2.0),
        model=model,
        x=x,
        intervention=InterventionOptions(engine="rerun"),
    )


def _raw_bytes(value: torch.Tensor) -> bytes:
    return value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _assert_snapshot_matches_fresh_clone(snapshot: Any, model: nn.Module) -> None:
    fresh = snapshot_capture_state(model)
    assert fresh is not None and snapshot is not None
    assert list(snapshot) == list(fresh)
    for name, value in fresh.items():
        shared = snapshot[name]
        assert (shared.dtype, shared.shape, shared.stride(), shared.device) == (
            value.dtype,
            value.shape,
            value.stride(),
            value.device,
        )
        assert _raw_bytes(shared) == _raw_bytes(value)


def test_unchanged_state_shares_every_source_clone(fresh_reruns: list[Trace]) -> None:
    model, x, trace = _capture()
    source = trace._runnable.capture_state
    _rerun(trace, model, x)
    rerun_state = fresh_reruns[-1]._runnable.capture_state
    assert set(rerun_state) == set(source)
    assert all(rerun_state[name] is source[name] for name in source)
    _assert_snapshot_matches_fresh_clone(rerun_state, model)


def test_a_changed_entry_is_cloned_and_the_source_is_untouched(
    fresh_reruns: list[Trace],
) -> None:
    model, x, trace = _capture()
    source = trace._runnable.capture_state
    recorded = _raw_bytes(source["fc1.weight"])
    with torch.no_grad():
        model.fc1.weight.add_(1.0)
    _rerun(trace, model, x)
    rerun_state = fresh_reruns[-1]._runnable.capture_state
    assert rerun_state["fc1.weight"] is not source["fc1.weight"]
    assert _raw_bytes(source["fc1.weight"]) == recorded
    assert all(rerun_state[name] is source[name] for name in source if name != "fc1.weight")
    _assert_snapshot_matches_fresh_clone(rerun_state, model)


def test_a_flipped_sign_bit_is_not_shared(fresh_reruns: list[Trace]) -> None:
    torch.manual_seed(0)
    model = _Net().eval()
    with torch.no_grad():
        model.fc2.bias.zero_()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(replayable=True))
    with torch.no_grad():
        model.fc2.bias.neg_()  # 0.0 -> -0.0: equal values, different bytes
    _rerun(trace, model, x)
    rerun_state = fresh_reruns[-1]._runnable.capture_state
    assert rerun_state["fc2.bias"] is not trace._runnable.capture_state["fc2.bias"]
    assert torch.signbit(rerun_state["fc2.bias"]).all()
    _assert_snapshot_matches_fresh_clone(rerun_state, model)


def _save_outcome(trace: Trace, path: Path) -> Any:
    try:
        trace.save(path, level="runnable")
    except Exception as exc:  # noqa: BLE001
        fields = getattr(exc, "fields", None) or {}
        return (type(exc).__name__, fields.get("code"))
    return "saved"


def test_runnable_saves_behave_as_with_full_clones(
    fresh_reruns: list[Trace], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = []
    for sharing in (True, False):
        if not sharing:
            monkeypatch.setattr(
                rerun_mod,
                "_shared_or_cloned_state_entry",
                lambda _source, _name, value: value.detach().clone(),
            )
        model, x, trace = _capture()
        fork = _rerun(trace, model, x)
        fresh = fresh_reruns[-1]
        shared = all(
            fresh._runnable.capture_state[name] is trace._runnable.capture_state[name]
            for name in trace._runnable.capture_state
        )
        assert shared is sharing
        outcomes.append(
            (
                _save_outcome(trace, tmp_path / f"source-{sharing}.tlspec"),
                _save_outcome(fork, tmp_path / f"fork-{sharing}.tlspec"),
                _save_outcome(fresh, tmp_path / f"fresh-{sharing}.tlspec"),
            )
        )
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == "saved"
