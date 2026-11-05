"""``tl.validate(num_threads=..., trace_observer=...)``: the public read of the validation trace.

The observer contract: called exactly once per forward/saved run, with the validation
``Trace`` after replay and before cleanup (or ``None`` at a precondition refusal), never
when the call raises; it changes no check and no verdict. ``num_threads`` pins torch's
intra-op threads for the run and restores them on every exit.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._validation_entry_options as entry_options
from torchlens.errors import InvalidArgumentError
from torchlens.validation import last_validation_failure


class _SmallNet(nn.Module):
    """Two linear layers with a ReLU between them."""

    def __init__(self) -> None:
        """Build the layers with fixed weights."""

        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two layers."""

        return self.fc2(torch.relu(self.fc1(x)))


class _ThreadRecorder(nn.Module):
    """Record the torch intra-op thread count every forward runs at."""

    def __init__(self) -> None:
        """Build one linear layer and an empty record."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Record the thread count, then run the layer."""

        _THREADS_SEEN.append(torch.get_num_threads())
        return self.fc(x)


# Module-level ledger: validation snapshots and restores model attributes.
_THREADS_SEEN: list[int] = []


class _UnsnapshotableState(nn.Module):
    """A plain generator attribute the validation state snapshot cannot copy."""

    def __init__(self) -> None:
        """Hold a generator and a step counter."""

        super().__init__()
        self.pending = (step for step in range(3))
        self.step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Change graph shape after the first execution."""

        output = x + 1 if self.step == 0 else x * 2
        self.step += 1
        return output


def _recording_observer(calls: list[dict[str, Any] | None]) -> Any:
    """Return an observer that appends the public reads of each trace it sees."""

    def observe(validation_trace: Any) -> None:
        if validation_trace is None:
            calls.append(None)
            return
        status = validation_trace.validation_replay_status
        calls.append(
            {
                "num_ops": len(validation_trace.ops),
                "fingerprint": validation_trace.provenance.fingerprint,
                "state": status.state,
                "reason": status.reason,
                "num_failed_ops": status.num_failed_ops,
            }
        )

    return observe


@pytest.mark.parametrize("scope", ["forward", "saved"])
def test_observer_sees_validation_trace_once(scope: str) -> None:
    """The observer runs once and its public reads match a separate inspection."""

    model, x = _SmallNet().eval(), torch.randn(2, 4)
    calls: list[dict[str, Any] | None] = []
    passed = tl.validate(model, x, scope=scope, trace_observer=_recording_observer(calls))
    assert passed is True
    assert len(calls) == 1
    seen = calls[0]
    assert seen is not None
    separate = tl.trace(model, x)
    assert seen["num_ops"] == len(separate.ops)
    assert seen["fingerprint"] == separate.provenance.fingerprint
    assert seen["state"] == "passed"
    assert seen["reason"] is None
    assert seen["num_failed_ops"] == 0


def test_observer_does_not_change_the_verdict() -> None:
    """The same pair validates the same with and without an observer."""

    model, x = _SmallNet().eval(), torch.randn(2, 4)
    calls: list[dict[str, Any] | None] = []
    without = tl.validate(model, x, seed=3)
    with_observer = tl.validate(model, x, seed=3, trace_observer=_recording_observer(calls))
    assert without is with_observer is True
    assert len(calls) == 1


def test_observer_gets_none_at_precondition_refusal() -> None:
    """A refusal before any capture calls the observer once with ``None``."""

    calls: list[dict[str, Any] | None] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        passed = tl.validate(
            _UnsnapshotableState(), torch.randn(3), trace_observer=_recording_observer(calls)
        )
    assert passed is False
    assert calls == [None]
    refusal = last_validation_failure()
    assert refusal is not None and refusal.check_name == "precondition_refusal"


def test_num_threads_pins_forwards_and_restores_after_success() -> None:
    """Every validation forward runs at ``num_threads``; the prior count comes back."""

    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        _THREADS_SEEN.clear()
        assert tl.validate(_ThreadRecorder().eval(), torch.randn(2, 4), num_threads=1) is True
        assert _THREADS_SEEN and set(_THREADS_SEEN) == {1}
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(prior)


def test_raising_observer_propagates_and_restores_torch_state() -> None:
    """An observer's exception reaches the caller with threads and determinism restored."""

    class ObserverBoom(RuntimeError):
        """The observer's own error."""

    calls: list[int] = []

    def boom(validation_trace: Any) -> None:
        calls.append(1)
        raise ObserverBoom("observer failed")

    prior = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(2)
    try:
        with pytest.raises(ObserverBoom):
            tl.validate(_SmallNet().eval(), torch.randn(2, 4), num_threads=1, trace_observer=boom)
        assert calls == [1]
        assert torch.get_num_threads() == 2
        assert torch.are_deterministic_algorithms_enabled() is prior_deterministic
        assert torch.is_deterministic_algorithms_warn_only_enabled() is prior_warn_only
        assert torch.is_grad_enabled()
    finally:
        torch.set_num_threads(prior)


@pytest.mark.parametrize("scope", ["backward", "intervention", "receptive_field"])
@pytest.mark.parametrize("keyword", ["num_threads", "trace_observer"])
def test_other_scopes_refuse_the_keywords(scope: str, keyword: str) -> None:
    """Scopes that do not run the forward harness refuse typed, before any capture."""

    calls: list[dict[str, Any] | None] = []
    value: Any = 1 if keyword == "num_threads" else _recording_observer(calls)
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.validate(_SmallNet(), torch.randn(2, 4), scope=scope, **{keyword: value})
    assert excinfo.value.fields["code"] == "validation_scope_invalid"
    assert excinfo.value.fields["keyword"] == keyword
    assert excinfo.value.fields["remedy"]
    assert calls == []


def test_non_torch_backend_refuses_the_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-torch backend refuses num_threads and trace_observer by their public names."""

    monkeypatch.setattr(
        entry_options, "resolve_backend_spec", lambda *args: SimpleNamespace(name="jax")
    )
    with pytest.raises(tl.errors.BackendUnsupportedError) as excinfo:
        entry_options.resolve_forward_validation_entry(
            None, object(), (), None, "same", "auto", 1, lambda trace: None
        )
    assert excinfo.value.fields["code"] == "backend_unsupported"
    assert excinfo.value.fields["options"] == ["num_threads", "trace_observer"]


def test_torch_backend_forwards_only_non_default_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults forward nothing; the observer travels as the harness's private keyword."""

    monkeypatch.setattr(
        entry_options, "resolve_backend_spec", lambda *args: SimpleNamespace(name="torch")
    )

    def observer(trace: Any) -> None:
        return None

    _, defaults = entry_options.resolve_forward_validation_entry(
        None, object(), (), None, "same", "auto", None, None
    )
    assert defaults == {}
    _, forwarded = entry_options.resolve_forward_validation_entry(
        None, object(), (), None, "same", None, 1, observer
    )
    assert forwarded == {"save_budget": None, "num_threads": 1, "_trace_observer": observer}
