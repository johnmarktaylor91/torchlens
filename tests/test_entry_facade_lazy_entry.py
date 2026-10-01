"""Lazy-module entry teaching refusals (quickstart memo B13 / B14, reconciled).

The wave-1a all-verbs entry refusal flipped off on the memo's own signal
(4.4) when the numbers-truth lane landed the completion unit: executed lazy
modules materialize during the ONE captured forward and never-run lazy
PARAMETERS stay at zero geometry in the inventory (pinned by
``tests/test_numbers_truth_execution.py`` -- not re-pinned here). The typed
``lazy_uninitialized`` teach now fires exactly where the request is
genuinely unanswerable: pending lazy BUFFERS (the capture-boundary
buffer-write tracker cannot index storage that does not exist yet), and the
armed-lane state baseline refuses typed (``state_baseline_unavailable``)
because a pending slot has no bytes to witness. The refusals leave the
model untouched, and the measured self-prime remedy actually works.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._robustness import check_lazy_state
from torchlens._runnable_state import snapshot_capture_state
from torchlens.errors import CaptureContextError, LazyStateUnsupportedError
from torchlens.utils.lazy_state import has_uninitialized_lazy_state, pending_lazy_state

pytestmark = pytest.mark.smoke


def _lazy_mlp() -> nn.Sequential:
    """Return a fresh model whose head is an un-materialized LazyLinear."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.LazyLinear(8))


def _lazy_bn() -> nn.Sequential:
    """Return a fresh model with un-materialized lazy BUFFERS (running stats)."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.LazyBatchNorm1d())


def _pending(model: nn.Module) -> bool:
    """Return whether any lazy module in ``model`` is still pending."""

    return has_uninitialized_lazy_state(model)


def test_entry_gate_tolerates_pending_lazy_params() -> None:
    """Pending lazy PARAMETERS alone never refuse at entry (completion landed)."""

    assert check_lazy_state(_lazy_mlp()) is None


def test_record_tolerates_pending_lazy_params() -> None:
    """tl.record on a pending-param model captures; the module materializes."""

    model = _lazy_mlp()
    recording = tl.record(model, torch.randn(2, 4), save=tl.func("relu"))
    assert recording.status == "complete"
    assert not _pending(model), "the captured forward materializes the lazy head"


def test_trace_refuses_lazy_buffer_model_typed_and_untouched() -> None:
    """Pending lazy BUFFERS refuse at entry: no storage to index pre-forward."""

    model = _lazy_bn()
    with pytest.raises(LazyStateUnsupportedError) as excinfo:
        tl.trace(model, torch.randn(3, 4))
    err = excinfo.value
    assert err.fields["code"] == "lazy_uninitialized"
    assert "BUFFERS" in str(err)
    assert "LazyBatchNorm1d" in str(err)
    assert "with torch.no_grad(): model(x)" in str(err)
    assert err.fields["pending_buffers"], "lazy running stats must be enumerated"
    buffer_names = {name for name, _ in err.fields["pending_buffers"]}
    assert {"1.running_mean", "1.running_var"} <= buffer_names
    assert _pending(model), "the refusal must leave the model untouched"


def test_record_refuses_lazy_buffer_model_typed() -> None:
    """tl.record rides the same buffer-scoped entry gate."""

    model = _lazy_bn()
    with pytest.raises(LazyStateUnsupportedError) as excinfo:
        tl.record(model, torch.randn(3, 4), save=tl.func("linear"))
    assert excinfo.value.fields["code"] == "lazy_uninitialized"
    assert _pending(model)


def test_self_prime_remedy_unlocks_capture() -> None:
    """The taught two-line self-prime materializes and capture then succeeds."""

    model = _lazy_bn()
    x = torch.randn(3, 4)
    with torch.no_grad():
        model(x)
    assert not _pending(model)
    log = tl.trace(model, x)
    assert log.num_ops >= 2
    log.cleanup()


def test_pending_enumeration_is_id_keyed_and_falsy_when_clear() -> None:
    """pending_lazy_state keys by id() and answers falsy after materialization."""

    model = _lazy_mlp()
    pending = pending_lazy_state(model)
    assert bool(pending) is True
    lazy_module = model[2]
    assert pending.modules[0][2] == id(lazy_module)
    param_ids = {tensor_id for _, tensor_id in pending.parameters}
    assert id(lazy_module.weight) in param_ids and id(lazy_module.bias) in param_ids
    with torch.no_grad():
        model(torch.randn(2, 4))
    assert bool(pending_lazy_state(model)) is False


def test_armed_lane_state_baseline_refuses_typed() -> None:
    """snapshot_capture_state refuses pending slots before any mutation."""

    model = _lazy_mlp()
    with pytest.raises(CaptureContextError) as excinfo:
        snapshot_capture_state(model)
    err = excinfo.value
    assert err.fields["code"] == "state_baseline_unavailable"
    assert set(err.fields["pending_slots"]) == {"2.weight", "2.bias"}
    assert "no_grad" in str(err)
    assert _pending(model)


def test_snapshot_clone_failures_degrade_to_none_not_crash() -> None:
    """The clone loop sits inside its guard: unclonable state answers None."""

    class _HostileTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN206
            if func is torch.Tensor.clone or getattr(func, "__name__", "") == "clone":
                raise RuntimeError("clone refused")
            return super().__torch_function__(func, types, args, kwargs or {})

    class _HostileState(nn.Module):
        def state_dict(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:  # type: ignore[override]
            return {"w": torch.randn(2).as_subclass(_HostileTensor)}

    assert snapshot_capture_state(_HostileState()) is None


def test_capture_failure_advisory_scopes_the_restoration_claim() -> None:
    """The failed-capture advisory no longer claims the MODEL was restored."""

    class _Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("mid-forward failure")

    with (
        pytest.warns(Warning) as caught,
        pytest.raises(RuntimeError, match="mid-forward failure"),
    ):
        tl.trace(_Boom(), torch.randn(2, 4))
    advisories = [
        str(w.message) for w in caught if "TorchLens capture attempt failed" in str(w.message)
    ]
    assert advisories, "the CaptureAttemptFailedWarning advisory must still fire"
    for message in advisories:
        assert "the model and torch environment were restored" not in message
        assert "not rolled back" in message
