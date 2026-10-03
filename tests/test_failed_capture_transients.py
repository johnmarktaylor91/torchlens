"""R11/R32: a FAILED capture's escaping trace drops its capture transients.

The success arms pop ``_output_attribution_input_tensors`` (live USER INPUT
tensors) and postprocess replaces the mutable ``capture_events`` alias with the
payload-free ``_capture_events`` home. The failure/interrupt arms reached
neither, so the trace escaping on ``exc.partial_log``:

* pickled the user's live input tensors through an undeclared attribute
  (``_output_attribution_input_tensors``), and
* retained EVERY activation payload and grad_fn handle of the failed run
  through the undeclared ``capture_events`` alias (the sidecar-release
  epilogue was success-path-only), measured GB-class on real models.

These tests are the red-capable seeds for the failure-axis scrub: they FAIL on
the pre-fix tree and pin the fix. Partial diagnostics (raw layers, nonfinite
summary, draw) must survive the scrub -- ``PartialTrace.from_trace``
materializes raw layers before the scrub runs.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import Trace

# Facade plumbing that legitimately owns no declared field (mirrors the
# lockstep gate's allowance; kept local so this file stays self-contained).
_FACADE_PLUMBING_ATTRS = frozenset({"_core", "_row", "_store", "_facets"})


class _Boom(nn.Module):
    """Model that fails mid-forward after producing real captured ops."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        torch.relu(self.fc(x))
        raise RuntimeError("boom")


class _Interrupt(BaseException):
    """Custom BaseException standing in for KeyboardInterrupt."""


class _BoomBase(nn.Module):
    """Model that raises a BaseException (interrupt axis) mid-forward."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        torch.relu(self.fc(x))
        raise _Interrupt("interrupted")


def _failed_partial() -> tl.partial.PartialTrace:
    with pytest.raises(RuntimeError) as excinfo:
        tl.trace(_Boom().eval(), torch.ones(1, 3))
    partial = getattr(excinfo.value, "partial_log", None)
    assert partial is not None, "failed torch capture must attach partial_log"
    return partial


def _undeclared(trace: Trace) -> list[str]:
    return sorted(set(vars(trace)) - set(type(trace).FIELD_POLICY) - _FACADE_PLUMBING_ATTRS)


def test_failed_capture_trace_carries_no_undeclared_attributes() -> None:
    """R11: every attribute the escaping failed trace carries is declared.

    Pre-fix this fails with exactly the two leaked names:
    ``capture_events`` and ``_output_attribution_input_tensors``.
    """

    partial = _failed_partial()
    assert _undeclared(partial.trace) == []


def test_failed_capture_drops_live_input_tensors() -> None:
    """R11 HIGH: live user input tensors never survive onto the partial."""

    partial = _failed_partial()
    assert "_output_attribution_input_tensors" not in vars(partial.trace)
    restored = pickle.loads(pickle.dumps(partial.trace))
    assert "_output_attribution_input_tensors" not in vars(restored)
    assert "capture_events" not in vars(restored)


def test_failed_capture_releases_runtime_sidecars() -> None:
    """R32-F1: the sidecar-release epilogue runs on the failure axis too.

    The mutable ``capture_events`` alias is replaced by the payload-free
    ``_capture_events`` home: grad_fn handles dropped, backend session
    detached, op-event payloads stripped to structural facts.
    """

    partial = _failed_partial()
    trace_dict = vars(partial.trace)
    assert "capture_events" not in trace_dict
    events = trace_dict.get("_capture_events")
    assert events is not None
    assert not events.grad_fn_handles_by_label_raw
    assert getattr(events, "backend_session", None) is None
    for event in events.op_events:
        core = getattr(event, "core", event)
        output = getattr(core, "output", None)
        tensor = getattr(output, "tensor", None)
        if tensor is not None:
            assert getattr(tensor, "payload", None) is None


def test_interrupted_capture_propagates_and_next_capture_is_clean() -> None:
    """The BaseException (interrupt) arm scrubs and does not poison the process."""

    model = _BoomBase().eval()
    with pytest.raises(_Interrupt):
        tl.trace(model, torch.ones(1, 3))
    trace = tl.trace(nn.Sequential(nn.Linear(3, 4)).eval(), torch.ones(1, 3))
    assert "capture_events" not in vars(trace)
    assert "_output_attribution_input_tensors" not in vars(trace)
    trace.cleanup()


@pytest.mark.smoke
def test_partial_diagnostics_survive_the_scrub() -> None:
    """Recovery stays intact: raw layers, nonfinite summary, DOT render."""

    partial = _failed_partial()
    assert len(partial.raw_layers) > 0
    assert isinstance(partial.first_nonfinite(), str)
    assert isinstance(partial.draw(), str)


def test_partial_trace_declares_a_field_policy() -> None:
    """R11: ``PartialTrace`` has a FIELD_POLICY so the lockstep gate can see it."""

    from torchlens.partial import PartialTrace

    policy = getattr(PartialTrace, "FIELD_POLICY", None)
    assert policy is not None, "PartialTrace must declare FIELD_POLICY"
    partial = _failed_partial()
    undeclared = set(vars(partial)) - set(policy) - _FACADE_PLUMBING_ATTRS
    assert not undeclared, f"PartialTrace carries undeclared attributes: {sorted(undeclared)}"
