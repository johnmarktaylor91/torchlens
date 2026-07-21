"""r55 corr_3 immunizer -- first lazy capture under ambient ``inference_mode``.

MED: ``import torchlens`` followed by the FIRST ``tl.trace`` inside
``torch.inference_mode()`` crashed before capture. ``_callable_safety`` is imported
lazily by the first capture, and its import-time pure-view probe read
``probe._version``; an inference tensor does not track a version counter, so the
probe raised ``RuntimeError: Inference tensors do not track version counter`` and
blocked producing a runnable ``.tlspec`` under an ambient inference-mode capture.

The fix composes ``torch.inference_mode(False)`` + ``torch.enable_grad()`` into
``_mode_free_probe_context`` so the classifier probe is never steered by ambient
grad/inference state. A fresh subprocess is the only faithful test: the probe runs
exactly once, at first lazy import, so an in-process test that already warmed the
import cannot exercise it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.smoke


_FIRST_CAPTURE_UNDER_INFERENCE = textwrap.dedent(
    """
    import torch
    import torch.nn as nn
    import torchlens as tl

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)
        def forward(self, x):
            return torch.relu(self.fc(x))

    # FIRST torchlens capture in this process, under an ambient inference_mode:
    # this triggers the lazy _callable_safety import + pure-view probe.
    with torch.inference_mode():
        log = tl.trace(M().eval(), torch.randn(2, 4), intervention_ready=True)
    assert len(list(log)) > 0
    print("OK", len(list(log)))
    """
)


def test_first_capture_under_inference_mode_succeeds_fresh_process() -> None:
    """A fresh process whose FIRST ``tl.trace`` is under ``inference_mode`` succeeds."""

    completed = subprocess.run(
        [sys.executable, "-c", _FIRST_CAPTURE_UNDER_INFERENCE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"first inference-mode capture crashed:\nSTDOUT:{completed.stdout}\n"
        f"STDERR:{completed.stderr}"
    )
    assert "Inference tensors do not track version counter" not in completed.stderr
    assert "OK" in completed.stdout


def test_pure_view_classification_identical_inside_and_outside_inference_mode() -> None:
    """The pure-view classification is unchanged whether or not inference mode is active."""

    from torchlens.utils import _callable_safety as cs

    outside = frozenset(cs._compute_pure_view_property_names())
    with torch.inference_mode():
        inside = frozenset(cs._compute_pure_view_property_names())
    assert inside == outside
    # The probe must not have leaked inference/grad state to the caller.
    assert torch.is_grad_enabled()


def test_mode_free_probe_context_neutralizes_inference_and_grad() -> None:
    """Inside the probe context, inference mode is off and grad is on, regardless of ambient."""

    from torchlens.utils._callable_safety import _mode_free_probe_context

    with torch.inference_mode(), torch.no_grad():
        assert not torch.is_grad_enabled()  # ambient: grad OFF
        with _mode_free_probe_context():
            probe = torch.randn(2, 3, requires_grad=True)
            # A version-tracked, autograd-live tensor -- the read that used to crash.
            _ = probe._version
            assert probe.requires_grad
            assert torch.is_grad_enabled()  # neutralized: grad ON inside the probe
        # The probe restores the caller's ambient (grad OFF) on exit.
        assert not torch.is_grad_enabled()
