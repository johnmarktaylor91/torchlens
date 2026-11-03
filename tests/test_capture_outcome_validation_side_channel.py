"""B1-04 (SF-39): validate-then-save must not hard-refuse.

``validate_forward_pass`` / ``validate_saved_outs`` are public methods on a
user-held Trace, and validation ENTRY unconditionally writes the failure side
channel: ``reset_validation_failure`` sets ``_last_validation_failure = None``
on every run (``validation/core.py``), and non-failing diagnostics accumulate
in ``_validation_diagnostics``. Neither name had a ``_PORTABLE_STATE_SPEC`` /
``_FIELD_POLICY`` row, so the portable scrub's undeclared-field audit rejected
them and a plain validate-then-save sequence died with

    ArtifactError: Trace._last_validation_failure is missing from
    _PORTABLE_STATE_SPEC

Both are session-time diagnostics carried on the trace as a side channel --
the ``_fast_run_session`` class -- so they are declared ``FieldPolicy.DROP``.
Like ``_fast_run_session`` they stay in plain pickle (in-process round-trips
legitimately carry session state); the DROP row governs the portable artifact.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _Small(nn.Module):
    """Two-op model with a real replayable graph."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.relu(self.fc(x))


def _traced() -> tuple[tl.Trace, torch.Tensor]:
    model = _Small().eval()
    x = torch.ones(1, 3)
    ground_truth = model(x)
    return tl.trace(model, x), ground_truth


def test_repeated_validate_save_cycles_stay_saveable(tmp_path) -> None:
    """Validation is re-entrant against saving; no one-shot escape."""

    trace, ground_truth = _traced()
    for index in range(3):
        trace.validate_forward_pass([ground_truth])
        trace.save(str(tmp_path / f"cycle_{index}.tlspec"))
