"""``run(inputs=)`` on a fork carrying value-edits discloses their inertness.

D1 (2026-08-19): ``fork.do(sel, edit)`` then ``fork.run(inputs=new_x)`` is
COHERENT -- a do() edit rewrites saved values (path 1), while a new-input run
is a fresh execution (path 2) -- but the un-edited VERIFIED result tripped a
real user flow. The contract is a plain DISCLOSURE at the run door, never a
refusal.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.errors import PendingValueEditsWarning


class Small(nn.Module):
    """Linear -> ReLU -> Linear."""

    def __init__(self) -> None:
        """Build the two linear stages."""

        super().__init__()
        self.a = nn.Linear(4, 4, bias=True)
        self.b = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the pipeline."""

        return self.b(torch.relu(self.a(x)))


def _traced() -> tuple[Small, torch.Tensor, tl.Trace]:
    """Return a live intervention-ready trace with its model kept alive."""

    torch.manual_seed(0)
    model = Small()
    x = torch.randn(2, 4)
    log = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True),
    )
    log._tl_test_model_keepalive = model
    return model, x, log


def test_new_input_run_on_value_edited_fork_discloses() -> None:
    """A fork with a do() value-edit warns on run(inputs=new_x)."""

    _model, _x, log = _traced()
    fork = log.fork()
    fork.do(
        tl.units("relu_1_2:1", [(0, 0)]).resolve(fork),
        tl.replace_with(torch.full((2, 4), 3.0)),
    )

    with pytest.warns(PendingValueEditsWarning, match="do NOT apply"):
        result = fork.run(inputs=torch.randn(2, 4))
    # Disclosure, not refusal: the run itself completes.
    assert result is not None


def test_new_input_run_on_pristine_trace_does_not_disclose() -> None:
    """No value-edits means no disclosure noise."""

    _model, _x, log = _traced()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PendingValueEditsWarning)
        result = log.run(inputs=torch.randn(2, 4))
    assert result is not None
