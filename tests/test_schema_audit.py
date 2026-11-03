"""Schema audit tests for intervention Phase 1 fields."""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens._trace_state import TraceState
from torchlens.intervention.types import (
    LAYER_PASS_LOG_FIELD_FORK_POLICY,
    ForkFieldPolicy,
    Relationship,
)


class _TinyModel(torch.nn.Module):
    """Minimal model for schema construction checks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a small graph that produces at least one operation node.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Relu output plus a scalar.
        """

        return torch.relu(x) + 1


def _capture_tiny_log() -> Trace:
    """Capture a tiny Trace for schema tests.

    Returns
    -------
    Trace
        Captured model log.
    """

    return tl.trace(_TinyModel(), torch.randn(2, 3))


def test_phase1_defaults_are_per_instance_and_fork_copy() -> None:
    """Container defaults are per-instance and interventions forks by copy."""

    log_a = Trace("a")
    log_b = Trace("b")
    log_a.state_history.append({"op": "sentinel"})
    assert log_b.state_history == []
    assert log_a.state is TraceState.PRISTINE
    assert all(
        value is Relationship.UNKNOWN for value in log_a.graph.relationship_evidence.values()
    )

    pass_a = next(iter(_capture_tiny_log()))
    pass_b = next(iter(_capture_tiny_log()))
    pass_a.edits.append({"op": "sentinel"})
    assert pass_b.edits == []
    assert LAYER_PASS_LOG_FIELD_FORK_POLICY["edits"] is ForkFieldPolicy.FORK_COPY
