"""Schema audit tests for intervention Phase 1 fields."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens._trace_state import TraceState
from torchlens.constants import LAYER_PASS_LOG_FIELD_ORDER, MODEL_LOG_FIELD_ORDER
from torchlens.data_classes.op_log import Op
from torchlens.data_classes.model_log import Trace
from torchlens.intervention.types import (
    LAYER_PASS_LOG_FIELD_FORK_POLICY,
    MODEL_LOG_FIELD_FORK_POLICY,
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


@pytest.mark.smoke
def test_ordered_fields_have_phase1_lifecycle_policies() -> None:
    """Every ordered field has portable, fork, and default-fill policy coverage."""

    assert set(MODEL_LOG_FIELD_ORDER) <= set(Trace.PORTABLE_STATE_SPEC)
    assert set(MODEL_LOG_FIELD_ORDER) <= set(MODEL_LOG_FIELD_FORK_POLICY)
    assert set(MODEL_LOG_FIELD_ORDER) <= set(Trace.DEFAULT_FILL_STATE)
    assert set(LAYER_PASS_LOG_FIELD_ORDER) <= set(Op.PORTABLE_STATE_SPEC)
    assert set(LAYER_PASS_LOG_FIELD_ORDER) <= set(LAYER_PASS_LOG_FIELD_FORK_POLICY)
    assert set(LAYER_PASS_LOG_FIELD_ORDER) <= set(Op.DEFAULT_FILL_STATE)


@pytest.mark.smoke
def test_phase1_defaults_are_per_instance_and_fork_copy() -> None:
    """Container defaults are per-instance and interventions forks by copy."""

    log_a = Trace("a")
    log_b = Trace("b")
    log_a.state_history.append({"op": "sentinel"})
    assert log_b.state_history == []
    assert log_a.state is TraceState.PRISTINE
    assert all(value is Relationship.UNKNOWN for value in log_a.relationship_evidence.values())

    pass_a = next(iter(_capture_tiny_log()))
    pass_b = next(iter(_capture_tiny_log()))
    pass_a.interventions.append({"op": "sentinel"})
    assert pass_b.interventions == []
    assert LAYER_PASS_LOG_FIELD_FORK_POLICY["interventions"] is ForkFieldPolicy.FORK_COPY


@pytest.mark.smoke
def test_layer_pass_construction_guard_and_direct_write_flag() -> None:
    """Construction is internal, but user out writes dirty the owning Trace."""

    log = _capture_tiny_log()
    layer = next(iter(log))
    fields_dict: dict[str, Any] = {
        field_name: getattr(layer, field_name) for field_name in LAYER_PASS_LOG_FIELD_ORDER
    }
    fields_dict["source_trace"] = log
    fields_dict["_construction_done"] = True
    log._has_direct_writes = False
    constructed = Op(fields_dict)
    assert constructed._construction_done is True
    assert log._has_direct_writes is False

    constructed.out = constructed.out
    assert log._has_direct_writes is True


@pytest.mark.smoke
def test_postprocess_internal_writes_do_not_mark_direct_write_dirty() -> None:
    """A normal capture leaves the direct-write flag unset after all postprocess steps."""

    log = _capture_tiny_log()
    assert log._has_direct_writes is False
    assert log.state is TraceState.PRISTINE
    for layer in log:
        assert hasattr(layer, "func_call_id")
        assert hasattr(layer, "container_path")
        assert hasattr(layer, "interventions")
        assert hasattr(layer, "_edge_uses")
        assert not hasattr(layer, "edge_uses")
        assert layer._construction_done is True
