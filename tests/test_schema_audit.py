"""Schema audit tests for intervention Phase 1 fields."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens._run_state import RunState
from torchlens.constants import LAYER_PASS_LOG_FIELD_ORDER, MODEL_LOG_FIELD_ORDER
from torchlens.data_classes.layer_pass_log import LayerPassLog
from torchlens.data_classes.model_log import ModelLog
from torchlens.intervention.types import (
    LAYER_PASS_LOG_FORK_POLICY,
    MODEL_LOG_FORK_POLICY,
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


def _capture_tiny_log() -> ModelLog:
    """Capture a tiny ModelLog for schema tests.

    Returns
    -------
    ModelLog
        Captured model log.
    """

    return tl.log_forward_pass(_TinyModel(), torch.randn(2, 3))


@pytest.mark.smoke
def test_ordered_fields_have_phase1_lifecycle_policies() -> None:
    """Every ordered field has portable, fork, and default-fill policy coverage."""

    assert set(MODEL_LOG_FIELD_ORDER) <= set(ModelLog.PORTABLE_STATE_SPEC)
    assert set(MODEL_LOG_FIELD_ORDER) <= set(MODEL_LOG_FORK_POLICY)
    assert set(MODEL_LOG_FIELD_ORDER) <= set(ModelLog.DEFAULT_FILL_STATE)
    assert set(LAYER_PASS_LOG_FIELD_ORDER) <= set(LayerPassLog.PORTABLE_STATE_SPEC)
    assert set(LAYER_PASS_LOG_FIELD_ORDER) <= set(LAYER_PASS_LOG_FORK_POLICY)
    assert set(LAYER_PASS_LOG_FIELD_ORDER) <= set(LayerPassLog.DEFAULT_FILL_STATE)


@pytest.mark.smoke
def test_phase1_defaults_are_per_instance_and_fork_copy() -> None:
    """Container defaults are per-instance and intervention_log forks by copy."""

    log_a = ModelLog("a")
    log_b = ModelLog("b")
    log_a.operation_history.append({"op": "sentinel"})
    assert log_b.operation_history == []
    assert log_a.run_state is RunState.PRISTINE
    assert all(value is Relationship.UNKNOWN for value in log_a.relationship_evidence.values())

    pass_a = next(iter(_capture_tiny_log()))
    pass_b = next(iter(_capture_tiny_log()))
    pass_a.intervention_log.append({"op": "sentinel"})
    assert pass_b.intervention_log == []
    assert LAYER_PASS_LOG_FORK_POLICY["intervention_log"] is ForkFieldPolicy.FORK_COPY


@pytest.mark.smoke
def test_layer_pass_construction_guard_and_direct_write_flag() -> None:
    """Construction is internal, but user activation writes dirty the owning ModelLog."""

    log = _capture_tiny_log()
    layer = next(iter(log))
    fields_dict: dict[str, Any] = {
        field_name: getattr(layer, field_name) for field_name in LAYER_PASS_LOG_FIELD_ORDER
    }
    fields_dict["source_model_log"] = log
    fields_dict["_construction_done"] = True
    log._has_direct_writes = False
    constructed = LayerPassLog(fields_dict)
    assert constructed._construction_done is True
    assert log._has_direct_writes is False

    constructed.activation = constructed.activation
    assert log._has_direct_writes is True


@pytest.mark.smoke
def test_postprocess_internal_writes_do_not_mark_direct_write_dirty() -> None:
    """A normal capture leaves the direct-write flag unset after all postprocess steps."""

    log = _capture_tiny_log()
    assert log._has_direct_writes is False
    assert log.run_state is RunState.PRISTINE
    for layer in log:
        assert hasattr(layer, "func_call_id")
        assert hasattr(layer, "output_path")
        assert hasattr(layer, "intervention_log")
        assert hasattr(layer, "edge_uses")
        assert layer._construction_done is True
