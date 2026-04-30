"""Lifecycle matrix audits for intervention-owned TorchLens fields."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._run_state import RunState
from torchlens.constants import LAYER_PASS_LOG_FIELD_ORDER, MODEL_LOG_FIELD_ORDER
from torchlens.data_classes.layer_pass_log import LayerPassLog
from torchlens.data_classes.model_log import ModelLog
from torchlens.intervention.types import LAYER_PASS_LOG_FORK_POLICY, MODEL_LOG_FORK_POLICY


class _LifecycleModel(nn.Module):
    """Small deterministic model for lifecycle transition tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a hookable operation and downstream arithmetic.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.relu(x) + 1


def _zero_hook(activation: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return a zero-valued activation.

    Parameters
    ----------
    activation:
        Activation supplied by TorchLens.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Zero-valued activation with the same shape as ``activation``.
    """

    del hook
    return activation * 0


def _capture_log() -> ModelLog:
    """Capture an intervention-ready test log.

    Returns
    -------
    ModelLog
        Captured model log.
    """

    return tl.log_forward_pass(
        _LifecycleModel(),
        torch.tensor([[-1.0, 2.0, 3.0]]),
        vis_opt="none",
        intervention_ready=True,
    )


def test_field_lifecycle_matrix_modellog() -> None:
    """Assert every ModelLog ordered field has lifecycle policy coverage."""

    ordered_fields = set(MODEL_LOG_FIELD_ORDER)
    assert ordered_fields <= set(ModelLog.PORTABLE_STATE_SPEC)
    assert ordered_fields <= set(MODEL_LOG_FORK_POLICY)
    assert ordered_fields <= set(ModelLog.DEFAULT_FILL_STATE)


def test_field_lifecycle_matrix_layer_pass_log() -> None:
    """Assert every LayerPassLog ordered field has lifecycle policy coverage."""

    ordered_fields = set(LAYER_PASS_LOG_FIELD_ORDER)
    assert ordered_fields <= set(LayerPassLog.PORTABLE_STATE_SPEC)
    assert ordered_fields <= set(LAYER_PASS_LOG_FORK_POLICY)
    assert ordered_fields <= set(LayerPassLog.DEFAULT_FILL_STATE)


def test_construction_done_lifecycle() -> None:
    """LayerPassLog construction starts guarded and ends with direct-write tracking enabled."""

    log = _capture_log()
    layer = next(iter(log))
    fields_dict = {
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
    assert log.run_state is RunState.DIRECT_WRITE_DIRTY


def test_run_state_transitions() -> None:
    """RunState transitions follow the allowed intervention lifecycle paths."""

    stale = _capture_log()
    assert stale.run_state is RunState.PRISTINE
    stale.attach_hooks(tl.func("relu"), _zero_hook, confirm_mutation=True)
    assert stale.run_state is RunState.SPEC_STALE
    stale.replay()
    assert stale.run_state is RunState.REPLAY_PROPAGATED

    rerun_log = _capture_log()
    assert rerun_log.run_state is RunState.PRISTINE
    rerun_log.attach_hooks(tl.func("relu"), _zero_hook, confirm_mutation=True)
    assert rerun_log.run_state is RunState.SPEC_STALE
    rerun_log.rerun(_LifecycleModel(), torch.tensor([[-1.0, 2.0, 3.0]]))
    assert rerun_log.run_state is RunState.RERUN_PROPAGATED

    live = tl.log_forward_pass(
        _LifecycleModel(),
        torch.tensor([[-1.0, 2.0, 3.0]]),
        vis_opt="none",
        intervention_ready=True,
        hooks={tl.func("relu"): _zero_hook},
    )
    assert live.run_state is RunState.LIVE_CAPTURED

    appended = _capture_log()
    appended.rerun(_LifecycleModel(), torch.tensor([[0.5, 1.0, 1.5]]), append=True)
    assert appended.run_state is RunState.APPENDED

    dirty = _capture_log()
    layer = next(layer for layer in dirty.layer_list if layer.func_name == "relu")
    layer.activation = layer.activation
    assert dirty.run_state is RunState.DIRECT_WRITE_DIRTY
