"""Phase 4a intervention capture plumbing tests."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens import _state
from torchlens.intervention.errors import InterventionReadyConflictError


class _TinyInterventionModel(torch.nn.Module):
    """Small model for intervention-ready capture checks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple loggable graph.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output shifted by one.
        """

        return torch.relu(x) + 1


class _MultiOutputModel(torch.nn.Module):
    """Model with a multi-output torch operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use one torch call that returns multiple output tensors.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Values plus indices from ``torch.max``.
        """

        values, indices = torch.max(x, dim=1)
        return values + indices.to(values.dtype)


@pytest.mark.smoke
def test_intervention_ready_sets_relationship_evidence() -> None:
    """``intervention_ready=True`` marks the log and seeds relationship evidence."""

    model = _TinyInterventionModel()
    x = torch.randn(2, 3)

    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)

    assert log.intervention_ready is True
    assert log.source_model_id == id(model)
    assert log.source_model_class == f"{type(model).__module__}.{type(model).__qualname__}"
    assert log.weight_fingerprint_at_capture is not None
    assert log.weight_fingerprint_full == log.weight_fingerprint_at_capture
    assert log.input_id_at_capture == id(x)
    assert log.input_shape_hash is not None
    assert log.save_function_args is False


@pytest.mark.smoke
def test_intervention_ready_rejects_nonempty_layers_to_save_list() -> None:
    """Selective two-pass intervention readiness is deferred beyond Phase 4a."""

    with pytest.raises(InterventionReadyConflictError):
        tl.log_forward_pass(
            _TinyInterventionModel(),
            torch.randn(2, 3),
            vis_opt="none",
            intervention_ready=True,
            layers_to_save=["relu"],
        )


@pytest.mark.smoke
def test_intervention_ready_allows_default_and_empty_layer_selections() -> None:
    """Only non-empty list selections conflict with intervention readiness."""

    tl.log_forward_pass(
        _TinyInterventionModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )
    tl.log_forward_pass(
        _TinyInterventionModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
        layers_to_save=[],
    )


@pytest.mark.smoke
def test_func_call_id_is_assigned_and_shared_for_multi_output_calls() -> None:
    """All outputs of one decorated torch call share one pre-call ``func_call_id``."""

    log = tl.log_forward_pass(
        _MultiOutputModel(),
        torch.randn(4, 5),
        vis_opt="none",
        intervention_ready=True,
    )

    ids = [layer.func_call_id for layer in log.layer_list if layer.func_call_id is not None]
    assert ids

    max_layers = [layer for layer in log.layer_list if layer.func_name == "max"]
    assert len(max_layers) == 2
    assert len({layer.func_call_id for layer in max_layers}) == 1


@pytest.mark.smoke
def test_active_logging_rejects_nested_entry_while_paused() -> None:
    """``pause_logging`` keeps ``_active_model_log`` set for nested-capture rejection."""

    with pytest.raises(RuntimeError, match="not re-entrant"):
        with _state.active_logging(object()):  # type: ignore[arg-type]
            with _state.pause_logging():
                assert _state._logging_enabled is False
                assert _state._active_model_log is not None
                with _state.active_logging(object()):  # type: ignore[arg-type]
                    pass

    assert _state._logging_enabled is False
    assert _state._active_model_log is None
