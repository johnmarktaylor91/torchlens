"""Phase 4a intervention capture plumbing tests."""

from __future__ import annotations

import pytest
import torch
from example_models import TinyReluAdd as _TinyInterventionModel

import torchlens as tl
from torchlens import _state
from torchlens.options import CaptureOptions


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


def test_intervention_ready_sets_relationship_evidence() -> None:
    """``replayable=True`` marks the log and seeds relationship evidence."""

    model = _TinyInterventionModel()
    x = torch.randn(2, 3)

    log = tl.trace(model, x, capture=CaptureOptions(replayable=True))

    assert log.is_replayable is True
    assert log.provenance.model_object_id == id(model)
    assert log.source.class_qualname == f"{type(model).__module__}.{type(model).__qualname__}"
    assert log.provenance.param_hash_quick is not None
    assert log.provenance.param_hash_full == log.provenance.param_hash_quick
    assert log.provenance.input_object_id == id(x)
    assert log.provenance.input_signature_hash is not None
    assert log.options.save_args is False


def test_intervention_ready_allows_default_and_empty_layer_selections() -> None:
    """Only non-empty list selections conflict with intervention readiness."""

    tl.trace(
        _TinyInterventionModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(replayable=True),
    )
    tl.trace(
        _TinyInterventionModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(replayable=True, layers_to_save=[]),
    )


def test_func_call_id_is_assigned_and_shared_for_multi_output_calls() -> None:
    """All outputs of one decorated torch call share one pre-call ``func_call_id``."""

    log = tl.trace(
        _MultiOutputModel(),
        torch.randn(4, 5),
        capture=CaptureOptions(replayable=True),
    )

    ids = [layer.func_call_id for layer in log.layer_list if layer.func_call_id is not None]
    assert ids

    max_layers = [layer for layer in log.layer_list if layer.func_name == "max"]
    assert len(max_layers) == 2
    assert len({layer.func_call_id for layer in max_layers}) == 1


def test_active_logging_rejects_nested_entry_while_paused() -> None:
    """``pause_logging`` keeps ``_active_trace`` set for nested-capture rejection."""

    with pytest.raises(RuntimeError, match="not re-entrant"):
        with _state.active_logging(object()):  # type: ignore[arg-type]
            with _state.pause_logging():
                assert _state._logging_enabled is False
                assert _state._active_trace is not None
                with _state.active_logging(object()):  # type: ignore[arg-type]
                    pass

    assert _state._logging_enabled is False
    assert _state._active_trace is None
