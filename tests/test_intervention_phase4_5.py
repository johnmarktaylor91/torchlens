"""Phase 4.5 capture-time runtime smoke tests."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens import _state
from torchlens._run_state import RunState


class _TinyReluModel(torch.nn.Module):
    """Small model with a returned ReLU out."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.latest: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply and return a ReLU out.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU out.
        """

        self.latest = torch.relu(x)
        return self.latest


class _TinyShiftModel(torch.nn.Module):
    """Small model that exercises input, operation, and output log fields."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a ReLU and scalar shift.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Shifted ReLU out.
        """

        return torch.relu(x) + 1


def _bad_hook(out: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
    """Raise from a live hook.

    Parameters
    ----------
    out:
        Activation at the hook site.
    hook:
        Hook context.

    Returns
    -------
    torch.Tensor
        This function never returns.

    Raises
    ------
    ValueError
        Always raised to exercise exception-path cleanup.
    """

    raise ValueError("boom")


@pytest.mark.smoke
def test_non_ready_capture_matches_baseline_runtime_fields() -> None:
    """Default captures populate baseline fields and skip intervention-only data."""

    log = tl.trace(_TinyShiftModel(), torch.randn(2, 3), vis_opt="none")

    assert log.intervention_ready is False
    assert log.run_state is RunState.PRISTINE
    assert log.num_ops == 2
    assert log.layer_list
    assert log.layer_dict_main_keys
    assert log.layer_dict_all_keys
    assert log.layer_labels
    assert log.layer_labels
    assert log.op_labels
    assert log.layer_num_calls
    assert log.input_layers == ["input_1"]
    assert log.output_layers == ["output_1"]
    assert log.ops_with_saved_outs
    assert all(layer.out is not None for layer in log.layer_list)
    assert all(layer.args_template is None for layer in log.layer_list)
    assert all(layer.kwargs_template is None for layer in log.layer_list)
    assert all(layer.edge_uses == [] for layer in log.layer_list)
    assert all(layer.container_path == () for layer in log.layer_list)


@pytest.mark.smoke
def test_intervention_ready_without_hooks_preserves_returned_values() -> None:
    """Intervention-ready path-aware traversal does not alter no-hook outputs."""

    x = torch.randn(2, 3)
    baseline_model = _TinyReluModel()
    ready_model = _TinyReluModel()

    baseline_log = tl.trace(
        baseline_model,
        x,
        vis_opt="none",
        intervention_ready=False,
    )
    ready_log = tl.trace(
        ready_model,
        x,
        vis_opt="none",
        intervention_ready=True,
    )

    assert baseline_model.latest is not None
    assert ready_model.latest is not None
    assert torch.equal(ready_model.latest, baseline_model.latest)
    assert torch.equal(
        ready_log[ready_log.output_layers[0]].out,
        baseline_log[baseline_log.output_layers[0]].out,
    )


@pytest.mark.smoke
def test_live_hook_exception_resets_runtime_state_and_allows_next_capture() -> None:
    """Hook-raised exceptions leave no active capture state behind."""

    model = _TinyReluModel()

    with pytest.raises(ValueError, match="boom"):
        tl.trace(
            model,
            torch.randn(2, 3),
            vis_opt="none",
            intervention_ready=True,
            hooks={tl.func("relu"): _bad_hook},
        )

    assert _state._logging_enabled is False
    assert _state._active_trace is None
    assert _state._active_hook_plan is None

    followup_log = tl.trace(model, torch.randn(2, 3), vis_opt="none")

    assert followup_log.intervention_ready is False
    assert followup_log.run_state is RunState.PRISTINE
    assert followup_log.layer_list


@pytest.mark.smoke
def test_live_hook_exception_resets_reentrancy_depth() -> None:
    """Hook-raised exceptions reset the hook reentrancy guard."""

    with pytest.raises(ValueError, match="boom"):
        tl.trace(
            _TinyReluModel(),
            torch.randn(2, 3),
            vis_opt="none",
            intervention_ready=True,
            hooks={tl.func("relu"): _bad_hook},
        )

    assert _state._hook_reentrancy_depth == 0
