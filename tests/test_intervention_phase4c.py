"""Phase 4c live hook execution tests."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._run_state import RunState
from torchlens.intervention.errors import LiveModeLabelError


class _ReluReturnModel(torch.nn.Module):
    """Model that exposes the returned ReLU activation for comparison."""

    def __init__(self) -> None:
        """Initialize the model with no captured output."""

        super().__init__()
        self.latest: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a single ReLU activation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU activation after live hooks.
        """

        self.latest = torch.relu(x)
        return self.latest


class _LinearModel(torch.nn.Module):
    """Single-module model for capture-time module selectors."""

    def __init__(self) -> None:
        """Initialize a linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear module.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear output.
        """

        return self.linear(x)


def _zero_hook(activation: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
    """Return a zeroed activation.

    Parameters
    ----------
    activation:
        Activation at the hook site.
    hook:
        Hook context.

    Returns
    -------
    torch.Tensor
        Zeroed activation with matching metadata.
    """

    return activation * 0


def _identity_hook(activation: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
    """Return an activation unchanged.

    Parameters
    ----------
    activation:
        Activation at the hook site.
    hook:
        Hook context.

    Returns
    -------
    torch.Tensor
        Original activation.
    """

    return activation


@pytest.mark.smoke
def test_live_func_hook_replaces_returned_and_saved_activation() -> None:
    """Live post-hooks run before saving so returned and saved activations match."""

    model = _ReluReturnModel()
    log = tl.log_forward_pass(
        model,
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
        hooks={tl.func("relu"): _zero_hook},
    )

    relu_layer = next(layer for layer in log.layer_list if layer.func_name == "relu")

    assert log.run_state is RunState.LIVE_CAPTURED
    assert model.latest is not None
    assert relu_layer.activation is not None
    assert torch.equal(model.latest, relu_layer.activation)
    assert torch.count_nonzero(relu_layer.activation) == 0
    assert len(relu_layer.intervention_log) == 1
    assert relu_layer.intervention_log[0].timing == "post"
    assert relu_layer.intervention_log[0].direction == "forward"


@pytest.mark.smoke
def test_live_label_error_for_finalized_style_label() -> None:
    """Finalized postprocess labels fail loudly in live capture."""

    with pytest.raises(LiveModeLabelError, match="tl.where"):
        tl.log_forward_pass(
            _ReluReturnModel(),
            torch.randn(2, 3),
            vis_opt="none",
            intervention_ready=True,
            hooks={tl.label("relu_4_27:2"): _identity_hook},
        )


@pytest.mark.smoke
def test_module_selector_matches_capture_time_module_context() -> None:
    """Module selectors can match live capture-time module context."""

    log = tl.log_forward_pass(
        _LinearModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
        hooks={tl.module("linear"): _zero_hook},
    )

    hooked_layers = [layer for layer in log.layer_list if layer.intervention_log]

    assert hooked_layers
    assert hooked_layers[0].activation is not None
    assert torch.count_nonzero(hooked_layers[0].activation) == 0


@pytest.mark.smoke
def test_raw_label_where_and_in_module_selectors_work_at_capture_time() -> None:
    """Raw labels, predicates, and module containment selectors resolve live."""

    raw_log = tl.log_forward_pass(
        _ReluReturnModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )
    raw_label = next(
        layer.layer_label_raw for layer in raw_log.layer_list if layer.func_name == "relu"
    )

    label_log = tl.log_forward_pass(
        _ReluReturnModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
        hooks={tl.label(raw_label): _zero_hook},
    )
    where_log = tl.log_forward_pass(
        _ReluReturnModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
        hooks={tl.where(lambda p: p.func_name == "relu"): _zero_hook},
    )
    in_module_log = tl.log_forward_pass(
        _LinearModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
        hooks={tl.in_module("linear"): _zero_hook},
    )

    assert any(layer.intervention_log for layer in label_log.layer_list)
    assert any(layer.intervention_log for layer in where_log.layer_list)
    assert any(layer.intervention_log for layer in in_module_log.layer_list)


@pytest.mark.smoke
def test_no_hooks_preserves_pristine_run_state() -> None:
    """Intervention-ready capture without hooks stays pristine."""

    log = tl.log_forward_pass(
        _ReluReturnModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )

    assert log.run_state is RunState.PRISTINE


@pytest.mark.smoke
def test_live_replacement_metadata_matches_saved_activation() -> None:
    """Hook replacement refreshes tensor metadata and saved-activation flags."""

    log = tl.log_forward_pass(
        _ReluReturnModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
        hooks={tl.func("relu"): _zero_hook},
    )
    relu_layer = next(layer for layer in log.layer_list if layer.func_name == "relu")

    assert relu_layer.activation is not None
    assert relu_layer.has_saved_activations is True
    assert relu_layer.tensor_shape == tuple(relu_layer.activation.shape)
    assert relu_layer.tensor_dtype == relu_layer.activation.dtype
    assert (
        relu_layer.tensor_memory
        == relu_layer.activation.nelement() * relu_layer.activation.element_size()
    )
