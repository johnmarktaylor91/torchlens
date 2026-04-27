"""Smoke tests for first-class backward-pass capture."""

import torch
from torch import nn
import pytest

import torchlens as tl


class _TinyBackwardModel(nn.Module):
    """Small MLP with view op coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        hidden = torch.relu(self.fc1(x))
        viewed = hidden.view(hidden.shape[0], 4)
        return self.fc2(viewed)


class _DoubleFn(torch.autograd.Function):
    """Custom autograd function for grad_fn classification tests."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Return doubled input."""
        return x * 2

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        """Return doubled upstream gradient."""
        return grad * 2


class _CustomModel(nn.Module):
    """Model using a custom autograd Function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return _DoubleFn.apply(x).sum()


def _logged_model(
    *,
    layers_to_save: str | list[str] | None = "all",
    gradients_to_save: str | list[str] | None = "all",
) -> tuple[nn.Module, torch.Tensor, tl.ModelLog]:
    """Create a logged tiny model.

    Returns
    -------
    tuple[nn.Module, torch.Tensor, tl.ModelLog]
        Model, input tensor, and model log.
    """
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    model_log = tl.log_forward_pass(
        model,
        x,
        layers_to_save=layers_to_save,
        gradients_to_save=gradients_to_save,
    )
    return model, x, model_log


def _output_loss(model_log: tl.ModelLog) -> torch.Tensor:
    """Return scalar sum loss from the logged output activation."""
    return model_log[model_log.output_layers[0]].activation.sum()


@pytest.mark.smoke
def test_log_backward_captures_per_layer_gradients() -> None:
    """log_backward captures saved per-layer gradients."""
    _model, _x, model_log = _logged_model()
    model_log.log_backward(_output_loss(model_log))
    assert model_log.has_gradients
    assert len(model_log.layers_with_saved_gradients) > 0
    assert all(
        model_log[label].gradient is not None for label in model_log.layers_with_saved_gradients
    )


@pytest.mark.smoke
def test_recording_backward_context_manager() -> None:
    """recording_backward accumulates multiple backward calls."""
    _model, _x, model_log = _logged_model()
    loss = _output_loss(model_log)
    with model_log.recording_backward():
        loss.backward(retain_graph=True)
        (loss * 2).backward()
    assert model_log.backward_num_passes == 2


@pytest.mark.smoke
def test_backward_graph_walk_includes_intervening_grad_fns() -> None:
    """The backward DAG includes grad_fns without forward LayerLog matches."""
    _model, _x, model_log = _logged_model()
    model_log.log_backward(_output_loss(model_log))
    assert any(grad_fn.is_intervening for grad_fn in model_log.grad_fn_logs.values())


@pytest.mark.smoke
def test_corresponding_grad_fn_back_pointer() -> None:
    """Forward LayerLogs link to corresponding GradFnLogs by identity."""
    _model, _x, model_log = _logged_model()
    model_log.log_backward(_output_loss(model_log))
    assert any(layer.corresponding_grad_fn is not None for layer in model_log.layer_list)
    assert any(
        grad_fn.corresponding_layer is not None
        and grad_fn.corresponding_layer.corresponding_grad_fn is grad_fn
        for grad_fn in model_log.grad_fns
    )


@pytest.mark.smoke
def test_grad_fn_naming_and_indexing() -> None:
    """GradFnLog labels and accessor indexing mirror layer lookup patterns."""
    _model, _x, model_log = _logged_model()
    model_log.log_backward(_output_loss(model_log))
    first_grad_fn = model_log.grad_fns[0]
    assert "_back_" in first_grad_fn.label
    assert first_grad_fn.label == first_grad_fn.label.lower()
    assert model_log.grad_fns[first_grad_fn.label] is first_grad_fn
    assert model_log.grad_fns[first_grad_fn.grad_fn_type] is first_grad_fn
    if first_grad_fn.num_passes:
        assert model_log.grad_fns[f"{first_grad_fn.label}:1"] is first_grad_fn.passes[1]
    assert list(model_log.grad_fns)


@pytest.mark.smoke
def test_gradients_to_save_default_matches_layers_to_save() -> None:
    """save_gradients uses layers_to_save when gradients_to_save is omitted."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    model_log = tl.log_forward_pass(model, x, layers_to_save=["relu"], save_gradients=True)
    model_log.log_backward(_output_loss(model_log))
    assert model_log.layers_with_saved_gradients
    assert all("relu" in label for label in model_log.layers_with_saved_gradients)


@pytest.mark.smoke
def test_gradients_to_save_independent_override() -> None:
    """gradients_to_save can be broader than layers_to_save."""
    _model, _x, model_log = _logged_model(layers_to_save="all", gradients_to_save=["relu"])
    model_log.log_backward(_output_loss(model_log))
    assert model_log.layers_with_saved_gradients
    assert all("relu" in label for label in model_log.layers_with_saved_gradients)


@pytest.mark.smoke
def test_auto_train_mode_when_backward_opted_in() -> None:
    """Explicit gradients_to_save auto-enables train_mode."""
    _model, _x, model_log = _logged_model()
    assert model_log.train_mode is True


@pytest.mark.smoke
def test_auto_train_mode_conflict_with_explicit_false() -> None:
    """Explicit train_mode=False conflicts with backward capture."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="requires train_mode=True"):
        tl.log_forward_pass(model, x, gradients_to_save="all", train_mode=False)


@pytest.mark.smoke
def test_gradient_postfunc_applied() -> None:
    """gradient_postfunc writes transformed gradients separately."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    model_log = tl.log_forward_pass(
        model,
        x,
        gradients_to_save="all",
        gradient_postfunc=lambda grad: torch.zeros_like(grad),
    )
    model_log.log_backward(_output_loss(model_log))
    assert all(
        torch.equal(
            model_log[label].transformed_gradient,
            torch.zeros_like(model_log[label].gradient),
        )
        for label in model_log.layers_with_saved_gradients
    )


@pytest.mark.smoke
def test_module_log_gradient_aggregation() -> None:
    """ModuleLog exposes aggregated gradients for contained layers."""
    _model, _x, model_log = _logged_model()
    model_log.log_backward(_output_loss(model_log))
    assert model_log.modules["fc2"].gradient is not None


@pytest.mark.smoke
def test_input_layer_gradient_access() -> None:
    """Input layers expose saved gradients after backward."""
    _model, _x, model_log = _logged_model()
    model_log.log_backward(_output_loss(model_log))
    assert model_log[model_log.input_layers[0]].gradient is not None


@pytest.mark.smoke
def test_param_layer_gradient_access() -> None:
    """ParamLog gradient metadata still works through the existing hook path."""
    model, _x, model_log = _logged_model()
    model_log.log_backward(_output_loss(model_log))
    assert any(param_log.has_grad for param_log in model_log.params)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.smoke
def test_custom_autograd_function_captured_with_is_custom_flag() -> None:
    """Custom autograd.Function grad_fns are captured and flagged."""
    model = _CustomModel()
    x = torch.randn(2, 3, requires_grad=True)
    model_log = tl.log_forward_pass(model, x, gradients_to_save="all")
    model_log.log_backward(_output_loss(model_log))
    assert any(grad_fn.is_custom for grad_fn in model_log.grad_fn_logs.values())


@pytest.mark.smoke
def test_implicit_hook_firing_preserved() -> None:
    """Calling backward outside log_backward still populates LayerLog gradients."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    model_log = tl.log_forward_pass(model, x, save_gradients=True)
    _output_loss(model_log).backward()
    assert model_log.layers_with_saved_gradients


@pytest.mark.smoke
def test_validate_backward_pass_correct() -> None:
    """validate_backward_pass returns True for correct capture."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert tl.validate_backward_pass(model, x)


@pytest.mark.smoke
def test_validate_backward_pass_perturbed() -> None:
    """validate_backward_pass returns False after perturbation sanity check."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert not tl.validate_backward_pass(model, x, perturb_saved_gradients=True)


@pytest.mark.smoke
def test_peak_memory_tracking_populated() -> None:
    """ModelLog stores flat backward peak-memory tracking metadata."""
    _model, _x, model_log = _logged_model()
    model_log.log_backward(_output_loss(model_log))
    assert model_log.has_backward_log
    assert isinstance(model_log.backward_peak_memory_bytes, int)
    assert model_log.backward_memory_backend in {"cpu", "cuda", "mps"}


@pytest.mark.smoke
def test_higher_order_grads_basic_support() -> None:
    """create_graph=True backward calls run through capture."""
    _model, _x, model_log = _logged_model()
    model_log.log_backward(_output_loss(model_log), create_graph=True)
    assert model_log.backward_num_passes == 1
