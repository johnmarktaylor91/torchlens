"""Smoke tests for first-class backward-pass capture."""

import torch
from torch import nn
import pytest

import torchlens as tl
from torchlens.data_classes.grad_fn_log import GradFnLog


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
        """Return doubled upstream grad."""
        return grad * 2


class _CustomModel(nn.Module):
    """Model using a custom autograd Function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return _DoubleFn.apply(x).sum()


def _logged_model(
    *,
    layers_to_save: str | list[str] | None = "all",
    grads_to_save: str | list[str] | None = "all",
) -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    """Create a logged tiny model.

    Returns
    -------
    tuple[nn.Module, torch.Tensor, tl.Trace]
        Model, input tensor, and model log.
    """
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        layers_to_save=layers_to_save,
        grads_to_save=grads_to_save,
    )
    return model, x, trace


def _output_loss(trace: tl.Trace) -> torch.Tensor:
    """Return scalar sum loss from the logged output out."""
    return trace[trace.output_layers[0]].out.sum()


@pytest.mark.smoke
def test_log_backward_captures_per_layer_grads() -> None:
    """log_backward captures saved per-layer grads."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.has_grads
    assert len(trace.ops_with_saved_grads) > 0
    assert all(trace[label].grad is not None for label in trace.ops_with_saved_grads)


@pytest.mark.smoke
def test_recording_backward_context_manager() -> None:
    """recording_backward accumulates multiple backward calls."""
    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    with trace.recording_backward():
        loss.backward(retain_graph=True)
        (loss * 2).backward()
    assert trace.backward_num_calls == 2


@pytest.mark.smoke
def test_backward_graph_walk_includes_intervening_grad_fns() -> None:
    """The backward DAG includes grad_fns without forward LayerLog matches."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert any(grad_fn.is_intervening for grad_fn in trace.grad_fn_logs.values())


def test_has_op_deprecation_property() -> None:
    """Legacy has_op access warns and returns the inverse of is_intervening."""

    grad_fn = GradFnLog(
        grad_fn_id=1,
        name="AddBackward0",
        module_path="torch.autograd",
        is_custom=False,
        label="addbackward0_1_1",
        grad_fn_type="addbackward0",
        grad_fn_type_num=1,
        grad_fn_total_num=1,
        is_intervening=False,
    )
    with pytest.warns(DeprecationWarning):
        assert grad_fn.has_op is True


@pytest.mark.smoke
def test_grad_fn_log_back_pointer() -> None:
    """Forward LayerLogs link to corresponding GradFnLogs by identity."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert any(layer.grad_fn_log is not None for layer in trace.layer_list)
    assert any(
        grad_fn.op is not None and grad_fn.op.grad_fn_log is grad_fn for grad_fn in trace.grad_fns
    )


@pytest.mark.smoke
def test_grad_fn_naming_and_indexing() -> None:
    """GradFnLog labels and accessor indexing mirror layer lookup patterns."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    first_grad_fn = trace.grad_fns[0]
    assert "_back_" in first_grad_fn.label
    assert first_grad_fn.label == first_grad_fn.label.lower()
    assert trace.grad_fns[first_grad_fn.label] is first_grad_fn
    assert trace.grad_fns[first_grad_fn.grad_fn_type] is first_grad_fn
    if first_grad_fn.num_calls:
        assert trace.grad_fns[f"{first_grad_fn.label}:1"] is first_grad_fn.ops[1]
    assert list(trace.grad_fns)


@pytest.mark.smoke
def test_grads_to_save_default_matches_layers_to_save() -> None:
    """save_grads uses layers_to_save when grads_to_save is omitted."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save=["relu"], save_grads=True)
    trace.log_backward(_output_loss(trace))
    assert trace.ops_with_saved_grads
    assert all("relu" in label for label in trace.ops_with_saved_grads)


@pytest.mark.smoke
def test_grads_to_save_independent_override() -> None:
    """grads_to_save can be broader than layers_to_save."""
    _model, _x, trace = _logged_model(layers_to_save="all", grads_to_save=["relu"])
    trace.log_backward(_output_loss(trace))
    assert trace.ops_with_saved_grads
    assert all("relu" in label for label in trace.ops_with_saved_grads)


@pytest.mark.smoke
def test_auto_train_mode_when_backward_opted_in() -> None:
    """Explicit grads_to_save auto-enables train_mode."""
    _model, _x, trace = _logged_model()
    assert trace.train_mode is True


@pytest.mark.smoke
def test_auto_train_mode_conflict_with_explicit_false() -> None:
    """Explicit train_mode=False conflicts with backward capture."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="requires train_mode=True"):
        tl.trace(model, x, grads_to_save="all", train_mode=False)


@pytest.mark.smoke
def test_grad_transform_applied() -> None:
    """grad_transform writes transformed grads separately."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        grads_to_save="all",
        grad_transform=lambda grad: torch.zeros_like(grad),
    )
    trace.log_backward(_output_loss(trace))
    assert all(
        torch.equal(
            trace[label].transformed_grad,
            torch.zeros_like(trace[label].grad),
        )
        for label in trace.ops_with_saved_grads
    )


@pytest.mark.smoke
def test_module_log_grad_aggregation() -> None:
    """ModuleLog exposes aggregated grads for contained layers."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.modules["fc2"].grad is not None


@pytest.mark.smoke
def test_input_layer_grad_access() -> None:
    """Input layers expose saved grads after backward."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace[trace.input_layers[0]].grad is not None


@pytest.mark.smoke
def test_param_layer_grad_access() -> None:
    """ParamLog grad metadata still works through the existing hook path."""
    model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert any(param_log.has_grad for param_log in trace.params)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.smoke
def test_custom_autograd_function_captured_with_is_custom_flag() -> None:
    """Custom autograd.Function grad_fns are captured and flagged."""
    model = _CustomModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, grads_to_save="all")
    trace.log_backward(_output_loss(trace))
    assert any(grad_fn.is_custom for grad_fn in trace.grad_fn_logs.values())


@pytest.mark.smoke
def test_implicit_hook_firing_preserved() -> None:
    """Calling backward outside log_backward still populates LayerLog grads."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, save_grads=True)
    _output_loss(trace).backward()
    assert trace.ops_with_saved_grads


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
    assert not tl.validate_backward_pass(model, x, perturb_saved_grads=True)


@pytest.mark.smoke
def test_peak_memory_tracking_populated() -> None:
    """Trace stores flat backward peak-memory tracking metadata."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.has_backward_pass
    assert isinstance(trace.backward_peak_memory, int)
    assert trace.backward_memory_backend in {"cpu", "cuda", "mps"}


@pytest.mark.smoke
def test_higher_order_grads_basic_support() -> None:
    """create_graph=True backward calls run through capture."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace), create_graph=True)
    assert trace.backward_num_calls == 1
