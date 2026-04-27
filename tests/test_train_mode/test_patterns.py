"""Training pattern tests for train-mode capture."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

import torchlens as tl
from .conftest import TinyResnetWithProbe


class SharedFrozenModule(nn.Module):
    """Model that calls one frozen module twice before a trainable head."""

    def __init__(self) -> None:
        """Initialize shared and trainable layers."""

        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)
        for param in self.shared.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reuse the shared module twice in one forward pass."""

        first = self.shared(x)
        second = self.shared(x + 1)
        return self.head(first + second)


class BranchMismatchModel(nn.Module):
    """Model whose graph changes based on input sign."""

    def __init__(self) -> None:
        """Initialize the branch layer."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a sign-dependent graph."""

        out = self.linear(x)
        if bool(x.sum() > 0):
            out = torch.relu(out)
        return out


def _assert_params_require_grad(module: nn.Module, expected: bool) -> None:
    """Assert all parameters in a module have the expected requires_grad value."""

    assert all(param.requires_grad is expected for param in module.parameters())


def test_train_mode_preserves_user_requires_grad(
    tiny_resnet_with_probe: TinyResnetWithProbe,
) -> None:
    """train_mode preserves frozen backbone params during and after capture."""

    observed_backbone_states: list[bool] = []

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        """Record backbone requires_grad state during the forward pass."""

        observed_backbone_states.append(
            all(
                param.requires_grad is False
                for param in tiny_resnet_with_probe.backbone.parameters()
            )
        )

    handle = tiny_resnet_with_probe.backbone.register_forward_pre_hook(hook)
    try:
        model_log = tl.log_forward_pass(
            tiny_resnet_with_probe,
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            random_seed=0,
        )
    finally:
        handle.remove()

    assert observed_backbone_states == [True]
    _assert_params_require_grad(tiny_resnet_with_probe.backbone, False)
    _assert_params_require_grad(tiny_resnet_with_probe.probe, True)

    saved = model_log[model_log.output_layers[0]].activation
    tiny_resnet_with_probe.zero_grad(set_to_none=True)
    saved.sum().backward()

    assert all(param.grad is None for param in tiny_resnet_with_probe.backbone.parameters())
    assert all(param.grad is not None for param in tiny_resnet_with_probe.probe.parameters())
    model_log.cleanup()


def test_train_mode_shared_module_requires_grad() -> None:
    """train_mode preserves requires_grad on shared modules called repeatedly."""

    model = SharedFrozenModule()
    observed_shared_states: list[bool] = []

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        """Record shared-module requires_grad state during each call."""

        observed_shared_states.append(
            all(param.requires_grad is False for param in model.shared.parameters())
        )

    handle = model.shared.register_forward_pre_hook(hook)
    try:
        model_log = tl.log_forward_pass(
            model,
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            random_seed=0,
        )
    finally:
        handle.remove()

    assert observed_shared_states == [True, True]
    _assert_params_require_grad(model.shared, False)
    _assert_params_require_grad(model.head, True)

    saved = model_log[model_log.output_layers[0]].activation
    model.zero_grad(set_to_none=True)
    saved.sum().backward()

    assert all(param.grad is None for param in model.shared.parameters())
    assert all(param.grad is not None for param in model.head.parameters())
    model_log.cleanup()


def test_save_new_activations_train_mode_inherits() -> None:
    """save_new_activations inherits train_mode by default."""

    model = SharedFrozenModule()
    model_log = tl.log_forward_pass(
        model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )

    model_log.save_new_activations(
        model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=None,
        random_seed=0,
    )

    saved = model_log[model_log.output_layers[0]].activation
    assert saved.grad_fn is not None
    assert model_log.train_mode is True
    model_log.cleanup()


def test_save_new_activations_train_mode_overrides() -> None:
    """Explicit save_new_activations train_mode overrides detach flags temporarily."""

    model = SharedFrozenModule()
    model_log = tl.log_forward_pass(
        model,
        torch.randn(3, 4, requires_grad=True),
        detach_saved_tensors=True,
        random_seed=0,
    )
    original_layer_flags = [layer.detach_saved_tensor for layer in model_log]

    model_log.save_new_activations(
        model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )

    saved = model_log[model_log.output_layers[0]].activation
    assert saved.grad_fn is not None
    assert model_log.detach_saved_tensors is True
    assert model_log.train_mode is False
    assert [layer.detach_saved_tensor for layer in model_log] == original_layer_flags
    model_log.cleanup()


def test_fastlog_train_mode_sugar_promotes_defaults() -> None:
    """fastlog train_mode promotes omitted defaults to keep-grad capture specs."""

    recording = tl.fastlog.record(
        SharedFrozenModule(),
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
    )

    grad_records = [
        record
        for record in recording
        if record.ram_payload is not None and record.ram_payload.grad_fn is not None
    ]
    assert grad_records


def test_fastlog_train_mode_explicit_default_op_false() -> None:
    """Explicit default_op=False remains disabled under train_mode sugar."""

    recording = tl.fastlog.record(
        SharedFrozenModule(),
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        default_op=False,
    )

    assert all(record.ctx.kind != "op" for record in recording)


def test_fastlog_train_mode_default_op_true_errors() -> None:
    """default_op=True contradicts train_mode keep-grad sugar."""

    with pytest.raises(tl.TrainingModeConfigError, match="default_op=True"):
        tl.fastlog.record(
            SharedFrozenModule(),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            default_op=True,
        )


def test_fastlog_train_mode_explicit_capspec_keepgrad_false_errors() -> None:
    """CaptureSpec keep_grad=False contradicts train_mode keep-grad sugar."""

    with pytest.raises(tl.TrainingModeConfigError, match="keep_grad=False"):
        tl.fastlog.record(
            SharedFrozenModule(),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            default_op=tl.fastlog.CaptureSpec(keep_grad=False),
        )


def test_save_new_activations_train_mode_restored_on_graph_mismatch() -> None:
    """save_new_activations restores override flags when fast-pass graph validation fails."""

    model = BranchMismatchModel()
    model_log = tl.log_forward_pass(
        model,
        torch.ones(2, 4, requires_grad=True),
        detach_saved_tensors=True,
        random_seed=0,
    )
    original_layer_flags = [layer.detach_saved_tensor for layer in model_log]

    def divergent_forward(self: BranchMismatchModel, x: torch.Tensor) -> torch.Tensor:
        """Run a different operation sequence from the exhaustive pass."""

        out = self.linear(x)
        return out + out

    model.forward = types.MethodType(divergent_forward, model)

    try:
        model_log.save_new_activations(
            model,
            torch.ones(2, 4, requires_grad=True),
            train_mode=True,
            random_seed=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected fast-pass graph mismatch")

    assert model_log.detach_saved_tensors is True
    assert model_log.train_mode is False
    assert [layer.detach_saved_tensor for layer in model_log] == original_layer_flags
    model_log.cleanup()
