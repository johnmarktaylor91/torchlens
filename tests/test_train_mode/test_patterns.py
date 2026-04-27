"""Training pattern tests for train-mode capture."""

from __future__ import annotations

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
