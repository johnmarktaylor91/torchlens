"""Regression tests for two-pass logging with in-place operations."""

import pytest
import torch
import torch.nn as nn

import torchlens as tl


def _assert_layer_present(model_log: tl.ModelLog, layer_label: str) -> None:
    """Assert that a model log contains a layer label.

    Parameters
    ----------
    model_log
        The TorchLens model log to inspect.
    layer_label
        The expected final layer label.
    """
    assert layer_label in [layer.layer_label for layer in model_log.layer_list]


@pytest.mark.smoke
def test_two_pass_succeeds_with_inplace_relu_module() -> None:
    """Two-pass logging should handle modules with in-place ReLU.

    Returns
    -------
    None
        This test only asserts that logging succeeds and expected labels exist.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10),
    )
    x = torch.randn(2, 3, 8, 8)

    model_log = tl.log_forward_pass(model, x, layers_to_save=["conv2d_1_1"])

    _assert_layer_present(model_log, "conv2d_1_1")
    model_log.cleanup()


def test_two_pass_succeeds_with_non_inplace_relu_module() -> None:
    """Two-pass logging should still handle non-in-place ReLU modules.

    Returns
    -------
    None
        This test only asserts that logging succeeds and expected labels exist.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10),
    )
    x = torch.randn(2, 3, 8, 8)

    model_log = tl.log_forward_pass(model, x, layers_to_save=["conv2d_1_1"])

    _assert_layer_present(model_log, "conv2d_1_1")
    model_log.cleanup()


def test_two_pass_preserves_identity_pass_through_detection() -> None:
    """Two-pass logging should still align true nn.Identity pass-through modules.

    Returns
    -------
    None
        This test only asserts that identity pass-through logging remains aligned.
    """
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Identity(),
        nn.Linear(20, 5),
    )
    x = torch.randn(4, 10)

    exhaustive_log = tl.log_forward_pass(model, x)
    identity_labels = [
        layer.layer_label for layer in exhaustive_log.layer_list if layer.layer_type == "identity"
    ]
    assert identity_labels
    identity_label = identity_labels[0]
    exhaustive_log.cleanup()

    model_log = tl.log_forward_pass(model, x, layers_to_save=[identity_label])

    _assert_layer_present(model_log, identity_label)
    assert model_log[identity_label].has_saved_activations
    model_log.cleanup()


class _InplaceFunctionBlock(nn.Module):
    """Small module that applies an in-place function-level ReLU."""

    def __init__(self) -> None:
        """Initialize the in-place function block.

        Returns
        -------
        None
            Initializer has no return value.
        """
        super().__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer followed by an in-place ReLU function.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            The transformed tensor.
        """
        x = self.linear(x)
        return torch.relu_(x)


def test_two_pass_succeeds_with_function_level_inplace_relu() -> None:
    """Two-pass logging should handle function-level in-place operations.

    Returns
    -------
    None
        This test only asserts that logging succeeds and expected labels exist.
    """
    model = nn.Sequential(
        _InplaceFunctionBlock(),
        nn.Linear(20, 5),
    )
    x = torch.randn(4, 10)

    model_log = tl.log_forward_pass(model, x, layers_to_save=["linear_1_1"])

    _assert_layer_present(model_log, "linear_1_1")
    model_log.cleanup()


def test_two_pass_succeeds_with_multiple_inplace_relu_modules() -> None:
    """Two-pass logging should handle stacks with multiple in-place ReLU modules.

    Returns
    -------
    None
        This test only asserts that logging succeeds for several selectors.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10),
    )
    x = torch.randn(2, 3, 8, 8)

    for selector in (["conv2d_1_1"], ["conv2d_2_3"], [-1]):
        model_log = tl.log_forward_pass(model, x, layers_to_save=selector)
        assert model_log is not None
        model_log.cleanup()


@pytest.mark.slow
def test_two_pass_succeeds_with_resnet50_inplace_relu_modules() -> None:
    """Two-pass logging should handle ResNet50's default in-place ReLU modules.

    Returns
    -------
    None
        This test only asserts that logging succeeds for previously failing selectors.
    """
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet50(weights=None).eval()
    x = torch.randn(2, 3, 224, 224)

    for selector in (["conv2d_1_1"], ["linear_1_175"], [-1]):
        model_log = tl.log_forward_pass(model, x, layers_to_save=selector)
        assert model_log is not None
        model_log.cleanup()
