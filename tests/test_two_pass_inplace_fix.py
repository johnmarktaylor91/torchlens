"""Regression tests for two-pass logging with in-place operations."""

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends.torch.aliasing import detect_torch_alias_contract
from torchlens.ir.intervention import FunctionEventInput
from torchlens.options import CaptureOptions


def _assert_layer_present(trace: tl.Trace, layer_label: str) -> None:
    """Assert that a model log contains a layer label.

    Parameters
    ----------
    trace
        The TorchLens model log to inspect.
    layer_label
        The expected final layer label.
    """
    assert layer_label in [layer.layer_label for layer in trace.layer_list]


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

    trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save=["conv2d_1_1"]))

    _assert_layer_present(trace, "conv2d_1_1")
    trace.cleanup()


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

    trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save=["conv2d_1_1"]))

    _assert_layer_present(trace, "conv2d_1_1")
    trace.cleanup()


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

    exhaustive_log = tl.trace(model, x)
    identity_labels = [
        layer.layer_label for layer in exhaustive_log.layer_list if layer.layer_type == "identity"
    ]
    assert identity_labels
    identity_label = identity_labels[0]
    exhaustive_log.cleanup()

    trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save=[identity_label]))

    _assert_layer_present(trace, identity_label)
    assert trace[identity_label].has_saved_activation
    trace.cleanup()


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

    trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save=["linear_1_1"]))

    _assert_layer_present(trace, "linear_1_1")
    trace.cleanup()


class _ForeachAddBlock(nn.Module):
    """Module that mutates list-held tensors with a foreach in-place op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a foreach in-place add over derived tensors."""

        left = x + 1
        right = x + 3
        torch._foreach_add_([left, right], 1.0)
        return left.sum() + right.sum()


class _OutBufferWriteBlock(nn.Module):
    """Module that writes an op result into a registered buffer via ``out=``."""

    def __init__(self) -> None:
        """Initialize the destination buffer."""

        super().__init__()
        self.register_buffer("acc", torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write ``x + x`` into ``self.acc`` and consume the result."""

        torch.add(x, x, out=self.acc)
        return self.acc.sum()


def test_identity_dedup_snapshots_foreach_inplace_outputs_after_mutation() -> None:
    """Saved foreach in-place outputs should reflect post-mutation values."""

    trace = tl.trace(_ForeachAddBlock(), torch.tensor([1.0, 2.0, 3.0]))
    foreach_outputs = [trace[label].out for label in trace.layer_labels if "foreachadd" in label]

    assert len(foreach_outputs) == 2
    assert torch.equal(foreach_outputs[0], torch.tensor([3.0, 4.0, 5.0]))
    assert torch.equal(foreach_outputs[1], torch.tensor([5.0, 6.0, 7.0]))


def test_identity_dedup_snapshots_out_kwarg_buffer_write_after_mutation() -> None:
    """Saved ``out=`` op payload should not reuse the pre-write buffer read."""

    trace = tl.trace(_OutBufferWriteBlock(), torch.tensor([1.0, 2.0, 3.0]))

    assert torch.equal(trace["add_1_1"].out, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.equal(trace["buffer_2"].out, torch.tensor([2.0, 4.0, 6.0]))


def test_alias_contract_detects_list_tensor_mutation_positions() -> None:
    """Alias detection should report mutations inside list tensor arguments."""

    before_left = torch.tensor([1.0])
    before_right = torch.tensor([2.0])
    left = before_left.clone()
    right = before_right.clone()
    torch._foreach_add_([left, right], 1.0)
    event_input = FunctionEventInput(
        func=torch._foreach_add_,
        func_name="_foreach_add_",
        func_qualname=None,
        args=([left, right], 1.0),
        kwargs={},
        raw_output=[left, right],
        arg_copies=([before_left, before_right], 1.0),
        kwarg_copies={},
        module_stack=(),
        is_bottom_level_func=True,
        func_call_id=1,
        expected_output_count=2,
    )

    semantics = detect_torch_alias_contract(event_input)

    assert semantics.mutated_input_positions == ((0, 0), (0, 1))
    assert semantics.aliased_output_inputs == ((0, 0), (0, 1))
    assert semantics.unknown_aliasing is False


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
        trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save=selector))
        assert trace is not None
        trace.cleanup()


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
        trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save=selector))
        assert trace is not None
        trace.cleanup()
