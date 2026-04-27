"""Edge-case tests for train-mode capture."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import torch
from torch import nn

import torchlens as tl


class ViewModel(nn.Module):
    """Model that returns a reshaped view before a trainable head."""

    def __init__(self) -> None:
        """Initialize the layer."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a view/reshape operation in the forward pass."""

        hidden = self.linear(x)
        return hidden.reshape(x.shape[0], 2, 2).view(x.shape[0], 4)


class InplaceReluModel(nn.Module):
    """Model that applies an in-place relu to a non-leaf tensor."""

    def __init__(self) -> None:
        """Initialize the layers."""

        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply an in-place relu after a linear layer."""

        hidden = self.linear(x)
        torch.relu_(hidden)
        return self.head(hidden)


class MixedGradModel(nn.Module):
    """Model with frozen and trainable parameter subsets."""

    def __init__(self) -> None:
        """Initialize frozen and trainable layers."""

        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.trainable = nn.Linear(4, 2)
        for param in self.frozen.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen then trainable layers."""

        return self.trainable(torch.relu(self.frozen(x)))


def _init_process_group(tmp_path: Path) -> bool:
    """Initialize a local single-rank process group if available."""

    if not torch.distributed.is_available():
        return False
    if torch.distributed.is_initialized():
        return True
    init_file = tmp_path / "ddp_init"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    return True


def _compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Compile a model with the lightweight eager backend."""

    compile_fn: Callable[..., torch.nn.Module] | None = getattr(torch, "compile", None)
    if compile_fn is None:
        pytest.skip("torch.compile is unavailable in this PyTorch version")
    return compile_fn(model, backend="eager")


def test_autocast_wrapping_slow_keeps_grad() -> None:
    """CPU autocast around slow train-mode capture preserves backpropagation."""

    model = nn.Linear(4, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model_log = tl.log_forward_pass(
            model,
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            random_seed=0,
        )
    saved = model_log[model_log.output_layers[0]].activation

    model.zero_grad(set_to_none=True)
    saved.float().sum().backward()

    assert saved.grad_fn is not None
    assert any(param.grad is not None for param in model.parameters())
    model_log.cleanup()


def test_ddp_wrapped_slow_keeps_local_module_grad(tmp_path: Path) -> None:
    """DDP unwrap populates LOCAL .module grads only; this is not DDP sync semantics."""

    if not _init_process_group(tmp_path):
        pytest.skip("torch.distributed is unavailable")
    ddp_model = torch.nn.parallel.DistributedDataParallel(nn.Linear(4, 2))

    model_log = tl.log_forward_pass(
        ddp_model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = model_log[model_log.output_layers[0]].activation

    ddp_model.module.zero_grad(set_to_none=True)
    saved.sum().backward()

    assert all(param.grad is not None for param in ddp_model.module.parameters())
    model_log.cleanup()


def test_view_reshape_ops_keep_grad() -> None:
    """Saving a viewed tensor still allows backward through the view chain."""

    model = ViewModel()
    model_log = tl.log_forward_pass(
        model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = model_log[model_log.output_layers[0]].activation

    model.zero_grad(set_to_none=True)
    saved.square().mean().backward()

    assert model.linear.weight.grad is not None
    model_log.cleanup()


def test_inplace_relu_keeps_grad_slow() -> None:
    """Saving after an in-place relu on a non-leaf tensor still backpropagates."""

    model = InplaceReluModel()
    model_log = tl.log_forward_pass(
        model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = model_log[model_log.output_layers[0]].activation

    model.zero_grad(set_to_none=True)
    saved.sum().backward()

    assert any(param.grad is not None for param in model.parameters())
    model_log.cleanup()


def test_no_grad_wrapping_forward_severs_grad_slow() -> None:
    """User-disabled grad around the forward keeps PyTorch's backward failure semantics."""

    model = nn.Linear(4, 2)
    no_grad = getattr(torch, "no_grad")
    with no_grad():
        model_log = tl.log_forward_pass(
            model,
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            random_seed=0,
        )
    saved = model_log[model_log.output_layers[0]].activation

    assert saved.grad_fn is None
    with pytest.raises(RuntimeError, match="does not require grad"):
        saved.sum().backward()
    model_log.cleanup()


def test_mixed_grad_model_slow() -> None:
    """Only the trainable parameter subset receives gradients."""

    model = MixedGradModel()
    model_log = tl.log_forward_pass(
        model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = model_log[model_log.output_layers[0]].activation

    model.zero_grad(set_to_none=True)
    saved.sum().backward()

    assert all(param.grad is None for param in model.frozen.parameters())
    assert all(param.grad is not None for param in model.trainable.parameters())
    model_log.cleanup()


def test_compile_wrapped_model_rejected_cross_link() -> None:
    """Edge-case cross-link: compiled models are rejected for train-mode capture."""

    with pytest.raises(RuntimeError, match="torch.compile"):
        tl.log_forward_pass(
            _compile_model(nn.Linear(4, 2)),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
        )
