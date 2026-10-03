"""Edge-case tests for train-mode capture."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_state_helpers import reset_compiled_model_unwrap_warning_state
from torchlens.distributed import has_vetted_snapshot


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
        trace = tl.trace(
            model,
            torch.randn(3, 4, requires_grad=True),
            capture=tl.options.CaptureOptions(backward_ready=True, random_seed=0),
        )
    saved = trace[trace.output_layers[0]].out

    model.zero_grad(set_to_none=True)
    saved.float().sum().backward()

    assert saved.grad_fn is not None
    assert any(param.grad is not None for param in model.parameters())
    trace.cleanup()


def test_ddp_wrapped_slow_keeps_local_module_grad(tmp_path: Path) -> None:
    """DDP unwrap populates LOCAL .module grads only; this is not DDP sync semantics.

    ``_init_process_group`` used to leave the group it created initialized
    for the rest of the process: every later ``tl.trace()`` call then saw
    ``torch.distributed.is_initialized() is True`` and lazily tried to arm
    collective capture (``maybe_auto_arm``), which warns on a torch build
    whose dispatcher namespaces are not a vetted census row -- a warning the
    suite's ``error::UserWarning:torchlens`` filter promotes to a hard
    failure. Under pytest-randomly this leak could land anywhere in the run
    order and broke hundreds of unrelated tests downstream (2026-10 fast-tier
    incident). Destroy only the group THIS test created, same convention as
    the other distributed fixtures in this suite.

    Capturing under a genuinely initialized process group also triggers that
    same disclosure on an unvetted torch build (F1, Lead ruling
    2026-10-01) -- the correct, honest product behavior (see
    ``test_distributed_boundary_gloo.py::TestUnvettedTorchRefusesArming`),
    expected here rather than treated as a surprise failure.
    """

    already_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not _init_process_group(tmp_path):
        pytest.skip("torch.distributed is unavailable")
    try:
        ddp_model = torch.nn.parallel.DistributedDataParallel(nn.Linear(4, 2))
        x = torch.randn(3, 4, requires_grad=True)
        capture = tl.options.CaptureOptions(backward_ready=True, random_seed=0)
        if has_vetted_snapshot():
            trace = tl.trace(ddp_model, x, capture=capture)
        else:
            with pytest.warns(UserWarning, match="uncaptured_collective_op"):
                trace = tl.trace(ddp_model, x, capture=capture)
        saved = trace[trace.output_layers[0]].out

        ddp_model.module.zero_grad(set_to_none=True)
        saved.sum().backward()

        assert all(param.grad is not None for param in ddp_model.module.parameters())
        trace.cleanup()
    finally:
        if not already_initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_ddp_wrapped_slow_does_not_leak_process_group(tmp_path: Path) -> None:
    """Regression for the 2026-10 fast-tier incident: this test must leave
    ``torch.distributed`` exactly as it found it, or every later capture in
    the process lazily (and, on an unvetted torch build, loudly) tries to
    arm collective capture against a group nobody else owns."""

    pre_existing = torch.distributed.is_available() and torch.distributed.is_initialized()
    test_ddp_wrapped_slow_keeps_local_module_grad(tmp_path)
    post = torch.distributed.is_available() and torch.distributed.is_initialized()
    assert post == pre_existing


def test_view_reshape_ops_keep_grad() -> None:
    """Saving a viewed tensor still allows backward through the view chain."""

    model = ViewModel()
    trace = tl.trace(
        model,
        torch.randn(3, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True, random_seed=0),
    )
    saved = trace[trace.output_layers[0]].out

    model.zero_grad(set_to_none=True)
    saved.square().mean().backward()

    assert model.linear.weight.grad is not None
    trace.cleanup()


def test_inplace_relu_keeps_grad_slow() -> None:
    """Saving after an in-place relu on a non-leaf tensor still backpropagates."""

    model = InplaceReluModel()
    trace = tl.trace(
        model,
        torch.randn(3, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True, random_seed=0),
    )
    saved = trace[trace.output_layers[0]].out

    model.zero_grad(set_to_none=True)
    saved.sum().backward()

    assert any(param.grad is not None for param in model.parameters())
    trace.cleanup()


def test_no_grad_wrapping_forward_severs_grad_slow() -> None:
    """User-disabled grad around the forward keeps PyTorch's backward failure semantics."""

    model = nn.Linear(4, 2)
    no_grad = getattr(torch, "no_grad")
    with no_grad():
        trace = tl.trace(
            model,
            torch.randn(3, 4, requires_grad=True),
            capture=tl.options.CaptureOptions(backward_ready=True, random_seed=0),
        )
    saved = trace[trace.output_layers[0]].out

    assert saved.grad_fn is None
    with pytest.raises(RuntimeError, match="does not require grad"):
        saved.sum().backward()
    trace.cleanup()


def test_mixed_grad_model_slow() -> None:
    """Only the trainable parameter subset receives grads."""

    model = MixedGradModel()
    trace = tl.trace(
        model,
        torch.randn(3, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True, random_seed=0),
    )
    saved = trace[trace.output_layers[0]].out

    model.zero_grad(set_to_none=True)
    saved.sum().backward()

    assert all(param.grad is None for param in model.frozen.parameters())
    assert all(param.grad is not None for param in model.trainable.parameters())
    trace.cleanup()


def test_compile_wrapped_model_unwrapped_cross_link() -> None:
    """Edge-case cross-link: compiled models trace through their eager source."""

    reset_compiled_model_unwrap_warning_state()
    with pytest.warns(UserWarning, match="compiled model detected"):
        trace = tl.trace(
            _compile_model(nn.Linear(4, 2)),
            torch.randn(3, 4, requires_grad=True),
            capture=tl.options.CaptureOptions(backward_ready=True),
        )
    assert trace.layer_list
    trace.cleanup()
