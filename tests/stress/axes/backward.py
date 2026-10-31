"""Backward axis: autograd inside a captured forward, and a backward-ready capture option.

The runner captures a forward and never calls backward afterwards, so the rows
here put autograd where a capture can see it: a loss and ``backward()`` run
inside the forward, a retained graph backed through twice, double backward, a
gradient taken on the input, a ``no_grad`` segment and a mid-forward
``detach``. Every row that runs autograd resets the parameter grads it reads
first, so repeated forwards (eager, capture, validation) return the same
values. The eager forward runs under ``no_grad``; the rows re-enable grad
locally where they need it.

Backward after the capture (``loss.backward()`` on a captured output,
``Trace.log_backward``, a second backward with ``retain_graph``) needs a
post-capture step the runner does not have; it is listed as remaining work.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ..generator import Inputs, ModelSpec, OptionsSpec

WIDTH = 8

OPTIONS: tuple[OptionsSpec, ...] = (
    OptionsSpec(
        "backward_ready",
        "capture with backward_ready=True (saved outputs kept graph-connected for backward)",
        capture_kwargs={"backward_ready": True},
    ),
)


class _Two(nn.Module):
    """Two linear layers the backward rows share."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(WIDTH, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)

    def _fresh_weight_grad(self) -> None:
        for param in self.parameters():
            param.grad = None


class LossBackwardInForward(_Two):
    """Compute a loss and call ``backward()`` mid-forward, then read ``fc1.weight.grad``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._fresh_weight_grad()
        with torch.enable_grad():
            loss = torch.relu(self.fc1(x.detach())).pow(2).mean()
            loss.backward()
        grad = self.fc1.weight.grad
        assert grad is not None
        return self.fc2(x) + grad.sum(dim=0)


class RetainGraphTwice(_Two):
    """Back through one retained graph twice; the grads accumulate to twice the first."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._fresh_weight_grad()
        with torch.enable_grad():
            loss = torch.tanh(self.fc1(x.detach())).sum()
            loss.backward(retain_graph=True)
            loss.backward()
        grad = self.fc1.weight.grad
        assert grad is not None
        return self.fc2(x) * grad.mean()


class DoubleBackward(_Two):
    """Take a gradient of a gradient (``create_graph=True``) on a detached copy of the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            y = torch.tanh(self.fc1(z)).pow(3).sum()
            (g,) = torch.autograd.grad(y, z, create_graph=True)
            (gg,) = torch.autograd.grad(g.pow(2).sum(), z)
        return self.fc2(x) + gg.detach()


class InputGradient(_Two):
    """Return the gradient of a scalar on the input, beside the forward output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(torch.relu(self.fc1(z)).sum(), z)
        return self.fc2(x) + g


class NoGradSegment(_Two):
    """Run the middle layer under ``torch.no_grad()`` inside the forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        with torch.no_grad():
            h = self.fc2(h) * 2.0
        return torch.relu(h) + x


class DetachMiddle(_Two):
    """Detach the activation between the two layers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x)).detach()
        return self.fc2(h) + h


class InputRequiresGrad(_Two):
    """A plain two-layer forward whose example input is a leaf that requires grad."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _input(_model: nn.Module) -> Inputs:
    return ((torch.randn(3, WIDTH),), {})


def _grad_input(_model: nn.Module) -> Inputs:
    return ((torch.randn(3, WIDTH).requires_grad_(True),), {})


def _model(
    row_id: str,
    factory: type[nn.Module],
    reason: str,
    example: Callable[[nn.Module], Inputs] = _input,
) -> ModelSpec:
    return ModelSpec(
        id=row_id,
        build=factory,
        example_input=example,
        reason=reason,
        tags=frozenset({"backward"}),
    )


MODELS: tuple[ModelSpec, ...] = (
    _model(
        "bw_loss_backward",
        LossBackwardInForward,
        "a loss and backward() run inside the captured forward, its grad read after",
    ),
    _model(
        "bw_retain_twice",
        RetainGraphTwice,
        "backward twice through one retained graph inside the forward",
    ),
    _model("bw_double", DoubleBackward, "double backward (grad of a grad) inside the forward"),
    _model("bw_input_grad", InputGradient, "a gradient on the input joins the forward output"),
    _model("bw_nograd_segment", NoGradSegment, "a no_grad segment in the middle of the forward"),
    _model("bw_detach_middle", DetachMiddle, "an activation detached between two layers"),
    _model(
        "bw_input_requires_grad",
        InputRequiresGrad,
        "the input is a leaf that requires grad, so every op is graph-connected",
        _grad_input,
    ),
)
