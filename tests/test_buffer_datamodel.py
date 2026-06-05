"""Regression tests for registered-buffer version capture."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl


TensorFactory = Callable[[], torch.Tensor]


class RecurrentReassign(nn.Module):
    """Top-level recurrent reassignment model."""

    def __init__(self, steps: int = 4) -> None:
        """Initialize the recurrent buffer."""

        super().__init__()
        self.steps = steps
        self.register_buffer("h", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run repeated buffer reassignment."""

        for _ in range(self.steps):
            self.h = torch.tanh(self.h + x)
        return self.h + x


class InplaceOps(nn.Module):
    """Explicit in-place mutator model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run in-place ``mul_`` and ``add_`` writes."""

        self.b.mul_(2)
        self.b.add_(x)
        return self.b + x


class CopyWrite(nn.Module):
    """Explicit ``copy_`` write model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Copy into the registered buffer."""

        self.b.copy_(x)
        return self.b + x


class DataCopyWrite(nn.Module):
    """Explicit ``.data.copy_`` write model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Copy through the buffer's ``.data`` tensor."""

        self.b.data.copy_(x)
        return self.b + x


class SliceWrite(nn.Module):
    """View/slice write model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into a slice of the buffer."""

        self.b[:2].copy_(x)
        return self.b.sum()


class SetItemWrite(nn.Module):
    """``__setitem__`` write model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into the buffer with item assignment."""

        self.b[:2] = x
        return self.b.sum()


class DirectBuffersWrite(nn.Module):
    """Direct ``_buffers`` reassignment model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Replace the buffer through ``_buffers``."""

        self._buffers["b"] = x + 1
        return self.b + x


class TwoLoops(nn.Module):
    """Same buffer written in two loops."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("h", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write the same buffer in two separate loops."""

        for _ in range(2):
            self.h = self.h + x
        for _ in range(2):
            self.h = torch.tanh(self.h)
        return self.h + x


class DualRoleInplace(nn.Module):
    """In-place write whose return is also used directly."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the in-place result and the mutated buffer."""

        y = self.b.add_(x)
        return y * self.b


class StaticReadOnly(nn.Module):
    """Static read-only buffer model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.arange(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read a static buffer."""

        return self.b + x


class DataSetter(nn.Module):
    """Unsupported ``.data = tensor`` model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Replace the buffer storage through the unsupported data setter."""

        self.b.data = x + 1
        return self.b


class AliasWrite(nn.Module):
    """Shared/overlapping registered-buffer model."""

    def __init__(self) -> None:
        """Initialize aliased registered buffers."""

        super().__init__()
        base = torch.zeros(4)
        self.register_buffer("b", base)
        self.register_buffer("c", base[:2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write through the aliased buffer view."""

        self.c.add_(x)
        return self.b.sum()


@pytest.mark.parametrize(
    ("model_factory", "input_factory", "expected_overwrites"),
    [
        (lambda: RecurrentReassign(), lambda: torch.ones(2), {"h": 4}),
        (lambda: InplaceOps(), lambda: torch.ones(2), {"b": 2}),
        (lambda: CopyWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: DataCopyWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: SliceWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: SetItemWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: DirectBuffersWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: TwoLoops(), lambda: torch.ones(2), {"h": 4}),
        (lambda: DualRoleInplace(), lambda: torch.ones(2), {"b": 1}),
        (lambda: StaticReadOnly(), lambda: torch.ones(2), {"b": 0}),
        (lambda: AliasWrite(), lambda: torch.ones(2), {"c": 1}),
        (
            lambda: nn.BatchNorm1d(3).train(),
            lambda: torch.randn(4, 3),
            {
                "num_batches_tracked": 1,
                "running_mean": 1,
                "running_var": 1,
            },
        ),
        (
            lambda: nn.BatchNorm2d(3).train(),
            lambda: torch.randn(2, 3, 4, 4),
            {
                "num_batches_tracked": 1,
                "running_mean": 1,
                "running_var": 1,
            },
        ),
        (
            lambda: nn.InstanceNorm1d(3, track_running_stats=True).train(),
            lambda: torch.randn(2, 3, 4),
            {"running_mean": 1, "running_var": 1},
        ),
    ],
)
def test_buffer_write_models_validate_and_expose_entities(
    model_factory: Callable[[], nn.Module],
    input_factory: TensorFactory,
    expected_overwrites: dict[str, int],
) -> None:
    """Validate stress models and assert buffer entity metadata."""

    model = model_factory()
    x = input_factory()
    assert tl.validation.validate_forward_pass(
        model_factory(), x.clone(), random_seed=123, validate_metadata=True
    )

    trace = tl.trace(model, x, save_arg_values=True)
    for address, overwrite_count in expected_overwrites.items():
        assert address in trace.buffers
        buffer = trace.buffers[address]
        assert buffer.versions
        assert buffer.final_value is not None
        assert buffer.num_overwrites == overwrite_count
    assert not any(op.is_buffer for op in trace.compute_ops)


def test_reassignment_double_count_is_exact() -> None:
    """Assert N top-level reassignments produce exactly N write events."""

    trace = tl.trace(RecurrentReassign(steps=5), torch.ones(2), save_arg_values=True)
    events = [event for event in trace._buffer_write_events if event.address == "h"]
    assert len(events) == 5
    assert trace.buffers["h"].num_overwrites == 5


class RecurrentCell(nn.Module):
    """RNN-style cell: reassigns a state buffer in a loop around a submodule.

    The inner ``nn.Linear`` makes the loop body recurrent, so loop detection
    engages over the reassigned buffer's version nodes. Regression guard for a
    crash where merging the initial buffer node left a dangling label in the
    output node's ``equivalent_ops`` (``'buffer_1_raw' is not a known raw
    label``) that loop detection then dereferenced mid-pass.
    """

    def __init__(self, dim: int = 8, steps: int = 4) -> None:
        """Initialize the recurrent cell and its hidden-state buffer."""

        super().__init__()
        self.steps = steps
        self.cell = nn.Linear(dim, dim)
        self.register_buffer("h", torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reset then recurrently reassign the hidden-state buffer."""

        self.h = torch.zeros_like(self.h)
        for _ in range(self.steps):
            self.h = torch.tanh(self.cell(x) + self.h)
        return self.h


def test_recurrent_cell_reassignment_does_not_break_loop_detection() -> None:
    """An RNN cell reassigning its state buffer must trace and validate."""

    model = RecurrentCell()
    x = torch.randn(1, 8)
    assert tl.validation.validate_forward_pass(
        RecurrentCell(), x.clone(), random_seed=7, validate_metadata=True
    )
    trace = tl.trace(model, x, save_arg_values=True)
    assert "h" in trace.buffers
    assert trace.buffers["h"].num_overwrites == 5  # one reset + four loop steps


class GradRecurrent(nn.Module):
    """Recurrent reassignment with learnable params (non-detached hidden state)."""

    def __init__(self, dim: int = 4, steps: int = 3) -> None:
        """Initialize the recurrent cell and its state buffer."""

        super().__init__()
        self.steps = steps
        self.lin = nn.Linear(dim, dim)
        self.register_buffer("h", torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reassign the state buffer through the autograd graph each step."""

        self.h = torch.zeros_like(self.h)
        for _ in range(self.steps):
            self.h = torch.tanh(self.lin(x) + self.h)
        return self.h.sum()


class GradBatchNorm(nn.Module):
    """Learnable model whose forward updates fused BatchNorm running stats."""

    def __init__(self, dim: int = 4) -> None:
        """Initialize the linear + BatchNorm stack."""

        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fused-buffer-writing forward."""

        return self.bn(self.lin(x)).sum()


def _param_grads(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return a fresh-backward gradient snapshot for every parameter."""

    model.zero_grad(set_to_none=True)
    model(x).backward()
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


@pytest.mark.parametrize("model_factory", [GradRecurrent, GradBatchNorm])
def test_buffer_capture_preserves_gradient_flow(
    model_factory: Callable[[], nn.Module],
) -> None:
    """Capture hooks must be observational: tracing must not break autograd.

    A reassigned state buffer carries ``grad_fn`` exactly like a non-detached
    RNN hidden state; the fused-write snapshot reads (never replaces) the live
    buffer. So gradients through a traced model must match an untraced run.
    """

    import copy

    torch.manual_seed(0)
    reference = model_factory().train()
    traced_model = copy.deepcopy(reference).train()
    x = torch.randn(8, 4)

    expected = _param_grads(reference, x.clone())

    # Tracing performs the fused/reassignment writes; it must leave the live
    # autograd path untouched, so a subsequent backward still matches.
    tl.trace(traced_model, x.clone())
    actual = _param_grads(traced_model, x.clone())

    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.allclose(expected[name], actual[name], atol=1e-5), name


def test_data_setter_reconciliation_raises() -> None:
    """Assert unsupported ``.data = tensor`` changes raise a diagnostic."""

    with pytest.raises(RuntimeError, match="unjournaled registered-buffer change"):
        tl.trace(DataSetter(), torch.ones(2), save_arg_values=True)
