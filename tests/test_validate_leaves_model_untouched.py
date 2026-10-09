"""``tl.validate`` hands the live model back without writing into its tensors.

Regression pin: every validate scope used to restore the model's state by
``load_state_dict`` even when the validation runs had not changed it. That
copies the snapshot into each parameter and buffer in place, so every
``_version`` counter moved while the bytes came back equal. A user who held an
autograd graph across the call (``y = model(x); tl.validate(model, x, ...);
y.sum().backward()``) then got torch's "modified by an inplace operation"
error at backward. Backward validation also dropped gradients the user had
already accumulated.

The oracles are an identically initialized model TorchLens never touched,
never the same call repeated.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

_X = torch.tensor([[1.0, -2.0, 0.5, 0.25], [0.5, 1.5, -1.0, 2.0], [-0.5, 0.0, 1.0, -1.5]])

# Every scope runs on the model with an eval BatchNorm (buffers) except
# receptive_field, which runs on the LayerNorm twin: its reference-mode capture
# refuses the BatchNorm buffers for an unrelated reason (see the xfail below).
_CASES = (
    ("forward", "batch"),
    ("saved", "batch"),
    ("intervention", "batch"),
    ("backward", "batch"),
    ("receptive_field", "layer"),
)


class _Model(nn.Module):
    """Linear stack with a normalization layer and a frozen layer.

    ``norm="batch"`` uses an eval-mode BatchNorm (registered buffers);
    ``norm="layer"`` uses a buffer-free LayerNorm.
    """

    def __init__(self, norm: str) -> None:
        """Build the layers and freeze ``frozen``."""

        super().__init__()
        self.inp = nn.Linear(4, 5)
        self.bn: nn.Module = nn.BatchNorm1d(5) if norm == "batch" else nn.LayerNorm(5)
        self.frozen = nn.Linear(5, 5)
        self.head = nn.Linear(5, 2)
        for param in self.frozen.parameters():
            param.requires_grad_(False)
        if isinstance(self.bn, nn.BatchNorm1d):
            with torch.no_grad():
                # Non-trivial running statistics so eval BatchNorm is not identity.
                self.bn.running_mean.copy_(torch.linspace(-0.5, 0.5, 5))
                self.bn.running_var.copy_(torch.linspace(0.5, 1.5, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack."""

        hidden = torch.relu(self.bn(self.inp(x)))
        return self.head(torch.tanh(self.frozen(hidden)) + hidden)


def _fresh(norm: str) -> _Model:
    """Return a deterministically initialized eval-mode model."""

    torch.manual_seed(11)
    return _Model(norm).eval()


def _tensor_state(model: nn.Module) -> dict[str, tuple[Any, ...]]:
    """Snapshot identity, storage, version, value and flags of every tensor.

    Parameters
    ----------
    model:
        Model under test.

    Returns
    -------
    dict[str, tuple[Any, ...]]
        ``name -> (id, data_ptr, _version, requires_grad, is_leaf, value)`` for
        every parameter and buffer.
    """

    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(model.named_buffers(remove_duplicate=False))
    return {
        name: (
            id(tensor),
            tensor.data_ptr(),
            tensor._version,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.detach().clone(),
        )
        for name, tensor in tensors.items()
    }


def _assert_state_unchanged(before: dict[str, tuple[Any, ...]], model: nn.Module) -> None:
    """Assert every parameter and buffer is exactly as ``before`` recorded."""

    after = _tensor_state(model)
    assert set(after) == set(before), "the parameter/buffer set changed"
    labels = ("identity", "data_ptr", "_version", "requires_grad", "is_leaf")
    for name, entry in before.items():
        for label, old, new in zip(labels, entry[:5], after[name][:5]):
            assert new == old, f"{name}: {label} changed from {old!r} to {new!r}"
        assert torch.equal(after[name][5], entry[5]), f"{name}: value changed"


def _validate(model: nn.Module, scope: str) -> None:
    """Run ``tl.validate`` at ``scope`` and assert it passed."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tl.validate(model, _X, scope=scope)
    if scope == "intervention":
        assert result.invariance, "baseline forward validation failed"
    elif scope == "receptive_field":
        assert isinstance(result, list)
    else:
        assert result is True, f"validate(scope={scope!r}) failed"


def _grads(model: nn.Module) -> dict[str, torch.Tensor | None]:
    """Return each parameter's grad (``None`` when unset)."""

    return {name: param.grad for name, param in model.named_parameters()}


def _assert_grads_equal(
    model: nn.Module, oracle: dict[str, torch.Tensor | None], *, scale: float = 1.0
) -> None:
    """Assert ``model``'s grads equal the oracle grads times ``scale``."""

    for name, grad in _grads(model).items():
        expected = oracle[name]
        if expected is None:
            assert grad is None, f"{name}: frozen parameter received a grad"
            continue
        assert grad is not None, f"{name}: grad missing"
        assert torch.equal(grad, expected * scale), f"{name}: grad differs from the oracle"


def _oracle_grads(norm: str) -> dict[str, torch.Tensor | None]:
    """Gradients of one eager forward/backward on a never-validated model."""

    oracle = _fresh(norm)
    oracle(_X).sum().backward()
    return _grads(oracle)


@pytest.mark.parametrize(("scope", "norm"), _CASES)
def test_validate_keeps_a_held_graph_backpropagatable(scope: str, norm: str) -> None:
    """Forward, validate, then backward through the held graph matches eager.

    Every parameter and buffer keeps its identity, storage, version counter,
    value and ``requires_grad`` flag across the call.
    """

    model = _fresh(norm)
    output = model(_X)
    before = _tensor_state(model)
    _validate(model, scope)
    _assert_state_unchanged(before, model)

    output.sum().backward()
    _assert_grads_equal(model, _oracle_grads(norm))


@pytest.mark.parametrize(("scope", "norm"), _CASES)
def test_validate_keeps_accumulated_grads(scope: str, norm: str) -> None:
    """Grads the user accumulated before validate survive it unchanged."""

    model = _fresh(norm)
    model(_X).sum().backward()
    held = _grads(model)
    held_values = {
        name: None if grad is None else grad.detach().clone() for name, grad in held.items()
    }
    output = model(_X)
    _validate(model, scope)
    for name, grad in _grads(model).items():
        assert grad is held[name], f"{name}: validate replaced the user's grad tensor"
        if grad is not None:
            assert torch.equal(grad, held_values[name]), f"{name}: grad value changed"

    output.sum().backward()
    _assert_grads_equal(model, _oracle_grads(norm), scale=2.0)


@pytest.mark.xfail(
    raises=tl.errors.MutatedReferenceError,
    strict=True,
    reason=(
        "receptive_field's reference-mode capture of an eval BatchNorm model fails its "
        "own metadata invariants: a buffer op's saved reference out reads a different "
        "version counter than the one stamped at capture (saved 2, current 0)"
    ),
)
def test_receptive_field_scope_on_a_batchnorm_model() -> None:
    """Receptive-field validation of an eval BatchNorm model (known capture bug)."""

    model = _fresh("batch")
    before = _tensor_state(model)
    _validate(model, "receptive_field")
    _assert_state_unchanged(before, model)
