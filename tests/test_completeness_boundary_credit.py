"""An untraced module output credits only the dispatch that built it.

When a module returns a tensor with no live label, module exit synthesizes a
functionless boundary Op and marks the module-forward token capture-accounted.
That credit covers the opaque construction of the exact boundary tensors. A
stale raw torch call elsewhere in the same module body (a callable bound
before TorchLens wrapped torch, as in IQL's ``hidden_activation=torch.relu``
default) is not represented by the boundary, so validation must fail on it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import rescue
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.user_funcs import _validate_forward_pass_torch


def _raw(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the original torch callable behind an installed TorchLens wrapper."""

    wrap_torch()
    return _state._decorated_to_orig.get(id(func), func)


class _OpaqueOutputChild(nn.Module):
    """Traced linear and relu, then a direct-aten output: one boundary, nothing dropped."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a direct-aten tanh of traced work."""

        return torch.ops.aten.tanh.default(torch.relu(self.fc(x)))


class _OpaqueTupleChild(nn.Module):
    """Two direct-aten outputs: each is its own boundary tensor."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two untraced outputs built by single direct-aten ops."""

        return torch.ops.aten.tanh.default(x), torch.ops.aten.sigmoid.default(x)


class _StaleHiddenChild(nn.Module):
    """IQL shape: stale relu hidden activation, stale tanh output activation."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.hidden_activation = _raw(torch.relu)
        self.output_activation = _raw(torch.tanh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc1, stale relu, fc2, stale tanh."""

        return self.output_activation(self.fc2(self.hidden_activation(self.fc1(x))))


class _StaleInnerLeaf(nn.Module):
    """A stale relu intermediate under a direct-aten output, two levels down."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.act = _raw(torch.relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a direct-aten tanh of a stale relu."""

        return torch.ops.aten.tanh.default(self.fc(self.act(x)))


class _Middle(nn.Module):
    """Plain container so the boundary-backed module sits at depth two."""

    def __init__(self) -> None:
        super().__init__()
        self.leaf = _StaleInnerLeaf()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the leaf and a traced op on its output."""

        return self.leaf(x) + 1.0


class _Parent(nn.Module):
    """Consume a child's output in a traced op."""

    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = child

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the child and scale its output (summing tuple outputs)."""

        out = self.child(x)
        if isinstance(out, tuple):
            return (out[0] + out[1]) * 2.0
        return out * 2.0


def _validate(model: nn.Module) -> bool:
    torch.manual_seed(0)
    return _validate_forward_pass_torch(
        model.eval(), [torch.randn(3, 4)], {}, random_seed=0, validate_metadata=True
    )


def _assert_completeness_failure(model: nn.Module) -> None:
    assert not _validate(model)
    failure = tl.validation.last_validation_failure()
    assert failure is not None
    assert "bfs_completeness" in failure.summary()


def test_single_opaque_output_op_still_validates() -> None:
    """The boundary still credits the one direct-aten op that built the module output."""

    assert _validate(_Parent(_OpaqueOutputChild())), tl.validation.last_validation_failure()


def test_tuple_of_opaque_outputs_still_validates() -> None:
    """Every tensor of a tuple result that is a boundary output is credited."""

    assert _validate(_Parent(_OpaqueTupleChild())), tl.validation.last_validation_failure()


# The stale op's output reaches the next traced op with no recorded parent; that
# provenance disclosure is expected alongside the completeness failure.
_NO_PROVENANCE = "ignore:TorchLens found tensor arguments with no graph:UserWarning"


@pytest.mark.filterwarnings(_NO_PROVENANCE)
def test_stale_hidden_activation_under_stale_output_fails_completeness() -> None:
    """IQL shape: the stale relu is not hidden by the stale tanh's boundary."""

    _assert_completeness_failure(_Parent(_StaleHiddenChild()))


@pytest.mark.filterwarnings(_NO_PROVENANCE)
def test_stale_op_in_nested_boundary_module_fails_completeness() -> None:
    """A stale op inside a depth-two module with a synthesized output boundary fails."""

    _assert_completeness_failure(_Parent(_Middle()))


@pytest.mark.filterwarnings(_NO_PROVENANCE)
def test_census_names_only_the_stale_relu_in_iql_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The primary capture's census names the dropped relu, never the boundary's tanh."""

    # Keep the primary capture: the rescue re-run would replace its diagnostics.
    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    wrap_torch(completeness_witness=True)
    torch.manual_seed(0)
    trace = tl.trace(_Parent(_StaleHiddenChild()).eval(), torch.randn(3, 4))
    operators = [row["operator"] for row in trace.completeness_diagnostics]
    assert "aten.relu.default" in operators
    assert "aten.tanh.default" not in operators
    assert trace.capture_verified is False
