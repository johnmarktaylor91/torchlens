"""The backward validator names, or survives, three model shapes it used to crash on.

Found by the behaviour net's backward cells on the classics corpus, each a single
``tl.validate(scope="backward")`` escaping with an untyped error:

* a ``dict`` subclass output whose ``__getattr__`` raises ``KeyError`` (the
  ``Pack`` idiom) defeated the ``hasattr(obj, "to_tuple")`` probe, which swallows
  only ``AttributeError``;
* a module output that is one leaf of a multi-output node (``split``,
  ``unbind``, an LSTM) whose sibling carried the gradient receives ``None`` in
  its tensor hook, which the stock pass tried to ``detach()``;
* a model holding non-leaf tensors with autograd history as module attributes
  (a key table computed from an embedding at construction) makes every backward
  after the first fail inside torch; that is the model's own backward contract,
  so it is refused typed as ``backward_validation_carried_graph``, and only with
  the evidence of such an attribute, never on the torch message alone.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ValidationError
from torchlens.options import CaptureOptions
from torchlens.validation._carried_graph import (
    carried_graph_attributes,
    is_freed_graph_error,
    rethrow_freed_graph,
)
from torchlens.validation._stock_layer_grads import (
    _first_leaf_tensor,
    _make_tensor_grad_hook,
    _StockModuleGradCollector,
    _tensor_leaves,
)


class _Pack(dict):
    """A ``dict`` whose attribute lookup raises ``KeyError`` for a missing name."""

    def __getattr__(self, name: str) -> object:
        return self[name]


class _PackModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> _Pack:
        hidden = torch.tanh(self.lin(x))
        return _Pack(logits=hidden, aux=hidden.sum(dim=-1))


class _Halves(nn.Module):
    """A child whose output is a multi-output node's tuple of leaves."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.unbind(x, 0)


class _UsesFirstHalf(nn.Module):
    """Only the first leaf of the child's output reaches the loss."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2)
        self.halves = _Halves()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, _second = self.halves(torch.tanh(self.lin(x)))
        return first


class _KeyTable(nn.Module):
    """Holds non-leaf tensors computed from a parameter at construction (EntNet's key table)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(4, 3)
        self.lin = nn.Linear(3, 2)
        self.keys = [self.emb(torch.tensor([index])).squeeze(0) for index in range(2)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.tanh(x + self.keys[0] + self.keys[1]))


def _sum(output: object) -> torch.Tensor:
    """Sum every floating-point tensor leaf; the validator may hand a container back as a list."""

    if isinstance(output, torch.Tensor):
        return output.float().sum()
    if isinstance(output, dict):
        return torch.stack([_sum(value) for value in output.values()]).sum()
    if isinstance(output, (list, tuple)):
        return torch.stack([_sum(item) for item in output]).sum()
    raise TypeError(f"no tensor leaf in {type(output).__name__}")


def test_dict_subclass_with_hostile_getattr_validates() -> None:
    x = torch.randn(4, 3)
    assert _tensor_leaves(_Pack(a=x)) == [x]
    assert _first_leaf_tensor(_Pack(a=x)) is x
    torch.manual_seed(0)
    model = _PackModel().eval()
    verdict = tl.validate(model, torch.randn(4, 3), scope="backward", seed=0, loss_fn=_sum)
    assert bool(verdict) is True


def test_tensor_grad_hook_tolerates_an_undefined_gradient() -> None:
    collector = _StockModuleGradCollector()
    key = ("halves", 1, 1)
    hook = _make_tensor_grad_hook(collector, key)
    assert hook(None) is None
    assert collector.stock_module_output_grads == {}
    grad = torch.ones(2)
    assert hook(grad) is grad
    assert torch.equal(collector.stock_module_output_grads[key], grad)


def test_multi_output_child_with_an_unused_leaf_validates() -> None:
    torch.manual_seed(0)
    model = _UsesFirstHalf().eval()
    verdict = tl.validate(model, torch.randn(2, 3), scope="backward", seed=0, loss_fn=_sum)
    assert bool(verdict) is True


def test_carried_graph_attributes_lists_graph_bearing_attributes() -> None:
    assert carried_graph_attributes(_KeyTable()) == ["keys[0]", "keys[1]"]
    assert carried_graph_attributes(nn.Linear(2, 2)) == []
    assert carried_graph_attributes(_PackModel()) == []


def test_carried_graph_refuses_typed_after_an_earlier_backward() -> None:
    torch.manual_seed(0)
    model = _KeyTable().eval()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, capture=CaptureOptions(backward_ready=True))
    trace.backward(trace.output_ops[0].out.sum())
    with pytest.raises(ValidationError) as info:
        tl.validate(model, x, scope="backward", seed=0, loss_fn=_sum)
    assert info.value.fields["code"] == "backward_validation_carried_graph"
    assert info.value.fields["carried_graph_attributes"] == ["keys[0]", "keys[1]"]
    assert "keys[0]" in str(info.value)
    assert info.value.__cause__ is not None and is_freed_graph_error(info.value.__cause__)


def test_carried_graph_refuses_typed_on_a_fresh_model() -> None:
    """The stock pass is the first backward; the captured pass is the second."""

    torch.manual_seed(0)
    model = _KeyTable().eval()
    with pytest.raises(ValidationError) as info:
        tl.validate(model, torch.randn(2, 3), scope="backward", seed=0, loss_fn=_sum)
    assert info.value.fields["code"] == "backward_validation_carried_graph"


def test_freed_graph_error_without_evidence_propagates_unchanged() -> None:
    freed = RuntimeError("Trying to backward through the graph a second time (or ...)")
    assert is_freed_graph_error(freed)
    with pytest.raises(RuntimeError) as info:
        rethrow_freed_graph(nn.Linear(2, 2), freed)
    assert info.value is freed
    other = RuntimeError("shape mismatch")
    assert not is_freed_graph_error(other)
    with pytest.raises(RuntimeError) as info:
        rethrow_freed_graph(_KeyTable(), other)
    assert info.value is other
