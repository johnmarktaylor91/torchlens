"""The opt-in ``kept_prepared`` block: reuse under an identity check, and a clean exit.

Inside the block a model keeps the session ``forward`` wrappers between
captures; the next capture reuses them only while the module tree, every
parameter object and each wrapped ``forward`` are unchanged, and prepares afresh
on any change. These tests show the reuse, a re-preparation for each kind of
change, that a capture inside the block records exactly what one outside it
records, that ``requires_grad`` is restored after every capture, and that the
model is left as a capture outside the block leaves it when the block exits, on
every path, exceptions included.
"""

from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _tensor_label
from torchlens.backends.torch import _prepared_cache as cache_mod
from torchlens.backends.torch._prepared_cache import kept_prepared


class _Block(nn.Module):
    """A linear layer, a buffer and a stateless activation."""

    offset: torch.Tensor

    def __init__(self) -> None:
        """Build the block."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.register_buffer("offset", torch.zeros(4))
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shift and activate."""

        return self.act(self.fc(x) + self.offset)


class _Model(nn.Module):
    """Two blocks and a head; the head can be told to raise."""

    def __init__(self) -> None:
        """Build the model."""

        super().__init__()
        self.a = _Block()
        self.b = _Block()
        self.head = nn.Linear(4, 2)
        self.fail = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both blocks and the head, or raise mid-forward when asked."""

        hidden = self.b(self.a(x))
        if self.fail:
            raise RuntimeError("planned failure")
        return self.head(hidden)


def _model() -> tuple[_Model, torch.Tensor]:
    """Return a seeded model in eval mode and an input."""

    torch.manual_seed(0)
    return _Model().eval(), torch.randn(3, 4)


def _forwards(model: nn.Module) -> dict[str, Any]:
    """Return each module's instance ``forward`` (``None`` when there is none), by name."""

    return {name: module.__dict__.get("forward") for name, module in model.named_modules()}


def _record(trace: Any) -> list[Any]:
    """Return what a capture recorded, in a form two captures can be compared by."""

    ops = [
        (
            op.label,
            op.func_name,
            tuple(op.parents),
            tuple(op.children),
            tuple(op.shape) if op.shape is not None else None,
            op.out.detach().clone() if isinstance(op.out, torch.Tensor) else None,
            tuple(op.modules),
        )
        for op in trace.ops
    ]
    modules = [(module.address, module.num_calls) for module in trace.modules]
    params = sorted((p.address, tuple(p.shape), p.is_trainable) for p in trace.params)
    return [ops, modules, params]


def _same_records(left: list[Any], right: list[Any]) -> bool:
    """Compare two ``_record`` results, tensors by value."""

    if left[1:] != right[1:] or len(left[0]) != len(right[0]):
        return False
    for one, two in zip(left[0], right[0], strict=True):
        if one[:5] != two[:5] or one[6] != two[6]:
            return False
        if (one[5] is None) != (two[5] is None):
            return False
        if one[5] is not None and not torch.equal(one[5], two[5]):
            return False
    return True


def _assert_untouched(model: nn.Module) -> None:
    """Assert ``model`` holds no TorchLens callable or kept state and copies cleanly."""

    assert all(fwd is None for fwd in _forwards(model).values())
    assert id(model) not in cache_mod._KEPT
    for tensor in [*model.parameters(), *model.buffers()]:
        assert not hasattr(tensor, "_tl")
        assert tensor not in _tensor_label._PARAM_REGISTRY
    assert not any(module in _tensor_label._MODULE_REGISTRY for module in model.modules())
    clone = copy.deepcopy(model)
    assert all(fwd is None for fwd in _forwards(clone).values())
    pickle.loads(pickle.dumps(model))


def test_second_capture_reuses_the_kept_wrappers() -> None:
    """Between captures in the block the wrappers stay; the next capture reuses them."""

    model, x = _model()
    with kept_prepared(model):
        tl.trace(model, x)
        kept = _forwards(model)
        assert kept[""] is None
        assert all(kept[name] is not None for name in ("a", "a.fc", "b.act", "head"))
        tl.trace(model, x)
        assert _forwards(model) == kept
    _assert_untouched(model)


def test_replaced_submodule_re_prepares() -> None:
    """A submodule replaced between captures is wrapped; the old one is put back."""

    model, x = _model()
    with kept_prepared(model):
        tl.trace(model, x)
        old_head = model.head
        model.head = nn.Linear(4, 2)
        trace = tl.trace(model, x)
        assert "forward" not in old_head.__dict__
        assert model.head.__dict__.get("forward") is not None
        assert any(module.address == "head" for module in trace.modules)
    _assert_untouched(model)
    assert "forward" not in old_head.__dict__


def test_added_and_removed_submodules_re_prepare() -> None:
    """A submodule added to or removed from the tree is seen by the next capture."""

    model, x = _model()
    with kept_prepared(model):
        tl.trace(model, x)
        before = _forwards(model)
        model.a.extra = nn.Identity()
        tl.trace(model, x)
        after = _forwards(model)
        assert after["a.extra"] is not None
        assert after["a.fc"] is not before["a.fc"]
        removed = model.a.extra
        del model.a.extra
        tl.trace(model, x)
        assert "forward" not in removed.__dict__
    _assert_untouched(model)


def test_replaced_parameter_re_prepares() -> None:
    """A parameter object swapped between captures makes the next capture prepare afresh."""

    model, x = _model()
    with kept_prepared(model):
        tl.trace(model, x)
        before = _forwards(model)
        model.head.weight = nn.Parameter(torch.ones(2, 4))
        trace = tl.trace(model, x)
        assert _forwards(model)["head"] is not before["head"]
        head_out = model.head(model.b(model.a(x)))
        assert torch.equal(trace.ops[-1].out, head_out)
    _assert_untouched(model)


def test_user_forward_set_between_captures_is_respected() -> None:
    """An instance ``forward`` the user sets between captures replaces the kept wrapper."""

    model, x = _model()
    calls: list[int] = []
    with kept_prepared(model):
        tl.trace(model, x)
        plain = type(model.head).forward.__get__(model.head)

        def counted(inp: torch.Tensor) -> torch.Tensor:
            """Count the call and run the class forward."""

            calls.append(1)
            return plain(inp)

        model.head.forward = counted
        tl.trace(model, x)
        assert calls == [1]
    assert model.head.__dict__.get("forward") is counted
    del model.head.forward
    _assert_untouched(model)


def test_capture_inside_the_block_records_what_one_outside_records() -> None:
    """Differential: the same model and input give the same records in and out of the block."""

    model, x = _model()
    outside = _record(tl.trace(model, x))
    with kept_prepared(model):
        first = _record(tl.trace(model, x))
        second = _record(tl.trace(model, x))
    after = _record(tl.trace(model, x))
    assert _same_records(outside, first)
    assert _same_records(outside, second)
    assert _same_records(outside, after)


def test_requires_grad_is_restored_after_every_capture() -> None:
    """User grad flags are the same after each capture in the block as before it."""

    model, x = _model()
    model.a.fc.weight.requires_grad_(False)
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    with kept_prepared(model):
        for _ in range(2):
            tl.trace(model, x)
            assert {name: p.requires_grad for name, p in model.named_parameters()} == flags
            assert not any(hasattr(p, "_tl") for p in model.parameters())
    _assert_untouched(model)


def test_exit_after_an_exception_in_the_block_leaves_the_model_untouched() -> None:
    """An exception raised in the block body still puts every kept wrapper back."""

    model, x = _model()
    with pytest.raises(ValueError, match="body"), kept_prepared(model):
        tl.trace(model, x)
        raise ValueError("body")
    _assert_untouched(model)


def test_a_failed_capture_in_the_block_keeps_the_block_sound() -> None:
    """A forward that raises inside the block: the next capture works, exit is clean."""

    model, x = _model()
    reference = _record(tl.trace(model, x))
    with kept_prepared(model):
        tl.trace(model, x)
        model.fail = True
        with pytest.raises(RuntimeError, match="planned failure"):
            tl.trace(model, x)
        model.fail = False
        assert _same_records(reference, _record(tl.trace(model, x)))
    _assert_untouched(model)


def test_nested_blocks_put_back_at_the_outermost_exit() -> None:
    """Re-entering the block for one model nests; only the outermost exit puts back."""

    model, x = _model()
    with kept_prepared(model):
        with kept_prepared(model):
            tl.trace(model, x)
        assert _forwards(model)["head"] is not None
        tl.trace(model, x)
    _assert_untouched(model)


def test_other_models_are_unaffected_by_a_block() -> None:
    """A model outside the block is prepared and cleaned per capture as before."""

    model, x = _model()
    other, y = _model()
    with kept_prepared(model):
        tl.trace(model, x)
        tl.trace(other, y)
        assert all(fwd is None for fwd in _forwards(other).values())
    _assert_untouched(model)
    _assert_untouched(other)


def test_kept_prepared_refuses_a_non_module() -> None:
    """Only an ``nn.Module`` can be kept prepared."""

    with pytest.raises(TypeError, match="nn.Module"), kept_prepared(lambda t: t):  # type: ignore[arg-type]
        pass
