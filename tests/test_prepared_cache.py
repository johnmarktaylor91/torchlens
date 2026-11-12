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
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _tensor_label
from torchlens.backends.torch import _prepared_cache as cache_mod
from torchlens.backends.torch._prepared_cache import kept_prepared

#: The head's ``forward`` object at each root forward call, in order. Kept outside the
#: model because a capture rewinds attributes the forward mutates.
_SEEN: list[Any] = []


@pytest.fixture(autouse=True)
def _clear_seen() -> Iterator[None]:
    """Start and end every test with an empty ``_SEEN``."""

    _SEEN.clear()
    yield
    _SEEN.clear()


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
    """Two blocks and a head; notes the head's ``forward`` object in ``_SEEN``; can raise."""

    def __init__(self) -> None:
        """Build the model."""

        super().__init__()
        self.a = _Block()
        self.b = _Block()
        self.head = nn.Linear(4, 2)
        self.fail = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both blocks and the head, or raise mid-forward when asked."""

        _SEEN.append(self.head.__dict__.get("forward"))
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


def _bare(model: nn.Module) -> bool:
    """Return whether no module of ``model`` holds an instance ``forward``."""

    return all(fwd is None for fwd in _forwards(model).values())


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


def _footprint(model: nn.Module) -> tuple[Any, ...]:
    """Return, by name, what TorchLens leaves on or about ``model`` between captures."""

    tensors = [*model.named_parameters(), *model.named_buffers()]
    return (
        sorted(name for name, fwd in _forwards(model).items() if fwd is not None),
        sorted(name for name, tensor in tensors if hasattr(tensor, "_tl")),
        sorted(name for name, p in model.named_parameters() if p in _tensor_label._PARAM_REGISTRY),
        sorted(name for name, m in model.named_modules() if m in _tensor_label._MODULE_REGISTRY),
    )


def _assert_untouched(model: nn.Module) -> None:
    """Assert ``model`` is left as a capture outside any block leaves its twin.

    The twin is a fresh ``_Model`` captured twice outside a block; ``model`` must
    match its footprint, hold no kept state, and deep-copy and pickle cleanly.
    """

    twin, x = _model()
    tl.trace(twin, x)
    tl.trace(twin, x)
    assert _footprint(model) == _footprint(twin)
    assert _bare(model)
    assert id(model) not in cache_mod._KEPT
    assert _bare(copy.deepcopy(model))
    pickle.loads(pickle.dumps(model))


def test_captures_in_the_block_reuse_one_wrapper_object() -> None:
    """Captures in the block install the same wrapper; captures outside build a new one."""

    model, x = _model()
    tl.trace(model, x)
    tl.trace(model, x)
    assert _SEEN[0] is not None and _SEEN[0] is not _SEEN[1]
    _SEEN.clear()
    with kept_prepared(model):
        tl.trace(model, x)
        assert _bare(model)
        tl.trace(model, x)
        tl.trace(model, x)
        assert _SEEN[0] is not None
        assert _SEEN[0] is _SEEN[1] is _SEEN[2]
        assert _bare(model)
    _assert_untouched(model)


def test_replaced_submodule_gets_fresh_wrappers() -> None:
    """A submodule replaced between captures is wrapped afresh; the old one stays bare."""

    model, x = _model()
    with kept_prepared(model):
        tl.trace(model, x)
        old_head = model.head
        model.head = nn.Linear(4, 2)
        trace = tl.trace(model, x)
        tl.trace(model, x)
        assert _SEEN[1] is not _SEEN[0]
        assert _SEEN[2] is _SEEN[1]
        assert "forward" not in old_head.__dict__
        assert torch.equal(trace.ops[-1].out, model.head(model.b(model.a(x))))
    _assert_untouched(model)


def test_added_and_removed_submodules_are_seen() -> None:
    """A submodule added to or removed from the tree is seen by the next capture."""

    model, x = _model()
    with kept_prepared(model):
        tl.trace(model, x)
        model.a.extra = nn.Identity()
        model.a.forward = lambda inp: model.a.extra(_Block.forward(model.a, inp))  # type: ignore[method-assign]
        added = tl.trace(model, x)
        assert any(module.address == "a.extra" for module in added.modules)
        assert _SEEN[1] is not _SEEN[0]
        del model.a.forward
        del model.a.extra
        removed = tl.trace(model, x)
        assert not any(module.address == "a.extra" for module in removed.modules)
    _assert_untouched(model)


def test_replaced_parameter_gets_fresh_wrappers() -> None:
    """A parameter object swapped between captures makes the next capture build afresh."""

    model, x = _model()
    with kept_prepared(model):
        tl.trace(model, x)
        model.head.weight = nn.Parameter(torch.ones(2, 4))
        trace = tl.trace(model, x)
        assert _SEEN[1] is not _SEEN[0]
        assert torch.equal(trace.ops[-1].out, model.head(model.b(model.a(x))))
        assert any(p.address == "head.weight" for p in trace.params)
    _assert_untouched(model)


def test_user_forward_set_between_captures_is_respected() -> None:
    """An instance ``forward`` the user sets between captures is wrapped and called."""

    model, x = _model()
    calls: list[int] = []
    with kept_prepared(model):
        tl.trace(model, x)
        plain = type(model.head).forward.__get__(model.head)

        def counted(inp: torch.Tensor) -> torch.Tensor:
            """Count the call and run the class forward."""

            calls.append(1)
            return plain(inp)

        model.head.forward = counted  # type: ignore[method-assign]
        tl.trace(model, x)
        assert calls == [1]
        assert _SEEN[1] is not _SEEN[0]
        assert model.head.__dict__.get("forward") is counted
        del model.head.forward
        tl.trace(model, x)
        assert calls == [1]
        assert _SEEN[2] is not _SEEN[1]
    _assert_untouched(model)


class _Head(nn.Linear):
    """A head whose class ``forward`` one test swaps mid-block (and puts back)."""


def test_changed_class_forward_gets_fresh_wrappers() -> None:
    """A class ``forward`` replaced between captures is the one the next capture runs."""

    model, x = _model()
    model.head = _Head(4, 2)
    try:
        with kept_prepared(model):
            tl.trace(model, x)
            _Head.forward = lambda self, inp: nn.Linear.forward(self, inp) * 2  # type: ignore[method-assign]
            trace = tl.trace(model, x)
            assert _SEEN[1] is not _SEEN[0]
            expected = nn.Linear.forward(model.head, model.b(model.a(x))) * 2
            assert torch.equal(trace.ops[-1].out, expected)
    finally:
        del _Head.forward
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
        with pytest.raises(Exception, match="planned failure"):
            tl.trace(model, x)
        model.fail = False
        assert _bare(model)
        assert _same_records(reference, _record(tl.trace(model, x)))
    _assert_untouched(model)


def test_nested_blocks_put_back_at_the_outermost_exit() -> None:
    """Re-entering the block for one model nests; only the outermost exit puts back."""

    model, x = _model()
    with kept_prepared(model):
        with kept_prepared(model):
            tl.trace(model, x)
        assert id(model) in cache_mod._KEPT
        tl.trace(model, x)
        assert _SEEN[0] is _SEEN[1]
    _assert_untouched(model)


def test_other_models_are_unaffected_by_a_block() -> None:
    """A model outside the block is prepared and cleaned per capture as before."""

    model, x = _model()
    other, y = _model()
    with kept_prepared(model):
        tl.trace(model, x)
        tl.trace(other, y)
        tl.trace(other, y)
        assert _SEEN[-1] is not _SEEN[-2]
        assert id(other) not in cache_mod._KEPT
    _assert_untouched(model)
    _assert_untouched(other)


def test_kept_prepared_refuses_a_non_module() -> None:
    """Only an ``nn.Module`` can be kept prepared."""

    with pytest.raises(TypeError, match="nn.Module"), kept_prepared(lambda t: t):  # type: ignore[arg-type]
        pass
