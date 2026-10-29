"""Declined-protocol redispatch: handler tables built while wrappers are installed.

A library imported after ``wrap_torch()`` keys its ``__torch_function__``
handler table on the namespace it reads at import time -- the TorchLens
WRAPPER -- while torch's protocol hands handlers the ORIGINAL. tensordict
0.14.3 is the observed case (Menagerie row m4387, f-dreamer_rssm): every
TensorDict ``torch.stack`` raised "Multiple dispatch failed ... all
__torch_function__ handlers returned NotImplemented" once the library was
imported post-wrap. The wrapper now re-offers a fully declined dispatch to the
same handlers under its own identity; these tests pin that, the negative
controls (a genuine decline still raises the same error, a handler never runs
twice), and the real tensordict reproduction in a fresh interpreter.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import _protocol_redispatch
from torchlens.utils import _torch_compat


def _ensure_wrapped() -> None:
    """Force the lazy torch wrap through the public capture path."""

    tl.trace(nn.Linear(2, 2), torch.randn(1, 2))


def _resolve(fn: Any) -> Any:
    """Follow the wrapper ledger to the original callable."""

    seen: set[int] = set()
    while id(fn) in _state._decorated_to_orig and id(fn) not in seen:
        seen.add(id(fn))
        fn = _state._decorated_to_orig[id(fn)]
    return fn


def _make_bag_type(table: dict[Callable[..., Any], Callable[..., Any]]) -> type:
    """Build a tensordict-shaped container whose handler consults ``table``."""

    class Bag:
        """Non-Tensor container dispatching ``__torch_function__`` by table lookup."""

        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        @classmethod
        def __torch_function__(
            cls,
            func: Callable[..., Any],
            types: tuple[type, ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            if func not in table:
                return NotImplemented
            return table[func](*args, **(kwargs or {}))

    return Bag


def _stack_bags(bags: list[Any], dim: int = 0) -> torch.Tensor:
    """Table implementation of ``torch.stack`` for bags (runs wrapped ops)."""

    return torch.stack([bag.tensor for bag in bags], dim) * 2


def _relu_bag(bag: Any, inplace: bool = False) -> torch.Tensor:
    """Table implementation of ``F.relu`` for a bag."""

    del inplace
    return F.relu(bag.tensor) + 1


@pytest.fixture
def post_wrap_bag() -> type:
    """A bag type whose table was keyed while wrappers were installed."""

    _ensure_wrapped()
    assert _resolve(torch.stack) is not torch.stack, "torch.stack must be wrapped here"
    return _make_bag_type({torch.stack: _stack_bags, F.relu: _relu_bag})


def test_post_wrap_table_dispatches_on_the_fast_path(post_wrap_bag: type) -> None:
    """A C builtin dispatch reaches a wrapper-keyed table outside any capture."""

    bags = [post_wrap_bag(torch.ones(2)), post_wrap_bag(torch.zeros(2))]
    out = torch.stack(bags)
    assert torch.equal(out, torch.tensor([[2.0, 2.0], [0.0, 0.0]]))


def test_post_wrap_table_dispatches_pure_python_functional(post_wrap_bag: type) -> None:
    """A pure-Python functional (``handle_torch_function`` path) reaches it too."""

    out = F.relu(post_wrap_bag(torch.tensor([-1.0, 3.0])))
    assert torch.equal(out, torch.tensor([1.0, 4.0]))


def _bag_stacking_model(bag_type: type) -> nn.Module:
    """A model that stacks two bags of ``bag_type`` inside forward."""

    class Model(nn.Module):
        """Stacks two bags inside forward."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.stack([bag_type(x), bag_type(x + 1)]).sum()

    return Model()


def test_post_wrap_table_capture_matches_original_keyed_table(post_wrap_bag: type) -> None:
    """Inside a capture, a wrapper-keyed table logs what an original-keyed one logs."""

    pre_wrap_bag = _make_bag_type({_resolve(torch.stack): _stack_bags})
    x = torch.randn(3)
    post_trace = tl.trace(_bag_stacking_model(post_wrap_bag), x)
    pre_trace = tl.trace(_bag_stacking_model(pre_wrap_bag), x)
    post_types = [op.layer_type for op in post_trace.layer_list]
    assert "stack" in post_types
    assert post_types == [op.layer_type for op in pre_trace.layer_list]
    expected = (torch.stack([x, x + 1]) * 2).sum()
    assert torch.allclose(post_trace.layer_list[-1].out, expected)
    assert tl.validate(_bag_stacking_model(post_wrap_bag), x, scope="forward")
    assert tl.validate(_bag_stacking_model(pre_wrap_bag), x, scope="forward")


def test_original_keyed_table_is_unchanged(post_wrap_bag: type) -> None:
    """A table keyed on originals (pre-wrap import) dispatches on the first offer."""

    del post_wrap_bag
    calls: list[Callable[..., Any]] = []

    def stack_impl(bags: list[Any], dim: int = 0) -> torch.Tensor:
        calls.append(torch.stack)
        return torch.stack([bag.tensor for bag in bags], dim)

    pre_wrap_bag = _make_bag_type({_resolve(torch.stack): stack_impl})
    out = torch.stack([pre_wrap_bag(torch.ones(1)), pre_wrap_bag(torch.ones(1))])
    assert out.shape == (2, 1)
    assert len(calls) == 1


def test_genuine_decline_raises_the_original_error(post_wrap_bag: type) -> None:
    """A function no handler accepts still raises torch's own TypeError."""

    with pytest.raises(TypeError, match=r"Multiple dispatch failed for 'torch\.cat'"):
        torch.cat([post_wrap_bag(torch.ones(1)), post_wrap_bag(torch.ones(1))])


def test_enclosing_call_never_reruns_a_handler_that_did_work() -> None:
    """A declined NESTED dispatch is not re-offered by the enclosing wrapper."""

    _ensure_wrapped()
    cat_calls: list[int] = []
    table: dict[Callable[..., Any], Callable[..., Any]] = {}
    bag_type = _make_bag_type(table)

    def cat_impl(bags: list[Any], dim: int = 0) -> torch.Tensor:
        cat_calls.append(1)
        # ``torch.unbind`` is in no table: this nested dispatch is declined.
        return torch.unbind(bags[0])

    table[torch.cat] = cat_impl
    with pytest.raises(TypeError, match=r"'torch\.unbind'"):
        torch.cat([bag_type(torch.ones(1)), bag_type(torch.ones(1))])
    assert cat_calls == [1]


def test_redispatch_stays_off_without_the_disabled_sentinel(
    post_wrap_bag: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``HAS_DISABLED_TORCH_FUNCTION_IMPL`` False the original error propagates."""

    monkeypatch.setattr(_torch_compat, "HAS_DISABLED_TORCH_FUNCTION_IMPL", False)
    with pytest.raises(TypeError, match="Multiple dispatch failed"):
        torch.stack([post_wrap_bag(torch.ones(1)), post_wrap_bag(torch.ones(1))])


def test_overloaded_args_follow_torch_protocol_order() -> None:
    """Exact Tensors and the disabled sentinel are skipped; subclasses go first."""

    class Base(torch.Tensor):
        """Tensor subclass with the default handler."""

    class Child(Base):
        """Subclass of ``Base``."""

    disabled = _torch_compat.get_disabled_torch_function_impl()
    assert disabled is not None
    base = torch.ones(1).as_subclass(Base)
    child = torch.ones(1).as_subclass(Child)
    param = nn.Parameter(torch.ones(1))
    ordered = _protocol_redispatch._overloaded_args(
        ([torch.ones(1), base, param], child), {"other": base}, disabled
    )
    assert [type(value) for value in ordered] == [Child, Base]


@pytest.mark.heavy
def test_real_tensordict_imported_after_wrap_torch() -> None:
    """tensordict imported after ``wrap_torch()`` stacks, cats and unbinds (m4387)."""

    pytest.importorskip("tensordict")
    script = textwrap.dedent(
        """
        import torch
        from torchlens.backends.torch.wrappers import wrap_torch

        wrap_torch()
        from tensordict import TensorDict

        td = TensorDict({"a": torch.zeros(3, 2)}, batch_size=[3])
        assert type(torch.stack([td, td])).__name__ == "TensorDict"
        assert type(torch.cat([td, td])).__name__ == "TensorDict"
        assert len(torch.unbind(td, 0)) == 3
        assert type(torch.zeros_like(td)).__name__ == "TensorDict"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("OK")
