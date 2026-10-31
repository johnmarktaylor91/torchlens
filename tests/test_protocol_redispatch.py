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
twice). The real tensordict package is not a declared extra, so its
end-to-end proof is the Menagerie m4387 row; the stand-in library here is the gate.
"""

from __future__ import annotations

import itertools
import sys
import traceback
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens import _state
from torchlens._utils import _torch_compat
from torchlens.backends.torch import _protocol_redispatch
from torchlens.capture.outcome import FailureOrigin, classify_failure_origin


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


# A tensordict-shaped library: a module-level handler table filled at IMPORT
# time by a registration decorator that reads the torch namespace.
_POST_WRAP_LIBRARY_SOURCE = """
import torch
import torch.nn.functional as F

HANDLED = {}


def implements(torch_function):
    def decorator(impl):
        HANDLED[torch_function] = impl
        return impl

    return decorator


class Bag:
    def __init__(self, tensor):
        self.tensor = tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func not in HANDLED:
            return NotImplemented
        return HANDLED[func](*args, **(kwargs or {}))


@implements(torch.stack)
def _stack(bags, dim=0):
    return torch.stack([bag.tensor for bag in bags], dim) * 2


@implements(torch.flatten)
def _flatten(bag, start_dim=0, end_dim=-1):
    raise RuntimeError("bag cannot flatten")


@implements(F.relu)
def _relu(bag, inplace=False):
    return F.relu(bag.tensor) + 1
"""
_library_ids = itertools.count()


@pytest.fixture
def post_wrap_library() -> Iterator[types.ModuleType]:
    """Import a fresh tensordict-shaped library while wrappers are installed."""

    _ensure_wrapped()
    assert _resolve(torch.stack) is not torch.stack, "torch.stack must be wrapped here"
    name = f"_tl_post_wrap_library_{next(_library_ids)}"
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        # Installed-package path, as tensordict's frames are (classifier zone "library").
        filename = f"/opt/venv/lib/python3/site-packages/{name}.py"
        exec(compile(_POST_WRAP_LIBRARY_SOURCE, filename, "exec"), module.__dict__)
        assert torch.stack in module.HANDLED
        assert _resolve(torch.stack) not in module.HANDLED
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def post_wrap_bag(post_wrap_library: types.ModuleType) -> type:
    """The post-wrap library's bag type."""

    return post_wrap_library.Bag


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
    post_types = [op.type for op in post_trace.layer_list]
    assert "stack" in post_types
    assert post_types == [op.type for op in pre_trace.layer_list]
    expected = (torch.stack([x, x + 1]) * 2).sum()
    assert torch.allclose(post_trace.layer_list[-1].out, expected)
    assert tl.validate(_bag_stacking_model(pre_wrap_bag), x, scope="forward")
    # Validation's pristine ground truth runs on unwrapped torch: the unwrap
    # aliased the library's wrapper-keyed table onto the originals first.
    assert tl.validate(_bag_stacking_model(post_wrap_bag), x, scope="forward")


def test_unwrap_aliases_post_wrap_tables_onto_originals(
    post_wrap_library: types.ModuleType,
) -> None:
    """Unwrapped torch dispatches a post-wrap table; the next wrap undoes the alias."""

    from torchlens.backends.torch.wrappers import unwrap_torch

    wrapped_stack = torch.stack
    imported_keys = set(post_wrap_library.HANDLED)
    unwrap_torch()
    try:
        assert torch.stack is _resolve(wrapped_stack)
        handled = post_wrap_library.HANDLED
        assert torch.stack in handled and wrapped_stack in handled
        bags = [post_wrap_library.Bag(torch.ones(1)), post_wrap_library.Bag(torch.ones(1))]
        assert torch.equal(torch.stack(bags), torch.full((2, 1), 2.0))
    finally:
        _ensure_wrapped()
    assert torch.stack is wrapped_stack
    # Back on the call path, the wrappers redispatch; the foreign table is again
    # exactly what its import built (wrapper keys only).
    assert _resolve(wrapped_stack) not in post_wrap_library.HANDLED
    assert set(post_wrap_library.HANDLED) == imported_keys


def test_alias_container_adds_originals_only_for_wrapper_keys() -> None:
    """Dicts and sets gain the original beside each wrapper key, nothing else."""

    _ensure_wrapped()
    wrapper, original = torch.cat, _resolve(torch.cat)
    table: dict[Any, Any] = {wrapper: "impl", "name": 1}
    members: set[Any] = {wrapper, 3}
    ledger, _ = _state.wrap_epoch_ledgers()
    try:
        assert _protocol_redispatch._alias_container(table, ledger) == 1
        assert _protocol_redispatch._alias_container(members, ledger) == 1
        assert table == {wrapper: "impl", "name": 1, original: "impl"}
        assert members == {wrapper, 3, original}
        assert _protocol_redispatch._alias_container(table, ledger) == 0
    finally:
        _protocol_redispatch.undo_table_aliases()
    assert table == {wrapper: "impl", "name": 1}
    assert members == {wrapper, 3}


def test_undo_leaves_a_re_registered_original_alone() -> None:
    """An original key whose value changed since the alias is the library's now."""

    _ensure_wrapped()
    wrapper, original = torch.cat, _resolve(torch.cat)
    table: dict[Any, Any] = {wrapper: "impl"}
    ledger, _ = _state.wrap_epoch_ledgers()
    assert _protocol_redispatch._alias_container(table, ledger) == 1
    table[original] = "re-registered"
    _protocol_redispatch.undo_table_aliases()
    assert table == {wrapper: "impl", original: "re-registered"}


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


def _capture_failure(
    forward: Callable[[torch.Tensor], Any], expected: type[BaseException] = TypeError
) -> BaseException:
    """Return the ``expected`` exception a capture of ``forward`` raises."""

    class Model(nn.Module):
        """Runs ``forward``."""

        def forward(self, x: torch.Tensor) -> Any:
            return forward(x)

    with pytest.raises(expected) as info:
        tl.trace(Model(), torch.ones(2))
    return info.value


def test_unrelated_type_error_in_capture_keeps_torch_traceback_and_user_origin() -> None:
    """A TypeError that is no declined dispatch re-raises from the wrapper frame."""

    exc = _capture_failure(lambda x: torch.matmul(x, "not a tensor"))
    assert classify_failure_origin(exc) is FailureOrigin.USER_OP
    frames = traceback.extract_tb(exc.__traceback__)
    assert "_protocol_redispatch.py" not in {Path(frame.filename).name for frame in frames}
    assert Path(frames[-1].filename).name == "wrappers.py"
    assert "out_orig = func(" in (frames[-1].line or "")


def test_genuine_decline_in_capture_classifies_user_op(post_wrap_bag: type) -> None:
    """A dispatch every handler declines twice is still the user's op."""

    exc = _capture_failure(lambda x: torch.cat([post_wrap_bag(x), post_wrap_bag(x)]))
    assert "Multiple dispatch failed for 'torch.cat'" in str(exc)
    assert classify_failure_origin(exc) is FailureOrigin.USER_OP


def test_handler_error_after_redispatch_classifies_user_op(post_wrap_bag: type) -> None:
    """A library handler's own error after redispatch is the user's op failing."""

    exc = _capture_failure(lambda x: torch.flatten(post_wrap_bag(x)), RuntimeError)
    assert "bag cannot flatten" in str(exc)
    frames = traceback.extract_tb(exc.__traceback__)
    assert "_protocol_redispatch.py" in {Path(frame.filename).name for frame in frames}
    assert classify_failure_origin(exc) is FailureOrigin.USER_OP
    # The classifier matches the re-offer line by source text; pin it exists.
    from torchlens.capture import outcome

    source = Path(_protocol_redispatch.__file__).read_text()
    assert outcome._PROTOCOL_REDISPATCH_MARKER in source
