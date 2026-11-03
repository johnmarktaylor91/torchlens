"""Regression coverage for capture-wrapper unwrapping's common fast path."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import torch
from pytest import MonkeyPatch

from torchlens import _state
from torchlens._utils import _callable_safety


def _reference_unwrap_capture_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``func`` unwrapped with the pre-fast-path algorithm."""

    current = func
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        original = _state._decorated_to_orig.get(id(current))
        if original is None:
            break
        current = original
    return current


def _authority_ids(
    unwrap: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> frozenset[int]:
    """Build callable-authority IDs with the supplied unwrapping implementation."""

    ids: set[int] = set()
    for funcs in torch.overrides.get_overridable_functions().values():
        for func in funcs:
            ids.add(id(func))
            ids.add(id(unwrap(func)))
    for func in torch.overrides.get_testing_overrides():
        ids.add(id(func))
        ids.add(id(unwrap(func)))
    return frozenset(ids)


def test_unwrap_fast_path_skips_cycle_guard_without_changing_cycle_results(
    monkeypatch: MonkeyPatch,
) -> None:
    """Skip cycle machinery on map misses while preserving every mapped-chain result."""

    first = MagicMock(name="first")
    second = MagicMock(name="second")
    terminal = MagicMock(name="terminal")
    decorated_to_orig = {
        id(first): second,
        id(second): terminal,
    }
    monkeypatch.setattr(_state, "_decorated_to_orig", decorated_to_orig)

    assert _callable_safety._unwrap_capture_wrapper(first) is terminal

    decorated_to_orig.clear()
    decorated_to_orig[id(first)] = first
    assert _callable_safety._unwrap_capture_wrapper(first) is first

    decorated_to_orig[id(first)] = second
    decorated_to_orig[id(second)] = first
    assert _callable_safety._unwrap_capture_wrapper(first) is first

    decorated_to_orig.clear()
    guard_constructor = MagicMock(side_effect=AssertionError("cycle guard allocated"))
    monkeypatch.setattr(_callable_safety, "set", guard_constructor, raising=False)

    assert _callable_safety._unwrap_capture_wrapper(terminal) is terminal
    guard_constructor.assert_not_called()


def test_callable_authority_matches_reference_before_and_after_wrap() -> None:
    """Keep the complete authority ID set identical across lazy wrapper installation."""

    before = _authority_ids(_callable_safety._unwrap_capture_wrapper)
    assert before == _authority_ids(_reference_unwrap_capture_wrapper)

    from torchlens.backends.torch.wrappers import wrap_torch

    wrap_torch()
    after = _authority_ids(_callable_safety._unwrap_capture_wrapper)
    assert after == _authority_ids(_reference_unwrap_capture_wrapper)

    _callable_safety._torch_overridable_callable_ids.cache_clear()
    assert _callable_safety._torch_overridable_callable_ids() == after
