"""A warm ``wrap_torch`` reuses what the previous install of the epoch verified.

The legacy-constructor layout verdicts are reused only for the identical roster
(each row the same class object); any other roster is a cache miss that is
probed in full and never flips ``HAS_LEGACY_CONSTRUCTOR_NEW_PATCH``. The belt
pre-filter reuses its all-swept snapshot only for the identical ``sys.modules``
values, compared by identity.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _torch_compat
from torchlens.backends.torch import belt, legacy_ctors
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch


@pytest.fixture
def probe_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Count layout probes run by the legacy-constructor install, after one warm trace."""

    tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
    if not _torch_compat.HAS_LEGACY_CONSTRUCTOR_NEW_PATCH:
        pytest.skip("legacy constructor patching is unavailable on this interpreter")
    calls: list[int] = []
    real_probe = legacy_ctors.probe_type_new_slot_patches

    def counting_probe(classes: Any) -> list[bool]:
        """Record the batch size, then probe for real."""

        calls.append(len(classes))
        return real_probe(classes)

    monkeypatch.setattr(legacy_ctors, "probe_type_new_slot_patches", counting_probe)
    yield calls


def test_identical_roster_reuses_the_verified_layouts(probe_calls: list[int]) -> None:
    """A second install in the epoch does only the installs: no layout probe."""

    wrap_torch()
    wrap_torch()
    assert probe_calls == []
    assert legacy_ctors.installed_legacy_constructor_classes()["FloatTensor"] is torch.FloatTensor


def test_rebound_class_is_a_cache_miss_not_a_degradation(
    probe_calls: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A roster row bound to another class is probed again; the capability flag stays on."""

    wrap_torch()
    assert probe_calls == []
    with monkeypatch.context() as patch:
        patch.setattr(torch, "HalfTensor", torch.FloatTensor)
        wrap_torch()
        assert len(probe_calls) == 1
        wrap_torch()
        assert len(probe_calls) == 1
    wrap_torch()
    assert len(probe_calls) == 2
    assert _torch_compat.HAS_LEGACY_CONSTRUCTOR_NEW_PATCH is True
    assert legacy_ctors.installed_legacy_constructor_classes()["HalfTensor"] is torch.HalfTensor


def test_unwrap_drops_the_verified_roster(probe_calls: list[int]) -> None:
    """After ``unwrap_torch`` the next install probes every layout again."""

    wrap_torch()
    assert legacy_ctors._PROBED_ROSTER is not None
    unwrap_torch()
    try:
        assert legacy_ctors._PROBED_ROSTER is None
        wrap_torch()
        assert len(probe_calls) == 1
    finally:
        wrap_torch()
    assert torch.FloatTensor(3).shape == (3,)


class _RaisingEq:
    """A ``sys.modules`` entry whose ``__eq__`` raises: the pre-filter must never call it."""

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("compared by equality")

    __hash__ = object.__hash__


def test_belt_prefilter_compares_entries_by_identity() -> None:
    """A replaced entry under the same name reads as new; ``__eq__`` is never called."""

    tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
    belt.sweep_stale_belt_references()
    assert belt._unswept_module_items() == []
    name = "_tl_warm_cache_entry"
    first = types.ModuleType(name)
    sys.modules[name] = first
    try:
        assert [key for key, _value in belt._unswept_module_items()] == [name]
        belt.sweep_stale_belt_references()
        assert belt._unswept_module_items() == []
        second = types.ModuleType(name)
        sys.modules[name] = second
        assert belt._unswept_module_items() == [(name, second)]
        belt.sweep_stale_belt_references()
        sys.modules[name] = _RaisingEq()
        assert [key for key, _value in belt._unswept_module_items()] == [name]
        belt.sweep_stale_belt_references()
        assert belt._unswept_module_items() == []
    finally:
        sys.modules.pop(name, None)
    assert belt._unswept_module_items() == []
