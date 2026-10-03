"""Lazy capability latches: registry completeness, restore, and un-poisoning.

The ``_torch_compat`` lazy ``HAS_*`` probes latch on first use, which makes
them the one capability class a test can POISON: a probe fired while
``sys.modules`` is stubbed latches the wrong verdict for the whole process
(the recorded ``b7fe953e`` incident -- later compat snapshots and
generated-doc gates flapped; it was fixed per-test rather than systemically).
These tests pin the systemic closure: the latch registry is complete, the
snapshot/restore API un-poisons exactly, and the suite-wide autouse fixture
(``tests/conftest.py::_restore_lazy_capability_probes``) bounds any mis-latch
to the test that caused it.
"""

from __future__ import annotations

import sys
import types

import pytest

from torchlens.utils import _torch_compat


class _FakeDTensor:
    """Stand-in DTensor type whose identity marks a poisoned latch."""


def test_lazy_probe_registry_is_complete() -> None:
    """Every ``*_PROBED`` latch and its family attrs live in the registry.

    A new lazy latch landing outside ``_LAZY_PROBE_FAMILIES`` would be
    invisible to the snapshot/restore fixture and re-open the poisoning class.
    """

    module_attrs = vars(_torch_compat)
    probed_attrs = {name for name in module_attrs if name.endswith("_PROBED")}
    assert probed_attrs == set(_torch_compat._LAZY_PROBE_FAMILIES), (
        "lazy *_PROBED latches and _LAZY_PROBE_FAMILIES diverged; register the "
        f"new latch: {sorted(probed_attrs ^ set(_torch_compat._LAZY_PROBE_FAMILIES))}"
    )
    for probed_attr, family in _torch_compat._LAZY_PROBE_FAMILIES.items():
        for attr in family:
            assert attr in module_attrs, f"{probed_attr} names unknown family attr {attr}"
    snapshot = _torch_compat.capability_probe_snapshot()
    assert "_LAZY_TORCH_IMPORTS_WARMED" in snapshot
    for probed_attr in probed_attrs:
        assert probed_attr in snapshot


def test_stubbed_probe_poison_is_undone_by_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe latched under a stubbed ``sys.modules`` restores exactly.

    This is the ``b7fe953e`` incident in miniature: the DTensor probe resolves
    a FAKE type from a stubbed module and latches ``HAS_DTENSOR`` for the
    process; ``restore_capability_probes`` must return every family attr to
    its pre-poison value so the next consumer re-probes the real runtime.
    """

    snapshot = _torch_compat.capability_probe_snapshot()

    fake_module = types.ModuleType("torch.distributed.tensor")
    fake_module.DTensor = _FakeDTensor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", fake_module)
    _torch_compat._DTENSOR_PROBED = False
    _torch_compat._DTENSOR_TYPE = None
    _torch_compat.HAS_DTENSOR = False

    resolved = _torch_compat.get_dtensor_type(force_probe=True)
    assert resolved is _FakeDTensor, "the stub did not reach the probe; test is vacuous"
    assert _torch_compat.HAS_DTENSOR is True
    assert _torch_compat._DTENSOR_PROBED is True

    _torch_compat.restore_capability_probes(snapshot)
    assert snapshot["_DTENSOR_PROBED"] == _torch_compat._DTENSOR_PROBED
    assert snapshot["HAS_DTENSOR"] == _torch_compat.HAS_DTENSOR
    assert _torch_compat._DTENSOR_TYPE is snapshot["_DTENSOR_TYPE"]
    assert _torch_compat._DTENSOR_TYPE is not _FakeDTensor


def test_poison_a_latch_and_lean_on_the_autouse_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberately leave a poisoned latch for the autouse fixture to clean."""

    fake_module = types.ModuleType("torch.distributed.tensor")
    fake_module.DTensor = _FakeDTensor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", fake_module)
    _torch_compat._DTENSOR_PROBED = False
    _torch_compat._DTENSOR_TYPE = None
    _torch_compat.HAS_DTENSOR = False
    assert _torch_compat.get_dtensor_type(force_probe=True) is _FakeDTensor
    assert _torch_compat._DTENSOR_TYPE is _FakeDTensor
    # NO manual restore: teardown must undo this or the next test fails.


def test_autouse_fixture_unpoisoned_the_prior_test() -> None:
    """Runs after the poisoning test above; the fake type must be gone."""

    assert _torch_compat._DTENSOR_TYPE is not _FakeDTensor, (
        "the autouse capability-probe restore did not undo the prior test's poisoned DTensor latch"
    )
