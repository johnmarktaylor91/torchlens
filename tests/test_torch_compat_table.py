"""The measured per-flag torch capability table and the declarative attribute probes."""

from __future__ import annotations

import pytest

from torchlens.utils import _torch_compat as tc
from torchlens.utils._torch_compat_table import (
    CAPABILITY_ROWS,
    MEASURED_TORCH_VERSIONS,
    CapabilityRow,
)


def test_table_rows_match_the_capability_flags_exactly() -> None:
    """Every snapshot flag has exactly one row and every row names a snapshot flag."""

    assert set(CAPABILITY_ROWS) == set(tc._CAPABILITY_ATTRS)
    assert all(row.flag == flag for flag, row in CAPABILITY_ROWS.items())


def test_absent_on_lists_only_measured_versions_in_order() -> None:
    """``absent_on`` is an ordered subset of the measured matrix."""

    order = {version: index for index, version in enumerate(MEASURED_TORCH_VERSIONS)}
    for row in CAPABILITY_ROWS.values():
        assert set(row.absent_on) <= set(order), row.flag
        assert list(row.absent_on) == sorted(row.absent_on, key=order.__getitem__), row.flag
        if not row.measured:
            assert row.absent_on == (), row.flag


def test_attribute_probed_flags_agree_with_the_live_probe() -> None:
    """A flag probed from its declared paths reads what those paths resolve to now."""

    assert set(tc._ATTRIBUTE_PROBES) <= set(CAPABILITY_ROWS)
    for flag in tc._ATTRIBUTE_PROBES:
        assert getattr(tc, flag) is tc._probe_attribute_paths(flag), flag


def test_the_compat_module_does_not_import_the_table() -> None:
    """The measured table is tooling data, kept out of the capture stack's import closure."""

    import ast
    import inspect

    tree = ast.parse(inspect.getsource(tc))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "_torch_compat_table" not in imported


def test_measured_class_covers_each_shape() -> None:
    """The classification distinguishes constant, varying and unmeasured rows."""

    every = MEASURED_TORCH_VERSIONS
    assert CapabilityRow("HAS_X").measured_class == "constant-true"
    assert CapabilityRow("HAS_X", absent_on=every).measured_class == "constant-false"
    assert CapabilityRow("HAS_X", absent_on=every[:1]).measured_class == "varying"
    assert CapabilityRow("HAS_X", measured=False).measured_class == "unmeasured"


def test_attribute_path_resolution_returns_none_for_absent_surfaces() -> None:
    """Missing modules and missing attribute segments resolve to ``None``, never raise."""

    assert tc._resolve_attribute_path("torch:_C._VariableFunctions") is not None
    assert tc._resolve_attribute_path("torch:_C.no_such_surface") is None
    assert tc._resolve_attribute_path("torch.utils._device:DeviceContext") is not None
    assert tc._resolve_attribute_path("torch.utils._device:DeviceContext.no_such") is None
    assert tc._resolve_attribute_path("torch._no_such_module:anything") is None


def test_attribute_probe_refuses_flags_without_paths() -> None:
    """A bespoke-probe flag cannot be evaluated declaratively by mistake."""

    bespoke = next(flag for flag in CAPABILITY_ROWS if flag not in tc._ATTRIBUTE_PROBES)
    with pytest.raises(ValueError, match="declares no attribute paths"):
        tc._probe_attribute_paths(bespoke)


def test_table_getter_flips_its_flag_when_the_surface_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A getter routed through its declared path still marks the capability missing."""

    monkeypatch.setattr(tc, "HAS_DEVICE_CONSTRUCTORS", True)
    monkeypatch.setattr(tc, "_resolve_attribute_path", lambda _spec: None)
    monkeypatch.setenv("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS", "1")
    assert tc.get_device_constructors() is None
    assert tc.HAS_DEVICE_CONSTRUCTORS is False


def test_snapshot_forces_the_meta_item_and_mha_switch_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot reports the real probe result, not the unprobed ``False`` placeholder."""

    monkeypatch.setattr(tc, "_META_ITEM_GUARD_PROBED", False)
    monkeypatch.setattr(tc, "HAS_META_ITEM_GUARD", False)
    monkeypatch.setattr(tc, "_MHA_FASTPATH_SWITCH_PROBED", False)
    monkeypatch.setattr(tc, "HAS_MHA_FASTPATH_SWITCH", False)
    snapshot = tc.get_torch_capability_snapshot()
    assert snapshot["HAS_META_ITEM_GUARD"] is tc._probe_meta_item_guard()
    assert snapshot["HAS_MHA_FASTPATH_SWITCH"] is tc._probe_mha_fastpath_switch()
