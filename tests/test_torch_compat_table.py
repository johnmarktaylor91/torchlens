"""The measured per-flag torch capability table and the declarative attribute probes."""

from __future__ import annotations

import pytest

from torchlens._utils import _torch_compat as tc
from torchlens._utils._torch_compat_table import (
    CAPABILITY_ROWS,
    MEASURED_TORCH_VERSIONS,
    CapabilityRow,
)


# Kept by the shrink-r4 coverage proof: sole branch coverage of _utils/_torch_compat.py.
def test_attribute_probed_flags_agree_with_the_live_probe() -> None:
    """A flag probed from its declared paths reads what those paths resolve to now."""

    assert set(tc._ATTRIBUTE_PROBES) <= set(CAPABILITY_ROWS)
    for flag in tc._ATTRIBUTE_PROBES:
        assert getattr(tc, flag) is tc._probe_attribute_paths(flag), flag


# Kept by the shrink-r4 coverage proof: sole branch coverage of _utils/_torch_compat_table.py.
def test_measured_class_covers_each_shape() -> None:
    """The classification distinguishes constant, varying and unmeasured rows."""

    every = MEASURED_TORCH_VERSIONS
    assert CapabilityRow("HAS_X").measured_class == "constant-true"
    assert CapabilityRow("HAS_X", absent_on=every).measured_class == "constant-false"
    assert CapabilityRow("HAS_X", absent_on=every[:1]).measured_class == "varying"
    assert CapabilityRow("HAS_X", measured=False).measured_class == "unmeasured"


# Kept by the shrink-r4 coverage proof: sole branch coverage of _utils/_torch_compat.py.
def test_attribute_path_resolution_returns_none_for_absent_surfaces() -> None:
    """Missing modules and missing attribute segments resolve to ``None``, never raise."""

    assert tc._resolve_attribute_path("torch:_C._VariableFunctions") is not None
    assert tc._resolve_attribute_path("torch:_C.no_such_surface") is None
    assert tc._resolve_attribute_path("torch.utils._device:DeviceContext") is not None
    assert tc._resolve_attribute_path("torch.utils._device:DeviceContext.no_such") is None
    assert tc._resolve_attribute_path("torch._no_such_module:anything") is None


# Kept by the shrink-r4 coverage proof: sole branch coverage of _utils/_torch_compat.py.
def test_attribute_probe_refuses_flags_without_paths() -> None:
    """A bespoke-probe flag cannot be evaluated declaratively by mistake."""

    bespoke = next(flag for flag in CAPABILITY_ROWS if flag not in tc._ATTRIBUTE_PROBES)
    with pytest.raises(ValueError, match="declares no attribute paths"):
        tc._probe_attribute_paths(bespoke)


# Kept by the shrink-r4 coverage proof: sole branch coverage of _utils/_torch_compat.py.
def test_table_getter_flips_its_flag_when_the_surface_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A getter routed through its declared path still marks the capability missing."""

    monkeypatch.setattr(tc, "HAS_DEVICE_CONSTRUCTORS", True)
    monkeypatch.setattr(tc, "_resolve_attribute_path", lambda _spec: None)
    monkeypatch.setenv("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS", "1")
    assert tc.get_device_constructors() is None
    assert tc.HAS_DEVICE_CONSTRUCTORS is False
