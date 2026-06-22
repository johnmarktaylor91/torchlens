"""Tests for the torch version-compatibility shim (torchlens/utils/_torch_compat.py).

The shim lets TorchLens run on torch 2.1+ by routing the two torch-2.4-only
autocast-query APIs (``is_autocast_enabled(device_type)`` /
``get_autocast_dtype(device_type)``) through version-neutral wrappers.

On the CI/dev torch (>=2.4) the shim's modern branch must be *byte-identical* to
calling torch directly. The legacy branch (torch 2.1-2.3) is exercised here via
the deprecated-but-present per-device helpers so its correctness is locked in even
when we cannot install an actual torch-2.1 environment.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from torchlens.utils import _torch_compat as tc
from torchlens.utils.rng import log_current_autocast_state

pytestmark = pytest.mark.smoke


def test_modern_branch_matches_torch_directly() -> None:
    """On torch>=2.4 the shim must equal torch's own per-device query exactly."""
    if not tc.AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED:
        pytest.skip("legacy torch (<2.4): modern branch not exercised here")
    for dev in ("cpu", "cuda"):
        assert tc.autocast_is_enabled(dev) == torch.is_autocast_enabled(dev)
        assert tc.autocast_get_dtype(dev) == torch.get_autocast_dtype(dev)


def test_modern_branch_observes_active_autocast() -> None:
    if not tc.AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED:
        pytest.skip("legacy torch (<2.4)")
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        assert tc.autocast_is_enabled("cpu") is True
        assert tc.autocast_get_dtype("cpu") == torch.bfloat16


def test_legacy_branch_logic_matches_modern() -> None:
    """Lock in the torch 2.1-2.3 fallback path correctness.

    The legacy device-specific helpers still exist (deprecated) on modern torch,
    so we can verify the fallback maps device_type -> the right helper and yields
    results identical to the modern per-device query.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for dev in ("cpu", "cuda"):
            legacy_enabled = tc._legacy_is_autocast_enabled(dev)
            legacy_dtype = tc._legacy_get_autocast_dtype(dev)
            # Compare against the modern reference where available.
            if tc.AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED:
                assert legacy_enabled == torch.is_autocast_enabled(dev)
                assert legacy_dtype == torch.get_autocast_dtype(dev)
            assert isinstance(legacy_enabled, bool)
            assert isinstance(legacy_dtype, torch.dtype)


def test_legacy_branch_observes_active_autocast() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            assert tc._legacy_is_autocast_enabled("cpu") is True
            assert tc._legacy_get_autocast_dtype("cpu") == torch.bfloat16


def test_legacy_branch_unsupported_device_raises_runtimeerror() -> None:
    """Unsupported device_type must raise RuntimeError (swallowed by rng.py)."""
    with pytest.raises(RuntimeError):
        tc._legacy_is_autocast_enabled("mps")
    with pytest.raises(RuntimeError):
        tc._legacy_get_autocast_dtype("xpu")


def test_log_current_autocast_state_unaffected() -> None:
    """The public capture path still returns the expected device-keyed dict."""
    state = log_current_autocast_state()
    assert set(state) <= {"cpu", "cuda"}
    for dev_state in state.values():
        assert set(dev_state) == {"enabled", "dtype"}
        assert isinstance(dev_state["enabled"], bool)
        assert isinstance(dev_state["dtype"], torch.dtype)
