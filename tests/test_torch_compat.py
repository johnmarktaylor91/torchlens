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

from types import SimpleNamespace
import warnings

import pytest
import torch

from torchlens.utils import _torch_compat as tc
from torchlens.utils.rng import log_current_autocast_state

# The functorch transform-level API (torch._C._functorch.maybe_current_level) is a
# torch-2.4+ addition, genuinely absent on the supported 2.1-2.3 range. Probe torch
# directly (independent of torchlens' own capability flag) so the capability expectations
# below stay correct across the whole supported torch matrix.
_HAS_FUNCTORCH_LEVEL_API = (
    getattr(getattr(getattr(torch, "_C", None), "_functorch", None), "maybe_current_level", None)
    is not None
)

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
    """The public capture path still returns the expected device-keyed dict.

    r35: the reserved ``__execution__`` grad/inference entry rides along with
    ``enabled=False`` so every autocast consumer skips it structurally.
    """

    state = log_current_autocast_state()
    assert set(state) <= {"cpu", "cuda", "__execution__"}
    execution = state["__execution__"]
    assert execution["enabled"] is False
    assert isinstance(execution["grad_enabled"], bool)
    assert isinstance(execution["inference_mode"], bool)
    for device, dev_state in state.items():
        if device.startswith("__"):
            continue
        assert set(dev_state) == {"enabled", "dtype"}
        assert isinstance(dev_state["enabled"], bool)
        assert isinstance(dev_state["dtype"], torch.dtype)


def _reset_capability(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Reset a capability flag and warning state for one degraded-path test.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.
    name:
        Capability flag to reset.

    Returns
    -------
    None
        Module state is reset for the current test.
    """

    monkeypatch.setattr(tc, name, True)
    if name == "HAS_DYNAMO_OPTIMIZED_MODULE":
        monkeypatch.setattr(tc, "_DYNAMO_OPTIMIZED_MODULE_TYPE", None)
        monkeypatch.setattr(tc, "_DYNAMO_OPTIMIZED_MODULE_PROBED", False)
    tc._warned_missing_capabilities.discard(name)


def test_capability_attrs_cover_all_has_flags() -> None:
    """Doctor capability inventory must cover every torch ``HAS_*`` flag."""

    has_flags = {name for name in vars(tc) if name.startswith("HAS_")}
    assert set(tc._CAPABILITY_ATTRS) == has_flags  # noqa: SLF001


def test_variable_functions_absence_falls_back_to_torch_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ``torch._C._VariableFunctions`` falls back to public exports."""

    _reset_capability(monkeypatch, "HAS_VARIABLE_FUNCTIONS")
    monkeypatch.setattr(tc, "_nested_getattr_or_none", lambda _root, _path: None)
    with pytest.warns(UserWarning, match="HAS_VARIABLE_FUNCTIONS"):
        names = tc.get_variable_function_names()
    assert names == list(getattr(torch, "__all__", ()))
    assert tc.HAS_VARIABLE_FUNCTIONS is False


def test_torch_vf_absence_skips_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``torch._VF`` returns ``None`` and marks decoration degraded."""

    _reset_capability(monkeypatch, "HAS_TORCH_VF")
    monkeypatch.delattr(torch, "_VF", raising=False)
    with pytest.warns(UserWarning, match="HAS_TORCH_VF"):
        assert tc.get_torch_vf_namespace() is None
    assert tc.HAS_TORCH_VF is False


@pytest.mark.parametrize(
    ("namespace_name", "flag_name"),
    [
        ("torch.func", "HAS_TORCH_FUNC"),
        ("torch._functorch.apis", "HAS_FUNCTORCH_APIS"),
    ],
)
def test_optional_torch_namespace_absence_marks_capability(
    monkeypatch: pytest.MonkeyPatch,
    namespace_name: str,
    flag_name: str,
) -> None:
    """Missing optional torch namespaces are skipped without raising."""

    _reset_capability(monkeypatch, flag_name)
    monkeypatch.setattr(tc, "_nested_getattr_or_none", lambda _root, _path: None)
    with pytest.warns(UserWarning, match=flag_name):
        assert tc.get_optional_torch_namespace(namespace_name) is None
    assert getattr(tc, flag_name) is False


def test_accumulate_grad_absence_uses_name_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing private AccumulateGrad class returns an empty isinstance target."""

    _reset_capability(monkeypatch, "HAS_ACCUMULATE_GRAD_CLASS")
    monkeypatch.setattr(tc, "_nested_getattr_or_none", lambda _root, _path: None)
    with pytest.warns(UserWarning, match="HAS_ACCUMULATE_GRAD_CLASS"):
        assert tc.get_accumulate_grad_class() == ()
    assert tc.HAS_ACCUMULATE_GRAD_CLASS is False


@pytest.mark.parametrize(
    ("helper_name", "flag_name", "expected"),
    [
        ("get_functorch_maybe_current_level", "HAS_FUNCTORCH_LEVEL_API", None),
        ("get_functorch_wrapped_tensor_checker", "HAS_FUNCTORCH_WRAPPED_TENSOR_API", None),
        ("get_fx_graph_module_type", "HAS_FX_GRAPH_MODULE", None),
    ],
)
def test_nested_private_helper_absence_marks_capability(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    flag_name: str,
    expected: object,
) -> None:
    """Nested private torch helpers degrade to their documented fallbacks."""

    _reset_capability(monkeypatch, flag_name)
    monkeypatch.setattr(tc, "_nested_getattr_or_none", lambda _root, _path: None)
    with pytest.warns(UserWarning, match=flag_name):
        assert getattr(tc, helper_name)() is expected
    assert getattr(tc, flag_name) is False


@pytest.mark.parametrize(
    ("helper_name", "flag_name", "expected"),
    [
        ("get_jit_builtin_table", "HAS_JIT_BUILTIN_TABLE", None),
        ("get_device_context_type", "HAS_DEVICE_CONTEXT_DISPATCH", None),
        ("get_current_function_mode_stack", "HAS_DEVICE_CONTEXT_DISPATCH", None),
        ("get_torch_function_mode_stack_length", "HAS_DEVICE_CONTEXT_DISPATCH", None),
        ("get_device_constructors", "HAS_DEVICE_CONSTRUCTORS", None),
        ("get_dynamo_optimized_module_type", "HAS_DYNAMO_OPTIMIZED_MODULE", None),
    ],
)
def test_imported_private_helper_absence_marks_capability(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    flag_name: str,
    expected: object,
) -> None:
    """Import-based private torch helpers degrade to their documented fallbacks."""

    _reset_capability(monkeypatch, flag_name)
    monkeypatch.setattr(tc, "_import_module_attr_or_none", lambda _module, _attr: None)
    with pytest.warns(UserWarning, match=flag_name):
        assert getattr(tc, helper_name)() is expected
    assert getattr(tc, flag_name) is False


def test_tensor_sequence_slot_fix_absence_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-CPython runtimes skip the tensor sequence-slot fix cleanly."""

    _reset_capability(monkeypatch, "HAS_TENSOR_SEQUENCE_SLOT_FIX")
    monkeypatch.setattr(tc.sys, "implementation", SimpleNamespace(name="pypy"))
    with pytest.warns(UserWarning, match="HAS_TENSOR_SEQUENCE_SLOT_FIX"):
        assert tc.fix_tensor_sequence_slot() is False
    assert tc.HAS_TENSOR_SEQUENCE_SLOT_FIX is False


def test_private_torch_capability_flags_present_on_supported_range() -> None:
    """Private torch integration flags should be present on the supported range."""
    snapshot = tc.get_torch_capability_snapshot()
    required_present = {
        "HAS_VARIABLE_FUNCTIONS",
        "HAS_TORCH_VF",
        "HAS_TORCH_FUNC",
        "HAS_FUNCTORCH_APIS",
        "HAS_FUNCTORCH_LEVEL_API",
        "HAS_FUNCTORCH_WRAPPED_TENSOR_API",
        "HAS_JIT_BUILTIN_TABLE",
        "HAS_DEVICE_CONTEXT_DISPATCH",
        "HAS_DEVICE_CONSTRUCTORS",
        "HAS_ACCUMULATE_GRAD_CLASS",
        "HAS_FX_GRAPH_MODULE",
        "HAS_DYNAMO_OPTIMIZED_MODULE",
        "HAS_TENSOR_SEQUENCE_SLOT_FIX",
    }

    assert required_present <= set(snapshot)
    # HAS_FUNCTORCH_LEVEL_API is torch-2.4+ only; do not require its value on 2.1-2.3.
    version_gated = set() if _HAS_FUNCTORCH_LEVEL_API else {"HAS_FUNCTORCH_LEVEL_API"}
    missing = sorted(name for name in required_present - version_gated if not snapshot[name])
    assert missing == []


def test_vf_and_functorch_guards_return_working_namespaces() -> None:
    """The guarded private namespaces should resolve to usable objects."""
    variable_names = tc.get_variable_function_names()
    torch_vf = tc.get_torch_vf_namespace()
    functorch_apis = tc.get_optional_torch_namespace("torch._functorch.apis")

    assert "add" in variable_names
    assert torch_vf is not None
    assert hasattr(torch_vf, "add")
    assert functorch_apis is not None
    assert hasattr(functorch_apis, "vmap")
    assert hasattr(functorch_apis, "grad")


def test_torch_capability_snapshot_contract() -> None:
    """Capability snapshot keys and values provide a named torch-private API signal."""
    snapshot = tc.get_torch_capability_snapshot()
    expected = {
        "HAS_AUTOCAST_DEVICE_TYPE_ARG": tc.AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED,
        "HAS_VARIABLE_FUNCTIONS": True,
        "HAS_TORCH_VF": True,
        "HAS_TORCH_FUNC": True,
        "HAS_FUNCTORCH_APIS": True,
        "HAS_FUNCTORCH_LEVEL_API": _HAS_FUNCTORCH_LEVEL_API,
        "HAS_FUNCTORCH_WRAPPED_TENSOR_API": True,
        "HAS_JIT_BUILTIN_TABLE": True,
        "HAS_DEVICE_CONTEXT_DISPATCH": True,
        "HAS_DEVICE_CONSTRUCTORS": True,
        "HAS_ACCUMULATE_GRAD_CLASS": True,
        "HAS_FX_GRAPH_MODULE": True,
        "HAS_DYNAMO_OPTIMIZED_MODULE": True,
        # CVE-2025-32434 fix presence (feature-detected; version-dependent, so mirror
        # the live capability like AUTOCAST rather than hardcoding a boolean).
        "HAS_SAFE_WEIGHTS_ONLY_LOAD": tc.HAS_SAFE_WEIGHTS_ONLY_LOAD,
        "HAS_TENSOR_SEQUENCE_SLOT_FIX": True,
        # r35 decision E: ambient execution-context knobs are feature-detected and
        # surfaced in the capability snapshot (values are runtime-dependent).
        "HAS_FLOAT32_MATMUL_PRECISION": tc.HAS_FLOAT32_MATMUL_PRECISION,
        "HAS_DETERMINISTIC_ALGORITHMS_QUERY": tc.HAS_DETERMINISTIC_ALGORITHMS_QUERY,
        "HAS_CUDA_MATMUL_TF32": tc.HAS_CUDA_MATMUL_TF32,
        "HAS_CUDNN_FLAGS": tc.HAS_CUDNN_FLAGS,
        "HAS_SDP_TOGGLES": tc.HAS_SDP_TOGGLES,
        # r53 hon_1: the global uninitialized-memory fill knob is an ambient
        # execution-context knob, feature-detected and surfaced like the others.
        "HAS_FILL_UNINITIALIZED_MEMORY": tc.HAS_FILL_UNINITIALIZED_MEMORY,
        "AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED": tc.AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED,
    }

    assert snapshot == expected
