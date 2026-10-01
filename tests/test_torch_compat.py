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
from types import SimpleNamespace
from typing import cast

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


class _UnreadableNamesSentinel:
    """Sentinel that fails if the compatibility helper reads ``names``."""

    @property
    def names(self) -> tuple[str | None, ...]:
        """Raise when the named-dimension metadata is inspected."""

        raise AssertionError("names must not be read when the capability is absent")


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
    if name == "HAS_DYNAMO_ORIG_CALLABLE_MARKER":
        monkeypatch.setattr(tc, "_DYNAMO_ORIG_CALLABLE_MARKER_PROBED", False)
    if name == "HAS_DISPATCH_MODE_STACK_QUERY":
        monkeypatch.setattr(tc, "_DISPATCH_MODE_STACK_FN", None)
        monkeypatch.setattr(tc, "_DISPATCH_MODE_STACK_PROBED", False)
    tc._warned_missing_capabilities.discard(name)


def test_capability_attrs_cover_all_has_flags() -> None:
    """Doctor capability inventory must cover every torch ``HAS_*`` flag exactly once.

    Set equality alone hid three duplicated rows (``HAS_C10D_GROUP_REGISTRY``,
    ``HAS_C10D_GROUP_SEQ``, ``HAS_C10D_ABORT_PG`` were each listed twice), so the
    row count is asserted too: the inventory is a declaration table, and a
    duplicated declaration is dead config.
    """

    has_flags = {name for name in vars(tc) if name.startswith("HAS_")}
    attrs = tc._CAPABILITY_ATTRS  # noqa: SLF001
    assert set(attrs) == has_flags
    duplicates = sorted({name for name in attrs if attrs.count(name) > 1})
    assert not duplicates, f"duplicated capability inventory rows: {duplicates}"


def test_untyped_storage_wrapper_cache_probe_matches_runtime() -> None:
    """Storage weak-key safety capability follows live wrapper identity behavior."""

    tensor = torch.empty(0)
    first = tensor.untyped_storage()
    second = tensor.untyped_storage()

    assert tc.HAS_CACHED_UNTYPED_STORAGE_WRAPPER is (first is second)


def test_variable_functions_absence_falls_back_to_torch_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ``torch._C._VariableFunctions`` degrades to an EMPTY roster.

    r-b7 R42-2: the historical fallback returned ``torch.__all__`` — the wrong
    namespace for a variable-function roster; had it ever been taken it would
    have silently mis-seeded the wrapper inventory. An empty list skips
    VF-based discovery honestly, with the flipped flag keeping it visible.
    """

    _reset_capability(monkeypatch, "HAS_VARIABLE_FUNCTIONS")
    monkeypatch.setattr(tc, "_nested_getattr_or_none", lambda _root, _path: None)
    with pytest.warns(UserWarning, match="HAS_VARIABLE_FUNCTIONS"):
        names = tc.get_variable_function_names()
    assert names == []
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
        ("get_jit_boolean_dispatch_table", "HAS_JIT_BOOLEAN_DISPATCH_TABLE", None),
        ("get_jit_overload_resolver_module", "HAS_JIT_OVERLOAD_RESOLVER", None),
        ("get_device_context_type", "HAS_DEVICE_CONTEXT_DISPATCH", None),
        ("get_current_function_mode_stack", "HAS_DEVICE_CONTEXT_DISPATCH", None),
        ("get_torch_function_mode_stack_length", "HAS_DEVICE_CONTEXT_DISPATCH", None),
        ("get_device_constructors", "HAS_DEVICE_CONSTRUCTORS", None),
        ("get_dynamo_optimized_module_type", "HAS_DYNAMO_OPTIMIZED_MODULE", None),
        ("get_current_dispatch_mode_stack", "HAS_DISPATCH_MODE_STACK_QUERY", None),
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
    if helper_name == "get_dynamo_optimized_module_type":
        # This helper DEFERS (returns None without degrading, by design) when
        # ``torch._dynamo.eval_frame`` is not yet imported -- so the degrade path
        # under test (module present, ``OptimizedModule`` attr absent, monkeypatched
        # below) is only reached once the module is in ``sys.modules``. Some torch
        # builds do not auto-import it, so place a stub to make the degrade contract
        # deterministic across the supported range.
        monkeypatch.setitem(tc.sys.modules, "torch._dynamo.eval_frame", SimpleNamespace())
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
        "HAS_JIT_BOOLEAN_DISPATCH_TABLE",
        "HAS_JIT_OVERLOAD_RESOLVER",
        "HAS_DEVICE_CONTEXT_DISPATCH",
        "HAS_DEVICE_CONSTRUCTORS",
        "HAS_ACCUMULATE_GRAD_CLASS",
        "HAS_FX_GRAPH_MODULE",
        "HAS_DYNAMO_OPTIMIZED_MODULE",
        "HAS_DYNAMO_ORIG_CALLABLE_MARKER",
        "HAS_TENSOR_SEQUENCE_SLOT_FIX",
        "HAS_DISPATCH_MODE_STACK_QUERY",
    }
    # NOT floor-required, verified rather than assumed: HAS_NAMED_TENSOR_API is
    # False on torch 2.13 (the named-tensor surface was REMOVED upstream), so its
    # degraded branch is live on NEW torch, not dead at the old floor.

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
        "HAS_VARIABLE_FUNCTIONS": True,
        "HAS_TORCH_VF": True,
        "HAS_TORCH_FUNC": True,
        "HAS_FUNCTORCH_APIS": True,
        "HAS_FUNCTORCH_LEVEL_API": _HAS_FUNCTORCH_LEVEL_API,
        "HAS_FUNCTORCH_WRAPPED_TENSOR_API": True,
        "HAS_JIT_BUILTIN_TABLE": True,
        # fix-capture-r3 R56: TorchScript compatibility while wrapped needs the
        # boolean-dispatch table (F.max_pool* family) and the overload resolver
        # (F.interpolate family). Present across the supported range --
        # hardcoded True as tripwires.
        "HAS_JIT_BOOLEAN_DISPATCH_TABLE": True,
        "HAS_JIT_OVERLOAD_RESOLVER": True,
        "HAS_DEVICE_CONTEXT_DISPATCH": True,
        "HAS_DEVICE_CONSTRUCTORS": True,
        "HAS_ACCUMULATE_GRAD_CLASS": True,
        # fix-bwgrad F6: the autograd graph-task-id resolver distinguishes
        # separate engine invocations for implicit backward pass boundaries.
        # torch._C._current_graph_task_id is present across the whole
        # supported range -- hardcoded True as a tripwire; a build without it
        # degrades to the open-bracket heuristic behind this named flag.
        "HAS_CURRENT_GRAPH_TASK_ID": True,
        "HAS_FX_GRAPH_MODULE": True,
        "HAS_NAMED_TENSOR_API": tc.HAS_NAMED_TENSOR_API,
        "HAS_CACHED_UNTYPED_STORAGE_WRAPPER": tc.HAS_CACHED_UNTYPED_STORAGE_WRAPPER,
        "HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE": (
            tc.HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE
        ),
        "HAS_DYNAMO_OPTIMIZED_MODULE": True,
        "HAS_DYNAMO_ORIG_CALLABLE_MARKER": tc.HAS_DYNAMO_ORIG_CALLABLE_MARKER,
        "HAS_DYNAMO_EXPLAIN": tc.HAS_DYNAMO_EXPLAIN,
        # F04 one-backward reads: GradientEdge addressing and Node prehooks
        # postdate the torch>=2.1 floor (OPTIONAL flags -- absence is a
        # healthy old install; the read refuses typed). Mirror the live
        # values so floor legs stay green.
        "HAS_GRADIENT_EDGE": tc.HAS_GRADIENT_EDGE,
        "HAS_NODE_PREHOOK": tc.HAS_NODE_PREHOOK,
        # L8 floor fixes: deterministic uninit-memory fill, device-agnostic
        # GradScaler, torch.nn.attention, torch.nn.RMSNorm, and tuple-dim
        # any()/all() all postdate the torch>=2.1 floor (OPTIONAL flags --
        # absence is a healthy old install with a real fallback or nothing to
        # shim). Build-dependent, so mirror the live post-snapshot capability.
        "HAS_DETERMINISTIC_FILL_FLAG": tc.HAS_DETERMINISTIC_FILL_FLAG,
        "HAS_AMP_GRADSCALER": tc.HAS_AMP_GRADSCALER,
        "HAS_NN_ATTENTION_MODULE": tc.HAS_NN_ATTENTION_MODULE,
        "HAS_RMSNORM_MODULE": tc.HAS_RMSNORM_MODULE,
        "HAS_REDUCE_TUPLE_DIM": tc.HAS_REDUCE_TUPLE_DIM,
        # CPU Half-dtype kernel coverage (addmm/layer_norm/nextafter) and
        # Float8 empty-fill under deterministic mode both postdate the torch
        # 2.1 floor: genuine old-torch CPU limitations. Build-dependent, so
        # mirror the live post-snapshot capability.
        "HAS_CPU_HALF_KERNELS": tc.HAS_CPU_HALF_KERNELS,
        "HAS_CPU_FLOAT8_DETERMINISTIC_FILL": tc.HAS_CPU_FLOAT8_DETERMINISTIC_FILL,
        # W21 cold-start: FSDP wrapper detection is lazily probed (never imports
        # torch.distributed.fsdp on plain captures); distributed availability is
        # build-dependent, so mirror the live post-snapshot capability.
        "HAS_FSDP_WRAPPER": tc.HAS_FSDP_WRAPPER,
        # Honest-distributed-detection probes, lazily resolved on the same
        # never-import-on-the-hot-path contract as FSDP above. Each one names the
        # graceful degradation it gates: without it, DTensor / device-mesh /
        # pipeline-stage detection falls back from an exact isinstance check to
        # structural namespace matching. Build-dependent, so mirror the live
        # post-snapshot capability.
        "HAS_DTENSOR": tc.HAS_DTENSOR,
        # F27: lazily probed at first Kineto extraction / memory-profile
        # read; the snapshot reads whatever the probe state is.
        "HAS_KINETO_INMEMORY_EVENTS": tc.HAS_KINETO_INMEMORY_EVENTS,
        "HAS_KINETO_EVENT_SCOPE": tc.HAS_KINETO_EVENT_SCOPE,
        "HAS_MEMORY_PROFILE": tc.HAS_MEMORY_PROFILE,
        "HAS_DEVICE_MESH": tc.HAS_DEVICE_MESH,
        "HAS_PIPELINING": tc.HAS_PIPELINING,
        # Dynamo/fake-mode boundary probes. Without them a compiled region reached
        # during capture dies inside the wrappers with a raw
        # InternalTorchDynamoError, and fake/functional tensors fall back to
        # structural name matching. Build-dependent, so mirror the live values.
        "HAS_DYNAMO_IS_COMPILING": tc.HAS_DYNAMO_IS_COMPILING,
        # r-b4 R26-1: the host-escape belt census check depends on this private
        # dispatch-stack query and FAILS CLOSED without it. Hardcoded True as a
        # tripwire: a torch build that loses the probe must fail this test
        # loudly, not degrade silently.
        "HAS_DISPATCH_MODE_STACK_QUERY": True,
        # r-b4 R26-2: previously-unrouted private probes, now behind named
        # flags. The four torch._C surfaces are present across the whole
        # supported range -- hardcoded True as tripwires. FakeTensorMode and
        # the DTensor geometry helper are build-dependent (distributed may be
        # absent), so mirror the live post-snapshot capability.
        "HAS_JIT_SCHEMA_ENUMERATION": True,
        "HAS_TENSORBASE_CLASS": True,
        "HAS_DISABLE_TORCH_FUNCTION": True,
        "HAS_VARIABLE_FUNCTIONS_CLASS": True,
        "HAS_FAKE_TENSOR_MODE": tc.HAS_FAKE_TENSOR_MODE,
        "HAS_DTENSOR_SHARD_GEOMETRY": tc.HAS_DTENSOR_SHARD_GEOMETRY,
        # L8/C2 funcol boundary capture degrades independently when group
        # resolution or dispatcher-level wait interposition is unavailable.
        # Both named flags are included in every diagnostic snapshot while
        # retaining their lazy-probe lifecycle.
        "HAS_FUNCOL_GROUP_RESOLUTION": tc.HAS_FUNCOL_GROUP_RESOLUTION,
        "HAS_FUNCOL_WAIT_INTERPOSITION": tc.HAS_FUNCOL_WAIT_INTERPOSITION,
        # fix/private-probe-routing: the remaining L8/C2 funcol private touches
        # (module object for the arm-time wraps, ACT class for inner-tensor
        # unwrapping) resolve through compat. Distributed availability is
        # build-dependent, so mirror the live post-snapshot capability.
        "HAS_FUNCOL_MODULE": tc.HAS_FUNCOL_MODULE,
        "HAS_ASYNC_COLLECTIVE_TENSOR": tc.HAS_ASYNC_COLLECTIVE_TENSOR,
        # L9 backward-residual private touches, compat-routed. The autograd
        # engine callback handle exists across the whole supported range --
        # hardcoded True as a tripwire; the non-reentrant _checkpoint_hook
        # class is version-dependent, so mirror the live capability.
        "HAS_CHECKPOINT_HOOK_CLASS": tc.HAS_CHECKPOINT_HOOK_CLASS,
        # torch 2.14 interposes _checkpoint_internal_hook between
        # _checkpoint_hook/_recomputation_hook and saved_tensors_hooks;
        # version-dependent, so mirror the live capability.
        "HAS_CHECKPOINT_INTERNAL_HOOK_CLASS": tc.HAS_CHECKPOINT_INTERNAL_HOOK_CLASS,
        "HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK": True,
        "HAS_TRACING_TENSOR_TYPES": tc.HAS_TRACING_TENSOR_TYPES,
        # Compile rung-2 probes: set_stance (torch >= 2.6) lets capture run
        # compiled callables through their original eager Python, and Dynamo's
        # compile counters back tl.debug.count_compiles. Build-dependent, so
        # mirror the live values.
        "HAS_SET_STANCE": tc.HAS_SET_STANCE,
        "HAS_TORCH_FUNCTION_STACK_SURGERY": tc.HAS_TORCH_FUNCTION_STACK_SURGERY,
        "HAS_DYNAMO_COMPILE_COUNTERS": tc.HAS_DYNAMO_COMPILE_COUNTERS,
        # fp8 dtypes exist on every torch build we support, but the set grew across
        # 2.x, so mirror the live value rather than hardcoding True.
        "HAS_FP8_DTYPES": tc.HAS_FP8_DTYPES,
        # c10d lifecycle/correlation surfaces read by collective boundary
        # capture (merge-ranks tier (b)/C0).
        "HAS_C10D_GROUP_REGISTRY": tc.HAS_C10D_GROUP_REGISTRY,
        "HAS_C10D_GROUP_SEQ": tc.HAS_C10D_GROUP_SEQ,
        "HAS_C10D_ABORT_PG": tc.HAS_C10D_ABORT_PG,
        "HAS_GENERATOR_CLONE_STATE": hasattr(torch.Generator, "clone_state"),
        "HAS_GENERATOR_GRAPHSAFE_GET_STATE": hasattr(torch.Generator, "graphsafe_get_state"),
        "HAS_GENERATOR_GRAPHSAFE_SET_STATE": hasattr(torch.Generator, "graphsafe_set_state"),
        "HAS_GENERATOR_PHILOX_STATE": hasattr(torch.Generator, "philox_state"),
        # CVE-2025-32434 fix presence (feature-detected; version-dependent, so mirror
        # the live capability like AUTOCAST rather than hardcoding a boolean).
        "HAS_SAFE_WEIGHTS_ONLY_LOAD": tc.HAS_SAFE_WEIGHTS_ONLY_LOAD,
        # r33 F-2: whether autograd default saved-tensors hooks can be peeked
        # (needed to know grad_fn saved-value reads are side-effect-free);
        # feature-detected, so mirror the live capability like AUTOCAST.
        "HAS_SAVED_TENSORS_HOOK_INTROSPECTION": tc.HAS_SAVED_TENSORS_HOOK_INTROSPECTION,
        # r32-abc Fix A: whether saved_tensors_hooks.__init__ can be patched to
        # scope user pack/unpack hook bodies as autograd-internal during capture;
        # feature-detected class shape, so mirror the live capability.
        "HAS_SAVED_TENSORS_HOOKS_PATCHABLE": tc.HAS_SAVED_TENSORS_HOOKS_PATCHABLE,
        # r32-abc Fix C: PEP 657 per-instruction columns and co_qualname gate
        # same-line ternary arm attribution and qualname scope resolution;
        # runtime capabilities of the interpreter, so mirror the live values.
        "HAS_CODE_POSITIONS": tc.HAS_CODE_POSITIONS,
        "HAS_CODE_QUALNAME": tc.HAS_CODE_QUALNAME,
        # Identity-shim sites (sf-fastpath): torch-internal `x is F.y` checks
        # normalized while wrappers are installed. The transformer flag and the
        # private expanded-weights machinery exist across the supported range
        # (hardcoded True); CausalBias postdates the floor, so mirror the live
        # value (its absence is OPTIONAL, a healthy old-torch install).
        "HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG": True,
        "HAS_ATTENTION_CAUSAL_BIAS": tc.HAS_ATTENTION_CAUSAL_BIAS,
        "HAS_EXPANDED_WEIGHTS_CONV_PICKER": True,
        "HAS_TENSOR_SEQUENCE_SLOT_FIX": True,
        # r-b7 R42-1/6: the six ambient execution-context HAS_* flags and the
        # HAS_AUTOCAST_DEVICE_TYPE_ARG alias are RETIRED — every control they
        # guarded is a public torch surface older than the 2.1 floor, so the
        # snapshot reads them directly and only the canonical autocast
        # spelling remains published.
        "AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED": tc.AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED,
    }

    assert snapshot == expected


def test_tensor_has_named_dims_reports_native_tensor_names() -> None:
    """Named-dimension inspection reports ordinary and named tensors accurately."""

    assert tc.tensor_has_named_dims(torch.ones(2, 3)) is False
    if not tc.HAS_NAMED_TENSOR_API:
        pytest.skip("native named-tensor API is unavailable")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        named = torch.ones(2, 3).refine_names("batch", "feature")
    assert tc.tensor_has_named_dims(named) is True


def test_tensor_has_named_dims_short_circuits_when_api_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent named-tensor support returns false without reading the input."""

    monkeypatch.setattr(tc, "HAS_NAMED_TENSOR_API", False)
    sentinel = cast(torch.Tensor, _UnreadableNamesSentinel())
    assert tc.tensor_has_named_dims(sentinel) is False


def test_completeness_census_check_fails_closed_on_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed dispatch-stack probe reads as census-INACTIVE (belt records).

    r-b4 R26-1: the historical inline ``except Exception: return True`` failed
    OPEN -- a private-API rename silently disarmed the host-escape belt inside
    the census-blind ``_disable_current_modes()`` regions it exists to cover.
    """

    from torchlens.backends.torch import completeness_witness as cw

    monkeypatch.setattr(tc, "get_current_dispatch_mode_stack", lambda: None)
    assert cw._completeness_census_active() is False


def test_completeness_census_check_reads_live_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The census check still discriminates active vs inactive census modes."""

    from torchlens.backends.torch import completeness_witness as cw

    census_mode = cw._CompletenessDispatchMode.__new__(cw._CompletenessDispatchMode)
    monkeypatch.setattr(tc, "get_current_dispatch_mode_stack", lambda: [census_mode])
    assert cw._completeness_census_active() is True
    monkeypatch.setattr(tc, "get_current_dispatch_mode_stack", lambda: [])
    assert cw._completeness_census_active() is False


def test_dispatch_mode_stack_probe_demotes_on_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe that raises is demoted to permanently absent, not fail-open."""

    def _raising_probe() -> list[object]:
        raise RuntimeError("private dispatch-stack API drifted")

    _reset_capability(monkeypatch, "HAS_DISPATCH_MODE_STACK_QUERY")
    monkeypatch.setattr(tc, "_DISPATCH_MODE_STACK_FN", _raising_probe)
    monkeypatch.setattr(tc, "_DISPATCH_MODE_STACK_PROBED", True)
    with pytest.warns(UserWarning, match="HAS_DISPATCH_MODE_STACK_QUERY"):
        assert tc.get_current_dispatch_mode_stack() is None
    assert tc.HAS_DISPATCH_MODE_STACK_QUERY is False
    assert tc._DISPATCH_MODE_STACK_FN is None


def test_dispatch_mode_stack_probe_resolves_on_supported_torch() -> None:
    """The dispatch-stack query resolves and returns a list on supported torch."""

    stack = tc.get_current_dispatch_mode_stack()
    assert isinstance(stack, list)
    assert tc.HAS_DISPATCH_MODE_STACK_QUERY is True


def test_dynamo_is_compiling_raising_probe_discloses_possibly_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising is_compiling probe flips the flag and reads possibly-compiling.

    r-b4 R26-3: the raising path used to return False WITHOUT flipping
    HAS_DYNAMO_IS_COMPILING -- the wrapper then logged data-free FakeTensors
    (the documented crash class) instead of taking the disclosed bypass.
    """

    def _raising_probe() -> bool:
        raise RuntimeError("dynamo probe drifted")

    _reset_capability(monkeypatch, "HAS_DYNAMO_IS_COMPILING")
    monkeypatch.setattr(tc, "_DYNAMO_IS_COMPILING_FN", _raising_probe)
    monkeypatch.setattr(tc, "_DYNAMO_IS_COMPILING_PROBED", True)
    with pytest.warns(UserWarning, match="HAS_DYNAMO_IS_COMPILING"):
        assert tc.dynamo_is_compiling() is True
    assert tc.HAS_DYNAMO_IS_COMPILING is False


def test_capability_warning_category_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """r-b4 R26-6d: degradation warnings carry TorchCapabilityWarning.

    CI suppression can key on the CATEGORY; the subclass still isinstance-
    matches UserWarning so existing filters keep working.
    """

    _reset_capability(monkeypatch, "HAS_TORCH_VF")
    monkeypatch.setattr(tc, "HAS_TORCH_VF", True)
    monkeypatch.setattr(torch, "_VF", None, raising=False)
    with pytest.warns(tc.TorchCapabilityWarning, match="HAS_TORCH_VF"):
        assert tc.get_torch_vf_namespace() is None
    assert issubclass(tc.TorchCapabilityWarning, UserWarning)


def test_tf_runtime_support_is_feature_probed() -> None:
    """r-b4 R26-6b: an odd version string no longer disables backend='tf'."""

    from torchlens.backends.default_specs import _tf_runtime_supported

    class _Backend:
        @staticmethod
        def backend() -> str:
            return "tensorflow"

    keras3 = SimpleNamespace(ops=object(), backend=_Backend, __version__="weird+build")
    odd_tf = SimpleNamespace(__version__="2.16.custom.oddity")
    assert _tf_runtime_supported(odd_tf, keras3) is True

    keras2 = SimpleNamespace(__version__="2.15.0")
    old_tf = SimpleNamespace(__version__="2.12.0")
    assert _tf_runtime_supported(old_tf, keras2) is False

    unparseable_keras2 = SimpleNamespace(__version__="not-a-version")
    assert _tf_runtime_supported(old_tf, unparseable_keras2) is False


class _RecordingStance:
    """Stance handle recording enter/exit calls."""

    def __init__(self) -> None:
        self.enters = 0
        self.exits = 0

    def __enter__(self) -> _RecordingStance:
        self.enters += 1
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.exits += 1


def test_force_eager_stance_scope_exit_owned_with_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once ``set_stance`` returns, ``__exit__`` runs on every exit path (R07).

    The stance is APPLIED by construction, so exit ownership must bind in the
    same guarded region -- a BaseException delivered after the handle exists
    (here: from the with-body) must still reach ``__exit__`` exactly once.
    """

    import torch._dynamo  # noqa: F401 - the scope requires Dynamo in sys.modules

    handle = _RecordingStance()
    monkeypatch.setattr(tc, "HAS_SET_STANCE", True)
    monkeypatch.setattr(torch.compiler, "set_stance", lambda mode: handle)

    class _Interrupt(KeyboardInterrupt):
        pass

    with pytest.raises(_Interrupt):
        with tc.force_eager_stance_scope() as active:
            assert active is True
            raise _Interrupt("body interrupted")

    assert handle.exits == 1


def test_force_eager_stance_scope_construction_failure_degrades_without_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing ``set_stance`` construction yields ``False`` and calls no exit."""

    import torch._dynamo  # noqa: F401 - the scope requires Dynamo in sys.modules

    monkeypatch.setenv("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS", "1")
    monkeypatch.setattr(tc, "HAS_SET_STANCE", True)

    def _refuse(mode: str) -> object:
        raise ValueError("stance refused")

    monkeypatch.setattr(torch.compiler, "set_stance", _refuse)

    with tc.force_eager_stance_scope() as active:
        assert active is False
    # The capability flag degraded (monkeypatch restores it at teardown).
    assert tc.HAS_SET_STANCE is False


def test_fill_uninitialized_memory_flag_visible_in_capability_snapshot() -> None:
    """``HAS_DETERMINISTIC_FILL_FLAG`` and friends are published, not silent.

    L8 floor fix: a torch build whose ``torch.utils.deterministic`` submodule
    is absent (torch 2.1.x) must still show the flag in the diagnostic
    snapshot (``tl.compat.report()`` / ``tl.utils.doctor()``), mirroring the
    live capability rather than dropping the key.
    """

    snapshot = tc.get_torch_capability_snapshot()
    assert snapshot["HAS_DETERMINISTIC_FILL_FLAG"] == tc.HAS_DETERMINISTIC_FILL_FLAG
    assert snapshot["HAS_AMP_GRADSCALER"] == tc.HAS_AMP_GRADSCALER
    assert snapshot["HAS_NN_ATTENTION_MODULE"] == tc.HAS_NN_ATTENTION_MODULE
    assert snapshot["HAS_RMSNORM_MODULE"] == tc.HAS_RMSNORM_MODULE
    assert snapshot["HAS_REDUCE_TUPLE_DIM"] == tc.HAS_REDUCE_TUPLE_DIM
    assert snapshot["HAS_CPU_HALF_KERNELS"] == tc.HAS_CPU_HALF_KERNELS
    assert snapshot["HAS_CPU_FLOAT8_DETERMINISTIC_FILL"] == tc.HAS_CPU_FLOAT8_DETERMINISTIC_FILL


def test_read_fill_uninitialized_memory_returns_none_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read degrades to ``None`` (never crashes) when the submodule is absent."""

    monkeypatch.setattr(tc, "_torch_deterministic_module", None)
    assert tc.read_fill_uninitialized_memory() is None


def test_write_fill_uninitialized_memory_noops_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The write is a no-op (never crashes) when the submodule is absent."""

    monkeypatch.setattr(tc, "_torch_deterministic_module", None)
    tc.write_fill_uninitialized_memory(True)  # must not raise


def test_read_fill_uninitialized_memory_reads_live_module_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read/write pair round-trips through a stubbed submodule when present."""

    stub = SimpleNamespace(fill_uninitialized_memory=True)
    monkeypatch.setattr(tc, "_torch_deterministic_module", stub)
    assert tc.read_fill_uninitialized_memory() is True
    tc.write_fill_uninitialized_memory(False)
    assert stub.fill_uninitialized_memory is False


def test_tensor_any_over_dims_matches_native_any_on_every_axis_combo() -> None:
    """The multi-axis ``any`` helper matches ``tensor.any(dim=<int>)`` chaining."""

    mask = torch.zeros((2, 3, 4), dtype=torch.bool)
    mask[1, 2, 3] = True
    for dims in ((0,), (1, 2), (0, 2), (0, 1, 2), ()):
        expected = mask
        for axis in sorted(dims, reverse=True):
            expected = expected.any(dim=axis)
        assert torch.equal(tc.tensor_any_over_dims(mask, dims), expected)


def test_tensor_any_over_dims_fallback_matches_native_tuple_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The torch-2.1 fallback path is byte-identical to the native tuple-dim call."""

    if not tc.HAS_REDUCE_TUPLE_DIM:
        pytest.skip("native tuple-dim any() unavailable; nothing to compare against")
    mask = torch.rand((3, 4, 5)) > 0.5
    monkeypatch.setattr(tc, "HAS_REDUCE_TUPLE_DIM", False)
    fallback = tc.tensor_any_over_dims(mask, (0, 2))
    monkeypatch.setattr(tc, "HAS_REDUCE_TUPLE_DIM", True)
    native = tc.tensor_any_over_dims(mask, (0, 2))
    assert torch.equal(fallback, native)


def test_probe_gradient_edge_is_functional_not_just_attribute_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe calls real autograd.grad, not just hasattr (torch 2.2-2.3 trap).

    L8 floor fix: torch 2.2-2.3 ships ``torch.autograd.graph.GradientEdge``
    but its OWN ``_make_grads`` internal crashes with
    ``AttributeError: 'GradientEdge' object has no attribute 'is_nested'``
    when a GradientEdge is used as an output with an explicit cotangent --
    exactly what the one-backward read engine does. The probe must catch
    this and return False on such a build.
    """

    def _broken_grad(*args: object, **kwargs: object) -> None:
        raise AttributeError("'GradientEdge' object has no attribute 'is_nested'")

    monkeypatch.setattr(torch.autograd, "grad", _broken_grad)
    assert tc._probe_gradient_edge() is False


def test_probe_gradient_edge_true_when_autograd_grad_succeeds() -> None:
    """On a healthy torch build the functional probe reports True."""

    if not tc.HAS_GRADIENT_EDGE:
        pytest.skip("this torch build genuinely lacks working GradientEdge support")
    assert tc._probe_gradient_edge() is True
