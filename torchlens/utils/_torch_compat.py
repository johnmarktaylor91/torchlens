"""Backward-compatible shims for torch APIs whose surface changed across versions.

TorchLens targets the *eager* PyTorch surface and only uses a tiny number of
version-sensitive APIs.  This module isolates those so that TorchLens can run on
**torch 2.1+** (instead of requiring torch 2.4+), while behaving *identically* on
modern torch.

The most user-visible incompatibility is autocast state introspection:

* On **torch >= 2.4**, the per-device query helpers take a ``device_type``
  argument::

      torch.is_autocast_enabled("cpu")
      torch.get_autocast_dtype("cuda")

* On **torch 2.1 - 2.3**, those functions take *no* argument (they query the
  CUDA/GPU autocast state only), and per-device queries route through the older
  device-specific helpers::

      torch.is_autocast_cpu_enabled()      # CPU enabled flag
      torch.get_autocast_cpu_dtype()       # CPU autocast dtype
      torch.is_autocast_enabled()          # CUDA/GPU enabled flag
      torch.get_autocast_gpu_dtype()       # CUDA/GPU autocast dtype

The ``device_type`` argument was added in torch 2.4.0 (see
https://github.com/pytorch/pytorch and the widely-hit downstream breakage
https://github.com/huggingface/transformers/issues/43508).  The legacy
device-specific helpers still exist on modern torch (as deprecated aliases), but
we *prefer the modern signature when available* so that we never emit deprecation
warnings on supported torch and so behavior tracks the canonical implementation.

We probe the modern signature **once** at import time (a capability probe, not a
brittle version-string parse) and bind the appropriate implementation.  This keeps
the hot path branch-free and makes the fallback robust to any future torch that
keeps the modern signature but changes its version string.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import ctypes
from dataclasses import dataclass
import importlib
import os
import sys
from typing import Any
import warnings

import torch

from ._torch_symbols import torch_attr

__all__ = [
    "AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED",
    "HAS_ACCUMULATE_GRAD_CLASS",
    "HAS_AUTOCAST_DEVICE_TYPE_ARG",
    "HAS_CUDA_MATMUL_TF32",
    "HAS_CUDNN_FLAGS",
    "HAS_DETERMINISTIC_ALGORITHMS_QUERY",
    "HAS_FILL_UNINITIALIZED_MEMORY",
    "HAS_FLOAT32_MATMUL_PRECISION",
    "HAS_SDP_TOGGLES",
    "apply_ambient_execution_context",
    "read_fill_uninitialized_memory",
    "snapshot_ambient_execution_context",
    "write_fill_uninitialized_memory",
    "HAS_DEVICE_CONTEXT_DISPATCH",
    "HAS_DEVICE_CONSTRUCTORS",
    "HAS_FUNCTORCH_APIS",
    "HAS_FUNCTORCH_LEVEL_API",
    "HAS_FUNCTORCH_WRAPPED_TENSOR_API",
    "HAS_FX_GRAPH_MODULE",
    "HAS_JIT_BUILTIN_TABLE",
    "HAS_NAMED_TENSOR_API",
    "HAS_DYNAMO_OPTIMIZED_MODULE",
    "HAS_SAFE_WEIGHTS_ONLY_LOAD",
    "HAS_TENSOR_SEQUENCE_SLOT_FIX",
    "HAS_TORCH_FUNC",
    "HAS_TORCH_VF",
    "HAS_VARIABLE_FUNCTIONS",
    "TorchCapabilitySnapshot",
    "RunnableTorchAlias",
    "autocast_get_dtype",
    "autocast_is_enabled",
    "get_accumulate_grad_class",
    "get_current_function_mode_stack",
    "get_device_constructors",
    "get_device_context_type",
    "get_dynamo_optimized_module_type",
    "get_functorch_maybe_current_level",
    "get_functorch_wrapped_tensor_checker",
    "get_fx_graph_module_type",
    "get_jit_builtin_table",
    "get_optional_torch_namespace",
    "get_torch_capability_snapshot",
    "get_torch_function_mode_stack_length",
    "get_torch_vf_namespace",
    "get_variable_function_names",
    "fix_tensor_sequence_slot",
    "mark_torch_capability_missing",
    "resolve_runnable_torch_alias",
    "tensor_has_named_dims",
]


TorchCapabilitySnapshot = dict[str, bool]
"""Stable mapping from torch capability name to availability."""

_CAPABILITY_WARNING_ENV = "TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS"
_warned_missing_capabilities: set[str] = set()


@dataclass(frozen=True, slots=True)
class RunnableTorchAlias:
    """One monotonic callable move used by sparse runnable reattachment.

    The table is capability-bounded rather than selected by parsing
    ``torch.__version__``: an entry applies only when its source is absent and
    its target exists in the current runtime. This makes an added entry able to
    turn an unresolved reference into a resolved alias without ever
    reinterpreting an exact current-runtime binding.
    """

    source: str
    target_namespace: str
    target_qualname: str | None
    provenance: str
    recorded_min_version: tuple[int, int]
    recorded_max_version: tuple[int, int]
    strip_target_prefix: str = ""


_RUNNABLE_TORCH_ALIASES: tuple[RunnableTorchAlias, ...] = (
    RunnableTorchAlias(
        "torch._C._nn.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_C._nn.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 13),
    ),
    RunnableTorchAlias(
        "_C._nn.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_C._nn.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 13),
    ),
    RunnableTorchAlias(
        "torch._VF.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_VF.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "_VF.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_VF.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._nn.gelu",
        "torch.nn.functional",
        "gelu",
        "private_to_public:_C._nn.gelu->torch.nn.functional.gelu",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._nn.softplus",
        "torch.nn.functional",
        "softplus",
        "private_to_public:_C._nn.softplus->torch.nn.functional.softplus",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._nn.cross_entropy_loss",
        "torch.nn.functional",
        "cross_entropy",
        "private_to_public:_C._nn.cross_entropy_loss->torch.nn.functional.cross_entropy",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._special.*",
        "torch.special",
        None,
        "private_to_public:_C._special.special_*->torch.special.*",
        (2, 1),
        (2, 12),
        "special_",
    ),
    RunnableTorchAlias(
        "torch._C._fft.*",
        "torch.fft",
        None,
        "private_to_public:_C._fft.fft_*->torch.fft.*",
        (2, 1),
        (2, 12),
        "fft_",
    ),
    RunnableTorchAlias(
        "torch._C._linalg.*",
        "torch.linalg",
        None,
        "private_to_public:_C._linalg.linalg_*->torch.linalg.*",
        (2, 1),
        (2, 13),
        "linalg_",
    ),
    RunnableTorchAlias(
        "torch._C._TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:torch._C._TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C.TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:torch._C.TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch.TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:torch.TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch.Tensor.*",
        "torch.Tensor",
        None,
        "tensor_prefix_drift:torch.Tensor->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "_TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:_TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "Tensor.*",
        "torch.Tensor",
        None,
        "tensor_prefix_drift:Tensor->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._VariableFunctions.*",
        "torch",
        None,
        "private_to_public:torch._C._VariableFunctions->torch",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._VariableFunctionsClass.*",
        "torch",
        None,
        "private_to_public:torch._C._VariableFunctionsClass->torch",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._VariableFunctionsClass.*",
        "torch",
        None,
        "private_to_public:torch._VariableFunctionsClass->torch",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "_VariableFunctionsClass.*",
        "torch",
        None,
        "private_to_public:_VariableFunctionsClass->torch",
        (2, 1),
        (2, 12),
    ),
)


def resolve_runnable_torch_alias(
    source_qualname: str,
    recorded_version: str | None = None,
) -> tuple[str, str, str] | None:
    """Return an explicit current-runtime successor for a recorded torch path.

    Parameters
    ----------
    source_qualname:
        Fully qualified recorded callable path.
    recorded_version:
        Torch version that produced the registry key. Aliases are bounded to
        supported minor releases. Unknown versions retain the compatibility
        behavior for legacy descriptors that predate this metadata.

    Returns
    -------
    tuple[str, str, str] | None
        Target namespace, target qualname, and stable provenance, or ``None``
        when the path has no explicit monotonic alias.
    """

    parsed_version = _torch_minor_version(recorded_version)
    for alias in _RUNNABLE_TORCH_ALIASES:
        if parsed_version is not None and not (
            alias.recorded_min_version <= parsed_version <= alias.recorded_max_version
        ):
            continue
        target_qualname: str | None
        if alias.source.endswith(".*"):
            prefix = alias.source[:-1]
            if not source_qualname.startswith(prefix):
                continue
            target_qualname = source_qualname[len(prefix) :]
        elif alias.source == source_qualname:
            target_qualname = alias.target_qualname
        else:
            continue
        if not target_qualname:
            return None
        if alias.strip_target_prefix and target_qualname.startswith(alias.strip_target_prefix):
            target_qualname = target_qualname[len(alias.strip_target_prefix) :]
        return alias.target_namespace, target_qualname, alias.provenance
    return None


def _torch_minor_version(version: str | None) -> tuple[int, int] | None:
    """Parse a torch major/minor prefix without inspecting the live runtime.

    Parameters
    ----------
    version:
        Producer torch version from runnable compatibility metadata.

    Returns
    -------
    tuple[int, int] | None
        Major/minor pair, or ``None`` when producer metadata is absent or not
        parseable (for example a legacy placeholder).
    """

    if version is None:
        return None
    components = version.split("+", maxsplit=1)[0].split(".")
    if len(components) < 2 or not components[0].isdigit() or not components[1].isdigit():
        return None
    return int(components[0]), int(components[1])


def _probe_device_type_arg_supported() -> bool:
    """Return True if ``torch.is_autocast_enabled`` accepts a ``device_type`` arg.

    True on torch >= 2.4, False on torch 2.1 - 2.3.  We call the modern signature
    inside a try/except rather than parsing ``torch.__version__`` so the result is
    derived from the actual runtime API, not a version string that could be a
    nightly/custom build.
    """
    try:
        # On torch 2.1-2.3 this raises TypeError ("takes no arguments" /
        # "takes 0 positional arguments but 1 was given").  On torch >= 2.4 it
        # returns a bool.
        torch.is_autocast_enabled("cpu")  # type: ignore[call-arg]
        return True
    except TypeError:
        return False


AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED: bool = _probe_device_type_arg_supported()
HAS_AUTOCAST_DEVICE_TYPE_ARG: bool = AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED


def _nested_getattr_or_none(root: Any, path: Iterable[str]) -> Any | None:
    """Return a nested attribute or ``None`` if any segment is absent.

    Parameters
    ----------
    root:
        Object from which attribute traversal starts.
    path:
        Attribute names to traverse in order.

    Returns
    -------
    Any | None
        Resolved object, or ``None`` when a segment is missing.
    """

    current = root
    for attr in path:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _import_module_attr_or_none(module_name: str, attr_name: str) -> Any | None:
    """Import a module attribute or return ``None`` if it is unavailable.

    Parameters
    ----------
    module_name:
        Module to import.
    attr_name:
        Attribute to read from the imported module.

    Returns
    -------
    Any | None
        Imported attribute, or ``None`` when the module or attribute is absent.
    """

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, attr_name, None)


def _probe_variable_functions() -> bool:
    """Return whether torch exposes the private C variable-functions table.

    Returns
    -------
    bool
        True when ``torch._C._VariableFunctions`` is importable.
    """

    return _nested_getattr_or_none(torch, ("_C", "_VariableFunctions")) is not None


def _probe_torch_vf() -> bool:
    """Return whether torch exposes the private ``torch._VF`` namespace.

    Returns
    -------
    bool
        True when ``torch._VF`` is present.
    """

    return getattr(torch, "_VF", None) is not None


def _probe_torch_func() -> bool:
    """Return whether torch exposes the public ``torch.func`` namespace.

    Returns
    -------
    bool
        True when ``torch.func`` is present.
    """

    return getattr(torch, "func", None) is not None


def _probe_functorch_apis() -> bool:
    """Return whether torch exposes the private ``torch._functorch.apis`` namespace.

    Returns
    -------
    bool
        True when ``torch._functorch.apis`` is present.
    """

    return (
        _nested_getattr_or_none(torch, ("_functorch", "apis")) is not None
        or importlib.util.find_spec("torch._functorch.apis") is not None
    )


def _probe_functorch_level_api() -> bool:
    """Return whether torch exposes functorch transform-level introspection.

    Returns
    -------
    bool
        True when ``torch._C._functorch.maybe_current_level`` is present.
    """

    return _nested_getattr_or_none(torch, ("_C", "_functorch", "maybe_current_level")) is not None


def _probe_functorch_wrapped_tensor_api() -> bool:
    """Return whether torch exposes functorch wrapped-tensor detection.

    Returns
    -------
    bool
        True when ``torch._C._functorch.is_functorch_wrapped_tensor`` is present.
    """

    return (
        _nested_getattr_or_none(torch, ("_C", "_functorch", "is_functorch_wrapped_tensor"))
        is not None
    )


def _probe_jit_builtin_table() -> bool:
    """Return whether TorchScript exposes the private builtin table.

    Returns
    -------
    bool
        True when ``torch.jit._builtins._builtin_table`` is present.
    """

    return _import_module_attr_or_none("torch.jit._builtins", "_builtin_table") is not None


def _probe_device_context_dispatch() -> bool:
    """Return whether DeviceContext stack APIs needed for factory injection exist.

    Returns
    -------
    bool
        True when DeviceContext and torch function-mode stack APIs are present.
    """

    return (
        _import_module_attr_or_none("torch.utils._device", "DeviceContext") is not None
        and _import_module_attr_or_none("torch.overrides", "_get_current_function_mode_stack")
        is not None
        and _import_module_attr_or_none("torch.overrides", "_len_torch_function_stack") is not None
    )


def _probe_device_constructors() -> bool:
    """Return whether torch exposes the private factory-constructor inventory.

    Returns
    -------
    bool
        True when ``torch.utils._device._device_constructors`` is present.
    """

    return _import_module_attr_or_none("torch.utils._device", "_device_constructors") is not None


def _probe_accumulate_grad_class() -> bool:
    """Return whether torch exposes the private AccumulateGrad class.

    Returns
    -------
    bool
        True when ``torch._C._functions.AccumulateGrad`` is present.
    """

    return _nested_getattr_or_none(torch, ("_C", "_functions", "AccumulateGrad")) is not None


def _probe_fx_graph_module() -> bool:
    """Return whether torch exposes ``torch.fx.GraphModule``.

    Returns
    -------
    bool
        True when the FX GraphModule type is importable.
    """

    return _nested_getattr_or_none(torch, ("fx", "GraphModule")) is not None


def _probe_named_tensor_api() -> bool:
    """Return whether native Torch named-tensor inspection/construction exists.

    Returns
    -------
    bool
        Whether the required native named-tensor surface is available.
    """

    return all(hasattr(torch.Tensor, attr) for attr in ("names", "has_names", "refine_names"))


def _probe_dynamo_optimized_module() -> bool:
    """Return whether torch exposes the private Dynamo OptimizedModule type.

    Returns
    -------
    bool
        True when ``torch._dynamo.eval_frame.OptimizedModule`` is importable.
    """

    return _import_module_attr_or_none("torch._dynamo.eval_frame", "OptimizedModule") is not None


class _PySequenceMethods(ctypes.Structure):
    """Minimal ctypes mirror of CPython's PySequenceMethods struct."""

    _fields_ = [
        ("sq_length", ctypes.c_void_p),
        ("sq_concat", ctypes.c_void_p),
        ("sq_repeat", ctypes.c_void_p),
        ("sq_item", ctypes.c_void_p),
        ("was_sq_slice", ctypes.c_void_p),
        ("sq_ass_item", ctypes.c_void_p),
        ("was_sq_ass_slice", ctypes.c_void_p),
        ("sq_contains", ctypes.c_void_p),
        ("sq_inplace_concat", ctypes.c_void_p),
        ("sq_inplace_repeat", ctypes.c_void_p),
    ]


class _PyTypeObject(ctypes.Structure):
    """Partial ctypes mirror of CPython's PyTypeObject up to tp_as_sequence."""

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("ob_size", ctypes.c_ssize_t),
        ("tp_name", ctypes.c_char_p),
        ("tp_basicsize", ctypes.c_ssize_t),
        ("tp_itemsize", ctypes.c_ssize_t),
        ("tp_dealloc", ctypes.c_void_p),
        ("tp_vectorcall_offset", ctypes.c_ssize_t),
        ("tp_getattr", ctypes.c_void_p),
        ("tp_setattr", ctypes.c_void_p),
        ("tp_as_async", ctypes.c_void_p),
        ("tp_repr", ctypes.c_void_p),
        ("tp_as_number", ctypes.c_void_p),
        ("tp_as_sequence", ctypes.POINTER(_PySequenceMethods)),
        ("tp_as_mapping", ctypes.c_void_p),
    ]


def _probe_tensor_sequence_slot_fix() -> bool:
    """Return whether the CPython tensor ``sq_item`` slot fix can run.

    Returns
    -------
    bool
        True on CPython when the expected ``torch.Tensor`` type layout is visible.
    """

    if sys.implementation.name != "cpython":
        return False
    try:
        type_obj = _PyTypeObject.from_address(id(torch.Tensor))
    except Exception:
        return False
    return type_obj.tp_name == b"Tensor" and bool(type_obj.tp_as_sequence)


def _probe_safe_weights_only_load() -> bool:
    """Return whether ``torch.load(..., weights_only=True)`` is CVE-2025-32434-safe.

    CVE-2025-32434 (critical, CVSS 9.3; affects torch <= 2.5.1, fixed in torch
    2.6.0) is a remote-code-execution reachable through ``torch.load`` *itself*
    even with ``weights_only=True``: the pre-2.6 weights-only unpickler could be
    bypassed. TorchLens routes an embedded-tensor reconstruction through
    ``torch.load(BytesIO, weights_only=True)`` (see
    ``torchlens._io._safe_unpickle._safe_load_from_bytes``), so on an affected
    runtime that "restricted" nested load is a working RCE and must fail closed.

    The patched behavior is internal to the unpickler and has NO version-string-
    free *behavioral* signature we can cheaply exercise, so -- consistent with this
    module's feature-detect convention (never parse ``torch.__version__`` for a
    behavioral branch) -- we probe the 2.6.0 serialization-hardening SURFACE that
    shipped *in* the fix release: the public
    ``torch.serialization.get_unsafe_globals_in_checkpoint`` inspection API plus the
    rewritten ``torch._weights_only_unpickler.get_globals_in_pkl`` GLOBAL-opcode
    handler. Both are ABSENT on every affected version (<= 2.5.1) and PRESENT on
    2.6.0+ (verified against the v2.5.1 and v2.6.0 sources). Requiring BOTH is the
    conservative, fail-closed direction: a false negative merely refuses an
    embedded load on a safe torch, whereas admitting it on a vulnerable torch would
    reopen the RCE.

    Returns
    -------
    bool
        ``True`` only when the running torch carries the CVE-2025-32434 fix.
    """

    try:
        import torch._weights_only_unpickler as _weights_only_unpickler
        import torch.serialization as _serialization
    except Exception:  # pragma: no cover - defensive; both import on supported torch.
        return False
    return hasattr(_serialization, "get_unsafe_globals_in_checkpoint") and hasattr(
        _weights_only_unpickler, "get_globals_in_pkl"
    )


HAS_VARIABLE_FUNCTIONS: bool = _probe_variable_functions()
HAS_TORCH_VF: bool = _probe_torch_vf()
HAS_TORCH_FUNC: bool = _probe_torch_func()
HAS_FUNCTORCH_APIS: bool = _probe_functorch_apis()
HAS_FUNCTORCH_LEVEL_API: bool = _probe_functorch_level_api()
HAS_FUNCTORCH_WRAPPED_TENSOR_API: bool = _probe_functorch_wrapped_tensor_api()
HAS_JIT_BUILTIN_TABLE: bool = _probe_jit_builtin_table()
HAS_DEVICE_CONTEXT_DISPATCH: bool = _probe_device_context_dispatch()
HAS_DEVICE_CONSTRUCTORS: bool = _probe_device_constructors()
HAS_ACCUMULATE_GRAD_CLASS: bool = _probe_accumulate_grad_class()
HAS_FX_GRAPH_MODULE: bool = _probe_fx_graph_module()
HAS_NAMED_TENSOR_API: bool = _probe_named_tensor_api()
HAS_DYNAMO_OPTIMIZED_MODULE: bool = False
HAS_SAFE_WEIGHTS_ONLY_LOAD: bool = _probe_safe_weights_only_load()
HAS_TENSOR_SEQUENCE_SLOT_FIX: bool = _probe_tensor_sequence_slot_fix()
_DYNAMO_OPTIMIZED_MODULE_TYPE: type[Any] | None = None
_DYNAMO_OPTIMIZED_MODULE_PROBED: bool = False

_CAPABILITY_ATTRS: tuple[str, ...] = (
    "HAS_AUTOCAST_DEVICE_TYPE_ARG",
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
    "HAS_NAMED_TENSOR_API",
    "HAS_DYNAMO_OPTIMIZED_MODULE",
    "HAS_SAFE_WEIGHTS_ONLY_LOAD",
    "HAS_TENSOR_SEQUENCE_SLOT_FIX",
    "HAS_FLOAT32_MATMUL_PRECISION",
    "HAS_DETERMINISTIC_ALGORITHMS_QUERY",
    "HAS_CUDA_MATMUL_TF32",
    "HAS_CUDNN_FLAGS",
    "HAS_SDP_TOGGLES",
    "HAS_FILL_UNINITIALIZED_MEMORY",
)


def mark_torch_capability_missing(capability_name: str, detail: str) -> None:
    """Mark a torch capability absent and emit at most one opt-out warning.

    Parameters
    ----------
    capability_name:
        Name of the module-level ``HAS_*`` flag to flip to ``False``.
    detail:
        Short user-facing detail describing the graceful degradation.

    Returns
    -------
    None
        The matching module-level flag is updated in place.
    """

    if capability_name not in _CAPABILITY_ATTRS:
        raise ValueError(f"unknown torch capability flag: {capability_name}")
    globals()[capability_name] = False
    if os.environ.get(_CAPABILITY_WARNING_ENV):
        return
    if capability_name in _warned_missing_capabilities:
        return
    _warned_missing_capabilities.add(capability_name)
    warnings.warn(
        f"TorchLens torch capability {capability_name} is unavailable; {detail}",
        UserWarning,
        stacklevel=3,
    )


def get_torch_capability_snapshot() -> TorchCapabilitySnapshot:
    """Return the current torch capability flags as a stable snapshot.

    Returns
    -------
    TorchCapabilitySnapshot
        Mapping from ``HAS_*`` flag name, plus the legacy
        ``AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED`` alias, to boolean availability.
    """

    # HAS_DYNAMO_OPTIMIZED_MODULE is lazily probed (see
    # get_dynamo_optimized_module_type) to avoid an unconditional
    # torch._dynamo import on the capture hot path. Diagnostic snapshot
    # consumers (tl.compat.report(), tl.utils.doctor()) are not on that hot
    # path, so force the probe here to report the real capability instead of
    # the pre-probe placeholder.
    get_dynamo_optimized_module_type()
    snapshot = {name: bool(globals()[name]) for name in _CAPABILITY_ATTRS}
    snapshot["AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED"] = bool(AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED)
    return snapshot


def tensor_has_named_dims(value: torch.Tensor) -> bool:
    """Return whether ``value`` has at least one named dimension.

    Parameters
    ----------
    value:
        Native tensor to inspect.

    Returns
    -------
    bool
        Whether at least one dimension has a non-null name.
    """

    if not HAS_NAMED_TENSOR_API:
        return False
    names = getattr(value, "names", None)
    return bool(names and any(name is not None for name in names))


def get_variable_function_names() -> list[str]:
    """Return torch variable-function names with a public fallback.

    Returns
    -------
    list[str]
        Names from ``torch._C._VariableFunctions`` when available, otherwise
        names exported by ``torch.__all__``.
    """

    variable_functions = _nested_getattr_or_none(torch, ("_C", "_VariableFunctions"))
    if variable_functions is None:
        mark_torch_capability_missing(
            "HAS_VARIABLE_FUNCTIONS",
            "falling back to torch.__all__ for function discovery",
        )
        return list(getattr(torch, "__all__", ()))
    return dir(variable_functions)


def get_torch_vf_namespace() -> Any | None:
    """Return ``torch._VF`` when present, otherwise ``None``.

    Returns
    -------
    Any | None
        Private ``torch._VF`` namespace, or ``None`` when unavailable.
    """

    namespace = getattr(torch, "_VF", None)
    if namespace is None:
        mark_torch_capability_missing(
            "HAS_TORCH_VF",
            "skipping torch._VF function decoration",
        )
    return namespace


def get_optional_torch_namespace(namespace_name: str) -> Any | None:
    """Return a torch namespace if it exists, otherwise ``None``.

    Parameters
    ----------
    namespace_name:
        Dotted namespace beginning with ``"torch"``.

    Returns
    -------
    Any | None
        Resolved namespace object, or ``None`` when unavailable.
    """

    if namespace_name == "torch":
        return torch
    prefix = "torch."
    if not namespace_name.startswith(prefix):
        raise ValueError(f"expected a torch namespace, got {namespace_name!r}")
    namespace = _nested_getattr_or_none(torch, namespace_name.removeprefix(prefix).split("."))
    if namespace is None and namespace_name == "torch.func":
        mark_torch_capability_missing(
            "HAS_TORCH_FUNC",
            "skipping torch.func transform-boundary decoration",
        )
    elif namespace is None and namespace_name == "torch._functorch.apis":
        mark_torch_capability_missing(
            "HAS_FUNCTORCH_APIS",
            "skipping torch._functorch.apis transform-boundary decoration",
        )
    return namespace


def get_accumulate_grad_class() -> Any:
    """Return the private AccumulateGrad class or an empty fallback tuple.

    Returns
    -------
    Any
        ``torch._C._functions.AccumulateGrad`` when present, otherwise ``()``.
    """

    accumulate_grad_cls = _nested_getattr_or_none(torch, ("_C", "_functions", "AccumulateGrad"))
    if accumulate_grad_cls is None:
        mark_torch_capability_missing(
            "HAS_ACCUMULATE_GRAD_CLASS",
            "using AccumulateGrad name matching instead of class matching",
        )
        return ()
    return accumulate_grad_cls


def get_functorch_maybe_current_level() -> Callable[[], Any] | None:
    """Return functorch transform-level introspection when available.

    Returns
    -------
    Callable[[], Any] | None
        ``torch._C._functorch.maybe_current_level`` or ``None`` when absent.
    """

    maybe_current_level = _nested_getattr_or_none(
        torch, ("_C", "_functorch", "maybe_current_level")
    )
    if maybe_current_level is None:
        mark_torch_capability_missing(
            "HAS_FUNCTORCH_LEVEL_API",
            "functorch transform-boundary detection is disabled",
        )
        return None
    return maybe_current_level


def get_functorch_wrapped_tensor_checker() -> Callable[[Any], bool] | None:
    """Return functorch wrapped-tensor detection when available.

    Returns
    -------
    Callable[[Any], bool] | None
        ``torch._C._functorch.is_functorch_wrapped_tensor`` or ``None``.
    """

    checker = _nested_getattr_or_none(torch, ("_C", "_functorch", "is_functorch_wrapped_tensor"))
    if checker is None:
        mark_torch_capability_missing(
            "HAS_FUNCTORCH_WRAPPED_TENSOR_API",
            "functorch wrapped-tensor equality handling is disabled",
        )
        return None
    return checker


def get_jit_builtin_table() -> dict[int, Any] | None:
    """Return TorchScript's private builtin table when available.

    Returns
    -------
    dict[int, Any] | None
        TorchScript builtin table, or ``None`` when unavailable.
    """

    builtin_table = _import_module_attr_or_none("torch.jit._builtins", "_builtin_table")
    if builtin_table is None:
        mark_torch_capability_missing(
            "HAS_JIT_BUILTIN_TABLE",
            "TorchScript wrapper registration is disabled",
        )
        return None
    return builtin_table


def get_device_context_type() -> type[Any] | None:
    """Return torch's DeviceContext type when available.

    Returns
    -------
    type[Any] | None
        ``torch.utils._device.DeviceContext`` or ``None`` when unavailable.
    """

    device_context = _import_module_attr_or_none("torch.utils._device", "DeviceContext")
    if device_context is None:
        mark_torch_capability_missing(
            "HAS_DEVICE_CONTEXT_DISPATCH",
            "torch.device context kwarg injection is disabled",
        )
        return None
    return device_context


def get_current_function_mode_stack() -> Iterable[Any] | None:
    """Return the current TorchFunctionMode stack when available.

    Returns
    -------
    Iterable[Any] | None
        Current mode stack, or ``None`` when stack introspection is unavailable.
    """

    stack_getter = _import_module_attr_or_none(
        "torch.overrides", "_get_current_function_mode_stack"
    )
    if stack_getter is None:
        mark_torch_capability_missing(
            "HAS_DEVICE_CONTEXT_DISPATCH",
            "torch.device context stack introspection is disabled",
        )
        return None
    return stack_getter()


def get_torch_function_mode_stack_length() -> int | None:
    """Return the TorchFunctionMode stack length when available.

    Returns
    -------
    int | None
        Active stack length, or ``None`` when unavailable.
    """

    stack_len = _import_module_attr_or_none("torch.overrides", "_len_torch_function_stack")
    if stack_len is None:
        mark_torch_capability_missing(
            "HAS_DEVICE_CONTEXT_DISPATCH",
            "torch.device context stack length introspection is disabled",
        )
        return None
    return int(stack_len())


def get_device_constructors() -> Any | None:
    """Return torch's private device-constructor inventory when available.

    Returns
    -------
    Any | None
        ``torch.utils._device._device_constructors`` or ``None`` when absent.
    """

    device_constructors = _import_module_attr_or_none("torch.utils._device", "_device_constructors")
    if device_constructors is None:
        mark_torch_capability_missing(
            "HAS_DEVICE_CONSTRUCTORS",
            "factory-function device injection inventory is disabled",
        )
        return None
    return device_constructors


def get_fx_graph_module_type() -> type[Any] | None:
    """Return ``torch.fx.GraphModule`` when available.

    Returns
    -------
    type[Any] | None
        FX GraphModule type, or ``None`` when unavailable.
    """

    graph_module_type = _nested_getattr_or_none(torch, ("fx", "GraphModule"))
    if graph_module_type is None:
        mark_torch_capability_missing(
            "HAS_FX_GRAPH_MODULE",
            "FX GraphModule compatibility detection is disabled",
        )
        return None
    return graph_module_type


def get_dynamo_optimized_module_type() -> type[Any] | None:
    """Return Dynamo's private ``OptimizedModule`` type when available.

    Returns
    -------
    type[Any] | None
        Dynamo OptimizedModule type, or ``None`` when unavailable.
    """

    global HAS_DYNAMO_OPTIMIZED_MODULE, _DYNAMO_OPTIMIZED_MODULE_PROBED
    global _DYNAMO_OPTIMIZED_MODULE_TYPE

    if not _DYNAMO_OPTIMIZED_MODULE_PROBED:
        _DYNAMO_OPTIMIZED_MODULE_TYPE = _import_module_attr_or_none(
            "torch._dynamo.eval_frame", "OptimizedModule"
        )
        HAS_DYNAMO_OPTIMIZED_MODULE = _DYNAMO_OPTIMIZED_MODULE_TYPE is not None
        _DYNAMO_OPTIMIZED_MODULE_PROBED = True
    optimized_module_type = _DYNAMO_OPTIMIZED_MODULE_TYPE
    if optimized_module_type is None:
        mark_torch_capability_missing(
            "HAS_DYNAMO_OPTIMIZED_MODULE",
            "torch.compile wrapper detection is falling back to class-name matching",
        )
        return None
    return optimized_module_type


def fix_tensor_sequence_slot() -> bool:
    """Clear the stale CPython ``sq_item`` slot on ``torch.Tensor`` when possible.

    Wrapping ``torch.Tensor.__getitem__`` on CPython can leave the sequence
    protocol's ``sq_item`` slot populated. If this capability is unavailable,
    TorchLens still captures normally, but scalar tensors may again look like
    sequences to CPython C APIs after wrap/unwrap cycles, which can break calls
    such as ``torch.tensor([zero_dim_tensor])``.

    Returns
    -------
    bool
        True when the slot was inspected and cleared or already absent; False
        when the private CPython layout was unavailable.
    """

    if sys.implementation.name != "cpython":
        mark_torch_capability_missing(
            "HAS_TENSOR_SEQUENCE_SLOT_FIX",
            "Tensor __getitem__ wrap/unwrap may leave scalar tensors sequence-like",
        )
        return False
    try:
        type_obj = _PyTypeObject.from_address(id(torch.Tensor))
    except Exception:
        mark_torch_capability_missing(
            "HAS_TENSOR_SEQUENCE_SLOT_FIX",
            "Tensor sequence-slot layout could not be inspected",
        )
        return False
    if type_obj.tp_name != b"Tensor" or not type_obj.tp_as_sequence:
        mark_torch_capability_missing(
            "HAS_TENSOR_SEQUENCE_SLOT_FIX",
            "Tensor sequence-slot layout did not match the expected CPython structure",
        )
        return False
    type_obj.tp_as_sequence.contents.sq_item = None
    return True


# --- Legacy (torch 2.1-2.3) per-device fallbacks -------------------------------
#
# These map a device_type string onto the old device-specific helpers.  We only
# special-case the two device types TorchLens captures ("cpu", "cuda"); any other
# device_type on legacy torch has no autocast-state query API and raises
# RuntimeError, which the caller in rng.py already swallows (it skips devices that
# can't be queried).


def _legacy_is_autocast_enabled(device_type: str) -> bool:
    """Return legacy autocast enabled state for a supported device type.

    Parameters
    ----------
    device_type:
        Device type accepted by legacy torch autocast helpers.

    Returns
    -------
    bool
        Whether autocast is enabled for ``device_type``.
    """

    if device_type == "cpu":
        return bool(torch.is_autocast_cpu_enabled())  # type: ignore[attr-defined]
    if device_type == "cuda":
        # The no-arg form queries the CUDA/GPU autocast flag on legacy torch.
        return bool(torch.is_autocast_enabled())
    raise RuntimeError(
        f"autocast state query not supported for device_type={device_type!r} "
        f"on torch {torch.__version__}"
    )


def _legacy_get_autocast_dtype(device_type: str) -> torch.dtype:
    """Return legacy autocast dtype for a supported device type.

    Parameters
    ----------
    device_type:
        Device type accepted by legacy torch autocast helpers.

    Returns
    -------
    torch.dtype
        Autocast dtype for ``device_type``.
    """

    if device_type == "cpu":
        return torch.get_autocast_cpu_dtype()  # type: ignore[attr-defined]
    if device_type == "cuda":
        return torch.get_autocast_gpu_dtype()  # type: ignore[attr-defined]
    raise RuntimeError(
        f"autocast dtype query not supported for device_type={device_type!r} "
        f"on torch {torch.__version__}"
    )


# --- Public, version-neutral entry points --------------------------------------
#
# Bound once at import time to the correct implementation.  On torch >= 2.4 these
# are exactly ``torch.is_autocast_enabled`` / ``torch.get_autocast_dtype`` (byte
# identical behavior); on torch 2.1-2.3 they route to the legacy helpers above.

if AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED:

    def autocast_is_enabled(device_type: str) -> bool:
        """Return whether autocast is enabled for ``device_type`` (torch>=2.4 path)."""
        return bool(torch.is_autocast_enabled(device_type))

    def autocast_get_dtype(device_type: str) -> torch.dtype:
        """Return the autocast dtype for ``device_type`` (torch>=2.4 path)."""
        return torch.get_autocast_dtype(device_type)

else:

    def autocast_is_enabled(device_type: str) -> bool:
        """Return whether autocast is enabled for ``device_type`` (torch 2.1-2.3 path)."""
        return _legacy_is_autocast_enabled(device_type)

    def autocast_get_dtype(device_type: str) -> torch.dtype:
        """Return the autocast dtype for ``device_type`` (torch 2.1-2.3 path)."""
        return _legacy_get_autocast_dtype(device_type)


# --- r35 decision E: ambient execution-context snapshot/restore ------------------
#
# Every knob below is a fragile cross-version torch surface, so it is probed here
# (the doctrinal home for such probes) behind named ``HAS_*`` capability flags.
# ``None`` in a snapshot means "this runtime does not expose the control"; every
# exposed control is recorded affirmatively, including its disabled state.


def _probe_float32_matmul_precision() -> bool:
    """Return whether this torch exposes the float32 matmul precision control."""

    return callable(getattr(torch, "get_float32_matmul_precision", None)) and callable(
        getattr(torch, "set_float32_matmul_precision", None)
    )


def _probe_deterministic_algorithms_query() -> bool:
    """Return whether deterministic-algorithms mode is both queryable and settable."""

    return (
        callable(getattr(torch, "are_deterministic_algorithms_enabled", None))
        and callable(getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None))
        and callable(getattr(torch, "use_deterministic_algorithms", None))
    )


def _probe_cuda_matmul_tf32() -> bool:
    """Return whether the CUDA matmul TF32 flag is exposed."""

    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    return matmul is not None and hasattr(matmul, "allow_tf32")


def _probe_cudnn_flags() -> bool:
    """Return whether the cuDNN backend flag set is exposed."""

    cudnn = getattr(torch.backends, "cudnn", None)
    return cudnn is not None and all(
        hasattr(cudnn, name) for name in ("allow_tf32", "deterministic", "benchmark", "enabled")
    )


def _probe_sdp_toggles() -> bool:
    """Return whether the scaled-dot-product attention backend toggles are exposed."""

    cuda_backend = getattr(torch.backends, "cuda", None)
    return cuda_backend is not None and all(
        callable(getattr(cuda_backend, name, None))
        for name in (
            "flash_sdp_enabled",
            "enable_flash_sdp",
            "mem_efficient_sdp_enabled",
            "enable_mem_efficient_sdp",
            "math_sdp_enabled",
            "enable_math_sdp",
        )
    )


def _probe_fill_uninitialized_memory() -> bool:
    """Return whether the deterministic uninit-memory fill knob is exposed.

    ``torch.utils.deterministic.fill_uninitialized_memory`` (torch >= 2.1)
    governs whether ``torch.use_deterministic_algorithms(True)`` deterministically
    fills the ``empty`` factory family. Feature-detected for the r53 hon_1/hon_2
    ambient wave; an absent knob records ``None`` (never a guessed boolean).
    """

    deterministic = getattr(getattr(torch, "utils", None), "deterministic", None)
    if deterministic is None:
        return False
    try:
        return isinstance(deterministic.fill_uninitialized_memory, bool)
    except (AttributeError, RuntimeError):
        return False


HAS_FLOAT32_MATMUL_PRECISION: bool = _probe_float32_matmul_precision()
HAS_DETERMINISTIC_ALGORITHMS_QUERY: bool = _probe_deterministic_algorithms_query()
HAS_CUDA_MATMUL_TF32: bool = _probe_cuda_matmul_tf32()
HAS_CUDNN_FLAGS: bool = _probe_cudnn_flags()
HAS_SDP_TOGGLES: bool = _probe_sdp_toggles()
HAS_FILL_UNINITIALIZED_MEMORY: bool = _probe_fill_uninitialized_memory()


def read_fill_uninitialized_memory() -> bool | None:
    """Return the deterministic uninit-memory fill flag, or ``None`` when absent.

    THE one sanctioned read of ``torch.utils.deterministic.fill_uninitialized_memory``
    (a module-``__getattr__`` property invisible to static typing): the ambient
    snapshot and the producer-side determinism refinement both route here.
    """

    if not HAS_FILL_UNINITIALIZED_MEMORY:
        return None
    return bool(torch.utils.deterministic.fill_uninitialized_memory)  # type: ignore[attr-defined]


def write_fill_uninitialized_memory(value: bool) -> None:
    """Set the deterministic uninit-memory fill flag (caller checks the flag)."""

    torch.utils.deterministic.fill_uninitialized_memory = bool(value)  # type: ignore[attr-defined]


def tensor_version_or_none(tensor: Any) -> int | None:
    """Return ``tensor._version`` for TorchLens-internal bookkeeping, or ``None``.

    r37 hon1_4: inference tensors REJECT ``_version`` with ``RuntimeError``
    ("Inference tensors do not track version counter"), which ``getattr(...,
    None)`` does not swallow -- every capture under ambient
    ``torch.inference_mode()`` crashed on TL's own dedup/reference/version
    bookkeeping reads. This helper is the ONE safe accessor for every
    TorchLens-owned ``_version`` read; an unavailable version degrades the
    optimization (dedup miss / conservative mutation fallback), never aborts
    capture. Genuine USER ``_version`` reads keep their native torch behavior.
    """

    try:
        version = tensor._version
    except (RuntimeError, AttributeError, NotImplementedError, TypeError):
        return None
    return int(version) if isinstance(version, int) else None


def snapshot_ambient_execution_context() -> dict[str, Any]:
    """Snapshot the capture-scoped ambient backend execution context (decision E).

    Returns
    -------
    dict[str, Any]
        Plain JSON-safe mapping of every decision-E ambient control. Controls the
        runtime does not expose are ``None``; exposed controls are recorded
        affirmatively (explicit ``False``), never omitted.
    """

    snapshot: dict[str, Any] = {
        "default_dtype": str(torch.get_default_dtype()),
        "default_device": str(getattr(torch, "get_default_device", lambda: "cpu")()),
        "float32_matmul_precision": (
            str(torch.get_float32_matmul_precision()) if HAS_FLOAT32_MATMUL_PRECISION else None
        ),
        "deterministic_algorithms": (
            bool(torch.are_deterministic_algorithms_enabled())
            if HAS_DETERMINISTIC_ALGORITHMS_QUERY
            else None
        ),
        "deterministic_algorithms_warn_only": (
            bool(torch.is_deterministic_algorithms_warn_only_enabled())
            if HAS_DETERMINISTIC_ALGORITHMS_QUERY
            else None
        ),
        "cuda_matmul_allow_tf32": (
            bool(torch.backends.cuda.matmul.allow_tf32) if HAS_CUDA_MATMUL_TF32 else None
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if HAS_CUDNN_FLAGS else None,
        "cudnn_deterministic": (
            bool(torch.backends.cudnn.deterministic) if HAS_CUDNN_FLAGS else None
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if HAS_CUDNN_FLAGS else None,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled) if HAS_CUDNN_FLAGS else None,
        "flash_sdp_enabled": (
            bool(torch.backends.cuda.flash_sdp_enabled()) if HAS_SDP_TOGGLES else None
        ),
        "mem_efficient_sdp_enabled": (
            bool(torch.backends.cuda.mem_efficient_sdp_enabled()) if HAS_SDP_TOGGLES else None
        ),
        "math_sdp_enabled": (
            bool(torch.backends.cuda.math_sdp_enabled()) if HAS_SDP_TOGGLES else None
        ),
        # r53 hon_1: the GLOBAL autograd/inference mode is a result-affecting
        # ambient control (a Python branch on ``torch.is_grad_enabled()`` /
        # ``is_inference_mode_enabled()`` is steered by it). Every supported
        # torch exposes both queries, so they are REQUIRED strict booleans --
        # never ``None``, never defaulted.
        "grad_enabled": bool(torch.is_grad_enabled()),
        "inference_mode": bool(torch.is_inference_mode_enabled()),
        # r53 hon_2: deterministic uninit-memory fill knob (feature-detected).
        "fill_uninitialized_memory": read_fill_uninitialized_memory(),
    }
    return snapshot


def apply_ambient_execution_context(values: dict[str, Any]) -> None:
    """Apply one recorded ambient execution context to the current runtime.

    Parameters
    ----------
    values:
        Mapping in the :func:`snapshot_ambient_execution_context` shape. ``None``
        entries (producer runtime did not expose the control) are skipped; the
        caller decides how to treat a recorded value this runtime cannot set.

    Raises
    ------
    RuntimeError
        If a recorded (non-``None``) control cannot be applied on this runtime.
    """

    def _require(flag: bool, name: str) -> None:
        if not flag:
            raise RuntimeError(
                f"Recorded ambient execution context field {name!r} is not "
                "supported by this torch runtime."
            )

    default_dtype = values.get("default_dtype")
    if default_dtype is not None:
        # r47 secD_1: ``torch_attr`` reads ``torch.__dict__`` (no lazy ``torch.__getattr__``).
        dtype = torch_attr(str(default_dtype).removeprefix("torch."))
        if not isinstance(dtype, torch.dtype):
            raise RuntimeError(f"Recorded default dtype {default_dtype!r} is unavailable.")
        if torch.get_default_dtype() is not dtype:
            torch.set_default_dtype(dtype)
    # r37 R4 (corr2-3/corr2-2): ``default_device`` is deliberately NOT applied here.
    # ``torch.set_default_device`` mutates the PROCESS-level TorchFunctionMode stack
    # (installing or replacing a DeviceContext), so setter-based apply/restore leaks
    # a mode into the caller (measured: a fresh implicit-CPU consumer ended with a
    # process DeviceContext installed) and corrupts nested caller modes. The recorded
    # default device is entered as a SCOPED ``with torch.device(recorded)`` context
    # around the run transaction (see ``_ambient_execution_context_restored``), whose
    # ``__exit__`` restores the caller's exact mode stack by construction on every
    # exit path -- no restore logic, no global mutation. The snapshot schema keeps
    # the ``default_device`` key unchanged.
    deterministic = values.get("deterministic_algorithms")
    if deterministic is not None:
        _require(HAS_DETERMINISTIC_ALGORITHMS_QUERY, "deterministic_algorithms")
        warn_only = bool(values.get("deterministic_algorithms_warn_only") or False)
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=warn_only)
    # ``torch.backends.cuda.matmul.allow_tf32`` and
    # ``torch.set_float32_matmul_precision`` are two views of ONE underlying
    # control (setting ``allow_tf32=True`` coerces precision to "high"). Apply
    # the coarse Boolean FIRST so the finer-grained recorded precision value
    # wins; the pair is snapshotted together, so the final state is coherent.
    cuda_tf32 = values.get("cuda_matmul_allow_tf32")
    if cuda_tf32 is not None:
        _require(HAS_CUDA_MATMUL_TF32, "cuda_matmul_allow_tf32")
        torch.backends.cuda.matmul.allow_tf32 = bool(cuda_tf32)
    precision = values.get("float32_matmul_precision")
    if precision is not None:
        _require(HAS_FLOAT32_MATMUL_PRECISION, "float32_matmul_precision")
        torch.set_float32_matmul_precision(str(precision))
    for field_name, attr in (
        ("cudnn_allow_tf32", "allow_tf32"),
        ("cudnn_deterministic", "deterministic"),
        ("cudnn_benchmark", "benchmark"),
        ("cudnn_enabled", "enabled"),
    ):
        recorded = values.get(field_name)
        if recorded is None:
            continue
        _require(HAS_CUDNN_FLAGS, field_name)
        setattr(torch.backends.cudnn, attr, bool(recorded))
    for field_name, setter in (
        ("flash_sdp_enabled", "enable_flash_sdp"),
        ("mem_efficient_sdp_enabled", "enable_mem_efficient_sdp"),
        ("math_sdp_enabled", "enable_math_sdp"),
    ):
        recorded = values.get(field_name)
        if recorded is None:
            continue
        _require(HAS_SDP_TOGGLES, field_name)
        getattr(torch.backends.cuda, setter)(bool(recorded))
    # r53 hon_2: the deterministic uninit-memory fill knob restores through the
    # ordinary setter path (module property setter; snapshot/apply transactional).
    fill_uninitialized = values.get("fill_uninitialized_memory")
    if fill_uninitialized is not None:
        _require(HAS_FILL_UNINITIALIZED_MEMORY, "fill_uninitialized_memory")
        write_fill_uninitialized_memory(bool(fill_uninitialized))
    # r53 hon_1: ``grad_enabled``/``inference_mode`` are deliberately NOT applied
    # here. Like ``default_device`` (r37 R4 above), they restore as SCOPED
    # contexts (``torch.set_grad_enabled`` / ``torch.inference_mode``) around the
    # run transaction in ``_ambient_execution_context_restored``, whose
    # ``__exit__`` restores the caller's exact mode on every exit path by
    # construction -- a setter-based apply of a THREAD-LOCAL autograd mode from a
    # ``finally`` block could race an interleaved caller context. The snapshot
    # schema keeps both keys for the ambient-coverage meta-test.


# --- r35 hon1_4: repr-independent torch structseq field discovery -----------------
#
# The ONLY sanctioned sources for ``torch.return_types`` structseq field names are
# the TYPE's ``__match_args__`` declaration (CPython 3.10+) and, on 3.9, an
# identity round-trip over the type's member descriptors. ``repr()`` parsing is
# FORBIDDEN: console wrap position injects phantom fields (``dtype=`` /
# ``grad_fn=`` at line start), flipping witness verdicts on tensor size alone.


def torch_structseq_field_names(value: Any) -> tuple[str, ...]:
    """Return the exact ordered public field names of a torch structseq value.

    Parameters
    ----------
    value:
        Candidate ``torch.return_types.*`` (PyStructSequence) instance.

    Returns
    -------
    tuple[str, ...]
        The validated ordered field names, or ``()`` when ``value`` is not a
        fully named torch structseq or its declaration cannot be PROVEN
        (producers refuse such a structseq; runtimes report typed
        unavailability -- never a guess, never a repr parse).
    """

    cls = type(value)
    if cls.__module__ != "torch.return_types" or not isinstance(value, tuple):
        return ()
    n_fields = getattr(value, "n_fields", None)
    n_unnamed = getattr(value, "n_unnamed_fields", 0)
    if not isinstance(n_fields, int) or n_fields <= 0 or n_unnamed:
        return ()
    match_args = getattr(cls, "__match_args__", None)
    if (
        isinstance(match_args, tuple)
        and len(match_args) == n_fields
        and len(set(match_args)) == n_fields
        and all(
            isinstance(name, str) and name.isidentifier() and not name.startswith("_")
            for name in match_args
        )
    ):
        return tuple(str(name) for name in match_args)
    # CPython 3.9 fallback: derive the name -> index bijection by IDENTITY
    # round-trip over the type's descriptors (``getattr(value, name) is
    # value[i]`` for exactly one ``i``). Refuse on any ambiguity or
    # non-bijection -- fail closed, never repr.
    candidates = [
        name
        for name, member in vars(cls).items()
        if not name.startswith("_") and hasattr(member, "__get__") and not callable(member)
    ]
    if len(candidates) != n_fields or len(value) != n_fields:
        return ()
    by_index: dict[int, str] = {}
    for name in candidates:
        try:
            attribute = getattr(value, name)
        except Exception:
            return ()
        matches = [index for index in range(n_fields) if value[index] is attribute]
        if len(matches) != 1 or matches[0] in by_index:
            return ()
        by_index[matches[0]] = name
    if len(by_index) != n_fields:
        return ()
    return tuple(by_index[index] for index in range(n_fields))
