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
import importlib
import os
from typing import Any
import warnings

import torch

__all__ = [
    "AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED",
    "HAS_ACCUMULATE_GRAD_CLASS",
    "HAS_AUTOCAST_DEVICE_TYPE_ARG",
    "HAS_DEVICE_CONTEXT_DISPATCH",
    "HAS_DEVICE_CONSTRUCTORS",
    "HAS_FUNCTORCH_APIS",
    "HAS_FUNCTORCH_LEVEL_API",
    "HAS_FUNCTORCH_WRAPPED_TENSOR_API",
    "HAS_FX_GRAPH_MODULE",
    "HAS_JIT_BUILTIN_TABLE",
    "HAS_TORCH_FUNC",
    "HAS_TORCH_VF",
    "HAS_VARIABLE_FUNCTIONS",
    "TorchCapabilitySnapshot",
    "autocast_get_dtype",
    "autocast_is_enabled",
    "get_accumulate_grad_class",
    "get_current_function_mode_stack",
    "get_device_constructors",
    "get_device_context_type",
    "get_functorch_maybe_current_level",
    "get_functorch_wrapped_tensor_checker",
    "get_fx_graph_module_type",
    "get_jit_builtin_table",
    "get_optional_torch_namespace",
    "get_torch_capability_snapshot",
    "get_torch_function_mode_stack_length",
    "get_torch_vf_namespace",
    "get_variable_function_names",
    "mark_torch_capability_missing",
]


TorchCapabilitySnapshot = dict[str, bool]
"""Stable mapping from torch capability flag name to availability."""

_CAPABILITY_WARNING_ENV = "TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS"
_warned_missing_capabilities: set[str] = set()


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
        Mapping from ``HAS_*`` flag name to boolean availability.
    """

    return {name: bool(globals()[name]) for name in _CAPABILITY_ATTRS}


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


def get_device_constructors() -> Callable[[], Iterable[Any]] | None:
    """Return torch's private device-constructor inventory when available.

    Returns
    -------
    Callable[[], Iterable[Any]] | None
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


# --- Legacy (torch 2.1-2.3) per-device fallbacks -------------------------------
#
# These map a device_type string onto the old device-specific helpers.  We only
# special-case the two device types TorchLens captures ("cpu", "cuda"); any other
# device_type on legacy torch has no autocast-state query API and raises
# RuntimeError, which the caller in rng.py already swallows (it skips devices that
# can't be queried).


def _legacy_is_autocast_enabled(device_type: str) -> bool:
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
