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

import contextlib
import importlib
import inspect
import sys
import types
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from importlib.machinery import PathFinder
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ..errors._base import TorchLensWarning
from ._torch_symbols import shadowed_torch_submodule, torch_attr

__all__ = [
    "AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED",
    "HAS_ACCUMULATE_GRAD_CLASS",
    "grad_copy_source_is_leaf",
    "HAS_C10D_ABORT_PG",
    "HAS_C10D_GROUP_REGISTRY",
    "HAS_C10D_GROUP_SEQ",
    "HAS_CURRENT_GRAPH_TASK_ID",
    "HAS_DYNAMO_EXPLAIN",
    "apply_ambient_execution_context",
    "restore_fp32_precision_controls",
    "read_legacy_fp32_controls",
    "AMBIENT_FP32_UNREPRESENTABLE_KEY",
    "snapshot_fp32_precision_controls",
    "read_fill_uninitialized_memory",
    "snapshot_ambient_execution_context",
    "write_fill_uninitialized_memory",
    "HAS_DEVICE_CONTEXT_DISPATCH",
    "HAS_DEVICE_CONSTRUCTORS",
    "HAS_DEVICE_MESH",
    "HAS_DISABLE_TORCH_FUNCTION",
    "HAS_DISPATCH_MODE_STACK_QUERY",
    "HAS_DTENSOR",
    "HAS_DTENSOR_SHARD_GEOMETRY",
    "HAS_DYNAMO_COMPILE_COUNTERS",
    "HAS_DYNAMO_IS_COMPILING",
    "HAS_FAKE_TENSOR_MODE",
    "HAS_FUNCOL_GROUP_RESOLUTION",
    "HAS_FUNCOL_WAIT_INTERPOSITION",
    "HAS_FUNCOL_MODULE",
    "HAS_ASYNC_COLLECTIVE_TENSOR",
    "HAS_CHECKPOINT_HOOK_CLASS",
    "HAS_CHECKPOINT_INTERNAL_HOOK_CLASS",
    "HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK",
    "HAS_KINETO_INMEMORY_EVENTS",
    "HAS_KINETO_EVENT_SCOPE",
    "HAS_MEMORY_PROFILE",
    "KINETO_EVENT_FIELD_CONTRACT",
    "HAS_JIT_SCHEMA_ENUMERATION",
    "HAS_TENSORBASE_CLASS",
    "HAS_VARIABLE_FUNCTIONS_CLASS",
    "HAS_FP8_DTYPES",
    "HAS_PIPELINING",
    "HAS_SET_STANCE",
    "HAS_TORCH_FUNCTION_STACK_SURGERY",
    "HAS_TRACING_TENSOR_TYPES",
    "HAS_FUNCTORCH_APIS",
    "HAS_FUNCTORCH_LEVEL_API",
    "HAS_FUNCTORCH_WRAPPED_TENSOR_API",
    "HAS_FX_GRAPH_MODULE",
    "HAS_GENERATOR_CLONE_STATE",
    "HAS_GENERATOR_GRAPHSAFE_GET_STATE",
    "HAS_GENERATOR_GRAPHSAFE_SET_STATE",
    "HAS_GENERATOR_PHILOX_STATE",
    "HAS_JIT_BUILTIN_TABLE",
    "HAS_JIT_BOOLEAN_DISPATCH_TABLE",
    "HAS_JIT_OVERLOAD_RESOLVER",
    "HAS_NAMED_TENSOR_API",
    "HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE",
    "HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE",
    "HAS_DYNAMO_OPTIMIZED_MODULE",
    "HAS_DYNAMO_ORIG_CALLABLE_MARKER",
    "HAS_FSDP_WRAPPER",
    "HAS_SAFE_WEIGHTS_ONLY_LOAD",
    "HAS_SAVED_TENSORS_HOOK_INTROSPECTION",
    "HAS_SAVED_TENSORS_HOOKS_PATCHABLE",
    "HAS_CODE_POSITIONS",
    "HAS_CODE_QUALNAME",
    "HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG",
    "HAS_ATTENTION_CAUSAL_BIAS",
    "HAS_EXPANDED_WEIGHTS_CONV_PICKER",
    "HAS_DETERMINISTIC_FILL_FLAG",
    "HAS_FP32_PRECISION_CONTROLS",
    "HAS_AMP_GRADSCALER",
    "HAS_NN_ATTENTION_MODULE",
    "HAS_RMSNORM_MODULE",
    "HAS_MHA_FASTPATH_SWITCH",
    "HAS_REDUCE_TUPLE_DIM",
    "HAS_CPU_HALF_KERNELS",
    "HAS_CPU_FLOAT8_DETERMINISTIC_FILL",
    "HAS_META_ITEM_GUARD",
    "tensor_any_over_dims",
    "get_mha_fastpath_enabled",
    "set_mha_fastpath_enabled",
    "force_mha_slow_path",
    "HAS_CACHED_UNTYPED_STORAGE_WRAPPER",
    "HAS_TENSOR_SEQUENCE_SLOT_FIX",
    "HAS_LEGACY_CONSTRUCTOR_NEW_PATCH",
    "HAS_TORCH_FUNC",
    "HAS_TORCH_VF",
    "HAS_VARIABLE_FUNCTIONS",
    "TorchCapabilitySnapshot",
    "TorchCapabilityWarning",
    "saved_tensors_default_hooks_active",
    "RunnableTorchAlias",
    "autocast_get_dtype",
    "autocast_is_enabled",
    "get_accumulate_grad_class",
    "get_current_dispatch_mode_stack",
    "get_current_function_mode_stack",
    "get_disable_torch_function_context",
    "get_dtensor_shard_geometry_fn",
    "get_fake_tensor_mode_class",
    "get_jit_all_schemas",
    "get_tensorbase_class",
    "get_variable_functions_class",
    "get_device_constructors",
    "get_device_context_type",
    "get_device_mesh_type",
    "get_dtensor_type",
    "get_gradient_edge_support",
    "get_reduce_tuple_dim_support",
    "get_cpu_half_kernels_support",
    "get_cpu_float8_deterministic_fill_support",
    "get_pipelining_module_types",
    "get_fp8_dtypes",
    "get_tracing_tensor_types",
    "dynamo_is_compiling",
    "force_eager_stance_scope",
    "get_dynamo_compile_counters",
    "get_dynamo_optimized_module_type",
    "get_dynamo_explain",
    "get_functorch_maybe_current_level",
    "get_functorch_wrapped_tensor_checker",
    "get_fsdp_wrapper_type",
    "get_fx_graph_module_type",
    "get_jit_builtin_table",
    "get_jit_boolean_dispatch_table",
    "get_jit_overload_resolver_module",
    "get_optional_torch_namespace",
    "get_torch_capability_snapshot",
    "kineto_events_from_profiler",
    "memory_profile_from_profiler",
    "probe_c10d_capabilities",
    "get_torch_function_mode_stack_length",
    "get_torch_function_stack_surgery",
    "get_torch_vf_namespace",
    "get_variable_function_names",
    "fix_tensor_sequence_slot",
    "mark_torch_capability_missing",
    "is_dynamo_compiled_callable",
    "resolve_runnable_torch_alias",
    "tensor_has_named_dims",
]


TorchCapabilitySnapshot = dict[str, bool]
"""Stable mapping from torch capability name to availability."""

_CAPABILITY_WARNING_ENV = "TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS"
_warned_missing_capabilities: set[str] = set()


class _ParameterAsSubclassProbeMode(TorchDispatchMode):
    """Redispatch operations unchanged for the Parameter subclass capability probe."""

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Redispatch one probe operation without changing its result.

        Parameters
        ----------
        func:
            Dispatcher operator invoked by the probe.
        types:
            Participating tensor subclass types.
        args:
            Positional operator arguments.
        kwargs:
            Keyword operator arguments.

        Returns
        -------
        Any
            Unmodified operator result.
        """

        del types
        return func(*args, **(kwargs or {}))


@dataclass(frozen=True, slots=True)
class RunnableTorchAlias:
    """One monotonic callable move used by sparse runnable reattachment.

    The table is capability-bounded rather than selected by parsing
    ``torch.__version__``. Source absence is established by the caller's
    allowlisted exact-resolution pass, which runs first and only reaches the
    alias stage when the recorded path does not resolve through a safe public
    namespace; the private/internal recorded paths in this table are therefore
    redirected to their public equivalents (a present internal binding such as
    ``torch._C._nn.linear`` is intentionally routed to its public wrapper for
    replay safety). This resolver additionally probes that an entry's *target*
    exists in the current runtime (:func:`_runtime_alias_target_exists`) and
    never hands back an alias whose target is absent -- so an unbounded/legacy
    version key can never manufacture a dead alias. A ``None`` maximum leaves
    a monotonic alias open above its minimum; target existence remains the
    capability gate in that case.
    """

    source: str
    target_namespace: str
    target_qualname: str | None
    provenance: str
    recorded_min_version: tuple[int, int]
    recorded_max_version: tuple[int, int] | None
    strip_target_prefix: str = ""


@dataclass(frozen=True, slots=True)
class _NormalizedDynamoBreak:
    """Version-neutral graph-break evidence from ``torch._dynamo.explain``."""

    reason: str
    source_file: str | None
    line_number: int | None


class _DynamoExplainOutputError(RuntimeError):
    """Raised when a Dynamo explain result has an unsupported runtime shape."""


_RUNNABLE_TORCH_ALIASES: tuple[RunnableTorchAlias, ...] = (
    RunnableTorchAlias(
        "torch._C._nn.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_C._nn.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 14),
    ),
    RunnableTorchAlias(
        "_C._nn.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_C._nn.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 14),
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
        (2, 14),
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
        None,
    ),
    RunnableTorchAlias(
        "torch._VariableFunctionsClass.*",
        "torch",
        None,
        "private_to_public:torch._VariableFunctionsClass->torch",
        (2, 1),
        None,
    ),
    RunnableTorchAlias(
        "_VariableFunctionsClass.*",
        "torch",
        None,
        "private_to_public:_VariableFunctionsClass->torch",
        (2, 1),
        None,
    ),
)


def _runtime_alias_target_exists(target_namespace: str, target_qualname: str) -> bool:
    """Return whether an alias target resolves in the running torch.

    Capability probe (feature detection, not a ``torch.__version__`` parse): the
    resolver only offers an alias whose recorded target actually exists in the
    current runtime, honoring :class:`RunnableTorchAlias`'s target-existence
    contract. The first hop off the top-level ``torch`` module reads
    ``torch.__dict__`` via :func:`torch_attr` (an identifier-only read that
    avoids the PEP-562 lazy-submodule-import hazard); deeper hops are off
    submodule/class roots that carry no lazy-import hazard, so they use
    ``getattr``.

    Parameters
    ----------
    target_namespace:
        Public namespace of the alias target (always rooted at ``torch``).
    target_qualname:
        Attribute path of the alias target within ``target_namespace``.

    Returns
    -------
    bool
        True only when every path segment resolves to a non-``None`` attribute.
    """

    parts = f"{target_namespace}.{target_qualname}".split(".")
    if parts[0] != "torch":
        return False
    current: Any = torch
    for part in parts[1:]:
        if current is torch:
            current = torch_attr(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return False
    return True


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
        Torch version that produced the registry key. Aliases may be bounded
        to supported minor releases; monotonic moves with an open upper bound
        rely on the live target-existence capability probe. Unknown versions
        retain compatibility behavior for descriptors that predate this
        metadata.

    Returns
    -------
    tuple[str, str, str] | None
        Target namespace, target qualname, and stable provenance, or ``None``
        when the path has no explicit monotonic alias.
    """

    parsed_version = _torch_minor_version(recorded_version)
    for alias in _RUNNABLE_TORCH_ALIASES:
        if parsed_version is not None and (
            parsed_version < alias.recorded_min_version
            or (
                alias.recorded_max_version is not None
                and parsed_version > alias.recorded_max_version
            )
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
        if not _runtime_alias_target_exists(alias.target_namespace, target_qualname):
            # Capability gate: never manufacture an alias whose recorded target
            # is absent from the running torch (the RunnableTorchAlias contract).
            # This also closes the unbounded/legacy-version bypass: even when the
            # version bounds are skipped, a matched entry can only resolve when
            # its target genuinely exists. A later entry may still match with a
            # live target, so keep scanning.
            continue
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
        torch.is_autocast_enabled("cpu")
        return True
    except TypeError:
        return False


AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED: bool = _probe_device_type_arg_supported()


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


def _probe_jit_boolean_dispatch_table() -> bool:
    """Return whether TorchScript exposes the private boolean-dispatch table.

    Returns
    -------
    bool
        True when ``torch._jit_internal.boolean_dispatched`` is present.
    """

    return _import_module_attr_or_none("torch._jit_internal", "boolean_dispatched") is not None


def _probe_jit_overload_resolver() -> bool:
    """Return whether TorchScript exposes the private overload resolver.

    Returns
    -------
    bool
        True when ``torch.jit._script._get_overloads`` is present.
    """

    return _import_module_attr_or_none("torch.jit._script", "_get_overloads") is not None


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


def _probe_current_graph_task_id() -> bool:
    """Return whether torch exposes the autograd graph-task-id resolver.

    Returns
    -------
    bool
        True when ``torch._C._current_graph_task_id`` is present.
    """

    return _nested_getattr_or_none(torch, ("_C", "_current_graph_task_id")) is not None


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


def _probe_cached_untyped_storage_wrapper() -> bool:
    """Return whether a tensor retains one stable untyped-storage Python wrapper.

    Torch 2.1 creates a fresh ``UntypedStorage`` wrapper on every
    ``Tensor.untyped_storage()`` call. Its wrapper destructor can leave a
    ``weakref.ref`` pointing at freed memory, so placing an ephemeral handle in a
    ``WeakKeyDictionary`` can later segfault CPython while clearing the weakref.
    Newer torch retains one wrapper on the tensor, which makes weak-key storage
    registries safe for the tensor's lifetime.

    The probe deliberately compares two live handles instead of constructing a
    weakref: exercising the broken weakref destructor would itself corrupt the
    interpreter on an unsupported runtime.

    Returns
    -------
    bool
        ``True`` when repeated calls return the same retained wrapper.
    """

    try:
        tensor = torch.empty(0)
        first = tensor.untyped_storage()
        second = tensor.untyped_storage()
    except (AttributeError, RuntimeError, TypeError):
        return False
    return first is second


def _probe_dynamo_orig_callable_marker() -> bool:
    """Return whether Dynamo publishes its original-callable marker contract.

    Returns
    -------
    bool
        True when ``torch._dynamo.eval_frame.innermost_fn`` recognizes the
        ``_torchdynamo_orig_callable`` marker used on compiled callables.
    """
    innermost_fn = _import_module_attr_or_none("torch._dynamo.eval_frame", "innermost_fn")
    code = getattr(innermost_fn, "__code__", None)
    return (
        callable(innermost_fn)
        and code is not None
        and ("_torchdynamo_orig_callable" in code.co_names)
    )


def _probe_dynamo_explain_module() -> bool:
    """Return whether the Dynamo package is discoverable without importing it.

    Returns
    -------
    bool
        Whether a later lazy probe can inspect ``torch._dynamo.explain``.
    """

    try:
        return importlib.util.find_spec("torch._dynamo") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _probe_legacy_constructor_new_patch() -> bool:
    """Return whether legacy ``torch.<dtype>Tensor`` constructors can be patched in place.

    The ctypes layout probe lives in ``_type_new_slot`` (a stdlib-only leaf);
    it is imported here, at probe time, so this module's eager import closure
    does not grow.

    Returns
    -------
    bool
        True on CPython when ``torch.FloatTensor``'s type layout matches the
        mirror ``backends/torch/legacy_ctors.py`` patches through.
    """

    from ._type_new_slot import probe_type_new_slot_patch

    return probe_type_new_slot_patch(getattr(torch, "FloatTensor", None))


def _probe_tensor_sequence_slot_fix() -> bool:
    """Return whether the CPython tensor ``sq_item`` slot fix can run.

    The ctypes layout mirror lives in the stdlib-only leaf ``_type_sequence_slot``,
    imported at probe time like ``_type_new_slot``.

    Returns
    -------
    bool
        True on CPython when the expected ``torch.Tensor`` type layout is visible.
    """

    from ._type_sequence_slot import sequence_slot_layout_problem

    return sequence_slot_layout_problem(torch.Tensor, b"Tensor") is None


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


def _probe_parameter_as_subclass_in_dispatch_mode() -> bool:
    """Return whether Parameter-to-Tensor subclass conversion works in a dispatch mode.

    Torch 2.1 rejects ``Parameter.as_subclass(torch.Tensor)`` while a
    :class:`TorchDispatchMode` is active because the returned raw tensor is already
    associated with a base ``Tensor`` Python object. Newer torch releases permit the
    conversion. TorchLens' completeness witness is itself a dispatch mode, so this
    behavioral probe mirrors the dynamic-parameter logging context without parsing a
    version string.

    Returns
    -------
    bool
        ``True`` when the conversion succeeds inside a redispatching mode.
    """

    parameter = torch.nn.Parameter(torch.empty(0))
    try:
        with _ParameterAsSubclassProbeMode():
            plain_tensor = parameter.as_subclass(torch.Tensor)
    except (RuntimeError, TypeError):
        return False
    return type(plain_tensor) is torch.Tensor


class _SubclassCtorProbe(torch.Tensor):
    """Plain strict Tensor subclass used only by the ctor-in-mode probe below."""


def _probe_subclass_ctor_in_dispatch_mode() -> bool:
    """Return whether constructing a STRICT Tensor subclass works in a dispatch mode.

    Torch 2.1 and 2.2 reject ``Tensor.as_subclass``/``_make_subclass`` (and the
    equivalent ``__new__`` path) into a custom (non-``torch.Tensor``) subclass
    while a :class:`TorchDispatchMode` is active: "Creating a new Tensor
    subclass X but the raw Tensor object is already associated to a python
    object of type Tensor." This reproduces on stock torch with a no-op mode
    (TorchLens's own wrapping is not the cause); newer torch permits the
    conversion. TorchLens's completeness witness and intervention-ready
    captures are themselves dispatch modes, so call sites that construct or
    convert into a strict subclass while one is active need this behavioral
    probe rather than a parsed version string. Sibling probe (Parameter-to-
    Tensor, the opposite direction, also version-gated but independently):
    :func:`_probe_parameter_as_subclass_in_dispatch_mode`.

    Returns
    -------
    bool
        ``True`` when the conversion succeeds inside a redispatching mode.
    """

    base = torch.empty(0)
    try:
        with _ParameterAsSubclassProbeMode():
            converted = base.as_subclass(_SubclassCtorProbe)
    except (RuntimeError, TypeError):
        return False
    return type(converted) is _SubclassCtorProbe


def _probe_roll_tensor_shifts() -> bool:
    """Return whether ``torch.roll`` accepts a bare 0-dim tensor ``shifts``.

    Older torch (verified: 2.8 rejects with ``RuntimeError: shifts required``;
    2.13 accepts) does not unpack a 0-dim tensor passed as the ``shifts``
    argument itself; the tuple spelling ``shifts=(tensor,)`` is accepted
    everywhere. Behavioral probe on a 1-element tensor, never a version parse.

    Returns
    -------
    bool
        ``True`` when the bare tensor ``shifts`` spelling executes.
    """

    try:
        torch.roll(torch.zeros(2), shifts=torch.zeros((), dtype=torch.long))  # type: ignore[arg-type]
    except (RuntimeError, TypeError):
        return False
    return True


_TOP_SAVED_TENSORS_DEFAULT_HOOKS_ARGS: tuple[bool, ...] | None = None


def _probe_saved_tensors_hook_introspection() -> bool:
    """Return whether autograd default saved-tensors hooks can be peeked.

    ``torch._C._autograd._top_saved_tensors_default_hooks`` returns the
    innermost installed default pack/unpack hook pair (or ``None``) without
    popping it. Non-reentrant ``torch.utils.checkpoint`` regions install such
    hooks, and any value they packed re-runs user code (the checkpoint
    RECOMPUTE) when read back, so capture-time stats collection needs this
    peek to know reading saved values is side-effect-free. The call signature
    gained an ``ignore_is_tracing`` argument over torch history; probe both
    spellings behaviorally, never a version parse.

    Returns
    -------
    bool
        ``True`` when the peek is callable outside any hooks context.
    """

    global _TOP_SAVED_TENSORS_DEFAULT_HOOKS_ARGS
    peek = getattr(getattr(torch._C, "_autograd", None), "_top_saved_tensors_default_hooks", None)
    if peek is None:
        return False
    for call_args in ((True,), ()):
        try:
            peek(*call_args)
        except (TypeError, RuntimeError):
            continue
        _TOP_SAVED_TENSORS_DEFAULT_HOOKS_ARGS = call_args
        return True
    return False


def saved_tensors_default_hooks_active() -> bool | None:
    """Return whether autograd default saved-tensors hooks are installed now.

    Returns
    -------
    bool | None
        ``True`` when a default pack/unpack hook pair is installed (values a
        grad_fn saved for backward may be hook-packed, so reading them runs
        the user's unpack hook -- e.g. a non-reentrant checkpoint recompute),
        ``False`` when none is installed, and ``None`` when the runtime cannot
        answer (introspection API absent).
    """

    if not HAS_SAVED_TENSORS_HOOK_INTROSPECTION:
        return None
    try:
        peek = torch._C._autograd._top_saved_tensors_default_hooks
        return peek(*(_TOP_SAVED_TENSORS_DEFAULT_HOOKS_ARGS or ())) is not None
    except (RuntimeError, TypeError):
        return None


def _probe_code_positions() -> bool:
    """Return whether code objects expose PEP 657 fine-grained positions.

    ``co_positions()`` (CPython 3.11+) yields per-instruction column offsets.
    Conditional-branch attribution uses them to tell same-line ternary /
    ``IfExp`` arms apart; without them the branch indexer degrades to
    line-only matching, which deliberately FAILS CLOSED (un-attributed, never
    mis-attributed) on same-line arms. Probe the capability behaviorally.

    Returns
    -------
    bool
        ``True`` when per-instruction position introspection is available.
    """

    positions = getattr(_probe_code_positions.__code__, "co_positions", None)
    if not callable(positions):
        return False
    try:
        next(iter(positions()), None)
    except (TypeError, ValueError):
        return False
    return True


def _probe_code_qualname() -> bool:
    """Return whether code objects expose ``co_qualname`` (CPython 3.11+).

    Frame-to-scope resolution for branch attribution prefers qualified names;
    without them, same-named nested functions resolve by name plus first line
    only and ambiguous matches fail closed.

    Returns
    -------
    bool
        ``True`` when ``co_qualname`` is available on code objects.
    """

    return isinstance(getattr(_probe_code_qualname.__code__, "co_qualname", None), str)


def _probe_saved_tensors_hooks_patchable() -> bool:
    """Return whether the saved-tensors-hooks context class accepts an init patch.

    ``torch.autograd.graph.saved_tensors_hooks`` is the public context manager
    that installs user pack/unpack hooks (``save_on_cpu`` and the non-reentrant
    checkpoint hook subclass it). TorchLens scopes user hook bodies as
    autograd-internal during an active capture by patching the class
    ``__init__`` (construction-time scoping) AND ``__enter__`` (use-time
    re-scoping, covering contexts constructed before the first capture) at
    wrap time, so the patch needs a plain-Python ``__init__`` with the
    ``(self, pack_hook, unpack_hook)`` shape and a plain-Python ``__enter__``
    taking only ``self``. Probe the shape behaviorally; never parse
    ``torch.__version__``.

    Returns
    -------
    bool
        ``True`` when the class exists and both ``__init__`` and ``__enter__``
        are patchable Python functions with the expected shapes.
    """

    hooks_cls = getattr(getattr(torch.autograd, "graph", None), "saved_tensors_hooks", None)
    if not isinstance(hooks_cls, type):
        return False
    init = hooks_cls.__dict__.get("__init__")
    enter = hooks_cls.__dict__.get("__enter__")
    if not isinstance(init, types.FunctionType) or not isinstance(enter, types.FunctionType):
        return False
    try:
        init_params = list(inspect.signature(init).parameters)
        enter_params = list(inspect.signature(enter).parameters)
    except (TypeError, ValueError):
        return False
    return init_params == ["self", "pack_hook", "unpack_hook"] and enter_params == ["self"]


def _probe_torch_function_stack_surgery() -> bool:
    """Return whether the torch-function mode stack supports pop/push surgery.

    Returns
    -------
    bool
        ``True`` when the private mode-stack primitives the weightsfree
        ambient-context absorption (W1-ABSORB / D19) depends on all exist:
        ``torch._C._len_torch_function_stack``, ``_get_function_stack_at``,
        ``_pop_torch_function_stack``, and ``_push_on_torch_function_stack``.
        The probe reads attributes only; where any is missing the absorption
        degrades to the typed refusal / disclosure path, never a guess.
    """

    c_module = getattr(torch, "_C", None)
    return all(
        callable(getattr(c_module, name, None))
        for name in (
            "_len_torch_function_stack",
            "_get_function_stack_at",
            "_pop_torch_function_stack",
            "_push_on_torch_function_stack",
        )
    )


def _probe_set_stance() -> bool:
    """Return whether this torch exposes ``torch.compiler.set_stance``.

    Returns
    -------
    bool
        ``True`` when ``torch.compiler.set_stance`` exists and is callable
        (public since torch 2.6). The probe reads the attribute only; it never
        imports ``torch._dynamo`` (calling ``set_stance`` does).
    """

    compiler_module = getattr(torch, "compiler", None)
    return callable(getattr(compiler_module, "set_stance", None))


def _probe_transformer_activation_fastpath_flag() -> bool:
    """Return whether ``TransformerEncoderLayer`` exposes the fastpath flag.

    Returns
    -------
    bool
        ``True`` when a constructed layer carries ``activation_relu_or_gelu``,
        the identity-checked (``activation is F.relu/F.gelu``) gate for the
        fused fastpath and nested-tensor paths. The probe constructs a tiny
        meta-device layer under ``fork_rng`` so it never touches user RNG
        streams or allocates real storage.
    """

    layer_cls = getattr(torch.nn, "TransformerEncoderLayer", None)
    if layer_cls is None:
        return False
    try:
        with torch.random.fork_rng(devices=[]):
            layer = layer_cls(d_model=2, nhead=1, dim_feedforward=2, device="meta")
    except Exception:
        return False
    return hasattr(layer, "activation_relu_or_gelu")


def _probe_attention_causal_bias() -> bool:
    """Return whether ``torch.nn.attention.bias.CausalBias`` is discoverable.

    NEVER executes ``torch.nn.attention.bias`` (r45/r49 lazy-import belt): that
    module's top level calls ``torch._dynamo.allow_in_graph(...)``, dragging the
    whole ``_dynamo``/``_inductor`` tree into every ``import torchlens``. When
    the USER has already imported the module, the class is inspected exactly;
    otherwise ``find_spec`` proves existence without execution (the parent
    ``torch.nn.attention`` package body is dynamo-free) and the exact
    ``__torch_function__`` site check is deferred to causal-bias shim install,
    which resolves only through ``sys.modules``.

    Returns
    -------
    bool
        ``True`` when the module is already imported and the class defines its
        own ``__torch_function__`` (the identity-dispatch site the causal-bias
        identity shim normalizes), or when the module exists but has not been
        imported yet (the shim installer re-reads ``sys.modules`` at each wrap).
        Absent on torch builds predating the ``torch.nn.attention`` namespace.

    Notes
    -----
    This probe NEVER imports ``torch.nn.attention.bias``: on torch 2.13 its
    module body reaches ``torch._dynamo``, whose import tree also drags in
    ``torch.distributed.fsdp`` -- breaking the W21 cold-start guarantee that a
    plain eager capture pays neither import. ``find_spec`` imports only the
    parent ``torch.nn.attention`` package (verified dynamo-free), and a process
    that never imported the module cannot hold a ``CausalBias`` instance, so
    the deferred read is exact, not heuristic.
    """

    module = sys.modules.get("torch.nn.attention.bias")
    if module is not None:
        causal_bias = getattr(module, "CausalBias", None)
        return causal_bias is not None and "__torch_function__" in vars(causal_bias)
    try:
        return importlib.util.find_spec("torch.nn.attention.bias") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def torch_module_exists_without_importing(module_path: str) -> bool:
    """Return whether ``module_path`` exists without importing it or any missing ancestor.

    Parameters
    ----------
    module_path:
        Dotted module path, e.g. ``"torch.distributed.pipeline.sync"``.

    Returns
    -------
    bool
        ``True`` when the module is imported or resolvable on disk.
    """

    parts = module_path.split(".")
    depth = next((d for d in range(len(parts), 0, -1) if ".".join(parts[:d]) in sys.modules), 0)
    search = getattr(sys.modules[".".join(parts[:depth])], "__path__", None) if depth else sys.path
    for index in range(depth, len(parts)):
        spec = PathFinder.find_spec(".".join(parts[: index + 1]), search) if search else None
        if spec is None:
            return False
        search = spec.submodule_search_locations
    return True


def _probe_expanded_weights_conv_picker() -> bool:
    """Return whether the private expanded-weights machinery is available.

    Returns
    -------
    bool
        ``True`` when ``conv_picker`` and ``ExpandedWeight`` (with its own
        ``__torch_function__``) exist -- the per-sample-grads dispatch sites
        the expanded-weights identity shims normalize.
    """

    conv_picker = _import_module_attr_or_none(
        "torch.nn.utils._expanded_weights.conv_utils", "conv_picker"
    )
    expanded_weight = _import_module_attr_or_none(
        "torch.nn.utils._expanded_weights.expanded_weights_impl", "ExpandedWeight"
    )
    return (
        conv_picker is not None
        and expanded_weight is not None
        and "__torch_function__" in vars(expanded_weight)
    )


def _probe_gradient_edge() -> bool:
    """Return whether torch's GradientEdge surface actually WORKS end to end.

    Returns
    -------
    bool
        ``True`` when ``torch.autograd.graph`` exposes both ``GradientEdge``
        and ``get_gradient_edge`` -- the ``(node, output slot)`` addressing
        pair the one-backward read engine seeds ``autograd.grad`` with -- AND
        a real ``autograd.grad`` call seeding an OUTPUT ``GradientEdge`` with
        an explicit cotangent succeeds. Attribute presence alone is
        insufficient: torch 2.2-2.3 ships the ``GradientEdge`` class, but its
        own internal ``torch.autograd._make_grads`` calls ``out.is_nested`` on
        every output -- including a bare ``GradientEdge``, which has no such
        attribute -- and raises ``AttributeError`` (fixed upstream by 2.4,
        matching this read's long-documented "2.4+" remedy text). Absent on
        older torch or broken on this intermediate band is a healthy old
        install either way, not a TorchLens degradation: the read refuses
        typed.
    """

    if (
        _import_module_attr_or_none("torch.autograd.graph", "GradientEdge") is None
        or _import_module_attr_or_none("torch.autograd.graph", "get_gradient_edge") is None
    ):
        return False
    from torch.autograd.graph import GradientEdge

    try:
        x = torch.randn(1, requires_grad=True)
        h = x * 2
        if h.grad_fn is None:
            # Never happens for a requires_grad leaf's product, but the
            # static annotation is ``Node | None``; treat an impossible-in-
            # practice None the same as any other probe failure below.
            return False
        torch.autograd.grad(
            [GradientEdge(h.grad_fn, h.output_nr)],
            [x],
            grad_outputs=[torch.ones(1)],
        )
    except Exception:
        return False
    return True


def get_gradient_edge_support(*, force_probe: bool = False) -> bool:
    """Return whether one-backward reads can rely on GradientEdge, lazily.

    Parameters
    ----------
    force_probe:
        Run the (real, one-op autograd) probe even on first call. Diagnostic
        surfaces set this to report the real build capability; the
        one-backward read engine (the only product consumer) calls this
        without it, so a plain ``import torchlens`` with no one-backward read
        never pays for an autograd engine invocation.

    Returns
    -------
    bool
        ``True`` once a real ``autograd.grad`` call seeding an output
        ``GradientEdge`` with an explicit cotangent has been verified to
        succeed on this process's torch build (see :func:`_probe_gradient_edge`
        for the torch 2.2-2.3 trap this guards against). Cached after the
        first call (``force_probe`` included).

    Notes
    -----
    Lazy by design: the first invocation of ``torch.autograd.grad`` in a
    process can pay a one-time autograd-engine-thread-pool initialization
    cost measured in tens of milliseconds, which must land on the first real
    one-backward read, never on a plain ``import torchlens`` with no
    autograd.grad call anywhere in the session (the import-hygiene budget).
    """

    global HAS_GRADIENT_EDGE, _GRADIENT_EDGE_PROBED

    if not _GRADIENT_EDGE_PROBED or force_probe:
        HAS_GRADIENT_EDGE = _probe_gradient_edge()
        _GRADIENT_EDGE_PROBED = True
    return HAS_GRADIENT_EDGE


def get_reduce_tuple_dim_support(*, force_probe: bool = False) -> bool:
    """Return whether ``Tensor.any``/``Tensor.all`` accept a tuple ``dim``, lazily.

    Parameters
    ----------
    force_probe:
        Run the probe even on first call. Diagnostic surfaces set this;
        :func:`tensor_any_over_dims` (the only product consumer) does not, so
        a plain ``import torchlens`` never pays for the tensor op.

    Returns
    -------
    bool
        See :func:`_probe_reduce_tuple_dim`. Cached after the first call.
    """

    global HAS_REDUCE_TUPLE_DIM, _REDUCE_TUPLE_DIM_PROBED

    if not _REDUCE_TUPLE_DIM_PROBED or force_probe:
        HAS_REDUCE_TUPLE_DIM = _probe_reduce_tuple_dim()
        _REDUCE_TUPLE_DIM_PROBED = True
    return HAS_REDUCE_TUPLE_DIM


def get_cpu_half_kernels_support(*, force_probe: bool = False) -> bool:
    """Return whether CPU addmm/layer_norm/nextafter accept float16, lazily.

    Parameters
    ----------
    force_probe:
        Run the probe even on first call. Diagnostic surfaces set this; no
        product code consumes this flag today (test-only), so a plain
        ``import torchlens`` never pays for the three tensor ops.

    Returns
    -------
    bool
        See :func:`_probe_cpu_half_kernels`. Cached after the first call.
    """

    global HAS_CPU_HALF_KERNELS, _CPU_HALF_KERNELS_PROBED

    if not _CPU_HALF_KERNELS_PROBED or force_probe:
        HAS_CPU_HALF_KERNELS = _probe_cpu_half_kernels()
        _CPU_HALF_KERNELS_PROBED = True
    return HAS_CPU_HALF_KERNELS


def get_cpu_float8_deterministic_fill_support(*, force_probe: bool = False) -> bool:
    """Return whether CPU empty-fill covers Float8 under determinism, lazily.

    Parameters
    ----------
    force_probe:
        Run the probe even on first call. Diagnostic surfaces set this; no
        product code consumes this flag today (test-only), so a plain
        ``import torchlens`` never pays for the transient global-determinism
        toggle and allocation.

    Returns
    -------
    bool
        See :func:`_probe_cpu_float8_deterministic_fill`. Cached after the
        first call.
    """

    global HAS_CPU_FLOAT8_DETERMINISTIC_FILL, _CPU_FLOAT8_DETERMINISTIC_FILL_PROBED

    if not _CPU_FLOAT8_DETERMINISTIC_FILL_PROBED or force_probe:
        HAS_CPU_FLOAT8_DETERMINISTIC_FILL = _probe_cpu_float8_deterministic_fill()
        _CPU_FLOAT8_DETERMINISTIC_FILL_PROBED = True
    return HAS_CPU_FLOAT8_DETERMINISTIC_FILL


def get_meta_item_guard_support(*, force_probe: bool = False) -> bool:
    """Return whether meta-tensor scalar extraction raises torch's own guard, lazily.

    Parameters
    ----------
    force_probe:
        Run the probe even on first call. Diagnostic surfaces set this; no
        product code consumes this flag today (test-only), so a plain
        ``import torchlens`` never pays for the transient meta-tensor
        allocation and ``.item()`` call.

    Returns
    -------
    bool
        See :func:`_probe_meta_item_guard`. Cached after the first call.
    """

    global HAS_META_ITEM_GUARD, _META_ITEM_GUARD_PROBED

    if not _META_ITEM_GUARD_PROBED or force_probe:
        HAS_META_ITEM_GUARD = _probe_meta_item_guard()
        _META_ITEM_GUARD_PROBED = True
    return HAS_META_ITEM_GUARD


def get_mha_fastpath_switch_support(*, force_probe: bool = False) -> bool:
    """Return whether ``torch.backends.mha`` exposes the fastpath switch, lazily.

    Parameters
    ----------
    force_probe:
        Run the probe even on first call. Diagnostic surfaces set this;
        :func:`get_mha_fastpath_enabled`, :func:`set_mha_fastpath_enabled`, and
        :func:`force_mha_slow_path` (the only product consumers) do not, so a
        plain ``import torchlens`` never pays for the ``find_spec`` /
        submodule-attribute lookup against ``torch.backends.mha``.

    Returns
    -------
    bool
        See :func:`_probe_mha_fastpath_switch`. Cached after the first call.
    """

    global HAS_MHA_FASTPATH_SWITCH, _MHA_FASTPATH_SWITCH_PROBED

    if not _MHA_FASTPATH_SWITCH_PROBED or force_probe:
        HAS_MHA_FASTPATH_SWITCH = _probe_mha_fastpath_switch()
        _MHA_FASTPATH_SWITCH_PROBED = True
    return HAS_MHA_FASTPATH_SWITCH


def _probe_node_prehook() -> bool:
    """Return whether autograd graph nodes support ``register_prehook``.

    Returns
    -------
    bool
        ``True`` when ``torch.autograd.graph.Node`` declares
        ``register_prehook`` -- the slot-targeted freeze mechanism behind the
        one-backward read's ``frozen=`` policy (M(reads) D6).
    """

    node_cls = _import_module_attr_or_none("torch.autograd.graph", "Node")
    return node_cls is not None and hasattr(node_cls, "register_prehook")


def _probe_fp32_precision_controls() -> bool:
    """Return whether torch exposes the ``fp32_precision`` backend controls.

    Returns
    -------
    bool
        ``True`` on torch >= 2.9, where ``torch.backends.fp32_precision`` and
        its per-backend children exist and the legacy
        ``set_float32_matmul_precision`` / ``allow_tf32`` setters write them.
        Absent on older torch, which is a healthy install: there is no
        second representation to leak, so nothing degrades.
    """

    return hasattr(torch.backends, "fp32_precision")


def _probe_deterministic_fill_flag() -> bool:
    """Return whether ``torch.utils.deterministic.fill_uninitialized_memory`` exists.

    Returns
    -------
    bool
        ``True`` when the ``torch.utils.deterministic`` submodule is
        discoverable (``find_spec``, never imported eagerly here to avoid
        paying for a module nothing else needs). Absent on torch 2.1.x (the
        submodule postdates the 2.1 floor), which is a healthy old install,
        not a degradation: :func:`read_fill_uninitialized_memory` returns
        ``None`` and :func:`write_fill_uninitialized_memory` no-ops.
    """

    try:
        return importlib.util.find_spec("torch.utils.deterministic") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _probe_amp_gradscaler() -> bool:
    """Return whether the device-agnostic ``torch.amp.GradScaler`` exists.

    Returns
    -------
    bool
        ``True`` when ``torch.amp`` exposes ``GradScaler`` directly (the
        device-agnostic constructor; torch 2.1-2.2 only ship the CUDA-specific
        ``torch.cuda.amp.GradScaler``). Absence is a healthy old install, not
        a degradation -- TorchLens product code never constructs a scaler.
    """

    return _import_module_attr_or_none("torch.amp", "GradScaler") is not None


def _probe_nn_attention_module() -> bool:
    """Return whether ``torch.nn.attention`` (``SDPBackend``/``sdpa_kernel``) exists.

    Returns
    -------
    bool
        ``True`` when the ``torch.nn.attention`` namespace exposes
        ``SDPBackend`` and ``sdpa_kernel``. The parent package body is
        dynamo-free (unlike ``torch.nn.attention.bias``, see
        :func:`_probe_attention_causal_bias`), so it is safe to import
        directly. Absent on torch 2.1-2.2 (the namespace postdates the 2.1
        floor), a healthy old install, not a degradation.

    Notes
    -----
    The absent case checks with ``find_spec`` FIRST (proves non-existence
    without executing anything) before ever calling
    ``importlib.import_module``: two full failed-import attempts (one per
    attribute) measurably inflated ``import torchlens`` on the torch
    2.1/2.2 floor, where the namespace genuinely does not exist, well past
    this module's own import-hygiene budget (``test_import_hygiene.py``).
    """

    try:
        if importlib.util.find_spec("torch.nn.attention") is None:
            return False
    except (ImportError, AttributeError, ValueError):
        return False
    return (
        _import_module_attr_or_none("torch.nn.attention", "SDPBackend") is not None
        and _import_module_attr_or_none("torch.nn.attention", "sdpa_kernel") is not None
    )


def _probe_rmsnorm_module() -> bool:
    """Return whether ``torch.nn.RMSNorm`` exists.

    Returns
    -------
    bool
        ``True`` when ``torch.nn`` exposes the built-in ``RMSNorm`` module
        (added torch 2.4). Absence is a healthy old install, not a
        degradation -- TorchLens classifies RMSNorm-family modules by class
        name, never by constructing ``torch.nn.RMSNorm`` itself.
    """

    return getattr(torch.nn, "RMSNorm", None) is not None


def _probe_mha_fastpath_switch() -> bool:
    """Return whether ``torch.backends.mha`` exposes the public fastpath switch.

    Returns
    -------
    bool
        ``True`` when ``torch.backends.mha`` exposes both
        ``get_fastpath_enabled`` and ``set_fastpath_enabled``. Absent on
        torch 2.1-2.2 (the submodule postdates the 2.1 floor): a healthy old
        install, not a degradation -- the fused ``nn.MultiheadAttention`` /
        ``nn.TransformerEncoderLayer`` fast path has no public global switch
        there, so :func:`force_mha_slow_path` falls back to flipping the
        affected modules' own ``training`` flag (both fast paths check
        ``self.training`` on every supported torch, switch or no switch).
    """

    try:
        if importlib.util.find_spec("torch.backends.mha") is None:
            return False
    except (ImportError, AttributeError, ValueError):
        return False
    return (
        _import_module_attr_or_none("torch.backends.mha", "get_fastpath_enabled") is not None
        and _import_module_attr_or_none("torch.backends.mha", "set_fastpath_enabled") is not None
    )


def _probe_cpu_half_kernels() -> bool:
    """Return whether common CPU kernels accept the ``torch.float16`` dtype.

    Returns
    -------
    bool
        ``True`` when CPU ``addmm``, ``layer_norm``, ``nextafter``, and
        ``aminmax`` all accept float16 operands. torch 2.1-2.2's CPU backend
        is missing these kernels for Half tensors (``addmm_impl_cpu_``,
        ``LayerNormKernelImpl``, ``nextafter_cpu``, and ``aminmax_cpu`` all
        raise ``"... not implemented for 'Half'"``); later torch ships them.
        Absence is a genuine torch CPU limitation on the floor, not a
        TorchLens degradation -- tests that exercise a half-precision CPU
        forward, or TorchLens's own ULP-step validation machinery
        (``torch.nextafter`` in ``validation/core.py`` and
        ``validation/exemptions.py``) or dense tensor-stats kernel
        (``torch.aminmax`` in ``stats/_stats_kernel.py``) on a Half-dtype
        output, skip on it.
    """

    try:
        half = torch.randn(2, 2, dtype=torch.float16)
        torch.addmm(half, half, half)
        torch.nn.functional.layer_norm(half, (2,))
        torch.nextafter(half, half)
        torch.aminmax(half)
    except RuntimeError:
        return False
    return True


def _probe_cpu_float8_deterministic_fill() -> bool:
    """Return whether CPU tensor allocation fills Float8 dtypes under determinism.

    Returns
    -------
    bool
        ``True`` when ``torch.empty(dtype=torch.float8_e4m3fn)`` succeeds with
        ``torch.use_deterministic_algorithms(True)`` active. torch 2.1-2.2's
        ``fill_empty_deterministic_`` CPU kernel does not cover Float8 dtypes,
        so any empty-tensor allocation under deterministic mode (TorchLens's
        own dtype-cast path included) raises
        ``RuntimeError: "fill_empty_deterministic_" not implemented for
        'Float8_e4m3fn'``. Absence is a genuine torch CPU limitation on the
        floor, not a TorchLens degradation.
    """

    float8_dtype = getattr(torch, "float8_e4m3fn", None)
    if float8_dtype is None:
        return False
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        torch.empty((1,), dtype=float8_dtype)
    except RuntimeError:
        return False
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
    return True


def _probe_reduce_tuple_dim() -> bool:
    """Return whether ``Tensor.any``/``Tensor.all`` accept a tuple ``dim``.

    Returns
    -------
    bool
        ``True`` when a boolean tensor's ``.any(dim=(0, 1))`` accepts a tuple
        of axes directly (added torch 2.2; torch 2.1 only accepts a single
        int). Absence is a healthy old install: :func:`tensor_any_over_dims`
        falls back to sequential single-axis reduction.
    """

    probe = torch.zeros((2, 2), dtype=torch.bool)
    try:
        probe.any(dim=(0, 1))
    except TypeError:
        return False
    return True


def _probe_meta_item_guard() -> bool:
    """Return whether ``Tensor.item()`` raises its own guard on meta tensors.

    Returns
    -------
    bool
        ``True`` when calling ``.item()`` (or any other scalar extraction,
        e.g. ``bool()``) on a meta tensor raises a plain ``RuntimeError``
        carrying torch's own "Tensor.item() cannot be called on meta
        tensors" guard. torch 2.1-2.2 have no such guard: the call instead
        falls all the way through to the aten dispatcher's generic
        ``NotImplementedError`` ("... not implemented for this backend").
        TorchLens's structure-only forward-boundary backstop
        (``backends/torch/structure_only_belt.py``) classifies a caught
        ``NotImplementedError`` as ``meta_kernel_unavailable`` and a generic
        ``RuntimeError`` as ``value_dependent_branch_unsupported`` by
        raising-frame PROVENANCE, never message text -- correctly, since
        the two exception FAMILIES genuinely differ here. Absence is a
        healthy old install surfacing the honest alternate typed refusal,
        not a TorchLens degradation.
    """

    try:
        torch.ones((), device="meta").item()
    except NotImplementedError:
        return False
    except RuntimeError:
        return True
    return False


HAS_VARIABLE_FUNCTIONS: bool = _probe_variable_functions()
HAS_TORCH_VF: bool = _probe_torch_vf()
HAS_TORCH_FUNC: bool = _probe_torch_func()
HAS_FUNCTORCH_APIS: bool = _probe_functorch_apis()
HAS_FUNCTORCH_LEVEL_API: bool = _probe_functorch_level_api()
HAS_FUNCTORCH_WRAPPED_TENSOR_API: bool = _probe_functorch_wrapped_tensor_api()
HAS_JIT_BUILTIN_TABLE: bool = _probe_jit_builtin_table()
HAS_JIT_BOOLEAN_DISPATCH_TABLE: bool = _probe_jit_boolean_dispatch_table()
HAS_JIT_OVERLOAD_RESOLVER: bool = _probe_jit_overload_resolver()
HAS_DEVICE_CONTEXT_DISPATCH: bool = _probe_device_context_dispatch()
HAS_DEVICE_CONSTRUCTORS: bool = _probe_device_constructors()
HAS_ACCUMULATE_GRAD_CLASS: bool = _probe_accumulate_grad_class()
HAS_CURRENT_GRAPH_TASK_ID: bool = _probe_current_graph_task_id()
HAS_FX_GRAPH_MODULE: bool = _probe_fx_graph_module()
HAS_NAMED_TENSOR_API: bool = _probe_named_tensor_api()
HAS_CACHED_UNTYPED_STORAGE_WRAPPER: bool = _probe_cached_untyped_storage_wrapper()
HAS_DYNAMO_OPTIMIZED_MODULE: bool = False
HAS_DYNAMO_ORIG_CALLABLE_MARKER: bool = False
HAS_DYNAMO_EXPLAIN: bool = _probe_dynamo_explain_module()
HAS_GENERATOR_CLONE_STATE: bool = hasattr(torch.Generator, "clone_state")
HAS_GENERATOR_GRAPHSAFE_GET_STATE: bool = hasattr(torch.Generator, "graphsafe_get_state")
HAS_GENERATOR_GRAPHSAFE_SET_STATE: bool = hasattr(torch.Generator, "graphsafe_set_state")
HAS_GENERATOR_PHILOX_STATE: bool = hasattr(torch.Generator, "philox_state")
HAS_SAFE_WEIGHTS_ONLY_LOAD: bool = _probe_safe_weights_only_load()
HAS_TENSOR_SEQUENCE_SLOT_FIX: bool = _probe_tensor_sequence_slot_fix()
# Legacy ``torch.<dtype>Tensor`` constructor capture (``backends/torch/legacy_ctors.py``).
HAS_LEGACY_CONSTRUCTOR_NEW_PATCH: bool = _probe_legacy_constructor_new_patch()
HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE: bool = _probe_parameter_as_subclass_in_dispatch_mode()
HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE: bool = _probe_subclass_ctor_in_dispatch_mode()
# r-b4 R26-5a: ROLL_TENSOR_SHIFTS_SUPPORTED is a TEST HELPER, not a published
# capability. It tracks a user-side torch spelling limitation (torch.roll with a
# bare 0-dim tensor `shifts`: 2.8 rejects, 2.13 accepts) on which TorchLens does
# NOT degrade -- zero library consumers exist, so publishing it made doctor WARN
# across most of the declared support range for nothing TorchLens does.
ROLL_TENSOR_SHIFTS_SUPPORTED: bool = _probe_roll_tensor_shifts()
HAS_SAVED_TENSORS_HOOK_INTROSPECTION: bool = _probe_saved_tensors_hook_introspection()
HAS_SAVED_TENSORS_HOOKS_PATCHABLE: bool = _probe_saved_tensors_hooks_patchable()
HAS_CODE_POSITIONS: bool = _probe_code_positions()
HAS_CODE_QUALNAME: bool = _probe_code_qualname()
HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG: bool = _probe_transformer_activation_fastpath_flag()
HAS_ATTENTION_CAUSAL_BIAS: bool = _probe_attention_causal_bias()
HAS_EXPANDED_WEIGHTS_CONV_PICKER: bool = _probe_expanded_weights_conv_picker()
HAS_GRADIENT_EDGE: bool = False
_GRADIENT_EDGE_PROBED: bool = False
HAS_NODE_PREHOOK: bool = _probe_node_prehook()
HAS_DETERMINISTIC_FILL_FLAG: bool = _probe_deterministic_fill_flag()
HAS_FP32_PRECISION_CONTROLS: bool = _probe_fp32_precision_controls()
HAS_AMP_GRADSCALER: bool = _probe_amp_gradscaler()
HAS_NN_ATTENTION_MODULE: bool = _probe_nn_attention_module()
HAS_RMSNORM_MODULE: bool = _probe_rmsnorm_module()
HAS_MHA_FASTPATH_SWITCH: bool = False
_MHA_FASTPATH_SWITCH_PROBED: bool = False
HAS_REDUCE_TUPLE_DIM: bool = False
_REDUCE_TUPLE_DIM_PROBED: bool = False
HAS_CPU_HALF_KERNELS: bool = False
_CPU_HALF_KERNELS_PROBED: bool = False
HAS_CPU_FLOAT8_DETERMINISTIC_FILL: bool = False
_CPU_FLOAT8_DETERMINISTIC_FILL_PROBED: bool = False
HAS_META_ITEM_GUARD: bool = False
_META_ITEM_GUARD_PROBED: bool = False
_DYNAMO_OPTIMIZED_MODULE_TYPE: type[Any] | None = None
_DYNAMO_OPTIMIZED_MODULE_PROBED: bool = False
_DYNAMO_ORIG_CALLABLE_MARKER_PROBED: bool = False
HAS_FSDP_WRAPPER: bool = False
_FSDP_WRAPPER_TYPE: type[Any] | None = None
_FSDP_WRAPPER_PROBED: bool = False
HAS_DTENSOR: bool = False
_DTENSOR_TYPE: type[Any] | None = None
_DTENSOR_PROBED: bool = False
HAS_DEVICE_MESH: bool = False
_DEVICE_MESH_TYPE: type[Any] | None = None
_DEVICE_MESH_PROBED: bool = False
HAS_PIPELINING: bool = False
_PIPELINING_TYPES: tuple[type[Any], ...] = ()
_PIPELINING_PROBED: bool = False
HAS_DYNAMO_IS_COMPILING: bool = False
_DYNAMO_IS_COMPILING_FN: Callable[[], bool] | None = None
_DYNAMO_IS_COMPILING_PROBED: bool = False
HAS_SET_STANCE: bool = _probe_set_stance()
HAS_TORCH_FUNCTION_STACK_SURGERY: bool = _probe_torch_function_stack_surgery()
HAS_DYNAMO_COMPILE_COUNTERS: bool = False
_DYNAMO_COMPILE_COUNTERS: Any | None = None
_DYNAMO_COMPILE_COUNTERS_PROBED: bool = False
HAS_TRACING_TENSOR_TYPES: bool = False
_TRACING_TENSOR_TYPES: tuple[type[Any], ...] = ()
_TRACING_TENSOR_TYPES_PROBED: bool = False
HAS_FP8_DTYPES: bool = False
_FP8_DTYPES: frozenset[Any] = frozenset()
_FP8_DTYPES_PROBED: bool = False
HAS_C10D_GROUP_REGISTRY: bool = False
_C10D_GROUP_REGISTRY_PROBED: bool = False
HAS_C10D_GROUP_SEQ: bool = False
_C10D_GROUP_SEQ_PROBED: bool = False
HAS_C10D_ABORT_PG: bool = False
_C10D_ABORT_PG_PROBED: bool = False
HAS_DISPATCH_MODE_STACK_QUERY: bool = False
_DISPATCH_MODE_STACK_FN: Callable[[], Any] | None = None
_DISPATCH_MODE_STACK_PROBED: bool = False
# r-b4 R26-2: previously-unrouted private probes, each with a named flag.
HAS_JIT_SCHEMA_ENUMERATION: bool = False
_JIT_SCHEMA_ENUMERATION_FN: Callable[[], Any] | None = None
_JIT_SCHEMA_ENUMERATION_PROBED: bool = False
HAS_TENSORBASE_CLASS: bool = False
_TENSORBASE_CLASS: type[Any] | None = None
_TENSORBASE_CLASS_PROBED: bool = False
HAS_DISABLE_TORCH_FUNCTION: bool = False
_DISABLE_TORCH_FUNCTION_CLS: Any | None = None
_DISABLE_TORCH_FUNCTION_PROBED: bool = False
HAS_VARIABLE_FUNCTIONS_CLASS: bool = False
_VARIABLE_FUNCTIONS_CLASS: Any | None = None
_VARIABLE_FUNCTIONS_CLASS_PROBED: bool = False
HAS_FAKE_TENSOR_MODE: bool = False
_FAKE_TENSOR_MODE_CLS: type[Any] | None = None
_FAKE_TENSOR_MODE_PROBED: bool = False
HAS_DTENSOR_SHARD_GEOMETRY: bool = False
_DTENSOR_SHARD_GEOMETRY_FN: Callable[..., Any] | None = None
_DTENSOR_SHARD_GEOMETRY_PROBED: bool = False
HAS_FUNCOL_GROUP_RESOLUTION: bool = False
_FUNCOL_GROUP_RESOLVERS: tuple[Callable[..., Any], Callable[..., Any]] | None = None
_FUNCOL_GROUP_RESOLUTION_PROBED: bool = False
HAS_FUNCOL_WAIT_INTERPOSITION: bool = False
_FUNCOL_WAIT_INTERPOSITION_PROBED: bool = False
_FUNCOL_WAIT_REDISPATCH: tuple[Any, Callable[[], Any]] | None = None
HAS_FUNCOL_MODULE: bool = False
_FUNCOL_MODULE_OBJ: Any | None = None
_FUNCOL_MODULE_PROBED: bool = False
HAS_ASYNC_COLLECTIVE_TENSOR: bool = False
_ASYNC_COLLECTIVE_TENSOR_TYPE: type[Any] | None = None
_ASYNC_COLLECTIVE_TENSOR_PROBED: bool = False
HAS_CHECKPOINT_HOOK_CLASS: bool = False
_CHECKPOINT_HOOK_CLASS: type[Any] | None = None
_CHECKPOINT_HOOK_CLASS_PROBED: bool = False
#: torch >= 2.14 interposes ``_checkpoint_internal_hook`` between
#: ``_checkpoint_hook``/``_recomputation_hook`` and ``saved_tensors_hooks``; its
#: ``__enter__``/``__exit__`` stash private cross-call identity state
#: (``_user_hooks``) directly on the pack-hook callable. Absence (torch 2.13 and
#: earlier) means the checkpoint-token classifier's hook-object replacement needs
#: no attribute preservation; presence means it must carry foreign attributes
#: forward or skip the replacement (see ``backward.py::_carry_foreign_hook_attrs``).
HAS_CHECKPOINT_INTERNAL_HOOK_CLASS: bool = False
_CHECKPOINT_INTERNAL_HOOK_CLASS: type[Any] | None = None
_CHECKPOINT_INTERNAL_HOOK_CLASS_PROBED: bool = False
HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK: bool = False
_AUTOGRAD_ENGINE_QUEUE_CALLBACK: Callable[..., Any] | None = None
_AUTOGRAD_ENGINE_QUEUE_CALLBACK_PROBED: bool = False
# torchnative W0.12/W2.1: the in-memory Kineto event field contract behind the
# device-time join's extraction adapter. Probed lazily at first extraction (a
# live profiler session is required to observe an event instance).
HAS_KINETO_INMEMORY_EVENTS: bool = False
_KINETO_INMEMORY_EVENTS_PROBED: bool = False
HAS_KINETO_EVENT_SCOPE: bool = False
_KINETO_EVENT_SCOPE_PROBED: bool = False
# torchnative W0.7 (test-oracle exception to E3): torch's still-alive private
# categorized memory profiler, consumed ONLY by the memory-timeline parity
# oracle while upstream keeps it. Probed lazily at first use.
HAS_MEMORY_PROFILE: bool = False
_MEMORY_PROFILE_PROBED: bool = False

#: The exact per-event field contract (torchnative W0.12) the extraction
#: adapter consumes: identity, integer-nanosecond interval, runtime
#: correlation, thread id, device coordinates, the user-annotation flag, and
#: the typed activity classification. ``scope`` is probed separately
#: (HAS_KINETO_EVENT_SCOPE) because its absence degrades only the backward
#: scope-1 body witness, never the join.
KINETO_EVENT_FIELD_CONTRACT: tuple[str, ...] = (
    "name",
    "start_ns",
    "duration_ns",
    "correlation_id",
    "start_thread_id",
    "device_type",
    "device_index",
    "is_user_annotation",
    "activity_type",
)

_CAPABILITY_ATTRS: tuple[str, ...] = (
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
    "HAS_CURRENT_GRAPH_TASK_ID",
    "HAS_FX_GRAPH_MODULE",
    "HAS_NAMED_TENSOR_API",
    "HAS_CACHED_UNTYPED_STORAGE_WRAPPER",
    "HAS_DYNAMO_OPTIMIZED_MODULE",
    "HAS_DYNAMO_ORIG_CALLABLE_MARKER",
    "HAS_DYNAMO_EXPLAIN",
    "HAS_FSDP_WRAPPER",
    "HAS_C10D_ABORT_PG",
    "HAS_C10D_GROUP_REGISTRY",
    "HAS_C10D_GROUP_SEQ",
    "HAS_DTENSOR",
    "HAS_DEVICE_MESH",
    "HAS_PIPELINING",
    "HAS_DISPATCH_MODE_STACK_QUERY",
    "HAS_JIT_SCHEMA_ENUMERATION",
    "HAS_TENSORBASE_CLASS",
    "HAS_DISABLE_TORCH_FUNCTION",
    "HAS_VARIABLE_FUNCTIONS_CLASS",
    "HAS_FAKE_TENSOR_MODE",
    "HAS_DTENSOR_SHARD_GEOMETRY",
    "HAS_FUNCOL_GROUP_RESOLUTION",
    "HAS_FUNCOL_WAIT_INTERPOSITION",
    "HAS_FUNCOL_MODULE",
    "HAS_ASYNC_COLLECTIVE_TENSOR",
    "HAS_CHECKPOINT_HOOK_CLASS",
    "HAS_CHECKPOINT_INTERNAL_HOOK_CLASS",
    "HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK",
    "HAS_KINETO_INMEMORY_EVENTS",
    "HAS_KINETO_EVENT_SCOPE",
    "HAS_MEMORY_PROFILE",
    "HAS_DYNAMO_IS_COMPILING",
    "HAS_SET_STANCE",
    "HAS_TORCH_FUNCTION_STACK_SURGERY",
    "HAS_DYNAMO_COMPILE_COUNTERS",
    "HAS_TRACING_TENSOR_TYPES",
    "HAS_FP8_DTYPES",
    "HAS_GENERATOR_CLONE_STATE",
    "HAS_GENERATOR_GRAPHSAFE_GET_STATE",
    "HAS_GENERATOR_GRAPHSAFE_SET_STATE",
    "HAS_GENERATOR_PHILOX_STATE",
    "HAS_SAFE_WEIGHTS_ONLY_LOAD",
    "HAS_TENSOR_SEQUENCE_SLOT_FIX",
    "HAS_LEGACY_CONSTRUCTOR_NEW_PATCH",
    "HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE",
    "HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE",
    "HAS_SAVED_TENSORS_HOOK_INTROSPECTION",
    "HAS_SAVED_TENSORS_HOOKS_PATCHABLE",
    "HAS_CODE_POSITIONS",
    "HAS_CODE_QUALNAME",
    "HAS_TRANSFORMER_ACTIVATION_FASTPATH_FLAG",
    "HAS_ATTENTION_CAUSAL_BIAS",
    "HAS_EXPANDED_WEIGHTS_CONV_PICKER",
    "HAS_GRADIENT_EDGE",
    "HAS_NODE_PREHOOK",
    "HAS_DETERMINISTIC_FILL_FLAG",
    "HAS_FP32_PRECISION_CONTROLS",
    "HAS_AMP_GRADSCALER",
    "HAS_NN_ATTENTION_MODULE",
    "HAS_RMSNORM_MODULE",
    "HAS_MHA_FASTPATH_SWITCH",
    "HAS_REDUCE_TUPLE_DIM",
    "HAS_CPU_HALF_KERNELS",
    "HAS_CPU_FLOAT8_DETERMINISTIC_FILL",
    "HAS_META_ITEM_GUARD",
)


# r-b7 R52-4: O(1) membership for mark_torch_capability_missing; the tuple above
# stays the ORDERED iteration authority (snapshot/report key order is stable).
_CAPABILITY_ATTR_SET: frozenset[str] = frozenset(_CAPABILITY_ATTRS)


_LAZY_PROBE_FAMILIES: dict[str, tuple[str, ...]] = {
    "_C10D_ABORT_PG_PROBED": ("HAS_C10D_ABORT_PG",),
    "_C10D_GROUP_REGISTRY_PROBED": ("HAS_C10D_GROUP_REGISTRY",),
    "_C10D_GROUP_SEQ_PROBED": ("HAS_C10D_GROUP_SEQ",),
    "_DEVICE_MESH_PROBED": ("HAS_DEVICE_MESH", "_DEVICE_MESH_TYPE"),
    "_DISABLE_TORCH_FUNCTION_PROBED": (
        "HAS_DISABLE_TORCH_FUNCTION",
        "_DISABLE_TORCH_FUNCTION_CLS",
    ),
    "_DISPATCH_MODE_STACK_PROBED": (
        "HAS_DISPATCH_MODE_STACK_QUERY",
        "_DISPATCH_MODE_STACK_FN",
    ),
    "_DTENSOR_PROBED": ("HAS_DTENSOR", "_DTENSOR_TYPE"),
    "_GRADIENT_EDGE_PROBED": ("HAS_GRADIENT_EDGE",),
    "_REDUCE_TUPLE_DIM_PROBED": ("HAS_REDUCE_TUPLE_DIM",),
    "_CPU_HALF_KERNELS_PROBED": ("HAS_CPU_HALF_KERNELS",),
    "_CPU_FLOAT8_DETERMINISTIC_FILL_PROBED": ("HAS_CPU_FLOAT8_DETERMINISTIC_FILL",),
    "_META_ITEM_GUARD_PROBED": ("HAS_META_ITEM_GUARD",),
    "_MHA_FASTPATH_SWITCH_PROBED": ("HAS_MHA_FASTPATH_SWITCH",),
    "_FUNCOL_GROUP_RESOLUTION_PROBED": (
        "HAS_FUNCOL_GROUP_RESOLUTION",
        "_FUNCOL_GROUP_RESOLVERS",
    ),
    "_FUNCOL_WAIT_INTERPOSITION_PROBED": (
        "HAS_FUNCOL_WAIT_INTERPOSITION",
        "_FUNCOL_WAIT_REDISPATCH",
    ),
    "_FUNCOL_MODULE_PROBED": ("HAS_FUNCOL_MODULE", "_FUNCOL_MODULE_OBJ"),
    "_ASYNC_COLLECTIVE_TENSOR_PROBED": (
        "HAS_ASYNC_COLLECTIVE_TENSOR",
        "_ASYNC_COLLECTIVE_TENSOR_TYPE",
    ),
    "_CHECKPOINT_HOOK_CLASS_PROBED": (
        "HAS_CHECKPOINT_HOOK_CLASS",
        "_CHECKPOINT_HOOK_CLASS",
    ),
    "_CHECKPOINT_INTERNAL_HOOK_CLASS_PROBED": (
        "HAS_CHECKPOINT_INTERNAL_HOOK_CLASS",
        "_CHECKPOINT_INTERNAL_HOOK_CLASS",
    ),
    "_AUTOGRAD_ENGINE_QUEUE_CALLBACK_PROBED": (
        "HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK",
        "_AUTOGRAD_ENGINE_QUEUE_CALLBACK",
    ),
    "_KINETO_INMEMORY_EVENTS_PROBED": ("HAS_KINETO_INMEMORY_EVENTS",),
    "_KINETO_EVENT_SCOPE_PROBED": ("HAS_KINETO_EVENT_SCOPE",),
    "_MEMORY_PROFILE_PROBED": ("HAS_MEMORY_PROFILE",),
    "_DTENSOR_SHARD_GEOMETRY_PROBED": (
        "HAS_DTENSOR_SHARD_GEOMETRY",
        "_DTENSOR_SHARD_GEOMETRY_FN",
    ),
    "_DYNAMO_COMPILE_COUNTERS_PROBED": (
        "HAS_DYNAMO_COMPILE_COUNTERS",
        "_DYNAMO_COMPILE_COUNTERS",
    ),
    "_DYNAMO_IS_COMPILING_PROBED": ("HAS_DYNAMO_IS_COMPILING", "_DYNAMO_IS_COMPILING_FN"),
    "_DYNAMO_OPTIMIZED_MODULE_PROBED": (
        "HAS_DYNAMO_OPTIMIZED_MODULE",
        "_DYNAMO_OPTIMIZED_MODULE_TYPE",
    ),
    "_DYNAMO_ORIG_CALLABLE_MARKER_PROBED": ("HAS_DYNAMO_ORIG_CALLABLE_MARKER",),
    "_FAKE_TENSOR_MODE_PROBED": ("HAS_FAKE_TENSOR_MODE", "_FAKE_TENSOR_MODE_CLS"),
    "_FP8_DTYPES_PROBED": ("HAS_FP8_DTYPES", "_FP8_DTYPES"),
    "_FSDP_WRAPPER_PROBED": ("HAS_FSDP_WRAPPER", "_FSDP_WRAPPER_TYPE"),
    "_JIT_SCHEMA_ENUMERATION_PROBED": (
        "HAS_JIT_SCHEMA_ENUMERATION",
        "_JIT_SCHEMA_ENUMERATION_FN",
    ),
    "_PIPELINING_PROBED": ("HAS_PIPELINING", "_PIPELINING_TYPES"),
    "_TENSORBASE_CLASS_PROBED": ("HAS_TENSORBASE_CLASS", "_TENSORBASE_CLASS"),
    "_TRACING_TENSOR_TYPES_PROBED": ("HAS_TRACING_TENSOR_TYPES", "_TRACING_TENSOR_TYPES"),
    "_VARIABLE_FUNCTIONS_CLASS_PROBED": (
        "HAS_VARIABLE_FUNCTIONS_CLASS",
        "_VARIABLE_FUNCTIONS_CLASS",
    ),
}
"""Every LAZY capability latch: ``*_PROBED`` flag -> the family attrs it gates.

The import-time ``HAS_*`` probes above run once against the real torch and are
process facts; the LAZY families here latch on FIRST USE, which makes them the
one capability class a test can poison: a probe fired while ``sys.modules`` is
stubbed latches the wrong verdict for the whole process (the recorded
``b7fe953e`` incident -- later compat snapshots and generated-doc gates flap).
``capability_probe_snapshot()`` / ``restore_capability_probes()`` exist so the
test suite can restore the pre-test latch state systemically instead of
per-test by hand; production code never calls them. A meta-test pins this
registry against the module's actual ``*_PROBED`` attrs so a new lazy latch
cannot land outside it.
"""


def capability_probe_snapshot() -> dict[str, object]:
    """Return the current lazy capability-latch state for exact restoration.

    Returns
    -------
    dict[str, object]
        Attribute name -> current value for every ``*_PROBED`` latch, its
        family attrs, and the lazy-import warm flag.
    """

    namespace = globals()
    snapshot: dict[str, object] = {"_LAZY_TORCH_IMPORTS_WARMED": _LAZY_TORCH_IMPORTS_WARMED}
    for probed_attr, family in _LAZY_PROBE_FAMILIES.items():
        snapshot[probed_attr] = namespace[probed_attr]
        for attr in family:
            snapshot[attr] = namespace[attr]
    return snapshot


def restore_capability_probes(snapshot: dict[str, object]) -> None:
    """Restore lazy capability latches captured by :func:`capability_probe_snapshot`.

    Un-poisons the ``b7fe953e`` incident class: a probe that latched under a
    stubbed ``sys.modules`` (or any other transient runtime shim) is reset to
    the snapshotted state, so the next consumer re-probes against the real
    runtime instead of inheriting the poisoned verdict for the process.

    Parameters
    ----------
    snapshot:
        Mapping returned by :func:`capability_probe_snapshot`.

    Returns
    -------
    None
        Module latch state is rebound in place.
    """

    globals().update(snapshot)


class TorchCapabilityWarning(TorchLensWarning):
    """Graceful torch-capability degradation warning (r-b4 R26-6d).

    A dedicated subclass so CI suppression can key on the CATEGORY instead of
    the message text (``pyproject.toml`` matched the literal message string, so
    editing the wording would have turned every legitimate matrix degradation
    into a suite-wide error under ``error::UserWarning:torchlens``). Subclasses
    ``TorchLensWarning`` (itself a ``UserWarning``), so existing category
    filters keep matching and the documented root covers it (R64-1).
    """


OPTIONAL_CAPABILITY_FLAGS: frozenset[str] = frozenset(
    {
        # PEP 657 / 3.11 code-object surfaces: an interpreter feature, not a
        # torch-private degradation -- CPython 3.10 installs are healthy.
        "HAS_CODE_POSITIONS",
        "HAS_CODE_QUALNAME",
        # torch.nn.attention.bias.CausalBias postdates the support floor
        # (namespace added mid-2.x): its absence on older torch is a healthy
        # install with nothing to shim, not a degradation.
        "HAS_ATTENTION_CAUSAL_BIAS",
        # Named-tensor API was REMOVED upstream (torch 2.13): its absence tracks
        # torch's own public surface, so there is nothing for TorchLens to
        # degrade on -- named-dim metadata simply cannot exist on such builds.
        "HAS_NAMED_TENSOR_API",
        # r7 R26 (opus b4 LM): the three ``torch.Generator`` method probes gate
        # rows of the RNG GENERATOR_METHOD_TABLE (utils/rng.py); a ``False``
        # DROPS the row, so there is no method left to monitor and NOTHING
        # degrades. Absent on the older half of the declared torch>=2.1 matrix
        # by upstream version, not by breakage.
        "HAS_GENERATOR_CLONE_STATE",
        "HAS_GENERATOR_GRAPHSAFE_GET_STATE",
        "HAS_GENERATOR_GRAPHSAFE_SET_STATE",
        # torch.Generator.philox_state postdates the torch>=2.1 floor (added
        # torch 2.14): its absence is a healthy older install with no method to
        # drop a row for (same rationale as the three generator flags above).
        "HAS_GENERATOR_PHILOX_STATE",
        # GradientEdge / Node.register_prehook postdate the torch>=2.1 floor:
        # their absence is a healthy old install with nothing to shim -- the
        # one-backward read refuses typed instead of degrading.
        "HAS_GRADIENT_EDGE",
        "HAS_NODE_PREHOOK",
        # _checkpoint_internal_hook (torch.utils.checkpoint) postdates the
        # torch>=2.1 floor (added torch 2.14): its absence on older torch means
        # checkpoint hook objects carry no private cross-call identity state to
        # preserve, not a degradation (backward.py's token-wrapper swap is
        # unconditionally safe there).
        "HAS_CHECKPOINT_INTERNAL_HOOK_CLASS",
        # Deterministic uninit-memory fill, device-agnostic GradScaler,
        # torch.nn.attention, torch.nn.RMSNorm, and tuple-dim any()/all() all
        # postdate the torch>=2.1 floor: their absence is a healthy old
        # install with a real fallback (fill flag) or nothing to shim
        # (TorchLens product code never constructs a GradScaler/RMSNorm or
        # calls sdpa_kernel itself; the tuple-dim reduction falls back to
        # sequential single-axis reduction).
        "HAS_DETERMINISTIC_FILL_FLAG",
        # fp32_precision controls (torch >= 2.9): absent means no second
        # precision representation exists to snapshot, not a degradation.
        "HAS_FP32_PRECISION_CONTROLS",
        "HAS_AMP_GRADSCALER",
        "HAS_NN_ATTENTION_MODULE",
        "HAS_RMSNORM_MODULE",
        "HAS_REDUCE_TUPLE_DIM",
        # torch.backends.mha (the nn.MultiheadAttention / nn.TransformerEncoderLayer
        # fused fast-path switch) postdates the torch>=2.1 floor: its absence is a
        # healthy old install with a real fallback (force_mha_slow_path flips the
        # affected modules' own training flag instead, since both fast paths check
        # self.training on every supported torch).
        "HAS_MHA_FASTPATH_SWITCH",
        # CPU Half-dtype kernel coverage (addmm/layer_norm/nextafter) and
        # Float8 empty-fill under deterministic mode both postdate the torch
        # 2.1 floor: genuine old-torch CPU limitations, not TorchLens
        # degradations -- tests that need them skip on these flags.
        "HAS_CPU_HALF_KERNELS",
        "HAS_CPU_FLOAT8_DETERMINISTIC_FILL",
        # torch's own "Tensor.item() cannot be called on meta tensors" guard
        # postdates the torch 2.1/2.2 floor: its absence means the identical
        # user situation (a value-dependent branch on a meta tensor) still
        # refuses typed, just via the sibling meta_kernel_unavailable code
        # instead of value_dependent_branch_unsupported -- a healthy old
        # install classified by the honest alternate path, not a degradation.
        "HAS_META_ITEM_GUARD",
        # torch.distributed._functional_collectives._resolve_group /
        # torch.distributed.distributed_c10d._resolve_process_group (funcol
        # group resolution, merge-ranks C2 recording) are torch-private
        # surfaces absent on torch 2.7.1 specifically (measured on the
        # nightly fast-tier leg, 2026-10-02) though present on both the
        # declared torch>=2.1 floor's neighbors and the canonical/newest
        # pins: a healthy install with nothing to shim -- captured funcol
        # calls refuse typed instead of recording correlated boundary nodes,
        # the same honest degradation every other optional probe here takes.
        "HAS_FUNCOL_GROUP_RESOLUTION",
        # torch._C._autograd._top_saved_tensors_default_hooks (peeks the
        # innermost installed default saved-tensors pack/unpack hook pair) is
        # probed EAGERLY at import time with no lazy latch, so its absence is
        # never merely "unprobed" -- on a torch build without it,
        # saved_tensors_default_hooks_active() honestly answers None instead
        # of degrading capture-time side-effect analysis; nothing shims it.
        # Absent on torch 2.7.1 specifically (measured on the nightly
        # fast-tier leg, 2026-10-02).
        "HAS_SAVED_TENSORS_HOOK_INTROSPECTION",
    }
)
"""Capability flags whose ``False`` is an absent OPTIONAL feature, not a degradation.

r-b4 R26-4: doctor/compat previously presented every ``False`` flag as a
graceful-degradation WARN, so every healthy torch-only / CPython-3.10 install
showed a permanent warning row with no actionable remedy -- training users to
ignore the row. Flags listed here still appear in the snapshot (with their
true value) but do not drive WARN status; only genuine degradations do.
"""


def mark_torch_capability_missing(capability_name: str, detail: str) -> None:
    """Mark a torch capability absent and emit at most one opt-out warning.

    Setting the ``TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS`` environment
    variable to a closed-vocabulary true spelling (``1``/``true``/``yes``/
    ``on``) suppresses the warning (the flag still flips, and the degradation
    stays visible through ``tl.compat.report()`` / ``tl.utils.doctor()``);
    false spellings and unset keep the warning, and unrecognized values
    refuse (``env_flag_invalid``) -- the historical any-non-empty-value
    truthiness meant ``=false`` SUPPRESSED, the exact silent-inversion class
    the closed_bool_env doctrine exists to kill. This is the knob's only
    documentation-of-record (grind b7 R47-7); keep it in sync with the
    ``_CAPABILITY_WARNING_ENV`` constant above.

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

    if capability_name not in _CAPABILITY_ATTR_SET:
        raise ValueError(f"unknown torch capability flag: {capability_name}")
    globals()[capability_name] = False
    from .env_flags import closed_bool_env

    if closed_bool_env(_CAPABILITY_WARNING_ENV):
        return
    if capability_name in _warned_missing_capabilities:
        return
    _warned_missing_capabilities.add(capability_name)
    warnings.warn(
        f"TorchLens torch capability {capability_name} is unavailable; {detail}",
        TorchCapabilityWarning,
        stacklevel=3,
    )


def get_torch_capability_snapshot() -> TorchCapabilitySnapshot:
    """Return the current torch capability flags as a stable snapshot.

    Returns
    -------
    TorchCapabilitySnapshot
        Mapping from ``HAS_*`` flag name, plus
        ``AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED`` (its own canonical spelling;
        the duplicate ``HAS_`` alias was retired in r-b7 R42-6), to boolean
        availability.
    """

    # HAS_DYNAMO_OPTIMIZED_MODULE, HAS_FSDP_WRAPPER, HAS_DTENSOR,
    # HAS_DEVICE_MESH, and HAS_PIPELINING are lazily probed (see
    # get_dynamo_optimized_module_type / get_fsdp_wrapper_type /
    # get_dtensor_type / get_device_mesh_type / get_pipelining_module_types) to
    # avoid unconditional torch._dynamo / torch.distributed imports on the
    # capture hot path. Diagnostic snapshot consumers (tl.compat.report(),
    # tl.utils.doctor()) are not on that hot path, so force the probes here to
    # report the real capabilities instead of the pre-probe placeholders.
    get_dynamo_optimized_module_type(force_probe=True)
    get_fsdp_wrapper_type(force_probe=True)
    get_dtensor_type(force_probe=True)
    get_gradient_edge_support(force_probe=True)
    get_reduce_tuple_dim_support(force_probe=True)
    get_cpu_half_kernels_support(force_probe=True)
    get_cpu_float8_deterministic_fill_support(force_probe=True)
    get_device_mesh_type(force_probe=True)
    get_pipelining_module_types(force_probe=True)
    get_tracing_tensor_types(force_probe=True)
    get_fp8_dtypes(force_probe=True)
    get_funcol_group_resolvers(force_probe=True)
    get_dynamo_compile_counters(force_probe=True)
    probe_c10d_capabilities(force_probe=True)
    get_current_dispatch_mode_stack()
    # r-b4 R26-2 lazily-probed accessors: resolve so the snapshot reports the
    # real capability, not the pre-probe placeholder. Degradation warnings are
    # deduped once-per-process and the probes are cached.
    get_jit_all_schemas()
    get_tensorbase_class()
    get_disable_torch_function_context()
    get_variable_functions_class()
    get_fake_tensor_mode_class()
    get_dtensor_shard_geometry_fn()
    dynamo_is_compiling()
    _ensure_dynamo_orig_callable_marker_probed()
    get_dynamo_explain()
    # L8/C2 + L9 privately-probed surfaces (fix/private-probe-routing): resolve
    # so the snapshot reports real capabilities. The funcol pair imports
    # torch.distributed like the DTensor/device-mesh forces above; the
    # checkpoint/engine pair is a cheap always-shipped getattr chain.
    get_funcol_module()
    get_async_collective_tensor_type(force_probe=True)
    probe_funcol_wait_interposition()
    get_checkpoint_hook_class()
    get_checkpoint_internal_hook_class()
    get_autograd_engine_queue_callback()
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
    """Return torch variable-function names, or none when the table is absent.

    Returns
    -------
    list[str]
        Names from ``torch._C._VariableFunctions`` when available, otherwise
        an empty list. The historical fallback returned ``torch.__all__`` —
        the WRONG namespace for a variable-function roster (r-b7 R42-2): had
        it ever been taken it would have silently mis-seeded the wrapper
        inventory. An empty roster honestly skips VF-based discovery, and the
        flipped ``HAS_VARIABLE_FUNCTIONS`` flag keeps the degradation visible.
    """

    variable_functions = _nested_getattr_or_none(torch, ("_C", "_VariableFunctions"))
    if variable_functions is None:
        mark_torch_capability_missing(
            "HAS_VARIABLE_FUNCTIONS",
            "skipping torch._C._VariableFunctions-based function discovery",
        )
        return []
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
    submodule = shadowed_torch_submodule(namespace_name, namespace)
    if submodule is not None:  # e.g. the torch.nn.utils.weight_norm module
        return submodule
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


_GRAD_COPY_NODE_NAMES = frozenset({"CloneBackward0", "ToCopyBackward0"})
_GRAD_COPY_CHAIN_LIMIT = 8


def grad_copy_source_is_leaf(copy: torch.Tensor) -> bool:
    """Return whether a grad-attached tensor copy was taken from an autograd leaf.

    Saved argument snapshots are copies (``clone`` / ``to``, possibly copies of
    copies) taken with grad attached, so the snapshot itself is a non-leaf.
    Walking down its chain of single-input copy nodes reaches the source: an
    ``AccumulateGrad`` node means the source was a leaf, any other node means
    it was an op output. A copy chain the source itself carried (a user
    ``clone()`` of a leaf) is indistinguishable here, so callers that know the
    producing op should prefer its recorded ``grad_fn``.

    Parameters
    ----------
    copy:
        Tensor snapshot that requires grad.

    Returns
    -------
    bool
        ``True`` when ``copy`` is itself a leaf or its copy chain ends at an
        ``AccumulateGrad`` node; ``False`` otherwise (including an over-long
        chain).
    """

    node = copy.grad_fn
    accumulate_grad_cls = get_accumulate_grad_class()
    for _ in range(_GRAD_COPY_CHAIN_LIMIT):
        if node is None:
            return True
        # Same match as the backward capture's AccumulateGrad test: the class
        # name, or the private class when this torch exposes it.
        if type(node).__name__ == "AccumulateGrad" or bool(
            accumulate_grad_cls and isinstance(node, accumulate_grad_cls)
        ):
            return True
        if type(node).__name__ not in _GRAD_COPY_NODE_NAMES:
            return False
        sources = [nxt for nxt, _ in getattr(node, "next_functions", ()) if nxt is not None]
        if len(sources) != 1:
            return False
        node = sources[0]
    return False


def get_current_graph_task_id_fn() -> Callable[[], Any] | None:
    """Return the autograd engine graph-task-id resolver when available.

    Returns
    -------
    Callable[[], Any] | None
        ``torch._C._current_graph_task_id`` when present, otherwise ``None``.
    """

    resolver = _nested_getattr_or_none(torch, ("_C", "_current_graph_task_id"))
    if resolver is None:
        mark_torch_capability_missing(
            "HAS_CURRENT_GRAPH_TASK_ID",
            "implicit backward pass boundaries cannot distinguish separate "
            "engine invocations and fall back to the open-bracket heuristic",
        )
        return None
    return resolver


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


def get_jit_boolean_dispatch_table() -> Any | None:
    """Return TorchScript's private boolean-dispatch table when available.

    The table maps each ``torch._jit_internal.boolean_dispatch`` product (the
    ``F.max_pool*`` family) to its dispatch record; TorchScript's sugared-value
    layer consults it BY OBJECT before attempting source compilation, so
    wrappers must be registered as additional keys to keep ``torch.jit.script``
    working while torch is wrapped.

    Returns
    -------
    Any | None
        The ``boolean_dispatched`` WeakKeyDictionary, or ``None`` when
        unavailable.
    """

    table = _import_module_attr_or_none("torch._jit_internal", "boolean_dispatched")
    if table is None:
        mark_torch_capability_missing(
            "HAS_JIT_BOOLEAN_DISPATCH_TABLE",
            "TorchScript boolean-dispatch wrapper registration is disabled",
        )
        return None
    return table


def get_jit_overload_resolver_module() -> Any | None:
    """Return the module holding TorchScript's private overload resolver.

    ``torch.jit._script._get_overloads`` is the one recursive-compilation
    entry the C++ sugared-value layer calls that does NOT honor
    ``__prepare_scriptable__``; the identity shim normalizes wrapped
    functionals there so overloaded ops (``F.interpolate``,
    ``F.adaptive_avg_pool2d/3d``) keep scripting while torch is wrapped.

    Returns
    -------
    Any | None
        The ``torch.jit._script`` module, or ``None`` when the resolver is
        unavailable.
    """

    resolver = _import_module_attr_or_none("torch.jit._script", "_get_overloads")
    if resolver is None:
        mark_torch_capability_missing(
            "HAS_JIT_OVERLOAD_RESOLVER",
            "TorchScript overload normalization for wrapped functionals is disabled",
        )
        return None
    return sys.modules.get("torch.jit._script")


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


def get_current_dispatch_mode_stack() -> list[Any] | None:
    """Return the current thread's TorchDispatchMode stack when available.

    Returns
    -------
    list[Any] | None
        Active ``TorchDispatchMode`` instances, or ``None`` when the private
        ``torch.utils._python_dispatch._get_current_dispatch_mode_stack`` probe
        is absent or raises. ``None`` means "cannot answer": the host-escape
        belt census check (its one consumer) must treat that as census-INACTIVE
        so the belt records the escape (fail closed) rather than silently
        standing down (r-b4 R26-1).

    Notes
    -----
    Probed once and cached as a bound callable because this sits on the
    host-value escape path (``item()`` / ``__bool__`` during capture). A probe
    that raises is demoted to permanently-absent so later calls short-circuit.
    """

    global HAS_DISPATCH_MODE_STACK_QUERY, _DISPATCH_MODE_STACK_FN, _DISPATCH_MODE_STACK_PROBED

    if not _DISPATCH_MODE_STACK_PROBED:
        candidate = _import_module_attr_or_none(
            "torch.utils._python_dispatch", "_get_current_dispatch_mode_stack"
        )
        _DISPATCH_MODE_STACK_FN = candidate if callable(candidate) else None
        HAS_DISPATCH_MODE_STACK_QUERY = _DISPATCH_MODE_STACK_FN is not None
        _DISPATCH_MODE_STACK_PROBED = True
    probe = _DISPATCH_MODE_STACK_FN
    if probe is None:
        mark_torch_capability_missing(
            "HAS_DISPATCH_MODE_STACK_QUERY",
            "the host-escape belt records without census-activity suppression (fail closed)",
        )
        return None
    try:
        return list(probe())
    except Exception:
        _DISPATCH_MODE_STACK_FN = None
        mark_torch_capability_missing(
            "HAS_DISPATCH_MODE_STACK_QUERY",
            "the host-escape belt records without census-activity suppression (fail closed)",
        )
        return None


def get_jit_all_schemas() -> list[Any] | None:
    """Return every registered dispatcher schema, or ``None`` when unavailable.

    Backs the arm-time collective-recognizer census (r-b4 R26-2). ``None``
    means the caller must refuse TYPED (an unvetted dispatcher surface can
    never silently read as "no uncaptured collectives").
    """

    global HAS_JIT_SCHEMA_ENUMERATION, _JIT_SCHEMA_ENUMERATION_FN
    global _JIT_SCHEMA_ENUMERATION_PROBED

    if not _JIT_SCHEMA_ENUMERATION_PROBED:
        candidate = _nested_getattr_or_none(torch, ("_C", "_jit_get_all_schemas"))
        _JIT_SCHEMA_ENUMERATION_FN = candidate if callable(candidate) else None
        HAS_JIT_SCHEMA_ENUMERATION = _JIT_SCHEMA_ENUMERATION_FN is not None
        _JIT_SCHEMA_ENUMERATION_PROBED = True
    probe = _JIT_SCHEMA_ENUMERATION_FN
    if probe is None:
        mark_torch_capability_missing(
            "HAS_JIT_SCHEMA_ENUMERATION",
            "distributed arming refuses typed (the recognizer census cannot be vetted)",
        )
        return None
    try:
        return list(probe())
    except Exception:
        _JIT_SCHEMA_ENUMERATION_FN = None
        mark_torch_capability_missing(
            "HAS_JIT_SCHEMA_ENUMERATION",
            "distributed arming refuses typed (the recognizer census cannot be vetted)",
        )
        return None


def get_tensorbase_class() -> type[Any] | None:
    """Return the C tensor base class (``TensorBase``/legacy ``_TensorBase``).

    The ONE probe for every consumer (completeness witness originals,
    callable-safety method-owner census; r-b4 R26-2). ``None`` flips the flag;
    consumers that structurally REQUIRE the class raise explicitly (never an
    ``assert`` stripped under ``python -O``).
    """

    global HAS_TENSORBASE_CLASS, _TENSORBASE_CLASS, _TENSORBASE_CLASS_PROBED

    if not _TENSORBASE_CLASS_PROBED:
        candidate = _nested_getattr_or_none(torch, ("_C", "TensorBase"))
        if candidate is None:
            candidate = _nested_getattr_or_none(torch, ("_C", "_TensorBase"))
        _TENSORBASE_CLASS = candidate if isinstance(candidate, type) else None
        HAS_TENSORBASE_CLASS = _TENSORBASE_CLASS is not None
        _TENSORBASE_CLASS_PROBED = True
    if _TENSORBASE_CLASS is None:
        mark_torch_capability_missing(
            "HAS_TENSORBASE_CLASS",
            "C-level tensor-base introspection is unavailable",
        )
    return _TENSORBASE_CLASS


def get_disable_torch_function_context() -> Any | None:
    """Return ``torch._C.DisableTorchFunction``, or ``None`` when unavailable.

    ``None`` means an ambient torch-FUNCTION mode CANNOT be neutralized for
    import-time probes (r-b4 R26-2): the caller's fallback does not neutralize
    such a mode, so the degradation must be disclosed by the flag rather than
    silently absorbed.
    """

    global HAS_DISABLE_TORCH_FUNCTION, _DISABLE_TORCH_FUNCTION_CLS
    global _DISABLE_TORCH_FUNCTION_PROBED

    if not _DISABLE_TORCH_FUNCTION_PROBED:
        _DISABLE_TORCH_FUNCTION_CLS = _nested_getattr_or_none(torch, ("_C", "DisableTorchFunction"))
        HAS_DISABLE_TORCH_FUNCTION = _DISABLE_TORCH_FUNCTION_CLS is not None
        _DISABLE_TORCH_FUNCTION_PROBED = True
    if _DISABLE_TORCH_FUNCTION_CLS is None:
        mark_torch_capability_missing(
            "HAS_DISABLE_TORCH_FUNCTION",
            "import-time probes cannot neutralize an ambient torch-function mode",
        )
    return _DISABLE_TORCH_FUNCTION_CLS


def get_variable_functions_class() -> Any | None:
    """Return ``torch._C._VariableFunctionsClass``, or ``None`` when unavailable.

    Backs the internal-torch-builtin registry-key lane (r-b4 R26-2): without
    it, a captured internal builtin is classified through the public ``torch``
    wrapper -- a DIFFERENT argument convention for replay -- so the downgrade
    must flip the flag instead of happening silently.
    """

    global HAS_VARIABLE_FUNCTIONS_CLASS, _VARIABLE_FUNCTIONS_CLASS
    global _VARIABLE_FUNCTIONS_CLASS_PROBED

    if not _VARIABLE_FUNCTIONS_CLASS_PROBED:
        _VARIABLE_FUNCTIONS_CLASS = _nested_getattr_or_none(
            torch, ("_C", "_VariableFunctionsClass")
        )
        HAS_VARIABLE_FUNCTIONS_CLASS = _VARIABLE_FUNCTIONS_CLASS is not None
        _VARIABLE_FUNCTIONS_CLASS_PROBED = True
    if _VARIABLE_FUNCTIONS_CLASS is None:
        mark_torch_capability_missing(
            "HAS_VARIABLE_FUNCTIONS_CLASS",
            "internal torch builtins record public-wrapper replay keys",
        )
    return _VARIABLE_FUNCTIONS_CLASS


def get_fake_tensor_mode_class() -> type[Any] | None:
    """Return torch's ``FakeTensorMode`` class, or ``None`` when unavailable.

    The runnable allocation preflight deliberately fails OPEN without it
    (refusing a legitimate run over a missing estimator would be worse), but
    the degradation flips ``HAS_FAKE_TENSOR_MODE`` so it is visible in
    doctor/compat instead of silently vanishing (r-b4 R26-2).
    """

    global HAS_FAKE_TENSOR_MODE, _FAKE_TENSOR_MODE_CLS, _FAKE_TENSOR_MODE_PROBED

    if not _FAKE_TENSOR_MODE_PROBED:
        candidate = _import_module_attr_or_none("torch._subclasses.fake_tensor", "FakeTensorMode")
        _FAKE_TENSOR_MODE_CLS = candidate if isinstance(candidate, type) else None
        HAS_FAKE_TENSOR_MODE = _FAKE_TENSOR_MODE_CLS is not None
        _FAKE_TENSOR_MODE_PROBED = True
    if _FAKE_TENSOR_MODE_CLS is None:
        mark_torch_capability_missing(
            "HAS_FAKE_TENSOR_MODE",
            "the runnable allocation preflight is disabled (falls back to the recorded bound)",
        )
    return _FAKE_TENSOR_MODE_CLS


def get_dtensor_shard_geometry_fn() -> Callable[..., Any] | None:
    """Return DTensor's shard-geometry helper, or ``None`` when unavailable.

    Backs the per-site dual geometry of the DTensor refusal (r-b4 R26-2).
    ``None`` distinguishes "the private API moved" (flag flips, visible in
    doctor/compat) from a legitimately off-mesh rank (helper present, raises
    at computation), which previously collapsed into one silent ``None``.
    """

    global HAS_DTENSOR_SHARD_GEOMETRY, _DTENSOR_SHARD_GEOMETRY_FN
    global _DTENSOR_SHARD_GEOMETRY_PROBED

    if not _DTENSOR_SHARD_GEOMETRY_PROBED:
        candidate = _import_module_attr_or_none(
            "torch.distributed.tensor._utils", "compute_local_shape_and_global_offset"
        )
        _DTENSOR_SHARD_GEOMETRY_FN = candidate if callable(candidate) else None
        HAS_DTENSOR_SHARD_GEOMETRY = _DTENSOR_SHARD_GEOMETRY_FN is not None
        _DTENSOR_SHARD_GEOMETRY_PROBED = True
    if _DTENSOR_SHARD_GEOMETRY_FN is None:
        mark_torch_capability_missing(
            "HAS_DTENSOR_SHARD_GEOMETRY",
            "refused-DTensor findings omit shard offsets (geometry helper unavailable)",
        )
    return _DTENSOR_SHARD_GEOMETRY_FN


def get_funcol_group_resolvers(
    *, force_probe: bool = False
) -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    """Return funcol's group resolver plus the c10d name resolver, or ``None``.

    Parameters
    ----------
    force_probe:
        Re-run the probe even when it already ran this process. Passed by
        :func:`get_torch_capability_snapshot` so the reported flag reflects the
        live runtime rather than a cached earlier answer.

    Backs the functional-collective boundary wraps (merge-ranks C2 recording):
    ``torch.distributed._functional_collectives._resolve_group`` maps every
    public funcol group spelling (ProcessGroup / group-name string / DeviceMesh /
    ``(mesh, dim)`` / rank lists) to a ProcessGroup or group name, and
    ``torch.distributed.distributed_c10d._resolve_process_group`` maps a group
    name to its live ProcessGroup. Both are torch-private surfaces, so their
    absence flips the named ``HAS_FUNCOL_GROUP_RESOLUTION`` flag (visible in
    ``doctor()`` / ``compat.report()``) instead of failing mid-capture; the
    boundary wrap then refuses captured funcol calls typed rather than letting
    an uncorrelatable collective execute silently.
    """

    global HAS_FUNCOL_GROUP_RESOLUTION, _FUNCOL_GROUP_RESOLVERS
    global _FUNCOL_GROUP_RESOLUTION_PROBED

    if force_probe or not _FUNCOL_GROUP_RESOLUTION_PROBED:
        resolve_group = _import_module_attr_or_none(
            "torch.distributed._functional_collectives", "_resolve_group"
        )
        resolve_name = _import_module_attr_or_none(
            "torch.distributed.distributed_c10d", "_resolve_process_group"
        )
        if callable(resolve_group) and callable(resolve_name):
            _FUNCOL_GROUP_RESOLVERS = (resolve_group, resolve_name)
        else:
            _FUNCOL_GROUP_RESOLVERS = None
        HAS_FUNCOL_GROUP_RESOLUTION = _FUNCOL_GROUP_RESOLVERS is not None
        _FUNCOL_GROUP_RESOLUTION_PROBED = True
    if _FUNCOL_GROUP_RESOLVERS is None:
        mark_torch_capability_missing(
            "HAS_FUNCOL_GROUP_RESOLUTION",
            "captured functional-collective (funcol) calls refuse typed instead "
            "of recording correlated boundary nodes (group resolvers unavailable)",
        )
    return _FUNCOL_GROUP_RESOLVERS


def probe_funcol_wait_interposition() -> bool:
    """Probe the dispatcher surfaces the plane-W wait interposition needs.

    The mode-independent completion authority registers a scoped
    ``torch.library.Library("_c10d_functional", "IMPL")`` wrapper for
    ``wait_tensor`` at the CPU key and redispatches below itself through
    ``torch._C._ExcludeDispatchKeyGuard`` (design-merge-ranks-c v5, 1.4c; probe
    P3a). All five surfaces (the guard trio, ``torch.library.Library``, and the
    ``torch.ops._c10d_functional.wait_tensor.default`` op handle) are
    feature-detected here; absence flips the named
    ``HAS_FUNCOL_WAIT_INTERPOSITION`` flag and funcol completions then stay
    honestly ``unobserved`` (fail-closed disclosure, never a crash). The
    resolved redispatch surfaces are served by
    :func:`get_funcol_wait_redispatch`, so no caller ever touches the private
    dispatcher spellings directly.
    """

    global HAS_FUNCOL_WAIT_INTERPOSITION, _FUNCOL_WAIT_INTERPOSITION_PROBED
    global _FUNCOL_WAIT_REDISPATCH

    if not _FUNCOL_WAIT_INTERPOSITION_PROBED:
        library_cls = getattr(getattr(torch, "library", None), "Library", None)
        internals = getattr(torch, "_C", None)
        guard_cls = getattr(internals, "_ExcludeDispatchKeyGuard", None)
        keyset_cls = getattr(internals, "DispatchKeySet", None)
        cpu_key = getattr(getattr(internals, "DispatchKey", None), "CPU", None)
        try:
            wait_op: Any | None = torch.ops._c10d_functional.wait_tensor.default
        except Exception:
            wait_op = None
        if (
            library_cls is not None
            and guard_cls is not None
            and keyset_cls is not None
            and cpu_key is not None
            and wait_op is not None
            and _import_module_attr_or_none(
                "torch.distributed._functional_collectives", "wait_tensor"
            )
            is not None
        ):

            def _exclude_cpu_dispatch_guard(
                _guard_cls: Any = guard_cls, _keyset_cls: Any = keyset_cls, _cpu_key: Any = cpu_key
            ) -> Any:
                """Build the below-CPU-key redispatch guard for one wait call."""

                return _guard_cls(_keyset_cls(_cpu_key))

            _FUNCOL_WAIT_REDISPATCH = (wait_op, _exclude_cpu_dispatch_guard)
        else:
            _FUNCOL_WAIT_REDISPATCH = None
        HAS_FUNCOL_WAIT_INTERPOSITION = _FUNCOL_WAIT_REDISPATCH is not None
        _FUNCOL_WAIT_INTERPOSITION_PROBED = True
    if not HAS_FUNCOL_WAIT_INTERPOSITION:
        mark_torch_capability_missing(
            "HAS_FUNCOL_WAIT_INTERPOSITION",
            "functional-collective completions stay completion_binding='unobserved' "
            "(dispatcher wait interposition unavailable on this torch build)",
        )
    return HAS_FUNCOL_WAIT_INTERPOSITION


def get_funcol_wait_redispatch() -> tuple[Any, Callable[[], Any]] | None:
    """Return the wait-interposition redispatch surfaces, or ``None``.

    Returns
    -------
    tuple[Any, Callable[[], Any]] | None
        ``(wait_op, exclude_cpu_guard)`` where ``wait_op`` is the resolved
        ``torch.ops._c10d_functional.wait_tensor.default`` op handle and
        ``exclude_cpu_guard()`` builds a fresh
        ``torch._C._ExcludeDispatchKeyGuard`` over the CPU dispatch key, or
        ``None`` when :func:`probe_funcol_wait_interposition` reports the
        capability absent (the flag flip and its disclosure happen there).
    """

    if not probe_funcol_wait_interposition():
        return None
    return _FUNCOL_WAIT_REDISPATCH


def get_funcol_module() -> Any | None:
    """Return ``torch.distributed._functional_collectives``, or ``None``.

    Backs the funcol boundary wrap install/uninstall (merge-ranks C2
    recording): the wraps must patch attributes on torch's REAL funcol module
    object (a copy would never be consulted by user ``funcol.all_reduce``
    calls), so this accessor hands out the module itself. The import is
    eager on first call -- callers only reach it at distributed arm/disarm
    time, never on the plain-capture hot path. Absence flips the named
    ``HAS_FUNCOL_MODULE`` flag (visible in ``doctor()`` / ``compat.report()``)
    and funcol calls then run unwrapped, exactly the pre-C2 disclosure class.
    """

    global HAS_FUNCOL_MODULE, _FUNCOL_MODULE_OBJ, _FUNCOL_MODULE_PROBED

    if not _FUNCOL_MODULE_PROBED:
        try:
            import torch.distributed._functional_collectives as funcol_module

            _FUNCOL_MODULE_OBJ = funcol_module
        except Exception:
            _FUNCOL_MODULE_OBJ = None
        HAS_FUNCOL_MODULE = _FUNCOL_MODULE_OBJ is not None
        _FUNCOL_MODULE_PROBED = True
    if _FUNCOL_MODULE_OBJ is None:
        mark_torch_capability_missing(
            "HAS_FUNCOL_MODULE",
            "functional-collective (funcol) calls run unwrapped; no funcol "
            "boundary nodes are recorded (module unavailable on this build)",
        )
    return _FUNCOL_MODULE_OBJ


def get_async_collective_tensor_type(*, force_probe: bool = False) -> type[Any] | None:
    """Return funcol's ``AsyncCollectiveTensor`` class without an eager import.

    Parameters
    ----------
    force_probe:
        Import the funcol namespace even when it has never been imported in
        this process. Diagnostic surfaces set this to report the real build
        capability; capture paths keep the default.

    Returns
    -------
    type[Any] | None
        The ACT wrapper class, or ``None`` when unavailable or when the lazy
        default defers the probe.

    Notes
    -----
    Mirrors :func:`get_dtensor_type`: a live value can only *be* an ACT if
    ``torch.distributed._functional_collectives`` is already in
    ``sys.modules``, so the default path never pays the ``torch.distributed``
    import on plain captures (the label chokepoint calls this per tensor
    read). A probed absence flips the named ``HAS_ASYNC_COLLECTIVE_TENSOR``
    flag: ACT unwrapping is then disabled and in-flight funcol destinations
    would surface as opaque wrapper subclasses instead of their inner
    tensors.
    """

    global HAS_ASYNC_COLLECTIVE_TENSOR, _ASYNC_COLLECTIVE_TENSOR_PROBED
    global _ASYNC_COLLECTIVE_TENSOR_TYPE

    if not _ASYNC_COLLECTIVE_TENSOR_PROBED:
        if not force_probe and "torch.distributed._functional_collectives" not in sys.modules:
            return None
        act_type = _import_module_attr_or_none(
            "torch.distributed._functional_collectives", "AsyncCollectiveTensor"
        )
        _ASYNC_COLLECTIVE_TENSOR_TYPE = act_type if isinstance(act_type, type) else None
        HAS_ASYNC_COLLECTIVE_TENSOR = _ASYNC_COLLECTIVE_TENSOR_TYPE is not None
        _ASYNC_COLLECTIVE_TENSOR_PROBED = True
    if _ASYNC_COLLECTIVE_TENSOR_TYPE is None:
        mark_torch_capability_missing(
            "HAS_ASYNC_COLLECTIVE_TENSOR",
            "AsyncCollectiveTensor unwrapping is disabled; in-flight funcol "
            "destinations stay opaque wrapper values",
        )
    return _ASYNC_COLLECTIVE_TENSOR_TYPE


def get_checkpoint_hook_class() -> type[Any] | None:
    """Return torch's private non-reentrant checkpoint hook class, or ``None``.

    Backs the L9 checkpoint-token classifier: a ``_checkpoint_hook`` context
    enter is the identity basis for minting non-reentrant checkpoint tokens.
    Absence flips the named ``HAS_CHECKPOINT_HOOK_CLASS`` flag and the
    classifier degrades fail-closed (degrade class D1: NO tokens are minted,
    so an unrecognized checkpoint variant can never mint a false token).
    """

    global HAS_CHECKPOINT_HOOK_CLASS, _CHECKPOINT_HOOK_CLASS
    global _CHECKPOINT_HOOK_CLASS_PROBED

    if not _CHECKPOINT_HOOK_CLASS_PROBED:
        resolved = _import_module_attr_or_none("torch.utils.checkpoint", "_checkpoint_hook")
        _CHECKPOINT_HOOK_CLASS = resolved if isinstance(resolved, type) else None
        HAS_CHECKPOINT_HOOK_CLASS = _CHECKPOINT_HOOK_CLASS is not None
        _CHECKPOINT_HOOK_CLASS_PROBED = True
    if _CHECKPOINT_HOOK_CLASS is None:
        mark_torch_capability_missing(
            "HAS_CHECKPOINT_HOOK_CLASS",
            "non-reentrant checkpoint token minting is disabled (fail-closed: "
            "checkpoint invocation evidence degrades, no false token is minted)",
        )
    return _CHECKPOINT_HOOK_CLASS


def get_checkpoint_internal_hook_class() -> type[Any] | None:
    """Return torch's private ``_checkpoint_internal_hook`` class, or ``None``.

    New in torch 2.14 (``torch/utils/checkpoint.py``): ``_checkpoint_hook`` and
    ``_recomputation_hook`` both subclass this intermediate class instead of
    ``torch.autograd.graph.saved_tensors_hooks`` directly. Its ``__enter__`` sets
    ``self.pack_hook._user_hooks`` (walked by ``_current_user_saved_tensors_hooks``
    to resolve nested-checkpoint identity) and its ``__exit__`` deletes that same
    attribute. Absence (torch 2.13 and earlier, or a torch without the private
    name) means hook objects carry no such cross-call identity state, so a
    hook-object replacement is always safe; presence means a replacement MUST
    carry the attribute forward or skip the swap (fail-closed), or torch's own
    ``__exit__`` raises reaching for state that moved to a different function
    object -- an exception inside ``__exit__`` that skips the matching
    ``_pop_saved_tensors_default_hooks()`` call and permanently corrupts torch's
    global saved-tensors-hooks stack for the rest of the process.
    """

    global HAS_CHECKPOINT_INTERNAL_HOOK_CLASS, _CHECKPOINT_INTERNAL_HOOK_CLASS
    global _CHECKPOINT_INTERNAL_HOOK_CLASS_PROBED

    if not _CHECKPOINT_INTERNAL_HOOK_CLASS_PROBED:
        resolved = _import_module_attr_or_none(
            "torch.utils.checkpoint", "_checkpoint_internal_hook"
        )
        _CHECKPOINT_INTERNAL_HOOK_CLASS = resolved if isinstance(resolved, type) else None
        HAS_CHECKPOINT_INTERNAL_HOOK_CLASS = _CHECKPOINT_INTERNAL_HOOK_CLASS is not None
        _CHECKPOINT_INTERNAL_HOOK_CLASS_PROBED = True
    return _CHECKPOINT_INTERNAL_HOOK_CLASS


def get_autograd_engine_queue_callback() -> Callable[..., Any] | None:
    """Return the autograd engine's ``queue_callback`` binding, or ``None``.

    Backs the L9 implicit-backward engine-drain close path: a final callback
    enqueued on ``torch.autograd.Variable._execution_engine`` closes the
    implicit backward pass when the engine drains. Absence flips the named
    ``HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK`` flag and the close falls back to
    the always-armed sync-point backstop (a disclosure-path difference, never
    a coverage gap).
    """

    global HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK, _AUTOGRAD_ENGINE_QUEUE_CALLBACK
    global _AUTOGRAD_ENGINE_QUEUE_CALLBACK_PROBED

    if not _AUTOGRAD_ENGINE_QUEUE_CALLBACK_PROBED:
        engine = _nested_getattr_or_none(torch, ("autograd", "Variable", "_execution_engine"))
        queue_callback = getattr(engine, "queue_callback", None) if engine is not None else None
        _AUTOGRAD_ENGINE_QUEUE_CALLBACK = queue_callback if callable(queue_callback) else None
        HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK = _AUTOGRAD_ENGINE_QUEUE_CALLBACK is not None
        _AUTOGRAD_ENGINE_QUEUE_CALLBACK_PROBED = True
    if _AUTOGRAD_ENGINE_QUEUE_CALLBACK is None:
        mark_torch_capability_missing(
            "HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK",
            "implicit backward passes close at the next sync point instead of "
            "at autograd engine drain (engine callback handle unavailable)",
        )
    return _AUTOGRAD_ENGINE_QUEUE_CALLBACK


def kineto_events_from_profiler(profiler: Any) -> tuple[Any, ...] | None:
    """Return one profiler's in-memory Kineto events, or ``None`` (W2.1).

    The device-time join consumes in-memory ``_KinetoEvent`` objects --
    integer-nanosecond intervals, runtime correlation IDs, thread IDs, the
    user-annotation flag, and the typed per-event activity classification --
    instead of writing and re-parsing a chrome-trace JSON temp file (the
    deleted 12.4x-inflation path). The event objects ride an undocumented
    torch seam, so extraction is feature-detected here, at the ONE sanctioned
    private-probe boundary: the first successful extraction validates the
    exact ``KINETO_EVENT_FIELD_CONTRACT`` on a live event and flips
    ``HAS_KINETO_INMEMORY_EVENTS`` (plus ``HAS_KINETO_EVENT_SCOPE`` for the
    separately-degradable ``scope`` field); any miss returns ``None`` and the
    caller demotes to its bounded streaming chrome-event extractor -- the
    adapter never guesses and records which path ran.

    Parameters
    ----------
    profiler:
        A CLOSED ``torch.profiler.profile`` instance (events are complete
        only after ``__exit__``).

    Returns
    -------
    tuple[Any, ...] | None
        The raw Kineto events, or ``None`` when the in-memory seam or its
        field contract is unavailable on this torch build.
    """

    global HAS_KINETO_INMEMORY_EVENTS, _KINETO_INMEMORY_EVENTS_PROBED
    global HAS_KINETO_EVENT_SCOPE, _KINETO_EVENT_SCOPE_PROBED

    try:
        inner = getattr(profiler, "profiler", None)
        results = getattr(inner, "kineto_results", None) if inner is not None else None
        raw_events = results.events() if results is not None else None
    except Exception:
        raw_events = None
    if raw_events is None:
        if not _KINETO_INMEMORY_EVENTS_PROBED:
            # Unprobed absence on THIS call is not proof about the build;
            # leave the flags unprobed so a later well-formed profiler can
            # still validate the contract.
            return None
        return None
    events = tuple(raw_events)
    if not _KINETO_INMEMORY_EVENTS_PROBED:
        sample = events[0] if events else None
        if sample is None:
            # An empty stream proves the seam exists but not the field
            # contract; extract nothing and stay unprobed.
            return events
        contract_ok = all(
            callable(getattr(sample, field_name, None))
            for field_name in KINETO_EVENT_FIELD_CONTRACT
        )
        HAS_KINETO_INMEMORY_EVENTS = contract_ok
        _KINETO_INMEMORY_EVENTS_PROBED = True
        HAS_KINETO_EVENT_SCOPE = contract_ok and callable(getattr(sample, "scope", None))
        _KINETO_EVENT_SCOPE_PROBED = True
    if not HAS_KINETO_INMEMORY_EVENTS:
        mark_torch_capability_missing(
            "HAS_KINETO_INMEMORY_EVENTS",
            "the device-time join demotes to the bounded streaming "
            "chrome-event extractor (in-memory Kineto event field contract "
            "unavailable on this torch build)",
        )
        return None
    return events


def memory_profile_from_profiler(profiler: Any) -> Any | None:
    """Return one closed profiler's private categorized memory profile.

    torchnative W0.7 (the DEADLINE item): torch deprecated its only
    categorized memory view, and the still-alive private categorizer
    (``profiler._memory_profile()``) is the ONE oracle the rebuilt
    categorized timeline can ever be checked against -- after upstream
    deletes it, the parity oracle can never be built again. This accessor is
    the test-oracle exception to the no-private-APIs rule: it
    feature-detects, flips the named ``HAS_MEMORY_PROFILE`` flag, and
    callers SKIP WITH A NAMED REASON on absence (the frozen torch-2.13
    golden fixture then carries the preserved behavior).

    Parameters
    ----------
    profiler:
        A closed ``torch.profiler.profile`` instance created with
        ``profile_memory=True, record_shapes=True, with_stack=True``.

    Returns
    -------
    Any | None
        The private memory-profile object, or ``None`` when the seam is
        gone (flag flipped, capability marked missing).
    """

    global HAS_MEMORY_PROFILE, _MEMORY_PROFILE_PROBED

    probe = getattr(profiler, "_memory_profile", None)
    if not _MEMORY_PROFILE_PROBED:
        HAS_MEMORY_PROFILE = callable(probe)
        _MEMORY_PROFILE_PROBED = True
    if not callable(probe):
        mark_torch_capability_missing(
            "HAS_MEMORY_PROFILE",
            "the memory-timeline parity oracle can no longer run live; the "
            "frozen torch-2.13 golden fixture is the remaining authority",
        )
        return None
    try:
        return probe()
    except Exception:
        return None


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


def get_torch_function_stack_surgery() -> (
    tuple[Callable[[], int], Callable[[], Any], Callable[[Any], Any]] | None
):
    """Return the torch-function mode-stack surgery trio when available.

    Returns
    -------
    tuple[Callable[[], int], Callable[[], Any], Callable[[Any], Any]] | None
        ``(len_stack, pop_stack, push_stack)`` -- the private ``torch._C``
        mode-stack primitives (``_len_torch_function_stack``,
        ``_pop_torch_function_stack``, ``_push_on_torch_function_stack``)
        the weightsfree ambient DeviceContext absorption (W1-ABSORB / D19)
        pops and restores caller modes with -- or ``None`` when the
        import-time ``HAS_TORCH_FUNCTION_STACK_SURGERY`` probe found any of
        the four surgery primitives missing. ``None`` means an ambient
        torch-function mode CANNOT be absorbed; the caller must refuse
        typed, never guess.
    """

    len_stack = _nested_getattr_or_none(torch, ("_C", "_len_torch_function_stack"))
    pop_stack = _nested_getattr_or_none(torch, ("_C", "_pop_torch_function_stack"))
    push_stack = _nested_getattr_or_none(torch, ("_C", "_push_on_torch_function_stack"))
    if not (
        HAS_TORCH_FUNCTION_STACK_SURGERY
        and callable(len_stack)
        and callable(pop_stack)
        and callable(push_stack)
    ):
        mark_torch_capability_missing(
            "HAS_TORCH_FUNCTION_STACK_SURGERY",
            "ambient torch-function modes cannot be absorbed for weights-free capture",
        )
        return None
    return len_stack, pop_stack, push_stack


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


def get_fsdp_wrapper_type(*, force_probe: bool = False) -> type[Any] | None:
    """Return torch's ``FullyShardedDataParallel`` type without eager import.

    Parameters
    ----------
    force_probe:
        Import ``torch.distributed.fsdp`` even when it has never been imported
        in this process. Diagnostic surfaces set this to report the real build
        capability; capture paths keep the default.

    Returns
    -------
    type[Any] | None
        FSDP wrapper type, or ``None`` when unavailable or when the lazy
        default defers the probe.

    Notes
    -----
    A live model can only be a ``FullyShardedDataParallel`` instance if
    ``torch.distributed.fsdp`` is already in ``sys.modules`` (Python registers
    a module before any class from it can be instantiated), so when it is
    absent the default path reports ``None`` without paying the FSDP import
    (~1.2s and hundreds of modules cold).
    """

    global HAS_FSDP_WRAPPER, _FSDP_WRAPPER_PROBED, _FSDP_WRAPPER_TYPE

    if not _FSDP_WRAPPER_PROBED:
        if not force_probe and "torch.distributed.fsdp" not in sys.modules:
            return None
        try:
            fsdp_type = _import_module_attr_or_none(
                "torch.distributed.fsdp", "FullyShardedDataParallel"
            )
        except RuntimeError:
            fsdp_type = None
        _FSDP_WRAPPER_TYPE = fsdp_type if isinstance(fsdp_type, type) else None
        HAS_FSDP_WRAPPER = _FSDP_WRAPPER_TYPE is not None
        _FSDP_WRAPPER_PROBED = True
    fsdp_wrapper_type = _FSDP_WRAPPER_TYPE
    if fsdp_wrapper_type is None:
        mark_torch_capability_missing(
            "HAS_FSDP_WRAPPER",
            "FSDP wrapper detection and rejection are disabled",
        )
        return None
    return fsdp_wrapper_type


# Module paths that must already be in ``sys.modules`` for a live DTensor /
# DeviceMesh / pipeline-stage object to exist. ``torch.distributed._tensor`` is
# the pre-2.5 spelling and ``torch.distributed.tensor`` the current one; both
# resolve to the same class object on torch that has the alias, so probing both
# is a capability probe, not a version parse.
_DTENSOR_MODULE_CANDIDATES: tuple[str, ...] = (
    "torch.distributed.tensor",
    "torch.distributed._tensor",
)
_PIPELINING_MODULE_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torch.distributed.pipelining", ("PipelineStage", "_PipelineStageBase")),
    ("torch.distributed.pipeline.sync", ("Pipe",)),
)


def get_dtensor_type(*, force_probe: bool = False) -> type[Any] | None:
    """Return torch's ``DTensor`` type without paying an eager import.

    Parameters
    ----------
    force_probe:
        Import the DTensor namespace even when it has never been imported in
        this process. Diagnostic surfaces set this to report the real build
        capability; capture paths keep the default.

    Returns
    -------
    type[Any] | None
        ``DTensor`` type, or ``None`` when unavailable or when the lazy default
        defers the probe.

    Notes
    -----
    Mirrors :func:`get_fsdp_wrapper_type`: a live tensor can only *be* a
    ``DTensor`` if its defining module is already in ``sys.modules``, because
    Python registers a module before any class from it can be instantiated. So
    when no candidate module has been imported, the default path reports
    ``None`` without paying the ``torch.distributed`` import cost.
    """

    global HAS_DTENSOR, _DTENSOR_PROBED, _DTENSOR_TYPE

    if not _DTENSOR_PROBED:
        if not force_probe and not any(name in sys.modules for name in _DTENSOR_MODULE_CANDIDATES):
            return None
        dtensor_type: Any = None
        for module_name in _DTENSOR_MODULE_CANDIDATES:
            try:
                dtensor_type = _import_module_attr_or_none(module_name, "DTensor")
            except RuntimeError:
                dtensor_type = None
            if isinstance(dtensor_type, type):
                break
        _DTENSOR_TYPE = dtensor_type if isinstance(dtensor_type, type) else None
        HAS_DTENSOR = _DTENSOR_TYPE is not None
        _DTENSOR_PROBED = True
    if _DTENSOR_TYPE is None:
        mark_torch_capability_missing(
            "HAS_DTENSOR",
            "exact DTensor detection falls back to structural tensor-subclass matching",
        )
        return None
    return _DTENSOR_TYPE


def get_device_mesh_type(*, force_probe: bool = False) -> type[Any] | None:
    """Return torch's ``DeviceMesh`` type without paying an eager import.

    Parameters
    ----------
    force_probe:
        Import ``torch.distributed.device_mesh`` even when it has never been
        imported in this process.

    Returns
    -------
    type[Any] | None
        ``DeviceMesh`` type, or ``None`` when unavailable or when the lazy
        default defers the probe.
    """

    global HAS_DEVICE_MESH, _DEVICE_MESH_PROBED, _DEVICE_MESH_TYPE

    if not _DEVICE_MESH_PROBED:
        if not force_probe and "torch.distributed.device_mesh" not in sys.modules:
            return None
        try:
            mesh_type = _import_module_attr_or_none("torch.distributed.device_mesh", "DeviceMesh")
        except RuntimeError:
            mesh_type = None
        _DEVICE_MESH_TYPE = mesh_type if isinstance(mesh_type, type) else None
        HAS_DEVICE_MESH = _DEVICE_MESH_TYPE is not None
        _DEVICE_MESH_PROBED = True
    if _DEVICE_MESH_TYPE is None:
        mark_torch_capability_missing(
            "HAS_DEVICE_MESH",
            "exact device-mesh detection falls back to structural attribute matching",
        )
        return None
    return _DEVICE_MESH_TYPE


def get_pipelining_module_types(*, force_probe: bool = False) -> tuple[type[Any], ...]:
    """Return torch pipeline-parallel stage/schedule types without eager import.

    Parameters
    ----------
    force_probe:
        Import the pipelining namespaces even when they have never been
        imported in this process.

    Returns
    -------
    tuple[type[Any], ...]
        Available pipeline-stage types. Empty when pipelining is unavailable or
        when the lazy default defers the probe.

    Notes
    -----
    Covers both the current ``torch.distributed.pipelining`` namespace and the
    removed-in-2.5 ``torch.distributed.pipeline.sync.Pipe`` wrapper, so a model
    built on either spelling is detected on the torch that has it.
    """

    global HAS_PIPELINING, _PIPELINING_PROBED, _PIPELINING_TYPES

    if not _PIPELINING_PROBED:
        candidate_modules = tuple(name for name, _attrs in _PIPELINING_MODULE_CANDIDATES)
        if not force_probe and not any(name in sys.modules for name in candidate_modules):
            return ()
        found: list[type[Any]] = []
        for module_name, attr_names in _PIPELINING_MODULE_CANDIDATES:
            for attr_name in attr_names:
                try:
                    candidate = _import_module_attr_or_none(module_name, attr_name)
                except RuntimeError:
                    candidate = None
                if isinstance(candidate, type):
                    found.append(candidate)
        _PIPELINING_TYPES = tuple(found)
        HAS_PIPELINING = bool(_PIPELINING_TYPES)
        _PIPELINING_PROBED = True
    if not _PIPELINING_TYPES:
        mark_torch_capability_missing(
            "HAS_PIPELINING",
            "exact pipeline-parallel detection falls back to structural namespace matching",
        )
        return ()
    return _PIPELINING_TYPES


# Tracing tensor subclasses: tensors that carry no real data, so every TorchLens
# operation that reads a value (safe_copy, torch.equal, .item(), memory
# accounting) is meaningless or fatal on them. ``FakeTensor`` backs Dynamo/AOT
# tracing and ``torch.export``; ``FunctionalTensor`` backs functionalization.
_TRACING_TENSOR_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("torch._subclasses.fake_tensor", "FakeTensor"),
    ("torch._subclasses.functional_tensor", "FunctionalTensor"),
)

_DYNAMO_IS_COMPILING_CANDIDATES: tuple[tuple[str, str], ...] = (
    # Public since torch 2.0; preferred so behavior tracks the canonical impl.
    ("torch.compiler", "is_compiling"),
    ("torch._dynamo", "is_compiling"),
)


def dynamo_is_compiling() -> bool:
    """Return whether Dynamo is currently tracing this frame.

    Returns
    -------
    bool
        True while a ``torch.compile`` region is being traced. False when the
        capability is structurally ABSENT (no probe resolved), so an absent
        probe degrades to "not compiling" rather than disabling capture. True
        when a resolved probe RAISES (r-b4 R26-3): "cannot answer" is treated
        as possibly-compiling so the wrapper takes the disclosed dynamo-region
        bypass (``capture_verified=False`` + ``dynamo_region_not_logged``)
        instead of silently logging data-free FakeTensors -- the documented
        crash class the guard exists to prevent. Both degraded paths flip
        ``HAS_DYNAMO_IS_COMPILING``.

    Notes
    -----
    Probed once and cached as a bound callable, because this is consulted on the
    capture logging path once per operation. Measured at ~0.5 us per call, which
    is under 1% of the per-op logging cost.
    """

    global HAS_DYNAMO_IS_COMPILING, _DYNAMO_IS_COMPILING_FN, _DYNAMO_IS_COMPILING_PROBED

    if not _DYNAMO_IS_COMPILING_PROBED:
        for module_name, attr_name in _DYNAMO_IS_COMPILING_CANDIDATES:
            try:
                candidate = _import_module_attr_or_none(module_name, attr_name)
            except RuntimeError:
                candidate = None
            if callable(candidate):
                _DYNAMO_IS_COMPILING_FN = candidate
                break
        HAS_DYNAMO_IS_COMPILING = _DYNAMO_IS_COMPILING_FN is not None
        _DYNAMO_IS_COMPILING_PROBED = True
    probe = _DYNAMO_IS_COMPILING_FN
    if probe is None:
        mark_torch_capability_missing(
            "HAS_DYNAMO_IS_COMPILING",
            "Dynamo-tracing detection is disabled; compiled regions may fail inside the wrappers",
        )
        return False
    try:
        return bool(probe())
    except Exception:
        mark_torch_capability_missing(
            "HAS_DYNAMO_IS_COMPILING",
            "Dynamo-tracing detection raised; frames are treated as possibly-compiling "
            "so compiled regions degrade to the disclosed bypass",
        )
        return True


_LAZY_TORCH_IMPORTS_WARMED: bool = False


def warm_lazy_torch_imports() -> None:
    """Force torch's lazy ``torch._compile``/``torch._dynamo`` cascade to run NOW.

    The first wrapped op of a capture can trigger torch's own lazy
    ``import torch._dynamo`` (``torch/_compile.py``), whose import cascade
    draws host entropy at module-exec time (``uuid.uuid4()`` in
    ``torch.distributed._composable.contract``, plus getrandbits/instance
    draws). Fired INSIDE the RNG channel-monitor window, those draws marked
    ``os.urandom``/getrandbits channels and permanently ceilinged the first
    selective runnable-capable capture of the process to UNVERIFIABLE -- a
    silent, order-dependent breach of the contract's "a plain deterministic
    capture records nothing" pin. The monitor calls this BEFORE arming any
    patch so the cascade runs outside every window. ``torch.backends.opt_einsum``
    (imported inside ``torch.einsum``) is warmed too: an in-window import's
    ``sys.meta_path`` finder frames become deep-inventory roots (see ``rng.py``).

    Failure is benign and intentionally unlatched: a partially-executed failed
    import is evicted from ``sys.modules``, so a later in-window retry re-runs
    the cascade and its draws are then honestly MARKED (the pre-warm's absence
    restores the old fail-closed ceiling, never a false-VERIFIED).

    Returns
    -------
    None
        ``sys.modules`` gains the warmed torch modules on success.
    """

    global _LAZY_TORCH_IMPORTS_WARMED

    if _LAZY_TORCH_IMPORTS_WARMED:
        return
    # The RNG-channel fix above (running this BEFORE the monitor's own
    # patches install) only protects THAT one census. The broader capture
    # logging/dispatch window is already armed by the time this runs (it
    # fires from inside ``run_and_log_inputs_through_model``), so the
    # cascade's own tensor ops -- torch._dynamo.variables.torch_function's
    # module-level ``populate_builtin_to_tensor_fn_map()`` calls
    # ``torch.ones`` plus several builtin unary ops (``abs``, ...) as an
    # import side effect -- went through TorchLens's OWN wrapped functions
    # and were recorded as unattributable "unmodeled_tensor_return" /
    # "caught_exception_control" dispatch-census facts, permanently
    # ceilinging the capture's witness completeness even though nothing in
    # the user's model caused them. ``pause_logging()`` is the general
    # mechanism for exactly this (critical invariant 2): it hides the
    # cascade from every capture-time witness, not just the RNG monitor.
    from .._state import pause_logging

    warmed = True
    with pause_logging():
        for module_name in ("torch._compile", "torch._dynamo", "torch.backends.opt_einsum"):
            try:
                importlib.import_module(module_name)
            except Exception:
                warmed = False
    _LAZY_TORCH_IMPORTS_WARMED = warmed


@contextlib.contextmanager
def force_eager_stance_scope() -> Iterator[bool]:
    """Force compiled callables to run their original eager Python for a scope.

    Yields
    ------
    bool
        ``True`` while a ``torch.compiler.set_stance("force_eager")`` stance is
        active for the scope; ``False`` when the capability is structurally
        unavailable (torch < 2.6), when Dynamo has never been imported in this
        process (so the stance would be a semantic no-op bought at Dynamo's
        import cost), or when entering the stance failed.

    Notes
    -----
    The stance (public API, torch >= 2.6; verified experimentally in the
    2026-08-12 design-review compile reconcile) runs the ORIGINAL Python inside every
    compiled callable reached in the scope, triggers zero new compiles during
    the scope (including on never-seen input shapes), and leaves every warm
    compiled artifact reproduced bitwise after exit. Reading the probe never
    imports ``torch._dynamo``. Skipping the stance when Dynamo is absent is
    exact for every callable that EXISTS at scope entry (compiling one imports
    Dynamo), but NOT for a callable created inside the scope: a forward whose
    first ``torch.compile`` happens mid-capture imports Dynamo after this
    check, so that callable runs compiled and is bypassed-and-disclosed
    (``dynamo_region_not_logged``) rather than eager-logged.
    """

    if not HAS_SET_STANCE or "torch._dynamo" not in sys.modules:
        yield False
        return
    # R07: the stance is APPLIED by ``set_stance(...)`` construction (its
    # function-call form), so ``__exit__`` ownership must be established in the
    # same guarded region as the construction -- binding the handle outside the
    # try left an async-interrupt window between the bind and the try entry
    # that stranded the process-wide stance. Once ``stance`` is bound, every
    # exit path (including BaseException) reaches the ``finally``. The one
    # residual is an interrupt INSIDE ``set_stance`` after it applied but
    # before it returned: without a public prior-stance getter that window is
    # unrecoverable and is accepted as atomic-API exposure.
    stance = None
    try:
        try:
            stance = torch.compiler.set_stance("force_eager")
        except Exception:
            mark_torch_capability_missing(
                "HAS_SET_STANCE",
                "torch.compiler.set_stance failed to engage; compiled callables keep the "
                "pre-2.6 bypass-and-disclose capture path",
            )
            yield False
            return
        # Constructing ``set_stance`` already applied the stance;
        # ``__enter__`` is a no-op today and keeps this robust if a future
        # torch moves application into the context protocol.
        stance.__enter__()
        yield True
    finally:
        if stance is not None:
            stance.__exit__(None, None, None)


def get_dynamo_compile_counters(*, force_probe: bool = False) -> Any | None:
    """Return Dynamo's live compilation-event counters mapping, if available.

    Parameters
    ----------
    force_probe:
        Import ``torch._dynamo.utils`` even when Dynamo has never been imported
        in this process.

    Returns
    -------
    Any | None
        The live ``torch._dynamo.utils.counters`` mapping (``counters["frames"]
        ["total"]`` counts frame compilations, including recompiles), or
        ``None`` when unavailable or when the lazy default defers the probe.

    Notes
    -----
    Same never-import-on-the-hot-path contract as
    :func:`get_tracing_tensor_types`: without ``force_probe`` the probe waits
    until ``torch._dynamo`` is already loaded, because a process that never
    imported Dynamo has no compilation events to count.
    """

    global HAS_DYNAMO_COMPILE_COUNTERS, _DYNAMO_COMPILE_COUNTERS
    global _DYNAMO_COMPILE_COUNTERS_PROBED

    if not _DYNAMO_COMPILE_COUNTERS_PROBED:
        if not force_probe and "torch._dynamo" not in sys.modules:
            return None
        try:
            candidate = _import_module_attr_or_none("torch._dynamo.utils", "counters")
        except RuntimeError:
            candidate = None
        if candidate is not None and hasattr(candidate, "__getitem__"):
            _DYNAMO_COMPILE_COUNTERS = candidate
        HAS_DYNAMO_COMPILE_COUNTERS = _DYNAMO_COMPILE_COUNTERS is not None
        _DYNAMO_COMPILE_COUNTERS_PROBED = True
    return _DYNAMO_COMPILE_COUNTERS


def probe_c10d_capabilities(*, force_probe: bool = False) -> dict[str, bool]:
    """Probe the c10d group-registry / group-seq / abort surfaces lazily.

    Parameters
    ----------
    force_probe:
        Probe even when ``torch.distributed`` has never been imported.

    Returns
    -------
    dict[str, bool]
        The three flag values. Each degradation is named through
        :func:`mark_torch_capability_missing` exactly once.

    Notes
    -----
    * ``HAS_C10D_GROUP_REGISTRY`` -- ``torch.distributed.distributed_c10d._world``
      exposes the live group registry (``pg_map``); restricted registry seeding
      and the two-alive-same-membership refusal read it.
    * ``HAS_C10D_GROUP_SEQ`` -- ``ProcessGroup._get_sequence_number_for_group``
      exists; the redundant ``c10d_group_seq`` cross-check on collective
      boundary records reads it (diagnostic, never correlation authority).
    * ``HAS_C10D_ABORT_PG`` -- ``torch.distributed._abort_process_group`` exists
      and is wrapped observationally by the group-lifecycle wraps.
    """

    global HAS_C10D_GROUP_REGISTRY, _C10D_GROUP_REGISTRY_PROBED
    global HAS_C10D_GROUP_SEQ, _C10D_GROUP_SEQ_PROBED
    global HAS_C10D_ABORT_PG, _C10D_ABORT_PG_PROBED

    needs_probe = not (
        _C10D_GROUP_REGISTRY_PROBED and _C10D_GROUP_SEQ_PROBED and _C10D_ABORT_PG_PROBED
    )
    if needs_probe and (force_probe or "torch.distributed" in sys.modules):
        c10d = _import_module_attr_or_none("torch.distributed", "distributed_c10d")
        if not _C10D_GROUP_REGISTRY_PROBED:
            world = getattr(c10d, "_world", None)
            HAS_C10D_GROUP_REGISTRY = isinstance(getattr(world, "pg_map", None), dict)
            _C10D_GROUP_REGISTRY_PROBED = True
            if not HAS_C10D_GROUP_REGISTRY:
                mark_torch_capability_missing(
                    "HAS_C10D_GROUP_REGISTRY",
                    "restricted registry seeding of pre-arming process groups "
                    "refuses (ambiguous_group_lifetime) instead of proving "
                    "single-generation membership",
                )
        if not _C10D_GROUP_SEQ_PROBED:
            process_group_type = _nested_getattr_or_none(
                torch, ("_C", "_distributed_c10d", "ProcessGroup")
            )
            HAS_C10D_GROUP_SEQ = hasattr(process_group_type, "_get_sequence_number_for_group")
            _C10D_GROUP_SEQ_PROBED = True
            if not HAS_C10D_GROUP_SEQ:
                mark_torch_capability_missing(
                    "HAS_C10D_GROUP_SEQ",
                    "collective boundary records omit the redundant "
                    "c10d_group_seq cross-check field",
                )
        if not _C10D_ABORT_PG_PROBED:
            HAS_C10D_ABORT_PG = callable(getattr(c10d, "_abort_process_group", None))
            _C10D_ABORT_PG_PROBED = True
            if not HAS_C10D_ABORT_PG:
                mark_torch_capability_missing(
                    "HAS_C10D_ABORT_PG",
                    "NCCL abort-recreate lifecycle events are observed only "
                    "through destroy_process_group; unobserved destruction "
                    "remains harmless by ordinal construction",
                )
    return {
        "HAS_C10D_GROUP_REGISTRY": HAS_C10D_GROUP_REGISTRY,
        "HAS_C10D_GROUP_SEQ": HAS_C10D_GROUP_SEQ,
        "HAS_C10D_ABORT_PG": HAS_C10D_ABORT_PG,
    }


def get_tracing_tensor_types(*, force_probe: bool = False) -> tuple[type[Any], ...]:
    """Return the data-free tracing tensor subclasses available in this build.

    Parameters
    ----------
    force_probe:
        Import the tracing-subclass namespaces even when they have never been
        imported in this process.

    Returns
    -------
    tuple[type[Any], ...]
        ``FakeTensor`` / ``FunctionalTensor`` types, empty when unavailable or
        when the lazy default defers the probe.

    Notes
    -----
    Same never-import-on-the-hot-path contract as :func:`get_fsdp_wrapper_type`:
    a live tensor can only *be* one of these classes if its defining module is
    already in ``sys.modules``.
    """

    global HAS_TRACING_TENSOR_TYPES, _TRACING_TENSOR_TYPES, _TRACING_TENSOR_TYPES_PROBED

    if not _TRACING_TENSOR_TYPES_PROBED:
        candidate_modules = tuple(name for name, _attr in _TRACING_TENSOR_CANDIDATES)
        if not force_probe and not any(name in sys.modules for name in candidate_modules):
            return ()
        found: list[type[Any]] = []
        for module_name, attr_name in _TRACING_TENSOR_CANDIDATES:
            try:
                candidate = _import_module_attr_or_none(module_name, attr_name)
            except RuntimeError:
                candidate = None
            if isinstance(candidate, type):
                found.append(candidate)
        _TRACING_TENSOR_TYPES = tuple(found)
        HAS_TRACING_TENSOR_TYPES = bool(_TRACING_TENSOR_TYPES)
        _TRACING_TENSOR_TYPES_PROBED = True
    if not _TRACING_TENSOR_TYPES:
        mark_torch_capability_missing(
            "HAS_TRACING_TENSOR_TYPES",
            "fake/functional tracing-tensor detection falls back to structural name matching",
        )
        return ()
    return _TRACING_TENSOR_TYPES


# Narrow-width float dtypes whose elementwise kernel coverage is incomplete.
# Every one of these is ``dtype.is_floating_point == True``, so code that branches
# on that flag reaches ops torch cannot run for them: on torch 2.13 CPU, ``isinf``,
# ``nan_to_num``, ``allclose``, and every reduction raise ``NotImplementedError``
# while ``isnan``, ``torch.equal``, ``abs``, and ``.float()`` work. Probed by name
# off ``torch`` (no submodule import, no version parsing) because the set grew over
# 2.x: e4m3fn/e5m2 arrived first, the "fnuz" variants next, e8m0fnu latest.
_NARROW_FLOAT_DTYPE_NAMES: tuple[str, ...] = (
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
)


def get_fp8_dtypes(*, force_probe: bool = False) -> frozenset[Any]:
    """Return the fp8 dtypes this torch build exposes.

    Parameters
    ----------
    force_probe:
        Accepted for symmetry with the other capability probes. The fp8 dtypes are
        plain ``torch`` attributes, so there is never a module to import and the
        probe is unconditional.

    Returns
    -------
    frozenset[Any]
        ``torch.dtype`` objects for the fp8 variants present, empty on a build with
        none. An empty set degrades every caller to its pre-fp8 behavior.

    Notes
    -----
    Probed once and cached; membership is checked per comparison on the validation
    path, so it must stay a hash lookup.
    """

    global HAS_FP8_DTYPES, _FP8_DTYPES, _FP8_DTYPES_PROBED

    del force_probe  # No import is required; the probe is always safe to run.
    if not _FP8_DTYPES_PROBED:
        found = []
        for name in _NARROW_FLOAT_DTYPE_NAMES:
            candidate = torch_attr(name)
            if isinstance(candidate, torch.dtype):
                found.append(candidate)
        _FP8_DTYPES = frozenset(found)
        HAS_FP8_DTYPES = bool(_FP8_DTYPES)
        _FP8_DTYPES_PROBED = True
    if not _FP8_DTYPES:
        mark_torch_capability_missing(
            "HAS_FP8_DTYPES",
            "fp8 dtypes are absent from this torch build; no fp8 comparison upcast is needed",
        )
    return _FP8_DTYPES


def get_dynamo_optimized_module_type(*, force_probe: bool = False) -> type[Any] | None:
    """Return Dynamo's private ``OptimizedModule`` type when available.

    Parameters
    ----------
    force_probe:
        Import ``torch._dynamo.eval_frame`` even when Dynamo has never been
        imported in this process. Diagnostic surfaces set this to report the
        real build capability; capture paths keep the default.

    Returns
    -------
    type[Any] | None
        Dynamo OptimizedModule type, or ``None`` when unavailable or when the
        lazy default defers the probe.

    Notes
    -----
    A live object can only be an ``OptimizedModule`` instance if
    ``torch._dynamo.eval_frame`` is already in ``sys.modules`` (Python
    registers a module before any class from it can be instantiated), so when
    it is absent the default path reports ``None`` without paying the
    ``torch._dynamo`` import (~0.9s and hundreds of modules cold).
    """

    global HAS_DYNAMO_OPTIMIZED_MODULE, _DYNAMO_OPTIMIZED_MODULE_PROBED
    global _DYNAMO_OPTIMIZED_MODULE_TYPE

    if not _DYNAMO_OPTIMIZED_MODULE_PROBED:
        if not force_probe and "torch._dynamo.eval_frame" not in sys.modules:
            return None
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


def _ensure_dynamo_orig_callable_marker_probed() -> None:
    """Populate the lazy Dynamo original-callable marker capability flag.

    Returns
    -------
    None
        Module-level capability state is updated at most once.
    """
    global HAS_DYNAMO_ORIG_CALLABLE_MARKER, _DYNAMO_ORIG_CALLABLE_MARKER_PROBED

    if _DYNAMO_ORIG_CALLABLE_MARKER_PROBED:
        return
    HAS_DYNAMO_ORIG_CALLABLE_MARKER = _probe_dynamo_orig_callable_marker()
    _DYNAMO_ORIG_CALLABLE_MARKER_PROBED = True


def is_dynamo_compiled_callable(value: Any) -> bool:
    """Return whether a non-module callable carries Dynamo's compile marker.

    Parameters
    ----------
    value:
        Candidate model or callable supplied to a capture API.

    Returns
    -------
    bool
        True only for a callable with Dynamo's direct original-callable marker
        on a runtime that exposes the marker contract.

    Notes
    -----
    The direct ``__dict__`` read avoids invoking arbitrary descriptors. The
    capability probe remains lazy and runs only after the cheap marker lookup
    succeeds, so ordinary eager-module capture pays no Dynamo import cost.
    """
    namespace = getattr(value, "__dict__", None)
    if not isinstance(namespace, dict):
        return False
    original = namespace.get("_torchdynamo_orig_callable")
    if not callable(original):
        return False
    _ensure_dynamo_orig_callable_marker_probed()
    return HAS_DYNAMO_ORIG_CALLABLE_MARKER


def get_dynamo_explain() -> Callable[..., Any] | None:
    """Return ``torch._dynamo.explain`` after a lazy capability probe.

    Returns
    -------
    Callable[..., Any] | None
        Dynamo explain callable, or ``None`` when unavailable.
    """

    if not HAS_DYNAMO_EXPLAIN:
        return None
    explain = _import_module_attr_or_none("torch._dynamo", "explain")
    if not callable(explain):
        mark_torch_capability_missing(
            "HAS_DYNAMO_EXPLAIN",
            "graph-break diagnostics are disabled",
        )
        return None
    return explain


def run_dynamo_explain(
    model: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Execute Dynamo explain across old and new invocation conventions.

    Parameters
    ----------
    model:
        Eager model or callable to inspect.
    args:
        Positional model arguments.
    kwargs:
        Keyword model arguments.

    Returns
    -------
    Any
        Raw version-specific explain output.

    Raises
    ------
    RuntimeError
        If the explain capability is unavailable.
    """

    explain = get_dynamo_explain()
    if explain is None:
        raise RuntimeError("torch._dynamo.explain is unavailable in this torch runtime")
    # Feature-detect the calling convention from the *signature* instead of
    # treating any TypeError raised by ``explain(model)`` as a legacy-signature
    # signal. The old blanket ``except TypeError`` misread an internal TypeError
    # (raised inside the modern one-arg convention) as a signature mismatch,
    # silently swallowed it, and re-invoked ``explain(model, *args, **kwargs)``
    # -- a double side effect that also masked the real error.
    if _explain_accepts_single_callable(explain):
        candidate = explain(model)
        if callable(candidate):
            return candidate(*args, **kwargs)
        return candidate
    return explain(model, *args, **kwargs)


def _explain_accepts_single_callable(explain: Callable[..., Any]) -> bool:
    """Return whether ``explain(model)`` is a valid one-argument call.

    Modern ``torch._dynamo.explain`` binds ``explain(f)`` (returning a callable)
    while the legacy convention requires ``explain(f, *args, **kwargs)``. We
    decide via :func:`inspect.signature`/``bind`` so that a genuine internal
    TypeError from the modern one-argument convention propagates instead of
    being mistaken for a signature mismatch.

    Parameters
    ----------
    explain:
        The ``torch._dynamo.explain`` callable resolved for this runtime.

    Returns
    -------
    bool
        True when ``explain(model)`` binds against the signature. When the
        signature cannot be introspected we conservatively assume the modern
        one-argument convention (torch's declared floor), which keeps internal
        errors visible rather than re-invoking with a hidden double side effect.
    """

    try:
        signature = inspect.signature(explain)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind(object())
    except TypeError:
        return False
    return True


def _dynamo_break_source(reason: Any) -> tuple[str | None, int | None]:
    """Extract one source location from a version-specific break reason.

    Parameters
    ----------
    reason:
        Dynamo break-reason object or mapping.

    Returns
    -------
    tuple[str | None, int | None]
        Source file and one-indexed line number when supplied by Dynamo.
    """

    user_stack = (
        reason.get("user_stack")
        if isinstance(reason, dict)
        else getattr(reason, "user_stack", None)
    )
    for frame in user_stack or ():
        source_file = (
            frame.get("filename")
            if isinstance(frame, dict)
            else getattr(frame, "filename", getattr(frame, "file", None))
        )
        line_number = (
            frame.get("lineno", frame.get("line"))
            if isinstance(frame, dict)
            else getattr(frame, "lineno", getattr(frame, "line", None))
        )
        if source_file is not None and isinstance(line_number, int):
            return str(source_file), line_number
    source_file = (
        reason.get("filename") if isinstance(reason, dict) else getattr(reason, "filename", None)
    )
    line_number = (
        reason.get("lineno", reason.get("line"))
        if isinstance(reason, dict)
        else getattr(reason, "lineno", getattr(reason, "line", None))
    )
    if source_file is not None and isinstance(line_number, int):
        return str(source_file), line_number
    return None, None


def normalize_dynamo_explain_output(output: Any) -> tuple[_NormalizedDynamoBreak, ...]:
    """Normalize supported Dynamo explain result shapes without version parsing.

    Supported shapes are attribute objects and mappings carrying
    ``break_reasons``, plus the historical tuple whose fourth element contains
    break reasons. Individual reasons may be objects, mappings, or strings.

    Parameters
    ----------
    output:
        Raw result from :func:`run_dynamo_explain`.

    Returns
    -------
    tuple[_NormalizedDynamoBreak, ...]
        Version-neutral graph-break evidence.

    Raises
    ------
    _DynamoExplainOutputError
        If the runtime result shape cannot be interpreted safely.
    """

    break_count: Any = None
    if isinstance(output, dict):
        reasons = output.get("break_reasons")
        break_count = output.get("graph_break_count")
    elif hasattr(output, "break_reasons"):
        reasons = getattr(output, "break_reasons")
        break_count = getattr(output, "graph_break_count", None)
    elif isinstance(output, tuple) and len(output) >= 4:
        reasons = output[3]
        break_count = output[2]
    else:
        raise _DynamoExplainOutputError(
            f"unsupported torch._dynamo.explain output type {type(output).__name__}"
        )
    if reasons is None:
        if break_count == 0:
            return ()
        raise _DynamoExplainOutputError("Dynamo explain output omitted break_reasons")
    if not isinstance(reasons, (list, tuple)):
        raise _DynamoExplainOutputError("Dynamo break_reasons is not a sequence")

    normalized: list[_NormalizedDynamoBreak] = []
    for item in reasons:
        reason_text: object | None
        if isinstance(item, str):
            reason_text = item
        elif isinstance(item, dict):
            reason_text = item.get("reason")
        else:
            reason_text = getattr(item, "reason", None)
        if reason_text is None:
            raise _DynamoExplainOutputError(
                f"unsupported Dynamo break reason type {type(item).__name__}"
            )
        source_file, line_number = _dynamo_break_source(item)
        normalized.append(_NormalizedDynamoBreak(str(reason_text), source_file, line_number))
    return tuple(normalized)


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

    from ._type_sequence_slot import clear_sequence_item_slot, sequence_slot_layout_problem

    problem = sequence_slot_layout_problem(torch.Tensor, b"Tensor")
    if problem is not None:
        mark_torch_capability_missing("HAS_TENSOR_SEQUENCE_SLOT_FIX", problem)
        return False
    clear_sequence_item_slot(torch.Tensor)
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
        return bool(torch.is_autocast_cpu_enabled())
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
        return torch.get_autocast_cpu_dtype()
    if device_type == "cuda":
        return torch.get_autocast_gpu_dtype()
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
# Every control below is a PUBLIC torch surface present since before the 2.1
# support floor (r-b7 R42-1 retired the six HAS_* flags that used to guard
# them: their False branches were unreachable on any supported torch, and the
# CUDA-named ones read True even on CUDA-less wheels) -- WITH ONE EXCEPTION:
# ``torch.utils.deterministic.fill_uninitialized_memory`` (just below)
# postdates the 2.1 floor -- the ``torch.utils.deterministic`` submodule does
# not exist at all on torch 2.1.x -- and is feature-detected through
# ``HAS_DETERMINISTIC_FILL_FLAG``. Snapshots record every control
# affirmatively; ``None`` survives as the feature-detected absence of the
# fill flag and, in the APPLY direction, as schema tolerance for artifacts
# recorded by older producers.

_torch_deterministic_module: types.ModuleType | None
if HAS_DETERMINISTIC_FILL_FLAG:
    import torch.utils.deterministic as _torch_deterministic_module
else:
    _torch_deterministic_module = None


def read_fill_uninitialized_memory() -> bool | None:
    """Return the deterministic uninit-memory fill flag, or ``None`` if absent.

    THE one sanctioned read of ``torch.utils.deterministic.fill_uninitialized_memory``
    (a module-``__getattr__`` property invisible to static typing): the ambient
    snapshot and the producer-side determinism refinement both route here.

    Returns
    -------
    bool | None
        The live flag value, or ``None`` on torch 2.1.x, where the
        ``torch.utils.deterministic`` submodule does not exist
        (``HAS_DETERMINISTIC_FILL_FLAG`` is ``False``) -- a healthy old
        install, not a degradation.
    """

    if _torch_deterministic_module is None:
        return None
    # ``fill_uninitialized_memory`` is a module-``__getattr__`` property;
    # static typing has no stub for it, so read it dynamically rather than
    # via attribute access (``getattr`` is accurate here, not a workaround).
    return bool(getattr(_torch_deterministic_module, "fill_uninitialized_memory"))


def write_fill_uninitialized_memory(value: bool) -> None:
    """Set the deterministic uninit-memory fill flag (caller checks the flag).

    No-ops on torch 2.1.x, where the ``torch.utils.deterministic`` submodule
    does not exist (``HAS_DETERMINISTIC_FILL_FLAG`` is ``False``) -- there is
    nothing to set and nothing to degrade.
    """

    if _torch_deterministic_module is None:
        return
    # See the matching comment in ``read_fill_uninitialized_memory``: this is
    # a module-``__getattr__`` property with no static stub.
    setattr(_torch_deterministic_module, "fill_uninitialized_memory", bool(value))


#: The ``fp32_precision`` controls (``HAS_FP32_PRECISION_CONTROLS``), root
#: first: restoring the children after the root pins every leaf exactly.
_FP32_PRECISION_OWNERS: tuple[str, ...] = (
    "",
    "cuda.matmul",
    "cudnn",
    "cudnn.conv",
    "cudnn.rnn",
    "mkldnn",
    "mkldnn.matmul",
    "mkldnn.conv",
    "mkldnn.rnn",
)


def _fp32_precision_owner(path: str) -> Any:
    """Return the ``torch.backends`` object owning ``path``'s control, or ``None``."""

    owner: Any = torch.backends
    for part in filter(None, path.split(".")):
        owner = getattr(owner, part, None)
    return owner if hasattr(owner, "fp32_precision") else None


def snapshot_fp32_precision_controls() -> dict[str, str]:
    """Return every ``fp32_precision`` control value, keyed by backend path.

    The legacy ``set_float32_matmul_precision`` and ``allow_tf32`` setters
    write these fields on torch >= 2.9, so re-applying a legacy snapshot does
    not restore them (``'none'`` reads back as ``'ieee'``). Callers that set
    the legacy controls transactionally snapshot these too. Empty when
    ``HAS_FP32_PRECISION_CONTROLS`` is ``False``.
    """

    if not HAS_FP32_PRECISION_CONTROLS:
        return {}
    snapshot: dict[str, str] = {}
    for path in _FP32_PRECISION_OWNERS:
        owner = _fp32_precision_owner(path)
        if owner is not None:
            snapshot[path] = str(owner.fp32_precision)
    return snapshot


def restore_fp32_precision_controls(snapshot: dict[str, str]) -> None:
    """Re-apply a :func:`snapshot_fp32_precision_controls` result exactly.

    Writes only the fields whose value changed, root first.
    """

    for path in _FP32_PRECISION_OWNERS:
        owner = _fp32_precision_owner(path) if path in snapshot else None
        if owner is not None and str(owner.fp32_precision) != snapshot[path]:
            owner.fp32_precision = snapshot[path]


#: Controls no legacy setter writes ('none' under any legacy-only history,
#: measured on torch 2.9.1 and 2.13.0): another value has no v2 record field.
_FP32_PRECISION_LEGACY_UNWRITTEN: tuple[str, ...] = (
    "",
    "cudnn",
    "mkldnn",
    "mkldnn.conv",
    "mkldnn.rnn",
)

#: Ambient-record fields that are legacy views of the ``fp32_precision`` controls.
_LEGACY_FP32_GETTERS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("float32_matmul_precision", lambda: str(torch.get_float32_matmul_precision())),
    ("cuda_matmul_allow_tf32", lambda: bool(torch.backends.cuda.matmul.allow_tf32)),
    ("cudnn_allow_tf32", lambda: bool(torch.backends.cudnn.allow_tf32)),
)

#: Session-only ambient-snapshot key (never a record field); the producer refuses on it.
AMBIENT_FP32_UNREPRESENTABLE_KEY = "_fp32_precision_unrepresentable"


def read_legacy_fp32_controls() -> tuple[dict[str, Any], tuple[str, ...]]:
    """Read the legacy fp32 views and name what the v2 ambient record cannot hold.

    Returns
    -------
    tuple[dict[str, Any], tuple[str, ...]]
        The legacy values (``None`` where the getter raised) and the
        unrepresentable parts of the live policy: each legacy field whose getter
        raised, then ``fp32_precision[<path>]=<value>`` for each control no legacy
        setter writes that is not ``'none'``. On torch >= 2.9 the legacy getters
        raise ``RuntimeError`` when the ``fp32_precision`` controls have no legacy
        equivalent (mixed cuDNN conv/RNN TF32, matmul or generic set via the new
        API); without those controls a raise is unexpected and propagates.
    """

    values: dict[str, Any] = {}
    unrepresentable: list[str] = []
    for name, read in _LEGACY_FP32_GETTERS:
        try:
            values[name] = read()
        except RuntimeError:
            if not HAS_FP32_PRECISION_CONTROLS:
                raise
            values[name] = None
            unrepresentable.append(name)
    controls = snapshot_fp32_precision_controls()
    unrepresentable.extend(
        f"fp32_precision[{path or 'generic'}]={controls[path]}"
        for path in _FP32_PRECISION_LEGACY_UNWRITTEN
        if controls.get(path, "none") != "none"
    )
    return values, tuple(unrepresentable)


def tensor_any_over_dims(tensor: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Reduce a boolean tensor with ``any`` over multiple axes at once.

    ``Tensor.any(dim=<tuple>)`` postdates the torch 2.1 floor (torch 2.1 only
    accepts a single int axis). THE one sanctioned multi-axis ``any`` reduction:
    product code that needs to OR-reduce several axes together routes here
    instead of calling ``tensor.any(dim=dims)`` directly.

    Parameters
    ----------
    tensor:
        Tensor to reduce (typically boolean).
    dims:
        Axes to reduce over. An empty tuple returns ``tensor`` unchanged.

    Returns
    -------
    torch.Tensor
        ``tensor`` with every axis in ``dims`` reduced away, identical to
        ``tensor.any(dim=dims)`` on torch>=2.2.

    Notes
    -----
    On torch 2.1 (``HAS_REDUCE_TUPLE_DIM`` is ``False``), falls back to
    sequential single-axis reduction, removing axes from highest to lowest so
    an already-normalized smaller target axis never shifts as a larger one is
    removed.
    """

    if not dims:
        return tensor
    if get_reduce_tuple_dim_support():
        return tensor.any(dim=dims)
    ndim = tensor.ndim
    result = tensor
    for axis in sorted({axis % ndim for axis in dims}, reverse=True):
        result = result.any(dim=axis)
    return result


def get_mha_fastpath_enabled() -> bool | None:
    """Return the public MHA fastpath switch, or ``None`` if absent.

    THE one sanctioned read of ``torch.backends.mha.get_fastpath_enabled()``.

    Returns
    -------
    bool | None
        The live switch value, or ``None`` on torch 2.1-2.2, where
        ``torch.backends.mha`` does not exist (``HAS_MHA_FASTPATH_SWITCH`` is
        ``False``) -- a healthy old install, not a degradation.
    """

    if not get_mha_fastpath_switch_support():
        return None
    return bool(torch.backends.mha.get_fastpath_enabled())


def set_mha_fastpath_enabled(value: bool) -> None:
    """Set the public MHA fastpath switch (caller checks the flag first).

    No-ops on torch 2.1-2.2, where ``torch.backends.mha`` does not exist
    (``HAS_MHA_FASTPATH_SWITCH`` is ``False``) -- there is nothing to set.
    """

    if not get_mha_fastpath_switch_support():
        return
    torch.backends.mha.set_fastpath_enabled(bool(value))


@contextlib.contextmanager
def force_mha_slow_path(model: torch.nn.Module) -> Iterator[None]:
    """Hold the fused MHA/TransformerEncoderLayer fast path OFF for ``model``.

    Uses the public ``torch.backends.mha`` switch when available
    (``HAS_MHA_FASTPATH_SWITCH``); otherwise falls back to setting the
    ``training`` ATTRIBUTE directly (never the recursive ``.train()``
    method, which would also flip nested dropout/norm submodules and could
    change their op counts) to ``True`` on every
    ``torch.nn.MultiheadAttention`` and ``torch.nn.TransformerEncoderLayer``
    instance reachable from ``model`` -- a REAL fallback, not a degradation:
    both fused fast paths independently check their OWN module's
    ``self.training`` on every supported torch version (the layer's native
    op and its inner attention's native op gate separately), so the public
    switch merely adds an earlier, version-independent gate on top. Every
    flag this context manager touches is restored to its prior value on
    exit, success or failure.

    Parameters
    ----------
    model:
        Module whose fused-attention submodules should run the unfused path
        for the duration of the ``with`` block.

    Yields
    ------
    None
        Nothing; the context is entered for its side effect.
    """

    prior_switch = get_mha_fastpath_enabled()
    fastpath_modules = [
        module
        for module in model.modules()
        if isinstance(module, (torch.nn.MultiheadAttention, torch.nn.TransformerEncoderLayer))
    ]
    saved_training = [(module, module.training) for module in fastpath_modules]
    try:
        if prior_switch is not None:
            set_mha_fastpath_enabled(False)
        else:
            for module in fastpath_modules:
                module.training = True
        yield
    finally:
        if prior_switch is not None:
            set_mha_fastpath_enabled(prior_switch)
        else:
            for module, was_training in saved_training:
                module.training = was_training


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
        affirmatively (explicit ``False``), never omitted. Legacy fp32 views with
        no legacy equivalent are ``None`` (see :func:`read_legacy_fp32_controls`).
    """

    legacy_fp32, _ = read_legacy_fp32_controls()
    snapshot: dict[str, Any] = {
        "default_dtype": str(torch.get_default_dtype()),
        "default_device": str(getattr(torch, "get_default_device", lambda: "cpu")()),
        "float32_matmul_precision": legacy_fp32["float32_matmul_precision"],
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cuda_matmul_allow_tf32": legacy_fp32["cuda_matmul_allow_tf32"],
        "cudnn_allow_tf32": legacy_fp32["cudnn_allow_tf32"],
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
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
        If a recorded default dtype is unavailable on this runtime. Every
        other decision-E control is a public torch surface older than the 2.1
        support floor, so recorded values apply directly (r-b7 R42-1).
    """

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
        warn_only = bool(values.get("deterministic_algorithms_warn_only") or False)
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=warn_only)
    # ``torch.backends.cuda.matmul.allow_tf32`` and
    # ``torch.set_float32_matmul_precision`` are two views of ONE underlying
    # control (setting ``allow_tf32=True`` coerces precision to "high"). Apply
    # the coarse Boolean FIRST so the finer-grained recorded precision value
    # wins; the pair is snapshotted together, so the final state is coherent.
    if HAS_FP32_PRECISION_CONTROLS:
        # A legacy-representable capture had these controls at 'none' (no legacy
        # setter writes them): reset any caller value before the legacy views.
        controls = snapshot_fp32_precision_controls()
        unwritten = [path for path in _FP32_PRECISION_LEGACY_UNWRITTEN if path in controls]
        restore_fp32_precision_controls({**controls, **dict.fromkeys(unwritten, "none")})
    cuda_tf32 = values.get("cuda_matmul_allow_tf32")
    if cuda_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(cuda_tf32)
    precision = values.get("float32_matmul_precision")
    if precision is not None:
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
        setattr(torch.backends.cudnn, attr, bool(recorded))
    for field_name, setter in (
        ("flash_sdp_enabled", "enable_flash_sdp"),
        ("mem_efficient_sdp_enabled", "enable_mem_efficient_sdp"),
        ("math_sdp_enabled", "enable_math_sdp"),
    ):
        recorded = values.get(field_name)
        if recorded is None:
            continue
        getattr(torch.backends.cuda, setter)(bool(recorded))
    # r53 hon_2: the deterministic uninit-memory fill knob restores through the
    # ordinary setter path (module property setter; snapshot/apply transactional).
    fill_uninitialized = values.get("fill_uninitialized_memory")
    if fill_uninitialized is not None:
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
# the TYPE's ``__match_args__`` declaration and, for types that do not declare one,
# an identity round-trip over the type's member descriptors. ``repr()`` parsing is
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
    # No usable ``__match_args__``: derive the name -> index bijection by IDENTITY
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
