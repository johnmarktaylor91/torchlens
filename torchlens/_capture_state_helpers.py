"""Internal model state, cache, and input helpers for public trace capture."""

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import hashlib
import json
import os
import types
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import torch
from torch import nn

from . import _state
from .data_classes.trace import Trace
from .utils._torch_compat import get_dynamo_optimized_module_type


def _clone_state_dict_with_metadata(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Clone a module ``state_dict`` while preserving PyTorch metadata.

    Parameters
    ----------
    model:
        Module whose state should be cloned.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Detached tensor clones plus any private ``_metadata`` needed by module
        implementations such as torchvision MNASNet during ``load_state_dict``.
    """

    original_state = model.state_dict()
    cloned_state = OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in original_state.items()
    )
    if hasattr(original_state, "_metadata"):
        cloned_state._metadata = copy.deepcopy(original_state._metadata)  # type: ignore[attr-defined]
    return cloned_state


_VALIDATION_DEEPCOPY_WARNING_TYPES: set[type[nn.Module]] = set()
_PLAIN_ATTR_IGNORED_NAMES = frozenset({"_parameters", "_buffers", "_modules"})
_PLAIN_ATTR_MAX_CONTAINER_ITEMS = 128
_PLAIN_ATTR_MAX_TENSOR_NUMEL = 4096
_COMPILED_MODEL_UNWRAP_WARNED = False


@dataclasses.dataclass(frozen=True)
class _PlainAttrIdentitySnapshot:
    """Identity snapshot for opaque but identity-stable plain attributes.

    Parameters
    ----------
    value:
        Original object to restore if the plain attribute is reassigned.
    value_type_name:
        Human-readable type name for diagnostics.
    """

    value: Any
    value_type_name: str


@dataclasses.dataclass(frozen=True)
class _PlainAttrManagedTensorSnapshot:
    """Snapshot marker for PyTorch-managed derived tensor attributes.

    Parameters
    ----------
    module:
        Owning module that exposes the derived tensor attribute.
    name:
        Attribute name on ``module``.
    shape:
        Tensor shape at snapshot time.
    dtype:
        Tensor dtype at snapshot time.
    device:
        Tensor device at snapshot time.
    manager:
        Human-readable manager kind for diagnostics.
    """

    module: nn.Module
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    manager: str


@dataclasses.dataclass(frozen=True)
class _CompiledSubmoduleSwap:
    """Snapshot of one temporarily unwrapped compiled submodule slot.

    Parameters
    ----------
    parent:
        Parent module that owns the child slot.
    name:
        Child name inside ``parent._modules``.
    compiled_module:
        Original compiled wrapper to restore.
    """

    parent: nn.Module
    name: str
    compiled_module: nn.Module


def reset_compiled_model_unwrap_warning_state() -> None:
    """Reset the process-local compiled-model unwrap warning flag.

    Returns
    -------
    None
        The next compiled-model unwrap emits the user-facing note again.
    """

    global _COMPILED_MODEL_UNWRAP_WARNED

    _COMPILED_MODEL_UNWRAP_WARNED = False


def _warn_compiled_model_unwrapped_once() -> None:
    """Emit the compiled-model eager-source note at most once per process.

    Returns
    -------
    None
        Emits a ``UserWarning`` only on the first call in the process.
    """

    global _COMPILED_MODEL_UNWRAP_WARNED

    if _COMPILED_MODEL_UNWRAP_WARNED:
        return
    _COMPILED_MODEL_UNWRAP_WARNED = True
    warnings.warn(
        "TorchLens: compiled model detected; tracing the eager source module it wraps "
        "(torch.compile semantics such as fusion are not traced).",
        UserWarning,
        stacklevel=3,
    )


def _compiled_model_orig_module(model: nn.Module) -> nn.Module | None:
    """Return the eager source module for a Dynamo compiled wrapper, if present.

    Parameters
    ----------
    model:
        Module to inspect.

    Returns
    -------
    nn.Module | None
        ``model._orig_mod`` when ``model`` is an OptimizedModule and the original
        object is an ``nn.Module``; otherwise ``None``.
    """

    optimized_module_type = get_dynamo_optimized_module_type()
    if optimized_module_type is None or not isinstance(model, optimized_module_type):
        return None
    orig_mod = getattr(model, "_orig_mod", None)
    if isinstance(orig_mod, nn.Module):
        return orig_mod
    return None


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the eager source module for a top-level compiled wrapper.

    Parameters
    ----------
    model:
        User-supplied model.

    Returns
    -------
    nn.Module
        ``model._orig_mod`` when ``model`` is a Dynamo OptimizedModule, otherwise
        ``model`` unchanged.
    """

    orig_mod = _compiled_model_orig_module(model)
    if orig_mod is None:
        return model
    _warn_compiled_model_unwrapped_once()
    return orig_mod


@contextmanager
def unwrap_compiled_submodules(model: nn.Module) -> Iterator[None]:
    """Temporarily replace compiled child slots with their eager source modules.

    Parameters
    ----------
    model:
        Root model whose descendants should execute through eager modules during
        TorchLens capture.

    Yields
    ------
    None
        Control while compiled child slots are unwrapped.
    """

    swaps: list[_CompiledSubmoduleSwap] = []
    try:
        traversal_queue: list[nn.Module] = [model]
        seen_module_ids: set[int] = set()
        while traversal_queue:
            parent = traversal_queue.pop()
            parent_id = id(parent)
            if parent_id in seen_module_ids:
                continue
            seen_module_ids.add(parent_id)
            for child_name, child_module in list(parent._modules.items()):
                if child_module is None:
                    continue
                orig_mod = _compiled_model_orig_module(child_module)
                if orig_mod is None:
                    traversal_queue.append(child_module)
                    continue
                swaps.append(
                    _CompiledSubmoduleSwap(
                        parent=parent,
                        name=child_name,
                        compiled_module=child_module,
                    )
                )
                parent._modules[child_name] = orig_mod
                traversal_queue.append(orig_mod)
    except BaseException:
        for swap in reversed(swaps):
            swap.parent._modules[swap.name] = swap.compiled_module
        raise

    if swaps:
        _warn_compiled_model_unwrapped_once()

    try:
        yield
    finally:
        for swap in reversed(swaps):
            swap.parent._modules[swap.name] = swap.compiled_module


def _is_identity_stable_plain_attr(value: Any) -> bool:
    """Return whether a plain attr should be tracked by object identity.

    Parameters
    ----------
    value:
        Attribute value to classify.

    Returns
    -------
    bool
        True for callable/type/module objects whose mutation signal is
        reassignment rather than value drift.
    """

    return isinstance(
        value,
        (
            types.FunctionType,
            types.MethodType,
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            type,
            nn.Module,
        ),
    ) or callable(value)


def _legacy_parametrization_manager(module: nn.Module, name: str) -> str | None:
    """Return the legacy PyTorch parametrization hook managing an attribute.

    Parameters
    ----------
    module:
        Module that owns the candidate plain tensor attribute.
    name:
        Attribute name to classify.

    Returns
    -------
    str | None
        Manager kind when the attribute is a computed view backed by registered
        module state, otherwise ``None``.
    """

    for hook in getattr(module, "_forward_pre_hooks", {}).values():
        if getattr(hook, "name", None) != name:
            continue
        hook_type = type(hook)
        hook_key = f"{hook_type.__module__}.{hook_type.__name__}"
        if (
            hook_key == "torch.nn.utils.weight_norm.WeightNorm"
            and f"{name}_g" in module._parameters
            and f"{name}_v" in module._parameters
        ):
            return "legacy_weight_norm"
        if (
            hook_key == "torch.nn.utils.spectral_norm.SpectralNorm"
            and f"{name}_orig" in module._parameters
            and f"{name}_u" in module._buffers
            and f"{name}_v" in module._buffers
        ):
            return "legacy_spectral_norm"
    return None


def _parametrize_manager(module: nn.Module, name: str) -> str | None:
    """Return whether PyTorch's parametrization API manages an attribute.

    Parameters
    ----------
    module:
        Module that owns the candidate plain tensor attribute.
    name:
        Attribute name to classify.

    Returns
    -------
    str | None
        Manager kind when the attribute is parametrized, otherwise ``None``.
    """

    try:
        from torch.nn.utils import parametrize
    except ImportError:
        return None
    try:
        if parametrize.is_parametrized(module, name):
            return "parametrize"
    except (AttributeError, ValueError, TypeError):
        return None
    return None


def _managed_plain_tensor_attr_snapshot(
    module: nn.Module,
    name: str,
    value: Any,
) -> _PlainAttrManagedTensorSnapshot | None:
    """Return a marker for a legitimate derived plain tensor attribute.

    Parameters
    ----------
    module:
        Module that owns the candidate attribute.
    name:
        Attribute name to classify.
    value:
        Current attribute value.

    Returns
    -------
    _PlainAttrManagedTensorSnapshot | None
        Snapshot marker when PyTorch registered state manages this plain tensor
        attribute, otherwise ``None``.
    """

    if not isinstance(value, torch.Tensor):
        return None
    manager = _parametrize_manager(module, name) or _legacy_parametrization_manager(module, name)
    if manager is None:
        return None
    return _PlainAttrManagedTensorSnapshot(
        module=module,
        name=name,
        shape=tuple(value.shape),
        dtype=value.dtype,
        device=value.device,
        manager=manager,
    )


def _snapshot_module_plain_attr_value(module: nn.Module, name: str, attr_path: str) -> Any:
    """Return a snapshot for a named plain attribute on a module.

    Parameters
    ----------
    module:
        Module that owns the plain attribute.
    name:
        Attribute name.
    attr_path:
        Human-readable module/attribute path for diagnostics.

    Returns
    -------
    Any
        Snapshot suitable for later comparison and restoration.
    """

    value = getattr(module, name)
    managed_snapshot = _managed_plain_tensor_attr_snapshot(module, name, value)
    if managed_snapshot is not None:
        return managed_snapshot
    return _snapshot_plain_attr_value(value, attr_path)


def _snapshot_plain_attr_value(value: Any, attr_path: str) -> Any:
    """Return a value snapshot for a plain module-tree attribute.

    Parameters
    ----------
    value:
        Attribute value to snapshot.
    attr_path:
        Human-readable module/attribute path for diagnostics.

    Returns
    -------
    Any
        Detached value snapshot that can later be compared and restored.

    Raises
    ------
    RuntimeError
        If the value is not small and value-comparable enough for the fallback
        validation restore. External objects remain unsupported in this path.
    """

    if isinstance(value, (_PlainAttrIdentitySnapshot, _PlainAttrManagedTensorSnapshot)):
        return value
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if _is_identity_stable_plain_attr(value):
        return _PlainAttrIdentitySnapshot(value=value, value_type_name=type(value).__name__)
    if isinstance(value, torch.Tensor):
        if value.numel() > _PLAIN_ATTR_MAX_TENSOR_NUMEL:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its tensor value has {value.numel()} "
                "elements. Non-registered large mutable state is unsupported."
            )
        return value.detach().clone()
    if isinstance(value, list):
        if len(value) > _PLAIN_ATTR_MAX_CONTAINER_ITEMS:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its list has {len(value)} items."
            )
        return [
            _snapshot_plain_attr_value(item, f"{attr_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        if len(value) > _PLAIN_ATTR_MAX_CONTAINER_ITEMS:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its tuple has {len(value)} items."
            )
        return tuple(
            _snapshot_plain_attr_value(item, f"{attr_path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        if len(value) > _PLAIN_ATTR_MAX_CONTAINER_ITEMS:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its dict has {len(value)} items."
            )
        snapshot = {}
        for key, item in value.items():
            try:
                key_snapshot = _snapshot_plain_attr_value(key, f"{attr_path}.<key>")
            except RuntimeError as exc:
                raise RuntimeError(
                    "TorchLens validation deepcopy fallback cannot snapshot plain "
                    f"attribute '{attr_path}' because one of its dict keys is unsupported."
                ) from exc
            try:
                hash(key_snapshot)
            except TypeError as exc:
                raise RuntimeError(
                    "TorchLens validation deepcopy fallback cannot snapshot plain "
                    f"attribute '{attr_path}' because one of its dict keys is unhashable "
                    "after snapshotting."
                ) from exc
            snapshot[key_snapshot] = _snapshot_plain_attr_value(item, f"{attr_path}[{key!r}]")
        return snapshot
    if isinstance(value, (set, frozenset)):
        if len(value) > _PLAIN_ATTR_MAX_CONTAINER_ITEMS:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its set has {len(value)} items."
            )
        snapshot_items = [_snapshot_plain_attr_value(item, f"{attr_path}.<item>") for item in value]
        try:
            return type(value)(snapshot_items)
        except TypeError as exc:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot snapshot plain "
                f"attribute '{attr_path}' because its set items are not hashable."
            ) from exc
    raise RuntimeError(
        "TorchLens validation deepcopy fallback cannot snapshot plain "
        f"attribute '{attr_path}' of type {type(value).__name__}. Non-registered "
        "external objects and other opaque mutable state are unsupported."
    )


def _plain_attr_values_equal(left: Any, right: Any, attr_path: str) -> bool:
    """Return whether two snapshotted plain-attribute values are equal by value.

    Parameters
    ----------
    left:
        First value.
    right:
        Second value.
    attr_path:
        Human-readable module/attribute path for diagnostics.

    Returns
    -------
    bool
        True when values match.

    Raises
    ------
    RuntimeError
        If the values cannot be compared without ambiguity.
    """

    if isinstance(left, _PlainAttrIdentitySnapshot) or isinstance(
        right, _PlainAttrIdentitySnapshot
    ):
        if not isinstance(left, _PlainAttrIdentitySnapshot) or not isinstance(
            right, _PlainAttrIdentitySnapshot
        ):
            return False
        return left.value is right.value
    if isinstance(left, _PlainAttrManagedTensorSnapshot) or isinstance(
        right, _PlainAttrManagedTensorSnapshot
    ):
        if not isinstance(left, _PlainAttrManagedTensorSnapshot) or not isinstance(
            right, _PlainAttrManagedTensorSnapshot
        ):
            return False
        return (
            left.module is right.module
            and left.name == right.name
            and left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and left.manager == right.manager
        )
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return bool(torch.equal(left, right))
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(
            _plain_attr_values_equal(left_item, right_item, f"{attr_path}[{index}]")
            for index, (left_item, right_item) in enumerate(zip(left, right))
        )
    if isinstance(left, tuple) or isinstance(right, tuple):
        if not isinstance(left, tuple) or not isinstance(right, tuple) or len(left) != len(right):
            return False
        return all(
            _plain_attr_values_equal(left_item, right_item, f"{attr_path}[{index}]")
            for index, (left_item, right_item) in enumerate(zip(left, right))
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(
            _plain_attr_values_equal(left[key], right[key], f"{attr_path}[{key!r}]") for key in left
        )
    if isinstance(left, (set, frozenset)) or isinstance(right, (set, frozenset)):
        try:
            return bool(left == right)
        except Exception as exc:
            raise RuntimeError(
                "TorchLens validation deepcopy fallback cannot compare plain "
                f"attribute '{attr_path}' by value."
            ) from exc
    try:
        result = left == right
    except Exception as exc:
        raise RuntimeError(
            "TorchLens validation deepcopy fallback cannot compare plain "
            f"attribute '{attr_path}' by value."
        ) from exc
    if isinstance(result, bool):
        return result
    raise RuntimeError(
        "TorchLens validation deepcopy fallback cannot compare plain "
        f"attribute '{attr_path}' because equality returned {type(result).__name__}."
    )


def _plain_attr_restore_value(snapshot: Any) -> Any:
    """Return the assignable value represented by a plain-attribute snapshot.

    Parameters
    ----------
    snapshot:
        Snapshot produced by :func:`_snapshot_plain_attr_value`.

    Returns
    -------
    Any
        Value to assign back to the module attribute.
    """

    if isinstance(snapshot, _PlainAttrIdentitySnapshot):
        return snapshot.value
    if isinstance(snapshot, _PlainAttrManagedTensorSnapshot):
        return getattr(snapshot.module, snapshot.name)
    return _snapshot_plain_attr_value(snapshot, "<snapshot>")


def _module_plain_attr_names(module: nn.Module) -> set[str]:
    """Return plain attribute names to snapshot for a module.

    Parameters
    ----------
    module:
        Module whose non-registered attributes should be inspected.

    Returns
    -------
    set[str]
        Attribute names excluding registered parameter, buffer, and child
        module storage.
    """

    return set(module.__dict__) - _PLAIN_ATTR_IGNORED_NAMES


class _ModuleTreePlainAttrSnapshot:
    """Snapshot of small value-comparable plain attributes in a module tree."""

    def __init__(self, model: nn.Module) -> None:
        """Capture a model's plain module-tree attributes.

        Parameters
        ----------
        model:
            Model whose ``modules()`` tree should be snapshotted.
        """

        self._entries: list[tuple[nn.Module, str, str, Any]] = []
        self._module_attr_names: dict[int, tuple[nn.Module, set[str], str]] = {}
        module_counts: dict[str, int] = {}
        for module in model.modules():
            module_type = type(module).__name__
            module_index = module_counts.get(module_type, 0)
            module_counts[module_type] = module_index + 1
            module_path = f"{module_type}[{module_index}]"
            attr_names = _module_plain_attr_names(module)
            self._module_attr_names[id(module)] = (module, attr_names, module_path)
            for name in sorted(attr_names):
                attr_path = f"{module_path}.{name}"
                self._entries.append(
                    (
                        module,
                        name,
                        attr_path,
                        _snapshot_module_plain_attr_value(module, name, attr_path),
                    )
                )

    def restore_changed_attrs(self) -> None:
        """Restore attributes whose values changed since the snapshot.

        Raises
        ------
        RuntimeError
            If any attribute cannot be compared, assigned, deleted, or verified
            after restoration.
        """

        for module, original_names, module_path in self._module_attr_names.values():
            added_names = _module_plain_attr_names(module) - original_names
            for name in sorted(added_names):
                try:
                    delattr(module, name)
                except Exception as exc:
                    raise RuntimeError(
                        "TorchLens validation deepcopy fallback could not remove "
                        f"new plain attribute '{module_path}.{name}' before the logged run."
                    ) from exc
        for module, name, attr_path, snapshot in self._entries:
            try:
                current_snapshot = _snapshot_module_plain_attr_value(module, name, attr_path)
            except AttributeError:
                current_snapshot = None
                changed = True
            else:
                changed = not _plain_attr_values_equal(current_snapshot, snapshot, attr_path)
            if not changed:
                continue
            restore_value = _plain_attr_restore_value(snapshot)
            try:
                setattr(module, name, restore_value)
            except Exception as exc:
                raise RuntimeError(
                    "TorchLens validation deepcopy fallback could not restore plain "
                    f"attribute '{attr_path}' before the logged run."
                ) from exc
            restored_snapshot = _snapshot_plain_attr_value(getattr(module, name), attr_path)
            if not _plain_attr_values_equal(restored_snapshot, snapshot, attr_path):
                raise RuntimeError(
                    "TorchLens validation deepcopy fallback restored plain "
                    f"attribute '{attr_path}', but verification by value failed."
                )


def _model_for_ground_truth_validation(
    model: nn.Module,
) -> tuple[nn.Module, _ModuleTreePlainAttrSnapshot | None]:
    """Return an isolated model for the validation ground-truth run.

    Parameters
    ----------
    model:
        Original model that will later be traced by TorchLens.

    Returns
    -------
    tuple[nn.Module, _ModuleTreePlainAttrSnapshot | None]
        A deep copy and no plain-attribute snapshot when possible; otherwise
        the original model plus a snapshot used to restore non-registered
        mutable state before the logged run.
    """

    try:
        return copy.deepcopy(model), None
    except Exception:
        model_type = type(model)
        if model_type not in _VALIDATION_DEEPCOPY_WARNING_TYPES:
            _VALIDATION_DEEPCOPY_WARNING_TYPES.add(model_type)
            warnings.warn(
                "TorchLens validate_forward_pass could not deepcopy the model for the "
                "ground-truth run; falling back to state_dict snapshot/restore. "
                "Non-registered mutable state may cause false negatives for this model.",
                RuntimeWarning,
                stacklevel=3,
            )
        return model, _ModuleTreePlainAttrSnapshot(model)


def decide_recording_of_batch(trace: Trace, predicate: Callable[[Trace], bool]) -> bool:
    """Retroactively keep or discard a captured batch log.

    Parameters
    ----------
    trace:
        Captured log to decide on.
    predicate:
        Callable receiving the log and returning whether to keep it.

    Returns
    -------
    bool
        True when the log was kept.
    """

    keep = bool(predicate(trace))
    if not keep:
        trace.cleanup()
    trace.recording_kept = keep
    return keep


def _qualname_for_model(model: nn.Module) -> str:
    """Return a stable class name for relationship evidence.

    Parameters
    ----------
    model:
        Model being captured.

    Returns
    -------
    str
        Module-qualified class name.
    """

    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__qualname__}"


def _fingerprint_model_weights(model: nn.Module) -> str:
    """Fingerprint model parameter metadata for relationship evidence.

    Phase 4a does not depend on tensor values. The deterministic scheme hashes
    ``(name, shape, dtype)`` for every named parameter, which is stable across
    devices and avoids retaining parameter references.

    Parameters
    ----------
    model:
        Model whose parameters should be fingerprinted.

    Returns
    -------
    str
        SHA-256 hex digest of parameter metadata.
    """

    entries = [
        (name, tuple(param.shape), str(param.dtype)) for name, param in model.named_parameters()
    ]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


def _iter_tensor_inputs(obj: Any) -> list[torch.Tensor]:
    """Collect tensor leaves from a nested input object.

    Parameters
    ----------
    obj:
        Arbitrary nested input object.

    Returns
    -------
    list[torch.Tensor]
        Tensor leaves in traversal order.
    """

    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=repr):
            tensors.extend(_iter_tensor_inputs(obj[key]))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_iter_tensor_inputs(item))
    return tensors


def _input_id_for_relationship_evidence(input_args: Any) -> int:
    """Return the input identity used for relationship evidence.

    Parameters
    ----------
    input_args:
        User-provided positional input container.

    Returns
    -------
    int
        ``id`` of the sole input tensor when available, otherwise ``id`` of
        the input container.
    """

    tensors = _iter_tensor_inputs(input_args)
    if len(tensors) == 1:
        return id(tensors[0])
    return id(input_args)


def _hash_input_signatures(input_args: Any, input_kwargs: Any) -> str:
    """Fingerprint input tensor shape metadata for relationship evidence.

    Parameters
    ----------
    input_args:
        Positional input container.
    input_kwargs:
        Keyword input container.

    Returns
    -------
    str
        SHA-256 hex digest over tensor shapes, dtypes, and devices.
    """

    tensors = _iter_tensor_inputs((input_args, input_kwargs))
    entries = [(tuple(tensor.shape), str(tensor.dtype), str(tensor.device)) for tensor in tensors]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


def _hash_tensor_content(tensor: torch.Tensor) -> str:
    """Return a content hash for a tensor.

    Parameters
    ----------
    tensor:
        Tensor to hash.

    Returns
    -------
    str
        SHA-256 digest over tensor metadata and CPU bytes.
    """

    with _state.pause_logging():
        cpu = tensor.detach().cpu().contiguous()
        if cpu.dtype is torch.bfloat16:
            cpu = cpu.to(torch.float32)
        payload = cpu.numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(repr((tuple(cpu.shape), str(cpu.dtype))).encode("utf-8"))
    hasher.update(payload)
    return hasher.hexdigest()


def _hash_nested_tensor_content(value: Any) -> str:
    """Return a deterministic content hash for nested tensor inputs.

    Parameters
    ----------
    value:
        Nested tensor container.

    Returns
    -------
    str
        SHA-256 digest.
    """

    tensors = _iter_tensor_inputs(value)
    entries = [_hash_tensor_content(tensor) for tensor in tensors]
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()


def _fingerprint_model_content(model: nn.Module) -> str:
    """Fingerprint model tensor contents for the capture cache.

    Parameters
    ----------
    model:
        Model to fingerprint.

    Returns
    -------
    str
        SHA-256 digest.
    """

    hasher = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode("utf-8"))
        hasher.update(_hash_tensor_content(tensor).encode("utf-8"))
    return hasher.hexdigest()


def _capture_cache_dir(cache_dir: str | Path | None) -> Path:
    """Resolve the capture-cache directory.

    Parameters
    ----------
    cache_dir:
        Optional user-specified directory.

    Returns
    -------
    pathlib.Path
        Cache directory path.
    """

    if cache_dir is not None:
        return Path(cache_dir)
    return Path(os.environ.get("TORCHLENS_CACHE_DIR", "~/.cache/torchlens")).expanduser()


def _capture_cache_key(
    model: nn.Module,
    input_args: Any,
    input_kwargs: Any,
    config: dict[str, Any],
) -> str:
    """Build a content-hash capture-cache key.

    Parameters
    ----------
    model:
        Model being captured.
    input_args:
        Positional inputs.
    input_kwargs:
        Keyword inputs.
    config:
        Capture configuration values.

    Returns
    -------
    str
        SHA-256 cache key.
    """

    payload = {
        "schema": 1,
        "torch": torch.__version__,
        "model": _fingerprint_model_content(model),
        "inputs": _hash_nested_tensor_content((input_args, input_kwargs)),
        "config": config,
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _facet_recipe_cache_key(
    recipes: list[Callable[[Any], dict[str, Any]]]
    | tuple[Callable[[Any], dict[str, Any]], ...]
    | None,
) -> tuple[str, ...]:
    """Return stable-ish recipe identities for capture cache separation.

    Parameters
    ----------
    recipes:
        Per-trace recipe functions supplied to ``trace``.

    Returns
    -------
    tuple[str, ...]
        Function module/qualname identities for cache configuration.
    """

    if recipes is None:
        return ()
    return tuple(f"{recipe.__module__}.{recipe.__qualname__}" for recipe in recipes)


def _capture_output_metadata_from_model_config(trace: Trace, model: nn.Module) -> None:
    """Capture portable output metadata from ``model.config`` into ``trace``.

    Parameters
    ----------
    trace:
        Trace receiving the in-band output metadata.
    model:
        Model being captured.
    """

    try:
        config = getattr(model, "config", None)
    except Exception:
        # ``config`` may be a property whose getter raises for reasons unrelated to
        # attribute existence (e.g. delegating to a submodule that only partially
        # implements it). This is best-effort output metadata capture, not a
        # validation check, so a raising getter degrades to "no config metadata"
        # rather than aborting the whole capture.
        return
    if config is None:
        return

    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, dict):
        normalized_id2label: dict[int, str] = {}
        for key, value in id2label.items():
            try:
                normalized_key = int(key)
            except (TypeError, ValueError):
                continue
            normalized_id2label[normalized_key] = str(value)
        trace.output_id2label = normalized_id2label or None

    num_labels = getattr(config, "num_labels", None)
    if num_labels is None and trace.output_id2label is not None:
        num_labels = len(trace.output_id2label)
    if num_labels is not None:
        try:
            trace.output_num_classes = int(num_labels)
        except (TypeError, ValueError):
            trace.output_num_classes = None


def _prepare_log_for_capture_cache(trace: Trace) -> None:
    """Detach non-leaf tensors and autograd objects before cache serialization.

    Parameters
    ----------
    trace:
        Log to make pickle-compatible in place.
    """

    for layer in getattr(trace, "layer_list", []):
        for field_name in (
            "out",
            "transformed_out",
            "grad",
            "transformed_grad",
        ):
            value = _raw_cache_payload_field(layer, field_name)
            if isinstance(value, torch.Tensor):
                layer._internal_set(field_name, value.detach().cpu())
        layer.grad_fn_handle = None
        layer._internal_set("saved_args", _detach_nested_for_cache(layer.saved_args))
        layer._internal_set("saved_kwargs", _detach_nested_for_cache(layer.saved_kwargs))
    for layer_log in getattr(trace, "layer_logs", {}).values():
        for field_name in ("transformed_out", "transformed_grad"):
            value = getattr(layer_log, field_name, None)
            if isinstance(value, torch.Tensor):
                setattr(layer_log, field_name, value.detach().cpu())
        layer_log.grad_fn_handle = None
    trace.__dict__.pop("_container_ordinals_by_output_op_label", None)
    trace.__dict__.pop("_container_ordinals_by_input_func_call_id", None)
    if trace.__dict__.get("_predicate_save_options") is not None:
        trace.__dict__["_predicate_save_options"] = "cache_predicate_capture"
    trace.__dict__.pop("_capture_config", None)
    trace.__dict__.pop("_stop_directive", None)
    build_state = trace.__dict__.get("_build_state")
    if build_state is not None:
        registry = getattr(build_state, "container_registry", None)
        if registry is not None:
            registry.clear_live_state()
        trace.__dict__.pop("_build_state", None)


def _raw_cache_payload_field(layer: Any, field_name: str) -> Any:
    """Return a cache payload field without invoking strict public accessors.

    Parameters
    ----------
    layer:
        Op-like object being prepared for pickle serialization.
    field_name:
        Payload field to read.

    Returns
    -------
    Any
        Raw slot value when present, otherwise ``None``.
    """

    try:
        return object.__getattribute__(layer, field_name)
    except AttributeError:
        return None


def _detach_nested_for_cache(value: Any) -> Any:
    """Detach tensors inside a nested cache payload.

    Parameters
    ----------
    value:
        Nested value.

    Returns
    -------
    Any
        Value with tensors detached to CPU.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_nested_for_cache(item) for item in value)
    if isinstance(value, list):
        return [_detach_nested_for_cache(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_nested_for_cache(item) for key, item in value.items()}
    return value


if TYPE_CHECKING:
    pass


def _unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """Return the underlying ``nn.Module`` if ``model`` is a data-parallel wrapper.

    Handles:
      * ``nn.DataParallel``              -> unwrap via ``.module``
      * ``nn.parallel.DistributedDataParallel`` -> unwrap via ``.module``
      * ``torch.distributed.fsdp.FullyShardedDataParallel`` -> raise

    FSDP cannot be unwrapped the same way: its parameters are sharded across
    ranks, so there is no single unsharded module to log. Users who want to
    log an FSDP-wrapped model should ``trace`` a rank-local
    *un-wrapped* copy of the underlying module instead.

    The function is kept under its original name to avoid churn at call sites;
    the historical ``_unwrap_data_parallel`` now covers the full data-parallel
    family.
    """
    # FSDP: fail loudly rather than silently mis-attributing sharded params.
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        pass
    else:
        if isinstance(model, FullyShardedDataParallel):
            raise RuntimeError(
                "torchlens.trace does not support "
                "FullyShardedDataParallel (FSDP): parameters are sharded "
                "across ranks and there is no unsharded module to log. "
                "Run trace on a rank-local copy of the underlying "
                "module (before FSDP wrapping) instead."
            )

    # DistributedDataParallel: unwrap via ``.module`` (same layout as DataParallel).
    try:
        from torch.nn.parallel import DistributedDataParallel
    except ImportError:
        pass
    else:
        if isinstance(model, DistributedDataParallel):
            return cast(nn.Module, model.module)

    # DataParallel: the original case this helper covered.
    if isinstance(model, nn.DataParallel):
        return cast(nn.Module, model.module)

    return model


def _reject_opaque_wrappers(model: nn.Module) -> None:
    """Raise a clear error if ``model`` is one of the opaque wrappers TorchLens cannot trace.

    TorchLens logs a model by wrapping every torch callable and running an
    ordinary Python forward pass.  The following wrappers all replace that
    Python execution with a traced / scripted / exported graph — by design,
    our wrappers don't see the original ops, so the Trace would be
    empty or misleading:

    * ``torch.jit.ScriptModule`` / ``torch.jit.RecursiveScriptModule``
      (``torch.jit.script`` / ``torch.jit.trace``) — the forward runs on the
      TorchScript interpreter, not Python, so no Python-level decoration fires.
    * ``torch.export.ExportedProgram`` — a serialised IR, not a callable
      ``nn.Module`` that can be re-executed in Python.
    * ``torch.distributed.fsdp.FullyShardedDataParallel`` — FSDP controls
      parameter materialization and sharding around forward execution in ways
      TorchLens cannot currently validate.

    In these cases the fix is the same: call ``trace`` on the
    *un-wrapped* model before scripting, exporting, or sharding.
    """
    # FullyShardedDataParallel
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except (ImportError, RuntimeError):
        pass
    else:
        if isinstance(model, FullyShardedDataParallel):
            raise RuntimeError(
                "torchlens.trace does not support "
                "FullyShardedDataParallel models: FSDP controls parameter "
                "materialization and sharding around forward execution in ways "
                "TorchLens cannot validate. Call trace on the "
                "underlying unwrapped nn.Module."
            )

    # torch.jit.script / torch.jit.trace -> ScriptModule
    if isinstance(model, torch.jit.ScriptModule):
        raise RuntimeError(
            "torchlens.trace does not support torch.jit ScriptModule "
            "or traced models: the forward runs on the TorchScript interpreter "
            "rather than Python, so TorchLens' function wrappers don't fire. "
            "Call trace on the original (un-scripted / un-traced) "
            "model."
        )

    # torch.export.ExportedProgram
    try:
        from torch.export import ExportedProgram
    except ImportError:
        pass
    else:
        if isinstance(model, ExportedProgram):
            raise RuntimeError(
                "torchlens.trace does not support "
                "torch.export.ExportedProgram: the exported IR is not a "
                "callable nn.Module that can be re-executed in Python. "
                "Call trace on the original nn.Module before "
                "export."
            )


def _move_tensors_to_device(obj: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a nested structure (lists, tuples, dicts) to *device*.

    Handles common dict-like types (OrderedDict, HuggingFace BatchEncoding, etc.)
    by attempting to reconstruct the original container type after moving values.
    NamedTuple subclasses (e.g. a GNN model's batch container, which is also an
    ``isinstance(obj, tuple)`` match) are reconstructed through their own type so
    downstream named-field access keeps working instead of silently degrading to
    a plain ``tuple``.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        moved_sequence = [_move_tensors_to_device(item, device) for item in obj]
        if not isinstance(obj, tuple):
            return type(obj)(moved_sequence)
        obj_type = type(obj)
        if hasattr(obj_type, "_fields"):
            return obj_type(*moved_sequence)
        return tuple(moved_sequence)
    elif isinstance(obj, collections.abc.MutableMapping):
        # Handles dict, UserDict, BatchEncoding, OrderedDict, etc.
        moved_mapping = {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
        if type(obj) is dict:
            return moved_mapping
        try:
            return cast(Any, type(obj))(moved_mapping)
        except Exception:
            return moved_mapping
    return obj
