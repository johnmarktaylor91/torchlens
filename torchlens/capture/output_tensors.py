"""Functions for logging output tensors produced by decorated torch operations.

This module handles the creation and population of LayerPassLog entries for every
tensor produced during a forward pass.  It covers both *exhaustive* mode (full
metadata collection) and *fast* mode (re-use of a previously logged graph with
new activations).

Architecture overview:
    Every decorated torch function wrapper calls ``log_function_output_tensors``
    after executing the original function.  This dispatcher routes to either:

    - ``log_function_output_tensors_exhaustive``: builds a complete
      ``fields_dict`` of ~80 fields per tensor, creates a LayerPassLog entry,
      updates family links (parent/child/sibling/spouse), and optionally
      saves the activation value.

    - ``log_function_output_tensors_fast``: skips metadata collection entirely.
      Increments counters to maintain alignment with the exhaustive pass,
      verifies the graph hasn't changed, and saves new activation values into
      the existing LayerPassLog entries.

Label format convention:
    Raw labels follow ``{layer_type}_{type_num}_{realtime_num}_raw``, e.g.
    ``"conv2d_3_47_raw"``.  During postprocessing, these are mapped to final
    labels like ``"conv2d_3:1"`` (layer 3, pass 1).

pause_logging usage:
    ``pause_logging()`` temporarily disables the logging toggle so that
    utility operations (e.g., ``get_tensor_memory_amount``, ``safe_copy``,
    ``activation_postfunc``) don't get logged as model operations.  It is
    used inside ``save_tensor_data`` and wherever helper functions call
    decorated torch methods on tensors.
"""

import copy
import dataclasses
import time
import warnings
from collections import defaultdict
from math import prod
from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Tuple, Union

import torch

from .. import _state as _st
from .._state import pause_logging
from ..utils.introspection import (
    _get_func_call_stack,
    get_attr_values_from_tensor_list,
    get_vars_of_type_from_obj,
)
from ..utils.tensor_utils import get_tensor_memory_amount, safe_copy, tensor_nanequal
from ..utils.collections import index_nested, ensure_iterable
from .flops import compute_backward_flops, compute_forward_flops
from ..data_classes.buffer_log import BufferLog
from ..data_classes.layer_pass_log import LayerPassLog
from ..intervention.types import (
    ArgComponent,
    CapturedArgTemplate,
    ContainerSpec,
    DataclassField,
    DictKey,
    EdgeUseRecord,
    FunctionRegistryKey,
    HFKey,
    LiteralTensor,
    LiteralValue,
    NamedField,
    OutputPathComponent,
    ParentRef,
    TupleIndex,
    Unsupported,
)
from ..intervention.hooks import make_live_site_proxy
from .arg_positions import (
    FUNC_ARG_SPECS,
    ArgSpec,
    extract_tensors_and_params,
    _cache_dynamic_spec,
    _normalize_func_name,
)

from .tensor_tracking import (
    _update_tensor_family_links,
    _locate_parent_tensors_in_args,
    _get_ancestors_from_parents,
    _add_backward_hook,
    _process_parent_param_passes,
    _make_raw_param_group_barcode,
    _get_operation_equivalence_type,
    _get_hash_from_args,
    _update_tensor_containing_modules,
)
from .._errors import TorchLensPostfuncError
from .._training_validation import TrainingModeConfigError
from ..data_classes.internal_types import FuncExecutionContext
from ..fastlog._predicate import _evaluate_keep_op
from ..fastlog._record_context import _build_record_context
from ..fastlog._state import get_active_recording_state
from ..fastlog.types import ActivationRecord, CaptureSpec
from .salient_args import extract_salient_args
from .source_tensors import _get_input_module_info

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog

_AUTOGRAD_SAVED_ATTR_PREFIX = "_saved_"
_UNSUPPORTED_OUTPUT_CONTAINER_WARNED: set[str] = set()


def _tensor_shape_or_none(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
    """Return a tensor shape tuple or ``None``.

    Parameters
    ----------
    tensor
        Tensor-like value to inspect.

    Returns
    -------
    tuple[int, ...] | None
        Shape tuple when a tensor is present.
    """

    return tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None


def _tensor_dtype_or_none(tensor: torch.Tensor | None) -> torch.dtype | None:
    """Return a tensor dtype or ``None``.

    Parameters
    ----------
    tensor
        Tensor-like value to inspect.

    Returns
    -------
    torch.dtype | None
        Dtype when a tensor is present.
    """

    return tensor.dtype if isinstance(tensor, torch.Tensor) else None


def _tensor_memory_or_none(tensor: torch.Tensor | None) -> int | None:
    """Return tensor memory bytes or ``None``.

    Parameters
    ----------
    tensor
        Tensor-like value to inspect.

    Returns
    -------
    int | None
        Memory amount when a tensor is present.
    """

    return get_tensor_memory_amount(tensor) if isinstance(tensor, torch.Tensor) else None


def _is_namedtuple_instance(value: Any) -> bool:
    """Return whether ``value`` is a namedtuple instance.

    Parameters
    ----------
    value
        Object to inspect.

    Returns
    -------
    bool
        True when the object behaves like a namedtuple instance.
    """

    return isinstance(value, tuple) and hasattr(value, "_fields")


def _is_hf_model_output(value: Any) -> bool:
    """Return whether ``value`` looks like a HuggingFace ``ModelOutput``.

    Parameters
    ----------
    value
        Object to inspect.

    Returns
    -------
    bool
        True when the object is a ``transformers.utils.ModelOutput`` instance
        or a duck-typed equivalent with ``keys`` and ``__getitem__``.
    """

    cls = type(value)
    if any(
        base.__module__.startswith("transformers") and base.__name__ == "ModelOutput"
        for base in cls.__mro__
    ):
        return True
    return (
        (cls.__module__.startswith("transformers") or cls.__name__.endswith("ModelOutput"))
        and hasattr(value, "keys")
        and hasattr(value, "__getitem__")
    )


def _container_type_ref(value: Any) -> tuple[str | None, str | None]:
    """Return the import-ish type reference for a container value.

    Parameters
    ----------
    value
        Container value.

    Returns
    -------
    tuple[str | None, str | None]
        ``(module, qualname)`` for the value's class.
    """

    cls = type(value)
    return cls.__module__, cls.__qualname__


def _build_container_spec(value: Any) -> ContainerSpec | None:
    """Build a replay container spec for a supported output container.

    Parameters
    ----------
    value
        Output object to describe.

    Returns
    -------
    ContainerSpec | None
        Container spec, or ``None`` for a single tensor / unsupported scalar.
    """

    child_specs: list[tuple[OutputPathComponent, ContainerSpec]] = []
    if _is_hf_model_output(value):
        keys = tuple(value.keys())
        for key in keys:
            child_spec = _build_container_spec(value[key])
            if child_spec is not None:
                child_specs.append((HFKey(key), child_spec))
        module, qualname = _container_type_ref(value)
        return ContainerSpec(
            kind="hf_model_output",
            length=len(keys),
            keys=keys,
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
        )
    if _is_namedtuple_instance(value):
        fields = tuple(value._fields)
        for field_name in fields:
            child_spec = _build_container_spec(getattr(value, field_name))
            if child_spec is not None:
                child_specs.append((NamedField(field_name), child_spec))
        module, qualname = _container_type_ref(value)
        return ContainerSpec(
            kind="namedtuple",
            length=len(value),
            fields=fields,
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = tuple(field.name for field in dataclasses.fields(value))
        for field_name in fields:
            child_spec = _build_container_spec(getattr(value, field_name))
            if child_spec is not None:
                child_specs.append((DataclassField(field_name), child_spec))
        module, qualname = _container_type_ref(value)
        return ContainerSpec(
            kind="dataclass",
            length=len(fields),
            fields=fields,
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
        )
    if isinstance(value, dict):
        keys = tuple(value.keys())
        for key in keys:
            child_spec = _build_container_spec(value[key])
            if child_spec is not None:
                child_specs.append((DictKey(key), child_spec))
        return ContainerSpec(
            kind="dict",
            length=len(keys),
            keys=keys,
            child_specs=tuple(child_specs),
        )
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            child_spec = _build_container_spec(item)
            if child_spec is not None:
                child_specs.append((TupleIndex(index), child_spec))
        return ContainerSpec(kind="tuple", length=len(value), child_specs=tuple(child_specs))
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_spec = _build_container_spec(item)
            if child_spec is not None:
                child_specs.append((TupleIndex(index), child_spec))
        return ContainerSpec(kind="list", length=len(value), child_specs=tuple(child_specs))
    return None


def _walk_supported_output_container(
    out: Any,
    *,
    root_spec: ContainerSpec,
    path: tuple[OutputPathComponent, ...],
) -> Iterator[tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]]:
    """Yield tensors from a supported output container.

    Parameters
    ----------
    out
        Output object or nested child object to traverse.
    root_spec
        Spec for the outermost output container.
    path
        Path accumulated from the outermost output container.

    Yields
    ------
    tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]
        Tensor, path, and root container spec.
    """

    if isinstance(out, torch.Tensor):
        if not isinstance(out, torch.nn.Parameter):
            yield out, path, root_spec
        return
    if _is_hf_model_output(out):
        for key in out.keys():
            yield from _walk_supported_output_container(
                out[key],
                root_spec=root_spec,
                path=(*path, HFKey(key)),
            )
        return
    if _is_namedtuple_instance(out):
        for field_name in out._fields:
            yield from _walk_supported_output_container(
                getattr(out, field_name),
                root_spec=root_spec,
                path=(*path, NamedField(field_name)),
            )
        return
    if dataclasses.is_dataclass(out) and not isinstance(out, type):
        for field in dataclasses.fields(out):
            yield from _walk_supported_output_container(
                getattr(out, field.name),
                root_spec=root_spec,
                path=(*path, DataclassField(field.name)),
            )
        return
    if isinstance(out, dict):
        for key, value in out.items():
            yield from _walk_supported_output_container(
                value,
                root_spec=root_spec,
                path=(*path, DictKey(key)),
            )
        return
    if isinstance(out, (list, tuple)):
        for index, item in enumerate(out):
            yield from _walk_supported_output_container(
                item,
                root_spec=root_spec,
                path=(*path, TupleIndex(index)),
            )


def _walk_output_tensors_with_paths(
    out: Any,
) -> Iterator[tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]]:
    """Yield each output tensor with its path inside the output container.

    Parameters
    ----------
    out
        Raw output object from a torch operation or model forward call.

    Yields
    ------
    tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]
        Output tensor, path inside the output container, and the outer
        container spec. Single tensor outputs use ``()`` and ``None``.
    """

    if isinstance(out, torch.Tensor):
        if not isinstance(out, torch.nn.Parameter):
            yield out, (), None
        return

    root_spec = _build_container_spec(out)
    if root_spec is None:
        if _literal_value_supported(out) or isinstance(out, torch.Size):
            return
        container_name = type(out).__qualname__
        if container_name not in _UNSUPPORTED_OUTPUT_CONTAINER_WARNED:
            _UNSUPPORTED_OUTPUT_CONTAINER_WARNED.add(container_name)
            warnings.warn(
                f"TorchLens intervention-ready output traversal does not support "
                f"{container_name}; falling back to BFS without stable output paths.",
                UserWarning,
                stacklevel=2,
            )
        for tensor in get_vars_of_type_from_obj(
            out, which_type=torch.Tensor, subclass_exceptions=[torch.nn.Parameter]
        ):
            yield tensor, (), None
        return

    yield from _walk_supported_output_container(out, root_spec=root_spec, path=())


def _function_registry_key(func: Callable) -> FunctionRegistryKey:
    """Build a portable registry key for a captured function.

    Parameters
    ----------
    func
        Function object being logged.

    Returns
    -------
    FunctionRegistryKey
        Best-effort function identity.
    """

    return FunctionRegistryKey(
        module=getattr(func, "__module__", None),
        qualname=getattr(func, "__qualname__", None),
        name=getattr(func, "__name__", None),
        registry_id=None,
    )


def _literal_value_supported(value: Any) -> bool:
    """Return whether ``value`` is a replay-safe literal.

    Parameters
    ----------
    value
        Value to classify.

    Returns
    -------
    bool
        True when the value can be stored directly in an argument template.
    """

    return isinstance(
        value,
        (int, float, bool, str, bytes, type(None), torch.dtype, torch.device, slice),
    )


def _classify_arg_component(value: Any, notes: list[str]) -> ArgComponent:
    """Classify a function argument value for replay templating.

    Parameters
    ----------
    value
        Argument value to classify.
    notes
        Accumulator for unsupported-value notes.

    Returns
    -------
    ArgComponent
        Tagged replay template component.
    """

    label = getattr(value, "tl_tensor_label_raw", None)
    if isinstance(label, str):
        return ParentRef(label)
    if isinstance(value, torch.Tensor):
        with pause_logging():
            return LiteralTensor(safe_copy(value))
    if _literal_value_supported(value):
        return LiteralValue(value)
    if isinstance(value, (list, tuple)):
        return tuple(_classify_arg_component(item, notes) for item in value)
    if isinstance(value, dict):
        return tuple((key, _classify_arg_component(item, notes)) for key, item in value.items())

    reason = f"unsupported argument type {type(value).__module__}.{type(value).__qualname__}"
    notes.append(reason)
    return Unsupported(reason=reason, value_type=type(value).__qualname__)


def _build_captured_arg_template(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CapturedArgTemplate:
    """Build a replay template from original function args and kwargs.

    Parameters
    ----------
    func
        Function object being logged.
    args
        Original positional args.
    kwargs
        Original keyword args.

    Returns
    -------
    CapturedArgTemplate
        Replay template for the function call.
    """

    notes: list[str] = []
    arg_components = tuple(_classify_arg_component(arg, notes) for arg in args)
    kwarg_components = tuple(
        (str(key), _classify_arg_component(value, notes)) for key, value in kwargs.items()
    )
    return CapturedArgTemplate(
        args=arg_components,
        kwargs=kwarg_components,
        func_id=_function_registry_key(func),
        notes=tuple(notes),
    )


def _arg_location_to_path(location: Any) -> tuple[OutputPathComponent, ...]:
    """Convert a parent-layer arg location to the MVP edge path schema.

    Parameters
    ----------
    location
        Location key from ``parent_layer_arg_locs``.

    Returns
    -------
    tuple[OutputPathComponent, ...]
        Path tuple mirroring the existing two-level arg-location scheme.
    """

    if isinstance(location, tuple):
        return location
    return (location,)


def _build_edge_use_records(
    self: "ModelLog",
    parent_layer_arg_locs: dict[str, dict[Any, str]],
    child_label: str,
    child_func_call_id: int,
) -> list[EdgeUseRecord]:
    """Build edge provenance records from existing parent arg locations.

    Parameters
    ----------
    self
        Active model log.
    parent_layer_arg_locs
        Existing parent-location map.
    child_label
        Raw label for the child tensor output.
    child_func_call_id
        Function call id for the child operation.

    Returns
    -------
    list[EdgeUseRecord]
        Edge provenance records.
    """

    edge_uses: list[EdgeUseRecord] = []
    for location, parent_label in parent_layer_arg_locs["args"].items():
        edge_uses.append(
            EdgeUseRecord(
                parent_label=parent_label,
                child_label=child_label,
                arg_kind="positional",
                arg_path=_arg_location_to_path(location),
                view_or_copy="unknown",
                parent_func_call_id=getattr(self[parent_label], "func_call_id", None),
                child_func_call_id=child_func_call_id,
            )
        )
    for location, parent_label in parent_layer_arg_locs["kwargs"].items():
        edge_uses.append(
            EdgeUseRecord(
                parent_label=parent_label,
                child_label=child_label,
                arg_kind="keyword",
                arg_path=_arg_location_to_path(location),
                view_or_copy="unknown",
                parent_func_call_id=getattr(self[parent_label], "func_call_id", None),
                child_func_call_id=child_func_call_id,
            )
        )
    return edge_uses


def log_function_output_tensors(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> None:
    """Dispatch to exhaustive or fast logging based on current logging mode.

    Called by every decorated torch function wrapper after executing the
    original function.  The mode was set in ``save_new_activations`` (fast)
    or ``log_forward_pass`` (exhaustive).
    """
    if self.logging_mode == "exhaustive":
        log_function_output_tensors_exhaustive(
            self,
            func,
            func_name,
            args,
            kwargs,
            arg_copies,
            kwarg_copies,
            out_orig,
            exec_ctx,
            is_bottom_level_func,
            func_call_id,
        )
    elif self.logging_mode == "fast":
        log_function_output_tensors_fast(
            self,
            func_name,
            args,
            kwargs,
            arg_copies,
            kwarg_copies,
            out_orig,
            exec_ctx,
            is_bottom_level_func,
            func_call_id,
        )
    elif self.logging_mode == "predicate":
        log_function_output_tensors_predicate(
            self,
            func_name,
            args,
            out_orig,
            is_bottom_level_func,
        )


def apply_live_hooks_to_outputs(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> Any:
    """Apply live hooks to function outputs before output logging.

    Parameters
    ----------
    self
        Active model log.
    func
        Original decorated function.
    func_name
        Torch function name.
    args
        Function positional arguments.
    kwargs
        Function keyword arguments.
    out_orig
        Function output after in-place safe-copy handling.
    exec_ctx
        Function execution metadata.
    is_bottom_level_func
        Whether the wrapper call is a bottom-level operation.
    func_call_id
        Function-call id allocated before calling ``func``.

    Returns
    -------
    Any
        Output object with hooked tensors replaced in place where possible.
    """

    if not _st._active_hook_plan or self.logging_mode != "exhaustive":
        return out_orig

    from ..intervention.runtime import _apply_live_hooks

    shared_fields, _parent_layer_entries, _arg_tensors, _parent_param_passes = (
        _build_shared_fields_dict(
            self, func, func_name, args, kwargs, out_orig, exec_ctx, func_call_id
        )
    )
    layer_type = shared_fields["layer_type"]
    replacements: dict[tuple[OutputPathComponent, ...], torch.Tensor] = {}
    predicted_layer_counter = self._layer_counter
    predicted_type_counter = self._raw_layer_type_counter[layer_type]

    for out, output_path, _container_spec in _iter_loggable_live_outputs(
        out_orig, is_bottom_level_func
    ):
        predicted_layer_counter += 1
        predicted_type_counter += 1
        raw_label = f"{layer_type}_{predicted_type_counter}_{predicted_layer_counter}_raw"
        site = make_live_site_proxy(
            layer_label_raw=raw_label,
            func_name=func_name,
            layer_type=layer_type,
            tensor=out,
            func_call_id=func_call_id,
            output_path=output_path,
            fields=shared_fields,
        )
        hooked = _apply_live_hooks(out, site=site, output_path=output_path)
        if hooked is not out:
            replacements[output_path] = hooked

    if not replacements:
        return out_orig
    if isinstance(out_orig, torch.Tensor):
        return replacements.get((), out_orig)
    return _replace_output_tensors_by_path(out_orig, replacements)


def _iter_loggable_live_outputs(
    out_orig: Any,
    is_bottom_level_func: bool,
) -> Iterator[tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]]:
    """Yield outputs that will be logged in exhaustive mode.

    Parameters
    ----------
    out_orig
        Function output.
    is_bottom_level_func
        Whether the wrapper call is a bottom-level operation.

    Yields
    ------
    tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]
        Tensor, output path, and container spec.
    """

    for out, output_path, container_spec in _walk_output_tensors_with_paths(out_orig):
        if _output_should_be_logged(out, is_bottom_level_func):
            yield out, output_path, container_spec


def _replace_output_tensors_by_path(
    out_orig: Any,
    replacements: dict[tuple[OutputPathComponent, ...], torch.Tensor],
) -> Any:
    """Return an output object with selected tensor paths replaced.

    Parameters
    ----------
    out_orig
        Original output container.
    replacements
        Replacement tensors keyed by output path.

    Returns
    -------
    Any
        Rebuilt output object when supported, otherwise the original object.
    """

    if () in replacements:
        return replacements[()]
    return _replace_output_value(out_orig, (), replacements)


def _replace_output_value(
    value: Any,
    path: tuple[OutputPathComponent, ...],
    replacements: dict[tuple[OutputPathComponent, ...], torch.Tensor],
) -> Any:
    """Recursively replace tensors in supported output containers.

    Parameters
    ----------
    value
        Current container value.
    path
        Path to the current value.
    replacements
        Replacement tensors keyed by full output path.

    Returns
    -------
    Any
        Rebuilt value.
    """

    if path in replacements:
        return replacements[path]
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        replaced_items = [
            _replace_output_value(item, (*path, NamedField(field)), replacements)
            for field, item in zip(value._fields, value)
        ]
        return type(value)(*replaced_items)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {
            field.name: _replace_output_value(
                getattr(value, field.name), (*path, DataclassField(field.name)), replacements
            )
            for field in dataclasses.fields(value)
        }
        return dataclasses.replace(value, **changes)
    if isinstance(value, tuple):
        return type(value)(
            _replace_output_value(item, (*path, TupleIndex(index)), replacements)
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _replace_output_value(item, (*path, TupleIndex(index)), replacements)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _replace_output_value(item, (*path, DictKey(key)), replacements)
            for key, item in value.items()
        }
    return value


def _apply_live_fire_records_to_entry(entry: LayerPassLog) -> None:
    """Attach pending live fire records and refresh saved tensor metadata.

    Parameters
    ----------
    entry
        Newly created layer-pass entry.

    Returns
    -------
    None
        The entry is updated in place through internal setters.
    """

    records = _st._pending_live_fire_records.pop(entry.tensor_label_raw, [])
    if not records:
        return
    entry.intervention_log.extend(records)
    tensor = entry.activation if isinstance(entry.activation, torch.Tensor) else None
    if tensor is not None:
        _set_saved_activation_metadata(entry, tensor)


def _set_saved_activation_metadata(entry: LayerPassLog, tensor: torch.Tensor) -> None:
    """Refresh activation metadata through internal setters.

    Parameters
    ----------
    entry
        Layer pass whose saved activation metadata should match ``tensor``.
    tensor
        Saved activation tensor.

    Returns
    -------
    None
        Metadata fields are updated atomically through ``_internal_set``.
    """

    entry._internal_set("tensor_shape", tuple(tensor.shape))
    entry._internal_set("tensor_dtype", tensor.dtype)
    entry._internal_set("tensor_memory", get_tensor_memory_amount(tensor))
    entry._internal_set("has_saved_activations", True)
    entry._internal_set(
        "transformed_activation_shape",
        _tensor_shape_or_none(entry.transformed_activation),
    )
    entry._internal_set(
        "transformed_activation_dtype",
        _tensor_dtype_or_none(entry.transformed_activation),
    )
    entry._internal_set(
        "transformed_activation_memory",
        _tensor_memory_or_none(entry.transformed_activation),
    )


def _record_predicate_output(
    ctx,
    out: torch.Tensor,
    spec: CaptureSpec,
) -> None:
    """Store a predicate-selected operation output."""

    if not spec.save_activation and not spec.save_metadata:
        return
    state = get_active_recording_state()
    ram_payload = None
    disk_payload = None
    transformed_ram_payload = None
    transformed_disk_payload = None
    if spec.save_activation:
        (
            ram_payload,
            disk_payload,
            transformed_ram_payload,
            transformed_disk_payload,
        ) = state.resolve_storage(out, spec, ctx=ctx)
    state.add_record(
        ActivationRecord(
            ctx=ctx,
            spec=spec,
            ram_payload=ram_payload,
            disk_payload=disk_payload,
            transformed_ram_payload=transformed_ram_payload,
            transformed_disk_payload=transformed_disk_payload,
        )
    )


def log_function_output_tensors_predicate(
    self,
    func_name: str,
    args: Tuple[Any],
    out_orig: Any,
    is_bottom_level_func: bool,
) -> None:
    """Predicate-mode logging for decorated torch function outputs."""

    state = get_active_recording_state()
    layer_type = func_name.lower().replace("_", "")
    arg_tensors, _ = _extract_arg_tensors_and_params(layer_type, args, {})
    parent_labels = tuple(get_attr_values_from_tensor_list(arg_tensors, "tl_tensor_label_raw"))
    out_iter = ensure_iterable(out_orig)

    for output_index, out in enumerate(out_iter):
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue
        self._layer_counter += 1
        self._raw_layer_type_counter[layer_type] += 1
        state.op_counts[layer_type] = state.op_counts.get(layer_type, 0) + 1
        state.op_index += 1
        state.event_index += 1
        creation_order = self._layer_counter
        layer_type_num = self._raw_layer_type_counter[layer_type]
        tensor_label_raw = f"{layer_type}_{layer_type_num}_{creation_order}_raw"
        out.tl_tensor_label_raw = tensor_label_raw
        module_frame = state.module_stack[-1] if state.module_stack else None
        ctx = _build_record_context(
            kind="op",
            layer_pass_log_or_op_data={
                "label": tensor_label_raw,
                "raw_label": tensor_label_raw,
                "tensor_label_raw": tensor_label_raw,
                "creation_order": creation_order,
                "layer_type": layer_type,
                "layer_type_num": layer_type_num,
                "func_name": func_name,
                "parent_labels": parent_labels,
                "tensor": out,
                "output_index": output_index,
                "is_bottom_level_func": is_bottom_level_func,
                "module_address": module_frame.module_address if module_frame else None,
                "module_type": module_frame.module_type if module_frame else None,
                "module_pass_index": module_frame.pass_index if module_frame else None,
            },
            module_stack=state.module_stack,
            history=tuple(state.history),
            op_counts=state.op_counts,
            pass_index=state.pass_index,
            event_index=state.event_index,
            op_index=state.op_index,
            time_since_pass_start=time.time() - self.pass_start_time,
            include_source_events=state.options.include_source_events,
            sample_id=state.sample_id,
        )
        try:
            spec = _evaluate_keep_op(ctx, state.options)
            _record_predicate_output(ctx, out, spec)
        except (TorchLensPostfuncError, TrainingModeConfigError):
            # Postfunc + train-mode validation errors are storage failures and
            # must surface directly to the caller, not be aggregated through
            # the predicate-failure pipeline. The orchestrator's outer
            # exception handler aborts disk storage before the raise reaches
            # the caller.
            raise
        except Exception as exc:
            state.handle_predicate_exception(ctx, exc)
        finally:
            state.append_context(ctx)


def _build_graph_relationship_fields(
    self,
    fields_dict: Dict[str, Any],
    parent_layer_labels: List[str],
    parent_layer_entries: List,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    out_orig: Any,
) -> None:
    """Populate graph structure fields: parents, children, ancestors, buffer/IO flags."""
    parent_layer_arg_locs = _locate_parent_tensors_in_args(self, parent_layer_entries, args, kwargs)
    input_ancestors, internally_initialized_ancestors = _get_ancestors_from_parents(
        parent_layer_entries
    )
    internal_parent_layer_labels = [
        label for label in parent_layer_labels if self[label].has_internally_initialized_ancestor
    ]

    fields_dict["parent_layers"] = parent_layer_labels
    fields_dict["parent_layer_arg_locs"] = parent_layer_arg_locs
    fields_dict["edge_uses"] = []
    fields_dict["root_ancestors"] = input_ancestors.union(internally_initialized_ancestors)
    fields_dict["child_layers"] = []
    fields_dict["has_children"] = False
    fields_dict["is_input_layer"] = False
    fields_dict["has_input_ancestor"] = len(input_ancestors) > 0
    fields_dict["input_ancestors"] = input_ancestors
    fields_dict["min_distance_from_input"] = None
    fields_dict["max_distance_from_input"] = None
    fields_dict["is_output_layer"] = False
    fields_dict["feeds_output"] = False
    fields_dict["is_final_output"] = False
    fields_dict["is_output_ancestor"] = False
    fields_dict["output_descendants"] = set()
    fields_dict["min_distance_from_output"] = None
    fields_dict["max_distance_from_output"] = None
    fields_dict["io_role"] = None
    fields_dict["is_buffer_layer"] = False
    fields_dict["buffer_address"] = None
    fields_dict["buffer_pass"] = None
    fields_dict["buffer_parent"] = None
    fields_dict["is_internally_initialized"] = len(parent_layer_labels) == 0
    fields_dict["has_internally_initialized_ancestor"] = len(internally_initialized_ancestors) > 0
    fields_dict["internally_initialized_parents"] = internal_parent_layer_labels
    fields_dict["internally_initialized_ancestors"] = internally_initialized_ancestors
    fields_dict["is_internally_terminated"] = False
    fields_dict["is_terminal_bool_layer"] = False
    fields_dict["bool_is_branch"] = False
    fields_dict["bool_context_kind"] = None
    fields_dict["bool_wrapper_kind"] = None
    fields_dict["bool_conditional_id"] = None
    fields_dict["in_cond_branch"] = False
    fields_dict["conditional_branch_stack"] = []
    fields_dict["conditional_branch_depth"] = 0
    fields_dict["cond_branch_start_children"] = []
    fields_dict["cond_branch_then_children"] = []
    fields_dict["cond_branch_elif_children"] = {}
    fields_dict["cond_branch_else_children"] = []
    fields_dict["cond_branch_children_by_cond"] = {}

    is_part_of_iterable_output = any(
        issubclass(type(out_orig), cls) for cls in [list, tuple, dict, set]
    )
    fields_dict["is_part_of_iterable_output"] = is_part_of_iterable_output


def _extract_arg_tensors_and_params(
    normalized_name: str,
    args: tuple,
    kwargs: dict,
) -> Tuple[List, List]:
    """O(1) tensor/param extraction via lookup table, with BFS fallback."""
    spec = FUNC_ARG_SPECS.get(normalized_name) or _st._dynamic_arg_specs.get(normalized_name)
    if spec is not None:
        return extract_tensors_and_params(spec, args, kwargs)  # type: ignore[arg-type]

    # Tier 3 fallback: BFS crawl once, then cache for subsequent calls
    all_args = list(args) + list(kwargs.values())
    arg_tensors = get_vars_of_type_from_obj(all_args, torch.Tensor, [torch.nn.Parameter])
    arg_parameters = get_vars_of_type_from_obj(all_args, torch.nn.parameter.Parameter)
    _cache_dynamic_spec(normalized_name, args, kwargs, arg_tensors, arg_parameters)
    return arg_tensors, arg_parameters


def _build_param_fields(
    self,
    fields_dict: Dict[str, Any],
    arg_parameters: List,
) -> Dict:
    """Populate parameter-involvement fields. Returns parent_param_passes dict."""
    parent_param_passes = _process_parent_param_passes(arg_parameters)
    indiv_param_barcodes = list(parent_param_passes.keys())

    parent_param_logs = []
    for param in arg_parameters:
        addr = getattr(param, "tl_param_address", None)
        if addr is not None and addr in self.param_logs:
            parent_param_logs.append(self.param_logs[addr])

    fields_dict["parent_params"] = arg_parameters
    fields_dict["parent_param_barcodes"] = indiv_param_barcodes
    fields_dict["parent_param_passes"] = parent_param_passes
    fields_dict["parent_param_logs"] = parent_param_logs
    fields_dict["parent_param_shapes"] = [tuple(param.shape) for param in arg_parameters]
    fields_dict["num_params_total"] = sum(
        prod(shape) for shape in fields_dict["parent_param_shapes"]
    )
    fields_dict["num_params_trainable"] = sum(
        pl.num_params for pl in parent_param_logs if pl.trainable
    )
    fields_dict["num_params_frozen"] = sum(
        pl.num_params for pl in parent_param_logs if not pl.trainable
    )
    with pause_logging():
        fields_dict["params_memory"] = sum(p.nelement() * p.element_size() for p in arg_parameters)
    return parent_param_passes


def _build_module_context_fields(
    self,
    fields_dict: Dict[str, Any],
    arg_tensors: List,
    parent_layer_entries: List,
) -> None:
    """Populate module nesting, address, and input/output status fields."""
    containing_modules = _get_input_module_info(self, arg_tensors)
    containing_module = containing_modules[-1] if containing_modules else None

    fields_dict["containing_module"] = containing_module
    fields_dict["containing_modules"] = containing_modules
    fields_dict["modules_entered"] = []
    fields_dict["modules_entered_argnames"] = defaultdict(list)
    fields_dict["module_passes_entered"] = []
    fields_dict["modules_exited"] = []
    fields_dict["module_passes_exited"] = []
    fields_dict["is_submodule_output"] = False
    fields_dict["is_leaf_module_output"] = False
    fields_dict["leaf_module_pass"] = None
    fields_dict["module_entry_exit_threads_inputs"] = {
        p.tensor_label_raw: p.module_entry_exit_thread_output[:] for p in parent_layer_entries
    }
    fields_dict["module_entry_exit_thread_output"] = []


def _build_shared_fields_dict(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    func_call_id: int,
) -> Tuple[Dict[str, Any], List, List, Dict]:
    """Build the fields_dict shared by all output tensors of a single function call.

    When a function produces multiple output tensors (e.g. ``torch.split``),
    many metadata fields are identical across outputs (function info, parent
    relationships, module context).  This function computes those shared fields
    once; per-tensor fields (shape, label, equivalence type) are added later
    in ``_log_output_tensor_info``.

    Returns:
        (fields_dict, parent_layer_entries, arg_tensors, parent_param_passes)
    """
    # Canonical layer_type: lowercase with underscores stripped (e.g. "conv2d").
    layer_type = func_name.lower().replace("_", "")

    # O(1) tensor/param extraction via lookup table (replaces BFS crawl)
    arg_tensors, arg_parameters = _extract_arg_tensors_and_params(layer_type, args, kwargs)

    # Separate tensor args (which define graph edges) from non-tensor args
    # (which become metadata and feed into operation_equivalence_type hashing).
    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {key: val for key, val in kwargs.items() if not _check_if_tensor_arg(val)}
    parent_layer_labels = get_attr_values_from_tensor_list(arg_tensors, "tl_tensor_label_raw")
    parent_layer_entries = [self[label] for label in parent_layer_labels]

    fields_dict: Dict[str, Any] = {}

    # General info
    fields_dict["layer_type"] = layer_type
    fields_dict["detach_saved_tensor"] = self.detach_saved_tensors
    fields_dict["output_device"] = self.output_device
    fields_dict["_construction_done"] = False
    fields_dict["intervention_log"] = []
    should_capture_template = bool(
        getattr(self, "intervention_ready", False) or getattr(self, "capture_full_args", False)
    )
    if should_capture_template:
        captured_template = _build_captured_arg_template(func, args, kwargs)
        fields_dict["captured_arg_template"] = captured_template
        fields_dict["captured_kwarg_template"] = captured_template if kwargs else None
    else:
        fields_dict["captured_arg_template"] = None
        fields_dict["captured_kwarg_template"] = None
    fields_dict["output_path"] = ()
    fields_dict["container_spec"] = None

    # Grad info
    fields_dict["gradient"] = None
    fields_dict["transformed_gradient"] = None
    fields_dict["save_gradients"] = self.save_gradients
    fields_dict["has_gradient"] = False
    fields_dict["grad_shape"] = None
    fields_dict["transformed_gradient_shape"] = None
    fields_dict["grad_dtype"] = None
    fields_dict["transformed_gradient_dtype"] = None
    fields_dict["grad_memory"] = 0
    fields_dict["transformed_gradient_memory"] = None

    # Function call info
    fields_dict["func_applied"] = func
    fields_dict["func_call_id"] = func_call_id
    fields_dict["func_name"] = func_name
    fields_dict["func_call_stack"] = _get_func_call_stack(
        self.num_context_lines,
        source_loading_enabled=self.save_source_context,
        disable_col_offset=False,
    )
    fields_dict["func_time"] = exec_ctx.time_elapsed
    fields_dict["func_rng_states"] = exec_ctx.rng_states
    fields_dict["func_autocast_state"] = exec_ctx.autocast_state
    fields_dict["func_argnames"] = _st._func_argnames.get(func_name.strip("_"), ())
    fields_dict["num_args"] = len(args) + len(kwargs)
    fields_dict["num_positional_args"] = len(args)
    fields_dict["num_keyword_args"] = len(kwargs)
    fields_dict["func_positional_args_non_tensor"] = non_tensor_args
    fields_dict["func_kwargs_non_tensor"] = non_tensor_kwargs
    fields_dict["func_non_tensor_args"] = non_tensor_args + list(non_tensor_kwargs.values())

    _build_graph_relationship_fields(
        self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
    )
    parent_param_passes = _build_param_fields(self, fields_dict, arg_parameters)
    _build_module_context_fields(self, fields_dict, arg_tensors, parent_layer_entries)

    # Function config — lightweight hyperparameter extraction, always on.
    param_shapes: Optional[List[Tuple]] = fields_dict.get("parent_param_shapes")
    fields_dict["func_config"] = extract_salient_args(
        layer_type,
        func_name,
        args,
        kwargs,
        param_shapes,
    )

    return fields_dict, parent_layer_entries, arg_tensors, parent_param_passes


def _classify_new_tensor_in_model_log(
    self,
    fields_dict: Dict[str, Any],
    fields_dict_onetensor: Dict[str, Any],
    new_tensor_label: str,
) -> None:
    """Update ModelLog categories (internally_initialized, merge points) for a new tensor."""
    if fields_dict["is_internally_initialized"]:
        self.internally_initialized_layers.append(new_tensor_label)
    if fields_dict["has_input_ancestor"] and any(
        (
            self[parent_layer].has_internally_initialized_ancestor
            and not self[parent_layer].has_input_ancestor
        )
        for parent_layer in fields_dict_onetensor["parent_layers"]
    ):
        self._layers_where_internal_branches_merge_with_input.append(new_tensor_label)


def _tag_tensor_and_track_variations(
    self,
    out: torch.Tensor,
    new_layer_entry,
    fields_dict_onetensor: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
) -> None:
    """Tag the output tensor with its label, add backward hook, and track parent content variations.

    Parent content variation tracking detects in-place mutations: if a parent
    tensor's value at function-call time (from arg_copies) differs from its
    saved activation, the pre-mutation value is recorded in
    ``children_tensor_versions``.  This is critical for validation replay,
    which needs the actual input values each child operation saw.
    """
    out.tl_tensor_label_raw = fields_dict_onetensor["tensor_label_raw"]  # type: ignore[attr-defined]
    if self.save_gradients:
        _add_backward_hook(self, out, out.tl_tensor_label_raw)  # type: ignore[attr-defined]

    for parent_label in new_layer_entry.parent_layers:
        parent = self[parent_label]
        if parent.has_saved_activations and self.save_function_args:
            parent_tensor_contents = _get_parent_contents(
                parent_label,
                arg_copies,
                kwarg_copies,
                new_layer_entry.parent_layer_arg_locs,
            )
            if not tensor_nanequal(parent_tensor_contents, parent.activation):
                parent.children_tensor_versions[new_layer_entry.tensor_label_raw] = (
                    parent_tensor_contents
                )
                parent.has_child_tensor_variations = True


def log_function_output_tensors_exhaustive(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> None:
    """Full metadata logging for each output tensor of a function call.

    For each loggable output tensor:
      1. Build per-tensor fields (label, shape, equivalence type, FLOPs).
      2. Create a LayerPassLog entry and optionally save activation data.
      3. Update bidirectional family links (parent→child, sibling, spouse).
      4. Tag the output tensor with ``tl_tensor_label_raw`` so downstream
         operations can identify it as a parent.
      5. Track parent content variations (for in-place mutation detection).

    Args:
        func: The original (unwrapped) function that was called.
        args: Positional arguments to the function.
        kwargs: Keyword arguments to the function.
        arg_copies: Pre-call copies of args (for child tensor variation tracking).
        kwarg_copies: Pre-call copies of kwargs.
        out_orig: Original output from the function (may be tensor, tuple, etc.).
        exec_ctx: Timing, RNG, and autocast state captured around the function call.
        is_bottom_level_func: True if this function was not called by another
            decorated function (i.e., it's a leaf in the decoration nesting).
    """
    fields_dict, parent_layer_entries, arg_tensors, parent_param_passes = _build_shared_fields_dict(
        self,
        func,
        func_name,
        args,
        kwargs,
        out_orig,
        exec_ctx,
        func_call_id,
    )

    if getattr(self, "intervention_ready", False):
        out_iter = list(_walk_output_tensors_with_paths(out_orig))
        autograd_saved_stats = _get_autograd_saved_stats_by_output_entries(out_iter)
    else:
        out_iter = [(out, (), None) for out in ensure_iterable(out_orig)]
        autograd_saved_stats = _get_autograd_saved_stats_by_output(out_orig)

    for i, (out, output_path, container_spec) in enumerate(out_iter):
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue

        # Each output tensor gets its own fields_dict to avoid shared-mutation bugs.
        # Shallow-copy mutable containers (list, dict, set); immutable values
        # (str, int, bool, None, tuple, torch.dtype) are safely shared.
        fields_dict_onetensor: dict[str, Any] = {}
        for key, value in fields_dict.items():
            if isinstance(value, (list, dict, set)):
                fields_dict_onetensor[key] = copy.copy(value)
            else:
                fields_dict_onetensor[key] = value
        # These nested structures need deep copies because they contain mutable
        # sub-containers that could be mutated independently per output tensor.
        for field in (
            "parent_layer_arg_locs",
            "containing_modules",
            "parent_param_passes",
        ):
            fields_dict_onetensor[field] = copy.deepcopy(fields_dict[field])
        fields_dict_onetensor["output_path"] = output_path
        fields_dict_onetensor["container_spec"] = container_spec
        if container_spec is not None:
            fields_dict_onetensor["is_part_of_iterable_output"] = True
        _log_output_tensor_info(
            self,
            out,
            i,
            args,
            kwargs,
            parent_param_passes,
            fields_dict_onetensor,
            autograd_saved_stats.get(i, (None, None)),
        )
        if getattr(self, "intervention_ready", False):
            fields_dict_onetensor["edge_uses"] = _build_edge_use_records(
                self,
                fields_dict_onetensor["parent_layer_arg_locs"],
                fields_dict_onetensor["tensor_label_raw"],
                func_call_id,
            )
        _make_layer_log_entry(
            self,
            out,
            fields_dict=fields_dict_onetensor,
            t_args=arg_copies,
            t_kwargs=kwarg_copies,
            activation_postfunc=self.activation_postfunc,
        )
        new_layer_entry = self[fields_dict_onetensor["tensor_label_raw"]]
        new_tensor_label = new_layer_entry.tensor_label_raw
        _update_tensor_family_links(self, new_layer_entry)

        _classify_new_tensor_in_model_log(
            self, fields_dict, fields_dict_onetensor, new_tensor_label
        )
        _tag_tensor_and_track_variations(
            self,
            out,
            new_layer_entry,
            fields_dict_onetensor,
            arg_copies,
            kwarg_copies,
        )


def _get_parent_contents(parent_label, arg_copies, kwarg_copies, parent_layer_arg_locs) -> Any:
    """Retrieve a parent tensor's pre-call value from the saved argument copies.

    Used for child tensor variation tracking: if a parent's value in arg_copies
    differs from its currently saved activation, the parent was mutated
    in-place between operations, and the variation is recorded.
    """
    for pos, label in parent_layer_arg_locs["args"].items():
        if label == parent_label:
            return index_nested(arg_copies, pos)
    for argname, label in parent_layer_arg_locs["kwargs"].items():
        if label == parent_label:
            return index_nested(kwarg_copies, argname)
    raise ValueError("Parent layer not found in function arguments.")


def log_function_output_tensors_fast(
    self,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> None:
    """Fast-path logging: save new activation values into existing graph entries.

    Skips all metadata collection.  Instead:
      1. Increment counters identically to the exhaustive pass (counter alignment).
      2. Reconstruct the raw label from counters and verify it maps to the same
         final label as the exhaustive pass (graph-change detection).
      3. Save activation data and update shape/dtype/timing metadata.

    If any counter, label, or parent mismatch is detected, raises ValueError
    telling the user to re-run ``log_forward_pass``.
    """
    # Minimal info collection — only what's needed for counter alignment and saving.
    layer_type = func_name.lower().replace("_", "")
    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {key: val for key, val in kwargs.items() if not _check_if_tensor_arg(val)}

    # O(1) tensor extraction via lookup table (replaces BFS crawl)
    arg_tensors, _ = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    out_iter = ensure_iterable(out_orig)

    for i, out in enumerate(out_iter):
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue
        # Mirror the exhaustive pass's counter increments exactly.
        # The raw label reconstructed here MUST match the exhaustive pass's label.
        self._layer_counter += 1
        self._raw_layer_type_counter[layer_type] += 1
        creation_order = self._layer_counter
        layer_type_num = self._raw_layer_type_counter[layer_type]
        tensor_label_raw = f"{layer_type}_{layer_type_num}_{creation_order}_raw"
        # Skip orphans — these were pruned from the graph during postprocessing.
        if tensor_label_raw in self.orphan_layers:
            continue
        # Map parent raw labels → final labels for graph-change verification.
        parent_layer_labels_raw = get_attr_values_from_tensor_list(
            arg_tensors, "tl_tensor_label_raw"
        )
        parent_layer_labels_orig = []
        for raw_label in parent_layer_labels_raw:
            if raw_label in self._raw_to_final_layer_labels:
                parent_layer_labels_orig.append(self._raw_to_final_layer_labels[raw_label])
            elif raw_label not in self.orphan_layers:
                raise ValueError(
                    f"Fast-path parent {raw_label} not found in raw→final label map "
                    f"and not in orphan_layers. The computational graph may have changed."
                )
        # Tag tensor so downstream ops can find this tensor's label.
        out.tl_tensor_label_raw = tensor_label_raw
        if tensor_label_raw not in self._raw_to_final_layer_labels:
            raise ValueError(
                "The computational graph changed for this forward pass compared to the original "
                "call to log_forward_pass (either due to different inputs or a different "
                "random seed), so save_new_activations failed. Please re-run "
                "log_forward_pass with the desired inputs."
            )
        orig_tensor_label = self._raw_to_final_layer_labels[tensor_label_raw]
        if orig_tensor_label in self.unlogged_layers:
            continue
        orig_layer_entry = self.layer_dict_main_keys[orig_tensor_label]
        previous_shape = orig_layer_entry.tensor_shape

        if self.save_gradients:
            _add_backward_hook(self, out, tensor_label_raw)  # Must pass RAW label (#86)

        # Structural integrity check: verify counter, type, label, and parents
        # all match the exhaustive pass.  Any mismatch means dynamic control flow
        # changed the graph and the fast pass cannot proceed.
        if (
            orig_layer_entry.creation_order != self._layer_counter
            or orig_layer_entry.layer_type != layer_type
            or orig_layer_entry.tensor_label_raw != tensor_label_raw
            or set(orig_layer_entry.parent_layers) != set(parent_layer_labels_orig)
        ):
            raise ValueError(
                "The computational graph changed for this forward pass compared to the original "
                "call to log_forward_pass (either due to different inputs or a different "
                "random seed), so save_new_activations failed. Please re-run "
                "log_forward_pass with the desired inputs."
            )

        # Save activation data if this layer is in the save list.
        if (self._layer_nums_to_save == "all") or (
            orig_layer_entry.creation_order in self._layer_nums_to_save
        ):
            self.layers_with_saved_activations.append(orig_layer_entry.layer_label)
            orig_layer_entry.save_tensor_data(
                out,
                arg_copies,
                kwarg_copies,
                self.save_function_args,
                self.activation_postfunc,
            )
            # Output layers are identity wrappers whose activation come
            # from their parent.  Propagate the parent's saved activation to
            # any child that is an output layer so postprocess_fast can find it.
            for child_layer in orig_layer_entry.child_layers:
                if child_layer in self.output_layers:
                    child_output = self.layer_dict_main_keys[child_layer]
                    if (
                        orig_layer_entry.has_child_tensor_variations
                        and child_layer in orig_layer_entry.children_tensor_versions
                    ):
                        # children_tensor_versions already has postfunc applied.
                        tensor_to_save = orig_layer_entry.children_tensor_versions[child_layer]
                        child_output._internal_set("activation", safe_copy(tensor_to_save))
                    else:
                        child_output._internal_set("activation", safe_copy(out))
                        if self.activation_postfunc is not None:
                            # pause_logging prevents activation_postfunc from
                            # triggering decorated torch ops that would be logged.
                            with pause_logging():
                                child_output._internal_set(
                                    "transformed_activation",
                                    self.activation_postfunc(child_output.activation),
                                )
                            if not getattr(self, "save_raw_activation", True):
                                child_output._internal_set("activation", None)
                    child_output.has_saved_activations = True
                    if child_output.activation is not None:
                        child_output.tensor_memory = get_tensor_memory_amount(
                            child_output.activation
                        )

        # Update lightweight metadata that may vary across inputs
        # (shape can differ for dynamic-shape models that still share graph structure).
        new_shape = tuple(out.shape)
        if previous_shape is not None and new_shape != previous_shape:
            import warnings

            warnings.warn(
                f"Tensor shape changed for '{orig_tensor_label}': "
                f"expected {previous_shape}, got {new_shape}. "
                f"The computational graph may have changed between passes."
            )
        orig_layer_entry.tensor_shape = new_shape
        orig_layer_entry.tensor_dtype = out.dtype
        orig_layer_entry.tensor_memory = get_tensor_memory_amount(out)
        (
            orig_layer_entry.autograd_saved_bytes,
            orig_layer_entry.autograd_saved_tensor_count,
        ) = _get_autograd_saved_stats_for_tensor(out)
        orig_layer_entry.func_time = exec_ctx.time_elapsed
        orig_layer_entry.func_rng_states = exec_ctx.rng_states
        orig_layer_entry.func_autocast_state = exec_ctx.autocast_state
        orig_layer_entry.func_positional_args_non_tensor = non_tensor_args
        orig_layer_entry.func_kwargs_non_tensor = non_tensor_kwargs

        # Update func_config — some may be input-dependent (e.g. interpolate size).
        orig_layer_entry.func_config = extract_salient_args(
            layer_type,
            func_name,
            args,
            kwargs,
            orig_layer_entry.parent_param_shapes,
        )


def _output_should_be_logged(out: Any, is_bottom_level_func: bool) -> bool:
    """Determine whether an output value should be logged as a new graph node.

    Two conditions must hold:
      1. ``out`` must be a torch.Tensor (non-tensor outputs like ints are skipped).
      2. Either the tensor is genuinely new (no ``tl_tensor_label_raw`` attribute),
         OR this is a bottom-level function.  Bottom-level functions are leaf
         operations in the decoration nesting — even if they return an already-
         labeled tensor (in-place ops), we log them to capture the operation.
         Non-bottom-level functions returning an already-labeled tensor are
         higher-level wrappers whose sub-operations were already logged.

    Returns:
        True if the output should be logged, False otherwise.
    """
    if type(out) is not torch.Tensor:
        return False

    if (not hasattr(out, "tl_tensor_label_raw")) or is_bottom_level_func:
        return True
    else:
        return False


def _check_if_tensor_arg(arg: Any) -> bool:
    """Helper function to check if an argument either is a tensor or is a list/tuple containing a tensor.

    Args:
        arg: argument

    Returns:
        True if arg is or contains a tensor, false otherwise
    """
    if issubclass(type(arg), torch.Tensor):
        return True
    elif type(arg) in [list, tuple]:
        for elt in arg:
            if issubclass(type(elt), torch.Tensor):
                return True
        return False
    elif type(arg) == dict:
        for val in arg.values():
            if issubclass(type(val), torch.Tensor):
                return True
        return False
    else:
        return False


def _iter_autograd_saved_candidates(grad_fn: Any) -> List[Any]:
    """Return accessible autograd-saved values from a grad_fn object.

    Parameters
    ----------
    grad_fn
        PyTorch autograd function object to inspect.

    Returns
    -------
    list
        Values exposed through ``saved_tensors`` and ``_saved_*`` attributes.
        Attribute access failures are ignored because PyTorch may release or
        guard some saved values.
    """
    saved_values: List[Any] = []
    try:
        saved_values.extend(getattr(grad_fn, "saved_tensors", ()))
    except Exception:
        pass

    for attr_name in grad_fn.__class__.__dict__:
        if not attr_name.startswith(_AUTOGRAD_SAVED_ATTR_PREFIX):
            continue
        try:
            saved_values.append(getattr(grad_fn, attr_name))
        except Exception:
            continue
    return saved_values


def _collect_tensor_values(value: Any) -> List[torch.Tensor]:
    """Collect tensor values from a shallow autograd-saved object.

    Parameters
    ----------
    value
        Value read from a grad_fn saved-tensor API.

    Returns
    -------
    list of torch.Tensor
        Tensor instances found in the value.
    """
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, torch.Tensor)]
    return []


def _add_autograd_saved_tensor(
    tensor: torch.Tensor,
    seen_data_ptrs: set[int],
) -> tuple[int, int]:
    """Return byte/count contribution for a deduped autograd-saved tensor.

    Parameters
    ----------
    tensor
        Saved tensor to measure.
    seen_data_ptrs
        Data pointers already counted for this operation.

    Returns
    -------
    tuple of int
        ``(bytes, tensor_count)`` contribution for this tensor.
    """
    try:
        data_ptr = tensor.data_ptr()
    except Exception:
        return 0, 0
    if data_ptr in seen_data_ptrs:
        return 0, 0
    seen_data_ptrs.add(data_ptr)
    with pause_logging():
        return tensor.numel() * tensor.element_size(), 1


def _get_autograd_saved_stats_by_output(
    output: Any,
) -> dict[int, tuple[Optional[int], Optional[int]]]:
    """Measure autograd-saved tensor bytes/counts for each tensor output.

    Parameters
    ----------
    output
        Raw function output from a decorated torch operation.

    Returns
    -------
    dict
        Mapping from output index to ``(autograd_saved_bytes,
        autograd_saved_tensor_count)``. Non-tensor outputs and tensors without
        ``grad_fn`` are omitted and handled by callers as ``None`` values.
    """
    stats_by_index: dict[int, tuple[Optional[int], Optional[int]]] = {}
    seen_grad_fns: set[int] = set()
    seen_data_ptrs: set[int] = set()

    for output_index, maybe_tensor in enumerate(ensure_iterable(output)):
        if not isinstance(maybe_tensor, torch.Tensor) or maybe_tensor.grad_fn is None:
            continue

        grad_fn = maybe_tensor.grad_fn
        grad_fn_id = id(grad_fn)
        if grad_fn_id in seen_grad_fns:
            stats_by_index[output_index] = (0, 0)
            continue
        seen_grad_fns.add(grad_fn_id)

        total_bytes = 0
        tensor_count = 0
        for saved_value in _iter_autograd_saved_candidates(grad_fn):
            for saved_tensor in _collect_tensor_values(saved_value):
                bytes_added, count_added = _add_autograd_saved_tensor(saved_tensor, seen_data_ptrs)
                total_bytes += bytes_added
                tensor_count += count_added
        stats_by_index[output_index] = (total_bytes, tensor_count)

    return stats_by_index


def _get_autograd_saved_stats_by_output_entries(
    output_entries: list[
        tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]
    ],
) -> dict[int, tuple[Optional[int], Optional[int]]]:
    """Measure autograd-saved tensor bytes/counts for path-aware outputs.

    Parameters
    ----------
    output_entries
        Path-aware output entries from ``_walk_output_tensors_with_paths``.

    Returns
    -------
    dict[int, tuple[Optional[int], Optional[int]]]
        Mapping from output entry index to autograd saved-byte/count stats.
    """

    stats_by_index: dict[int, tuple[Optional[int], Optional[int]]] = {}
    seen_grad_fns: set[int] = set()
    seen_data_ptrs: set[int] = set()

    for output_index, (maybe_tensor, _, _) in enumerate(output_entries):
        if maybe_tensor.grad_fn is None:
            continue

        grad_fn = maybe_tensor.grad_fn
        grad_fn_id = id(grad_fn)
        if grad_fn_id in seen_grad_fns:
            stats_by_index[output_index] = (0, 0)
            continue
        seen_grad_fns.add(grad_fn_id)

        total_bytes = 0
        tensor_count = 0
        for saved_value in _iter_autograd_saved_candidates(grad_fn):
            for saved_tensor in _collect_tensor_values(saved_value):
                bytes_added, count_added = _add_autograd_saved_tensor(saved_tensor, seen_data_ptrs)
                total_bytes += bytes_added
                tensor_count += count_added
        stats_by_index[output_index] = (total_bytes, tensor_count)

    return stats_by_index


def _get_autograd_saved_stats_for_tensor(
    tensor: torch.Tensor,
) -> tuple[Optional[int], Optional[int]]:
    """Measure autograd-saved tensor bytes/counts for a single tensor output.

    Parameters
    ----------
    tensor
        Tensor output from a decorated torch operation.

    Returns
    -------
    tuple
        ``(autograd_saved_bytes, autograd_saved_tensor_count)``. Both values
        are ``None`` when no grad_fn exists.
    """
    if tensor.grad_fn is None:
        return None, None
    stats_by_index = _get_autograd_saved_stats_by_output(tensor)
    return stats_by_index.get(0, (None, None))


def _log_output_tensor_info(
    self,
    t: torch.Tensor,
    i: int,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    parent_param_passes: Dict[str, int],
    fields_dict: Dict[str, Any],
    autograd_saved_stats: tuple[Optional[int], Optional[int]],
) -> None:
    """Populate per-tensor fields that differ across outputs of a single function call.

    This includes:
      - Counter-based label generation (``tensor_label_raw``).
      - Operation equivalence type assignment (used by loop detection to
        identify structurally identical operations across forward-pass iterations).
      - FLOPs computation.
      - Shape, dtype, and memory size.

    Label format: ``"{layer_type}_{type_num}_{realtime_num}_raw"``
      - ``layer_type``: normalized function name (e.g. "conv2d")
      - ``type_num``: how many times this layer_type has been seen (monotonic)
      - ``realtime_num``: global operation counter across all types (monotonic)

    Args:
        t: The output tensor.
        i: Index of this tensor in a multi-output function call (0 for single outputs).
        args: Positional args to the function that created the tensor.
        kwargs: Keyword args to the function.
        parent_param_passes: Dict mapping param barcodes to their current pass number.
        fields_dict: Per-tensor fields dict to populate (mutated in place).
    """
    layer_type = fields_dict["layer_type"]
    indiv_param_barcodes = list(parent_param_passes.keys())
    self._layer_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    creation_order = self._layer_counter
    layer_type_num = self._raw_layer_type_counter[layer_type]
    tensor_label_raw = f"{layer_type}_{layer_type_num}_{creation_order}_raw"

    # Determine operation equivalence type — the fingerprint used by loop detection
    # to group structurally identical operations (same layer across passes).
    if len(parent_param_passes) > 0:
        # Parameterized ops: equivalence is defined by the exact set of parameters
        # used, combined with the operation type.  E.g., two conv2d calls using the
        # same weight+bias tensors are the same layer on different passes.
        operation_equivalence_type = _make_raw_param_group_barcode(indiv_param_barcodes, layer_type)
        fields_dict["operation_equivalence_type"] = operation_equivalence_type
        self.layers_with_params[operation_equivalence_type].append(tensor_label_raw)
        fields_dict["pass_num"] = len(self.layers_with_params[operation_equivalence_type])
    else:
        # Non-parameterized ops: equivalence is a hash of the operation type,
        # non-tensor args, output index, and containing module.  Each unique
        # non-param operation is seen only once (pass_num=1).
        operation_equivalence_type = _get_operation_equivalence_type(
            args, kwargs, i, layer_type, fields_dict
        )
        fields_dict["operation_equivalence_type"] = operation_equivalence_type
        fields_dict["pass_num"] = 1

    # equivalent_operations is a DIRECT reference to the ModelLog-level set —
    # all entries sharing this equivalence type point to the same set object.
    self.equivalent_operations[operation_equivalence_type].add(tensor_label_raw)
    fields_dict["equivalent_operations"] = self.equivalent_operations[operation_equivalence_type]

    # In-place ops return the same tensor object, which already has tl_tensor_label_raw.
    fields_dict["func_is_inplace"] = hasattr(t, "tl_tensor_label_raw")
    fields_dict["grad_fn_name"] = type(t.grad_fn).__name__
    fields_dict["grad_fn_id"] = id(t.grad_fn) if t.grad_fn is not None else None
    # Autograd Function objects do not consistently support weak references.
    # Keep the object only until explicit backward capture has registered hooks;
    # the backward finalizer clears these strong refs to avoid pinning graphs.
    fields_dict["grad_fn_object"] = t.grad_fn
    fields_dict["corresponding_grad_fn"] = None

    if fields_dict["is_part_of_iterable_output"]:
        fields_dict["iterable_output_index"] = i
    else:
        fields_dict["iterable_output_index"] = None

    if (t.dtype == torch.bool) and (t.dim()) == 0:
        fields_dict["is_scalar_bool"] = True
        try:
            fields_dict["scalar_bool_value"] = t.item()
        except RuntimeError:
            # .item() forbidden inside torch.vmap context
            fields_dict["scalar_bool_value"] = None
    else:
        fields_dict["is_scalar_bool"] = False
        fields_dict["scalar_bool_value"] = None

    # General info
    fields_dict["tensor_label_raw"] = tensor_label_raw
    fields_dict["layer_total_num"] = None
    fields_dict["recurrent_group"] = []
    fields_dict["creation_order"] = creation_order
    fields_dict["operation_num"] = None
    fields_dict["source_model_log"] = self
    fields_dict["_pass_finished"] = False

    # Other labeling info
    fields_dict["layer_label"] = None
    fields_dict["layer_label_short"] = None
    fields_dict["layer_label_w_pass"] = None
    fields_dict["layer_label_w_pass_short"] = None
    fields_dict["layer_label_no_pass"] = None
    fields_dict["layer_label_no_pass_short"] = None
    fields_dict["layer_type"] = layer_type
    fields_dict["layer_label_raw"] = tensor_label_raw
    fields_dict["layer_type_num"] = layer_type_num
    fields_dict["num_passes"] = 1
    fields_dict["lookup_keys"] = []

    # Saved tensor info
    fields_dict["activation"] = None
    fields_dict["transformed_activation"] = None
    fields_dict["has_saved_activations"] = False
    fields_dict["activation_postfunc"] = self.activation_postfunc
    fields_dict["extra_data"] = {}
    fields_dict["args_captured"] = False
    fields_dict["captured_args"] = None
    fields_dict["captured_kwargs"] = None
    fields_dict["tensor_shape"] = tuple(t.shape)
    fields_dict["transformed_activation_shape"] = None
    fields_dict["tensor_dtype"] = t.dtype
    fields_dict["transformed_activation_dtype"] = None
    with pause_logging():
        fields_dict["tensor_memory"] = t.nelement() * t.element_size()
    fields_dict["transformed_activation_memory"] = None
    (
        fields_dict["autograd_saved_bytes"],
        fields_dict["autograd_saved_tensor_count"],
    ) = autograd_saved_stats

    # FLOPs computation
    fields_dict["flops_forward"] = compute_forward_flops(
        fields_dict.get("func_name"),  # type: ignore[arg-type]
        fields_dict["tensor_shape"],
        fields_dict.get("parent_param_shapes", []),
        args,
        kwargs,
    )
    fields_dict["flops_backward"] = compute_backward_flops(
        fields_dict.get("func_name"),  # type: ignore[arg-type]
        fields_dict["flops_forward"],
    )

    # Child tensor variation tracking
    fields_dict["has_child_tensor_variations"] = False
    fields_dict["children_tensor_versions"] = {}

    # If internally initialized, fix this information:
    if len(fields_dict["parent_layers"]) == 0:
        fields_dict["is_internally_initialized"] = True
        fields_dict["has_internally_initialized_ancestor"] = True
        fields_dict["internally_initialized_parents"] = []
        fields_dict["internally_initialized_ancestors"] = {tensor_label_raw}


def _make_layer_log_entry(
    self,
    t: torch.Tensor,
    fields_dict: Dict,
    t_args: Optional[Tuple] = None,
    t_kwargs: Optional[Dict] = None,
    activation_postfunc: Optional[Callable] = None,
):
    """Create a LayerPassLog (or BufferLog) entry and register it in ModelLog.

    Instantiates the appropriate log class from ``fields_dict``, conditionally
    saves activation data (if this layer is in ``_layer_nums_to_save``), and
    appends the entry to ``_raw_layer_dict`` and ``_raw_layer_labels_list``.

    Args:
        t: The tensor to log.
        fields_dict: Complete field dictionary (~80 fields) for the log entry.
        t_args: Positional arguments to the function that created the tensor.
        t_kwargs: Keyword arguments to the function that created the tensor.
        activation_postfunc: Optional transform applied to activations before saving.
    """
    if t_args is None:
        t_args = []  # type: ignore[assignment]
    if t_kwargs is None:
        t_kwargs = {}

    if fields_dict.get("is_buffer_layer"):
        new_entry = BufferLog(fields_dict)
    else:
        new_entry = LayerPassLog(fields_dict)  # type: ignore[assignment]
    if (self._layer_nums_to_save == "all") or (
        new_entry.creation_order in self._layer_nums_to_save
    ):
        new_entry.save_tensor_data(
            t,
            t_args,  # type: ignore[arg-type]
            t_kwargs,
            self.save_function_args,
            activation_postfunc,
        )
        self.layers_with_saved_activations.append(new_entry.tensor_label_raw)
    _apply_live_fire_records_to_entry(new_entry)
    self._raw_layer_dict[new_entry.tensor_label_raw] = new_entry
    self._raw_layer_labels_list.append(new_entry.tensor_label_raw)

    return new_entry
