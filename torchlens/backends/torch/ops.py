"""Functions for logging output tensors produced by decorated torch operations.

This module handles the creation and population of Op entries for every
tensor produced during a forward pass.  It covers both *exhaustive* mode (full
metadata collection) and *fast* mode (re-use of a previously logged graph with
new outs).

Architecture overview:
    Every decorated torch function wrapper calls ``log_function_output_tensors``
    after executing the original function.  This dispatcher routes to either:

    - ``log_function_output_tensors_exhaustive``: builds a complete
      ``fields_dict`` of ~80 fields per tensor, creates a Op entry,
      updates family links (parent/child/sibling/spouse), and optionally
      saves the out value.

    - ``log_function_output_tensors_fast``: skips metadata collection entirely.
      Increments counters to maintain alignment with the exhaustive pass,
      verifies the graph hasn't changed, and saves new out values into
      the existing Op entries.

Label format convention:
    Raw labels follow ``{layer_type}_{type_num}_{realtime_num}_raw``, e.g.
    ``"conv2d_3_47_raw"``.  During postprocessing, these are mapped to final
    labels like ``"conv2d_3:1"`` (layer 3, pass 1).

pause_logging usage:
    ``pause_logging()`` temporarily disables the logging toggle so that
    utility operations (e.g., ``get_memory_amount``, ``safe_copy``,
    ``activation_transform``) don't get logged as model operations.  It is
    used inside ``save_activation`` and wherever helper functions call
    decorated torch custom_methods on tensors.
"""

import copy
import dataclasses
import re
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator, cast

import torch

from ... import _state as _st
from ..._state import pause_logging
from ._tl import get_label_list, get_param_meta, get_tensor_label, get_tensor_meta, set_tensor_label
from .aliasing import (
    detect_torch_alias_contract,
    detect_torch_output_alias_contract,
    get_parent_contents_for_contract_position,
    parent_label_has_alias_contract,
)
from . import module_stack as _mstack
from ...errors import CaptureError
from ...fastlog._halt import HaltSignal
from ...quantities import Bytes
from ...utils.introspection import (
    _get_code_context,
    _get_tensors_and_params_from_obj,
    get_vars_of_type_from_obj,
)
from ...utils.display import _timed_phase
from ...utils.tensor_utils import (
    get_memory_amount_from_metadata,
    is_functorch_wrapped_tensor,
    safe_copy,
    safe_to,
    tensor_nanequal,
)
from ...utils.collections import index_nested, ensure_iterable
from ...capture.flops import compute_backward_flops, compute_forward_flops
from ...capture.projections import LiveOpView
from ...data_classes.op import (
    Op,
    _dtype_or_none,
    _dedup_saved_activation_out,
    _effective_activation_save_mode,
    _memory_or_none,
    _recursive_safe_copy,
    _shape_or_none,
    _stamp_reference_out,
    apply_transform,
    validate_streaming_transform_output,
    validate_train_mode_transform_output,
)
from ...ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
    OpEvent,
    OutputRef,
    OutputVersionEvent,
    ParentEdge,
)
from ...ir.intervention import FireResult, FunctionEventInput
from ...ir.container import (
    ContainerSpec,
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
    get_registered_container,
)
from ...ir.container_registry import ContainerLeafOccurrence, FuncSite, Phase, Role
from ...ir.refs import ParamRef, TensorRef
from ...ir.predicate import RetroactiveCaptureDecision
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...intervention.selectors import (
    BaseSelector,
    CompositeSelector,
    FollowedBySelector,
    label as make_label_selector,
)
from ...intervention.types import (
    ArgComponent,
    CapturedArgTemplate,
    EdgeUseRecord,
    FunctionRegistryKey,
    LiteralTensor,
    LiteralValue,
    ParentRef,
    Unsupported,
)
from ...intervention.hooks import make_live_site_proxy, normalize_hook_plan
from ...intervention.runtime import active_intervention_context
from ...capture.arg_positions import (
    FUNC_ARG_SPECS,
    ArgSpec,
    extract_tensors_and_params,
    _cache_dynamic_spec,
    _normalize_func_name,
)

from .tensor_tracking import (
    _add_tensor_backward_hook,
    _append_module_suffix_to_equivalence_class,
    _get_ancestors_from_parents,
    _get_equivalence_class,
    _get_hash_from_args,
    _locate_parent_tensors_in_args,
    _make_raw_param_group_barcode,
    _process_parent_param_ops,
)

from ..._errors import TorchLensPostfuncError
from ..._io import BlobRef
from ...fastlog._storage_resolver import _resolve_storage
from ..._training_validation import TrainingModeConfigError
from ...data_classes.internal_types import FuncExecutionContext
from ...capture.predicates import (
    _evaluate_halt,
    _evaluate_intervene_op,
    _evaluate_keep_op,
    _is_halt_only_capture,
    build_op_record_context,
)
from ...capture.projections import (
    append_projected_event,
    get_active_recording_state,
)
from ...fastlog.exceptions import PredicateError
from ...fastlog.types import (
    ActivationRecord,
    CaptureSpec,
    ModuleStackFrame,
    RecordContext,
    StorageIntent,
)
from ...capture.salient_args import extract_salient_args

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


_SHARED_FIELDS_TO_SHALLOW_COPY_PER_OUTPUT = (
    "interventions",
    "non_tensor_pos_args",
    "non_tensor_kwargs",
    "func_non_tensor_args",
    "transform_config",
    "parent_params",
    "_param_barcodes",
    "parent_param_ops",
    "_param_logs",
    "param_shapes",
    "parents",
    "_edge_uses",
    "root_ancestors",
    "children",
    "input_ancestors",
    "output_descendants",
    "internal_source_parents",
    "internal_source_ancestors",
    "in_conditionals",
    "terminal_bool_for",
    "conditional_branch_stack",
    "conditional_entry_children",
    "conditional_then_children",
    "conditional_elif_children",
    "conditional_else_children",
    "conditional_arm_children",
    "modules",
    "module_call_stack",
    "module_entry_arg_keys",
    "input_to_module_calls",
    "output_of_modules",
    "output_of_module_calls",
    "func_config",
)
_SHARED_FIELDS_TO_DEEP_COPY_PER_OUTPUT = ("parent_arg_positions",)


def _should_keep_alias_mutation_contract(trace: "Trace") -> bool:
    """Return whether mutation-position alias contracts can be consumed.

    Parameters
    ----------
    trace
        Active trace object.

    Returns
    -------
    bool
        True when replay, intervention, backward, or validation paths may use
        mutation-position metadata.
    """

    save_grads = getattr(trace, "save_grads", None)
    return bool(
        getattr(trace, "save_arg_values", False)
        or getattr(trace, "intervention_ready", False)
        or getattr(trace, "backward_ready", False)
        or save_grads not in (None, False)
        or getattr(trace, "_validation_active", False)
    )


_AUTOGRAD_SAVED_ATTR_PREFIX = "_saved_"
_UNSUPPORTED_OUTPUT_CONTAINER_WARNED: set[str] = set()
_LIVE_FIRE_RESULTS_ATTR = "_tl_live_fire_results"
TRANSFORM_FUNC_NAMES = frozenset(
    {
        "vmap",
        "grad",
        "grad_and_value",
        "jacrev",
        "jacfwd",
        "hessian",
        "vjp",
        "jvp",
        "linearize",
        "vjpfn",
        "jvpfn",
        "linearizefn",
    }
)
"""Function names reserved for torch.func transform boundary ops."""


@dataclass(slots=True)
class _RetainedLookbackPayload:
    """Payload retained for one retroactive-save candidate."""

    raw_out: torch.Tensor | None
    transformed_out: torch.Tensor | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    activation_memory: int
    transformed_shape: tuple[int, ...] | None
    transformed_dtype: torch.dtype | None
    transformed_memory: int | None


@dataclass(slots=True)
class _RetainedLookbackCandidate:
    """One bounded lookback candidate with payload and event identity."""

    raw_label: str
    payload: _RetainedLookbackPayload
    marked: bool = False


@dataclass(slots=True)
class _OutputTensorEntry:
    """One output entry prepared for exhaustive tensor logging."""

    value: Any
    container_path: tuple[OutputPathComponent, ...]
    container_spec: ContainerSpec | None
    autograd_stats: tuple[int | None, int | None]


def _snapshot_exhaustive_module_stack(self: "Trace") -> list[tuple[str, int]]:
    """Return the raw hook-stack module context.

    Parameters
    ----------
    self:
        Active trace.

    Returns
    -------
    list[tuple[str, int]]
        Raw ``(module_address, pass_index)`` stack snapshot.
    """

    return [
        (frame.address, frame.pass_index)
        for frame in _mstack.snapshot(self._exhaustive_module_stack)
    ]


def _tensor_ref_from_fields(tensor: torch.Tensor, fields_dict: dict[str, Any]) -> TensorRef:
    """Build an IR tensor reference from one raw fields dictionary.

    Parameters
    ----------
    tensor
        Tensor represented by the operation event.
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    TensorRef
        Backend-neutral tensor metadata and optional payload reference.
    """

    return TensorRef(
        label_raw=fields_dict["_label_raw"],
        shape=fields_dict["shape"],
        dtype=str(fields_dict["dtype"]),
        device=str(tensor.device),
        requires_grad=tensor.requires_grad,
        memory=fields_dict["activation_memory"],
        payload=fields_dict["out"],
        blob_ref=(
            BlobRef(blob_id=fields_dict["_pending_blob_id"], kind="out")  # type: ignore[arg-type]
            if fields_dict.get("_pending_blob_id") is not None
            else None
        ),
        backend_handle_id=str(id(tensor)),
    )


def _module_frames_from_fields(fields_dict: dict[str, Any]) -> tuple[ModuleFrame, ...]:
    """Convert raw module stack tuples to IR module frames.

    Parameters
    ----------
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    tuple[ModuleFrame, ...]
        Module stack snapshot for the operation event.
    """

    return tuple(
        ModuleFrame(
            address=address,
            address_normalized=None,
            module_type="",
            call_index=pass_index,
            fx_qualpath=None,
            entry_argnames=(),
        )
        for address, pass_index in fields_dict["modules"]
    )


def _param_refs_from_fields(fields_dict: dict[str, Any]) -> tuple[ParamRef, ...]:
    """Convert raw parameter metadata to IR parameter references.

    Parameters
    ----------
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    tuple[ParamRef, ...]
        Parameter references for the operation event.
    """

    refs: list[ParamRef] = []
    for barcode, param in zip(fields_dict["_param_barcodes"], fields_dict["parent_params"]):
        param_meta = get_param_meta(param)
        param_address = (
            ""
            if param_meta is None or param_meta.param_address is None
            else param_meta.param_address
        )
        refs.append(
            ParamRef(
                barcode=barcode,
                address=param_address,
                shape=tuple(param.shape),
                dtype=str(param.dtype),
                trainable=bool(param.requires_grad),
                module_address=None,
            )
        )
    return tuple(refs)


def _parent_edges_from_fields(fields_dict: dict[str, Any]) -> tuple[ParentEdge, ...]:
    """Convert raw parent labels to IR parent edges.

    Parameters
    ----------
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    tuple[ParentEdge, ...]
        Parent edges for the operation event.
    """

    positions_by_label: dict[str, tuple[Any, str]] = {}
    for location, label in fields_dict["parent_arg_positions"]["args"].items():
        positions_by_label.setdefault(label, (location, "arg"))
    for location, label in fields_dict["parent_arg_positions"]["kwargs"].items():
        positions_by_label.setdefault(label, (location, "kwarg"))

    return tuple(
        ParentEdge(
            parent_label_raw=label,
            arg_position=positions_by_label.get(label, (None, "arg"))[0],
            edge_use=positions_by_label.get(label, (None, "arg"))[1],
        )
        for label in fields_dict["parents"]
    )


def _op_event_from_log(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    fire_results: tuple[FireResult, ...] = (),
) -> OpEvent:
    """Build an ``OpEvent`` that mirrors a just-constructed ``Op``.

    Parameters
    ----------
    fields_dict
        Raw field mapping used to construct ``op_log``.
    tensor
        Live output tensor for backend metadata.
    fire_results
        Live intervention fire results associated with this output.

    Returns
    -------
    OpEvent
        Frozen operation event appended to ``CaptureEvents``.
    """

    tensor_ref = _tensor_ref_from_fields(tensor, fields_dict)
    transformed_ref = (
        None
        if fields_dict["transformed_out"] is None
        else TensorRef(
            label_raw=fields_dict["_label_raw"],
            shape=fields_dict["transformed_out_shape"],
            dtype=str(fields_dict["transformed_out_dtype"]),
            device=fields_dict["output_device"],
            requires_grad=None,
            memory=fields_dict["transformed_activation_memory"],
            payload=fields_dict["transformed_out"],
            blob_ref=(
                BlobRef(
                    blob_id=fields_dict["_pending_transformed_out_blob_id"],
                    kind="transformed_out",
                )  # type: ignore[arg-type]
                if fields_dict.get("_pending_transformed_out_blob_id") is not None
                else None
            ),
            backend_handle_id=None,
        )
    )
    backend_semantics = fields_dict.get("backend_semantics")
    if backend_semantics is None:
        backend_semantics = BackendSemantics(
            backend_grad_handle=fields_dict["grad_fn_handle"],
            grad_fn_class_name=fields_dict["grad_fn_class_name"],
            autograd_memory=fields_dict["autograd_memory"],
            num_autograd_tensors=fields_dict["num_autograd_tensors"],
            mutated_input_positions=(),
            aliased_output_inputs=(),
            unknown_aliasing=False,
            bytes_delta_at_call=fields_dict["bytes_delta_at_call"],
            bytes_peak_at_call=fields_dict["bytes_peak_at_call"],
        )
    return OpEvent(
        kind="source" if fields_dict["is_input"] or fields_dict["is_buffer"] else "op",
        label_raw=fields_dict["_label_raw"],
        layer_label_raw=fields_dict["_layer_label_raw"],
        layer_type=fields_dict["type"],
        raw_index=fields_dict["raw_index"],
        type_index=fields_dict["type_index"],
        step_index=fields_dict["step_index"] or 0,
        source_trace=trace,
        source_trace_id=str(id(trace)),
        tracing_finished=fields_dict["_tracing_finished"],
        construction_done=fields_dict["_construction_done"],
        function=FunctionCallRef(
            func=fields_dict["func"],
            func_name=fields_dict["func_name"],
            func_qualname=fields_dict["func_qualname"],
            func_call_id=fields_dict["func_call_id"],
            code_context=tuple(fields_dict["code_context"]),
            func_duration=fields_dict["func_duration"],
            flops_forward=fields_dict["flops_forward"],
            flops_backward=fields_dict["flops_backward"],
            func_rng_states=fields_dict["func_rng_states"],
            func_autocast_state=fields_dict["func_autocast_state"],
            arg_names=tuple(fields_dict["arg_names"]),
            num_args_total=fields_dict["num_args_total"],
            num_pos_args=fields_dict["num_pos_args"],
            num_kwargs=fields_dict["num_kwargs"],
            non_tensor_pos_args=tuple(fields_dict["non_tensor_pos_args"]),
            non_tensor_kwargs=tuple(fields_dict["non_tensor_kwargs"].items()),
            func_non_tensor_args=tuple(fields_dict["func_non_tensor_args"]),
            is_inplace=fields_dict["is_inplace"],
            func_config=tuple(fields_dict["func_config"].items()),
        ),
        output=OutputRef(
            tensor=tensor_ref,
            transformed_tensor=transformed_ref,
            has_saved_activation=fields_dict["has_saved_activation"],
            output_device=fields_dict["output_device"],
            activation_transform=fields_dict["activation_transform"],
            detach_saved_activations=fields_dict["detach_saved_activations"],
            visualizer_path=fields_dict["visualizer_path"],
            multi_output_index=fields_dict["multi_output_index"],
            in_multi_output=fields_dict["in_multi_output"],
            container_path=tuple(fields_dict["container_path"]),
            container_spec=fields_dict["container_spec"],
            child_versions=tuple(fields_dict["out_versions_by_child"].items()),
        ),
        templates=ArgTemplateRef(
            saved_args=fields_dict["saved_args"],
            saved_kwargs=fields_dict["saved_kwargs"],
            args_template=fields_dict["args_template"],
            kwargs_template=fields_dict["kwargs_template"],
            has_saved_args=fields_dict["has_saved_args"],
        ),
        parents=_parent_edges_from_fields(fields_dict),
        parent_arg_positions=copy.deepcopy(fields_dict["parent_arg_positions"]),
        _edge_uses=tuple(
            fields_dict["_edge_uses"]
            or _build_edge_use_records(
                trace,
                fields_dict["parent_arg_positions"],
                fields_dict["_label_raw"],
                fields_dict["func_call_id"],
            )
        ),
        params=_param_refs_from_fields(fields_dict),
        parent_params=tuple(fields_dict["parent_params"]),
        module_stack=_module_frames_from_fields(fields_dict),
        modules=tuple(fields_dict["modules"]),
        backend_semantics=backend_semantics,
        policy=CapturePolicy(
            must_keep_topology=True,
            save_payload=fields_dict["has_saved_activation"],
            requires_isolation=fields_dict["is_inplace"],
            save_args=fields_dict["has_saved_args"],
            save_code=bool(fields_dict["code_context"]),
            save_rng=bool(fields_dict["func_rng_states"]),
            save_grad=fields_dict["save_grads"],
            stream=False,
            save_mode=getattr(trace, "save_mode", "copy"),
        ),
        predicate_matched=True,
        pass_index=fields_dict["pass_index"],
        grad_fn_class_qualname=fields_dict["grad_fn_class_qualname"],
        grad_fn_handle=fields_dict["grad_fn_handle"],
        equivalence_class=fields_dict["equivalence_class"],
        is_transform=bool(fields_dict.get("is_transform", False)),
        transform_kind=fields_dict.get("transform_kind"),
        transform_chain=tuple(fields_dict.get("transform_chain") or ()),
        transform_config={
            **dict(fields_dict.get("transform_config") or {}),
            "_tl_annotations": dict(fields_dict.get("annotations") or {}),
        },
        transform_fn_name=fields_dict.get("transform_fn_name"),
        transform_fn_qualname=fields_dict.get("transform_fn_qualname"),
        transform_fn_source=fields_dict.get("transform_fn_source"),
        unattributed_tensor_args=tuple(fields_dict.get("unattributed_tensor_args") or ()),
        is_output_parent=fields_dict["is_output_parent"],
        has_internal_source_ancestor=fields_dict["has_internal_source_ancestor"],
        internal_source_ancestors=frozenset(fields_dict["internal_source_ancestors"]),
        input_ancestors=frozenset(fields_dict["input_ancestors"]),
        root_ancestors=frozenset(fields_dict["root_ancestors"]),
        func_call_id=fields_dict["func_call_id"],
        is_bottom_level=True,
        is_scalar_bool=fields_dict["is_scalar_bool"],
        bool_value=fields_dict["bool_value"],
        intervention_fired=bool(fire_results),
        intervention_replaced=fields_dict["intervention_replaced"],
        fire_results=fire_results,
        intervention_template_ref=None,
    )


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


def _torch_return_type_fields(value: Any) -> tuple[str, ...]:
    """Return public field names for a ``torch.return_types`` structseq.

    Parameters
    ----------
    value
        Object to inspect.

    Returns
    -------
    tuple[str, ...]
        Field names when PyTorch exposes a named structseq, otherwise ``()``.
    """

    cls = type(value)
    if cls.__module__ != "torch.return_types":
        return ()
    if not isinstance(value, tuple):
        return ()
    n_fields = getattr(value, "n_fields", None)
    n_unnamed = getattr(value, "n_unnamed_fields", 0)
    if not isinstance(n_fields, int) or n_fields <= 0 or n_unnamed:
        return ()
    field_names = tuple(re.findall(r"^\s*([A-Za-z_]\w*)=", repr(value), flags=re.MULTILINE))
    if len(field_names) != n_fields:
        return ()
    return field_names


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

    if _literal_value_supported(value) or isinstance(value, torch.Size):
        return ContainerSpec(kind="literal", literal_value=value)
    child_specs: list[tuple[OutputPathComponent, ContainerSpec]] = []
    registered = get_registered_container(type(value))
    if registered is not None:
        children, aux_data = registered.flatten(value)
        for index, item in enumerate(children):
            child_spec = _build_container_spec(item)
            if child_spec is not None:
                child_specs.append((TupleIndex(index), child_spec))
        module, qualname = _container_type_ref(value)
        return ContainerSpec(
            kind="registered",
            length=len(children),
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
            aux_data=aux_data,
        )
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
    torch_fields = _torch_return_type_fields(value)
    if _is_namedtuple_instance(value) or torch_fields:
        fields = torch_fields or tuple(value._fields)
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
    registered = get_registered_container(type(out))
    if registered is not None:
        children, _aux_data = registered.flatten(out)
        for index, item in enumerate(children):
            yield from _walk_supported_output_container(
                item,
                root_spec=root_spec,
                path=(*path, TupleIndex(index)),
            )
        return
    if _is_hf_model_output(out):
        for key in out.keys():
            yield from _walk_supported_output_container(
                out[key],
                root_spec=root_spec,
                path=(*path, HFKey(key)),
            )
        return
    torch_fields = _torch_return_type_fields(out)
    if _is_namedtuple_instance(out) or torch_fields:
        fields = torch_fields or tuple(out._fields)
        for field_name in fields:
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


def _function_registry_key(func: Callable[..., Any]) -> FunctionRegistryKey:
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

    from torchlens.intervention.resolver import function_registry_key_from_callable

    return function_registry_key_from_callable(func)


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

    label = None if isinstance(value, torch.nn.Parameter) else get_tensor_label(value)
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


def _build_args_template(
    func: Callable[..., Any],
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
        Location key from ``parent_arg_positions``.

    Returns
    -------
    tuple[OutputPathComponent, ...]
        Path tuple mirroring the existing two-level arg-location scheme.
    """

    if isinstance(location, tuple):
        return location
    return (location,)


def _build_edge_use_records(
    self: "Trace",
    parent_arg_positions: dict[str, dict[Any, str]],
    child_label: str,
    child_func_call_id: int,
) -> list[EdgeUseRecord]:
    """Build edge provenance records from existing parent arg locations.

    Parameters
    ----------
    self
        Active model log.
    parent_arg_positions
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

    _edge_uses: list[EdgeUseRecord] = []
    for location, parent_label in parent_arg_positions["args"].items():
        _edge_uses.append(
            EdgeUseRecord(
                parent_label=parent_label,
                child_label=child_label,
                arg_kind="positional",
                arg_path=_arg_location_to_path(location),
                view_or_copy="unknown",
                parent_func_call_id=self.capture_events.live_index.require_event(
                    parent_label
                ).func_call_id,
                child_func_call_id=child_func_call_id,
            )
        )
    for location, parent_label in parent_arg_positions["kwargs"].items():
        _edge_uses.append(
            EdgeUseRecord(
                parent_label=parent_label,
                child_label=child_label,
                arg_kind="keyword",
                arg_path=_arg_location_to_path(location),
                view_or_copy="unknown",
                parent_func_call_id=self.capture_events.live_index.require_event(
                    parent_label
                ).func_call_id,
                child_func_call_id=child_func_call_id,
            )
        )
    return _edge_uses


def _tensor_has_known_provenance(value: torch.Tensor) -> bool:
    """Return whether a tensor carries TorchLens input/op/buffer provenance.

    Parameters
    ----------
    value:
        Tensor argument to inspect.

    Returns
    -------
    bool
        True when the tensor is a Parameter or has TorchLens tensor metadata.
    """

    if isinstance(value, torch.nn.Parameter):
        return True
    meta = get_tensor_meta(value)
    return meta is not None and any(
        item is not None for item in (meta.label_raw, meta.address, meta.buffer_source)
    )


def _unattributed_tensor_arg_positions(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[str, ...]:
    """Find tensor arguments that will not become graph parents or known sources.

    Parameters
    ----------
    args:
        Positional function arguments.
    kwargs:
        Keyword function arguments.

    Returns
    -------
    tuple[str, ...]
        Stable argument-position strings such as ``"arg1"`` or ``"kw:mask.0"``.
    """

    positions: list[str] = []

    def visit(value: Any, path: str) -> None:
        """Append unattributed tensor positions under ``path``."""

        if isinstance(value, torch.Tensor):
            if not _tensor_has_known_provenance(value):
                positions.append(path)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")

    for index, arg in enumerate(args):
        visit(arg, f"arg{index}")
    for key, value in kwargs.items():
        visit(value, f"kw:{key}")
    return tuple(positions)


def log_function_output_tensors(
    self: "Trace",
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> None:
    """Dispatch to exhaustive or fast logging based on current logging mode.

    Called by every decorated torch function wrapper after executing the
    original function.  The mode was set in ``save_new_outs`` (fast)
    or ``trace`` (exhaustive).
    """
    if self.capture_mode == "exhaustive":
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
    elif self.capture_mode == "fast":
        log_function_output_tensors_fast(
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
    elif self.capture_mode == "predicate":
        log_function_output_tensors_predicate(
            self,
            func,
            func_name,
            args,
            kwargs,
            arg_copies,
            kwarg_copies,
            out_orig,
            is_bottom_level_func,
            func_call_id,
        )


def apply_live_hooks_to_outputs(
    self: "Trace",
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
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
        Fired hook results are stored temporarily on the tensor being logged.
    """

    predicate_intervene_active = _trace_intervene_options(self) is not None
    if (
        not _st._active_hook_plan
        and not predicate_intervene_active
        or self.capture_mode not in {"exhaustive", "predicate"}
    ):
        return out_orig
    if (
        self.capture_mode == "predicate"
        and predicate_intervene_active
        and not _st._active_hook_plan
    ):
        return _apply_predicate_mode_interventions_to_outputs(
            self,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            out_orig=out_orig,
            is_bottom_level_func=is_bottom_level_func,
            func_call_id=func_call_id,
        )

    from ...intervention.runtime import _apply_live_hooks

    shared_fields, _parent_layer_entries, _arg_tensors, _parent_param_ops = (
        _build_shared_fields_dict(
            self, func, func_name, args, kwargs, out_orig, exec_ctx, func_call_id
        )
    )
    layer_type = shared_fields["type"]
    replacements: dict[tuple[OutputPathComponent, ...], torch.Tensor] = {}
    loggable_outputs = list(_iter_loggable_live_outputs(out_orig, is_bottom_level_func))
    events = self.capture_events
    events.raw_layer_counter = self._layer_counter
    events.raw_layer_type_counter = dict(self._raw_layer_type_counter)
    reserved_labels = events.reserve_label_block(layer_type, len(loggable_outputs))

    for reserved, (out, container_path, _container_spec) in zip(reserved_labels, loggable_outputs):
        raw_label = reserved.label_raw
        site_fields = dict(shared_fields)
        site_fields["raw_index"] = reserved.raw_index
        site_fields["type_index"] = reserved.type_index
        site_fields["_label_raw"] = raw_label
        site_fields["_layer_label_raw"] = raw_label
        site_fields["pass_index"] = 1
        site_fields["step_index"] = None
        site_fields["container_path"] = container_path
        site = make_live_site_proxy(
            _layer_label_raw=raw_label,
            func_name=func_name,
            layer_type=layer_type,
            tensor=out,
            func_call_id=func_call_id,
            container_path=container_path,
            fields=site_fields,
        )
        hooked = out
        all_fire_results: list[FireResult] = []
        if _st._active_hook_plan:
            hooked, fire_results = _apply_live_hooks(
                hooked, site=site, container_path=container_path
            )
            all_fire_results.extend(fire_results)
        if predicate_intervene_active:
            hooked, fire_results = _apply_predicate_intervention(
                self,
                func_name=func_name,
                out=hooked,
                site=site,
                site_fields=site_fields,
                parent_labels=tuple(site_fields.get("parents", ())),
                output_index=_live_output_index(container_path),
                is_bottom_level_func=is_bottom_level_func,
                container_path=container_path,
            )
            all_fire_results.extend(fire_results)
        fire_results = tuple(all_fire_results)
        if fire_results:
            _set_tensor_live_fire_results(hooked, fire_results)
        if hooked is not out:
            replacements[container_path] = hooked

    if not replacements:
        return out_orig
    if isinstance(out_orig, torch.Tensor):
        return replacements.get((), out_orig)
    return _replace_output_tensors_by_path(out_orig, replacements)


def _apply_predicate_mode_interventions_to_outputs(
    trace: "Trace",
    *,
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> Any:
    """Apply predicate interventions in fastlog predicate mode."""

    options = _trace_intervene_options(trace)
    if options is None:
        return out_orig
    state = get_active_recording_state()
    layer_type = func_name.lower().replace("_", "")
    arg_tensors, _ = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    parent_labels = tuple(get_label_list(arg_tensors))
    replacements: dict[tuple[OutputPathComponent, ...], torch.Tensor] = {}
    loggable_outputs = list(_iter_loggable_live_outputs(out_orig, is_bottom_level_func))
    output_ordinal = 0
    for out, container_path, _container_spec in loggable_outputs:
        output_ordinal += 1
        raw_index = trace._layer_counter + output_ordinal
        type_index = trace._raw_layer_type_counter[layer_type] + output_ordinal
        raw_label = f"{layer_type}_{type_index}_{raw_index}_raw"
        module_frame = state.module_stack[-1] if state.module_stack else None
        ctx = build_op_record_context(
            kind="op",
            label=raw_label,
            raw_label=raw_label,
            raw_index=raw_index,
            layer_type=layer_type,
            type_index=type_index,
            func_name=func_name,
            parent_labels=parent_labels,
            tensor=out,
            output_index=_live_output_index(container_path),
            is_bottom_level_func=is_bottom_level_func,
            module_stack=state.module_stack,
            history=tuple(state.history),
            op_counts=state.op_counts,
            pass_index=state.pass_index,
            event_index=state.event_index + output_ordinal,
            step_index=state.step_index + output_ordinal,
            capture_start_time=trace.capture_start_time,
            include_source_events=state.options.include_source_events,
            sample_id=state.sample_id,
            address=module_frame.address if module_frame else None,
            module_type=module_frame.module_type if module_frame else None,
            module_pass_index=module_frame.pass_index if module_frame else None,
            is_transform=func_name in TRANSFORM_FUNC_NAMES,
            transform_kind=func_name if func_name in TRANSFORM_FUNC_NAMES else None,
        )
        decision = _evaluate_intervene_op(ctx, options)
        if decision is None:
            continue
        site = make_live_site_proxy(
            _layer_label_raw=raw_label,
            func_name=func_name,
            layer_type=layer_type,
            tensor=out,
            func_call_id=func_call_id,
            container_path=container_path,
            fields={
                "_label_raw": raw_label,
                "_layer_label_raw": raw_label,
                "raw_index": raw_index,
                "type_index": type_index,
            },
        )
        hook_entries = normalize_hook_plan(
            decision.hook,
            default_site_target=make_label_selector(ctx.raw_label or ctx.label),
        )
        from ...intervention.runtime import _apply_live_hooks

        with active_intervention_context(
            intervention_spec=_st._active_intervention_spec,
            hook_plan=hook_entries,
        ):
            hooked, fire_results = _apply_live_hooks(out, site=site, container_path=container_path)
        if fire_results:
            _set_tensor_live_fire_results(hooked, fire_results)
        if hooked is not out:
            replacements[container_path] = hooked
    if not replacements:
        return out_orig
    if isinstance(out_orig, torch.Tensor):
        return replacements.get((), out_orig)
    return _replace_output_tensors_by_path(out_orig, replacements)


def _trace_intervene_options(trace: "Trace") -> Any | None:
    """Return trace predicate options when ``intervene`` is configured."""

    options = getattr(trace, "_predicate_save_options", None)
    if options is None or options.intervene is None:
        return None
    return options


def _live_output_index(container_path: tuple[OutputPathComponent, ...]) -> int | None:
    """Return the first tuple/list path index for a live output."""

    if not container_path:
        return None
    first = container_path[0]
    if isinstance(first, TupleIndex):
        return first.index
    if isinstance(first, int):
        return first
    return None


def _apply_predicate_intervention(
    trace: "Trace",
    *,
    func_name: str,
    out: torch.Tensor,
    site: Any,
    site_fields: dict[str, Any],
    parent_labels: tuple[str, ...],
    output_index: int | None,
    is_bottom_level_func: bool,
    container_path: tuple[OutputPathComponent, ...],
) -> tuple[torch.Tensor, tuple[FireResult, ...]]:
    """Evaluate and apply a current-op predicate intervention."""

    options = _trace_intervene_options(trace)
    if options is None:
        return out, ()
    ctx = _build_trace_predicate_context(
        trace,
        site_fields,
        out,
        parent_labels=parent_labels,
        output_index=output_index,
        is_bottom_level_func=is_bottom_level_func,
    )
    _cache_trace_predicate_context(trace, ctx, container_path)
    decision = _evaluate_intervene_op(ctx, options)
    if decision is None:
        return out, ()
    hook_entries = normalize_hook_plan(
        decision.hook,
        default_site_target=make_label_selector(ctx.raw_label or ctx.label),
    )
    from ...intervention.runtime import _apply_live_hooks

    with active_intervention_context(
        intervention_spec=_st._active_intervention_spec,
        hook_plan=hook_entries,
    ):
        return _apply_live_hooks(out, site=site, container_path=container_path)


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

    for out, container_path, container_spec in _walk_output_tensors_with_paths(out_orig):
        if _output_should_be_logged(out, is_bottom_level_func):
            yield out, container_path, container_spec


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


def _set_tensor_live_fire_results(
    tensor: torch.Tensor,
    fire_results: tuple[FireResult, ...],
) -> None:
    """Attach transient live hook results to the tensor that will be logged.

    Parameters
    ----------
    tensor
        Tensor returned by live hook handling.
    fire_results
        Typed hook fire results for the corresponding raw output site.

    Returns
    -------
    None
        The tensor receives best-effort transient metadata.
    """

    try:
        setattr(tensor, _LIVE_FIRE_RESULTS_ATTR, fire_results)
    except Exception:
        pass


def _pop_tensor_live_fire_results(tensor: torch.Tensor) -> tuple[FireResult, ...]:
    """Return and clear transient live hook results attached to ``tensor``.

    Parameters
    ----------
    tensor
        Tensor about to be materialized into an ``Op``.

    Returns
    -------
    tuple[FireResult, ...]
        Hook fire results associated with the tensor, if any.
    """

    fire_results = getattr(tensor, _LIVE_FIRE_RESULTS_ATTR, ())
    try:
        delattr(tensor, _LIVE_FIRE_RESULTS_ATTR)
    except Exception:
        pass
    return tuple(fire_results)


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
    if _is_hf_model_output(value):
        key_values = {
            key: _replace_output_value(item, (*path, HFKey(key)), replacements)
            for key, item in value.items()
        }
        return type(value)(**key_values)
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


def _record_predicate_output(
    ctx: Any,
    out: torch.Tensor,
    spec: CaptureSpec,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Store a predicate-selected operation output."""

    if not spec.save_out and not spec.save_metadata:
        return None, None
    state = get_active_recording_state()
    ram_payload = None
    disk_payload = None
    transformed_ram_payload = None
    transformed_disk_payload = None
    if spec.save_out:
        (
            ram_payload,
            disk_payload,
            transformed_ram_payload,
            transformed_disk_payload,
        ) = state.resolve_storage(out, spec, ctx=ctx)
    if state.storage_intent.on_disk:
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
    return ram_payload, transformed_ram_payload


def log_function_output_tensors_predicate(
    self: "Trace",
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    out_orig: Any,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> None:
    """Predicate-mode logging for decorated torch function outputs."""

    state = get_active_recording_state()
    layer_type = func_name.lower().replace("_", "")
    arg_tensors, _ = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    parent_labels = tuple(get_label_list(arg_tensors))
    out_iter = list(_iter_loggable_live_outputs(out_orig, is_bottom_level_func))

    for output_index, (out, container_path, _container_spec) in enumerate(out_iter):
        self._layer_counter += 1
        self._raw_layer_type_counter[layer_type] += 1
        state.op_counts[layer_type] = state.op_counts.get(layer_type, 0) + 1
        state.step_index += 1
        state.event_index += 1
        raw_index = self._layer_counter
        type_index = self._raw_layer_type_counter[layer_type]
        _label_raw = f"{layer_type}_{type_index}_{raw_index}_raw"
        set_tensor_label(out, _label_raw)
        module_frame = state.module_stack[-1] if state.module_stack else None
        with _timed_phase(self, "ctx_build:record_context"):
            ctx = build_op_record_context(
                kind="op",
                label=_label_raw,
                raw_label=_label_raw,
                raw_index=raw_index,
                layer_type=layer_type,
                type_index=type_index,
                func_name=func_name,
                parent_labels=parent_labels,
                tensor=out,
                output_index=_live_output_index(container_path) or output_index,
                is_bottom_level_func=is_bottom_level_func,
                module_stack=state.module_stack,
                history=tuple(state.history),
                op_counts=state.op_counts,
                pass_index=state.pass_index,
                event_index=state.event_index,
                step_index=state.step_index,
                capture_start_time=self.capture_start_time,
                include_source_events=state.options.include_source_events,
                sample_id=state.sample_id,
                address=module_frame.address if module_frame else None,
                module_type=module_frame.module_type if module_frame else None,
                module_pass_index=module_frame.pass_index if module_frame else None,
                is_transform=bool(getattr(func, "__tl_is_transform_boundary__", False))
                or func_name in TRANSFORM_FUNC_NAMES,
                transform_kind=getattr(func, "__tl_transform_kind__", None)
                or (func_name if func_name in TRANSFORM_FUNC_NAMES else None),
            )
        try:
            halt_only = _is_halt_only_capture(state.options)
            if halt_only:
                _evaluate_halt(ctx, state.options, frontier_output=out)
                continue
            if out.grad_fn is not None:
                state.grad_fn_to_context[out.grad_fn] = ctx
            spec = _evaluate_keep_op(ctx, state.options)
            if isinstance(spec, RetroactiveCaptureDecision):
                raise PredicateError(
                    "tl.followed_by(...) retroactive save is only supported by trace"
                )
            ram_payload, transformed_ram_payload = _record_predicate_output(ctx, out, spec)
            grad_fn_handle = out.grad_fn if isinstance(out, torch.Tensor) else None
            func_event_input = FunctionEventInput(
                func=func,
                func_name=func_name,
                func_qualname=getattr(func, "__qualname__", None),
                args=args,
                kwargs=kwargs,
                raw_output=out_orig,
                arg_copies=arg_copies,
                kwarg_copies=kwarg_copies,
                module_stack=(),
                is_bottom_level_func=is_bottom_level_func,
                func_call_id=func_call_id,
                expected_output_count=len(out_iter),
            )
            detect_backend_semantics = (
                detect_torch_alias_contract
                if _should_keep_alias_mutation_contract(self)
                else detect_torch_output_alias_contract
            )
            backend_semantics = detect_backend_semantics(
                func_event_input,
                backend_grad_handle=grad_fn_handle,
                grad_fn_class_name=type(grad_fn_handle).__name__
                if grad_fn_handle is not None
                else None,
                autograd_memory=None,
                num_autograd_tensors=None,
            )
            function_ref = FunctionCallRef(
                func=func,
                func_name=func_name,
                func_qualname=getattr(func, "__qualname__", None),
                func_call_id=func_call_id,
                code_context=(),
                func_duration=None,
                flops_forward=None,
                flops_backward=None,
                func_rng_states=None,
                func_autocast_state=None,
                arg_names=(),
                num_args_total=len(args) + len(kwargs),
                num_pos_args=len(args),
                num_kwargs=len(kwargs),
                non_tensor_pos_args=(),
                non_tensor_kwargs=tuple(
                    (key, value)
                    for key, value in kwargs.items()
                    if not isinstance(value, torch.Tensor)
                ),
                func_non_tensor_args=(),
                is_inplace=False,
                func_config=(),
            )
            append_projected_event(
                self,
                ctx,
                spec,
                tensor=out,
                ram_payload=ram_payload,
                transformed_ram_payload=transformed_ram_payload,
                predicate_matched=spec.save_out or spec.save_metadata,
                backend_semantics=backend_semantics,
                function=function_ref,
                container_path=container_path,
            )
            _evaluate_halt(ctx, state.options, frontier_output=out)
        except HaltSignal:
            raise
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
            if not halt_only:
                if not any(
                    event.raw_index == raw_index
                    for event in getattr(getattr(self, "capture_events", None), "op_events", ())
                ):
                    append_projected_event(
                        self,
                        ctx,
                        CaptureSpec(save_out=False, save_metadata=False),
                        tensor=out,
                        predicate_matched=False,
                    )
                state.append_context(ctx)


def _build_graph_relationship_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    parent_layer_labels: list[str],
    parent_layer_entries: list[Op],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
) -> None:
    """Populate graph structure fields: parents, children, ancestors, buffer/IO flags."""
    out_kwarg_label = None
    out_kwarg = kwargs.get("out")
    if isinstance(out_kwarg, torch.Tensor):
        out_kwarg_label = get_tensor_label(out_kwarg)
    if out_kwarg_label is not None and out_kwarg_label not in parent_layer_labels:
        parent_layer_labels = [*parent_layer_labels, out_kwarg_label]
        parent_layer_entries = [
            *parent_layer_entries,
            cast(
                Op, LiveOpView(self, self.capture_events.live_index.require_event(out_kwarg_label))
            ),
        ]
    parent_arg_positions = _locate_parent_tensors_in_args(self, parent_layer_entries, args, kwargs)
    input_ancestors, internal_source_ancestors = _get_ancestors_from_parents(parent_layer_entries)
    internal_parent_layer_labels = [
        label
        for label in parent_layer_labels
        if self.capture_events.live_index.require_event(label).has_internal_source_ancestor
    ]

    fields_dict["parents"] = parent_layer_labels
    fields_dict["parent_arg_positions"] = parent_arg_positions
    fields_dict["_edge_uses"] = []
    fields_dict["root_ancestors"] = input_ancestors.union(internal_source_ancestors)
    fields_dict["children"] = []
    fields_dict["has_children"] = False
    fields_dict["is_input"] = False
    fields_dict["has_input_ancestor"] = len(input_ancestors) > 0
    fields_dict["input_ancestors"] = input_ancestors
    fields_dict["min_distance_from_input"] = None
    fields_dict["max_distance_from_input"] = None
    fields_dict["is_output"] = False
    fields_dict["is_output_parent"] = False
    fields_dict["is_final_output"] = False
    fields_dict["has_output_descendant"] = False
    fields_dict["output_descendants"] = set()
    fields_dict["is_orphan"] = False
    fields_dict["min_distance_to_output"] = None
    fields_dict["max_distance_to_output"] = None
    fields_dict["io_role"] = None
    fields_dict["is_buffer"] = False
    fields_dict["address"] = None
    fields_dict["buffer_pass"] = None
    fields_dict["buffer_source"] = None
    fields_dict["buffer_write_kind"] = None
    fields_dict["buffer_value_changed"] = None
    fields_dict["buffer_replay_validated"] = None
    fields_dict["buffer_source_func_name"] = None
    fields_dict["is_internal_source"] = len(parent_layer_labels) == 0
    fields_dict["has_internal_source_ancestor"] = len(internal_source_ancestors) > 0
    fields_dict["internal_source_parents"] = internal_parent_layer_labels
    fields_dict["internal_source_ancestors"] = internal_source_ancestors
    fields_dict["is_internal_sink"] = False
    fields_dict["is_terminal_bool"] = False
    fields_dict["is_terminal_conditional_bool"] = False
    fields_dict["conditional_context_kind"] = None
    fields_dict["conditional_wrapper_kind"] = None
    fields_dict["terminal_conditional_id"] = None
    fields_dict["in_conditionals"] = []
    fields_dict["terminal_bool_for"] = None
    fields_dict["is_in_conditional_body"] = False
    fields_dict["conditional_branch_stack"] = []
    fields_dict["conditional_branch_depth"] = 0
    fields_dict["conditional_entry_children"] = []
    fields_dict["conditional_then_children"] = []
    fields_dict["conditional_elif_children"] = {}
    fields_dict["conditional_else_children"] = []
    fields_dict["conditional_arm_children"] = {}

    in_multi_output = any(issubclass(type(out_orig), cls) for cls in [list, tuple, dict, set])
    fields_dict["in_multi_output"] = in_multi_output


def _extract_arg_tensors_and_params(
    normalized_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[list[torch.Tensor], list[torch.nn.Parameter]]:
    """O(1) tensor/param extraction via lookup table, with BFS fallback."""
    spec = FUNC_ARG_SPECS.get(normalized_name) or _st._dynamic_arg_specs.get(normalized_name)
    if spec is not None:
        return extract_tensors_and_params(spec, args, kwargs)  # type: ignore[arg-type]

    # Tier 3 fallback: BFS crawl once, then cache for subsequent calls.
    all_args = list(args) + list(kwargs.values())
    arg_tensors, arg_parameters = _get_tensors_and_params_from_obj(all_args)
    _cache_dynamic_spec(normalized_name, args, kwargs, arg_tensors, arg_parameters)
    return arg_tensors, arg_parameters


def _build_param_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    arg_parameters: list[torch.nn.Parameter],
) -> dict[str, int]:
    """Populate parameter-involvement fields. Returns parent_param_ops dict."""
    parent_param_ops = _process_parent_param_ops(arg_parameters)
    indiv_param_barcodes = list(parent_param_ops.keys())

    _param_logs = []
    for param in arg_parameters:
        param_meta = get_param_meta(param)
        addr = None if param_meta is None else param_meta.param_address
        if addr is not None and addr in self.param_logs:
            _param_logs.append(self.param_logs[addr])

    fields_dict["parent_params"] = arg_parameters
    fields_dict["_param_barcodes"] = indiv_param_barcodes
    fields_dict["parent_param_ops"] = parent_param_ops
    fields_dict["_param_logs"] = _param_logs
    fields_dict["param_shapes"] = [tuple(param.shape) for param in arg_parameters]
    fields_dict["num_params"] = sum(prod(shape) for shape in fields_dict["param_shapes"])
    fields_dict["num_params_trainable"] = sum(
        pl.num_params for pl in _param_logs if pl.is_trainable
    )
    fields_dict["num_params_frozen"] = sum(
        pl.num_params for pl in _param_logs if not pl.is_trainable
    )
    with pause_logging():
        fields_dict["param_memory"] = sum(p.nelement() * p.element_size() for p in arg_parameters)
    return parent_param_ops


def _build_module_context_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    arg_tensors: list[torch.Tensor],
    parent_layer_entries: list[Op],
) -> None:
    """Populate module nesting, address, and input/output status fields."""
    modules = _snapshot_exhaustive_module_stack(self)
    module = modules[-1] if modules else None

    fields_dict["module"] = module
    fields_dict["modules"] = modules
    fields_dict["module_call_stack"] = []
    fields_dict["module_entry_arg_keys"] = defaultdict(list)
    fields_dict["input_to_module_calls"] = []
    fields_dict["output_of_modules"] = []
    fields_dict["output_of_module_calls"] = []
    fields_dict["is_module_output"] = False
    fields_dict["is_atomic_module"] = False
    fields_dict["atomic_module_call"] = None


def _build_shared_fields_dict(
    self: "Trace",
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    func_call_id: int,
) -> tuple[dict[str, Any], list[Op], list[torch.Tensor], dict[str, int]]:
    """Build the fields_dict shared by all output tensors of a single function call.

    When a function produces multiple output tensors (e.g. ``torch.split``),
    many metadata fields are identical across outputs (function info, parent
    relationships, module context).  This function computes those shared fields
    once; per-tensor fields (shape, label, equivalence type) are added later
    in ``_log_output_tensor_info``.

    Returns:
        (fields_dict, parent_layer_entries, arg_tensors, parent_param_ops)
    """
    # Canonical layer_type: lowercase with underscores stripped (e.g. "conv2d").
    layer_type = func_name.lower().replace("_", "")

    # O(1) tensor/param extraction via lookup table (replaces BFS crawl)
    arg_tensors, arg_parameters = _extract_arg_tensors_and_params(layer_type, args, kwargs)

    # Separate tensor args (which define graph edges) from non-tensor args
    # (which become metadata and feed into equivalence_class hashing).
    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {key: val for key, val in kwargs.items() if not _check_if_tensor_arg(val)}
    parent_layer_labels = get_label_list(arg_tensors)
    parent_layer_entries = [
        cast(Op, LiveOpView(self, self.capture_events.live_index.require_event(label)))
        for label in parent_layer_labels
    ]

    fields_dict: dict[str, Any] = {}

    # General info
    fields_dict["type"] = layer_type
    fields_dict["detach_saved_activations"] = self.detach_saved_activations
    fields_dict["output_device"] = self.output_device
    fields_dict["_construction_done"] = False
    fields_dict["interventions"] = []
    should_capture_template = bool(
        getattr(self, "intervention_ready", False) or getattr(self, "save_arg_templates", False)
    )
    if should_capture_template:
        captured_template = _build_args_template(func, args, kwargs)
        fields_dict["args_template"] = captured_template
        fields_dict["kwargs_template"] = captured_template if kwargs else None
    else:
        fields_dict["args_template"] = None
        fields_dict["kwargs_template"] = None
    fields_dict["container_path"] = ()
    fields_dict["container_spec"] = None
    fields_dict["multi_output_name"] = None

    # Grad info
    fields_dict["grad"] = None
    fields_dict["transformed_grad"] = None
    fields_dict["save_grads"] = getattr(self, "save_grads", None) not in (None, False)
    fields_dict["has_grad"] = False
    fields_dict["grad_shape"] = None
    fields_dict["transformed_grad_shape"] = None
    fields_dict["grad_dtype"] = None
    fields_dict["transformed_grad_dtype"] = None
    fields_dict["gradient_memory"] = 0
    fields_dict["transformed_gradient_memory"] = None

    # Function call info
    fields_dict["func"] = func
    fields_dict["func_call_id"] = func_call_id
    fields_dict["func_name"] = func_name
    fields_dict["func_qualname"] = getattr(func, "__qualname__", None)
    code_context_cache = getattr(self, "_code_context_cache", None)
    if code_context_cache is None:
        code_context_cache = {}
        self._code_context_cache = code_context_cache
    fields_dict["code_context"] = _get_code_context(
        self.num_context_lines,
        source_loading_enabled=self.save_code_context,
        disable_col_offset=False,
        context_cache=code_context_cache,
    )
    fields_dict["func_duration"] = exec_ctx.time_elapsed
    fields_dict["func_rng_states"] = exec_ctx.rng_states
    fields_dict["func_autocast_state"] = exec_ctx.autocast_state
    fields_dict["arg_names"] = _st._arg_names.get(func_name.strip("_"), ())
    fields_dict["num_args_total"] = len(args) + len(kwargs)
    fields_dict["num_pos_args"] = len(args)
    fields_dict["num_kwargs"] = len(kwargs)
    fields_dict["non_tensor_pos_args"] = non_tensor_args
    fields_dict["non_tensor_kwargs"] = non_tensor_kwargs
    fields_dict["func_non_tensor_args"] = non_tensor_args + list(non_tensor_kwargs.values())

    _build_graph_relationship_fields(
        self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
    )
    parent_param_ops = _build_param_fields(self, fields_dict, arg_parameters)
    _build_module_context_fields(self, fields_dict, arg_tensors, parent_layer_entries)
    is_transform = bool(getattr(func, "__tl_is_transform_boundary__", False)) or (
        func_name in TRANSFORM_FUNC_NAMES
    )
    fields_dict["is_transform"] = is_transform
    fields_dict["transform_kind"] = getattr(func, "__tl_transform_kind__", None) or (
        func_name if is_transform else None
    )
    fields_dict["transform_chain"] = tuple(getattr(func, "__tl_transform_tags__", ()))
    fields_dict["transform_config"] = dict(getattr(func, "__tl_transform_config__", {}))
    fields_dict["transform_fn_name"] = getattr(func, "__tl_transform_fn_name__", None)
    fields_dict["transform_fn_qualname"] = getattr(func, "__tl_transform_fn_qualname__", None)
    fields_dict["transform_fn_source"] = getattr(func, "__tl_transform_fn_source__", None)
    fields_dict["unattributed_tensor_args"] = _unattributed_tensor_arg_positions(args, kwargs)

    # Function config — lightweight hyperparameter extraction, always on.
    param_shapes = cast(list[tuple[int, ...]] | None, fields_dict.get("param_shapes"))
    fields_dict["func_config"] = extract_salient_args(
        layer_type,
        func_name,
        args,
        kwargs,
        param_shapes,
    )

    return fields_dict, parent_layer_entries, arg_tensors, parent_param_ops


def _copy_shared_fields_for_output(fields_dict: dict[str, Any]) -> dict[str, Any]:
    """Return an isolated per-output copy of shared exhaustive fields.

    Parameters
    ----------
    fields_dict
        Shared field mapping for all tensor outputs of one wrapped function call.

    Returns
    -------
    dict[str, Any]
        Field mapping that can be mutated for one output tensor without changing
        sibling output entries.
    """

    fields_dict_onetensor = fields_dict.copy()
    for field in _SHARED_FIELDS_TO_SHALLOW_COPY_PER_OUTPUT:
        fields_dict_onetensor[field] = copy.copy(fields_dict[field])
    for field in _SHARED_FIELDS_TO_DEEP_COPY_PER_OUTPUT:
        fields_dict_onetensor[field] = copy.deepcopy(fields_dict[field])
    return fields_dict_onetensor


def _classify_new_tensor_in_trace(
    self: "Trace",
    fields_dict: dict[str, Any],
    new_tensor_label: str,
) -> None:
    """Update Trace categories for a new tensor.

    Args:
        self: Trace object being populated.
        fields_dict: Shared field values for this wrapped output.
        new_tensor_label: Raw label for the new tensor op.
    """
    if fields_dict["is_internal_source"]:
        self.internal_source_ops.append(new_tensor_label)


def _tag_tensor_and_track_variations(
    self: "Trace",
    out: torch.Tensor,
    new_layer_entry: Op,
    fields_dict_onetensor: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
) -> None:
    """Tag the output tensor with its label, add backward hook, and track parent content variations.

    Parent content variation tracking detects in-place mutations: if a parent
    tensor's value at function-call time (from arg_copies) differs from its
    saved out, the pre-mutation value is recorded in
    ``out_versions_by_child``.  This is critical for validation replay,
    which needs the actual input values each child operation saw.
    """
    out_label = fields_dict_onetensor["_label_raw"]
    set_tensor_label(out, out_label)
    _add_tensor_backward_hook(self, out, out_label)

    child_event = self.capture_events.live_index.require_event(new_layer_entry._label_raw)
    for parent_label in new_layer_entry.parents:
        parent_event = self.capture_events.live_index.require_event(parent_label)
        contract = child_event.backend_semantics
        parent_tensor_contents = _get_parent_output_version_snapshot(
            self,
            parent_label,
            new_layer_entry.parent_arg_positions,
            contract.mutated_input_positions,
            contract.aliased_output_inputs,
            parent_event.output.has_saved_activation,
            parent_event.output.tensor.payload,
            arg_copies,
            kwarg_copies,
        )
        if parent_tensor_contents is not None:
            self.capture_events.append_output_version(
                OutputVersionEvent(
                    parent_raw_label=parent_label,
                    child_raw_label=new_layer_entry._label_raw,
                    child_output_path=tuple(fields_dict_onetensor["container_path"]),
                    payload=parent_tensor_contents,
                    transform_state=fields_dict_onetensor["activation_transform"],
                    detach_grad_policy=fields_dict_onetensor["detach_saved_activations"],
                )
            )


def _get_parent_output_version_snapshot(
    self: "Trace",
    parent_label: str,
    parent_arg_positions: dict[str, dict[Any, str]],
    mutated_input_positions: tuple[object, ...],
    aliased_output_inputs: tuple[object, ...],
    parent_has_saved_activation: bool,
    parent_saved_output: Any,
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
) -> Any | None:
    """Return the pre-call parent snapshot needed for child-version replay.

    Parameters
    ----------
    self
        Active trace.
    parent_label
        Parent label in the same label space as ``parent_arg_positions``.
    parent_arg_positions
        Mapping from argument positions to parent labels.
    mutated_input_positions
        Input positions the backend contract says may be mutated.
    aliased_output_inputs
        Input positions the backend contract says may alias the output.
    parent_has_saved_activation
        Whether the parent currently has a saved activation payload.
    parent_saved_output
        Parent's currently saved activation payload, if any.
    arg_copies
        Pre-call positional argument copies.
    kwarg_copies
        Pre-call keyword argument copies.

    Returns
    -------
    Any | None
        Pre-call parent value when replay needs a child-specific version;
        otherwise ``None``.
    """

    contract_positions = tuple(mutated_input_positions) + tuple(aliased_output_inputs)
    should_snapshot_by_contract = parent_label_has_alias_contract(
        parent_label,
        parent_arg_positions,
        contract_positions,
    )
    should_snapshot_by_value = parent_has_saved_activation
    if not (self.save_arg_values and (should_snapshot_by_contract or should_snapshot_by_value)):
        return None

    parent_tensor_contents = get_parent_contents_for_contract_position(
        parent_label,
        arg_copies,
        kwarg_copies,
        parent_arg_positions,
    )
    if should_snapshot_by_contract or not tensor_nanequal(
        parent_tensor_contents,
        parent_saved_output,
    ):
        return parent_tensor_contents
    return None


def _track_fast_parent_output_versions(
    self: "Trace",
    orig_layer_entry: Op,
    backend_semantics: BackendSemantics,
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
) -> None:
    """Record child-version snapshots while refreshing an existing fast trace.

    Parameters
    ----------
    self
        Trace being refreshed in fast mode.
    orig_layer_entry
        Existing child operation matched by counter alignment.
    backend_semantics
        Backend alias/mutation contract detected for the current call.
    arg_copies
        Pre-call positional argument copies.
    kwarg_copies
        Pre-call keyword argument copies.

    Returns
    -------
    None
        Mutates parent ``out_versions_by_child`` fields.
    """

    for parent_label in orig_layer_entry.parents:
        parent_layer = self[parent_label]
        parent_tensor_contents = _get_parent_output_version_snapshot(
            self,
            parent_label,
            orig_layer_entry.parent_arg_positions,
            backend_semantics.mutated_input_positions,
            backend_semantics.aliased_output_inputs,
            parent_layer.has_saved_activation,
            parent_layer.out if parent_layer.has_saved_activation else None,
            arg_copies,
            kwarg_copies,
        )
        if parent_tensor_contents is None:
            continue
        parent_layer.out_versions_by_child[orig_layer_entry.layer_label] = parent_tensor_contents
        parent_layer.has_out_variations = True


def log_function_output_tensors_exhaustive(
    self: "Trace",
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> None:
    """Full metadata logging for each output tensor of a function call.

    For each loggable output tensor:
      1. Build per-tensor fields (label, shape, equivalence type, FLOPs).
      2. Create a Op entry and optionally save out data.
      3. Update bidirectional family links (parent→child, sibling, spouse).
      4. Tag the output tensor with ``_tl.label_raw`` so downstream
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
    with _timed_phase(self, "ctx_build:shared_fields"):
        (
            fields_dict,
            parent_layer_entries,
            arg_tensors,
            parent_param_ops,
        ) = _build_shared_fields_dict(
            self,
            func,
            func_name,
            args,
            kwargs,
            out_orig,
            exec_ctx,
            func_call_id,
        )

    output_entries = _partition_output_entries_with_autograd_stats(out_orig)
    _register_call_output_container_snapshot(
        self,
        out_orig,
        output_entries=output_entries,
        func_call_id=func_call_id,
        event_index=int(fields_dict.get("raw_index") or func_call_id),
    )
    expected_output_count = len(output_entries)
    loggable_output_count = sum(
        1
        for output_entry in output_entries
        if _output_should_be_logged(output_entry.value, is_bottom_level_func)
    )
    use_single_output_fields = (
        loggable_output_count == 1
        and len(output_entries) == 1
        and output_entries[0].container_spec is None
    )
    shared_func_event_input = FunctionEventInput(
        func=func,
        func_name=func_name,
        func_qualname=getattr(func, "__qualname__", None),
        args=args,
        kwargs=kwargs,
        raw_output=out_orig,
        arg_copies=arg_copies,
        kwarg_copies=kwarg_copies,
        module_stack=tuple(_module_frames_from_fields(fields_dict)),
        is_bottom_level_func=is_bottom_level_func,
        func_call_id=func_call_id,
        expected_output_count=expected_output_count,
    )

    for i, output_entry in enumerate(output_entries):
        out = output_entry.value
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue
        out_tensor = cast(torch.Tensor, out)

        fields_dict_onetensor = (
            fields_dict if use_single_output_fields else _copy_shared_fields_for_output(fields_dict)
        )
        fields_dict_onetensor["container_path"] = output_entry.container_path
        fields_dict_onetensor["container_spec"] = output_entry.container_spec
        if output_entry.container_spec is not None:
            fields_dict_onetensor["in_multi_output"] = True
        _log_output_tensor_info(
            self,
            out_tensor,
            i,
            args,
            kwargs,
            parent_param_ops,
            fields_dict_onetensor,
            output_entry.autograd_stats,
        )
        detect_backend_semantics = (
            detect_torch_alias_contract
            if _should_keep_alias_mutation_contract(self)
            else detect_torch_output_alias_contract
        )
        fields_dict_onetensor["backend_semantics"] = detect_backend_semantics(
            shared_func_event_input,
            backend_grad_handle=fields_dict_onetensor["grad_fn_handle"],
            grad_fn_class_name=fields_dict_onetensor["grad_fn_class_name"],
            autograd_memory=fields_dict_onetensor["autograd_memory"],
            num_autograd_tensors=fields_dict_onetensor["num_autograd_tensors"],
            bytes_delta_at_call=fields_dict_onetensor["bytes_delta_at_call"],
            bytes_peak_at_call=fields_dict_onetensor["bytes_peak_at_call"],
        )
        fire_results = _pop_tensor_live_fire_results(out_tensor)
        if fire_results:
            fields_dict_onetensor["fire_results"] = fire_results
            fields_dict_onetensor["interventions"] = [
                result.fire_record for result in fire_results if result.fire_record is not None
            ]
            fields_dict_onetensor["intervention_replaced"] = any(
                result.replaced for result in fire_results
            )
        if getattr(self, "intervention_ready", False):
            fields_dict_onetensor["_edge_uses"] = _build_edge_use_records(
                self,
                fields_dict_onetensor["parent_arg_positions"],
                fields_dict_onetensor["_label_raw"],
                func_call_id,
            )
        new_layer_entry = cast(
            Op,
            _make_layer_log_entry(
                self,
                out_tensor,
                fields_dict=fields_dict_onetensor,
                t_args=arg_copies,
                t_kwargs=kwarg_copies,
                activation_transform=self.activation_transform,
            ),
        )
        new_tensor_label = new_layer_entry._label_raw

        _classify_new_tensor_in_trace(self, fields_dict, new_tensor_label)
        _tag_tensor_and_track_variations(
            self,
            out_tensor,
            new_layer_entry,
            fields_dict_onetensor,
            arg_copies,
            kwarg_copies,
        )
        options = getattr(self, "_predicate_save_options", None)
        if options is not None and options.halt is not None:
            halt_ctx = _build_trace_predicate_context(self, fields_dict_onetensor, out_tensor)
            _evaluate_halt(halt_ctx, options, frontier_output=out_tensor)


def _get_parent_contents(
    parent_label: str,
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    parent_arg_positions: dict[str, dict[Any, str]],
) -> Any:
    """Retrieve a parent tensor's pre-call value from the saved argument copies.

    Used for child tensor variation tracking: if a parent's value in arg_copies
    differs from its currently saved out, the parent was mutated
    in-place between operations, and the variation is recorded.
    """
    for pos, label in parent_arg_positions["args"].items():
        if label == parent_label:
            return index_nested(arg_copies, pos)
    for argname, label in parent_arg_positions["kwargs"].items():
        if label == parent_label:
            return index_nested(kwarg_copies, argname)
    raise ValueError("Parent layer not found in function arguments.")


def log_function_output_tensors_fast(
    self: "Trace",
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
    func_call_id: int,
) -> None:
    """Fast-path logging: save new out values into existing graph entries.

    Skips all metadata collection.  Instead:
      1. Increment counters identically to the exhaustive pass (counter alignment).
      2. Reconstruct the raw label from counters and verify it maps to the same
         final label as the exhaustive pass (graph-change detection).
      3. Save out data and update shape/dtype/timing metadata.

    If any counter, label, or parent mismatch is detected, raises ValueError
    telling the user to re-run ``trace``.
    """
    # Minimal info collection — only what's needed for counter alignment and saving.
    layer_type = func_name.lower().replace("_", "")
    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {key: val for key, val in kwargs.items() if not _check_if_tensor_arg(val)}

    # O(1) tensor extraction via lookup table (replaces BFS crawl)
    arg_tensors, _ = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    out_iter = ensure_iterable(out_orig)
    expected_output_count = len(out_iter)
    func_event_input = FunctionEventInput(
        func=func,
        func_name=func_name,
        func_qualname=getattr(func, "__qualname__", None),
        args=args,
        kwargs=kwargs,
        raw_output=out_orig,
        arg_copies=arg_copies,
        kwarg_copies=kwarg_copies,
        module_stack=(),
        is_bottom_level_func=is_bottom_level_func,
        func_call_id=func_call_id,
        expected_output_count=expected_output_count,
    )

    for i, out in enumerate(out_iter):
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue
        # Mirror the exhaustive pass's counter increments exactly.
        # The raw label reconstructed here MUST match the exhaustive pass's label.
        self._layer_counter += 1
        self._raw_layer_type_counter[layer_type] += 1
        raw_index = self._layer_counter
        type_index = self._raw_layer_type_counter[layer_type]
        _label_raw = f"{layer_type}_{type_index}_{raw_index}_raw"
        # Skip orphans — these were pruned from the graph during postprocessing.
        if _label_raw in self._orphan_labels:
            continue
        # Map parent raw labels → final labels for graph-change verification.
        parent_layer_labels_raw = get_label_list(arg_tensors)
        parent_layer_labels_orig = []
        for raw_label in parent_layer_labels_raw:
            if raw_label in self._raw_to_final_layer_labels:
                parent_layer_labels_orig.append(self._raw_to_final_layer_labels[raw_label])
            elif raw_label not in self._orphan_labels:
                raise ValueError(
                    f"Fast-path parent {raw_label} not found in raw→final label map "
                    f"and not in _orphan_labels. The computational graph may have changed."
                )
        # Tag tensor so downstream ops can find this tensor's label.
        set_tensor_label(out, _label_raw)
        if _label_raw not in self._raw_to_final_layer_labels:
            raise ValueError(
                "The computational graph changed for this forward pass compared to the original "
                "call to trace (either due to different inputs or a different "
                "random seed), so save_new_outs failed. Please re-run "
                "trace with the desired inputs."
            )
        orig_tensor_label = self._raw_to_final_layer_labels[_label_raw]
        orig_layer_entry = self[orig_tensor_label]
        previous_shape = orig_layer_entry.shape

        _add_tensor_backward_hook(self, out, _label_raw)  # Must pass RAW label (#86)
        grad_fn_cls = type(out.grad_fn) if out.grad_fn is not None else None
        orig_layer_entry.grad_fn_class_name = None if grad_fn_cls is None else grad_fn_cls.__name__
        orig_layer_entry.grad_fn_class_qualname = (
            None if grad_fn_cls is None else f"{grad_fn_cls.__module__}.{grad_fn_cls.__qualname__}"
        )
        orig_layer_entry.grad_fn_object_id = id(out.grad_fn) if out.grad_fn is not None else None
        orig_layer_entry.grad_fn_handle = out.grad_fn

        # Structural integrity check: verify counter, type, label, and parents
        # all match the exhaustive pass.  Any mismatch means dynamic control flow
        # changed the graph and the fast pass cannot proceed.
        if (
            orig_layer_entry.raw_index != self._layer_counter
            or orig_layer_entry.layer_type != layer_type
            or orig_layer_entry._label_raw != _label_raw
            or set(orig_layer_entry.parents) != set(parent_layer_labels_orig)
        ):
            raise ValueError(
                "The computational graph changed for this forward pass compared to the original "
                "call to trace (either due to different inputs or a different "
                "random seed), so save_new_outs failed. Please re-run "
                "trace with the desired inputs."
            )
        detect_backend_semantics = (
            detect_torch_alias_contract
            if _should_keep_alias_mutation_contract(self)
            else detect_torch_output_alias_contract
        )
        backend_semantics = detect_backend_semantics(
            func_event_input,
            backend_grad_handle=orig_layer_entry.grad_fn_handle,
            grad_fn_class_name=orig_layer_entry.grad_fn_class_name,
            autograd_memory=orig_layer_entry.autograd_memory,
            num_autograd_tensors=orig_layer_entry.num_autograd_tensors,
            bytes_delta_at_call=orig_layer_entry.bytes_delta_at_call,
            bytes_peak_at_call=orig_layer_entry.bytes_peak_at_call,
        )
        _track_fast_parent_output_versions(
            self,
            orig_layer_entry,
            backend_semantics,
            arg_copies,
            kwarg_copies,
        )

        # Save out data if this layer is in the save list.
        layer_nums_to_save = cast(Any, self._layer_nums_to_save)
        if (layer_nums_to_save == "all") or (orig_layer_entry.raw_index in layer_nums_to_save):
            orig_layer_entry.save_activation(
                out,
                arg_copies,
                kwarg_copies,
                self.save_arg_values,
                self.activation_transform,
            )
            # Output layers are identity wrappers whose out come
            # from their parent.  Propagate the parent's saved out to
            # any child that is an output layer so postprocess_fast can find it.
            for child_layer in orig_layer_entry.children:
                if child_layer in self.output_layers:
                    child_output = self[child_layer]
                    if (
                        orig_layer_entry.has_out_variations
                        and child_layer in orig_layer_entry.out_versions_by_child
                    ):
                        # out_versions_by_child already has transform applied.
                        tensor_to_save = orig_layer_entry.out_versions_by_child[child_layer]
                        child_output._internal_set("out", safe_copy(tensor_to_save))
                    else:
                        child_output._internal_set("out", safe_copy(out))
                        if self.activation_transform is not None:
                            # pause_logging prevents activation_transform from
                            # triggering decorated torch ops that would be logged.
                            with pause_logging():
                                child_output._internal_set(
                                    "transformed_out",
                                    self.activation_transform(child_output.out),
                                )
                            if not getattr(self, "save_raw_activations", True):
                                child_output._internal_set("out", None)
                    child_output.has_saved_activation = True
                    if self.save_arg_values:
                        child_output.has_saved_args = True
                        child_output._internal_set("saved_args", [safe_copy(child_output.out)])
                        child_output._internal_set("saved_kwargs", {})
                    if child_output.out is not None:
                        child_output.activation_memory = Bytes(
                            get_memory_amount_from_metadata(
                                child_output.out,
                                tuple(child_output.out.shape),
                                child_output.out.dtype,
                            )
                        )

        # Update lightweight metadata that may vary across inputs
        # (shape can differ for dynamic-shape models that still share graph structure).
        new_shape = tuple(out.shape)
        if previous_shape is not None and new_shape != previous_shape:
            import warnings

            warnings.warn(
                f"Tensor shape changed for '{orig_tensor_label}': "
                f"expected {previous_shape}, got {new_shape}. "
                f"The computational graph may have changed between ops."
            )
        orig_layer_entry.shape = new_shape
        orig_layer_entry.dtype = out.dtype
        orig_layer_entry.activation_memory = Bytes(
            get_memory_amount_from_metadata(out, new_shape, out.dtype)
        )
        (
            orig_layer_entry.autograd_memory,
            orig_layer_entry.num_autograd_tensors,
        ) = _get_autograd_saved_stats_for_tensor(out)
        orig_layer_entry.bytes_delta_at_call = 0
        orig_layer_entry.bytes_peak_at_call = 0
        orig_layer_entry.func_duration = exec_ctx.time_elapsed
        orig_layer_entry.func_rng_states = exec_ctx.rng_states
        orig_layer_entry.func_autocast_state = exec_ctx.autocast_state
        orig_layer_entry.non_tensor_pos_args = non_tensor_args
        orig_layer_entry.non_tensor_kwargs = non_tensor_kwargs

        # Update func_config — some may be input-dependent (e.g. interpolate size).
        orig_layer_entry.func_config = extract_salient_args(
            layer_type,
            func_name,
            args,
            kwargs,
            orig_layer_entry.param_shapes,
        )


def _output_should_be_logged(out: Any, is_bottom_level_func: bool) -> bool:
    """Determine whether an output value should be logged as a new graph node.

    Two conditions must hold:
      1. ``out`` must be a torch.Tensor (non-tensor outputs like ints are skipped).
      2. Either the tensor is genuinely new (no ``_tl.label_raw`` value),
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

    if (get_tensor_label(out) is None) or is_bottom_level_func:
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


def _iter_autograd_saved_candidates(grad_fn_handle: Any) -> list[Any]:
    """Return accessible autograd-saved values from a grad_fn_handle object.

    Parameters
    ----------
    grad_fn_handle
        PyTorch autograd function object to inspect.

    Returns
    -------
    list
        Values exposed through ``saved_tensors`` and ``_saved_*`` attributes.
        Attribute access failures are ignored because PyTorch may release or
        guard some saved values.
    """
    saved_values: list[Any] = []
    try:
        saved_values.extend(getattr(grad_fn_handle, "saved_tensors", ()))
    except Exception:
        pass

    for attr_name in grad_fn_handle.__class__.__dict__:
        if not attr_name.startswith(_AUTOGRAD_SAVED_ATTR_PREFIX):
            continue
        try:
            saved_values.append(getattr(grad_fn_handle, attr_name))
        except Exception:
            continue
    return saved_values


def _collect_tensor_values(value: Any) -> list[torch.Tensor]:
    """Collect tensor values from a shallow autograd-saved object.

    Parameters
    ----------
    value
        Value read from a grad_fn_handle saved-tensor API.

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
    if is_functorch_wrapped_tensor(tensor):
        return 0, 0
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
) -> dict[int, tuple[int | None, int | None]]:
    """Measure autograd-saved tensor bytes/counts for each tensor output.

    Parameters
    ----------
    output
        Raw function output from a decorated torch operation.

    Returns
    -------
    dict
        Mapping from output index to ``(autograd_memory,
        num_autograd_tensors)``. Non-tensor outputs and tensors without
        ``grad_fn_handle`` are omitted and handled by callers as ``None`` values.
    """
    stats_by_index: dict[int, tuple[int | None, int | None]] = {}
    seen_grad_fns: set[int] = set()
    seen_data_ptrs: set[int] = set()

    for output_index, maybe_tensor in enumerate(ensure_iterable(output)):
        if not isinstance(maybe_tensor, torch.Tensor) or maybe_tensor.grad_fn is None:
            continue

        grad_fn_handle = maybe_tensor.grad_fn
        grad_fn_object_id = id(grad_fn_handle)
        if grad_fn_object_id in seen_grad_fns:
            stats_by_index[output_index] = (0, 0)
            continue
        seen_grad_fns.add(grad_fn_object_id)

        total_bytes = 0
        tensor_count = 0
        for saved_value in _iter_autograd_saved_candidates(grad_fn_handle):
            for saved_tensor in _collect_tensor_values(saved_value):
                bytes_added, count_added = _add_autograd_saved_tensor(saved_tensor, seen_data_ptrs)
                total_bytes += bytes_added
                tensor_count += count_added
        stats_by_index[output_index] = (total_bytes, tensor_count)

    return stats_by_index


def _partition_output_entries_with_autograd_stats(output: Any) -> list[_OutputTensorEntry]:
    """Collect output entries and autograd stats in one pass.

    Parameters
    ----------
    output
        Raw function output from a decorated torch operation.

    Returns
    -------
    list[_OutputTensorEntry]
        Output entries in logging order, each paired with its autograd saved
        tensor byte/count stats when available.
    """

    raw_entries: list[tuple[Any, tuple[OutputPathComponent, ...], ContainerSpec | None]] = list(
        _walk_output_tensors_with_paths(output)
    )
    if not raw_entries:
        raw_entries = [(out, (), None) for out in ensure_iterable(output)]

    partitioned_entries: list[_OutputTensorEntry] = []
    seen_grad_fns: set[int] = set()
    seen_data_ptrs: set[int] = set()
    for maybe_tensor, container_path, container_spec in raw_entries:
        autograd_stats: tuple[int | None, int | None] = (None, None)
        if isinstance(maybe_tensor, torch.Tensor) and maybe_tensor.grad_fn is not None:
            grad_fn_handle = maybe_tensor.grad_fn
            grad_fn_object_id = id(grad_fn_handle)
            if grad_fn_object_id in seen_grad_fns:
                autograd_stats = (0, 0)
            else:
                seen_grad_fns.add(grad_fn_object_id)
                total_bytes = 0
                tensor_count = 0
                for saved_value in _iter_autograd_saved_candidates(grad_fn_handle):
                    for saved_tensor in _collect_tensor_values(saved_value):
                        bytes_added, count_added = _add_autograd_saved_tensor(
                            saved_tensor, seen_data_ptrs
                        )
                        total_bytes += bytes_added
                        tensor_count += count_added
                autograd_stats = (total_bytes, tensor_count)
        partitioned_entries.append(
            _OutputTensorEntry(
                value=maybe_tensor,
                container_path=container_path,
                container_spec=container_spec,
                autograd_stats=autograd_stats,
            )
        )

    return partitioned_entries


def _register_call_output_container_snapshot(
    trace: "Trace",
    output: Any,
    *,
    output_entries: list[_OutputTensorEntry],
    func_call_id: int,
    event_index: int,
) -> None:
    """Register a function-call output container snapshot when opt-in capture is active.

    Parameters
    ----------
    trace
        Active trace.
    output
        Raw function output object.
    output_entries
        Path-aware tensor output entries from the existing output walker.
    func_call_id
        Function call identifier for the output boundary.
    event_index
        Capture event index used for ordering.
    """

    if not (
        getattr(trace, "intervention_ready", False)
        or getattr(trace, "_capture_output_structure", False)
    ):
        return
    spec = next((entry.container_spec for entry in output_entries if entry.container_spec), None)
    if spec is None:
        return
    registry = trace._ensure_build_state().container_registry
    registry.register_snapshot(
        output,
        site=FuncSite(func_call_id=func_call_id, position="return"),
        role=Role.CALL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=event_index,
        spec=spec,
        leaf_occurrences=_container_leaf_occurrences_from_entries(output_entries),
        reconstructable=True,
    )


def _container_leaf_occurrences_from_entries(
    output_entries: list[_OutputTensorEntry],
) -> tuple[ContainerLeafOccurrence, ...]:
    """Build ordered leaf occurrences from path-aware output entries.

    Parameters
    ----------
    output_entries
        Path-aware tensor output entries.

    Returns
    -------
    tuple[ContainerLeafOccurrence, ...]
        Occurrence records preserving repeated tensors at multiple paths.
    """

    occurrences: list[ContainerLeafOccurrence] = []
    for occ_index, entry in enumerate(output_entries):
        producer_label = (
            get_tensor_label(entry.value) if isinstance(entry.value, torch.Tensor) else None
        )
        occurrences.append(
            ContainerLeafOccurrence(
                path=entry.container_path,
                producer_op_label=producer_label,
                tensor_identity=producer_label,
                occ_index=occ_index,
            )
        )
    return tuple(occurrences)


def _get_autograd_saved_stats_for_tensor(
    tensor: torch.Tensor,
) -> tuple[int | None, int | None]:
    """Measure autograd-saved tensor bytes/counts for a single tensor output.

    Parameters
    ----------
    tensor
        Tensor output from a decorated torch operation.

    Returns
    -------
    tuple
        ``(autograd_memory, num_autograd_tensors)``. Both values
        are ``None`` when no grad_fn_handle exists.
    """
    if tensor.grad_fn is None:
        return None, None
    stats_by_index = _get_autograd_saved_stats_by_output(tensor)
    return stats_by_index.get(0, (None, None))


def _log_output_tensor_info(
    self: "Trace",
    t: torch.Tensor,
    i: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    parent_param_ops: dict[str, int],
    fields_dict: dict[str, Any],
    autograd_saved_stats: tuple[int | None, int | None],
) -> None:
    """Populate per-tensor fields that differ across outputs of a single function call.

    This includes:
      - Counter-based label generation (``_label_raw``).
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
        parent_param_ops: Dict mapping param barcodes to their current pass number.
        fields_dict: Per-tensor fields dict to populate (mutated in place).
    """
    layer_type = fields_dict["type"]
    indiv_param_barcodes = list(parent_param_ops.keys())
    self._layer_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    raw_index = self._layer_counter
    type_index = self._raw_layer_type_counter[layer_type]
    _label_raw = f"{layer_type}_{type_index}_{raw_index}_raw"

    # Determine operation equivalence type — the fingerprint used by loop detection
    # to group structurally identical operations (same layer across ops).
    if len(parent_param_ops) > 0:
        # Parameterized ops: equivalence is defined by the exact set of parameters
        # used, combined with the operation type.  E.g., two conv2d calls using the
        # same weight+bias tensors are the same layer on different ops.
        output_index = i if fields_dict["in_multi_output"] else None
        equivalence_class = _make_raw_param_group_barcode(
            indiv_param_barcodes,
            layer_type,
            output_index=output_index,
        )
        base_equivalence_class = equivalence_class
        self.layers_with_params[equivalence_class].append(_label_raw)
        fields_dict["pass_index"] = len(self.layers_with_params[equivalence_class])
        equivalence_class = _append_module_suffix_to_equivalence_class(
            equivalence_class, fields_dict["modules"]
        )
        fields_dict["equivalence_class"] = equivalence_class
    else:
        # Non-parameterized ops: equivalence is a hash of the operation type,
        # non-tensor args, output index, and containing module.  Each unique
        # non-param operation is seen only once (pass_index=1).
        logged_func_name = fields_dict["func_name"]
        is_inplace_output = logged_func_name.startswith("__i") or (
            logged_func_name.endswith("_") and not logged_func_name.startswith("__")
        )
        equivalence_layer_type = f"{layer_type}_inplace" if is_inplace_output else layer_type
        equivalence_class = _get_equivalence_class(
            args, kwargs, i, equivalence_layer_type, fields_dict
        )
        base_equivalence_class = equivalence_class
        equivalence_class = _append_module_suffix_to_equivalence_class(
            equivalence_class, fields_dict["modules"]
        )
        fields_dict["equivalence_class"] = equivalence_class
        fields_dict["pass_index"] = 1

    # equivalent_ops is a DIRECT reference to the Trace-level set —
    # all entries sharing this equivalence type point to the same set object.
    # Defensive: if equivalence_class isn't yet registered (e.g. user-injected
    # tensor through intervention/raw-hook with a fresh hash), create the
    # set on demand rather than crashing with KeyError.
    if base_equivalence_class not in self.op_equivalence_classes:
        self.op_equivalence_classes[base_equivalence_class] = set()
    self.op_equivalence_classes[base_equivalence_class].add(_label_raw)
    fields_dict["equivalent_ops"] = self.op_equivalence_classes[base_equivalence_class]

    # In-place ops return the same tensor object, which already has a raw label.
    fields_dict["is_inplace"] = get_tensor_label(t) is not None
    grad_fn_cls = type(t.grad_fn) if t.grad_fn is not None else None
    fields_dict["grad_fn_class_name"] = None if grad_fn_cls is None else grad_fn_cls.__name__
    fields_dict["grad_fn_class_qualname"] = (
        None if grad_fn_cls is None else f"{grad_fn_cls.__module__}.{grad_fn_cls.__qualname__}"
    )
    fields_dict["grad_fn_object_id"] = id(t.grad_fn) if t.grad_fn is not None else None
    # Autograd Function objects do not consistently support weak references.
    # Keep the object only until explicit backward capture has registered hooks;
    # the backward finalizer clears these strong refs to avoid pinning graphs.
    fields_dict["grad_fn_handle"] = t.grad_fn
    fields_dict["grad_fn"] = None

    if fields_dict["in_multi_output"]:
        fields_dict["multi_output_index"] = i
    else:
        fields_dict["multi_output_index"] = None

    if (t.dtype == torch.bool) and (t.dim()) == 0:
        fields_dict["is_scalar_bool"] = True
        try:
            fields_dict["bool_value"] = t.item()
        except RuntimeError:
            # .item() forbidden inside torch.vmap context
            fields_dict["bool_value"] = None
    else:
        fields_dict["is_scalar_bool"] = False
        fields_dict["bool_value"] = None

    # General info
    fields_dict["_label_raw"] = _label_raw
    fields_dict["step_index"] = None
    fields_dict["recurrent_ops"] = []
    fields_dict["raw_index"] = raw_index
    fields_dict["step_index"] = None
    fields_dict["source_trace"] = self
    fields_dict["_tracing_finished"] = False

    # Other labeling info
    fields_dict["layer_label"] = None
    fields_dict["layer_label_short"] = None
    fields_dict["label"] = None
    fields_dict["label_short"] = None
    fields_dict["layer_label"] = None
    fields_dict["layer_label_short"] = None
    fields_dict["type"] = layer_type
    fields_dict["_layer_label_raw"] = _label_raw
    fields_dict["type_index"] = type_index
    fields_dict["num_passes"] = 1
    fields_dict["lookup_keys"] = []

    # Saved tensor info
    fields_dict["out"] = None
    fields_dict["transformed_out"] = None
    fields_dict["has_saved_activation"] = False
    fields_dict["activation_transform"] = self.activation_transform
    fields_dict["annotations"] = {}
    fields_dict["intervention_replaced"] = False
    fields_dict["fire_results"] = ()
    fields_dict["has_saved_args"] = False
    fields_dict["saved_args"] = None
    fields_dict["saved_kwargs"] = None
    fields_dict["shape"] = tuple(t.shape)
    fields_dict["transformed_out_shape"] = None
    fields_dict["dtype"] = t.dtype
    fields_dict["transformed_out_dtype"] = None
    fields_dict["activation_memory"] = get_memory_amount_from_metadata(
        t,
        fields_dict["shape"],
        fields_dict["dtype"],
    )
    fields_dict["transformed_activation_memory"] = None
    fields_dict["visualizer_path"] = None
    fields_dict["bytes_delta_at_call"] = 0
    fields_dict["bytes_peak_at_call"] = 0
    (
        fields_dict["autograd_memory"],
        fields_dict["num_autograd_tensors"],
    ) = autograd_saved_stats

    # FLOPs computation
    fields_dict["flops_forward"] = compute_forward_flops(
        fields_dict.get("func_name"),  # type: ignore[arg-type]
        fields_dict["shape"],
        fields_dict.get("param_shapes", []),
        args,
        kwargs,
    )
    fields_dict["flops_backward"] = compute_backward_flops(
        fields_dict.get("func_name"),  # type: ignore[arg-type]
        fields_dict["flops_forward"],
    )

    # Child tensor variation tracking
    fields_dict["has_out_variations"] = False
    fields_dict["out_versions_by_child"] = {}

    # If internally initialized, fix this information:
    if len(fields_dict["parents"]) == 0:
        fields_dict["is_internal_source"] = True
        fields_dict["has_internal_source_ancestor"] = True
        fields_dict["internal_source_parents"] = []
        fields_dict["internal_source_ancestors"] = {_label_raw}


def _save_activation_fields(
    trace: "Trace",
    fields_dict: dict[str, Any],
    t: torch.Tensor,
    t_args: tuple[Any, ...],
    t_kwargs: dict[str, Any],
    activation_transform: Callable[..., Any] | None,
) -> None:
    """Save activation data directly into a live field dictionary.

    Parameters
    ----------
    trace
        Active trace.
    fields_dict
        Mutable live field mapping.
    t
        Output tensor to save.
    t_args
        Positional function arguments.
    t_kwargs
        Keyword function arguments.
    activation_transform
        Optional output transform.

    Returns
    -------
    None
        Mutates ``fields_dict``.
    """

    writer = getattr(trace, "_out_writer", None)
    try:
        save_mode = _effective_activation_save_mode(
            trace,
            func_name=fields_dict.get("func_name"),
            is_inplace=bool(fields_dict.get("is_inplace", False)),
        )
        raw_out = safe_copy(
            t,
            fields_dict["detach_saved_activations"],
            save_mode=save_mode,
        )
        if fields_dict["output_device"] not in [str(raw_out.device), "same"]:
            raw_out = safe_to(raw_out, fields_dict["output_device"])
        _stamp_reference_out(fields_dict["annotations"], raw_out, save_mode)

        fields_dict["shape"] = tuple(raw_out.shape)
        fields_dict["dtype"] = raw_out.dtype
        fields_dict["activation_memory"] = get_memory_amount_from_metadata(
            raw_out,
            fields_dict["shape"],
            fields_dict["dtype"],
        )

        save_raw_activations = getattr(trace, "save_raw_activations", True)
        store_raw = save_raw_activations or activation_transform is None
        if store_raw:
            raw_out = _dedup_saved_activation_out(
                trace,
                t,
                raw_out,
                fields_dict["_layer_label_raw"],
                fields_dict["annotations"],
                getattr(trace, "save_arg_values", False),
            )
        fields_dict["out"] = raw_out if store_raw else None
        fields_dict["transformed_out"] = None
        fields_dict["transformed_out_shape"] = None
        fields_dict["transformed_out_dtype"] = None
        fields_dict["transformed_activation_memory"] = None
        if activation_transform is not None:
            transformed_out = apply_transform(
                label=fields_dict.get("_layer_label_raw"),
                raw_label=fields_dict.get("_label_raw"),
                func_name=fields_dict.get("func_name"),
                tensor=raw_out,
                transform=activation_transform,
                transform_kind="activation",
                streaming_active=writer is not None,
            )
            validate_train_mode_transform_output(
                raw_tensor=raw_out,
                transformed_tensor=transformed_out,
                transform_kind="activation",
                backward_ready=fields_dict.get(
                    "backward_ready", getattr(trace, "backward_ready", False)
                ),
                label=fields_dict.get("_layer_label_raw"),
            )
            validate_streaming_transform_output(
                transformed_tensor=transformed_out,
                transform_kind="activation",
                streaming_active=writer is not None,
                label=fields_dict.get("_layer_label_raw"),
            )
            fields_dict["transformed_out"] = transformed_out
            fields_dict["transformed_out_shape"] = _shape_or_none(transformed_out)
            fields_dict["transformed_out_dtype"] = _dtype_or_none(transformed_out)
            fields_dict["transformed_activation_memory"] = _memory_or_none(transformed_out)
        fields_dict["has_saved_activation"] = True

        _stream_activation_fields(trace, fields_dict)

        out_sink = getattr(trace, "_out_sink", None)
        if out_sink is not None and isinstance(fields_dict["out"], torch.Tensor):
            out_sink(fields_dict["_label_raw"], fields_dict["out"])

        if trace.save_arg_values:
            fields_dict["has_saved_args"] = True
            fields_dict["saved_args"] = [_recursive_safe_copy(arg) for arg in t_args]
            fields_dict["saved_kwargs"] = {
                key: _recursive_safe_copy(value) for key, value in t_kwargs.items()
            }
        else:
            fields_dict["saved_args"] = None
            fields_dict["saved_kwargs"] = None
    except Exception as exc:
        if writer is not None:
            writer.abort(f"Failed while saving out for {fields_dict['_label_raw']}: {exc}")
        raise


def _stream_activation_fields(trace: "Trace", fields_dict: dict[str, Any]) -> None:
    """Write saved activation tensors during capture when streaming is active.

    Parameters
    ----------
    trace:
        Active trace whose writer receives tensor blobs.
    fields_dict:
        Mutable live field mapping for the captured operation.

    Returns
    -------
    None
        Mutates pending blob-id fields when blobs are written.
    """

    writer = getattr(trace, "_out_writer", None)
    if writer is None or not getattr(trace, "_in_exhaustive_pass", False):
        return

    label = fields_dict["_label_raw"]
    for tensor_field, pending_field, kind in (
        ("out", "_pending_blob_id", "out"),
        ("transformed_out", "_pending_transformed_out_blob_id", "transformed_out"),
    ):
        tensor = fields_dict.get(tensor_field)
        if tensor is None:
            continue
        blob_id = writer.next_blob_id()
        fields_dict[pending_field] = blob_id
        writer.write_blob(blob_id, tensor, kind=kind, label=label)


def _save_predicate_activation_fields(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    spec: CaptureSpec,
    ctx: RecordContext,
    activation_transform: Callable[..., Any] | None,
) -> None:
    """Save a ``trace(save=...)`` activation through the shared storage resolver.

    Parameters
    ----------
    trace:
        Active trace.
    fields_dict:
        Mutable operation fields for the current output.
    tensor:
        Live tensor selected by the predicate.
    spec:
        Resolved capture spec for this output.
    ctx:
        Predicate context used for transform error messages.
    activation_transform:
        Optional activation transform configured for the trace.
    """

    options = getattr(trace, "_predicate_save_options", None)
    streaming = None if options is None else options.streaming
    intent = StorageIntent(
        in_ram=streaming is None or streaming.bundle_path is None or streaming.retain_in_memory,
        on_disk=streaming is not None and streaming.bundle_path is not None,
    )
    (
        ram_payload,
        disk_payload,
        transformed_ram_payload,
        transformed_disk_payload,
    ) = _resolve_storage(
        tensor,
        spec,
        intent,
        activation_transform=activation_transform,
        save_raw_activations=getattr(trace, "save_raw_activations", True),
        ctx=ctx,
        kind="activation",
    )
    metadata_tensor = ram_payload
    if metadata_tensor is None:
        metadata_tensor = disk_payload
    if metadata_tensor is None:
        metadata_tensor = tensor
    fields_dict["shape"] = tuple(metadata_tensor.shape)
    fields_dict["dtype"] = metadata_tensor.dtype
    fields_dict["activation_memory"] = get_memory_amount_from_metadata(
        metadata_tensor,
        fields_dict["shape"],
        fields_dict["dtype"],
    )
    fields_dict["out"] = ram_payload
    fields_dict["transformed_out"] = transformed_ram_payload
    transformed_metadata = transformed_ram_payload
    if transformed_metadata is None:
        transformed_metadata = transformed_disk_payload
    fields_dict["transformed_out_shape"] = _shape_or_none(transformed_metadata)
    fields_dict["transformed_out_dtype"] = _dtype_or_none(transformed_metadata)
    fields_dict["transformed_activation_memory"] = _memory_or_none(transformed_metadata)
    fields_dict["has_saved_activation"] = True
    _stream_predicate_payloads(
        trace,
        fields_dict,
        disk_payload=disk_payload,
        transformed_disk_payload=transformed_disk_payload,
    )


def _stream_predicate_payloads(
    trace: "Trace",
    fields_dict: dict[str, Any],
    *,
    disk_payload: torch.Tensor | None,
    transformed_disk_payload: torch.Tensor | None,
) -> None:
    """Write predicate-selected disk payloads during forward capture.

    Parameters
    ----------
    trace:
        Active trace whose writer receives payload blobs.
    fields_dict:
        Mutable operation fields receiving pending blob ids.
    disk_payload:
        Detached raw payload for disk storage.
    transformed_disk_payload:
        Detached transformed payload for disk storage.
    """

    writer = getattr(trace, "_out_writer", None)
    if writer is None:
        return
    label = fields_dict["_label_raw"]
    for payload, pending_field, kind in (
        (disk_payload, "_pending_blob_id", "out"),
        (transformed_disk_payload, "_pending_transformed_out_blob_id", "transformed_out"),
    ):
        if payload is None:
            continue
        blob_id = writer.next_blob_id()
        fields_dict[pending_field] = blob_id
        writer.write_blob(blob_id, payload, kind=kind, label=label)


def _module_stack_frames_from_fields(fields_dict: dict[str, Any]) -> tuple[ModuleStackFrame, ...]:
    """Project exhaustive op fields into predicate module-stack frames.

    Parameters
    ----------
    fields_dict:
        Live exhaustive op field mapping.

    Returns
    -------
    tuple[ModuleStackFrame, ...]
        Module frames suitable for ``RecordContext.module_stack``.
    """

    frames: list[ModuleStackFrame] = []
    for module_address, module_pass in fields_dict.get("modules", ()):
        frames.append(
            ModuleStackFrame(
                address=str(module_address),
                module_type="",
                module_id=0,
                pass_index=int(module_pass),
            )
        )
    return tuple(frames)


def _append_trace_predicate_context(trace: "Trace", ctx: RecordContext) -> None:
    """Append a trace selective-save context to the bounded runtime window.

    Parameters
    ----------
    trace:
        Active trace.
    ctx:
        Context to append.

    Returns
    -------
    None
        Mutates trace-owned predicate runtime state.
    """

    all_contexts = getattr(trace, "_predicate_all_contexts", None)
    if all_contexts is None:
        all_contexts = []
        trace._predicate_all_contexts = all_contexts
    all_contexts.append(ctx)
    history_size = int(getattr(trace, "_predicate_history_size", 8))
    if history_size == 0:
        return
    history = getattr(trace, "_predicate_history", None)
    if history is None:
        history = deque()
        trace._predicate_history = history
    history.append(ctx)
    while len(history) > history_size:
        history.popleft()


def _trace_followed_by_candidate_selector(trace: "Trace") -> BaseSelector | None:
    """Return the candidate selector for ``candidate & followed_by(successor)``."""

    options = getattr(trace, "_predicate_save_options", None)
    predicate = None if options is None else options.keep_op
    if not isinstance(predicate, CompositeSelector) or predicate.operator != "and":
        return None
    left, right = predicate.selectors
    if isinstance(right, FollowedBySelector) and isinstance(left, BaseSelector):
        return left
    if isinstance(left, FollowedBySelector) and isinstance(right, BaseSelector):
        return right
    return None


def _retain_lookback_candidate(
    trace: "Trace",
    ctx: RecordContext,
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
) -> None:
    """Retain one candidate payload in the bounded retroactive-save window."""

    if ctx.raw_label is None:
        return
    lookback = int(getattr(trace, "_predicate_lookback", 0))
    if lookback <= 0:
        return
    policy = str(getattr(trace, "_predicate_lookback_payload_policy", "metadata_only"))
    if policy == "metadata_only":
        return
    candidate_selector = _trace_followed_by_candidate_selector(trace)
    if candidate_selector is None or not candidate_selector(ctx):
        return
    payload = _copy_lookback_payload(trace, fields_dict, tensor, policy)
    candidates = getattr(trace, "_predicate_lookback_candidates", None)
    if candidates is None:
        candidates = deque()
        trace._predicate_lookback_candidates = candidates
    candidates.append(_RetainedLookbackCandidate(raw_label=ctx.raw_label, payload=payload))
    while len(candidates) > lookback:
        candidates.popleft()


def _copy_lookback_payload(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    policy: str,
) -> _RetainedLookbackPayload:
    """Copy a candidate tensor according to the lookback payload policy."""

    detach = policy != "grad_connected"
    if policy == "disk_spilled":
        warnings.warn(
            "lookback_payload_policy='disk_spilled' currently retains the bounded candidate "
            "payload in memory before final bundle streaming.",
            RuntimeWarning,
            stacklevel=3,
        )
    raw_out = safe_copy(tensor, detach)
    if fields_dict["output_device"] not in [str(raw_out.device), "same"]:
        raw_out = safe_to(raw_out, fields_dict["output_device"])
    transformed_out = None
    if policy == "transformed" and trace.activation_transform is not None:
        transformed_out = apply_transform(
            label=fields_dict.get("_layer_label_raw"),
            raw_label=fields_dict.get("_label_raw"),
            func_name=fields_dict.get("func_name"),
            tensor=raw_out,
            transform=trace.activation_transform,
            transform_kind="activation",
            streaming_active=False,
        )
        validate_train_mode_transform_output(
            raw_tensor=raw_out,
            transformed_tensor=transformed_out,
            transform_kind="activation",
            backward_ready=fields_dict.get(
                "backward_ready", getattr(trace, "backward_ready", False)
            ),
            label=fields_dict.get("_layer_label_raw"),
        )
    store_raw = policy != "transformed" or getattr(trace, "save_raw_activations", True)
    raw_shape = tuple(raw_out.shape)
    raw_dtype = raw_out.dtype
    return _RetainedLookbackPayload(
        raw_out=raw_out if store_raw else None,
        transformed_out=transformed_out,
        shape=raw_shape,
        dtype=raw_dtype,
        activation_memory=get_memory_amount_from_metadata(raw_out, raw_shape, raw_dtype),
        transformed_shape=_shape_or_none(transformed_out),
        transformed_dtype=_dtype_or_none(transformed_out),
        transformed_memory=_memory_or_none(transformed_out),
    )


def _apply_retroactive_decision(
    trace: "Trace",
    decision: RetroactiveCaptureDecision,
) -> None:
    """Mark retained candidate events as saved after a successor matches."""

    policy = str(getattr(trace, "_predicate_lookback_payload_policy", "metadata_only"))
    if policy == "metadata_only":
        raise PredicateError(
            "tl.followed_by(...) requires lookback_payload_policy other than "
            "'metadata_only' so candidate payloads are retained."
        )
    candidates = getattr(trace, "_predicate_lookback_candidates", ())
    by_label = {
        candidate.raw_label: candidate
        for candidate in candidates
        if isinstance(candidate, _RetainedLookbackCandidate)
    }
    for raw_label in decision.target_raw_labels:
        candidate = by_label.get(raw_label)
        if candidate is None:
            warnings.warn(
                f"followed_by target {raw_label!r} was not retained in the payload lookback "
                "window; increase lookback.",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        if candidate.marked:
            continue
        candidate.marked = True
        _replace_event_with_retained_payload(trace, raw_label, candidate.payload)


def _replace_event_with_retained_payload(
    trace: "Trace",
    raw_label: str,
    payload: _RetainedLookbackPayload,
) -> None:
    """Replace a frozen op event with a saved-output version."""

    event = trace.capture_events.op_event_by_label_raw.get(raw_label)
    if event is None:
        return
    tensor_ref = dataclasses.replace(
        event.output.tensor,
        shape=payload.shape,
        dtype=str(payload.dtype),
        device=str(payload.raw_out.device)
        if payload.raw_out is not None
        else event.output.tensor.device,
        requires_grad=payload.raw_out.requires_grad
        if payload.raw_out is not None
        else event.output.tensor.requires_grad,
        memory=payload.activation_memory,
        payload=payload.raw_out,
        backend_handle_id=str(id(payload.raw_out)) if payload.raw_out is not None else None,
    )
    transformed_ref = None
    if payload.transformed_out is not None:
        transformed_ref = TensorRef(
            label_raw=raw_label,
            shape=payload.transformed_shape,
            dtype=str(payload.transformed_dtype),
            device=str(payload.transformed_out.device),
            requires_grad=payload.transformed_out.requires_grad,
            memory=payload.transformed_memory,
            payload=payload.transformed_out,
            blob_ref=None,
            backend_handle_id=str(id(payload.transformed_out)),
        )
    output_ref = dataclasses.replace(
        event.output,
        tensor=tensor_ref,
        transformed_tensor=transformed_ref,
        has_saved_activation=True,
    )
    updated_event = dataclasses.replace(event, output=output_ref, predicate_matched=True)
    trace.capture_events.op_event_by_label_raw[raw_label] = updated_event
    for index, existing_event in enumerate(trace.capture_events.op_events):
        if existing_event.label_raw == raw_label:
            trace.capture_events.op_events[index] = updated_event
            break


def _build_trace_predicate_context(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    *,
    parent_labels: tuple[str, ...] | None = None,
    output_index: int | None = None,
    is_bottom_level_func: bool = True,
) -> RecordContext:
    """Build the shared trace predicate context for one op.

    Parameters
    ----------
    trace:
        Active trace.
    fields_dict:
        Live exhaustive op fields for the tensor.
    tensor:
        Output tensor being considered.

    Parameters
    ----------
    trace:
        Active trace.
    fields_dict:
        Live exhaustive op fields for the tensor.
    tensor:
        Output tensor being considered.
    parent_labels:
        Optional raw parent labels supplied by the caller.
    output_index:
        Optional output index supplied by the caller.
    is_bottom_level_func:
        Whether this is a bottom-level function output.

    Returns
    -------
    RecordContext
        Predicate context shared by save and intervention slots.
    """

    history = tuple(getattr(trace, "_predicate_history", ()))
    raw_label = fields_dict["_label_raw"]
    return build_op_record_context(
        kind="op",
        label=raw_label,
        raw_label=raw_label,
        raw_index=int(fields_dict["raw_index"]),
        layer_type=str(fields_dict["type"]),
        type_index=int(fields_dict["type_index"]),
        func_name=fields_dict.get("func_name"),
        parent_labels=tuple(fields_dict.get("parents", ()))
        if parent_labels is None
        else parent_labels,
        tensor=tensor,
        output_index=fields_dict.get("multi_output_index")
        if output_index is None
        else output_index,
        is_bottom_level_func=is_bottom_level_func,
        module_stack=_module_stack_frames_from_fields(fields_dict),
        history=history,
        op_counts=dict(getattr(trace, "_raw_layer_type_counter", {})),
        pass_index=int(fields_dict.get("pass_index", 0)),
        event_index=int(fields_dict["raw_index"]),
        step_index=fields_dict.get("step_index"),
        capture_start_time=float(getattr(trace, "capture_start_time", time.time())),
        include_source_events=False,
        sample_id=None,
        address=fields_dict.get("module"),
        module_type=None,
        module_pass_index=None,
        is_transform=bool(fields_dict.get("is_transform", False)),
        transform_kind=fields_dict.get("transform_kind"),
    )


def _trace_predicate_context_key(
    raw_label: str,
    pass_index: int,
    container_path: tuple[Any, ...],
) -> tuple[str, int, tuple[Any, ...]]:
    """Return the runtime key for cached trace predicate contexts."""

    return (raw_label, pass_index, container_path)


def _cache_trace_predicate_context(
    trace: "Trace",
    ctx: RecordContext,
    container_path: tuple[Any, ...],
) -> None:
    """Cache a pre-save predicate context for the current op."""

    contexts = getattr(trace, "_predicate_current_contexts", None)
    if contexts is None:
        contexts = {}
        trace._predicate_current_contexts = contexts
    contexts[
        _trace_predicate_context_key(
            ctx.raw_label or ctx.label,
            ctx.pass_index,
            container_path,
        )
    ] = ctx


def _pop_trace_predicate_context(
    trace: "Trace",
    fields_dict: dict[str, Any],
) -> RecordContext | None:
    """Pop a cached predicate context for an op, if one exists."""

    contexts = getattr(trace, "_predicate_current_contexts", None)
    if contexts is None:
        return None
    key = _trace_predicate_context_key(
        str(fields_dict["_label_raw"]),
        int(fields_dict.get("pass_index", 0)),
        tuple(fields_dict.get("container_path", ())),
    )
    return contexts.pop(key, None)


def _evaluate_trace_save_predicate(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
) -> tuple[CaptureSpec | None, RecordContext | None]:
    """Evaluate ``trace(save=...)`` for one exhaustive op, if configured.

    Parameters
    ----------
    trace:
        Active trace.
    fields_dict:
        Live exhaustive op fields for the tensor.
    tensor:
        Output tensor being considered.

    Returns
    -------
    CaptureSpec | None
        Capture decision when selective predicate save is active, otherwise ``None``.
    """

    options = getattr(trace, "_predicate_save_options", None)
    if options is None:
        return None, None
    if options.keep_op is None:
        return None, None
    ctx = _pop_trace_predicate_context(trace, fields_dict)
    if ctx is None:
        ctx = _build_trace_predicate_context(trace, fields_dict, tensor)
    raw_label = str(fields_dict["_label_raw"])
    try:
        decision = _evaluate_keep_op(ctx, options)
    finally:
        _append_trace_predicate_context(trace, ctx)
    if isinstance(decision, RetroactiveCaptureDecision):
        _apply_retroactive_decision(trace, decision)
        spec = CaptureSpec(save_out=False, save_metadata=True)
    else:
        spec = decision
    decisions = getattr(trace, "_predicate_save_decisions", None)
    if decisions is None:
        decisions = {}
        trace._predicate_save_decisions = decisions
    decisions[
        (
            raw_label,
            int(fields_dict.get("pass_index", 0)),
            tuple(fields_dict.get("container_path", ())),
        )
    ] = spec
    return spec, ctx


def _make_layer_log_entry(
    self: "Trace",
    t: torch.Tensor,
    fields_dict: dict[str, Any],
    t_args: tuple[Any, ...] | None = None,
    t_kwargs: dict[str, Any] | None = None,
    activation_transform: Callable[..., Any] | None = None,
) -> Any:
    """Create a Op (or Buffer) entry and register it in Trace.

    Instantiates the appropriate log class from ``fields_dict``, conditionally
    saves out data (if this layer is in ``_layer_nums_to_save``), and
    appends the entry to ``_raw_layer_dict`` and ``_raw_layer_labels_list``.

    Args:
        t: The tensor to log.
        fields_dict: Complete field dictionary (~80 fields) for the log entry.
        t_args: Positional arguments to the function that created the tensor.
        t_kwargs: Keyword arguments to the function that created the tensor.
        activation_transform: Optional transform applied to outs before saving.
    """
    if t_args is None:
        t_args = ()
    if t_kwargs is None:
        t_kwargs = {}

    fire_results = tuple(fields_dict.pop("fire_results", ()))
    from ...capture.projections import LiveOpView

    keep_by_predicate = True
    module_filter = getattr(self, "module_filter", None)
    if module_filter is not None:
        keep_by_predicate = bool(module_filter(SimpleNamespace(**fields_dict)))
    layer_nums_to_save = cast(Any, self._layer_nums_to_save)
    raw_index = cast(int, fields_dict["raw_index"])
    predicate_spec, predicate_ctx = _evaluate_trace_save_predicate(self, fields_dict, t)
    if predicate_spec is None:
        save_this_activation = (layer_nums_to_save == "all") or (raw_index in layer_nums_to_save)
    else:
        save_this_activation = predicate_spec.save_out
    if keep_by_predicate and save_this_activation:
        if predicate_spec is None or predicate_ctx is None:
            with _timed_phase(self, "clone_save:activation_fields"):
                _save_activation_fields(
                    self,
                    fields_dict,
                    t,
                    t_args,
                    t_kwargs,
                    activation_transform,
                )
        else:
            with _timed_phase(self, "clone_save:activation_fields"):
                _save_predicate_activation_fields(
                    self,
                    fields_dict,
                    t,
                    predicate_spec,
                    predicate_ctx,
                    activation_transform,
                )
    op_event = _op_event_from_log(self, fields_dict, t, fire_results)
    self.capture_events.append(op_event)
    if op_event.grad_fn_handle is not None:
        self.capture_events.grad_fn_handles_by_label_raw[op_event.label_raw] = (
            op_event.grad_fn_handle
        )
    new_entry = LiveOpView(self, op_event)
    if predicate_ctx is not None and not save_this_activation:
        _retain_lookback_candidate(self, predicate_ctx, fields_dict, t)
    _raise_if_nonfinite_requested(self, t, new_entry)

    return new_entry


def _raise_if_nonfinite_requested(self: Any, tensor: torch.Tensor, entry: Any) -> None:
    """Raise a structured capture error if ``raise_on_nan`` finds a non-finite tensor.

    Parameters
    ----------
    self:
        Active ``Trace`` instance.
    tensor:
        Tensor output produced by the just-logged operation.
    entry:
        Newly registered layer pass log for ``tensor``.

    Raises
    ------
    CaptureError
        If ``self.raise_on_nan`` is enabled and ``tensor`` contains NaN or Inf.
    """

    if not getattr(self, "raise_on_nan", False) or tensor.numel() == 0:
        return
    try:
        with pause_logging():
            has_nonfinite = bool(
                (~torch.isfinite(safe_copy(tensor, detach_tensor=True))).any().item()
            )
    except (RuntimeError, TypeError):
        return
    if not has_nonfinite:
        return

    raw_label = getattr(entry, "_label_raw", getattr(entry, "_layer_label_raw", "unknown"))
    func_name = getattr(entry, "func_name", "unknown")
    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    parents = list(getattr(entry, "parents", []) or [])
    message = (
        "TorchLens capture stopped at first non-finite tensor: "
        f"op={func_name!r}, layer={raw_label!r}, shape={shape}, dtype={dtype}."
    )
    raise CaptureError(
        message,
        affected_sites=[raw_label],
        op=func_name,
        layer=raw_label,
        shape=shape,
        dtype=str(dtype),
        parents=parents,
    )
