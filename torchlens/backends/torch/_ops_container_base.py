"""Exhaustive drafts and foundational output-container helpers."""

import copy
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from ...ir.container import (
    ContainerSpec,
    DictKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
)
from ...ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
)
from ...ir.intervention import FireResult
from ...utils._torch_compat import (
    torch_structseq_field_names,
)
from ...utils.introspection import (
    get_vars_of_type_from_obj,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace
    from ...ir.op_record import OpRecord

if TYPE_CHECKING:
    from .ops import (
        _SAFE_DEFAULT_FACTORIES,
        _build_container_spec,
        _build_edge_use_records,
        _exhaustive_capture_policy,
        _exhaustive_freeze_refs,
        _exhaustive_output_ref,
        _literal_value_supported,
        _param_refs_from_fields,
        _parent_edges_from_fields,
        _resolve_call_function_ref,
    )

__all__ = (
    "_op_record_from_log",
    "_is_namedtuple_instance",
    "_torch_return_type_fields",
    "_non_iterable_type_error",
    "_iter_sequence_items",
    "_try_build_container_spec",
    "_fallback_address_to_path",
    "_is_hf_model_output",
    "_container_type_ref",
    "_safe_default_factory_name",
    "_mapping_reconstruction",
    "_object_holds_tensor",
    "_leaf_is_reconstructable",
)


def _op_record_from_log(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    fire_results: tuple[FireResult, ...] = (),
    module_stack: tuple[ModuleFrame, ...] | None = None,
    call_ref_box: list[FunctionCallRef] | None = None,
) -> "OpRecord":
    """Exhaustive-pipeline decomposed freeze: ``OpCore`` + facets.

    Reads the final exhaustive draft (``fields_dict``) through the shared
    ref builders; facet PRESENCE mirrors
    ``op_record_from_event`` applied to the equivalent compat event (S5:
    absent facet != fabricated empty facet).
    """

    from ...ir.op_record import (
        AncestryFacet,
        AnnotationsFacet,
        AutogradFacet,
        ControlFacet,
        GraphFacet,
        InterventionFacet,
        ModulesFacet,
        OpCore,
        OpRecord,
        ParamsFacet,
        PolicyFacet,
        TransformFacet,
    )

    tensor_ref, module_stack, transformed_ref, backend_semantics = _exhaustive_freeze_refs(
        fields_dict, tensor, module_stack
    )
    core = OpCore(
        seq=0,
        kind="source" if fields_dict["is_input"] or fields_dict["is_buffer"] else "op",
        label_raw=fields_dict["_label_raw"],
        layer_label_raw=fields_dict["_layer_label_raw"],
        layer_type=fields_dict["type"],
        raw_index=fields_dict["raw_index"],
        type_index=fields_dict["type_index"],
        step_index=fields_dict["step_index"] or 0,
        pass_index=fields_dict["pass_index"],
        parents=_parent_edges_from_fields(fields_dict),
        output=_exhaustive_output_ref(fields_dict, tensor_ref, transformed_ref),
        is_bottom_level=True,
        func_call_id=fields_dict["func_call_id"],
    )
    modules = tuple(fields_dict["modules"])
    input_ancestors = frozenset(fields_dict["input_ancestors"])
    internal_source_ancestors = frozenset(fields_dict["internal_source_ancestors"])
    root_ancestors = frozenset(fields_dict["root_ancestors"])
    has_internal_source_ancestor = fields_dict["has_internal_source_ancestor"]
    grad_fn_class_qualname = fields_dict["grad_fn_class_qualname"]
    # The legacy event's transform_config = {**user config, "_tl_annotations"}
    # and the adapter pops the two smuggled channels back out; the direct
    # builder starts from the clean user mapping and never smuggles.
    user_transform_config = dict(fields_dict.get("transform_config") or {})
    user_transform_config.pop("_tl_annotations", None)
    fn_code_location = user_transform_config.pop("fn_code_location", None)
    annotations_payload = dict(fields_dict.get("annotations") or {})
    is_transform = bool(fields_dict.get("is_transform", False))
    transform_kind = fields_dict.get("transform_kind")
    transform_chain = tuple(fields_dict.get("transform_chain") or ())
    transform_fn_name = fields_dict.get("transform_fn_name")
    transform_fn_qualname = fields_dict.get("transform_fn_qualname")
    transform_fn_source = fields_dict.get("transform_fn_source")
    params = _param_refs_from_fields(fields_dict)
    parent_params = tuple(fields_dict["parent_params"])
    is_scalar_bool = fields_dict["is_scalar_bool"]
    bool_value = fields_dict["bool_value"]
    intervention_replaced = fields_dict["intervention_replaced"]
    return OpRecord(
        core=core,
        function=_resolve_call_function_ref(fields_dict, call_ref_box),
        templates=ArgTemplateRef(
            saved_args=fields_dict["saved_args"],
            saved_kwargs=fields_dict["saved_kwargs"],
            args_template=fields_dict["args_template"],
            kwargs_template=fields_dict["kwargs_template"],
            has_saved_args=fields_dict["has_saved_args"],
        ),
        graph=GraphFacet(
            parent_arg_positions=copy.deepcopy(fields_dict["parent_arg_positions"]),
            edge_uses=tuple(
                fields_dict["_edge_uses"]
                or _build_edge_use_records(
                    trace,
                    fields_dict["parent_arg_positions"],
                    fields_dict["_label_raw"],
                    fields_dict["func_call_id"],
                    func_name=fields_dict.get("func_name"),
                )
            ),
            unattributed_tensor_args=tuple(fields_dict.get("unattributed_tensor_args") or ()),
            dropped_edge_tensor_args=tuple(fields_dict.get("dropped_edge_tensor_args") or ()),
            is_output_parent=fields_dict["is_output_parent"],
            input_was_parameter=bool(fields_dict.get("input_was_parameter", False)),
            equivalence_class=fields_dict["equivalence_class"],
        ),
        modules_facet=(
            ModulesFacet(module_stack=module_stack, modules=modules)
            if module_stack or modules
            else None
        ),
        ancestry=(
            AncestryFacet(
                input_ancestors=input_ancestors,
                internal_source_ancestors=internal_source_ancestors,
                root_ancestors=root_ancestors,
                has_internal_source_ancestor=has_internal_source_ancestor,
            )
            if (
                input_ancestors
                or internal_source_ancestors
                or root_ancestors
                or has_internal_source_ancestor
            )
            else None
        ),
        autograd=(
            AutogradFacet(grad_fn_class_qualname=grad_fn_class_qualname)
            if grad_fn_class_qualname is not None
            else None
        ),
        transform=(
            TransformFacet(
                is_transform=is_transform,
                transform_kind=transform_kind,
                transform_chain=transform_chain,
                transform_config=user_transform_config,
                transform_fn_name=transform_fn_name,
                transform_fn_qualname=transform_fn_qualname,
                transform_fn_source=transform_fn_source,
                fn_code_location=fn_code_location,
            )
            if (
                is_transform
                or transform_kind is not None
                or transform_chain
                or user_transform_config
                or transform_fn_name is not None
                or transform_fn_qualname is not None
                or transform_fn_source is not None
                or fn_code_location is not None
            )
            else None
        ),
        control=(
            ControlFacet(is_scalar_bool=is_scalar_bool, bool_value=bool_value)
            if is_scalar_bool is not None or bool_value is not None
            else None
        ),
        params_facet=(
            ParamsFacet(params=params, parent_params=parent_params)
            if params or parent_params
            else None
        ),
        annotations_facet=(
            AnnotationsFacet(annotations=annotations_payload) if annotations_payload else None
        ),
        policy_facet=PolicyFacet(
            backend_semantics=backend_semantics,
            policy=_exhaustive_capture_policy(trace, fields_dict),
            predicate_matched=True,
            tracing_finished=fields_dict["_tracing_finished"],
            construction_done=fields_dict["_construction_done"],
        ),
        intervention=(
            InterventionFacet(
                intervention_fired=bool(fire_results),
                intervention_replaced=intervention_replaced,
                fire_results=fire_results,
            )
            if (bool(fire_results) or intervention_replaced or fire_results)
            else None
        ),
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

    r35 hon1_4: delegates to the shared repr-independent helper in
    ``utils/_torch_compat.py`` (``__match_args__`` primary; identity round-trip
    fallback when it is absent; refusal otherwise). Structural facts must NEVER derive
    from ``repr()``/``str()`` of tensor-bearing values -- a wrapped tensor repr
    used to inject phantom ``dtype=`` fields and flip witness verdicts on
    tensor size alone.

    Parameters
    ----------
    value
        Object to inspect.

    Returns
    -------
    tuple[str, ...]
        Field names when PyTorch exposes a named structseq, otherwise ``()``.
    """

    return torch_structseq_field_names(value)


def _non_iterable_type_error(exc: TypeError) -> bool:
    """Return whether ``exc`` represents an opaque non-iterable object.

    Parameters
    ----------
    exc
        TypeError raised while attempting output-container iteration.

    Returns
    -------
    bool
        True when the exception text matches Python's non-iterable diagnostics.
    """

    return "not iterable" in str(exc)


def _iter_sequence_items(value: Any) -> tuple[tuple[int, Any], ...] | None:
    """Return indexed sequence items, or no items for opaque non-iterables.

    Parameters
    ----------
    value
        Candidate list/tuple output container.

    Returns
    -------
    tuple[tuple[int, Any], ...] | None
        Enumerated child values. ``None`` means the object raised a
        non-iterable ``TypeError`` and should be treated as an opaque leaf.
    """

    try:
        return tuple(enumerate(value))
    except TypeError as exc:
        if _non_iterable_type_error(exc):
            return None
        raise


def _try_build_container_spec(
    value: Any,
    *,
    _depth: int = 0,
    _in_progress: set[int] | None = None,
) -> ContainerSpec | None:
    """Build a child container spec, treating opaque non-iterables as leaves.

    Parameters
    ----------
    value
        Child output value to describe.
    _depth
        Internal recursion depth forwarded to the guarded builder (r-b4 R27-4).
    _in_progress
        Internal path-scoped container-id set forwarded to the guarded builder.

    Returns
    -------
    ContainerSpec | None
        Child container spec, or ``None`` when the child is an opaque leaf.
    """

    try:
        return _build_container_spec(value, _depth=_depth, _in_progress=_in_progress)
    except TypeError as exc:
        if _non_iterable_type_error(exc):
            return None
        raise


def _fallback_address_to_path(address: list[tuple[str, Any]]) -> tuple[OutputPathComponent, ...]:
    """Convert a generic introspection address to a typed output path suffix.

    Parameters
    ----------
    address
        Programmatic address emitted by :func:`get_vars_of_type_from_obj`.

    Returns
    -------
    tuple[OutputPathComponent, ...]
        Best-effort typed path components for an opaque output subtree.
    """

    path: list[OutputPathComponent] = []
    for kind, value in address:
        if kind == "attr":
            path.append(NamedField(str(value)))
        elif kind == "ind" and isinstance(value, int):
            path.append(TupleIndex(value))
        elif kind == "ind":
            path.append(DictKey(value))
        else:
            path.append(str(value))
    return tuple(path)


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


def _safe_default_factory_name(factory: Any) -> str | None:
    """Return the allowlisted name of a defaultdict factory, or ``None``."""

    for name, ctor in _SAFE_DEFAULT_FACTORIES.items():
        if factory is ctor:
            return name
    return None


def _mapping_reconstruction(
    value: Mapping[Any, Any],
) -> tuple[str | None, str | None, Any] | None:
    """Return faithful-rebuild metadata for a mapping output container.

    Returns ``(type_module, type_qualname, aux_data)`` for a mapping we can rebuild
    with the EXACT type (plain ``dict`` -> no type ref; ``OrderedDict``;
    ``defaultdict`` with an allowlisted factory). Returns ``None`` for a mapping we
    cannot reconstruct exactly (custom ``Mapping``, unknown ``dict`` subclass,
    unsafe ``defaultdict`` factory) so the caller records it opaquely rather than
    silently collapsing it to a plain ``dict`` / bare tensor under a VERIFIED run.
    """

    if type(value) is dict:
        return (None, None, None)
    if type(value) is OrderedDict:
        return ("collections", "OrderedDict", None)
    if type(value) is defaultdict:
        factory = value.default_factory
        if factory is None:
            return ("collections", "defaultdict", {"default_factory": None})
        name = _safe_default_factory_name(factory)
        if name is not None:
            return ("collections", "defaultdict", {"default_factory": name})
        return None
    return None


def _object_holds_tensor(value: Any) -> bool:
    """Return whether ``value`` opaquely contains at least one tensor."""

    for _tensor in get_vars_of_type_from_obj(
        value,
        which_type=torch.Tensor,
        subclass_exceptions=[torch.nn.Parameter],
        search_depth=5,
    ):
        return True
    return False


def _leaf_is_reconstructable(item: Any) -> bool:
    """Return whether a childless (``_build_container_spec is None``) output leaf can be restored.

    A tensor leaf is filled from the flat leaf stream and an encodable literal
    is rebuilt in place. ANY other leaf makes the enclosing container
    non-reconstructable (recorded ``opaque``), so replay is honestly
    UNVERIFIABLE and the runnable save refuses at preflight instead of
    advertising a contract the codec cannot fill: a pure non-tensor object
    (``memoryview``, an arbitrary object with no tensors) AND a tensor-HOLDING
    opaque object (HuggingFace ``DynamicCache``) alike. The latter used to pass
    as a leaf (W051-HONESTY H1): the spec then declared one slot where the
    capture's BFS fallback held several tensors, and the rebuild ran dry with a
    bare ``ValueError``. The BFS tolerance that still CAPTURES those tensors
    lives in the output walker, not here; this predicate is the spec builder's.
    """

    if isinstance(item, torch.Tensor):
        return True
    return bool(_literal_value_supported(item) or isinstance(item, torch.Size))
