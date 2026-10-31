"""Output-container contracts, reconstruction, and tensor walking."""

import dataclasses
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..._ir.container import (
    ContainerSpec,
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
    get_registered_container,
    mapping_extra_instance_state,
    namedtuple_extra_instance_state,
    namedtuple_type_can_carry_instance_state,
    reconstruction_is_lossy,
)
from ..._ir.container_registry import OUTPUT_TREE_MAX_DEPTH
from ..._utils.introspection import (
    get_vars_of_type_from_obj,
)

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from .ops import (
        _UNSUPPORTED_OUTPUT_CONTAINER_WARNED,
        _container_type_ref,
        _fallback_address_to_path,
        _is_hf_model_output,
        _is_namedtuple_instance,
        _iter_sequence_items,
        _leaf_is_reconstructable,
        _literal_value_supported,
        _mapping_reconstruction,
        _torch_return_type_fields,
        _try_build_container_spec,
    )

__all__ = (
    "_build_container_spec",
    "_build_container_spec_unguarded",
    "_known_output_container_children",
    "_walk_supported_output_container",
    "_prove_runnable_output_lossless",
    "_prove_runnable_output_lossless_unguarded",
    "runnable_output_losslessness",
    "_walk_output_tensors_with_paths",
)


def _build_container_spec(
    value: Any,
    *,
    _depth: int = 0,
    _in_progress: set[int] | None = None,
) -> ContainerSpec | None:
    """Build a replay container spec for a supported output container.

    Over-deep and self-referential subtrees degrade to the honest ``opaque``
    lane (r-b4 R27-4) instead of exhausting the interpreter stack; the cycle
    guard is path-scoped so DAG-shaped outputs keep one spec per occurrence.

    Parameters
    ----------
    value
        Output object to describe.
    _depth
        Internal recursion depth (callers must not supply this).
    _in_progress
        Internal path-scoped container-id set (callers must not supply this).

    Returns
    -------
    ContainerSpec | None
        Container spec, or ``None`` for a single tensor / unsupported scalar.
    """

    if _literal_value_supported(value) or isinstance(value, torch.Size):
        return ContainerSpec(kind="literal", literal_value=value)
    if _in_progress is None:
        _in_progress = set()
    value_id = id(value)
    if _depth >= OUTPUT_TREE_MAX_DEPTH or value_id in _in_progress:
        module, qualname = _container_type_ref(value)
        return ContainerSpec(kind="opaque", type_module=module, type_qualname=qualname)
    _in_progress.add(value_id)
    try:
        return _build_container_spec_unguarded(value, _depth=_depth, _in_progress=_in_progress)
    finally:
        _in_progress.discard(value_id)


def _build_container_spec_unguarded(
    value: Any,
    *,
    _depth: int,
    _in_progress: set[int],
) -> ContainerSpec | None:
    """Build one guarded output container node's spec (dispatch body)."""

    child_specs: list[tuple[OutputPathComponent, ContainerSpec]] = []
    registered = get_registered_container(type(value))
    if registered is not None:
        children, aux_data = registered.flatten(value)
        for index, item in enumerate(children):
            child_spec = _try_build_container_spec(
                item, _depth=_depth + 1, _in_progress=_in_progress
            )
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
        reconstructable = True
        for key in keys:
            child = value[key]
            child_spec = _try_build_container_spec(
                child, _depth=_depth + 1, _in_progress=_in_progress
            )
            if child_spec is not None:
                child_specs.append((HFKey(key), child_spec))
                if child_spec.kind == "opaque":
                    reconstructable = False
            elif not _leaf_is_reconstructable(child):
                reconstructable = False
        module, qualname = _container_type_ref(value)
        if not reconstructable:
            return ContainerSpec(kind="opaque", type_module=module, type_qualname=qualname)
        return ContainerSpec(
            kind="hf_model_output",
            length=len(keys),
            keys=keys,
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
            lossy_reconstruction=reconstruction_is_lossy(value, keys),
        )
    torch_fields = _torch_return_type_fields(value)
    if type(value).__module__ == "torch.return_types" and not torch_fields:
        # r35 hon1_4: a structseq whose field declaration cannot be PROVEN
        # (no valid __match_args__, no identity bijection) is recorded opaque
        # so the runnable producer refuses it typed -- never silently recorded
        # as a plain positional tuple (a lossy type substitution).
        module, qualname = _container_type_ref(value)
        return ContainerSpec(kind="opaque", type_module=module, type_qualname=qualname)
    if _is_namedtuple_instance(value) or torch_fields:
        fields = torch_fields or tuple(value._fields)
        reconstructable = True
        for field_name in fields:
            child = getattr(value, field_name)
            child_spec = _try_build_container_spec(
                child, _depth=_depth + 1, _in_progress=_in_progress
            )
            if child_spec is not None:
                child_specs.append((NamedField(field_name), child_spec))
                if child_spec.kind == "opaque":
                    reconstructable = False
            elif not _leaf_is_reconstructable(child):
                reconstructable = False
        module, qualname = _container_type_ref(value)
        if not reconstructable:
            return ContainerSpec(kind="opaque", type_module=module, type_qualname=qualname)
        return ContainerSpec(
            kind="namedtuple",
            length=len(value),
            fields=fields,
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
            # r37 secB_1 capture-side parity with dataclass/hf: the inert
            # ``tuple.__new__`` rebuild drops non-field instance state.
            lossy_reconstruction=namedtuple_extra_instance_state(value),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = tuple(field.name for field in dataclasses.fields(value))
        reconstructable = True
        for field_name in fields:
            child = getattr(value, field_name)
            child_spec = _try_build_container_spec(
                child, _depth=_depth + 1, _in_progress=_in_progress
            )
            if child_spec is not None:
                child_specs.append((DataclassField(field_name), child_spec))
                if child_spec.kind == "opaque":
                    reconstructable = False
            elif not _leaf_is_reconstructable(child):
                reconstructable = False
        module, qualname = _container_type_ref(value)
        if not reconstructable:
            return ContainerSpec(kind="opaque", type_module=module, type_qualname=qualname)
        return ContainerSpec(
            kind="dataclass",
            length=len(fields),
            fields=fields,
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
            lossy_reconstruction=reconstruction_is_lossy(value, fields),
        )
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        # A tensor leaf under a non-str/int key cannot be bound to the frozen
        # runnable slot-path vocabulary, so such a mapping is not runnable; record
        # it opaque (honest-reject at save) rather than crashing at descriptor build.
        # ``bool`` keys are ``int`` subclasses and remain representable.
        reconstructable = all(isinstance(key, (str, int)) for key in keys)
        for key in keys:
            child = value[key]
            child_spec = _try_build_container_spec(
                child, _depth=_depth + 1, _in_progress=_in_progress
            )
            if child_spec is not None:
                child_specs.append((DictKey(key), child_spec))
                if child_spec.kind == "opaque":
                    reconstructable = False
            elif not _leaf_is_reconstructable(child):
                reconstructable = False
        recon = _mapping_reconstruction(value)
        if recon is None or not reconstructable:
            # A mapping we cannot faithfully rebuild (custom Mapping, unknown dict
            # subclass, unsafe defaultdict factory, or an unrepresentable leaf).
            # Record it opaquely so producer preflight refuses to advertise runnable
            # and the run is UNVERIFIABLE, never a silent bare-tensor/plain-dict.
            module, qualname = _container_type_ref(value)
            return ContainerSpec(kind="opaque", type_module=module, type_qualname=qualname)
        type_module, type_qualname, aux = recon
        return ContainerSpec(
            kind="dict",
            length=len(keys),
            keys=keys,
            type_module=type_module,
            type_qualname=type_qualname,
            child_specs=tuple(child_specs),
            aux_data=aux,
        )
    if isinstance(value, tuple):
        items = _iter_sequence_items(value)
        if items is None:
            return None
        reconstructable = True
        for index, item in items:
            child_spec = _try_build_container_spec(
                item, _depth=_depth + 1, _in_progress=_in_progress
            )
            if child_spec is not None:
                child_specs.append((TupleIndex(index), child_spec))
                if child_spec.kind == "opaque":
                    reconstructable = False
            elif not _leaf_is_reconstructable(item):
                reconstructable = False
        if not reconstructable:
            module, qualname = _container_type_ref(value)
            return ContainerSpec(kind="opaque", type_module=module, type_qualname=qualname)
        return ContainerSpec(kind="tuple", length=len(value), child_specs=tuple(child_specs))
    if isinstance(value, list):
        items = _iter_sequence_items(value)
        if items is None:
            return None
        reconstructable = True
        for index, item in items:
            child_spec = _try_build_container_spec(
                item, _depth=_depth + 1, _in_progress=_in_progress
            )
            if child_spec is not None:
                child_specs.append((TupleIndex(index), child_spec))
                if child_spec.kind == "opaque":
                    reconstructable = False
            elif not _leaf_is_reconstructable(item):
                reconstructable = False
        if not reconstructable:
            module, qualname = _container_type_ref(value)
            return ContainerSpec(kind="opaque", type_module=module, type_qualname=qualname)
        return ContainerSpec(kind="list", length=len(value), child_specs=tuple(child_specs))
    return None


def _known_output_container_children(
    out: Any,
) -> list[tuple[OutputPathComponent, Any]] | None:
    """Return ``(component, child)`` pairs for a recognized output container.

    ``None`` means the container kind is UNRECOGNIZED (the caller falls through
    to the opaque-boundary tensor fallback). An empty list means recognized
    with nothing to walk (the historical list/tuple ``_iter_sequence_items``
    ``None`` early-return).
    """

    registered = get_registered_container(type(out))
    if registered is not None:
        children, _aux_data = registered.flatten(out)
        return [(TupleIndex(index), item) for index, item in enumerate(children)]
    if _is_hf_model_output(out):
        return [(HFKey(key), out[key]) for key in out.keys()]
    torch_fields = _torch_return_type_fields(out)
    if _is_namedtuple_instance(out) or torch_fields:
        fields = torch_fields or tuple(out._fields)
        return [(NamedField(field_name), getattr(out, field_name)) for field_name in fields]
    if dataclasses.is_dataclass(out) and not isinstance(out, type):
        return [
            (DataclassField(field.name), getattr(out, field.name))
            for field in dataclasses.fields(out)
        ]
    if isinstance(out, dict):
        return [(DictKey(key), value) for key, value in out.items()]
    if isinstance(out, (list, tuple)):
        items = _iter_sequence_items(out)
        if items is None:
            return []
        return [(TupleIndex(index), item) for index, item in items]
    return None


def _walk_supported_output_container(
    out: Any,
    *,
    root_spec: ContainerSpec,
    path: tuple[OutputPathComponent, ...],
    _depth: int = 0,
    _in_progress: set[int] | None = None,
) -> Iterator[tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]]:
    """Yield tensors from a supported output container.

    Bounded by the same ceiling/cycle policy as the spec builders (r-b4 R27-4):
    an over-deep or self-referential subtree is treated exactly like an
    unrecognized container -- its tensors are yielded at the opaque boundary
    path by the fallback below, matching the ``opaque`` spec node the builder
    records at that same depth. The cycle guard is path-scoped, so DAG-shaped
    outputs are walked once per occurrence.

    Parameters
    ----------
    out
        Output object or nested child object to traverse.
    root_spec
        Spec for the outermost output container.
    path
        Path accumulated from the outermost output container.
    _depth
        Internal recursion depth (callers must not supply this).
    _in_progress
        Internal path-scoped container-id set (callers must not supply this).

    Yields
    ------
    tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]
        Tensor, path, and root container spec.
    """

    if isinstance(out, torch.Tensor):
        if not isinstance(out, torch.nn.Parameter):
            yield out, path, root_spec
        return
    if _in_progress is None:
        _in_progress = set()
    out_id = id(out)
    if _depth < OUTPUT_TREE_MAX_DEPTH and out_id not in _in_progress:
        children = _known_output_container_children(out)
        if children is not None:
            _in_progress.add(out_id)
            try:
                for component, child in children:
                    yield from _walk_supported_output_container(
                        child,
                        root_spec=root_spec,
                        path=(*path, component),
                        _depth=_depth + 1,
                        _in_progress=_in_progress,
                    )
            finally:
                _in_progress.discard(out_id)
            return
    # Unrecognized nested container (e.g. transformers DynamicCache nested inside
    # an HF ModelOutput, or a detectron2 Instances inside a list). The structured
    # walk cannot descend into this subtree to assign deeper stable paths, so every
    # tensor it holds is attributed to the path of the opaque container boundary
    # itself -- the same depth at which ``_build_container_spec`` records the opaque
    # slot as a childless leaf. Yielding tensors here is mandatory: otherwise
    # capture silently drops them (e.g. GPT-2's past_key_values), shrinking the
    # output set from 3 tensors to 1. ``root_spec`` must be propagated (not None);
    # it is the outer container spec used as ``output_structure``, and dropping it
    # leaves ``output_structure`` unset so it is later back-filled from an
    # unrelated output layer, producing a structure whose leaf paths disagree with
    # these output paths (caught by the module_hierarchy invariant). search_depth=5
    # matches the whole-output BFS fallback; DynamicCache's tensors live at
    # depth ~5 and are missed by the default depth of 3.
    for tensor, _address, fallback_address in get_vars_of_type_from_obj(
        out,
        which_type=torch.Tensor,
        subclass_exceptions=[torch.nn.Parameter],
        search_depth=5,
        return_addresses=True,
    ):
        yield tensor, (*path, *_fallback_address_to_path(fallback_address)), root_spec


def _prove_runnable_output_lossless(
    value: Any,
    *,
    _depth: int = 0,
    _in_progress: set[int] | None = None,
) -> tuple[bool, str]:
    """Positively prove one MODEL-output subtree is losslessly reconstructable (r35 I1).

    A runnable claim requires refuse-unless-proved: the proof establishes an exact
    root kind, recursively supported child kinds, and fully encodable literal
    leaves. It deliberately does NOT extend ``_leaf_is_reconstructable``'s
    "childless leaf that still holds tensors" tolerance (that tolerance exists
    only for non-runnable BFS capture): an opaque tensor holder, and any
    unordered container (``set``/``frozenset``/subclasses) at ANY depth and ANY
    cardinality (zero, one, or many tensors), fails the proof.

    Returns
    -------
    tuple[bool, str]
        ``(proved, reason)`` where ``reason`` names the first failure.
    """

    if isinstance(value, torch.nn.Parameter):
        return False, "parameter_output"
    if isinstance(value, torch.Tensor):
        return True, ""
    if isinstance(value, (set, frozenset)):
        return False, f"unordered_container:{type(value).__qualname__}"
    if _literal_value_supported(value) or isinstance(value, torch.Size):
        return True, ""
    # r-b4 R27-4: refuse-unless-proved makes the guard trivial -- an over-deep or
    # self-referential subtree simply cannot be proved lossless.
    if _in_progress is None:
        _in_progress = set()
    if _depth >= OUTPUT_TREE_MAX_DEPTH:
        return False, "output_tree_depth_exceeded"
    if id(value) in _in_progress:
        return False, f"output_tree_cycle:{type(value).__qualname__}"
    _in_progress.add(id(value))
    try:
        return _prove_runnable_output_lossless_unguarded(
            value, _depth=_depth, _in_progress=_in_progress
        )
    finally:
        _in_progress.discard(id(value))


def _prove_runnable_output_lossless_unguarded(
    value: Any,
    *,
    _depth: int,
    _in_progress: set[int],
) -> tuple[bool, str]:
    """Prove one guarded output container node lossless (dispatch body)."""

    registered = get_registered_container(type(value))
    if registered is not None:
        if not getattr(registered, "state_complete", False):
            instance_dict = getattr(value, "__dict__", None)
            if isinstance(instance_dict, dict) and any(
                item is not None for item in instance_dict.values()
            ):
                # r37 3-ADJ-3: registration without the explicit state_complete
                # declaration cannot prove the hooks round-trip this instance's
                # extra state -- refuse at save rather than drop it on replay.
                return False, f"registered_container_instance_state:{type(value).__qualname__}"
        children, _aux = registered.flatten(value)
        for item in children:
            proved, reason = _prove_runnable_output_lossless(
                item, _depth=_depth + 1, _in_progress=_in_progress
            )
            if not proved:
                return False, reason
        return True, ""
    if _is_hf_model_output(value):
        for key in value.keys():
            proved, reason = _prove_runnable_output_lossless(
                value[key], _depth=_depth + 1, _in_progress=_in_progress
            )
            if not proved:
                return False, reason
        return True, ""
    torch_fields = _torch_return_type_fields(value)
    if _is_namedtuple_instance(value) or torch_fields:
        fields = torch_fields or tuple(value._fields)
        if type(value).__module__ == "torch.return_types" and not torch_fields:
            return False, "structseq_fields_unprovable"
        if namedtuple_type_can_carry_instance_state(type(value)):
            # r39 corr1-1: refuse at save any namedtuple TYPE capable of carrying dropped
            # instance state (an unslotted subclass), even when THIS instance's ``__dict__``
            # is currently empty -- load cannot see the original instance and secB_1 forbids
            # trusting a persisted "no extras" flag, so instance-emptiness is unprovable at
            # load. This is the SAME structural type-level criterion the load-time gate
            # (``_spec_node_reconstruction_lossy`` -> ``namedtuple_type_can_carry_instance_state``)
            # applies, closing the producer/consumer disagreement that let an unslotted
            # subclass save a permanently-UNVERIFIABLE artifact. Plain ``collections.namedtuple``
            # / ``typing.NamedTuple`` / ``__slots__=()`` subclasses / torch structseq (no
            # instance ``__dict__``) stay admitted.
            return False, f"namedtuple_instance_state:{type(value).__qualname__}"
        for field_name in fields:
            proved, reason = _prove_runnable_output_lossless(
                getattr(value, field_name), _depth=_depth + 1, _in_progress=_in_progress
            )
            if not proved:
                return False, reason
        return True, ""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            proved, reason = _prove_runnable_output_lossless(
                getattr(value, field.name), _depth=_depth + 1, _in_progress=_in_progress
            )
            if not proved:
                return False, reason
        return True, ""
    if isinstance(value, Mapping):
        recon = _mapping_reconstruction(value)
        if recon is None:
            return False, f"unreconstructable_mapping:{type(value).__qualname__}"
        if mapping_extra_instance_state(value):
            # r37 3-ADJ-2: OrderedDict/defaultdict instances can carry attributes
            # the key/value rebuild drops -- refuse at save.
            return False, f"mapping_instance_state:{type(value).__qualname__}"
        for key in value.keys():
            if not isinstance(key, (str, int)):
                return False, f"unsupported_mapping_key:{type(key).__qualname__}"
            proved, reason = _prove_runnable_output_lossless(
                value[key], _depth=_depth + 1, _in_progress=_in_progress
            )
            if not proved:
                return False, reason
        return True, ""
    if isinstance(value, (list, tuple)):
        if type(value) not in {list, tuple}:
            # An exotic sequence subclass cannot be rebuilt with its exact type.
            return False, f"unsupported_sequence_type:{type(value).__qualname__}"
        items = _iter_sequence_items(value)
        if items is None:
            return False, f"opaque_sequence:{type(value).__qualname__}"
        for _index, item in items:
            proved, reason = _prove_runnable_output_lossless(
                item, _depth=_depth + 1, _in_progress=_in_progress
            )
            if not proved:
                return False, reason
        return True, ""
    return False, f"opaque_leaf:{type(value).__qualname__}"


def runnable_output_losslessness(
    out: Any,
    output_entries: Sequence[tuple[torch.Tensor, tuple[OutputPathComponent, ...], Any]],
) -> dict[str, Any]:
    """Build the explicit model-output traversal proof result (r35 I1, decision B).

    Combines the positive structural losslessness proof with the tensor-leaf /
    typed-path bijection over the walked output entries: duplicate paths, a
    spec/leaf-count disagreement, or any fallback traversal breaks the proof.

    Parameters
    ----------
    out:
        Raw model output object.
    output_entries:
        Entries yielded by :func:`_walk_output_tensors_with_paths` for ``out``.

    Returns
    -------
    dict[str, Any]
        ``{"lossless", "reason", "root_type", "used_fallback", "leaf_count",
        "duplicate_paths"}`` -- the single flag set the runnable producer
        consumes (no dual flags).
    """

    proved, reason = _prove_runnable_output_lossless(out)
    paths = [tuple(path) for _tensor, path, _spec in output_entries]
    duplicate_paths = len(paths) != len({repr(p) for p in paths})
    used_fallback = False
    if not isinstance(out, torch.Tensor) and not (
        _literal_value_supported(out) or isinstance(out, torch.Size)
    ):
        used_fallback = _try_build_container_spec(out) is None
    lossless = proved and not duplicate_paths and not used_fallback
    if not lossless and not reason:
        reason = "duplicate_output_paths" if duplicate_paths else "fallback_traversal"
    # r39 corr2_5: an EXPLICIT closed root kind, so the live provider never INFERS a bare
    # tensor from "one leaf, no spec, no path" -- an opaque set/frozenset/custom container the
    # traversal fell back on produces that SAME signature. ``bare_tensor_root`` is the single
    # positive fact the live bare-tensor fast path is gated on; a non-tensor opaque root is
    # ``opaque`` (never blessed), a proven container is ``supported_container``, and a
    # tensor-free literal/Size root is ``literal_only``.
    bare_tensor_root = isinstance(out, torch.Tensor) and not isinstance(out, torch.nn.Parameter)
    if bare_tensor_root:
        root_kind = "bare_tensor"
    elif not lossless:
        root_kind = "opaque"
    elif _literal_value_supported(out) or isinstance(out, torch.Size):
        root_kind = "literal_only"
    else:
        root_kind = "supported_container"
    return {
        "lossless": bool(lossless),
        "reason": reason if not lossless else "",
        "root_type": type(out).__qualname__,
        "root_kind": root_kind,
        "bare_tensor_root": bool(bare_tensor_root),
        "used_fallback": bool(used_fallback),
        "leaf_count": len(paths),
        "duplicate_paths": bool(duplicate_paths),
    }


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

    root_spec = _try_build_container_spec(out)
    if root_spec is None:
        if _literal_value_supported(out) or isinstance(out, torch.Size):
            return
        fallback_tensors = list(
            get_vars_of_type_from_obj(
                out, which_type=torch.Tensor, subclass_exceptions=[torch.nn.Parameter]
            )
        )
        if not fallback_tensors:
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
        for tensor in fallback_tensors:
            yield tensor, (), None
        return

    yield from _walk_supported_output_container(out, root_spec=root_spec, path=())
