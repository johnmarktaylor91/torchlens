"""Input and label record helpers for the RUN stage.

Moved verbatim from ``torchlens._capture.trace`` by the capture seam.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._ir.container_registry import ModelSite, Phase, Role, walk_container
from .._runnable.seam import runnable_trace_state
from ..backends.registry import (
    get_backend_spec,
)

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
from .._data_classes._lookup_keys import _give_user_feedback_about_lookup_key
from ._forward_helpers import _backend_name_for_trace


def _select_raw_indexes_by_retention_predicate(
    self: "Trace", predicate: "Callable[[Any], bool]"
) -> list[int]:
    """Evaluate a bare retention predicate per finalized op, strict bool.

    Bare retention predicates (the ``save_grads=`` callable spelling) are
    evaluated per finalized op with a layer-like ctx and a strict-bool return
    -- mirroring the halt-slot bool contract so a truthy tensor cannot
    silently select everything.

    Parameters
    ----------
    self:
        Finished trace whose ``layer_list`` supplies the candidate ops.
    predicate:
        User callable evaluated per op record.

    Returns
    -------
    list[int]
        Sorted unique raw indexes of the ops the predicate selected.

    Raises
    ------
    PredicateError
        If the predicate returns a non-bool (code ``predicate_return_invalid``).
    """

    from ..recording.exceptions import PredicateError

    selected_raw_indexes: set[int] = set()
    for layer_entry in self._layer_list:
        decision = predicate(layer_entry)
        if not isinstance(decision, bool):
            raise PredicateError(
                "save_grads predicate must return bool. "
                "Remedy: return True or False from the save_grads predicate.",
                ctx=layer_entry,
                result=decision,
                code="predicate_return_invalid",
            )
        if decision:
            selected_raw_indexes.add(layer_entry.raw_index)
    return sorted(selected_raw_indexes)


def _get_op_nums_from_user_labels(
    self: "Trace", which_layers: str | list[str | int] | None
) -> list[int] | str:
    """Resolve user-provided layer identifiers to internal raw_index values.

    Supports exact key match, substring match across all lookup keys, and the
    special sentinel ``"all"`` (which ops through as-is).  Returns sorted
    unique raw operation numbers for refresh projection.
    """
    if which_layers == "all":
        return which_layers
    elif which_layers in [None, "none", "None", "NONE", []]:
        return []

    from .._selection.selectors import BaseSelector

    if isinstance(which_layers, BaseSelector):
        return sorted(
            {
                site.raw_index
                for site in self._resolve_sites(
                    which_layers,
                    strict=False,
                    max_fanout=max(1, len(self._layer_list)),
                )
            }
        )

    if callable(which_layers):
        return _select_raw_indexes_by_retention_predicate(self, which_layers)

    if not isinstance(which_layers, list):
        which_layers = [which_layers]  # type: ignore[list-item]
    raw_layer_nums_to_save: set[int] = set()
    for layer_key in which_layers:
        if isinstance(layer_key, BaseSelector):
            raw_layer_nums_to_save.update(
                site.raw_index
                for site in self._resolve_sites(
                    layer_key,
                    strict=False,
                    max_fanout=max(1, len(self._layer_list)),
                )
            )
            continue
        if isinstance(layer_key, str) and ":" not in layer_key:
            matching_layer_passes = [
                layer_entry
                for layer_entry in self._layer_list
                if layer_key in {layer_entry.layer_label, layer_entry.short_layer_label}
            ]
            if matching_layer_passes:
                raw_layer_nums_to_save.update(
                    layer_entry.raw_index for layer_entry in matching_layer_passes
                )
                continue
        if layer_key in self._lookup_keys_to_layer_num_dict:
            raw_layer_nums_to_save.add(self._lookup_keys_to_layer_num_dict[layer_key])
            continue

        keys_with_substr = [key for key in self._layer_dict_all_keys if str(layer_key) in str(key)]
        if len(keys_with_substr) > 0:
            for key in keys_with_substr:
                raw_layer_nums_to_save.add(self._layer_dict_all_keys[key].raw_index)
            continue

        _give_user_feedback_about_lookup_key(self, layer_key, "query_multiple")

    raw_layer_nums_to_save = sorted(raw_layer_nums_to_save)  # type: ignore[assignment]
    return raw_layer_nums_to_save  # type: ignore[return-value]


def _register_model_input_container_snapshots(
    trace: "Trace",
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
) -> None:
    """Register top-level model input containers before forward invocation.

    Parameters
    ----------
    trace:
        Active trace.
    input_args:
        Normalized positional model inputs.
    input_kwargs:
        Normalized keyword model inputs.
    """

    if not getattr(trace, "_capture_container_structure", False):
        return
    capability = get_backend_spec(
        str(_backend_name_for_trace(trace))
    ).capabilities.input_container_structure
    if capability == "none":
        return
    registry = trace._wrapper_runtime_ws.container_registry
    first_spec = None
    for index, arg in enumerate(input_args):
        result = walk_container(arg, role=Role.MODEL_INPUT, capability=capability)
        if result is None:
            continue
        if first_spec is None:
            first_spec = result.spec
        registry.register_snapshot(
            arg,
            site=ModelSite(model_ref="self:1", position=("arg", index)),
            role=Role.MODEL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=0,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
        registry.register_snapshot(
            arg,
            site=ModelSite(model_ref="self:1", position=("arg", index)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=0,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
    for key, value in input_kwargs.items():
        result = walk_container(value, role=Role.MODEL_INPUT, capability=capability)
        if result is None:
            continue
        if first_spec is None:
            first_spec = result.spec
        registry.register_snapshot(
            value,
            site=ModelSite(model_ref="self:1", position=("kwarg", key)),
            role=Role.MODEL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=0,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
        registry.register_snapshot(
            value,
            site=ModelSite(model_ref="self:1", position=("kwarg", key)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=0,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
    if first_spec is not None:
        trace.__dict__["_provenance_input_structure"] = first_spec


_OPAQUE_INPUT_LEAF = object()
"""Sentinel marking a non-tensor input subtree that cannot be witnessed.

Recorded in place of the children under a mapping key that is not representable
in the frozen literal grammar. The runnable producer treats it as an opaque
(value-free) leaf, so the run honestly downgrades to ``UNVERIFIABLE`` instead of
silently skipping the subtree.
"""


def _record_runnable_input_literal_leaves(
    trace: "Trace",
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
) -> None:
    """Stash capture-time non-tensor model-input leaves for runnable honesty.

    A sparse runnable descriptor replays the *recorded taken-path* DAG, which is
    only valid for the recorded inputs. Non-tensor Python inputs
    (``bool``/``int``/``float``/``str``/``None`` literal leaves) can steer
    Python-level control flow that TorchLens never observes as an op, so a
    changed non-tensor input can silently make the recorded path wrong. TorchLens
    binds only tensor input leaves at run time, so without this record a changed
    non-tensor input is invisible and the run would falsely report a verified,
    attested -- but numerically wrong -- result.

    This records the model-boundary non-tensor leaves (site position, container
    path, and immutable literal value) so the sparse producer can witness them
    and the runnable executor can diverge on a changed non-tensor input instead
    of silently replaying the recorded path. It runs only when replay templates
    are captured (the runnable prerequisite), touches no tensors, and stores an
    in-memory list consumed by the producer at save time.

    Parameters
    ----------
    trace:
        Active trace.
    input_args:
        Normalized positional model inputs.
    input_kwargs:
        Normalized keyword model inputs.
    """

    if not bool(getattr(trace, "is_replayable", False)):
        return

    from torchlens._input_walk import tagged_mapping_key_component, walk_input_boundary
    from torchlens._io.runnable import EMPTY_CONTAINER_PATH_MARKER

    leaves: list[tuple[object, tuple[str | int, ...], Any]] = []

    def _walk_site(position: object, value: Any) -> None:
        """Record one boundary site's non-tensor leaves through the shared traversal.

        Container dispatch is single-sourced in ``torchlens._input_walk`` (r65
        Cluster Y): this walker only declares WHAT it records. A mapping child under
        a literal-grammar key is recorded at its tagged path (bool keys stay distinct
        from equal-valued int keys in the leaf-path set); a child under a
        NON-representable key (enum, object, bytes, ...) cannot be
        re-derived at run time, so the whole subtree is recorded as one OPAQUE marker
        leaf -- downgrading witness coverage to UNVERIFIABLE rather than silently
        dropping it (a silently skipped leaf under an exotic key is the false-VERIFIED
        money bug this walker exists to prevent). An EMPTY container adds no child
        leaf, so it is witnessed by a synthetic marker leaf carrying its KIND at
        ``(*path, EMPTY_CONTAINER_PATH_MARKER)`` so an added/removed/kind-changed
        empty container (which can steer ``'flag' in d`` / ``if not lst`` control
        flow) diverges instead of silently replaying the recorded path. r42 corr1_2:
        dataclass containers descend by DECLARED FIELD with the same leaf vocabulary,
        so a tensor-only dataclass input records ZERO opaque leaves (stays fully
        witnessable -> VERIFIED) while a genuinely-opaque field still surfaces.
        """

        walk_input_boundary(
            value,
            (),
            key_component=tagged_mapping_key_component,
            on_leaf=lambda leaf, p: leaves.append((position, p, leaf)),
            on_empty_container=lambda kind, p: leaves.append(
                (position, (*p, EMPTY_CONTAINER_PATH_MARKER), kind)
            ),
            on_opaque_key_subtree=lambda _child, p: leaves.append(
                (position, p, _OPAQUE_INPUT_LEAF)
            ),
        )

    for index, arg in enumerate(input_args):
        _walk_site(("arg", index), arg)
    for key, value in input_kwargs.items():
        _walk_site(("kwarg", key), value)

    if leaves:
        runnable_trace_state(trace).input_nontensor_leaves = tuple(leaves)


def _record_runnable_input_structure(
    trace: "Trace",
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
) -> None:
    """Stash ONE per-site input-boundary structure snapshot for runnable honesty (r67 C2).

    One traversal per normalized model-input site (the snapshot spine in
    ``torchlens._input_walk``) records the complete site set/arity and every nested node
    -- kind, exact ``(module, qualname)`` type, declared child schema, ordered
    type-strict-encoded mapping keys, registered aux, and the instance-state proof. The
    runnable producer persists the records as REQUIRED structure facts (positive proof:
    a site without a snapshot, or a snapshot carrying a refusal, refuses the runnable
    save); the executor re-derives the runtime snapshot with the SAME function and
    diverges on any node mismatch. Analysis captures are unaffected.
    """

    if not bool(getattr(trace, "is_replayable", False)):
        return

    from torchlens._input_walk import snapshot_input_boundary

    snapshots: list[dict[str, Any]] = []
    for index, arg in enumerate(input_args):
        record = snapshot_input_boundary(arg)
        record["position"] = ["arg", index]
        snapshots.append(record)
    for key, value in input_kwargs.items():
        record = snapshot_input_boundary(value)
        record["position"] = ["kwarg", str(key)]
        snapshots.append(record)
    runnable_trace_state(trace).input_structure = tuple(snapshots)


def _record_runnable_input_tensor_sites(
    trace: "Trace",
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
) -> None:
    """Index model-input TENSOR leaves by object identity for metadata-read witnessing.

    A Python-level metadata predicate read on a model input (``x.is_contiguous()`` /
    ``x.stride()`` / ``x.requires_grad``) can steer control flow that TorchLens never
    observes as an op: the input contract checks only shape+dtype, so a same-shape
    runtime input differing in layout or grad flag would silently replay the wrong
    recorded path. The completeness-witness scoped patch observes such reads during
    the runnable forward; this map lets it attribute a read RECEIVER back to its
    model-boundary site (position, container path) so the producer can witness the
    read fact and the executor can diverge on a mismatched runtime input. Keys are
    ``id(tensor)`` -- stable for the forward's duration because ``input_args`` /
    ``input_kwargs`` hold strong references until the capture completes. It runs only
    for intervention-ready captures and stores no tensors.

    Parameters
    ----------
    trace:
        Active trace.
    input_args:
        Normalized positional model inputs.
    input_kwargs:
        Normalized keyword model inputs.
    """

    if not bool(getattr(trace, "is_replayable", False)):
        return

    from torchlens._input_walk import raw_mapping_key_component, walk_input_boundary

    sites: dict[int, tuple[object, tuple[str | int, ...]]] = {}
    # (tensor, site) leaves so the completeness witness can additionally index model-input
    # leaves by STORAGE identity (r31): a metadata read routed through a ``.data`` / ``.detach()``
    # alias shares the leaf's storage but is a distinct object the id map above misses.
    tensor_leaves: list[tuple[Any, tuple[object, tuple[str | int, ...]]]] = []

    def _walk_site(position: object, value: Any) -> None:
        """Index one boundary site's tensor leaves by identity through the shared traversal.

        Container dispatch is single-sourced in ``torchlens._input_walk`` (r65
        Cluster Y), so this walker descends EXACTLY the container set the literal
        walker descends -- including dataclasses (the r64 Finding-1 false-VERIFIED: a
        missing dataclass branch here recorded no metadata witness for
        ``box.x.is_contiguous()`` on a dataclass input field, so a same-value
        non-contiguous twin replayed the captured branch as VERIFIED). Mapping
        children are indexed under EVERY key with the RAW key component (the declared
        dual vocabulary, residual R6): a fact site whose path carries a
        non-representable key simply fails literal encoding at witness time and is
        dropped, and the literal-leaf walker independently records such a subtree as
        an OPAQUE leaf that downgrades the run to UNVERIFIABLE -- so a dropped
        metadata fact under an exotic key can never yield a false VERIFIED.
        """

        def _index_tensor(tensor: Any, path: tuple[Any, ...]) -> None:
            """Record one tensor leaf's identity-keyed site."""

            site = (position, path)
            sites[id(tensor)] = site
            tensor_leaves.append((tensor, site))

        walk_input_boundary(
            value, (), key_component=raw_mapping_key_component, on_tensor=_index_tensor
        )

    for index, arg in enumerate(input_args):
        _walk_site(("arg", index), arg)
    for key, value in input_kwargs.items():
        _walk_site(("kwarg", key), value)

    if sites:
        runnable_trace_state(trace).input_tensor_sites = sites
        from ..backends.torch._completeness_boundaries import record_runnable_input_storage_sites

        record_runnable_input_storage_sites(trace, tensor_leaves)


def _record_runnable_module_training_modes(trace: "Trace", model: Any) -> None:
    """Stash the capture-time per-module ``training`` mode for runnable honesty.

    ``self.training`` is module state that is NOT part of the ``state_dict`` and is not a
    model input, yet it steers mode-sensitive ops (BatchNorm running-stats vs batch-stats,
    Dropout on/off). The runnable VERIFIED oracle is a *fresh instance in the captured mode*
    on the given inputs, so the captured mode is DECLARED state the replay reproduces.
    Recording it (per submodule -- submodules can differ) lets the producer declare the mode
    as a witness fact; a mode-sensitive op replayed without a recorded mode fact is downgraded
    to UNVERIFIABLE (fail closed). It runs for every capture (D18 widened the former
    intervention-ready gate so the live refresh projector's mode-claim belt holds on the
    default path; the widening is verdict-inert on the sparse side because the runnable
    producer refuses non-intervention-ready captures outright), touches no tensors, and
    stores an in-memory map consumed by the producer at save time and by the projector belt.

    Parameters
    ----------
    trace:
        Active trace.
    model:
        The prepared source model whose per-module ``training`` flags are recorded.
    """

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    modes: dict[str, bool] = {}
    try:
        for name, module in named_modules():
            address = name or "self"
            modes[address] = bool(getattr(module, "training", False))
    except (AttributeError, TypeError):
        return
    if modes:
        runnable_trace_state(trace).module_training_modes = modes
