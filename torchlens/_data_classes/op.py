"""Op: per-operation metadata for a single invocation of a layer.

Each Op records everything about one tensor operation in the
forward pass: the output tensor itself, the function that produced it,
its parents/children in the computation graph, module containment,
parameter usage, timing, RNG state, and more.

For recurrent models, the same "layer" may execute multiple times; each
execution is a separate Op with a distinct ``pass_index``.  The
aggregate view across ops is provided by :class:`Layer`.

Field categories (matching the LAYER_PASS_LOG_FIELD_ORDER in constants.py):

1. **General info** - raw/final labels, operation numbering, back-reference
   to the owning Trace.
2. **Label info** - human-readable labels in various formats (with/without
   pass qualifier, short form, etc.).
3. **Saved tensor info** - the tensor contents, shape, dtype, size, device
   transfer settings, out transform, and function arguments.
4. **Child tensor variations** - tracks per-child input values for
   validation replay (``out_versions_by_child`` stores RAW values
   because validation compares against ``saved_args``).
5. **Gradient info** - grad tensor and metadata (always stored as a
   detached SNAPSHOT via ``log_tensor_grad``, even under
   ``save_mode="reference"``/``"view"`` -- autograd may accumulate into
   the observed gradient in place, so an alias would silently rewrite
   the recorded value).
6. **Function call info** - the applied function, call stack, timing,
   FLOPs, RNG state, arg metadata, grad_fn_handle, inplace flag.
7. **Param info** - which parameters were used, their shapes and sizes.
8. **Equivalence info** - loop-detection equivalence type and groups.
9. **Graph info** - parent/child/sibling/spouse edges, input/output
   ancestry, distances, buffer/internal-init status.
10. **Conditional info** - boolean branching metadata.
11. **Module info** - module entry/exit tracking, nesting depth,
    bottom-level submodule output status.
"""

import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch

from .._constants import ARG_EXPRESSIONS_FIELD, LAYER_PASS_LOG_FIELD_ORDER
from .._errors import (
    ArgumentTypeError,
    InvalidArgumentError,
    PayloadUnavailableError,
    RecordBindingError,
)
from .._io import (
    TLSPEC_VERSION,
    FieldPolicy,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from .._ir.refs import DeviceRef, DtypeRef
from .._quantities import Bytes, Duration, Flops, Macs, as_bytes, as_duration, as_flops
from .._selection.selection import _SelectionOperand
from .._trace_core.fact_blocks import OP_FACT_FIELDS
from .._trace_core.groups import GroupRef
from .._trace_core.op_store import (
    _CSR,
    _FACT,
    _MISSING,
    DetachedOpStore,
    OpStoreLayout,
    PooledCell,
)
from .._trace_core.relation_views import (
    OP_BITSET_VIEW_FIELDS,
    OP_DATAFLOW_FIELDS,
    OP_FROZENSET_VIEW_FIELDS,
    OP_GROUP_VIEW_FIELDS,
    OP_TUPLE_VIEW_FIELDS,
    materialize_dataflow_view,
)
from .._trace_state import TraceState
from .._utils._record_group_table import OP_STORAGE_ATTRS, op_attr, record_member
from .._utils.arg_handling import copy_arg_tree
from .._utils.facade import refuse_removed_member
from .._utils.tensor_utils import (
    concatenate_batch_tensors,
    get_memory_amount_from_metadata,
)
from ..intervention._types import (
    LAYER_PASS_LOG_FIELD_FORK_POLICY,
    EdgeUseRecord,
    FunctionRegistryKey,
)
from ..intervention.errors import DirectActivationWriteWarning
from . import _doors, _familiar_names, _labels, _op_cost, _op_relations, _op_save, _removed_members
from ._accessor_base import Accessor
from ._backend_capability_guards import raise_if_no_backward_capture
from ._layer_accessors import OpAccessor
from ._op_aten import OpAtenMixin
from ._op_container_census import (  # noqa: F401  (re-exported for the pooling tests)
    _census_container_cells,
    _count_container_ids,
)
from ._op_dedup import (  # noqa: F401  (re-exported for backends.torch importers)
    _dedup_cached_identity_out,
    _dedup_saved_activation_out,
)
from ._op_out_read import (
    _out_read_state,
    _refuse_read_without_trace,
    _synthesized_out_read_due,
    _validate_reference_out_not_mutated,
)
from ._op_row_release import (  # noqa: F401  (re-exported for cleanup and backend importers)
    _ROW_RELEASE_PROPERTIES,
    bind_op_facade,
    register_row_release_property,
    release_op_row,
)
from ._op_save import (  # noqa: F401  (re-exported for backends and postprocess importers)
    _dtype_or_none,
    _effective_activation_save_mode,
    _memory_or_none,
    _shape_or_none,
    _stamp_reference_out,
)
from ._op_transforms import (  # noqa: F401  (re-exported for backends.torch importers)
    apply_transform,
    train_mode_tripwire_armed,
    validate_streaming_transform_output,
    validate_train_mode_transform_output,
)
from ._record_groups import OpControlFlow, OpCost
from ._record_groups_autograd import OpAutograd
from ._record_groups_call import OpCall
from ._record_groups_graph import OpGraph
from ._record_groups_module_context import OpModuleContext
from ._record_groups_payload import OpPayload
from ._record_groups_source import OpSource
from ._state_adapter import state_items, state_restore
from .container import (
    container_from_op,
    containers_from_op,
    input_containers_from_op,
    output_containers_from_op,
)
from .field_policy import (
    build_record_field_policy_table,
    default_fill_state_from_policy,
    fork_policy_from_policy,
    portable_state_spec_from_policy,
)

_LAYER_PASS_LOG_FIELD_ORDER_SET = frozenset(LAYER_PASS_LOG_FIELD_ORDER)
_DIRECT_WRITE_GUARDED_FIELDS = frozenset(
    {
        "out",
        "transformed_out",
        "grad",
        "transformed_grad",
        "edits",
    }
)
# ``Op.__getattribute__``/``__setattr__`` run on EVERY attribute touch during
# capture, so the plain ``object.__getattribute__`` global lookup (globals miss
# -> builtins hit -> type attribute resolution) was itself measurable. Bind the
# unbound slot accessors once at import.
_object_getattribute = object.__getattribute__


_object_setattr = object.__setattr__
# Fields whose reads go through the lazy-materialization path in
# ``Op.__getattribute__``; every other name short-circuits straight to the slot.
_LAZY_READ_FIELDS = frozenset({"grad", "out"})
# ``equivalent_ops``/``recurrent_ops`` were copy-on-read fields here; the
# live group views (shared ``GroupRef`` cells, cached immutable ``_GroupViewField``
# reads) make sharing alias-safe without per-read copies.
_INTERCEPTED_READ_FIELDS = _LAZY_READ_FIELDS
_LAYER_PASS_LOG_DEFAULT_FILL: dict[str, Any] = {
    "_source_trace_ref": None,
    "out_ref": None,
    "grad_ref": None,
    "_pending_blob_id": None,
    "_pending_transformed_out_blob_id": None,
    "_pending_grad_blob_id": None,
    "_pending_transformed_grad_blob_id": None,
    "annotations": {},
    "is_constant": False,
    "autograd_memory": None,
    "num_autograd_tensors": None,
    "bytes_delta_at_call": None,
    "bytes_peak_at_call": None,
    "transformed_out": None,
    "transformed_out_shape": None,
    "transformed_out_dtype": None,
    "dtype_ref": None,
    "device_ref": None,
    "backend_address": None,
    "resolver_status": "resolved",
    "input_was_parameter": False,
    "transformed_activation_memory": None,
    "visualizer_path": None,
    "transformed_grad": None,
    "transformed_grad_shape": None,
    "transformed_grad_dtype": None,
    "transformed_gradient_memory": None,
    "func_call_id": None,
    "func_id": None,
    "container_path": (),
    "multi_output_name": None,
    "is_edited": False,
    "edits": [],
    "container_spec": None,
    "args_template": None,
    "kwargs_template": None,
    "_edge_uses": [],
    "edge_substitutions": {},
    "edge_replacement_stamps": {},
    "var_names": [],
    "is_orphan": False,
    "_address_normalized": None,
    "_construction_done": True,
}
# Typed container defaults for every non-Optional container field in
# `LAYER_PASS_LOG_FIELD_ORDER` (same fix as `trace._MODEL_LOG_CONTAINER_DEFAULTS`):
# an absent legacy field restores as its declared container, not ``None``, and
# plain builtin types let ``coerce_container_typed_state`` (``Op.__setstate__``)
# repair a present-but-wrong-typed legacy value.
_LAYER_PASS_LOG_CONTAINER_DEFAULTS: dict[str, Any] = {
    "lookup_keys": [],
    "annotations": {},
    "shape": (),
    "out_versions_by_child": {},
    "code_context": [],
    "func_rng_states": {},
    "func_autocast_state": {},
    "arg_names": (),
    "non_tensor_pos_args": [],
    "non_tensor_kwargs": {},
    "func_non_tensor_args": [],
    "transform_chain": (),
    "transform_config": {},
    "unattributed_tensor_args": (),
    "dropped_edge_tensor_args": (),
    # Relation view fields: the declared restore type is the
    # IMMUTABLE view — ``coerce_container_typed_state`` normalizes legacy
    # list/set state to tuple/frozenset on load, so loaded traces present the
    # same immutable relation surface as live finished captures.
    "parent_params": (),
    "_param_barcodes": (),
    "parent_param_ops": {},
    "_param_logs": (),
    "param_shapes": [],
    "equivalent_ops": frozenset(),
    "recurrent_ops": (),
    "parents": (),
    "parent_arg_positions": {},
    "root_ancestors": frozenset(),
    "children": (),
    "input_ancestors": frozenset(),
    "output_descendants": frozenset(),
    "internal_source_parents": (),
    "internal_source_ancestors": frozenset(),
    "in_conditionals": (),
    "conditional_branch_stack": (),
    "conditional_entry_children": (),
    "conditional_then_children": (),
    "conditional_elif_children": {},
    "conditional_else_children": (),
    "conditional_arm_children": {},
    "modules": (),
    "module_call_stack": (),
    "input_to_module_calls": (),
    "module_entry_arg_keys": {},
    "output_of_modules": (),
    "output_of_module_calls": (),
    "func_config": {},
}
_LAYER_PASS_LOG_DEFAULT_FILL = {
    **dict.fromkeys(LAYER_PASS_LOG_FIELD_ORDER),
    **_LAYER_PASS_LOG_CONTAINER_DEFAULTS,
    **_LAYER_PASS_LOG_DEFAULT_FILL,
}
_OP_PROPERTY_BACKED_FIELD_NAMES = frozenset(
    {
        "source_trace",
        "input_ops",
        "input_activations",
        "input_shapes",
        "input_dtypes",
        "input_memory",
        "num_inputs",
        "is_in_conditional_body",
        "is_internally_initialized",
        "raw_label",
        ARG_EXPRESSIONS_FIELD,
    }
)
_OP_DYNAMIC_SLOT_NAMES = (
    "_source_trace_ref",
    "out_ref",
    "grad_ref",
    "_pending_blob_id",
    "_pending_transformed_out_blob_id",
    "_pending_grad_blob_id",
    "_pending_transformed_grad_blob_id",
    "_grad_records",
    "_facets_cache",
    "_receptive_field_cache",
    "_projective_field_cache",
    "_arg_expressions_cache",
    "_is_in_conditional_body",
    "_construction_done",
)
_OP_SLOT_NAMES = tuple(
    dict.fromkeys(
        [
            *(
                field_name
                for field_name in LAYER_PASS_LOG_FIELD_ORDER
                if field_name not in _OP_PROPERTY_BACKED_FIELD_NAMES
            ),
            *_OP_DYNAMIC_SLOT_NAMES,
        ]
    )
)
# The seam: ``_OP_SLOT_NAMES`` is no longer a ``__slots__`` tuple but the
# declared stored-field universe of the Op row store. One shared layout maps
# each stored field to its column id; ``Op`` itself carries only
# ``(_core, _row)`` and one generated data descriptor per stored field (see
# ``_OpField`` / ``_install_op_field_descriptors`` at the bottom of this
# module). All storage magic (``_slot``, ``_internal_set``, the ancestor
# bitset overlays in ``backends/torch/ops.py``, ``state_items`` /
# ``state_restore``) already speaks the descriptor protocol, so behavior is
# unchanged while the physical cells live in per-trace columns.
_OP_STORE_LAYOUT = OpStoreLayout(_OP_SLOT_NAMES)
#: ``(persisted field name, attribute that holds it)`` in declared order; a
#: concern-group member keeps its persisted name under a private attribute.
_OP_STATE_ATTRS = tuple((name, OP_STORAGE_ATTRS.get(name, name)) for name in _OP_SLOT_NAMES)
# Bulk-ingress seed constants: the converted-at-construction fields and the
# layout ids of the cells `__init__` overrides after the one-pass dict seed.
_INIT_BYTES_FIELDS = (
    "activation_memory",
    "transformed_activation_memory",
    "autograd_memory",
    "bytes_delta_at_call",
    "bytes_peak_at_call",
    "gradient_memory",
    "transformed_gradient_memory",
)
_FID_SOURCE_TRACE_REF = _OP_STORE_LAYOUT.fid_by_name["_source_trace_ref"]
_FID_IS_IN_CONDITIONAL_BODY = _OP_STORE_LAYOUT.fid_by_name["_is_in_conditional_body"]
_FID_GRAD_RECORDS = _OP_STORE_LAYOUT.fid_by_name["_grad_records"]
_FID_CONSTRUCTION_DONE = _OP_STORE_LAYOUT.fid_by_name["_construction_done"]
_FIDS_INIT_NONE = tuple(
    _OP_STORE_LAYOUT.fid_by_name[field_name]
    for field_name in (
        "out_ref",
        "grad_ref",
        "_pending_blob_id",
        "_pending_transformed_out_blob_id",
        "_pending_grad_blob_id",
        "_pending_transformed_grad_blob_id",
    )
)

# ---------------------------------------------------------------------------
# Post-capture metadata pooling (RAM)
#
# A finished graph stores the SAME immutable metadata value over and over: one
# ``"torch.float32"`` dtype name per op, one ``Bytes(0)`` per unused memory
# counter, the same module address in every op of a module, the same ancestor
# label in hundreds of ancestor sets.  Each of those is a separate Python
# object, so per-node footprint grows with (#ops x #repeated facts) instead of
# with the number of DISTINCT facts.
#
# ``_compaction.compact_op_metadata`` runs one pass at the freeze seam and
# replaces each such value with a single pooled instance.  The pool is local to
# that pass and dropped afterwards, so nothing leaks process-wide the way
# ``sys.intern`` would.
#
# The pass is VALUE-PRESERVING by construction:
#   * only exact-class IMMUTABLE values are pooled (str/bytes, the frozen
#     DtypeRef/DeviceRef refs, the int/float quantity subclasses, and tuples /
#     frozensets built purely out of those).  Sharing an immutable object can
#     never alias-corrupt an owner.
#   * mutable containers (list/set/dict) keep their own identity and
#     mutability; only their ELEMENTS are swapped for pooled equals.
#   * anything else -- tensors, Ops, arbitrary user objects -- is returned
#     untouched and never enters the pool.
# ---------------------------------------------------------------------------

# Exact immutable classes safe to share (exact class keeps a pooled value's type;
# dtype/device are process value objects, e.g. in ``func_autocast_state``). Only a
# ZERO ``Duration`` pools (the fixed placeholder of rows that ran nothing): measured
# timings match only by chance, and sharing them made the saved graph's identity
# (pickle memo back-references, ``metadata.pkl`` bytes) depend on the clock.
_POOLED_CLASSES = frozenset(
    {str, bytes, DtypeRef, DeviceRef, Bytes, Flops, Macs, Duration, torch.dtype, torch.device}
)
_NONE_POOL_KEY = (type(None), None)
# Slots skipped by the pass: tensor payloads, replay/attestation raw values,
# live handles, and lazily rebuilt caches.  Skipping is a cost/robustness
# choice, not a correctness one -- the pooling helpers already refuse every
# value in them.
_UNPOOLED_SLOTS = frozenset(
    {
        "out",
        "transformed_out",
        "grad",
        "transformed_grad",
        "saved_args",
        "saved_kwargs",
        "args_template",
        "kwargs_template",
        "out_versions_by_child",
        "func_rng_states",
        "activation_transform",
        "edits",
        # Per-op UNIQUE injected-op identity record (tlspec v9 entry-dark
        # slot): a mutable mapping distinct per injected op, never a shared
        # internable value. None on every model op until writes it.
        "injection_provenance",
        "parent_params",
        "_param_logs",
        "_grad_records",
        "func",
        "grad_fn",
        "grad_fn_handle",
        "out_ref",
        "grad_ref",
        "_source_trace_ref",
        "_facets_cache",
        "_receptive_field_cache",
        "_projective_field_cache",
        "_arg_expressions_cache",
        # One canonical set object is shared by every Op of an equivalence class,
        # so walking it per Op would re-pool the SAME N labels N times (a 512-step
        # loop: 524k member visits for 4 distinct groups) and would mutate one
        # shared container from N owners.  Nothing is lost: a group's members are
        # the very ``op.label`` strings the rename pass read out of
        # ``_raw_to_final_op_labels``, and those slots ARE pooled, so no distinct
        # duplicate of a label survives this skip.  Pooling is a cost choice, not
        # a correctness one (see the pooling block above).
        "equivalent_ops",
        # Same sharing argument: one canonical list per recurrence group, whose
        # members are final op labels already pooled through their own slots.
        "recurrent_ops",
    }
)
# Attribute names (a concern-group member reads under its private storage name).
_POOLED_SLOTS = tuple(
    OP_STORAGE_ATTRS.get(name, name) for name in _OP_SLOT_NAMES if name not in _UNPOOLED_SLOTS
)
# Depth of container nesting the pass descends into.  Every field that carries
# real repetition (ancestor sets, label lists, config dicts) is at depth 0-1.
_POOL_MAX_DEPTH = 2

# ---------------------------------------------------------------------------
# Mutable-container CELL pooling (the memory slice).
#
# The freeze-seam compaction additionally replaces whole mutable-container
# CELLS -- exact ``dict``/``list``/``set`` (and top-level ``defaultdict``)
# values whose full content is provably immutable AND either empty or
# repeated across cells -- with one shared ``PooledCell`` per distinct
# content. The facade descriptors hydrate a fresh exact-type container per
# row on first read and cache it back (``_FACT`` semantics), so per-row
# identity/mutation contracts are unchanged while an uninspected row retains
# no per-row container. Op-store cells pool only through the explicit field
# allowlist below (fields whose post-freeze lifecycle is read-or-reassign);
# kind-table cells (Module/ModuleCall/Param/...) pool generically -- their
# repetition (empty hook lists, identical ``custom_attributes``...) is the
# census-dominant tail. An in-store alias census guards every replacement:
# a container object reachable from more than one swept cell never pools.
# ---------------------------------------------------------------------------

#: Op-store fields sanctioned for whole-cell container pooling.
_POOLED_CONTAINER_FIELDS = frozenset(
    {
        "annotations",
        "edits",
        "var_names",
        "_grad_records",
        "out_versions_by_child",
        "func_rng_states",
        "conditional_elif_children",
        "conditional_arm_children",
        "func_autocast_state",
        "transform_config",
        "module_entry_arg_keys",
        "parent_param_ops",
        "parent_arg_positions",
    }
)

#: Exact cell classes the container pooling considers.
_MUTABLE_CELL_CLASSES = (dict, list, set)

#: ``defaultdict.default_factory`` values safe to share and rebuild.
_POOLABLE_DEFAULT_FACTORIES = frozenset({list, set, dict, int, tuple})

#: Nesting cap for container pool keys (cycles and pathological nesting
#: simply refuse to pool).
_CONTAINER_KEY_MAX_DEPTH = 4
_CONTAINER_KEY_MAX_MEMBERS = 1024


def _container_pool_key(
    value: Any,
    depth: int,
    visited_ids: list[int],
    member_budget: list[int] | None = None,
) -> Any:
    """Return an injective hashable content key for one mutable container.

    ``None`` means "do not pool": unknown member types, subclassed
    containers, non-builtin factories, or nesting past the cap. Immutable
    members key through :func:`_pool_key` (class-tagged, float-bit exact),
    so the key is injective exactly like the immutable pool's. Every
    visited mutable container's ``id`` lands in ``visited_ids`` for the
    alias guard.
    """

    if member_budget is None:
        member_budget = [_CONTAINER_KEY_MAX_MEMBERS]
    if depth > _CONTAINER_KEY_MAX_DEPTH or member_budget[0] <= 0:
        return None
    cls = value.__class__
    if cls is defaultdict:
        if depth:
            return None
        factory = value.default_factory
        if factory is not None and factory not in _POOLABLE_DEFAULT_FACTORIES:
            return None
        visited_ids.append(id(value))
        items = _dict_member_keys(value, depth, visited_ids, member_budget)
        return None if items is None else ("dd", factory, items)
    if cls is dict:
        visited_ids.append(id(value))
        items = _dict_member_keys(value, depth, visited_ids, member_budget)
        return None if items is None else ("d", items)
    if cls is list:
        visited_ids.append(id(value))
        member_keys = []
        for member in value:
            member_budget[0] -= 1
            if member_budget[0] < 0:
                return None
            member_key = _container_member_key(member, depth, visited_ids, member_budget)
            if member_key is None:
                return None
            member_keys.append(member_key)
        return ("l", tuple(member_keys))
    if cls is set:
        visited_ids.append(id(value))
        member_keys = []
        for member in value:
            member_budget[0] -= 1
            if member_budget[0] < 0:
                return None
            member_key = _pool_key(member)
            if member_key is None:
                return None
            member_keys.append(member_key)
        return ("s", frozenset(member_keys))
    return None


def _dict_member_keys(
    value: Any,
    depth: int,
    visited_ids: list[int],
    member_budget: list[int],
) -> Any:
    """Key the items of one dict-shaped container, or ``None`` to refuse."""

    items = []
    for key, member in value.items():
        member_budget[0] -= 1
        if member_budget[0] < 0:
            return None
        key_key = _pool_key(key)
        if key_key is None:
            return None
        member_key = _container_member_key(member, depth, visited_ids, member_budget)
        if member_key is None:
            return None
        items.append((key_key, member_key))
    return tuple(items)


def _container_member_key(
    member: Any,
    depth: int,
    visited_ids: list[int],
    member_budget: list[int],
) -> Any:
    """Key one container member: immutable leaf or nested exact container."""

    immutable_key = _pool_key(member)
    if immutable_key is not None:
        return ("i", immutable_key)
    if member.__class__ in _MUTABLE_CELL_CLASSES:
        nested = _container_pool_key(member, depth + 1, visited_ids, member_budget)
        return None if nested is None else ("m", nested)
    return None


def _pool_container_cells(
    stores: list[tuple[Any, Any]], container_pool: dict[Any, PooledCell]
) -> None:
    """Pool duplicate/empty immutable-content container cells across stores.

    Parameters
    ----------
    stores:
        ``(store, fids)`` pairs; ``fids`` is an iterable of sanctioned field
        ids or ``None`` for every field (kind tables). Only building-phase
        row stores participate (``rows_building()`` returning ``None`` skips
        the store) -- pooling always precedes the physical freeze.
    container_pool:
        Pass-local ``content key -> PooledCell`` table shared across stores,
        so equal content pools trace-wide.
    """

    census = _census_container_cells(stores)
    if census is None:
        return
    id_counts, cell_sites, singleton_sites = census
    # A cell whose own id is aliased is skipped before its key is built: that
    # id is the first one the key walk visits, so the alias guard would refuse
    # it anyway. Empty exact dict/list/set cells arrive with their key.
    candidates: list[tuple[Any, int, Any, Any]] = []
    key_counts: dict[Any, int] = {}
    for row_cells, fid, value, key in cell_sites:
        if id_counts[id(value)] > 1:
            continue
        if key is None:
            visited_ids: list[int] = []
            key = _container_pool_key(value, 0, visited_ids)
            if key is None:
                continue
            if any(id_counts[oid] > 1 for oid in visited_ids):
                continue
        candidates.append((row_cells, fid, value, key))
        key_counts[key] = key_counts.get(key, 0) + 1
    for row_cells, fid, value, key in candidates:
        # Empty containers always pool (all empties of a class share ONE
        # cell+prototype). Non-empty content needs >= 3 occurrences: a pooled
        # key retains 2 objects (cell + detached prototype), so pooling a
        # pair is object-neutral before any read and negative after.
        if value and key_counts[key] < 3:
            continue
        cell = container_pool.get(key)
        if cell is None:
            cell = container_pool[key] = PooledCell.from_value(value)
        row_cells[fid] = cell
    # Singleton-label compaction (kind tables only): a
    # one-element list holding exactly one str (per-record label/address
    # lists — distinct content per row, so PooledCell's >= 3 threshold never
    # reaches them) stores the bare element, registered on the store so
    # decode fires only for the exact registered object. Same alias census
    # as pooling: a list reachable from more than one swept cell never
    # compacts (breaking its mutation coupling is not sanctioned).
    for store, row_idx, fid, row_cells, value in singleton_sites:
        if (
            row_cells[fid] is value  # not already pooled above
            and len(value) == 1
            and value[0].__class__ is str
            and id_counts[id(value)] == 1
        ):
            element = value[0]
            row_cells[fid] = element
            store.register_compacted_singleton(row_idx, fid, element)


def _pool_key(value: Any) -> Any:
    """Return an injective hashable pool key for ``value``, or ``None``.

    Injectivity is what keeps the pass value-preserving: two values share a key
    if and only if they are indistinguishable. The class is always part of the
    key (so ``True`` never collapses into ``1``, nor ``Bytes(0)`` into ``0``),
    and floats key on their exact bit pattern (so ``-0.0`` never collapses into
    ``0.0``).

    Parameters
    ----------
    value:
        Candidate immutable value.

    Returns
    -------
    Any
        A hashable key, or ``None`` when ``value`` is not a poolable immutable.
    """

    cls = value.__class__
    if cls is str or cls is bytes or cls is int or cls is bool:
        return (cls, value)
    if cls is float:
        return (cls, value.hex())
    if cls in _POOLED_CLASSES and (cls is not Duration or not value):
        return (cls, value.hex()) if cls is Duration else (cls, value)
    if value is None:
        return _NONE_POOL_KEY
    if cls is tuple or cls is frozenset:
        member_keys = []
        for member in value:
            member_key = _pool_key(member)
            if member_key is None:
                return None
            member_keys.append(member_key)
        return (cls, tuple(member_keys) if cls is tuple else frozenset(member_keys))
    return None


def _pool_value(value: Any, pool: dict[Any, Any]) -> Any:
    """Return the pooled twin of an immutable ``value``, or ``value`` itself.

    Parameters
    ----------
    value:
        Candidate value from an Op slot or from inside one of its containers.
    pool:
        Pass-local ``pool key -> canonical instance`` table.

    Returns
    -------
    Any
        An object of the same exact class that is ``==`` to ``value``, or
        ``value`` unchanged when it is not a poolable immutable.
    """

    cls = value.__class__
    if cls in _POOLED_CLASSES and (cls is not Duration or not value):
        key = (cls, value.hex()) if cls is Duration else (cls, value)
    elif (cls is tuple or cls is frozenset) and value:
        key = _pool_key(value)
        if key is None:
            return value
    else:
        return value
    pooled = pool.get(key)
    if pooled is not None:
        return pooled
    if cls is tuple or cls is frozenset:
        # Rebuild so nested immutables are shared too; the rebuilt collection
        # is the canonical one from here on.
        value = cls(_pool_value(member, pool) for member in value)
    pool[key] = value
    return value


def _pool_container_members(container: Any, pool: dict[Any, Any], depth: int) -> None:
    """Swap poolable members of a mutable container for their pooled twins.

    The container object itself is never replaced, so its identity, class,
    mutability, and (for lists/dicts) ordering are preserved exactly.

    Parameters
    ----------
    container:
        A ``list``, ``set``, or mapping owned by one Op.
    pool:
        Pass-local pooling table.
    depth:
        Current nesting depth; recursion stops at ``_POOL_MAX_DEPTH``.
    """

    cls = container.__class__
    recurse = depth < _POOL_MAX_DEPTH
    pooled_classes = _POOLED_CLASSES
    pool_get = pool.get
    if cls is list:
        for index, member in enumerate(container):
            member_cls = member.__class__
            # Inline the common case (a pooled string) so the vast majority of
            # members never pay a Python call.
            if member_cls is str:
                key = (str, member)
                pooled = pool_get(key)
                if pooled is None:
                    pool[key] = member
                elif pooled is not member:
                    container[index] = pooled
            elif member_cls in pooled_classes or member_cls is tuple or member_cls is frozenset:
                pooled = _pool_value(member, pool)
                if pooled is not member:
                    container[index] = pooled
            elif recurse and (
                member_cls is list
                or member_cls is set
                or member_cls is dict
                or isinstance(member, dict)
            ):
                _pool_container_members(member, pool, depth + 1)
    elif cls is set:
        # Rebuilding through clear()/update() also drops the over-allocation
        # left behind by the discard/remove calls postprocessing makes.
        pooled_members = {_pool_value(member, pool) for member in container}
        container.clear()
        container.update(pooled_members)
    elif isinstance(container, dict):
        # Keys are dict-literal constants in the producers and are already
        # shared by the compiler, so only values are pooled -- which also keeps
        # insertion order untouched.  Rebinding an existing key never resizes a
        # dict, so iterating ``items()`` directly is safe.
        for key, member in container.items():
            member_cls = member.__class__
            if member_cls is str:
                pool_key = (str, member)
                pooled = pool_get(pool_key)
                if pooled is None:
                    pool[pool_key] = member
                elif pooled is not member:
                    container[key] = pooled
            elif member_cls in pooled_classes or member_cls is tuple or member_cls is frozenset:
                pooled = _pool_value(member, pool)
                if pooled is not member:
                    container[key] = pooled
            elif recurse and (
                member_cls is list
                or member_cls is set
                or member_cls is dict
                or isinstance(member, dict)
            ):
                _pool_container_members(member, pool, depth + 1)


def _clear_property_backed_state_fields(state: dict[str, Any]) -> None:
    """Remove state keys that are represented by computed Op properties."""

    for field_name in _OP_PROPERTY_BACKED_FIELD_NAMES:
        state.pop(field_name, None)


def _recursive_safe_copy(val: Any) -> Any:
    """Compatibility alias for recursive input-argument copying."""

    return copy_arg_tree(val)


def _dtype_ref_or_none(value: Any) -> DtypeRef | None:
    """Return a neutral dtype reference for a dtype-like value."""

    return DtypeRef.from_value(value)


def _device_ref_from_metadata(out: Any, output_device: Any) -> DeviceRef | None:
    """Return a neutral device reference from payload or output-device metadata."""

    if isinstance(out, torch.Tensor):
        return DeviceRef.from_value(out.device)
    if output_device in (None, "same"):
        return None
    return DeviceRef.from_value(output_device)


def _summarize_value(value: Any) -> str:
    """Return a compact, human-readable string for one argument value."""

    if isinstance(value, torch.Tensor):
        shape = "x".join(str(int(d)) for d in value.shape) or "scalar"
        return f"Tensor({shape}, {str(value.dtype).replace('torch.', '')})"
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return text


class GradientRecord:
    """Saved gradient payload observed for one op during one backward pass.

    Parameters
    ----------
    owner:
        Op that received the gradient.
    ordinal:
        One-based local ordinal for this owner.
    backward_pass_number:
        One-based global backward pass number.
    grad:
        Saved raw gradient tensor, if retained.
    transformed_grad:
        Saved transformed gradient tensor, if retained.
    shape:
        Observed raw gradient shape.
    dtype:
        Observed raw gradient dtype string.
    memory:
        Observed raw gradient memory in bytes.
    timestamp:
        Event timestamp.
    """

    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "owner": FieldPolicy.WEAKREF_STRIP,
        "ordinal": FieldPolicy.KEEP,
        "backward_pass_number": FieldPolicy.KEEP,
        "grad": FieldPolicy.BLOB,
        "transformed_grad": FieldPolicy.BLOB,
        "shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "memory": FieldPolicy.KEEP,
        "timestamp": FieldPolicy.KEEP,
    }

    def __init__(
        self,
        *,
        owner: Any,
        ordinal: int,
        backward_pass_number: int,
        grad: torch.Tensor | None,
        transformed_grad: Any | None,
        shape: tuple[int, ...] | None,
        dtype: str | None,
        memory: int | None,
        timestamp: float,
    ) -> None:
        """Initialize one backward-gradient payload record for an operation.

        Parameters
        ----------
        owner:
            Object that owns the gradient record.
        ordinal:
            Position of this record within the owner.
        backward_pass_number:
            Backward pass index that produced the gradient.
        grad:
            Raw gradient tensor, if retained.
        transformed_grad:
            Post-processed gradient payload, if retained.
        shape:
            Gradient shape.
        dtype:
            Gradient dtype string.
        memory:
            Gradient memory in bytes.
        timestamp:
            Capture timestamp.
        """

        self.owner = owner
        self.ordinal = ordinal
        self.backward_pass_number = backward_pass_number
        self.grad = grad
        self.transformed_grad = transformed_grad
        self.shape = shape
        self.dtype = dtype
        self.memory = Bytes(memory or 0)
        self.timestamp = timestamp

    @property
    def is_saved(self) -> bool:
        """Return whether this record retained a payload."""

        return self.grad is not None or self.transformed_grad is not None

    @property
    def transformed_grad_shape(self) -> tuple[int, ...] | None:
        """Return the transformed gradient payload shape, when tensor-like."""

        return _shape_or_none(self.transformed_grad)

    @property
    def transformed_grad_dtype(self) -> torch.dtype | None:
        """Return the transformed gradient payload dtype, when tensor-like."""

        return _dtype_or_none(self.transformed_grad)

    @property
    def transformed_gradient_memory(self) -> Bytes | None:
        """Return transformed gradient payload memory, when tensor-like."""

        return _memory_or_none(self.transformed_grad)


class GradientRecordAccessor(Accessor[GradientRecord]):
    """Accessor for per-owner gradient records."""

    def __init__(self, records: list[GradientRecord]) -> None:
        """Initialize from records in local ordinal order."""

        super().__init__({str(record.ordinal): record for record in records}, item_list=records)

    def for_pass(self, pass_number: int) -> GradientRecord:
        """Return the gradient record for a one-based backward pass number."""

        matches = [record for record in self._list if record.backward_pass_number == pass_number]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise InvalidArgumentError(
                f"Multiple gradient records participated in pass {pass_number}; "
                "use positional indexing on .grads",
                code="gradient_pass_ambiguous",
                remedy="use positional indexing on .grads to pick one record",
                pass_number=pass_number,
            )
        available = [record.backward_pass_number for record in self._list]
        raise KeyError(
            f"Gradient record for backward pass {pass_number} not found; participated in "
            f"passes {available}."
        )


def _summarize_call_args(saved_args: Any, non_tensor_pos_args: Any) -> str | None:
    """Build a human-readable summary of an Op's positional arguments.

    Prefers the fully-saved positional args when available; otherwise falls
    back to the captured non-tensor positional args. Returns ``None`` when no
    positional-argument information was captured.
    """

    source = saved_args if saved_args is not None else non_tensor_pos_args
    if source is None:
        return None
    try:
        items = list(source)
    except TypeError:
        return _summarize_value(source)
    if not items:
        return ""
    return ", ".join(_summarize_value(item) for item in items)


def _summarize_call_kwargs(saved_kwargs: Any, non_tensor_kwargs: Any) -> str | None:
    """Build a human-readable summary of an Op's keyword arguments.

    Prefers the fully-saved keyword args when available; otherwise falls back to
    the captured non-tensor keyword args. Returns ``None`` when no
    keyword-argument information was captured.
    """

    source = saved_kwargs if saved_kwargs is not None else non_tensor_kwargs
    if source is None:
        return None
    if isinstance(source, dict):
        pairs = source.items()
    else:
        try:
            pairs = list(source)  # type: ignore[assignment]
        except TypeError:
            return _summarize_value(source)
    rendered = []
    for pair in pairs:
        try:
            key, value = pair
        except (TypeError, ValueError):
            rendered.append(_summarize_value(pair))
            continue
        rendered.append(f"{key}={_summarize_value(value)}")
    if not rendered:
        return ""
    return ", ".join(rendered)


def _set_saved_out_metadata(entry: "Op", tensor: torch.Tensor) -> None:
    """Refresh saved output metadata from a replacement tensor.

    Parameters
    ----------
    entry:
        Layer pass whose saved output metadata should match ``tensor``.
    tensor:
        Saved output tensor.

    Returns
    -------
    None
        Metadata fields are updated through internal setters.
    """

    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    entry._internal_set("shape", shape)
    entry._internal_set("dtype", dtype)
    entry._internal_set("dtype_ref", DtypeRef.from_value(dtype))
    entry._internal_set("device_ref", DeviceRef.from_value(tensor.device))
    entry._internal_set(
        "activation_memory",
        Bytes(get_memory_amount_from_metadata(tensor, shape, dtype)),
    )
    entry._internal_set("has_saved_activation", True)
    entry._internal_set("transformed_out_shape", _shape_or_none(entry.payload.transformed_out))
    entry._internal_set("transformed_out_dtype", _dtype_or_none(entry.payload.transformed_out))
    entry._internal_set(
        "transformed_activation_memory", _memory_or_none(entry.payload.transformed_out)
    )


if TYPE_CHECKING:
    import pandas as pd

    from .._io.lazy import LazyActivationRef
    from ..receptive_field._view import ReceptiveFieldView
    from .func_call_location import FuncCallLocation
    from .layer import Layer
    from .param import Param
    from .trace import Trace


def _warn_synthesized_out_read(source_trace: Any) -> None:
    """Fire the once-per-trace synthesized-value warning for an ``Op.out`` read when due."""

    if _synthesized_out_read_due(source_trace):
        from .._quickstart._gate import warn_on_raw_read

        warn_on_raw_read(source_trace, site="Op.out")


class Op(OpAtenMixin, _SelectionOperand, _removed_members.RemovedMemberMixin):
    """Metadata for a single tensor operation (one pass of one layer).

    Constructed from a dict whose keys must exactly match
    ``LAYER_PASS_LOG_FIELD_ORDER`` (enforced at init time).  Every
    attribute is set explicitly (not via a loop) so that IDE
    autocompletion works.

    Notable design points:

    * ``_tracing_finished`` mirrors the owning Trace's flag. Methods like
      ``__str__`` branch on it to show raw vs final labels.
    * ``source_trace`` is a direct reference to the owning Trace.
      This creates a circular reference (Trace -> layer_list -> entry ->
      source_trace -> Trace) that is broken by ``cleanup()``.
    * ``fx_qualpath`` and ``fx_call_index`` expose metadata that mirrors
      ``torch.fx.symbolic_trace`` naming conventions, computed independently
      using TorchLens rules.  ``fx_qualpath`` is not a lookup key, so
      ``trace[label]`` does not accept it.  Combine
      ``fx_qualpath.replace(".", "_")`` with ``fx_call_index`` when an
      FX-style name form is needed.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> op = trace.ops[1]
    >>> type(op).__name__, op.func_name
    ('Op', 'relu')
    >>> tuple(op.out.shape)
    (1, 4, 8, 8)
    """

    __slots__ = ("_core", "_row")

    if TYPE_CHECKING:
        # Static field declarations for type checkers ONLY (invisible at
        # runtime, so the frozen debugger/DX surface is unchanged). The
        # explicitly typed names carry the annotations the former inline
        # `self.x: T = ...` assignments declared; every other stored field
        # was already inferred as Any from the untyped fields_dict.
        annotations: dict[str, Any]
        dtype_ref: DtypeRef | None
        device_ref: DeviceRef | None
        backend_address: str | None
        resolver_status: str
        _cost_activation_memory: Bytes | None
        _cost_transformed_activation_memory: Bytes | None
        visualizer_path: str | None
        _cost_autograd_memory: Bytes | None
        _autograd_num_autograd_tensors: int | None
        _cost_bytes_delta_at_call: Bytes | None
        _cost_bytes_peak_at_call: Bytes | None
        gradient_memory: Bytes | None
        transformed_gradient_memory: Bytes | None
        func_id: FunctionRegistryKey | None
        _source_frames: list["FuncCallLocation"]
        _source_var_names: list[str]
        _cost_duration: Duration | None
        _cost_flops_forward: Flops | None
        _cost_flops_backward: Flops | None
        _param_barcodes: list[Any]
        _param_logs: list["Param"]
        _cost_param_memory: Bytes
        _graph_is_orphan: bool
        _source_fx_path: str | None
        _source_fx_call_index: int
        _out_ref: "LazyActivationRef" | None
        _grad_ref: "LazyActivationRef" | None
        _pending_blob_id: str | None
        _pending_transformed_out_blob_id: str | None
        _pending_grad_blob_id: str | None
        _pending_transformed_grad_blob_id: str | None
        _grad_records: list[GradientRecord]

        _label_raw: Any
        _layer_label_raw: Any
        step_index: Any
        raw_index: Any
        ordinal_index: Any
        _tracing_finished: Any
        _construction_done: Any
        label: Any
        short_label: Any
        layer_label: Any
        short_layer_label: Any
        type: Any
        type_index: Any
        pass_number: Any
        num_passes: Any
        lookup_keys: Any
        out: Any
        has_saved_activation: Any
        _output_device: Any
        _activation_transform: Any
        edits: Any
        is_edited: Any
        _detach_saved_activations: Any
        _call_has_saved_args: Any
        _call_saved_args: Any
        _call_saved_kwargs: Any
        _call_args_template: Any
        _call_kwargs_template: Any
        shape: Any
        _payload_transformed_out_shape: Any
        dtype: Any
        _payload_transformed_out_dtype: Any
        _payload_transformed_out: Any
        has_out_variations: Any
        out_versions_by_child: Any
        grad: Any
        _payload_transformed_grad: Any
        save_grads: Any
        has_grad: Any
        _autograd_grad_shape: Any
        _payload_transformed_grad_shape: Any
        _autograd_grad_dtype: Any
        _payload_transformed_grad_dtype: Any
        func: Any
        _call_call_id: Any
        func_name: Any
        _source_func_qualname: Any
        _call_rng_states: Any
        _call_autocast_state: Any
        _call_arg_names: Any
        _call_num_args_total: Any
        _call_num_pos_args: Any
        _call_num_kwargs: Any
        _call_non_tensor_pos_args: Any
        _call_non_tensor_kwargs: Any
        _call_non_tensor_args: Any
        is_inplace: Any
        _autograd_grad_fn_class_name: Any
        _autograd_grad_fn_class_qualname: Any
        _grad_fn_object_id: Any
        _autograd_grad_fn_handle: Any
        _autograd_grad_fn: Any
        in_multi_output: Any
        multi_output_index: Any
        multi_output_name: Any
        container_path: Any
        container_spec: Any
        _call_is_func_transform: Any
        _call_func_transform_kind: Any
        _call_func_transform_chain: Any
        _call_func_transform_config: Any
        _call_func_transform_fn_name: Any
        _call_func_transform_fn_qualname: Any
        _call_func_transform_fn_source: Any
        _call_unattributed_tensor_args: Any
        _call_dropped_edge_tensor_args: Any
        _graph_parent_params: Any
        _graph_parent_param_ops: Any
        param_shapes: Any
        num_params: Any
        num_params_trainable: Any
        num_params_frozen: Any
        _graph_equivalence_class: Any
        _graph_equivalent_ops: Any
        _graph_recurrent_ops: Any
        _graph_site_key: str | None
        injection_provenance: dict[str, Any] | None
        episode_step: int | None
        tl_authored_root: bool | None
        parents: Any
        _call_parent_arg_positions: Any
        _edge_uses: Any
        _graph_edge_substitutions: dict[Any, Any]
        edge_replacement_stamps: dict[Any, Any]
        _graph_root_ancestors: Any
        children: Any
        _graph_has_children: Any
        is_input: Any
        input_was_parameter: Any
        _graph_has_input_ancestor: Any
        input_ancestors: Any
        _graph_min_distance_from_input: Any
        _graph_max_distance_from_input: Any
        is_output: Any
        _graph_is_output_parent: Any
        _graph_is_final_output: Any
        _graph_has_output_descendant: Any
        output_descendants: Any
        _graph_io_role: Any
        _graph_min_distance_to_output: Any
        _graph_max_distance_to_output: Any
        _graph_is_buffer: Any
        is_constant: bool
        buffer_address: Any
        buffer_pass: Any
        _graph_buffer_source: Any
        buffer_write_kind: Any
        buffer_value_changed: Any
        buffer_replay_validated: Any
        _graph_buffer_source_func_name: Any
        _graph_is_internal_source: Any
        _graph_has_internal_source_ancestor: Any
        _graph_internal_source_parents: Any
        _graph_internal_source_ancestors: Any
        is_internal_sink: Any
        _control_flow_is_terminal_bool: Any
        _control_flow_is_terminal_conditional_bool: Any
        _control_flow_context_kind: Any
        _control_flow_wrapper_kind: Any
        terminal_conditional_id: Any
        _control_flow_is_scalar_bool: Any
        _control_flow_bool_value: Any
        _control_flow_in_conditionals: Any
        _control_flow_terminal_bool_for: Any
        _control_flow_branch_stack: Any
        _control_flow_branch_depth: Any
        _control_flow_entry_children: Any
        _control_flow_then_children: Any
        _control_flow_elif_children: Any
        _control_flow_else_children: Any
        _control_flow_arm_children: Any
        module: Any
        _address_normalized: Any
        modules: Any
        _module_context_call_stack: Any
        _module_context_input_to_module_calls: Any
        _module_context_entry_arg_keys: Any
        _module_context_output_of_modules: Any
        _module_context_output_of_module_calls: Any
        _module_context_is_module_output: Any
        _module_context_is_atomic_module: Any
        _module_context_atomic_module_call: Any
        _call_config: Any
        _source_trace_ref: Any
        _facets_cache: Any
        _receptive_field_cache: Any
        _projective_field_cache: Any
        _arg_expressions_cache: Any
        _is_in_conditional_body: Any

    _PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_label_raw": FieldPolicy.KEEP,
        "_layer_label_raw": FieldPolicy.KEEP,
        "raw_label": FieldPolicy.DROP,
        "step_index": FieldPolicy.KEEP,
        "raw_index": FieldPolicy.KEEP,
        "ordinal_index": FieldPolicy.KEEP,
        "source_trace": FieldPolicy.DROP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
        "_tracing_finished": FieldPolicy.KEEP,
        "_construction_done": FieldPolicy.DROP,
        "_is_in_conditional_body": FieldPolicy.KEEP,
        "label": FieldPolicy.KEEP,
        "short_label": FieldPolicy.KEEP,
        "layer_label": FieldPolicy.KEEP,
        "short_layer_label": FieldPolicy.KEEP,
        "type": FieldPolicy.KEEP,
        "type_index": FieldPolicy.KEEP,
        "pass_number": FieldPolicy.KEEP,
        "num_passes": FieldPolicy.KEEP,
        "lookup_keys": FieldPolicy.KEEP,
        "out": FieldPolicy.BLOB,
        "transformed_out": FieldPolicy.BLOB,
        "has_saved_activation": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "activation_transform": FieldPolicy.DROP,
        "annotations": FieldPolicy.KEEP,
        "edits": FieldPolicy.KEEP,
        "is_edited": FieldPolicy.KEEP,
        "detach_saved_activations": FieldPolicy.KEEP,
        "has_saved_args": FieldPolicy.KEEP,
        "saved_args": FieldPolicy.BLOB_RECURSIVE,
        "saved_kwargs": FieldPolicy.BLOB_RECURSIVE,
        "args_template": FieldPolicy.DROP,
        "kwargs_template": FieldPolicy.DROP,
        "input_ops": FieldPolicy.DROP,
        "input_activations": FieldPolicy.DROP,
        "input_shapes": FieldPolicy.DROP,
        "input_dtypes": FieldPolicy.DROP,
        "input_memory": FieldPolicy.DROP,
        "num_inputs": FieldPolicy.DROP,
        "shape": FieldPolicy.KEEP,
        "transformed_out_shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "dtype_ref": FieldPolicy.KEEP,
        "transformed_out_dtype": FieldPolicy.KEEP,
        "device_ref": FieldPolicy.KEEP,
        "backend_address": FieldPolicy.KEEP,
        "resolver_status": FieldPolicy.KEEP,
        "activation_memory": FieldPolicy.KEEP,
        "transformed_activation_memory": FieldPolicy.KEEP,
        "visualizer_path": FieldPolicy.KEEP,
        "autograd_memory": FieldPolicy.KEEP,
        "num_autograd_tensors": FieldPolicy.KEEP,
        "bytes_delta_at_call": FieldPolicy.KEEP,
        "bytes_peak_at_call": FieldPolicy.KEEP,
        "has_out_variations": FieldPolicy.KEEP,
        "out_versions_by_child": FieldPolicy.BLOB_RECURSIVE,
        "grad": FieldPolicy.BLOB,
        "transformed_grad": FieldPolicy.BLOB,
        "save_grads": FieldPolicy.KEEP,
        "has_grad": FieldPolicy.KEEP,
        "grad_shape": FieldPolicy.KEEP,
        "transformed_grad_shape": FieldPolicy.KEEP,
        "grad_dtype": FieldPolicy.KEEP,
        "transformed_grad_dtype": FieldPolicy.KEEP,
        "gradient_memory": FieldPolicy.KEEP,
        "transformed_gradient_memory": FieldPolicy.KEEP,
        "func": FieldPolicy.DROP,
        "func_id": FieldPolicy.KEEP,
        "func_call_id": FieldPolicy.KEEP,
        "func_name": FieldPolicy.KEEP,
        "func_qualname": FieldPolicy.KEEP,
        "code_context": FieldPolicy.KEEP,
        "var_names": FieldPolicy.KEEP,
        ARG_EXPRESSIONS_FIELD: FieldPolicy.KEEP,
        "func_duration": FieldPolicy.KEEP,
        "flops_forward": FieldPolicy.KEEP,
        "flops_backward": FieldPolicy.KEEP,
        "func_rng_states": FieldPolicy.BLOB_RECURSIVE,
        "func_autocast_state": FieldPolicy.KEEP,
        "arg_names": FieldPolicy.KEEP,
        "num_args_total": FieldPolicy.KEEP,
        "num_pos_args": FieldPolicy.KEEP,
        "num_kwargs": FieldPolicy.KEEP,
        "non_tensor_pos_args": FieldPolicy.KEEP,
        "non_tensor_kwargs": FieldPolicy.KEEP,
        "func_non_tensor_args": FieldPolicy.KEEP,
        "is_inplace": FieldPolicy.KEEP,
        "grad_fn_class_name": FieldPolicy.KEEP,
        "grad_fn_class_qualname": FieldPolicy.KEEP,
        "grad_fn_object_id": FieldPolicy.KEEP,
        "grad_fn_handle": FieldPolicy.DROP,
        "grad_fn": FieldPolicy.DROP,
        "in_multi_output": FieldPolicy.KEEP,
        "multi_output_index": FieldPolicy.KEEP,
        "multi_output_name": FieldPolicy.KEEP,
        "container_path": FieldPolicy.KEEP,
        "container_spec": FieldPolicy.KEEP,
        "is_transform": FieldPolicy.KEEP,
        "transform_kind": FieldPolicy.KEEP,
        "transform_chain": FieldPolicy.KEEP,
        "transform_config": FieldPolicy.KEEP,
        "transform_fn_name": FieldPolicy.KEEP,
        "transform_fn_qualname": FieldPolicy.KEEP,
        "transform_fn_source": FieldPolicy.KEEP,
        "unattributed_tensor_args": FieldPolicy.KEEP,
        "dropped_edge_tensor_args": FieldPolicy.KEEP,
        "parent_params": FieldPolicy.KEEP,
        "_param_barcodes": FieldPolicy.KEEP,
        "parent_param_ops": FieldPolicy.KEEP,
        "_param_logs": FieldPolicy.KEEP,
        "param_shapes": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "param_memory": FieldPolicy.KEEP,
        "equivalence_class": FieldPolicy.KEEP,
        "equivalent_ops": FieldPolicy.KEEP,
        "recurrent_ops": FieldPolicy.KEEP,
        # site_key_v1 structural-position identity: persists as of tlspec v8
        # with byte-exact recomputation at load (_io/forgery_validation.py).
        "site_key": FieldPolicy.KEEP,
        # tlspec v9 entry-dark injected-op identity:
        # None on every model op until the log_injections writer lands;
        # fail-closed shape validation at load (_io/forgery_validation.py).
        "injection_provenance": FieldPolicy.KEEP,
        # tlspec v9 entry-dark slots: the episode-step stamp
        # and the TL-authored-root marker;
        # None on every op today, fail-closed load validation from day one.
        "episode_step": FieldPolicy.KEEP,
        "tl_authored_root": FieldPolicy.KEEP,
        "parents": FieldPolicy.KEEP,
        "parent_arg_positions": FieldPolicy.KEEP,
        "_edge_uses": FieldPolicy.KEEP,
        # L6 stage 3: occurrence-granular edge-substitution store + save-time
        # corroboration stamps; persist as of tlspec v8 (BLOB_RECURSIVE may
        # carry unit-term masks). Uncorroborated tier-(ii) entries FAIL
        # validation; the audit digest relation validates at load.
        "edge_substitutions": FieldPolicy.BLOB_RECURSIVE,
        "edge_replacement_stamps": FieldPolicy.KEEP,
        "root_ancestors": FieldPolicy.KEEP,
        "children": FieldPolicy.KEEP,
        "has_children": FieldPolicy.KEEP,
        "is_input": FieldPolicy.KEEP,
        "input_was_parameter": FieldPolicy.KEEP,
        "has_input_ancestor": FieldPolicy.KEEP,
        "input_ancestors": FieldPolicy.KEEP,
        "min_distance_from_input": FieldPolicy.KEEP,
        "max_distance_from_input": FieldPolicy.KEEP,
        "is_output": FieldPolicy.KEEP,
        "is_output_parent": FieldPolicy.KEEP,
        "is_final_output": FieldPolicy.KEEP,
        "has_output_descendant": FieldPolicy.KEEP,
        "output_descendants": FieldPolicy.KEEP,
        "is_orphan": FieldPolicy.KEEP,
        "min_distance_to_output": FieldPolicy.KEEP,
        "max_distance_to_output": FieldPolicy.KEEP,
        "io_role": FieldPolicy.KEEP,
        "is_buffer": FieldPolicy.KEEP,
        "is_constant": FieldPolicy.KEEP,
        "address": FieldPolicy.KEEP,
        "buffer_pass": FieldPolicy.KEEP,
        "buffer_source": FieldPolicy.KEEP,
        "buffer_write_kind": FieldPolicy.KEEP,
        "buffer_value_changed": FieldPolicy.KEEP,
        "buffer_replay_validated": FieldPolicy.KEEP,
        "buffer_source_func_name": FieldPolicy.KEEP,
        "is_internal_source": FieldPolicy.KEEP,
        "is_internally_initialized": FieldPolicy.DROP,
        "has_internal_source_ancestor": FieldPolicy.KEEP,
        "internal_source_parents": FieldPolicy.KEEP,
        "internal_source_ancestors": FieldPolicy.KEEP,
        "is_internal_sink": FieldPolicy.KEEP,
        "is_terminal_bool": FieldPolicy.KEEP,
        "is_terminal_conditional_bool": FieldPolicy.KEEP,
        "conditional_context_kind": FieldPolicy.KEEP,
        "conditional_wrapper_kind": FieldPolicy.KEEP,
        "terminal_conditional_id": FieldPolicy.KEEP,
        "is_scalar_bool": FieldPolicy.KEEP,
        "bool_value": FieldPolicy.KEEP,
        "in_conditionals": FieldPolicy.KEEP,
        "terminal_bool_for": FieldPolicy.KEEP,
        "is_in_conditional_body": FieldPolicy.DROP,
        "conditional_branch_stack": FieldPolicy.KEEP,
        "conditional_branch_depth": FieldPolicy.KEEP,
        "conditional_entry_children": FieldPolicy.KEEP,
        "conditional_then_children": FieldPolicy.KEEP,
        "conditional_elif_children": FieldPolicy.KEEP,
        "conditional_else_children": FieldPolicy.KEEP,
        "conditional_arm_children": FieldPolicy.KEEP,
        "module": FieldPolicy.KEEP,
        "_address_normalized": FieldPolicy.KEEP,
        "modules": FieldPolicy.KEEP,
        "fx_qualpath": FieldPolicy.KEEP,
        "fx_call_index": FieldPolicy.KEEP,
        "module_call_stack": FieldPolicy.KEEP,
        "module_entry_arg_keys": FieldPolicy.KEEP,
        "input_to_module_calls": FieldPolicy.KEEP,
        "output_of_modules": FieldPolicy.KEEP,
        "output_of_module_calls": FieldPolicy.KEEP,
        "is_module_output": FieldPolicy.KEEP,
        "is_atomic_module": FieldPolicy.KEEP,
        "atomic_module_call": FieldPolicy.KEEP,
        "func_config": FieldPolicy.BLOB_RECURSIVE,
        "out_ref": FieldPolicy.DROP,
        "grad_ref": FieldPolicy.DROP,
        "_grad_records": FieldPolicy.BLOB_RECURSIVE,
        "_receptive_field_cache": FieldPolicy.DROP,
        "_projective_field_cache": FieldPolicy.DROP,
        "_arg_expressions_cache": FieldPolicy.DROP,
        "_pending_blob_id": FieldPolicy.DROP,
        "_pending_transformed_out_blob_id": FieldPolicy.DROP,
        "_pending_grad_blob_id": FieldPolicy.DROP,
        "_pending_transformed_grad_blob_id": FieldPolicy.DROP,
        # Lazily-populated facet-view cache slot: popped by __getstate__,
        # FORK_RECONSTRUCT on fork, disposable. Declared so the Op slot
        # universe carries no shadow storage outside _FIELD_POLICY.
        "_facets_cache": FieldPolicy.DROP,
    }
    _FIELD_POLICY = build_record_field_policy_table(
        LAYER_PASS_LOG_FIELD_ORDER,
        _PORTABLE_STATE_SPEC,
        fork_policy=LAYER_PASS_LOG_FIELD_FORK_POLICY,
        default_fill_state=_LAYER_PASS_LOG_DEFAULT_FILL,
        schema_key="op",
    )
    _PORTABLE_STATE_SPEC = portable_state_spec_from_policy(_FIELD_POLICY)
    _FIELD_FORK_POLICY = fork_policy_from_policy(_FIELD_POLICY)
    _DEFAULT_FILL_STATE = default_fill_state_from_policy(_FIELD_POLICY)

    def _slot(self, name: str, default: Any = None) -> Any:
        """Return one physical slot value, or ``default`` when it is unset."""

        try:
            return _object_getattribute(self, OP_STORAGE_ATTRS.get(name, name))
        except AttributeError:
            return default

    def __getattribute__(
        self,
        name: str,
        # Bound once at definition time: this method runs on EVERY attribute
        # read (150-200k per trace), so even the cached LOAD_GLOBAL for these
        # two names was measurable; LOAD_FAST via default args is cheaper.
        _getattribute: Callable[[Any, str], Any] = _object_getattribute,
        _lazy_fields: frozenset = _INTERCEPTED_READ_FIELDS,
    ) -> Any:
        """Materialize lazy grads, copy shared groups, reject unsaved predicate outs."""

        if name not in _lazy_fields:
            return _getattribute(self, name)
        if name == "grad":
            slot = _object_getattribute(self, "_slot")
            records = slot("_grad_records")
            if records:
                saved = [record for record in records if record.grad is not None]
                if len(saved) == 1:
                    grad = saved[0].grad
                    if isinstance(grad, torch.Tensor) and slot("grad") is None:
                        object.__getattribute__(self, "_internal_set")("grad", grad)
                    return grad
                if len(saved) > 1:
                    label = slot("label") or slot("layer_label") or slot("_label_raw")
                    passes = [record.backward_pass_number for record in saved]
                    raise InvalidArgumentError(
                        f"op {label} has gradients saved from multiple backward passes {passes}; "
                        "use op.autograd.grads[...] / op.autograd.grad_for(bwd=k)",
                        code="gradient_pass_ambiguous",
                        remedy="use op.autograd.grads[...] or op.autograd.grad_for(bwd=k) to pick one pass",
                        label=str(label),
                    )
            grad = slot("grad")
            if grad is None and slot("grad_ref") is not None:
                return object.__getattribute__(self, "payload").materialize_grad()
            return grad
        if name == "out":
            slot = _object_getattribute(self, "_slot")
            out = slot("out")
            source_ref = slot("_source_trace_ref")
            source_trace = None if source_ref is None else source_ref()
            if out is None and source_trace is None and source_ref is not None:
                _getattribute(self, "source_trace")  # collected Trace: refuse typed, never None
            # Lazy-payload honesty gate (P0 part 1): a lazy blob ref materializes (sha-verified)
            # instead of reading None; live streamed captures (retain_in_memory=False) keep None.
            if (
                out is None
                and slot("out_ref") is not None
                and (
                    getattr(source_trace, "_predicate_save_options", None) is not None
                    or getattr(source_trace, "payload_load_status", None) == "loaded_lazy"
                )
            ):
                return object.__getattribute__(self, "payload").materialize_out()
            if (
                slot("_tracing_finished")
                and not slot("has_saved_activation", False)
                and (
                    getattr(source_trace, "_predicate_save_options", None) is not None
                    # P0 part 4: a lazy load refuses unsaved rows; eager loads keep None-for-unsaved.
                    or getattr(source_trace, "payload_load_status", None) == "loaded_lazy"
                )
            ):
                label = slot("label") or slot("layer_label") or slot("_label_raw")
                raise PayloadUnavailableError(
                    f"op {label} was not saved; no saved payload is available",
                    code="activation_not_saved",
                    remedy="re-run with save=... to retain this activation",
                    label=str(label),
                )
            if out is None:
                _refuse_read_without_trace(self, out_trace=source_trace)
            _warn_synthesized_out_read(source_trace)
            if not getattr(source_trace, "_postprocessing_active", False):
                _validate_reference_out_not_mutated(_out_read_state(slot, out))
        return _getattribute(self, name)

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            """Miss hook: a removed public member redirects typed (``facade_redirect``)."""

            if not name.startswith("_"):
                refuse_removed_member("Op", name, _removed_members.REMOVED_MEMBERS["Op"])
                _refuse_read_without_trace(self, member=name)
            raise _familiar_names.missing_attribute_error(type(self).__name__, name)

    def __setattr__(
        self,
        name: str,
        value: Any,
        # Same definition-time binding as ``__getattribute__``: capture writes
        # every Op field through here, so the two globals are hoisted to
        # LOAD_FAST default args.
        _setattr: Callable[[Any, str, Any], None] = _object_setattr,
        _guarded_fields: frozenset = _DIRECT_WRITE_GUARDED_FIELDS,
    ) -> None:
        """Mark owning logs dirty when user code directly writes guarded fields.

        Parameters
        ----------
        name:
            Attribute being written.
        value:
            New attribute value.
        """

        # Guarded-name test first: it is a pure frozenset probe that is False for
        # almost every write, so the (side-effect-free) construction-done slot
        # read is skipped entirely on the hot path.
        if name in _guarded_fields and self._slot("_construction_done", False):
            owner = self._slot("_source_trace_ref")
            trace = owner() if owner is not None else None
            if trace is not None:
                object.__setattr__(trace, "_has_direct_writes", True)
                object.__setattr__(trace, "state", TraceState.DIRECT_WRITE_DIRTY)
                if not getattr(trace, "_warned_direct_write", False):
                    warnings.warn(
                        "DirectActivationWriteWarning: direct Op out writes "
                        "are not recipe edits; replay/rerun propagation will overlay them.",
                        DirectActivationWriteWarning,
                        stacklevel=2,
                    )
                    object.__setattr__(trace, "_warned_direct_write", True)
        _setattr(self, name, value)

    def _internal_set(self, attr: str, value: Any) -> None:
        """Set an attribute without marking the owner dirty.

        Parameters
        ----------
        attr:
            Attribute name to set.
        value:
            Value to assign.
        """

        _object_setattr(self, OP_STORAGE_ATTRS.get(attr, attr), value)

    def _compact_metadata(self, pool: dict[Any, Any]) -> None:
        """Replace repeated immutable metadata with pooled shared instances.

        Called once per Op by :func:`~torchlens._data_classes._compaction.compact_op_metadata`
        at the freeze seam. Every field keeps a value that is ``==`` to, and of the
        same exact class as, the one it had before; only the object identity of
        immutable values is collapsed. See the pooling block near the top of
        this module for the safety argument.

        Parameters
        ----------
        pool:
            Pass-local pooling table shared by every Op of one Trace.
        """

        getattribute = _object_getattribute
        pooled_classes = _POOLED_CLASSES
        pool_get = pool.get
        for name in _POOLED_SLOTS:
            try:
                value = getattribute(self, name)
            except AttributeError:
                continue
            # Most slots hold None or a plain scalar; the class ladder below is
            # ordered so those fall through without a single Python call.
            if value is None:
                continue
            cls = value.__class__
            if cls in pooled_classes and (cls is not Duration or not value):
                key = (cls, value.hex()) if cls is Duration else (cls, value)
                pooled = pool_get(key)
                if pooled is None:
                    pool[key] = value
                elif pooled is not value:
                    _object_setattr(self, name, pooled)
            elif cls is list or cls is set or cls is dict:
                _pool_container_members(value, pool, 0)
            elif cls is tuple or cls is frozenset:
                pooled = _pool_value(value, pool)
                if pooled is not value:
                    _object_setattr(self, name, pooled)
            elif isinstance(value, dict):
                _pool_container_members(value, pool, 0)

    def _append_tensor_from(self, other: "Op", field_name: str) -> None:
        """Append one tensor field from another pass along batch dimension 0.

        Parameters
        ----------
        other:
            New chunk pass with a compatible tensor field.
        field_name:
            Tensor attribute name to concatenate.
        """

        current_value = getattr(self, op_attr(field_name))
        other_value = getattr(other, op_attr(field_name))
        if isinstance(current_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            self._internal_set(field_name, concatenate_batch_tensors(current_value, other_value))

    def __init__(self, fields_dict: dict[str, Any], *, _store: Any = None) -> None:
        """Initialise from a complete fields dictionary.

        Args:
            fields_dict: Dict with values for all fields defined in
                ``LAYER_PASS_LOG_FIELD_ORDER``.  Missing or extra keys
                raise ``ValueError``.
            _store: Private ingress hook: the owning trace's ``OpRowStore``
                when materialize step 0 constructs this op (a shared row is
                appended). Every other construction path (``copy()``, direct
                user construction, preview backends) gets a detached
                single-row store.
        """
        # Validate that fields_dict has exactly the expected keys:
        if "_address_normalized" not in fields_dict:
            fields_dict["_address_normalized"] = None
        if "fx_qualpath" not in fields_dict:
            fields_dict["fx_qualpath"] = None
        if "fx_call_index" not in fields_dict:
            fields_dict["fx_call_index"] = 0
        if "ordinal_index" not in fields_dict:
            fields_dict["ordinal_index"] = -1
        if "grad_fn" not in fields_dict:
            fields_dict["grad_fn"] = None
        if fields_dict.get("dtype_ref") is None:
            fields_dict["dtype_ref"] = _dtype_ref_or_none(fields_dict.get("dtype"))
        if fields_dict.get("device_ref") is None:
            fields_dict["device_ref"] = _device_ref_from_metadata(
                fields_dict.get("out"), fields_dict.get("output_device")
            )
        if fields_dict.get("backend_address") is None:
            fields_dict["backend_address"] = fields_dict.get("address")
        if fields_dict.get("resolver_status") is None:
            fields_dict["resolver_status"] = "resolved"
        for derived_field in (
            "input_ops",
            "input_activations",
            "input_shapes",
            "input_dtypes",
            "input_memory",
            "num_inputs",
            "raw_label",
            ARG_EXPRESSIONS_FIELD,
            "is_internally_initialized",
        ):
            if derived_field not in fields_dict:
                fields_dict[derived_field] = None
        fields_dict_key_set = set(fields_dict.keys())
        if fields_dict_key_set != _LAYER_PASS_LOG_FIELD_ORDER_SET:
            error_str = "Error initializing Op:"
            missing_fields = _LAYER_PASS_LOG_FIELD_ORDER_SET - fields_dict_key_set
            extra_fields = fields_dict_key_set - _LAYER_PASS_LOG_FIELD_ORDER_SET
            if len(missing_fields) > 0:
                error_str += f"\n\t- Missing fields {', '.join(missing_fields)}"
            if len(extra_fields) > 0:
                error_str += f"\n\t- Extra fields {', '.join(extra_fields)}"
            raise ValueError(error_str)

        # One-pass layout-ordered row seed. The former 185 explicit
        # descriptor assignments were pure stores during construction (the
        # __setattr__ guard is inert until _construction_done flips), so
        # normalizing the converted fields into the dict and building the
        # row cells directly is value-identical at ~40% of the cost. Static
        # field types live in the TYPE_CHECKING declaration block on the
        # class; runtime attribute surface comes from the generated field
        # descriptors.
        fd = fields_dict
        for field_name in _INIT_BYTES_FIELDS:
            fd[field_name] = as_bytes(fd[field_name])
        fd["param_memory"] = Bytes(fd["param_memory"] or 0)
        fd["func_duration"] = as_duration(fd["func_duration"])
        fd["flops_forward"] = as_flops(fd["flops_forward"])
        fd["flops_backward"] = as_flops(fd["flops_backward"])
        get = fd.get
        cells = [get(field_name, _MISSING) for field_name in _OP_STORE_LAYOUT.names]
        # Store as weakref to break the circular reference
        # (Trace -> layer_list -> entry -> Trace).
        source_trace = fd["source_trace"]
        cells[_FID_SOURCE_TRACE_REF] = (
            weakref.ref(source_trace) if source_trace is not None else None
        )
        # The public alias value backs the `_is_in_conditional_body` cell
        # (exactly what the compatibility property setter does).
        cells[_FID_IS_IN_CONDITIONAL_BODY] = fd["is_in_conditional_body"]
        for none_fid in _FIDS_INIT_NONE:
            cells[none_fid] = None
        cells[_FID_GRAD_RECORDS] = []
        cells[_FID_CONSTRUCTION_DONE] = True
        set_ = object.__setattr__
        if _store is None:
            store: Any = DetachedOpStore(_OP_STORE_LAYOUT)
        else:
            store = _store
        row = store.adopt_row(cells)
        set_(self, "_core", store)
        set_(self, "_row", row)

    @property
    def cost(self) -> OpCost:
        """Compute, memory and timing facts of this Op (``op.cost.flops_forward``)."""

        return OpCost(self)

    @property
    def control_flow(self) -> OpControlFlow:
        """Conditional and branch facts of this Op (``op.control_flow.depth``)."""

        return OpControlFlow(self)

    @property
    def call(self) -> OpCall:
        """How this Op's function was called (``op.call.config``)."""

        return OpCall(self)

    @property
    def source(self) -> OpSource:
        """Where this Op's call sits in the user's source (``op.source.frames``)."""

        return OpSource(self)

    @property
    def autograd(self) -> OpAutograd:
        """Autograd facts and saved gradients of this Op (``op.autograd.grad_fn``)."""

        return OpAutograd(self)

    @property
    def module_context(self) -> OpModuleContext:
        """Module-nesting facts of this Op (``op.module_context.call_depth``)."""

        return OpModuleContext(self)

    @property
    def graph(self) -> OpGraph:
        """Graph position of this Op (``op.graph.has_parents``)."""

        return OpGraph(self)

    @property
    def payload(self) -> OpPayload:
        """Saved transformed payloads and lazy materialization (``op.payload.transformed_out``)."""

        return OpPayload(self)

    @property
    def macs_backward(self) -> Macs | None:
        """Backward MACs are not derivable and always ``None``."""

        return _op_cost.macs_backward(self)

    @property
    def macs_total(self) -> Macs | None:
        """Total MACs are not derivable and always ``None``."""

        return _op_cost.macs_total(self)

    @property
    def param_names(self) -> list[str]:
        """Return short names of parameters consumed by this Op.

        Returns
        -------
        list[str]
            Parameter names in consumed-parameter order.
        """

        return [param.name for param in self._param_logs]

    @property
    def param_dtypes(self) -> list[torch.dtype]:
        """Return dtypes of parameters consumed by this Op.

        Returns
        -------
        list[torch.dtype]
            Parameter dtypes in consumed-parameter order.
        """

        return [param.dtype for param in self._param_logs]

    @property
    def num_param_tensors_trainable(self) -> int:
        """Number of trainable parameter tensors consumed by this Op."""

        return sum(1 for param in self._param_logs if param.is_trainable)

    @property
    def num_param_tensors_frozen(self) -> int:
        """Number of frozen parameter tensors consumed by this Op."""

        return sum(1 for param in self._param_logs if not param.is_trainable)

    @property
    def has_trainable_params(self) -> bool:
        """Whether this Op consumes at least one trainable parameter."""

        return self.num_params_trainable > 0

    @property
    def has_frozen_params(self) -> bool:
        """Whether this Op consumes at least one frozen parameter."""

        return self.num_params_frozen > 0

    @property
    def _graph_is_compute(self) -> bool:
        """Whether this Op executed a torch function rather than a boundary sentinel."""

        return not (self.is_input or self.is_output or self._graph_is_buffer)

    @property
    def fx_label(self) -> str | None:
        """Return a torch.fx-style label for this Op when module context is available."""

        return _labels.fx_label(self._source_fx_path, self._source_fx_call_index)

    @property
    def trace(self) -> "Trace":
        """Alias for the owning Trace back-reference."""

        return self.source_trace

    @property
    def _autograd_grads(self) -> GradientRecordAccessor:
        """Per-pass gradient records saved for this Op.

        One record per backward pass that reached this op's output, each the
        gradient after this trace's ``tl.grad(site)`` edit chain.
        """

        trace = self.source_trace
        raise_if_no_backward_capture(trace, plural_subject="op.autograd.grads or saved_grad_ops")
        records = self._slot("_grad_records")
        if records is None:
            records = []
            self._internal_set("_grad_records", records)
        return GradientRecordAccessor(records)

    @property
    def _autograd_derived_grad(self) -> Any | None:
        """Return this op's derived intermediate gradient payload, if available."""

        return _doors.intervention()._op_edits.derived_grad(self)  # record-groups: flat

    def _autograd_grad_for(self, *, bwd: int) -> torch.Tensor:
        """Return the saved gradient tensor for one backward pass."""

        return _doors.intervention()._op_edits.grad_for(self, bwd=bwd)  # record-groups: flat

    def _record_gradient(
        self,
        *,
        backward_pass_number: int,
        grad: torch.Tensor | None,
        transformed_grad: Any | None,
        shape: tuple[int, ...] | None,
        dtype: str | None,
        memory: int | None,
        timestamp: float,
    ) -> GradientRecord:
        """Append or replace the projected gradient record for one pass.

        Parameters
        ----------
        backward_pass_number:
            One-based global backward pass number.
        grad:
            Saved raw gradient tensor, if retained.
        transformed_grad:
            Saved transformed gradient tensor, if retained.
        shape:
            Observed raw gradient shape.
        dtype:
            Observed raw gradient dtype string.
        memory:
            Observed raw gradient memory in bytes.
        timestamp:
            Event timestamp.

        Returns
        -------
        GradientRecord
            Appended record.
        """

        records = self._slot("_grad_records")
        if records is None:
            records = []
            self._internal_set("_grad_records", records)
        records[:] = [
            record for record in records if record.backward_pass_number != backward_pass_number
        ]
        record = GradientRecord(
            owner=self,
            ordinal=len(records) + 1,
            backward_pass_number=backward_pass_number,
            grad=grad,
            transformed_grad=transformed_grad,
            shape=shape,
            dtype=dtype,
            memory=memory,
            timestamp=timestamp,
        )
        records.append(record)
        return record

    def _clear_gradient_records(self) -> None:
        """Clear projected per-pass gradient records in place."""

        records = self._slot("_grad_records")
        if records is None:
            self._internal_set("_grad_records", [])
            return
        records.clear()

    @property
    def grad_fn_label(self) -> str | None:
        """Stable GradFn label for this Op, or ``None`` when no GradFn was captured.

        Stored form of the cross-class reference: ``grad_fn`` resolves the
        TorchLens GradFn record, while ``grad_fn_label`` is the portable label.
        """

        grad_fn = self._autograd_grad_fn
        if grad_fn is None:
            return None
        return cast("str | None", getattr(grad_fn, "label", None))

    @property
    def layer(self) -> "Layer":
        """Parent Layer record resolved via ``self.trace.layers[self.layer_label]``.

        Raises
        ------
        KeyError
            When the parent Layer label cannot be resolved on the owning Trace.
        """

        trace = self._source_trace_or_error()
        return cast("Layer", trace.layers[self.layer_label])

    @property
    def args_summary(self) -> str | None:
        """Human-readable summary of this Op's positional arguments.

        The Op-level source for ``Layer.args_summary``. Returns ``None`` when no
        positional-argument information was captured.
        """

        return _summarize_call_args(self._call_saved_args, self._call_non_tensor_pos_args)

    @property
    def kwargs_summary(self) -> str | None:
        """Human-readable summary of this Op's keyword arguments.

        Op-level source for ``Layer.kwargs_summary``. Returns ``None`` when no
        keyword-argument information was captured.
        """

        return _summarize_call_kwargs(self._call_saved_kwargs, self._call_non_tensor_kwargs)

    @property
    def multi_output_type(self) -> type | str | None:
        """Python class of the multi-output container this Op came from."""

        return _op_relations.multi_output_type(self)

    @property
    def container(self) -> Any:
        """Return the runtime container view for this op, when available."""

        return container_from_op(self)

    @property
    def output_containers(self) -> tuple[Any, ...]:
        """Return output container views associated with this op."""

        return output_containers_from_op(self)

    @property
    def input_containers(self) -> tuple[Any, ...]:
        """Return input container views consumed by this op."""

        return input_containers_from_op(self)

    @property
    def containers(self) -> tuple[Any, ...]:
        """Return all input and output container views associated with this op."""

        return containers_from_op(self)

    @property
    def atomic_module_call_label(self) -> str | None:
        """Stable ModuleCall label for an atomic-module output Op, else ``None``.

        Stored form of the cross-class reference; ``atomic_module_call`` resolves
        the ModuleCall record.
        """

        return cast("str | None", self._module_context_atomic_module_call)

    @property
    def _graph_has_parents(self) -> bool:
        """Whether this layer has any parent layers."""
        return len(self.parents) > 0

    @property
    def edge_uses(self) -> tuple[EdgeUseRecord, ...]:
        """Per-edge parent-use records captured for this Op.

        Returns
        -------
        tuple[EdgeUseRecord, ...]
            Immutable view of the stored edge-use records, including repeated
            same-parent uses in separate argument slots.
        """

        return tuple(self._slot("_edge_uses") or ())

    def _own_label_spellings(self) -> set[str]:
        """Return every spelling a relation list may use for THIS op."""

        return _labels.own_label_spellings(self)

    @property
    def _graph_siblings(self) -> list[str]:
        """Layers sharing at least one parent (excluding output layers and this op)."""

        return _op_relations.siblings(self)

    @property
    def _graph_has_siblings(self) -> bool:
        """Whether this layer shares parents with other layers."""
        return len(self._graph_siblings) > 0

    @property
    def _graph_co_parents(self) -> list[str]:
        """Layers sharing at least one child (excluding output layers and this op)."""

        return _op_relations.co_parents(self)

    @property
    def _graph_has_co_parents(self) -> bool:
        """Whether this layer shares children with other layers."""
        return len(self._graph_co_parents) > 0

    @property
    def _control_flow_is_in_conditional_body(self) -> bool:
        """Whether this op ran inside a conditional arm (the Layer rule; test ops are outside)."""

        return bool(self._control_flow_branch_stack) or any(
            role.role == "body" for role in self._control_flow_in_conditionals or []
        )

    @_control_flow_is_in_conditional_body.setter
    def _control_flow_is_in_conditional_body(self, value: bool) -> None:
        """Set the cached conditional-body predicate used during postprocessing."""

        object.__setattr__(self, "_is_in_conditional_body", value)

    @_control_flow_is_in_conditional_body.deleter
    def _control_flow_is_in_conditional_body(self) -> None:
        """Delete the cached conditional-body predicate during cleanup."""

        try:
            object.__delattr__(self, "_is_in_conditional_body")
        except AttributeError:
            pass

    @property
    def has_params(self) -> bool:
        """Whether this operation used model parameters."""
        return len(self._param_barcodes) > 0

    @property
    def input_ops(self) -> "OpAccessor":
        """Accessor over graph-parent Op records in ``parents`` order."""

        return OpAccessor(_op_relations.input_ops_by_index(self))

    @input_ops.deleter
    def input_ops(self) -> None:
        """Ignore cleanup deletion for derived input Op access."""

    @property
    def _graph_buffer_source_ops(self) -> "OpAccessor":
        """Accessor over buffer source Ops that feed this Op."""

        return OpAccessor(_op_relations.buffer_source_ops_by_index(self))

    @_graph_buffer_source_ops.deleter
    def _graph_buffer_source_ops(self) -> None:
        """Ignore cleanup deletion for derived buffer source Op access."""

    @property
    def _graph_buffer_sink_ops(self) -> "OpAccessor":
        """Accessor over buffer sink Ops directly consuming this Op."""

        return OpAccessor(_op_relations.buffer_sink_ops_by_index(self))

    @_graph_buffer_sink_ops.deleter
    def _graph_buffer_sink_ops(self) -> None:
        """Ignore cleanup deletion for derived buffer sink Op access."""

    @property
    def input_activations(self) -> tuple[torch.Tensor | None, ...]:
        """Saved parent activations consumed by this Op, as references."""

        return _op_relations.input_activations(self)

    @input_activations.deleter
    def input_activations(self) -> None:
        """Ignore cleanup deletion for derived input activations."""

    @property
    def input_shapes(self) -> tuple[Any | None, ...]:
        """Shapes of saved parent activations in ``parents`` order."""

        return tuple(
            None if activation is None else activation.shape
            for activation in self.input_activations
        )

    @input_shapes.deleter
    def input_shapes(self) -> None:
        """Ignore cleanup deletion for derived input shapes."""

    @property
    def input_dtypes(self) -> tuple[torch.dtype | None, ...]:
        """Dtypes of saved parent activations in ``parents`` order."""

        return tuple(
            None if activation is None else activation.dtype
            for activation in self.input_activations
        )

    @input_dtypes.deleter
    def input_dtypes(self) -> None:
        """Ignore cleanup deletion for derived input dtypes."""

    @property
    def _cost_input_memory(self) -> Bytes:
        """Sum of activation bytes across saved graph-parent Ops (``op.cost.input_memory``)."""

        return Bytes(
            sum(
                int(record_member(parent, "activation_memory", 0) or 0)
                for parent in self.input_ops.values()
                if parent.has_saved_activation
            )
        )

    @_cost_input_memory.deleter
    def _cost_input_memory(self) -> None:
        """Ignore cleanup deletion for derived input memory."""

    @property
    def _graph_num_inputs(self) -> int:
        """Number of graph-parent input Ops."""

        return len(self.parents)

    @_graph_num_inputs.deleter
    def _graph_num_inputs(self) -> None:
        """Ignore cleanup deletion for derived input count."""

    @property
    def in_submodule(self) -> bool:
        """Whether this operation was computed inside a submodule."""
        return self.module is not None

    @property
    def gradient_transform(self) -> Callable[..., Any] | None:
        """Transform used for this Op's saved gradient, or ``None`` when unset.

        Mirrors ``activation_transform`` for the backward side. Reads the
        trace-level ``grad_transform`` that was applied to this Op's gradient.
        """

        trace = self._source_trace
        if trace is None:
            return None
        return cast("Callable[..., Any] | None", trace._options_grad_transform)

    @property
    def is_buffer_source(self) -> bool:
        """Whether this Op represents a buffer boundary (overwrites a buffer).

        Glossary name for the stored ``is_buffer`` flag.
        """

        return bool(self._graph_is_buffer)

    @property
    def has_saved_grad(self) -> bool:
        """Whether this Op's gradient was saved.

        Glossary name for the backward-side saved predicate; mirrors
        ``has_saved_activation``.
        """

        return bool(self.has_grad)

    @property
    def _streaming_label(self) -> str:
        """Best available label for sink/writer callbacks during or after postprocess."""

        return _labels.streaming_label(self)

    @property
    def source_trace(self) -> "Trace":
        """Back-reference to the owning Trace (stored as weakref).

        Never returns ``None``: an Op detached from its Trace (standalone
        pickle strips the weakref; cleanup clears it) refuses with the same
        typed ``RecordBindingError`` family as the collected-Trace case, so
        no ``None`` can escape behind the ``-> Trace`` signature and crash a
        caller untyped -- the ``Layer`` fix (1a2b715e) applied to the
        sibling record class it never reached.
        """
        ref = self._slot("_source_trace_ref")
        if ref is None:
            raise RecordBindingError(
                "This Op is not bound to a Trace (standalone pickle, "
                "cleanup, or a record never attached to a Trace)",
                code="record_not_bound",
                remedy="read the op through a live Trace accessor",
            )
        obj = ref()
        if obj is None:
            raise RecordBindingError(
                "the Trace this Op was read from has been garbage-collected: an Op holds its Trace weakly, so a chained read on a temporary (tl.trace(...)[key].out, t.fork().do(...)[key].out) loses it as soon as the record is taken",
                code="trace_reference_collected",
                remedy="keep the Trace (t = tl.trace(...)) or fork (f = t.fork().do(...)) in a variable while reading its records",
            )
        return cast("Trace", obj)

    @source_trace.setter
    def source_trace(self, value: "Trace | None") -> None:
        """Set the owning Trace back-reference.

        Parameters
        ----------
        value:
            Owning model log, or ``None`` to clear the reference.
        """
        self._source_trace_ref = weakref.ref(value) if value is not None else None

    def _source_trace_or_none(self) -> "Trace | None":
        """Owning Trace, or ``None`` when detached (internal quiet spelling)."""
        ref = self._slot("_source_trace_ref")
        obj = ref() if ref is not None else None
        return cast("Trace | None", obj)

    @property
    def _source_trace(self) -> "Trace | None":
        """Owning Trace, if bound and still alive (tolerant internal read).

        Property alias of :meth:`_source_trace_or_none` so both landed
        internal spellings stay valid.
        """

        return self._source_trace_or_none()

    def _source_trace_or_error(self) -> "Trace":
        """Return the owning Trace, or raise a detached-log error.

        Returns
        -------
        Trace
            Source Trace that owns this operation log.

        Raises
        ------
        AttributeError
            If this operation log is detached from its source Trace.
        """

        ref = self._slot("_source_trace_ref")
        source = ref() if ref is not None else None
        if source is None or getattr(source, "_loaded_from_bundle", False):
            raise AttributeError(
                "This Op is detached from its source Trace "
                "(perhaps loaded from disk or after cleanup). "
                "Use trace.do(label, transform) directly."
            )
        return cast("Trace", source)

    def do(
        self,
        edit: Any,
        /,
        *,
        model: Any = None,
        inputs: Any = None,
        intervention: Any = None,
    ) -> "Trace":
        """Apply an intervention to this op through the owning Trace."""

        return _doors.intervention()._op_edits.do(
            self, edit, model=model, inputs=inputs, intervention=intervention
        )

    def __tl_state_items__(self) -> Any:
        """Yield live state ``(field_name, value)`` pairs in declared order.

        The state-protocol hook. Enumeration follows the declared
        ``_OP_SLOT_NAMES`` order and reads through
        ``object.__getattribute__`` -- exactly what the former per-slot walk
        did -- so class-level compatibility overlays (the ancestor-bitset
        properties installed by ``backends/torch/ops.py``) keep
        materializing their public values into pickle/fork state instead of
        leaking compact internal encodings. Unset cells are skipped; a bare
        ``object.__new__`` shell with no bound store yields nothing.
        """

        getattribute = _object_getattribute
        for name, attr in _OP_STATE_ATTRS:
            try:
                yield name, getattribute(self, attr)
            except AttributeError:
                continue

    def __tl_state_restore__(self, mapping: dict[str, Any]) -> None:
        """Install ``mapping`` onto this op through the descriptor protocol.

        Binds a detached single-row store when this op is a bare shell
        (``state_new`` fork/scrub shells), then assigns exactly like the
        generic fallback did: one ``object.__setattr__`` per field, which
        resolves the generated field descriptors and the compatibility
        properties identically to the former slot layout.
        """

        _ensure_detached_store(self)
        storage_attrs = OP_STORAGE_ATTRS
        for field_name, field_value in mapping.items():
            _object_setattr(self, storage_attrs.get(field_name, field_name), field_value)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with weakrefs stripped."""
        state = dict(state_items(self))
        # Argument templates hold the model's live parameters. A pickled op has no
        # func, so its templates cannot drive replay; only a replayable capture keeps
        # them (its pickle form is unchanged), so a default trace pickles at its old size.
        if not getattr(self._source_trace_or_none(), "is_replayable", False):
            state["args_template"] = None
            state["kwargs_template"] = None
        state["_source_trace_ref"] = None
        state.pop("_facets_cache", None)
        state.pop("_receptive_field_cache", None)
        state.pop("_projective_field_cache", None)
        state["func"] = None
        state["grad_fn_handle"] = None
        # user transform callables (FieldPolicy.DROP) serialize to the
        # loaded-artifact form (None) -- a lambda transform= made pickle.dumps
        # crash on every op record while Trace.save succeeded on the same trace.
        if state.get("activation_transform") is not None:
            state["activation_transform"] = None
        if state.get("grad_transform") is not None:
            state["grad_transform"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state produced by ``__getstate__``."""
        _ensure_detached_store(self)
        read_tlspec_version(state, cls_name=type(self).__name__, cls=type(self))
        resolver_status_was_present = "resolver_status" in state
        default_fill_state(state, defaults=self._DEFAULT_FILL_STATE)
        # Repair present-but-wrong-typed container fields from legacy states
        # (e.g. `_param_barcodes` serialized as a `set` where a `list` is now
        # declared). `default_fill_state` only fills absent keys; this closes
        # the same gap `Trace.__setstate__` already closes for its own fields.
        coerce_container_typed_state(state, self._DEFAULT_FILL_STATE)
        _clear_property_backed_state_fields(state)
        if state.get("dtype_ref") is None:
            state["dtype_ref"] = _dtype_ref_or_none(state.get("dtype"))
        if state.get("device_ref") is None:
            state["device_ref"] = _device_ref_from_metadata(
                state.get("out"), state.get("output_device")
            )
        if not resolver_status_was_present:
            state["resolver_status"] = "resolved"
        for field_name in (
            "activation_memory",
            "transformed_activation_memory",
            "autograd_memory",
            "gradient_memory",
            "transformed_gradient_memory",
            "param_memory",
            "bytes_delta_at_call",
            "bytes_peak_at_call",
        ):
            if state.get(field_name) is not None:
                state[field_name] = Bytes(state[field_name])
        if state.get("func_duration") is not None:
            state["func_duration"] = as_duration(state["func_duration"])
        for field_name in ("flops_forward", "flops_backward"):
            if state.get(field_name) is not None:
                state[field_name] = Flops(state[field_name])
        object.__setattr__(self, "_construction_done", False)
        state_restore(self, state)
        object.__setattr__(self, "_construction_done", bool(state.get("_construction_done", True)))

    # ********************************************
    # *********** User-Facing Functions **********
    # ********************************************

    @property
    def facets(self) -> Any:
        """Return the lazy semantic facet view for this Op."""

        cache = self._slot("_facets_cache")
        if cache is None:
            from .._semantic.facets import FacetView

            cache = FacetView(self)
            object.__setattr__(self, "_facets_cache", cache)
        return cache

    @facets.deleter
    def facets(self) -> None:
        """Drop the cached semantic facet view for this Op."""

        try:
            object.__delattr__(self, "_facets_cache")
        except AttributeError:
            pass

    def __selection__(self) -> object:
        """Lift this op's whole output as an ACT selection term (one pass)."""

        from .._selection.selection import _selection_from_op

        return _selection_from_op(self)

    @property
    def receptive_field(self) -> "ReceptiveFieldView":
        """Return the lazy receptive-field query view for this Op."""

        return _doors.receptive_field()._op_fields.receptive_field(self)

    @receptive_field.deleter
    def receptive_field(self) -> None:
        """Drop the cached receptive-field query view for this Op."""

        try:
            object.__delattr__(self, "_receptive_field_cache")
        except AttributeError:
            pass

    @property
    def projective_field(self) -> "ReceptiveFieldView":
        """Return the lazy source-anchored projective-field query view."""

        return _doors.receptive_field()._op_fields.projective_field(self)

    @projective_field.deleter
    def projective_field(self) -> None:
        """Drop the cached source-anchored projective-field query view."""

        try:
            object.__delattr__(self, "_projective_field_cache")
        except AttributeError:
            pass

    @property
    def raw_label(self) -> str:
        """Return the public raw ordinal label without the raw-label suffix."""

        return _labels.public_raw_label(self._label_raw)

    @raw_label.setter
    def raw_label(self, value: Any) -> None:
        """Accept compatibility writes for the computed raw label."""

        if value not in {None, self.raw_label}:
            raise InvalidArgumentError(
                f"raw_label is derived from _label_raw and cannot be set to {value!r}",
                code="derived_field_assignment_invalid",
                remedy="do not assign raw_label; it is derived from _label_raw",
                field="raw_label",
            )

    @property
    def _source_arg_expressions(self) -> list[str]:
        """Return source expressions used for this op's call arguments.

        Returns
        -------
        list[str]
            Argument expressions resolved lazily from the captured call-site AST.
            Ambiguous or unavailable source returns an empty list.
        """

        cached = self._slot("_arg_expressions_cache")
        if cached is not None:
            return cast(list[str], cached)
        from .._postprocess.ast_branches import resolve_arg_expressions

        resolved = resolve_arg_expressions(self._source_frames, self.func_name)
        object.__setattr__(self, "_arg_expressions_cache", resolved)
        return resolved

    @_source_arg_expressions.setter
    def _source_arg_expressions(self, value: Any) -> None:
        """Set or clear the lazy argument-expression cache."""

        if value is None:
            try:
                object.__delattr__(self, "_arg_expressions_cache")
            except AttributeError:
                pass
            return
        object.__setattr__(self, "_arg_expressions_cache", list(value))

    @_source_arg_expressions.deleter
    def _source_arg_expressions(self) -> None:
        """Drop the cached argument-expression parse result."""

        try:
            object.__delattr__(self, "_arg_expressions_cache")
        except AttributeError:
            pass

    @property
    def _call_is_internally_initialized(self) -> bool:
        """Return whether this op is an internally initialized source op.

        Returns
        -------
        bool
            Alias of ``is_internal_source`` for tabular compatibility.
        """

        return bool(self._graph_is_internal_source)

    @_call_is_internally_initialized.setter
    def _call_is_internally_initialized(self, value: Any) -> None:
        """Accept compatibility writes for the computed internal-source alias."""

        if value is not None and bool(value) != bool(self._graph_is_internal_source):
            raise InvalidArgumentError(
                "is_internally_initialized is derived from is_internal_source and "
                f"cannot be set to {value!r}",
                code="derived_field_assignment_invalid",
                remedy="do not assign is_internally_initialized; it mirrors is_internal_source",
                field="is_internally_initialized",
            )

    # ********************************************
    # ************* Logging Functions ************
    # ********************************************

    def _copy_for_output(self, *, _store: Any = None) -> "Op":
        """Clone this op for immediate conversion into a synthetic output node."""

        return _op_save._copy_for_output(self, _store=_store)

    def _copy_with_shallow_fields(
        self,
        extra_shallow_fields: frozenset[str],
        *,
        _store: Any = None,
    ) -> "Op":
        """Clone this op while sharing fields the caller replaces immediately."""

        return _op_save._copy_with_shallow_fields(self, extra_shallow_fields, _store=_store)

    def _save_activation(
        self,
        t: torch.Tensor,
        t_args: list[Any] | tuple[Any, ...],
        t_kwargs: dict[str, Any],
        save_args: bool,
        activation_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Save the output tensor (and optionally args) for this operation."""

        _op_save.save_activation(self, t, (t_args, t_kwargs), save_args, activation_transform)

    def _save_call_args(
        self,
        t_args: list[Any] | tuple[Any, ...],
        t_kwargs: dict[str, Any],
        save_args: bool,
    ) -> None:
        """Store copies of the call's tensor args and kwargs, or clear them."""

        if save_args:
            self._call_has_saved_args = True
            self._internal_set("saved_args", [copy_arg_tree(arg) for arg in t_args])
            self._internal_set("saved_kwargs", {k: copy_arg_tree(v) for k, v in t_kwargs.items()})
        else:
            self._internal_set("saved_args", None)
            self._internal_set("saved_kwargs", None)

    def _log_tensor_grad(
        self,
        grad: torch.Tensor,
        prebuilt: "tuple[torch.Tensor | None, Any | None] | None" = None,
    ) -> None:
        """Save the grad tensor for this layer's output."""

        from ..backends.torch.tensor_tracking import _copy_grad_payload

        _op_save._log_tensor_grad(self, grad, _copy_grad_payload, prebuilt)

    def _apply_transform(
        self,
        tensor: torch.Tensor,
        transform: Callable[..., Any],
        *,
        transform_kind: str,
        streaming_active: bool,
    ) -> Any:
        """Apply a user transform with logging paused and rich error context."""

        return _op_save._apply_transform(
            self,
            tensor,
            transform,
            transform_kind=transform_kind,
            streaming_active=streaming_active,
        )

    def _transform_error_message(
        self,
        *,
        transform_kind: str,
        tensor: torch.Tensor,
        streaming_active: bool,
    ) -> str:
        """Build context for an out or grad transform failure."""

        return _op_save._transform_error_message(
            self, transform_kind=transform_kind, tensor=tensor, streaming_active=streaming_active
        )

    def _validate_train_mode_transform_output(
        self,
        raw_tensor: torch.Tensor,
        output: Any,
        *,
        transform_kind: str,
        transform: Any = None,
    ) -> None:
        """Validate differentiability requirements for train-mode transform outputs."""

        _op_save._validate_train_mode_transform_output(
            self, raw_tensor, output, transform_kind=transform_kind, transform=transform
        )

    def _validate_streaming_transform_output(
        self,
        output: Any,
        *,
        transform_kind: str,
        streaming_active: bool,
    ) -> None:
        """Validate transformed tensors before streaming bundle finalization."""

        _op_save._validate_streaming_transform_output(
            self, output, transform_kind=transform_kind, streaming_active=streaming_active
        )

    def _abort_streaming_writer(self, message: str) -> None:
        """Abort the active streaming writer when one is attached."""

        _op_save._abort_streaming_writer(self, message)

    def _stream_tensor_blob(
        self,
        writer: Any,
        *,
        tensor_field: str,
        pending_field: str,
        kind: str,
    ) -> None:
        """Stream one tensor field when present."""

        _op_save._stream_tensor_blob(
            self, writer, tensor_field=tensor_field, pending_field=pending_field, kind=kind
        )

    # ********************************************
    # ************* Fetcher Functions ************
    # ********************************************

    def _resolve_relation_record(self, label: str) -> "Op | None":
        """Resolve one relation label through the mainline, then orphans."""

        return _op_relations._resolve_relation_record(self, label)

    def get_children(self) -> list["Op"]:
        """Return child Op objects for this pass."""

        return _op_relations.get_children(self)

    def get_parents(self) -> list["Op"]:
        """Return parent Op objects for this pass."""

        return _op_relations.get_parents(self)

    def _repr_html_(self) -> str:
        """Return the notebook Op card.

        Identity / value / context zones with the PROOF row; cost wrappers
        pass through ``str()`` untouched; the budgeted array view rides the
        ported truncation protocol. Never raises: any internal failure
        degrades to a one-line ``card unavailable`` fragment, and the
        treescope-bridge suppression sentinel replaces the card exactly
        when the bridge just rendered this object.
        """

        from ..report._notebook.cards import op_repr_html

        return op_repr_html(self)

    def show(
        self,
        kind: Literal["auto", "heatmap", "channels", "rgb", "hist"] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Display this pass's saved out.

        Parameters
        ----------
        kind:
            Picture kind. ``None`` (the default) or ``"auto"`` chooses from tensor shape.
        **kwargs:
            Forwarded to the tensor display helper.

        Returns
        -------
        Any
            Matplotlib figure when plotting is available, otherwise a text
            fallback explaining why no plot was produced.
        """

        from ..viz._tensor_display import show_tensor

        return show_tensor(self, method=kind or "auto", **kwargs)

    @property
    def params(self) -> Any:
        """Access parameter metadata by address, short name, or index."""
        from .param import ParamAccessor

        param_dict = {pl.address: pl for pl in self._param_logs}
        return ParamAccessor(param_dict)

    def to_pandas(self) -> "pd.DataFrame":
        """Export this Op as a one-row pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``LAYER_PASS_LOG_FIELD_ORDER``.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        row = {
            field_name: getattr(self, OP_STORAGE_ATTRS.get(field_name, field_name))
            for field_name in LAYER_PASS_LOG_FIELD_ORDER
        }
        row["is_internally_initialized"] = bool(row["is_internally_initialized"])
        for field_name in (
            "min_distance_from_input",
            "max_distance_from_input",
            "min_distance_to_output",
            "max_distance_to_output",
        ):
            if row[field_name] is not None:
                row[field_name] = float(row[field_name])
        return pd.DataFrame([row], columns=LAYER_PASS_LOG_FIELD_ORDER)

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self) -> str:
        """Render a bounded Op display from captured metadata and cached values."""

        from ._value_repr import op_card

        return op_card(self)

    def __repr__(self) -> str:
        """Render a bounded Op display from captured metadata and cached values."""

        from ._value_repr import op_repr_line

        return op_repr_line(self)


# ---------------------------------------------------------------------------
# The facade: generated per-field data descriptors over (_core, _row)
#
# ``Op`` stores nothing per instance beyond the store handle and its row id.
# Each declared stored field is a real data descriptor (visible to dir(),
# debugger enumeration, and ``vars(Op)`` -- the ancestor-bitset overlay in
# ``backends/torch/ops.py`` captures these exactly as it captured the former
# slot member descriptors). ``_MISSING`` cells reproduce the exact unset-slot
# ``AttributeError`` shapes (message + ``name``/``obj`` on reads; bare-name
# args on deletes), so ``_slot()``, ``getattr`` defaults, and cleanup paths
# behave byte-identically.
# ---------------------------------------------------------------------------

# Bound member-descriptor accessors: C-speed reads of the two real slots that
# do NOT re-enter ``Op.__getattribute__`` (descriptor bodies run on every
# field touch, so each avoided Python-level re-entry is measurable).
_CORE_GET = Op.__dict__["_core"].__get__
_ROW_GET = Op.__dict__["_row"].__get__


def _ensure_detached_store(op: "Op") -> None:
    """Bind a fresh detached single-row store when ``op`` is a bare shell."""

    try:
        _CORE_GET(op)
    except AttributeError:
        _object_setattr(op, "_core", DetachedOpStore(_OP_STORE_LAYOUT))
        _object_setattr(op, "_row", 0)


def _detach_op_husk(op: "Op") -> None:
    """Rebind a cleared op to an empty detached row.

    Cleanup / removal paths clear every field and historically left an empty
    slotted husk; the facade equivalent also drops the reference to the
    shared row store so a user-held husk cannot pin the whole trace core.
    """

    _object_setattr(op, "_core", DetachedOpStore(_OP_STORE_LAYOUT))
    _object_setattr(op, "_row", 0)


class _OpField:
    """Data descriptor for one stored ``Op`` field backed by the row store."""

    __slots__ = ("_name", "_fid")

    def __init__(self, name: str, fid: int) -> None:
        """Bind the descriptor to its declared field name and column id."""

        self._name = name
        self._fid = fid

    def __repr__(self) -> str:
        """Return a diagnostic representation naming the backed field."""

        return f"<Op field descriptor {self._name!r}>"

    def __get__(self, op: Any, owner: Any = None) -> Any:
        """Read the backing cell; unset cells raise like an unset slot.

        A ``PooledCell`` (the duplicate/empty-container pooling) hydrates
        a fresh exact-type container on first read and caches it back, so
        identity is stable across reads and per-row in-place mutation stays
        isolated — the ``_FACT`` semantics.
        """

        if op is None:
            return self
        store = _CORE_GET(op)
        row = _ROW_GET(op)
        value = store.cell_get(row, self._fid)
        if value is _MISSING:
            name = self._name
            raise AttributeError(
                f"{type(op).__name__!r} object has no attribute {name!r}",
                name=name,
                obj=op,
            )
        if value.__class__ is PooledCell:
            value = value.hydrate()
            store.cell_set(row, self._fid, value)
        return value

    def __set__(self, op: Any, value: Any) -> None:
        """Write the backing cell (base while building, overlay after freeze)."""

        _CORE_GET(op).cell_set(_ROW_GET(op), self._fid, value)

    def __delete__(self, op: Any) -> None:
        """Delete the backing cell; unset cells raise like an unset slot."""

        if not _CORE_GET(op).cell_del(_ROW_GET(op), self._fid):
            raise AttributeError(self._name)


#: Internal storage encodings sanctioned to live in finished relation cells.
#: Extended at import time by compat overlays whose reads materialize
#: immutable views (the ancestor bitset in ``backends/torch/ops.py``); a
#: refresh re-run can assign such an encoding onto a detached-backed op.
_RELATION_CELL_ENCODINGS: tuple[type, ...] = ()


def register_relation_cell_encoding(encoding_type: type) -> None:
    """Sanction ``encoding_type`` as a finished relation-cell storage value."""

    global _RELATION_CELL_ENCODINGS
    if encoding_type not in _RELATION_CELL_ENCODINGS:
        _RELATION_CELL_ENCODINGS = (*_RELATION_CELL_ENCODINGS, encoding_type)


class _RelationViewField(_OpField):
    """Descriptor for one immutable-view relation field.

    While a shared ``OpRowStore`` is BUILDING, writes stage raw mutable
    containers (postprocess mutates them in place). Once the store is sealed
    — and always on detached single-row stores (copy/pickle/fork/loaded) —
    writes normalize ``list``/``set`` values to the field's immutable view
    type, so a finished record can never re-expose a mutable relation
    container regardless of which write path assigned it.
    """

    __slots__ = ("_view_type",)

    def __init__(self, name: str, fid: int, view_type: type) -> None:
        """Bind the descriptor with its immutable view type."""

        super().__init__(name, fid)
        self._view_type = view_type

    def __set__(self, op: Any, value: Any) -> None:
        """Write the cell, normalizing to the view type once finished.

        "Finished" is any of: a sealed store, a store whose relation freeze
        already ran (preview backends convert without sealing), or a detached
        single-row store. Building-phase writes stay raw so postprocess can
        keep mutating its staging containers in place.

        Finished-store assignment is CLOSED over container types: any
        ``list``/``set``/``tuple``/``frozenset`` INSTANCE (subclasses
        included — an exact-type check let a mutable subclass bypass the
        view) normalizes to the field's view type, ``None`` passes through,
        and every other value raises ``TypeError`` so a finished record can
        never re-expose a mutable relation container regardless of which
        write path assigned it.
        """

        store = _CORE_GET(op)
        if store.frozen or store.dataflow_edges is not None or store.__class__ is DetachedOpStore:
            if isinstance(value, (list, set, frozenset, tuple)):
                if value.__class__ is not self._view_type:
                    value = self._view_type(value)
            elif value is not None and not isinstance(value, _RELATION_CELL_ENCODINGS):
                raise ArgumentTypeError(
                    f"cannot assign {type(value).__name__!r} to finished relation "
                    f"field {self._name!r}; expected list/set/tuple/frozenset "
                    f"(normalized to {self._view_type.__name__}) or None",
                    code="relation_assignment_type_invalid",
                    remedy="assign a list/set/tuple/frozenset or None to relation fields",
                    field=self._name,
                )
        store.cell_set(_ROW_GET(op), self._fid, value)


class _GroupViewField(_RelationViewField):
    """Descriptor for the two group-membership fields.

    A finished cell holds THE one shared ``GroupRef`` of its membership
    group; reads resolve to the group's cached immutable view (``frozenset``
    for ``equivalent_ops``, ``tuple`` for ``recurrent_ops``) — O(1) and LIVE
    through removal scrub, which rebinds the group row once for every
    member. This natively replaces the historical copy-on-read barrier
    (a fresh mutable copy per read, O(group) each): immutable views cannot
    alias-corrupt the group, so sharing is safe by construction.
    """

    __slots__ = ()

    def __get__(self, op: Any, owner: Any = None) -> Any:
        """Read the cell, resolving group refs to their live view."""

        if op is None:
            return self
        value = _CORE_GET(op).cell_get(_ROW_GET(op), self._fid)
        if value.__class__ is GroupRef:
            return value.view()
        if value is _MISSING:
            name = self._name
            raise AttributeError(
                f"{type(op).__name__!r} object has no attribute {name!r}",
                name=name,
                obj=op,
            )
        return value


class _DataflowField(_RelationViewField):
    """Descriptor for the two CSR-backed dataflow fields.

    A ``_CSR`` cell means the value lives in the store's edge-occurrence
    table: the first read rematerializes the interned tuple view and caches
    it back into the row (row cell on small sealed stores, sparse overlay on
    transposed ones), so identity is stable across reads and uninspected
    rows retain no per-row container.
    """

    __slots__ = ()

    def __get__(self, op: Any, owner: Any = None) -> Any:
        """Read the cell, rematerializing CSR-backed views on demand."""

        if op is None:
            return self
        store = _CORE_GET(op)
        row = _ROW_GET(op)
        value = store.cell_get(row, self._fid)
        if value is _CSR:
            value = materialize_dataflow_view(store, row, self._name)
            store.cell_set(row, self._fid, value)
            return value
        if value is _MISSING:
            name = self._name
            raise AttributeError(
                f"{type(op).__name__!r} object has no attribute {name!r}",
                name=name,
                obj=op,
            )
        return value


class _FactField(_OpField):
    """Descriptor for one shared-fact field (FunctionCall/ParamAlias blocks).

    A ``_FACT`` cell means the value lives in the store's shared fact block:
    the first read hydrates the field's exact public container type for THIS
    row and caches it back, so identity is stable across reads, per-row
    in-place mutation stays isolated (a fresh container per row, exactly the
    earlier per-row semantics), and uninspected rows retain no per-row container. A
    direct write replaces the sentinel with a per-row cell value and never
    mutates the shared block.
    """

    __slots__ = ()

    def __get__(self, op: Any, owner: Any = None) -> Any:
        """Read the cell, hydrating shared-fact sentinels on demand."""

        if op is None:
            return self
        store = _CORE_GET(op)
        row = _ROW_GET(op)
        value = store.cell_get(row, self._fid)
        if value is _FACT:
            value = store.fact_blocks.hydrate(row, self._name)
            store.cell_set(row, self._fid, value)
            return value
        if value is _MISSING:
            name = self._name
            raise AttributeError(
                f"{type(op).__name__!r} object has no attribute {name!r}",
                name=name,
                obj=op,
            )
        return value


def _install_op_field_descriptors() -> None:
    """Install the per-field data descriptors on the ``Op`` class.

    Most stored fields get a plain ``_OpField``; the declared relation
    families get view-normalizing descriptors (``_RelationViewField``), the
    dataflow pair additionally rematerializes from the CSR
    (``_DataflowField``), and the shared-fact fields hydrate from the
    fact blocks (``_FactField``).
    """

    tuple_view_names = frozenset(OP_TUPLE_VIEW_FIELDS)
    frozenset_view_names = frozenset(OP_FROZENSET_VIEW_FIELDS + OP_BITSET_VIEW_FIELDS)
    dataflow_names = frozenset(OP_DATAFLOW_FIELDS)
    existing = vars(Op)
    for fid, name in enumerate(_OP_SLOT_NAMES):
        attr = OP_STORAGE_ATTRS.get(name, name)
        if attr in existing:
            raise RuntimeError(f"Op facade collision: {attr!r} is already defined on Op")
        if name in dataflow_names:
            descriptor: _OpField = _DataflowField(name, fid, tuple)
        elif name in OP_GROUP_VIEW_FIELDS:
            descriptor = _GroupViewField(name, fid, OP_GROUP_VIEW_FIELDS[name])
        elif name in tuple_view_names:
            descriptor = _RelationViewField(name, fid, tuple)
        elif name in frozenset_view_names:
            descriptor = _RelationViewField(name, fid, frozenset)
        elif name in OP_FACT_FIELDS:
            descriptor = _FactField(name, fid)
        else:
            descriptor = _OpField(name, fid)
        setattr(Op, attr, descriptor)


_install_op_field_descriptors()

#: The generated field descriptor classes, for :func:`release_op_row`'s facade
#: check. No descriptor class overrides ``_OpField.__delete__``.
_ROW_RELEASE_FIELD_CLASSES = frozenset(
    {_OpField, _RelationViewField, _GroupViewField, _DataflowField, _FactField}
)
bind_op_facade(Op, (attr for _, attr in _OP_STATE_ATTRS), _ROW_RELEASE_FIELD_CLASSES)


# The tlspec v8 coordinated bump retired this class's pre-release
# registrations (site_key, edge_substitutions, edge_replacement_stamps);
# their persisting policies are declared directly in _FIELD_POLICY above.
