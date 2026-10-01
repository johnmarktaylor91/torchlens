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

import copy
import hashlib
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    cast,
)

import torch

from .._errors import (
    ArgumentTypeError,
    InvalidArgumentError,
    MutatedReferenceError,
    PayloadUnavailableError,
    RecordBindingError,
    TorchLensPostfuncError,
)
from .._io import (
    TLSPEC_VERSION,
    FieldPolicy,
    TorchLensIOError,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from .._save_budget import SaveBudgetExceededError
from .._state import pause_logging
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
from .._training_validation import _NON_GRAD_DTYPES, TrainingModeConfigError
from .._transport import digest_byte_view
from ..backends.torch._tl import mark_detached_saved_activation
from ..constants import ARG_EXPRESSIONS_FIELD, LAYER_PASS_LOG_FIELD_ORDER, RAW_LABEL_SUFFIX
from ..intervention.errors import DirectActivationWriteWarning
from ..intervention.types import (
    LAYER_PASS_LOG_FIELD_FORK_POLICY,
    EdgeUseRecord,
    FunctionRegistryKey,
)
from ..ir.refs import DeviceRef, DtypeRef
from ..quantities import (
    Bytes,
    Duration,
    Flops,
    Macs,
    as_bytes,
    as_duration,
    as_flops,
    as_macs,
)
from ..selection import _SelectionOperand
from ..utils._torch_compat import tensor_version_or_none
from ..utils.arg_handling import copy_arg_tree
from ..utils.display import tensor_stats_summary
from ..utils.tensor_utils import (
    SaveMode,
    concatenate_batch_tensors,
    copy_tensor_payload,
    get_memory_amount,
    get_memory_amount_from_metadata,
    is_functorch_wrapped_tensor,
    print_override,
    safe_to,
)
from ._accessor_base import Accessor
from ._backend_capability_guards import raise_if_no_backward_capture
from ._repr import format_config_items, format_shape_list
from ._state_adapter import state_items, state_restore
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
        "interventions",
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
# ``equivalent_ops``/``recurrent_ops`` historically lived here too as
# copy-on-read fields (a fresh mutable copy per read, protecting the ONE
# canonical container shared by every group member from alias corruption).
# The M7 live group views replace that barrier natively: cells hold a shared
# ``GroupRef`` and reads resolve to the group's cached IMMUTABLE view
# (``_GroupViewField``), so sharing is alias-safe without per-read copies.
_INTERCEPTED_READ_FIELDS = _LAZY_READ_FIELDS
_WARNED_REFERENCE_SAVE_MODE = False
_LAYER_PASS_LOG_DEFAULT_FILL: dict[str, Any] = {
    "_source_trace_ref": None,
    "out_ref": None,
    "grad_ref": None,
    "_pending_blob_id": None,
    "_pending_transformed_out_blob_id": None,
    "_pending_grad_blob_id": None,
    "_pending_transformed_grad_blob_id": None,
    "annotations": {},
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
    "intervention_replaced": False,
    "interventions": [],
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
# `LAYER_PASS_LOG_FIELD_ORDER`. Same defect and fix as
# `trace._MODEL_LOG_CONTAINER_DEFAULTS`: the blanket ``{field: None}`` base
# below makes an absent (legacy/partial-state) container field restore as
# ``None`` instead of its declared list/dict/set/tuple, which then crashes real
# consumer code (membership/iteration in ``finalization.py``,
# ``loop_detection.py``, ``invariants.py``). Plain builtin types are used so
# ``coerce_container_typed_state`` (called from ``Op.__setstate__``) also
# repairs a present-but-wrong-typed legacy value (e.g. a ``set`` where a
# ``list`` is now declared).
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
    # Relation view fields (M6, JMT-FORK-1): the declared restore type is the
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
# The M5 seam: ``_OP_SLOT_NAMES`` is no longer a ``__slots__`` tuple but the
# declared stored-field universe of the Op row store. One shared layout maps
# each stored field to its column id; ``Op`` itself carries only
# ``(_core, _row)`` and one generated data descriptor per stored field (see
# ``_OpField`` / ``_install_op_field_descriptors`` at the bottom of this
# module). All storage magic (``_slot``, ``_internal_set``, the ancestor
# bitset overlays in ``backends/torch/ops.py``, ``state_items`` /
# ``state_restore``) already speaks the descriptor protocol, so behavior is
# unchanged while the physical cells live in per-trace columns.
_OP_STORE_LAYOUT = OpStoreLayout(_OP_SLOT_NAMES)
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

# Exact classes that are safe to share.  Exact-class (not isinstance) matching
# keeps subclasses out, so a pooled value always has the original's type.
# ``torch.dtype``/``torch.device`` are immutable value objects (process
# singletons for dtypes), so sharing references is safe; they appear inside
# per-op config dicts such as ``func_autocast_state``.
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
        "interventions",
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
_POOLED_SLOTS = tuple(name for name in _OP_SLOT_NAMES if name not in _UNPOOLED_SLOTS)
# Depth of container nesting the pass descends into.  Every field that carries
# real repetition (ancestor sets, label lists, config dicts) is at depth 0-1.
_POOL_MAX_DEPTH = 2

# ---------------------------------------------------------------------------
# Mutable-container CELL pooling (the M14 memory slice).
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
        "interventions",
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

_DEEPCOPY_IMMUTABLE_TYPES = (
    str,
    bytes,
    int,
    float,
    bool,
    type(None),
    torch.dtype,
    torch.device,
    Bytes,
    Duration,
    Flops,
)

_OUTPUT_NODE_REPLACED_FIELDS = frozenset(
    {
        "_edge_uses",
        "_label_raw",
        "_layer_label_raw",
        "_param_barcodes",
        "_param_logs",
        "activation_memory",
        "annotations",
        "arg_names",
        "atomic_module_call",
        "autograd_memory",
        "bytes_delta_at_call",
        "bytes_peak_at_call",
        "children",
        "code_context",
        "container_path",
        "container_spec",
        "dropped_edge_tensor_args",
        "dtype",
        "equivalence_class",
        "equivalent_ops",
        "func",
        "func_config",
        "func_duration",
        "func_name",
        "func_non_tensor_args",
        "func_rng_states",
        "grad_fn_class_name",
        "has_children",
        "has_out_variations",
        "has_output_descendant",
        "input_to_module_calls",
        "internal_source_parents",
        "intervention_replaced",
        "interventions",
        "io_role",
        "is_atomic_module",
        "is_buffer",
        "is_final_output",
        "is_input",
        "is_internal_source",
        "is_module_output",
        "is_output",
        "is_transform",
        "layer_type",
        "module",
        "module_call_stack",
        "modules",
        "non_tensor_kwargs",
        "non_tensor_pos_args",
        "num_args_total",
        "num_autograd_tensors",
        "num_kwargs",
        "num_params",
        "num_params_frozen",
        "num_params_trainable",
        "num_passes",
        "num_pos_args",
        "out",
        "out_versions_by_child",
        "output_descendants",
        "output_of_module_calls",
        "output_of_modules",
        "param_memory",
        "param_shapes",
        "parent_arg_positions",
        "parent_param_ops",
        "parent_params",
        "parents",
        "pass_index",
        "raw_index",
        "recurrent_ops",
        "saved_args",
        "saved_kwargs",
        "shape",
        "transform_chain",
        "transform_config",
        "transform_fn_name",
        "transform_fn_qualname",
        "transform_fn_source",
        "transform_kind",
        "transformed_activation_memory",
        "transformed_out",
        "transformed_out_dtype",
        "transformed_out_shape",
        "unattributed_tensor_args",
        "var_names",
    }
)


def _copy_op_field_value(value: Any) -> Any:
    """Copy one Op field without deep-copy dispatch for immutable values.

    Parameters
    ----------
    value:
        Stored field value.

    Returns
    -------
    Any
        The original immutable value or an independent deep copy.
    """

    if isinstance(value, _DEEPCOPY_IMMUTABLE_TYPES):
        return value
    if isinstance(value, tuple) and all(
        isinstance(member, _DEEPCOPY_IMMUTABLE_TYPES) for member in value
    ):
        return value
    return copy.deepcopy(value)


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


def _count_container_ids(
    value: Any,
    id_counts: dict[int, int],
    expanded_ids: set[int],
    depth: int,
) -> None:
    """Count every exact builtin MUTABLE container id reachable from one cell.

    The alias census behind the pooling guard: a container whose id is seen
    more than once across ALL swept cells is aliased in-store, and replacing
    any of its cell appearances would break the alias for later in-place
    mutation, so such cells never pool.

    Only ``dict``/``defaultdict``/``list``/``set`` ids matter (the pool key
    visits exactly those). Hashability bounds the walk: dict KEYS and
    ``set``/``frozenset`` members must be hashable, so no exact builtin
    mutable container can hide below them — dict values, list members, and
    tuple members are the only recursion edges.
    """

    cls = value.__class__
    if cls is dict or cls is defaultdict:
        oid = id(value)
        id_counts[oid] = id_counts.get(oid, 0) + 1
        if depth < 6 and oid not in expanded_ids:
            expanded_ids.add(oid)
            for member in value.values():
                _count_container_ids(member, id_counts, expanded_ids, depth + 1)
    elif cls is list:
        oid = id(value)
        id_counts[oid] = id_counts.get(oid, 0) + 1
        if depth < 6 and oid not in expanded_ids:
            expanded_ids.add(oid)
            for member in value:
                _count_container_ids(member, id_counts, expanded_ids, depth + 1)
    elif cls is set:
        oid = id(value)
        id_counts[oid] = id_counts.get(oid, 0) + 1
    elif cls is tuple and depth < 6 and id(value) not in expanded_ids:
        expanded_ids.add(id(value))
        for member in value:
            _count_container_ids(member, id_counts, expanded_ids, depth + 1)


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

    # ONE scan over every cell: the alias census recurses only into cells
    # whose class can hold a container (everything else is skipped by an
    # inline class test — the vast majority of cells are scalars/None), and
    # the same scan collects the pool candidates (container-classed cells at
    # sanctioned fids) and the singleton-compaction sites (kind-table list
    # cells), so neither later pass re-walks rows x fields.
    id_counts: dict[int, int] = {}
    expanded_ids: set[int] = set()
    cell_sites: list[tuple[Any, int, Any]] = []
    singleton_sites: list[tuple[Any, int, int, Any, Any]] = []
    any_swept = False
    for store, fids in stores:
        rows = store.rows_building()
        if rows is None:
            continue
        any_swept = True
        fid_set = None if fids is None else frozenset(fids)
        for row_idx, row_cells in enumerate(rows):
            for fid, value in enumerate(row_cells):
                cls = value.__class__
                if cls is tuple:
                    _count_container_ids(value, id_counts, expanded_ids, 0)
                    continue
                if not (cls is dict or cls is list or cls is set or cls is defaultdict):
                    continue
                _count_container_ids(value, id_counts, expanded_ids, 0)
                if fid_set is None or fid in fid_set:
                    cell_sites.append((row_cells, fid, value))
                if fid_set is None and cls is list:
                    singleton_sites.append((store, row_idx, fid, row_cells, value))
    if not any_swept:
        return
    candidates: list[tuple[Any, int, Any, Any]] = []
    key_counts: dict[Any, int] = {}
    for row_cells, fid, value in cell_sites:
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
    # Singleton-label compaction (M14 slice 2, kind tables only): a
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
    if cls in _POOLED_CLASSES:
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
    if cls in _POOLED_CLASSES:
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


def _shape_or_none(value: Any) -> tuple[int, ...] | None:
    """Return a tensor shape tuple, or ``None`` for non-tensor values."""

    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _dtype_or_none(value: Any) -> torch.dtype | None:
    """Return a tensor dtype, or ``None`` for non-tensor values."""

    return value.dtype if isinstance(value, torch.Tensor) else None


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


def _memory_or_none(value: Any) -> Bytes | None:
    """Return tensor memory in bytes, or ``None`` for non-tensor values."""

    if not isinstance(value, torch.Tensor):
        return None
    return as_bytes(get_memory_amount_from_metadata(value, tuple(value.shape), value.dtype))


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
    backward_pass_index:
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

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "owner": FieldPolicy.WEAKREF_STRIP,
        "ordinal": FieldPolicy.KEEP,
        "backward_pass_index": FieldPolicy.KEEP,
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
        backward_pass_index: int,
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
        backward_pass_index:
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
        self.backward_pass_index = backward_pass_index
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

    def for_pass(self, pass_index: int) -> GradientRecord:
        """Return the gradient record for a one-based backward pass number."""

        matches = [record for record in self._list if record.backward_pass_index == pass_index]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise InvalidArgumentError(
                f"Multiple gradient records participated in pass {pass_index}; "
                "use positional indexing on .grads",
                code="gradient_pass_ambiguous",
                remedy="use positional indexing on .grads to pick one record",
                pass_index=pass_index,
            )
        available = [record.backward_pass_index for record in self._list]
        raise KeyError(
            f"Gradient record for backward pass {pass_index} not found; participated in "
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


def apply_transform(
    *,
    label: str | None,
    tensor: torch.Tensor,
    transform: Callable[..., Any],
    transform_kind: str,
    streaming_active: bool = False,
    raw_label: str | None = None,
    func_name: str | None = None,
) -> Any:
    """Apply a user transform with logging paused and contextual errors.

    Parameters
    ----------
    label:
        Raw layer label for error context, or ``None`` when unavailable.
    tensor:
        Raw tensor passed to the user transform.
    transform:
        Callable applied to ``tensor``.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active.
    raw_label:
        Raw layer label for error context when it differs from ``label``.
    func_name:
        Function name for error context.

    Returns
    -------
    Any
        Value returned by ``transform``.
    """

    # R36: a cpu_async payload may still be an in-flight pinned buffer; a
    # user transform is a host-side byte read and must never observe partial
    # bytes. No-op unless async fence events are actually pending.
    from ..utils.tensor_utils import synchronize_pending_cpu_async_copies

    synchronize_pending_cpu_async_copies()
    try:
        with pause_logging():
            return transform(tensor)
    except Exception as exc:
        raise TorchLensPostfuncError(
            transform_error_message(
                label=label,
                raw_label=raw_label,
                func_name=func_name,
                tensor=tensor,
                transform_kind=transform_kind,
                streaming_active=streaming_active,
            )
        ) from exc


def transform_error_message(
    *,
    label: str | None,
    raw_label: str | None = None,
    func_name: str | None = None,
    tensor: torch.Tensor,
    transform_kind: str,
    streaming_active: bool,
) -> str:
    """Build context for an out or grad transform failure.

    Parameters
    ----------
    label:
        Raw layer label for error context, or ``None`` when unavailable.
    raw_label:
        Raw layer label for error context when it differs from ``label``.
    func_name:
        Function name for error context.
    tensor:
        Raw tensor passed to the transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active.

    Returns
    -------
    str
        Contextual error message.
    """

    return (
        f"{transform_kind}_transform raised for layer {label} "
        f"(raw={raw_label or label}, func={func_name}, "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"streaming_active={streaming_active})."
    )


def validate_train_mode_transform_output(
    *,
    raw_tensor: torch.Tensor,
    transformed_tensor: Any,
    transform_kind: str,
    backward_ready: bool,
    label: str | None = None,
) -> None:
    """Validate differentiability requirements for train-mode transform outputs.

    Parameters
    ----------
    raw_tensor:
        Raw tensor passed to the transform.
    transformed_tensor:
        Value returned by the transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    backward_ready:
        Whether TorchLens is preserving autograd graph connectivity.
    label:
        Raw layer label for error context, or ``None`` when unavailable.

    Returns
    -------
    None
        Raises if the transformed value violates train-mode requirements.
    """

    if not backward_ready or not raw_tensor.requires_grad:
        return
    if not isinstance(transformed_tensor, torch.Tensor):
        raise TrainingModeConfigError(
            f"{transform_kind}_transform must return a torch.Tensor while backward_ready=True "
            f"for layer {label}. "
            "Remedy: return a differentiable torch.Tensor from the transform.",
            code="transform_not_differentiable",
        )
    if transformed_tensor.dtype in _NON_GRAD_DTYPES:
        raise TrainingModeConfigError(
            f"backward_ready=True with non-grad dtype {transformed_tensor.dtype} on layer "
            f"{label}. Integer and bool dtypes cannot propagate grads. "
            "Remedy: return a floating-dtype tensor from the transform.",
            code="transform_not_differentiable",
        )
    if not transformed_tensor.requires_grad or (
        transformed_tensor.grad_fn is None and transformed_tensor is not raw_tensor
    ):
        raise TrainingModeConfigError(
            f"{transform_kind}_transform returned a tensor disconnected from the autograd "
            "graph (grad_fn is None) while backward_ready=True. The transformed out "
            "must remain differentiable. "
            "Remedy: keep the transform on the autograd graph (no detach/no_grad).",
            code="transform_not_differentiable",
        )


def validate_streaming_transform_output(
    *,
    transformed_tensor: Any,
    transform_kind: str,
    streaming_active: bool,
    label: str | None = None,
) -> None:
    """Validate transformed tensors before streaming bundle finalization.

    Parameters
    ----------
    transformed_tensor:
        Value returned by the user transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active for this trace.
    label:
        Raw layer label for error context, or ``None`` when unavailable.

    Returns
    -------
    None
        Raises if streaming cannot serialize the transformed value.
    """

    if not streaming_active:
        return
    if not isinstance(transformed_tensor, torch.Tensor):
        raise TorchLensIOError(
            f"Streaming save requires {transform_kind}_transform outputs to be "
            f"torch.Tensor instances, but layer {label} produced "
            f"{type(transformed_tensor).__name__}."
        )
    if transformed_tensor.layout != torch.strided:
        raise TorchLensIOError(
            f"Streaming save does not support sparse {transform_kind}_transform outputs "
            f"for layer {label}."
        )


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
    entry._internal_set("transformed_out_shape", _shape_or_none(entry.transformed_out))
    entry._internal_set("transformed_out_dtype", _dtype_or_none(entry.transformed_out))
    entry._internal_set("transformed_activation_memory", _memory_or_none(entry.transformed_out))


def _warn_reference_save_mode_once() -> None:
    """Emit the reference-mode mutation warning once per process."""

    global _WARNED_REFERENCE_SAVE_MODE
    if _WARNED_REFERENCE_SAVE_MODE:
        return
    warnings.warn(
        "save_mode='reference' stores source tensors by reference; reading a mutated "
        "saved tensor raises MutatedReferenceError.",
        UserWarning,
        stacklevel=3,
    )
    _WARNED_REFERENCE_SAVE_MODE = True


def _effective_activation_save_mode(
    trace: "Trace | None",
    *,
    func_name: str | None,
    is_inplace: bool = False,
) -> SaveMode:
    """Return the effective saved-activation mode for one operation."""

    save_mode = cast(SaveMode, getattr(trace, "save_mode", "copy"))
    if save_mode not in {"copy", "reference", "view", "cpu_async"}:
        raise InvalidArgumentError(
            "save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'; "
            f"received {save_mode!r}",
            code="save_mode_invalid",
            remedy="set save_mode to 'copy', 'reference', 'view', or 'cpu_async'",
            argument="save_mode",
        )
    if save_mode == "reference":
        _warn_reference_save_mode_once()
        if is_inplace or (func_name is not None and func_name.endswith("_")):
            return "copy"
    return save_mode


def _stamp_reference_out(
    annotations: dict[str, Any], raw_out: torch.Tensor, save_mode: SaveMode
) -> None:
    """Store reference-mode mutation metadata for a saved output."""

    if save_mode != "reference":
        return
    annotations["save_mode"] = "reference"
    annotations["saved_out_version"] = tensor_version_or_none(raw_out)


def _validate_reference_out_not_mutated(state: dict[str, Any]) -> None:
    """Raise if a reference-mode saved output has changed since capture."""

    annotations = state.get("annotations") or {}
    if annotations.get("save_mode") != "reference":
        return
    out = state.get("out")
    if not isinstance(out, torch.Tensor):
        return
    saved_version = annotations.get("saved_out_version")
    current_version = tensor_version_or_none(out)
    if saved_version is not None and current_version != saved_version:
        label = state.get("label") or state.get("layer_label") or state.get("_label_raw")
        raise MutatedReferenceError(
            f"saved reference out for op {label!r} was mutated after capture "
            f"(saved _version={saved_version}, current _version={current_version}). "
            "Use save_mode='copy' for isolated saved tensors."
        )


def _tensor_content_hash(value: torch.Tensor) -> str:
    """Return a CPU content hash for a tensor.

    Parameters
    ----------
    value:
        Tensor to hash.

    Returns
    -------
    str
        SHA-256 digest.

    Notes
    -----
    The digest frames the LOGICAL dtype so a bfloat16 tensor can never
    collide with the float32 tensor of the same values (content-mode dedup
    aliasing across dtypes). The payload is hashed through the buffer
    protocol (no whole-payload ``tobytes`` copy), the transport is the
    shared zero-copy-when-possible ``to_cpu_contiguous`` (r7 b5 R35-1: the
    old ``safe_copy(...).cpu().contiguous()`` paid one unconditional full
    clone for an already-contiguous CPU tensor and materialized twice for a
    CUDA/permuted source), and bf16 hashes its OWN bytes -- the uint8
    reinterpret view needs no numpy-transport upcast (R35 fable: the
    bf16->f32 copy was pointless once the logical dtype was framed; this
    digest is process-local, so the byte change is invisible).
    """

    if is_functorch_wrapped_tensor(value):
        return f"functorch_wrapped_tensor:{id(value)}"

    with pause_logging():
        # r8 R35: conj/neg resolution + the uint8 reinterpret live in the
        # ONE transport authority (``_transport.digest_byte_view``); the
        # local resolve guard this site pioneered moved there so the other
        # digest sites cannot drift from it.
        logical_dtype = str(value.dtype)
        shape = tuple(value.shape)
        payload = digest_byte_view(value)
        hasher = hashlib.sha256()
        hasher.update(repr((shape, logical_dtype)).encode("utf-8"))
        hasher.update(payload)
    return hasher.hexdigest()


def _dedup_cached_identity_out(
    trace: "Trace | None",
    source_tensor: torch.Tensor,
    annotations: dict[str, Any],
    save_arg_values: bool,
) -> torch.Tensor | None:
    """Return the already-saved payload for this live source, or ``None``.

    Parameters
    ----------
    trace:
        Trace that owns the per-pass dedup caches.
    source_tensor:
        Live output tensor about to be copied for retention.
    annotations:
        Mutable annotation dictionary for the saved output.
    save_arg_values:
        Whether argument values are being saved (disables activation dedup).

    Notes
    -----
    Pre-copy identity probe: the historical order CLONED the payload first
    and only consulted the identity cache afterwards, discarding the fresh
    clone on every hit — a full wasted payload copy per repeated-source save
    (dedup-after-copy ordering). Hit semantics, annotations, and the miss
    path (which still inserts post-copy via
    :func:`_dedup_saved_activation_out`) are unchanged.
    """

    if trace is None or save_arg_values or source_tensor.is_meta:
        return None
    if getattr(trace, "_out_dedup_mode", "identity") != "identity":
        return None
    identity_cache = getattr(trace, "_out_identity_cache", None)
    if identity_cache is None:
        return None
    source_key = id(source_tensor)
    from ..backends.torch.completeness_witness import internal_scalar_read

    with internal_scalar_read():
        source_version = tensor_version_or_none(source_tensor)
    cached = identity_cache.get(source_key)
    if cached is None:
        return None
    cached_source, cached_label, cached_out, cached_version, cached_ordinal = cached
    if cached_source is source_tensor and cached_version == source_version:
        # B3R4-R21-1: the annotation carries the trace-local dense dedup
        # ordinal, never the raw ``id()`` bookkeeping key -- a memory address
        # in a persisted field made same-program artifacts byte-differ per
        # process and exposed a meaningless public value.
        annotations["dedup_source_id"] = cached_ordinal
        annotations["dedup_source_version"] = source_version
        annotations["dedup_reference_label"] = cached_label
        return cast(torch.Tensor, cached_out)
    return None


def _dedup_saved_activation_out(
    trace: "Trace | None",
    source_tensor: torch.Tensor,
    raw_out: torch.Tensor,
    label: str,
    annotations: dict[str, Any],
    save_arg_values: bool,
) -> torch.Tensor:
    """Return a deduplicated saved activation payload when configured.

    Parameters
    ----------
    trace:
        Trace that owns the per-pass dedup caches.
    source_tensor:
        Live output tensor before ``safe_copy`` created ``raw_out``.
    raw_out:
        Copied activation payload.
    label:
        Raw layer label for the saved output.
    annotations:
        Mutable annotation dictionary for the saved output.
    save_arg_values:
        Whether argument values are being saved. Argument snapshots consume
        independent payloads, so activation dedup is disabled in that mode.

    Returns
    -------
    torch.Tensor
        Either ``raw_out`` or a previously saved payload for the same live
        source tensor.
    """

    if trace is None or save_arg_values or raw_out.is_meta:
        return raw_out

    mode = getattr(trace, "_out_dedup_mode", "identity")
    if mode == "none":
        return raw_out

    if mode == "content":
        hash_cache = getattr(trace, "_out_hash_cache", None)
        if hash_cache is None:
            hash_cache = {}
            setattr(trace, "_out_hash_cache", hash_cache)
        # R36: the content digest is a host-side byte read; a cpu_async
        # payload may still be an in-flight pinned buffer. No-op unless
        # async fence events are pending.
        from ..utils.tensor_utils import synchronize_pending_cpu_async_copies

        synchronize_pending_cpu_async_copies()
        out_hash = _tensor_content_hash(raw_out)
        if out_hash in hash_cache:
            annotations["dedup_out_hash"] = out_hash
            annotations["dedup_reference_label"] = hash_cache[out_hash][0]
            return hash_cache[out_hash][1]
        hash_cache[out_hash] = (label, raw_out)
        return raw_out

    identity_cache = getattr(trace, "_out_identity_cache", None)
    if identity_cache is None:
        identity_cache = {}
        setattr(trace, "_out_identity_cache", identity_cache)

    source_key = id(source_tensor)
    # r65: TorchLens's OWN dedup-bookkeeping ``_version`` read runs under the explicit
    # internal-read marker so the r65 state-metadata property observer never mistakes it
    # for a user ``._version`` read on a registered buffer/param receiver (unmarked it
    # fires for every saved state source and would spuriously refuse any model whose
    # consumed buffer was ever mutated in place before capture). Imported lazily:
    # ``data_classes`` sits below the torch backend in the layering.
    from ..backends.torch.completeness_witness import internal_scalar_read

    with internal_scalar_read():
        source_version = tensor_version_or_none(source_tensor)
    cached = identity_cache.get(source_key)
    if cached is not None:
        cached_source, cached_label, cached_out, cached_version, cached_ordinal = cached
        if cached_source is source_tensor and cached_version == source_version:
            # B3R4-R21-1: dense trace-local ordinal, never the raw ``id()``.
            annotations["dedup_source_id"] = cached_ordinal
            annotations["dedup_source_version"] = source_version
            annotations["dedup_reference_label"] = cached_label
            return cached_out

    # Dense ordinal in cache-insertion (execution) order: deterministic across
    # processes for the same captured program, unlike the ``id()`` slot key. A
    # replaced slot (id reuse after a mismatch) keeps its original ordinal so
    # ordinals stay unique within the cache.
    ordinal = cached[4] if cached is not None else len(identity_cache) + 1
    identity_cache[source_key] = (source_tensor, label, raw_out, source_version, ordinal)
    return raw_out


if TYPE_CHECKING:
    import pandas as pd

    from .._io.lazy import LazyActivationRef
    from ..receptive_field._view import ReceptiveFieldView
    from .func_call_location import FuncCallLocation
    from .layer import Layer, OpAccessor
    from .module import Module
    from .param import Param
    from .trace import Trace


class Op(_SelectionOperand):
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
        activation_memory: Bytes | None
        transformed_activation_memory: Bytes | None
        visualizer_path: str | None
        autograd_memory: Bytes | None
        num_autograd_tensors: int | None
        bytes_delta_at_call: Bytes | None
        bytes_peak_at_call: Bytes | None
        gradient_memory: Bytes | None
        transformed_gradient_memory: Bytes | None
        func_id: FunctionRegistryKey | None
        code_context: list["FuncCallLocation"]
        var_names: list[str]
        func_duration: Duration | None
        flops_forward: Flops | None
        flops_backward: Flops | None
        _param_barcodes: list[Any]
        _param_logs: list["Param"]
        param_memory: Bytes
        is_orphan: bool
        fx_qualpath: str | None
        fx_call_index: int
        out_ref: "LazyActivationRef" | None
        grad_ref: "LazyActivationRef" | None
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
        label_short: Any
        layer_label: Any
        layer_label_short: Any
        type: Any
        type_index: Any
        pass_index: Any
        num_passes: Any
        lookup_keys: Any
        out: Any
        has_saved_activation: Any
        output_device: Any
        activation_transform: Any
        interventions: Any
        intervention_replaced: Any
        detach_saved_activations: Any
        has_saved_args: Any
        saved_args: Any
        saved_kwargs: Any
        args_template: Any
        kwargs_template: Any
        shape: Any
        transformed_out_shape: Any
        dtype: Any
        transformed_out_dtype: Any
        transformed_out: Any
        has_out_variations: Any
        out_versions_by_child: Any
        grad: Any
        transformed_grad: Any
        save_grads: Any
        has_grad: Any
        grad_shape: Any
        transformed_grad_shape: Any
        grad_dtype: Any
        transformed_grad_dtype: Any
        func: Any
        func_call_id: Any
        func_name: Any
        func_qualname: Any
        func_rng_states: Any
        func_autocast_state: Any
        arg_names: Any
        num_args_total: Any
        num_pos_args: Any
        num_kwargs: Any
        non_tensor_pos_args: Any
        non_tensor_kwargs: Any
        func_non_tensor_args: Any
        is_inplace: Any
        grad_fn_class_name: Any
        grad_fn_class_qualname: Any
        grad_fn_object_id: Any
        grad_fn_handle: Any
        grad_fn: Any
        in_multi_output: Any
        multi_output_index: Any
        multi_output_name: Any
        container_path: Any
        container_spec: Any
        is_transform: Any
        transform_kind: Any
        transform_chain: Any
        transform_config: Any
        transform_fn_name: Any
        transform_fn_qualname: Any
        transform_fn_source: Any
        unattributed_tensor_args: Any
        dropped_edge_tensor_args: Any
        parent_params: Any
        parent_param_ops: Any
        param_shapes: Any
        num_params: Any
        num_params_trainable: Any
        num_params_frozen: Any
        equivalence_class: Any
        equivalent_ops: Any
        recurrent_ops: Any
        site_key: str | None
        parents: Any
        parent_arg_positions: Any
        _edge_uses: Any
        edge_substitutions: dict[Any, Any]
        edge_replacement_stamps: dict[Any, Any]
        root_ancestors: Any
        children: Any
        has_children: Any
        is_input: Any
        input_was_parameter: Any
        has_input_ancestor: Any
        input_ancestors: Any
        min_distance_from_input: Any
        max_distance_from_input: Any
        is_output: Any
        is_output_parent: Any
        is_final_output: Any
        has_output_descendant: Any
        output_descendants: Any
        io_role: Any
        min_distance_to_output: Any
        max_distance_to_output: Any
        is_buffer: Any
        address: Any
        buffer_pass: Any
        buffer_source: Any
        buffer_write_kind: Any
        buffer_value_changed: Any
        buffer_replay_validated: Any
        buffer_source_func_name: Any
        is_internal_source: Any
        has_internal_source_ancestor: Any
        internal_source_parents: Any
        internal_source_ancestors: Any
        is_internal_sink: Any
        is_terminal_bool: Any
        is_terminal_conditional_bool: Any
        conditional_context_kind: Any
        conditional_wrapper_kind: Any
        terminal_conditional_id: Any
        is_scalar_bool: Any
        bool_value: Any
        in_conditionals: Any
        terminal_bool_for: Any
        conditional_branch_stack: Any
        conditional_branch_depth: Any
        conditional_entry_children: Any
        conditional_then_children: Any
        conditional_elif_children: Any
        conditional_else_children: Any
        conditional_arm_children: Any
        module: Any
        _address_normalized: Any
        modules: Any
        module_call_stack: Any
        input_to_module_calls: Any
        module_entry_arg_keys: Any
        output_of_modules: Any
        output_of_module_calls: Any
        is_module_output: Any
        is_atomic_module: Any
        atomic_module_call: Any
        func_config: Any
        _source_trace_ref: Any
        _facets_cache: Any
        _receptive_field_cache: Any
        _projective_field_cache: Any
        _arg_expressions_cache: Any
        _is_in_conditional_body: Any

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
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
        "label_short": FieldPolicy.KEEP,
        "layer_label": FieldPolicy.KEEP,
        "layer_label_short": FieldPolicy.KEEP,
        "type": FieldPolicy.KEEP,
        "type_index": FieldPolicy.KEEP,
        "pass_index": FieldPolicy.KEEP,
        "num_passes": FieldPolicy.KEEP,
        "lookup_keys": FieldPolicy.KEEP,
        "out": FieldPolicy.BLOB,
        "transformed_out": FieldPolicy.BLOB,
        "has_saved_activation": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "activation_transform": FieldPolicy.DROP,
        "annotations": FieldPolicy.KEEP,
        "interventions": FieldPolicy.KEEP,
        "intervention_replaced": FieldPolicy.KEEP,
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
        ARG_EXPRESSIONS_FIELD: FieldPolicy.DROP,
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
        # universe carries no shadow storage outside FIELD_POLICY.
        "_facets_cache": FieldPolicy.DROP,
    }
    FIELD_POLICY = build_record_field_policy_table(
        LAYER_PASS_LOG_FIELD_ORDER,
        PORTABLE_STATE_SPEC,
        fork_policy=LAYER_PASS_LOG_FIELD_FORK_POLICY,
        default_fill_state=_LAYER_PASS_LOG_DEFAULT_FILL,
        schema_key="op",
    )
    PORTABLE_STATE_SPEC = portable_state_spec_from_policy(FIELD_POLICY)
    FIELD_FORK_POLICY = fork_policy_from_policy(FIELD_POLICY)
    DEFAULT_FILL_STATE = default_fill_state_from_policy(FIELD_POLICY)

    def _slot(self, name: str, default: Any = None) -> Any:
        """Return one physical slot value, or ``default`` when it is unset."""

        try:
            return _object_getattribute(self, name)
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
                    passes = [record.backward_pass_index for record in saved]
                    raise InvalidArgumentError(
                        f"op {label} has gradients saved from multiple backward passes {passes}; "
                        "use op.grads[...] / op.grad_for(bwd=k)",
                        code="gradient_pass_ambiguous",
                        remedy="use op.grads[...] or op.grad_for(bwd=k) to pick one pass",
                        label=str(label),
                    )
            grad = slot("grad")
            if grad is None and slot("grad_ref") is not None:
                return object.__getattribute__(self, "materialize_grad")()
            return grad
        if name == "out":
            slot = _object_getattribute(self, "_slot")
            out = slot("out")
            source_ref = slot("_source_trace_ref")
            source_trace = None if source_ref is None else source_ref()
            if (
                out is None
                and slot("out_ref") is not None
                and getattr(source_trace, "_predicate_save_options", None) is not None
            ):
                return object.__getattribute__(self, "materialize_out")()
            if (
                slot("_tracing_finished")
                and not slot("has_saved_activation", False)
                and getattr(source_trace, "_predicate_save_options", None) is not None
            ):
                label = slot("label") or slot("layer_label") or slot("_label_raw")
                raise PayloadUnavailableError(
                    f"op {label} was not saved; no saved payload is available",
                    code="activation_not_saved",
                    remedy="re-run with save=... to retain this activation",
                    label=str(label),
                )
            if not getattr(source_trace, "_postprocessing_active", False):
                state = {
                    "out": out,
                    "annotations": slot("annotations"),
                    "has_saved_activation": slot("has_saved_activation", False),
                    "detach_saved_activations": slot("detach_saved_activations"),
                    "label": slot("label"),
                    "layer_label": slot("layer_label"),
                    "_label_raw": slot("_label_raw"),
                }
                _validate_reference_out_not_mutated(state)
        return _getattribute(self, name)

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

        _object_setattr(self, attr, value)

    def _compact_metadata(self, pool: dict[Any, Any]) -> None:
        """Replace repeated immutable metadata with pooled shared instances.

        Called once per Op by :func:`~torchlens.data_classes._compaction.compact_op_metadata`
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
            if cls in pooled_classes:
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

        current_value = getattr(self, field_name)
        other_value = getattr(other, field_name)
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
    def layer_type(self) -> str:
        """Return the operation type token used by existing internal callers."""

        return cast(str, self.type)

    @layer_type.setter
    def layer_type(self, value: str) -> None:
        """Set the operation type token through the legacy internal name."""

        self.type = value

    @property
    def macs_forward(self) -> Macs | None:
        """Forward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return as_macs(self.flops_forward // 2 if self.flops_forward is not None else None)

    @property
    def macs_backward(self) -> Macs | None:
        """Backward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return as_macs(self.flops_backward // 2 if self.flops_backward is not None else None)

    @property
    def flops_total(self) -> Flops:
        """Approximate total FLOPs for this Op.

        Returns
        -------
        Flops
            Forward plus backward FLOPs, treating unknown halves as zero.
        """

        return Flops((self.flops_forward or 0) + (self.flops_backward or 0))

    @property
    def macs_total(self) -> Macs:
        """Approximate total MACs for this Op.

        Returns
        -------
        Macs
            Forward plus backward MACs.
        """

        return Macs(self.flops_total // 2)

    @property
    def bytes_read(self) -> Bytes | None:
        """Theoretical ideal read-once traffic for this operation.

        This access-time model uses only recorded shape/dtype metadata. It is
        not measured hardware traffic and can diverge under caching, kernel
        fusion, or backend implementation details. Logical view/alias operations
        report zero bytes moved; insufficient metadata returns ``None``.

        Returns
        -------
        Bytes | None
            Theoretical bytes read once from inputs and parameters.
        """

        from ..debug._cost import theoretical_op_bytes

        return theoretical_op_bytes(self)[0]

    @property
    def bytes_written(self) -> Bytes | None:
        """Theoretical ideal write-once traffic for this operation.

        This access-time model uses only recorded shape/dtype metadata. It is
        not measured hardware traffic and can diverge under caching, kernel
        fusion, or backend implementation details. Logical view/alias operations
        report zero bytes moved; insufficient metadata returns ``None``.

        Returns
        -------
        Bytes | None
            Theoretical bytes written once for the logical output.
        """

        from ..debug._cost import theoretical_op_bytes

        return theoretical_op_bytes(self)[1]

    @property
    def arithmetic_intensity(self) -> float | None:
        """Return theoretical forward FLOPs per ideal traffic byte.

        The value is ``flops_forward / (bytes_read + bytes_written)`` using the
        theoretical ideal read-once/write-once model. It is not measured roofline
        intensity and can diverge under kernel fusion, caching, or real memory
        transactions. Missing FLOPs/traffic and zero-traffic logical views return
        ``None``.

        Returns
        -------
        float | None
            Theoretical forward arithmetic intensity in FLOPs per byte.
        """

        bytes_read = self.bytes_read
        bytes_written = self.bytes_written
        if self.flops_forward is None or bytes_read is None or bytes_written is None:
            return None
        total_bytes = int(bytes_read) + int(bytes_written)
        if total_bytes == 0:
            return None
        return float(self.flops_forward) / total_bytes

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
    def is_compute_op(self) -> bool:
        """Whether this Op executed a torch function rather than a boundary sentinel."""

        return not (self.is_input or self.is_output or self.is_buffer)

    @property
    def fx_label(self) -> str | None:
        """Return a torch.fx-style label for this Op when module context is available."""

        if self.fx_qualpath is None:
            return None
        return f"{self.fx_qualpath.replace('.', '_')}_{self.fx_call_index}"

    @property
    def trace(self) -> "Trace":
        """Alias for the owning Trace back-reference."""

        return self.source_trace

    @property
    def grad_fn_cls(self) -> type[Any] | None:
        """Return the live grad_fn_handle class when the autograd object is retained.

        Returns
        -------
        type[Any] | None
            Runtime grad_fn_handle class, or ``None`` when the object is unavailable.
        """

        grad_fn_handle = self.grad_fn_handle
        return None if grad_fn_handle is None else type(grad_fn_handle)

    @property
    def grads(self) -> GradientRecordAccessor:
        """Per-pass gradient records saved for this Op."""

        trace = self.source_trace
        raise_if_no_backward_capture(trace, plural_subject="op.grads or saved_grad_ops")
        records = self._slot("_grad_records")
        if records is None:
            records = []
            self._internal_set("_grad_records", records)
        return GradientRecordAccessor(records)

    @property
    def derived_grad(self) -> Any | None:
        """Return this op's derived intermediate gradient payload, if available.

        Returns
        -------
        Any | None
            Backend-owned gradient payload from ``trace.intermediate_derived_grads``,
            or ``None`` when this op has no exact intermediate-derived record.
        """

        trace = self._source_trace
        records = getattr(trace, "intermediate_derived_grads", None)
        if records is None:
            return None
        if self.label not in records:
            return None
        record = records[self.label]
        if record.provenance.get("status") != "exact":
            return None
        return record.grad

    def grad_for(self, *, bwd: int) -> torch.Tensor:
        """Return the saved gradient tensor for one backward pass.

        Parameters
        ----------
        bwd:
            One-based backward pass number.

        Returns
        -------
        torch.Tensor
            Saved raw gradient tensor for the requested pass.
        """

        record = self.grads.for_pass(bwd)
        if record.grad is None:
            raise PayloadUnavailableError(
                f"op {self.label} has no saved gradient payload for backward pass {bwd}",
                code="gradient_not_saved",
                remedy="capture with gradient saving enabled for this op and pass",
                label=str(self.label),
                backward_pass=bwd,
            )
        return record.grad

    def _record_gradient(
        self,
        *,
        backward_pass_index: int,
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
        backward_pass_index:
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
            record for record in records if record.backward_pass_index != backward_pass_index
        ]
        record = GradientRecord(
            owner=self,
            ordinal=len(records) + 1,
            backward_pass_index=backward_pass_index,
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

        grad_fn = self.grad_fn
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

        return _summarize_call_args(self.saved_args, self.non_tensor_pos_args)

    @property
    def kwargs_summary(self) -> str | None:
        """Human-readable summary of this Op's keyword arguments.

        Op-level source for ``Layer.kwargs_summary``. Returns ``None`` when no
        keyword-argument information was captured.
        """

        return _summarize_call_kwargs(self.saved_kwargs, self.non_tensor_kwargs)

    @property
    def multi_output_type(self) -> type | str | None:
        """Python class of the multi-output container this Op came from.

        Resolves ``container_spec.type_module``/``type_qualname`` to a runtime class
        through the single default-deny container resolver in
        ``torchlens.ir.container`` -- the SAME resolver used by output-container
        reconstruction. That resolver NEVER imports an attacker-named module (the
        ``ContainerSpec`` fields are portable, bundle-controlled strings that ride
        through the safe unpickler untouched, so importing one would execute its
        top-level code = arbitrary-code-execution); it resolves only from a module
        already present in ``sys.modules`` and only to a type that structurally
        matches the recorded container kind. When the type cannot be resolved
        WITHOUT importing (its defining module is not loaded), fall back to the
        qualified name string exactly as before, rather than importing it. ``None``
        when this Op is not from a multi-output container.

        Raises
        ------
        ContainerReconstructionError
            If the reference resolves to an already-loaded type that is not
            admissible for the recorded container kind (a tampered spec). This is
            the security tripwire firing, not a legit-capture path.
        """

        from ..ir.container import resolve_container_type

        if not self.in_multi_output:
            return None
        spec = self.container_spec
        if spec is None:
            return None
        resolved = resolve_container_type(spec)
        if resolved is not None:
            return resolved
        # Unresolvable without importing an untrusted, bundle-controlled module
        # name. Preserve the historical graceful string fallback WITHOUT importing.
        type_module = getattr(spec, "type_module", None)
        type_qualname = getattr(spec, "type_qualname", None)
        if type_qualname is None:
            return None
        if type_module:
            return f"{type_module}.{type_qualname}"
        return type_qualname

    @property
    def container(self) -> Any:
        """Return the runtime container view for this op, when available.

        Returns
        -------
        Container | None
            Computed container view, degraded path-only view, or ``None``.
        """

        from .container import container_from_op

        return container_from_op(self)

    @property
    def output_containers(self) -> tuple[Any, ...]:
        """Return output container views associated with this op.

        Returns
        -------
        tuple[Container, ...]
            Registry-backed output container views, or a legacy single view.
        """

        from .container import output_containers_from_op

        return output_containers_from_op(self)

    @property
    def input_containers(self) -> tuple[Any, ...]:
        """Return input container views consumed by this op.

        Returns
        -------
        tuple[Container, ...]
            Registry-backed input container views keyed by this call site.
        """

        from .container import input_containers_from_op

        return input_containers_from_op(self)

    @property
    def containers(self) -> tuple[Any, ...]:
        """Return all input and output container views associated with this op.

        Returns
        -------
        tuple[Container, ...]
            Deduplicated union of input and output container views.
        """

        from .container import containers_from_op

        return containers_from_op(self)

    @property
    def is_module_input(self) -> bool:
        """Whether this Op's output feeds into at least one ModuleCall as an input.

        The Op itself is OUTSIDE the module (the upstream producer). Equivalent to
        ``bool(self.input_to_module_calls)``. Direction-of-data-flow framing:
        inputs come FROM outside the module.
        """

        return bool(self.input_to_module_calls)

    @property
    def atomic_module_call_label(self) -> str | None:
        """Stable ModuleCall label for an atomic-module output Op, else ``None``.

        Stored form of the cross-class reference; ``atomic_module_call`` resolves
        the ModuleCall record.
        """

        return cast("str | None", self.atomic_module_call)

    @property
    def atomic_module_address(self) -> str | None:
        """Module address (no ``:N``) for an atomic-module output Op, else ``None``.

        Derived from ``atomic_module_call_label`` by stripping the ``:N`` pass
        suffix; pairs with the ``atomic_module`` resolver.
        """

        label = self.atomic_module_call_label
        if label is None:
            return None
        return label.rsplit(":", 1)[0]

    @property
    def atomic_module(self) -> "Module | None":
        """Module record resolved from ``atomic_module_address``, or ``None``.

        Returns ``None`` when this Op is not an atomic-module output or when the
        owning Trace is unavailable.
        """

        address = self.atomic_module_address
        if address is None:
            return None
        trace = self._source_trace
        if trace is None:
            return None
        try:
            return cast("Module", trace.modules[address])
        except (KeyError, TypeError):
            return None

    @property
    def has_parents(self) -> bool:
        """Whether this layer has any parent layers."""
        return len(self.parents) > 0

    @property
    def num_parents(self) -> int:
        """Number of distinct parent Ops feeding this Op."""

        return len(self.parents)

    @property
    def num_children(self) -> int:
        """Number of distinct child Ops fed by this Op."""

        return len(self.children)

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
        """Return every spelling a relation list may use for THIS op.

        On a finished trace a parent's ``children`` (or a child's ``parents``)
        may record this op under either its bare ``layer_label`` or -- on a
        multi-pass layer -- its pass-qualified ``layer_label:pass`` spelling,
        so self-exclusion in ``siblings``/``co_parents`` must cover BOTH
        (excluding only the bare spelling appended the op itself on every
        multi-pass layer). Unfinished traces still use raw labels.
        """

        _finished_trace = self._source_trace_or_none()
        _finished = self._tracing_finished or (
            _finished_trace is not None and _finished_trace._tracing_finished
        )
        if not _finished:
            return {self._label_raw}
        spellings = {self.layer_label}
        pass_index = self.pass_index
        if isinstance(pass_index, int):
            spellings.add(f"{self.layer_label}:{pass_index}")
        return spellings

    @property
    def siblings(self) -> list[str]:
        """Layers sharing at least one parent (excluding output layers and this op)."""
        ml = self._source_trace
        if ml is None:
            return []
        siblings = []
        seen = self._own_label_spellings()
        for parent_label in self.parents:
            try:
                parent = ml[parent_label]
            except (KeyError, ValueError):
                try:
                    parent = ml.orphans[parent_label]
                except KeyError:
                    continue
            for child_label in parent.children:
                if child_label not in seen:
                    seen.add(child_label)
                    try:
                        child = ml[child_label]
                    except (KeyError, ValueError):
                        try:
                            child = ml.orphans[child_label]
                        except KeyError:
                            continue
                    if not child.is_output:
                        siblings.append(child_label)
        return siblings

    @property
    def has_siblings(self) -> bool:
        """Whether this layer shares parents with other layers."""
        return len(self.siblings) > 0

    @property
    def co_parents(self) -> list[str]:
        """Layers sharing at least one child (excluding output layers and this op)."""
        ml = self._source_trace
        if ml is None:
            return []
        spouses = []
        seen = self._own_label_spellings()
        for child_label in self.children:
            try:
                child = ml[child_label]
            except (KeyError, ValueError):
                try:
                    child = ml.orphans[child_label]
                except KeyError:
                    continue
            for parent_label in child.parents:
                if parent_label not in seen:
                    seen.add(parent_label)
                    try:
                        parent = ml[parent_label]
                    except (KeyError, ValueError):
                        try:
                            parent = ml.orphans[parent_label]
                        except KeyError:
                            continue
                    if not parent.is_output:
                        spouses.append(parent_label)
        return spouses

    @property
    def has_co_parents(self) -> bool:
        """Whether this layer shares children with other layers."""
        return len(self.co_parents) > 0

    @property
    def is_in_conditional(self) -> bool:
        """Whether this op participates in any conditional role."""

        return bool(self.in_conditionals)

    @property
    def is_in_conditional_evaluation(self) -> bool:
        """Whether this op computes a conditional arm condition."""

        return any(role.role == "evaluation" for role in self.in_conditionals or [])

    @property
    def is_in_conditional_body(self) -> bool:
        """Whether this op is in a conditional arm body."""

        if self.has_output_descendant and not self.conditional_entry_children:
            return False
        return bool(self._slot("_is_in_conditional_body", False)) or any(
            role.role == "body" for role in self.in_conditionals or []
        )

    @is_in_conditional_body.setter
    def is_in_conditional_body(self, value: bool) -> None:
        """Set the cached conditional-body predicate used during postprocessing."""

        object.__setattr__(self, "_is_in_conditional_body", value)

    @is_in_conditional_body.deleter
    def is_in_conditional_body(self) -> None:
        """Delete the cached conditional-body predicate during cleanup."""

        try:
            object.__delattr__(self, "_is_in_conditional_body")
        except AttributeError:
            pass

    @property
    def conditional_depth(self) -> int:
        """Number of distinct conditionals this op participates in."""

        return len({role.conditional_id for role in self.in_conditionals or []})

    @property
    def uses_params(self) -> bool:
        """Whether this operation used model parameters."""
        return len(self._param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this operation."""
        return len(self._param_barcodes)

    @property
    def input_ops(self) -> "OpAccessor":
        """Accessor over graph-parent Op records in ``parents`` order."""

        from .layer import OpAccessor

        trace = self._source_trace_or_error()
        parent_ops = {}
        for parent_index, parent_label in enumerate(self.parents, start=1):
            try:
                parent_ops[parent_index] = trace.ops[parent_label]
            except KeyError:
                if parent_label in trace.layer_dict_all_keys:
                    parent_ops[parent_index] = trace.layer_dict_all_keys[parent_label]
        return OpAccessor(parent_ops)

    @input_ops.deleter
    def input_ops(self) -> None:
        """Ignore cleanup deletion for derived input Op access."""

    @property
    def buffer_source_ops(self) -> "OpAccessor":
        """Accessor over buffer source Ops that feed this Op."""

        from .layer import OpAccessor

        trace = self._source_trace_or_error()
        source_ops = {}
        for parent_index, parent_label in enumerate(self.parents, start=1):
            try:
                parent = trace.ops[parent_label]
            except KeyError:
                if parent_label not in trace.layer_dict_all_keys:
                    continue
                parent = trace.layer_dict_all_keys[parent_label]
            if parent.is_buffer and parent.buffer_write_kind is None:
                source_ops[parent_index] = parent
        return OpAccessor(source_ops)

    @buffer_source_ops.deleter
    def buffer_source_ops(self) -> None:
        """Ignore cleanup deletion for derived buffer source Op access."""

    @property
    def buffer_sink_ops(self) -> "OpAccessor":
        """Accessor over buffer sink Ops directly consuming this Op."""

        from .layer import OpAccessor

        trace = self._source_trace_or_error()
        sink_ops = {}
        for child_index, child_label in enumerate(self.children, start=1):
            try:
                child = trace.ops[child_label]
            except KeyError:
                if child_label not in trace.layer_dict_all_keys:
                    continue
                child = trace.layer_dict_all_keys[child_label]
            if child.is_buffer and child.buffer_write_kind is not None:
                sink_ops[child_index] = child
        return OpAccessor(sink_ops)

    @buffer_sink_ops.deleter
    def buffer_sink_ops(self) -> None:
        """Ignore cleanup deletion for derived buffer sink Op access."""

    @property
    def input_activations(self) -> tuple[torch.Tensor | None, ...]:
        """Saved parent activations consumed by this Op, as references.

        Returned tensors are not copied. Mutating them mutates TorchLens saved
        state. For in-place-modified parents, the per-child version consumed by
        this Op is returned when available.
        """

        trace = self._source_trace_or_error()
        activations: list[torch.Tensor | None] = []
        child_labels = (
            self.layer_label,
            self.label,
            self._label_raw,
            self._layer_label_raw,
        )
        # ``Trace.ops`` is a cache-checking property, so resolving it per parent
        # made a wide fan-in operation (concatenation) pay that check once per
        # parent; the accessor cannot change while this loop runs.
        ops = trace.ops
        for parent_label in self.parents:
            parent: Op | None
            try:
                parent = ops[parent_label]
            except KeyError:
                parent = cast("Op | None", trace.layer_dict_all_keys.get(parent_label))
            if parent is None:
                activations.append(None)
                continue
            if not parent.has_saved_activation:
                activations.append(None)
                continue
            child_versions = getattr(parent, "out_versions_by_child", {}) or {}
            consumed = next(
                (child_versions[label] for label in child_labels if label in child_versions),
                parent.out,
            )
            activations.append(consumed if isinstance(consumed, torch.Tensor) else None)
        return tuple(activations)

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
    def input_memory(self) -> Bytes:
        """Sum of activation bytes across saved graph-parent Ops."""

        return Bytes(
            sum(
                int(getattr(parent, "activation_memory", 0) or 0)
                for parent in self.input_ops.values()
                if parent.has_saved_activation
            )
        )

    @input_memory.deleter
    def input_memory(self) -> None:
        """Ignore cleanup deletion for derived input memory."""

    @property
    def num_inputs(self) -> int:
        """Number of graph-parent input Ops."""

        return len(self.parents)

    @num_inputs.deleter
    def num_inputs(self) -> None:
        """Ignore cleanup deletion for derived input count."""

    @property
    def in_submodule(self) -> bool:
        """Whether this operation was computed inside a submodule."""
        return self.module is not None

    @property
    def module_call_depth(self) -> int:
        """Depth of ``module_call_stack``, the op's active ModuleCall nesting.

        ``module_call_stack`` holds the same containment fact as ``modules``
        (root-first ModuleCall labels active for this op; B3R7-R05-1 tied the
        two by invariant), so the length of either is this depth.
        """
        return len(self.modules)

    @property
    def input_to_modules(self) -> list[str]:
        """Module addresses (no ``:N``) whose input this Op's output fed.

        Derived from ``input_to_module_calls`` by stripping the ``:N`` pass suffix
        and de-duplicating, mirroring the ``output_of_modules`` /
        ``output_of_module_calls`` split. Use ``input_to_module_calls`` for the
        ModuleCall-label list.
        """

        seen: dict[str, None] = {}
        for call_label in self.input_to_module_calls:
            address = str(call_label).rsplit(":", 1)[0]
            seen.setdefault(address, None)
        return list(seen)

    @property
    def gradient_transform(self) -> Callable[..., Any] | None:
        """Transform used for this Op's saved gradient, or ``None`` when unset.

        Mirrors ``activation_transform`` for the backward side. Reads the
        trace-level ``grad_transform`` that was applied to this Op's gradient.
        """

        trace = self._source_trace
        if trace is None:
            return None
        return cast("Callable[..., Any] | None", getattr(trace, "grad_transform", None))

    @property
    def is_buffer_source(self) -> bool:
        """Whether this Op represents a buffer boundary (overwrites a buffer).

        Glossary name for the stored ``is_buffer`` flag.
        """

        return bool(self.is_buffer)

    @property
    def has_saved_gradient(self) -> bool:
        """Whether this Op's gradient was saved.

        Glossary name for the backward-side saved predicate; mirrors
        ``has_saved_activation``.
        """

        return bool(self.has_grad)

    @property
    def tensor(self) -> Any:
        """Alias for the raw saved out."""

        return self.out

    @property
    def ops(self) -> tuple["Op", ...]:
        """Tuple containing this pass for aggregate-compatible iteration.

        Returns
        -------
        tuple[Op, ...]
            Single-entry tuple containing this pass log.
        """

        return (self,)

    @property
    def _streaming_label(self) -> str:
        """Best available label for sink/writer callbacks during or after postprocess.

        Returns
        -------
        str
            Pass-qualified label when available, otherwise the current layer label.
        """

        for candidate in (
            self.label,
            self.layer_label,
            self._layer_label_raw,
            self._label_raw,
        ):
            if candidate is not None:
                return str(candidate)
        return "<unknown>"

    @property
    def source_trace(self) -> "Trace":
        """Back-reference to the owning Trace (stored as weakref).

        Never returns ``None``: an Op detached from its Trace (standalone
        pickle strips the weakref; cleanup clears it) refuses with the same
        typed ``RecordBindingError`` family as the collected-Trace case, so
        no ``None`` can escape behind the ``-> Trace`` signature and crash a
        caller untyped -- the r4 ``Layer`` fix (1a2b715e) applied to the
        sibling record class it never reached (r5 b7-opus R52-B).
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
                "Trace has been garbage-collected",
                code="trace_reference_collected",
                remedy="keep the owning Trace alive while reading its records",
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
        transform: Any,
        *,
        model: Any = None,
        x: Any = None,
        intervention: Any = None,
    ) -> "Trace":
        """Apply an intervention to this op through the owning Trace.

        Parameters
        ----------
        transform:
            Transform or hook to apply to this operation's output.
        model:
            Model required when ``intervention.engine="rerun"``.
        x:
            Input required when ``intervention.engine="rerun"``.
        intervention:
            Grouped intervention options (``InterventionOptions``).

        Returns
        -------
        Trace
            Source Trace after applying the intervention.
        """

        return self._source_trace_or_error().do(
            self.layer_label,
            transform,
            model=model,
            x=x,
            intervention=intervention,
        )

    def set(
        self,
        value: Any,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
        """Set this op's out recipe through the owning Trace.

        Parameters
        ----------
        value:
            Static replacement value or one-shot callable.
        strict:
            Whether site resolution should reject non-portable selectors.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.

        Returns
        -------
        Trace
            Source Trace with a stale intervention recipe.
        """

        return self._source_trace_or_error().set(
            self.layer_label,
            value,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )

    def attach_hooks(
        self,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
    ) -> Any:
        """Attach sticky hooks to this op through the owning Trace.

        Parameters
        ----------
        hook:
            Hook or helper to attach to this operation.
        *extra_hooks:
            Additional hooks to compose on this operation in left-to-right order.
        strict:
            Whether site resolution should reject non-portable selectors.
        prepend:
            Whether new sticky hooks should run before existing sticky hooks.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.

        Returns
        -------
        Any
            Trace or scoped removable hook handle, matching ``Trace.attach_hooks``.
        """

        return self._source_trace_or_error().attach_hooks(
            self.layer_label,
            hook,
            *extra_hooks,
            strict=strict,
            prepend=prepend,
            confirm_mutation=confirm_mutation,
        )

    def materialize_out(
        self,
        *,
        map_location: str | torch.device = "cpu",
        payload_hints: Any | None = None,
    ) -> Any:
        """Materialize this layer's saved out from a lazy bundle ref.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.
        payload_hints:
            Optional backend payload hints used during materialization.

        Returns
        -------
        Any
            Materialized out payload.

        Raises
        ------
        TorchLensIOError
            If no out ref is available for this layer.

        Examples
        --------
        >>> import torchlens as tl
        >>> trace = tl.load("demo_bundle", lazy=True)
        >>> tensor = trace["linear_1_1"].materialize_out()
        >>> tensor.shape
        torch.Size([2, 3])
        """

        current_out = self._slot("out")
        if current_out is not None:
            return current_out
        if self.out_ref is None:
            raise TorchLensIOError("no out_ref to materialize from")
        self._internal_set(
            "out",
            self.out_ref.materialize(map_location=map_location, payload_hints=payload_hints),
        )
        return self.out

    def materialize_grad(
        self,
        *,
        map_location: str | torch.device = "cpu",
        payload_hints: Any | None = None,
    ) -> Any:
        """Materialize this layer's saved grad from a lazy bundle ref.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.
        payload_hints:
            Optional backend payload hints used during materialization.

        Returns
        -------
        Any
            Materialized grad payload.

        Raises
        ------
        TorchLensIOError
            If no grad ref is available for this layer.

        Examples
        --------
        >>> import torchlens as tl
        >>> trace = tl.load("demo_bundle", lazy=True)
        >>> grad = trace["linear_1_1"].materialize_grad()
        >>> grad.shape
        torch.Size([2, 3])
        """

        grad = self._slot("grad")
        if grad is not None:
            return grad
        if self.grad_ref is None:
            raise TorchLensIOError("no grad_ref to materialize from")
        self._internal_set(
            "grad",
            self.grad_ref.materialize(map_location=map_location, payload_hints=payload_hints),
        )
        return self.grad

    def __tl_state_items__(self) -> Any:
        """Yield live state ``(field_name, value)`` pairs in declared order.

        The M2 state-protocol hook. Enumeration follows the declared
        ``_OP_SLOT_NAMES`` order and reads through
        ``object.__getattribute__`` -- exactly what the former per-slot walk
        did -- so class-level compatibility overlays (the ancestor-bitset
        properties installed by ``backends/torch/ops.py``) keep
        materializing their public values into pickle/fork state instead of
        leaking compact internal encodings. Unset cells are skipped; a bare
        ``object.__new__`` shell with no bound store yields nothing.
        """

        getattribute = _object_getattribute
        for name in _OP_SLOT_NAMES:
            try:
                yield name, getattribute(self, name)
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
        for field_name, field_value in mapping.items():
            _object_setattr(self, field_name, field_value)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with weakrefs stripped."""
        state = dict(state_items(self))
        state["_source_trace_ref"] = None
        state.pop("_facets_cache", None)
        state.pop("_receptive_field_cache", None)
        state.pop("_projective_field_cache", None)
        state["func"] = None
        state["grad_fn_handle"] = None
        # R10-7: user transform callables (FieldPolicy.DROP) serialize to the
        # loaded-artifact form (None) -- a lambda transform= made pickle.dumps
        # crash on every op record while tl.save succeeded on the same trace.
        if state.get("activation_transform") is not None:
            state["activation_transform"] = None
        if state.get("grad_transform") is not None:
            state["grad_transform"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state produced by ``__getstate__``."""
        _ensure_detached_store(self)
        read_tlspec_version(state, cls_name=type(self).__name__)
        resolver_status_was_present = "resolver_status" in state
        default_fill_state(
            state,
            defaults=self.DEFAULT_FILL_STATE,
        )
        # Repair present-but-wrong-typed container fields from legacy states
        # (e.g. `_param_barcodes` serialized as a `set` where a `list` is now
        # declared). `default_fill_state` only fills absent keys; this closes
        # the same gap `Trace.__setstate__` already closes for its own fields.
        coerce_container_typed_state(state, self.DEFAULT_FILL_STATE)
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
            from ..semantic import FacetView

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

        from ..selection import _selection_from_op

        return _selection_from_op(self)

    @property
    def receptive_field(self) -> "ReceptiveFieldView":
        """Return the lazy receptive-field query view for this Op."""

        from ..receptive_field import _engine
        from ..receptive_field._view import ReceptiveFieldView

        trace = self.source_trace
        solution = _engine.solve(trace)
        cache = self._slot("_receptive_field_cache")
        if cache is not None and cache._solution is not solution:
            for op in trace.layer_list:
                if op._slot("_receptive_field_cache") is not None:
                    object.__setattr__(
                        op,
                        "_receptive_field_cache",
                        ReceptiveFieldView(op, solution),
                    )
            cache = self._slot("_receptive_field_cache")
        if cache is None:
            cache = ReceptiveFieldView(self, solution)
            object.__setattr__(self, "_receptive_field_cache", cache)
        return cache

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

        from ..receptive_field._view import ReceptiveFieldView

        cache = self._slot("_projective_field_cache")
        current = ReceptiveFieldView.projective(self)
        if cache is None or cache._solution is not current._solution:
            cache = current
            object.__setattr__(self, "_projective_field_cache", cache)
        return cache

    @projective_field.deleter
    def projective_field(self) -> None:
        """Drop the cached source-anchored projective-field query view."""

        try:
            object.__delattr__(self, "_projective_field_cache")
        except AttributeError:
            pass

    @property
    def raw_label(self) -> str:
        """Return the public raw ordinal label without the raw-label suffix.

        Returns
        -------
        str
            Realtime ordinal identity for this op before equivalent-op grouping.
        """

        label = self._label_raw
        if isinstance(label, str) and label.endswith(RAW_LABEL_SUFFIX):
            return label[: -len(RAW_LABEL_SUFFIX)]
        return str(label)

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
    def arg_expressions(self) -> list[str]:
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
        from ..postprocess.ast_branches import resolve_arg_expressions

        resolved = resolve_arg_expressions(self.code_context, self.func_name)
        object.__setattr__(self, "_arg_expressions_cache", resolved)
        return resolved

    @arg_expressions.setter
    def arg_expressions(self, value: Any) -> None:
        """Set or clear the lazy argument-expression cache."""

        if value is None:
            try:
                object.__delattr__(self, "_arg_expressions_cache")
            except AttributeError:
                pass
            return
        object.__setattr__(self, "_arg_expressions_cache", list(value))

    @arg_expressions.deleter
    def arg_expressions(self) -> None:
        """Drop the cached argument-expression parse result."""

        try:
            object.__delattr__(self, "_arg_expressions_cache")
        except AttributeError:
            pass

    @property
    def is_internally_initialized(self) -> bool:
        """Return whether this op is an internally initialized source op.

        Returns
        -------
        bool
            Alias of ``is_internal_source`` for tabular compatibility.
        """

        return bool(self.is_internal_source)

    @is_internally_initialized.setter
    def is_internally_initialized(self, value: Any) -> None:
        """Accept compatibility writes for the computed internal-source alias."""

        if value is not None and bool(value) != bool(self.is_internal_source):
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

    def copy(self, *, _store: Any = None) -> "Op":
        """Return a selective-depth copy of this entry.

        ``_store`` is the private internal-synthesis hook: postprocess output
        node synthesis passes the owning trace's row store so the clone is
        appended as a shared columnar row. Public callers omit it and receive
        a detached single-row copy.

        Most fields are ``copy.deepcopy``'d so the clone is fully independent.
        However, certain fields are shallow-copied (shared by reference) because:

        * ``func``, ``grad_fn_class_name``, ``grad_fn_handle`` - function or
          autograd handle objects, immutable/shared.
        * ``source_trace`` - must point to the same Trace instance.
        * ``func_rng_states`` - large state dicts, not mutated after capture.
        * ``saved_args``, ``saved_kwargs``, ``args_template``,
          ``kwargs_template`` - may contain large tensors or structured
          templates; deep-copying them is expensive and unnecessary.
        * ``parent_params`` - references to nn.Parameters, must stay shared.
        * ``out``, ``transformed_out``, ``transformed_grad``,
          ``out_versions_by_child`` - large tensors;
          shared references are safe since they're replaced (not mutated).
        * ``container_spec`` - frozen dataclass shared by sibling output leaves.

        Returns:
            A new Op (or subclass) with the same field values.
        """
        return self._copy_with_shallow_fields(frozenset(), _store=_store)

    def _copy_for_output(self, *, _store: Any = None) -> "Op":
        """Clone this op for immediate conversion into a synthetic output node.

        Parameters
        ----------
        _store:
            Optional owning row store for the synthesized node.

        Returns
        -------
        Op
            Clone whose soon-to-be-replaced fields avoid unnecessary deep copies.
        """

        return self._copy_with_shallow_fields(_OUTPUT_NODE_REPLACED_FIELDS, _store=_store)

    def _copy_with_shallow_fields(
        self,
        extra_shallow_fields: frozenset[str],
        *,
        _store: Any = None,
    ) -> "Op":
        """Clone this op while sharing fields the caller replaces immediately.

        Parameters
        ----------
        extra_shallow_fields:
            Additional fields safe to share for this construction path.
        _store:
            Optional owning row store for the clone.

        Returns
        -------
        Op
            Selective-depth clone.
        """

        fields_dict = {}
        fields_not_to_deepcopy = {
            "func",
            "grad_fn_class_name",
            "grad_fn_handle",
            "source_trace",
            "func_rng_states",
            "saved_args",
            "saved_kwargs",
            "args_template",
            "kwargs_template",
            "parent_params",
            "out",
            "transformed_out",
            "transformed_grad",
            "out_versions_by_child",
            "container_spec",
        } | extra_shallow_fields
        from .._trace_core.op_store import row_clone_scope

        # The whole-schema getattr loop is a ROW-CLONE read, not a set of
        # per-column dependencies: under a combined step audit these reads
        # are tagged category (d) via the clone scope (design-ppdag-v3
        # §2.4d) — legal on row-creating steps, a finding elsewhere. The
        # scope is one dict lookup when no audit is armed; clone WRITES
        # bypass interception entirely (adopt_row).
        with row_clone_scope(object.__getattribute__(self, "_core")):
            for field in LAYER_PASS_LOG_FIELD_ORDER:
                if field not in fields_not_to_deepcopy:
                    fields_dict[field] = _copy_op_field_value(getattr(self, field, None))
                else:
                    fields_dict[field] = getattr(self, field, None)
        copied_entry = type(self)(fields_dict, _store=_store)
        return copied_entry

    def save_activation(
        self,
        t: torch.Tensor,
        t_args: list[Any] | tuple[Any, ...],
        t_kwargs: dict[str, Any],
        save_arg_values: bool,
        activation_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Save the output tensor (and optionally args) for this operation.

        Flow:
        1. Clone the tensor via ``safe_copy`` (strips tl_ attributes to avoid
           logging the copy operation).
        2. Move to ``output_device`` if different from the tensor's current device.
        3. Apply ``activation_transform`` inside ``pause_logging()`` to prevent
           the transform's own tensor ops from being logged.
        4. Optionally deep-copy function args/kwargs via ``_recursive_safe_copy``.

        Args:
            t: The output tensor of the operation.
            t_args: Positional arguments passed to the operation.
            t_kwargs: Keyword arguments passed to the operation.
            save_arg_values: Whether to deep-copy and store args/kwargs.
            activation_transform: Optional transform applied to the tensor
                before storing (e.g. detach, to-numpy, normalize).
        """
        trace = self._source_trace
        writer = getattr(trace, "_out_writer", None) if trace is not None else None
        try:
            save_mode = _effective_activation_save_mode(
                trace,
                func_name=self.func_name,
                is_inplace=bool(self.is_inplace),
            )
            budget = getattr(trace, "_save_budget_accountant", None)
            target_device = t.device
            if save_mode == "cpu_async":
                target_device = torch.device("cpu")
            elif self.output_device not in ("same", str(t.device)):
                target_device = torch.device(self.output_device)
            budget_reservation = (
                None
                if budget is None
                else budget.admit(
                    self._layer_label_raw,
                    target_device,
                    get_memory_amount_from_metadata(t, tuple(t.shape), t.dtype),
                )
            )
            save_raw_activations = getattr(trace, "save_raw_activations", True)
            store_raw = save_raw_activations or activation_transform is None
            # Pre-copy identity probe (dedup-after-copy ordering): a hit
            # reuses the already-saved payload and skips the clone entirely.
            # Restricted to plain "copy" mode -- reference/view copies are
            # free and cpu_async has fence side effects.
            dedup_cached_out = (
                _dedup_cached_identity_out(trace, t, self.annotations, save_arg_values)
                if store_raw and save_mode == "copy"
                else None
            )
            if dedup_cached_out is not None:
                raw_out = dedup_cached_out
            else:
                # Clone the tensor, optionally detaching from autograd graph.
                raw_out = copy_tensor_payload(
                    t,
                    detach_tensor=self.detach_saved_activations,
                    save_mode=save_mode,
                )
                # Move to the user-requested output device if needed.
                if self.output_device not in [str(raw_out.device), "same"]:
                    raw_out = safe_to(raw_out, self.output_device)
                _stamp_reference_out(self.annotations, raw_out, save_mode)

            self.shape = tuple(raw_out.shape)
            self.dtype = raw_out.dtype
            self.activation_memory = Bytes(
                get_memory_amount_from_metadata(raw_out, self.shape, self.dtype)
            )

            if store_raw:
                if dedup_cached_out is None:
                    raw_out = _dedup_saved_activation_out(
                        trace,
                        t,
                        raw_out,
                        self._layer_label_raw,
                        self.annotations,
                        save_arg_values,
                    )
                if isinstance(raw_out, torch.Tensor):
                    mark_detached_saved_activation(t, raw_out, self._layer_label_raw)
            self._internal_set("out", raw_out if store_raw else None)

            self._internal_set("transformed_out", None)
            self.transformed_out_shape = None
            self.transformed_out_dtype = None
            self.transformed_activation_memory = None
            if activation_transform is not None:
                self._internal_set(
                    "transformed_out",
                    self._apply_transform(
                        raw_out,
                        activation_transform,
                        transform_kind="activation",
                        streaming_active=writer is not None,
                    ),
                )
                self._validate_train_mode_transform_output(
                    raw_out,
                    self.transformed_out,
                    transform_kind="activation",
                )
                self._validate_streaming_transform_output(
                    self.transformed_out,
                    transform_kind="activation",
                    streaming_active=writer is not None,
                )
                self.transformed_out_shape = _shape_or_none(self.transformed_out)
                self.transformed_out_dtype = _dtype_or_none(self.transformed_out)
                self.transformed_activation_memory = _memory_or_none(self.transformed_out)
            if budget is not None:
                budget.commit(budget_reservation, (self.out, self.transformed_out))
        except Exception as exc:
            if writer is not None:
                writer.abort(f"Failed while saving out for {self._streaming_label}: {exc}")
                if isinstance(exc, (SaveBudgetExceededError, TorchLensPostfuncError)):
                    raise
                raise TorchLensIOError(
                    f"Streaming out save failed for {self._streaming_label}."
                ) from exc
            raise

        self.has_saved_activation = True

        if trace is not None:
            out_sink = getattr(trace, "_out_sink", None)
            if out_sink is not None and isinstance(self.out, torch.Tensor):
                out_sink(self._streaming_label, self.out)

            if writer is not None and trace._wrapper_runtime_ws.in_exhaustive_pass:
                self._stream_tensor_blob(
                    writer,
                    tensor_field="out",
                    pending_field="_pending_blob_id",
                    kind="out",
                )
                self._stream_tensor_blob(
                    writer,
                    tensor_field="transformed_out",
                    pending_field="_pending_transformed_out_blob_id",
                    kind="transformed_out",
                )

        # Tensor args and kwargs:
        if save_arg_values:
            self.has_saved_args = True
            self._internal_set("saved_args", [copy_arg_tree(arg) for arg in t_args])
            self._internal_set(
                "saved_kwargs",
                {k: copy_arg_tree(v) for k, v in t_kwargs.items()},
            )
        else:
            self._internal_set("saved_args", None)
            self._internal_set("saved_kwargs", None)

    def log_tensor_grad(
        self,
        grad: torch.Tensor,
        prebuilt: "tuple[torch.Tensor | None, Any | None] | None" = None,
    ) -> None:
        """Save the grad tensor for this layer's output.

        Called by the backward hook registered during the forward pass.
        The grad is ``detach().clone()``'d - a bare copy, not deep-copied -
        so it's independent of the autograd graph but cheap to store.

        Args:
            grad: The grad tensor flowing back through this operation.
            prebuilt: Budget-charged ``(raw_payload, transformed_payload)``
                pair already built (and transform-validated) by the
                event-sidecar path for this exact grad. When given, the slot
                REUSES those objects: no second clone, no second
                ``grad_transform`` execution, no uncharged retention
                (grind-r6 b5 R34-N1/R35-N1).
        """
        trace = self._source_trace
        raw_grad = grad
        self.grad_shape = tuple(raw_grad.shape)
        self.grad_dtype = raw_grad.dtype
        self.gradient_memory = Bytes(get_memory_amount(raw_grad))
        grad_transform = getattr(trace, "grad_transform", None)
        self._internal_set("transformed_grad", None)
        self.transformed_grad_shape = None
        self.transformed_grad_dtype = None
        self.transformed_gradient_memory = None
        writer = getattr(trace, "_out_writer", None) if trace is not None else None
        if prebuilt is not None:
            raw_payload, transformed_payload = prebuilt
            if grad_transform is not None:
                self._internal_set("transformed_grad", transformed_payload)
                self.transformed_grad_shape = _shape_or_none(self.transformed_grad)
                self.transformed_grad_dtype = _dtype_or_none(self.transformed_grad)
                self.transformed_gradient_memory = _memory_or_none(self.transformed_grad)
            self._internal_set("grad", raw_payload)
            self.has_grad = True
            if writer is not None and getattr(trace, "_defer_streaming_bundle_finalization", False):
                self._stream_tensor_blob(
                    writer,
                    tensor_field="grad",
                    pending_field="_pending_grad_blob_id",
                    kind="grad",
                )
                self._stream_tensor_blob(
                    writer,
                    tensor_field="transformed_grad",
                    pending_field="_pending_transformed_grad_blob_id",
                    kind="transformed_grad",
                )
            return
        # Admit BEFORE the transform/clone allocate (r8 R34, fable F1): a
        # policy-DENIED label reaching this legacy slot (callable/selector
        # save_grads with _grad_op_nums_to_save == "all") used to retain a
        # full uncharged grad clone -- invisible to a tight save_budget.
        # Source-sized reservation, transform delta reconciled at commit,
        # exactly the primary-site contract.
        budget = getattr(trace, "_save_budget_accountant", None) if trace is not None else None
        grad_reservation = None
        if budget is not None:
            grad_reservation = budget.admit(
                str(getattr(self, "_layer_label_raw", "<grad>")),
                raw_grad.device,
                int(raw_grad.numel() * raw_grad.element_size()),
                site="primary",
            )
        if grad_transform is not None:
            self._internal_set(
                "transformed_grad",
                self._apply_transform(
                    raw_grad,
                    grad_transform,
                    transform_kind="grad",
                    streaming_active=writer is not None,
                ),
            )
            self._validate_train_mode_transform_output(
                raw_grad,
                self.transformed_grad,
                transform_kind="grad",
            )
            self._validate_streaming_transform_output(
                self.transformed_grad,
                transform_kind="grad",
                streaming_active=writer is not None,
            )
            self.transformed_grad_shape = _shape_or_none(self.transformed_grad)
            self.transformed_grad_dtype = _dtype_or_none(self.transformed_grad)
            self.transformed_gradient_memory = _memory_or_none(self.transformed_grad)

        save_raw_gradients = getattr(trace, "save_raw_gradients", True)
        store_raw = save_raw_gradients or grad_transform is None
        # Gradient payloads are ALWAYS genuine snapshots: under
        # save_mode="reference"/"view" an aliased payload would be silently
        # rewritten by later user mutation of the seed gradient or by
        # AccumulateGrad accumulating in place on the next backward, and no
        # wrapped in-place op exists on the grad path to disclose it. Route
        # through the same chokepoint the event-sidecar path uses.
        from ..backends.torch.tensor_tracking import _copy_grad_payload

        save_mode = cast(SaveMode, getattr(trace, "save_mode", "copy"))
        self._internal_set(
            "grad",
            _copy_grad_payload(raw_grad, save_mode=save_mode) if store_raw else None,
        )
        self.has_grad = True
        if budget is not None and grad_reservation is not None:
            budget.commit(grad_reservation, (self.grad, self.transformed_grad))
        if writer is not None and getattr(trace, "_defer_streaming_bundle_finalization", False):
            self._stream_tensor_blob(
                writer,
                tensor_field="grad",
                pending_field="_pending_grad_blob_id",
                kind="grad",
            )
            self._stream_tensor_blob(
                writer,
                tensor_field="transformed_grad",
                pending_field="_pending_transformed_grad_blob_id",
                kind="transformed_grad",
            )

    def _apply_transform(
        self,
        tensor: torch.Tensor,
        transform: Callable[..., Any],
        *,
        transform_kind: str,
        streaming_active: bool,
    ) -> Any:
        """Apply a user transform with logging paused and rich error context."""

        return apply_transform(
            label=self._streaming_label,
            raw_label=self._layer_label_raw,
            func_name=self.func_name,
            tensor=tensor,
            transform=transform,
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

        return transform_error_message(
            label=self._streaming_label,
            raw_label=self._layer_label_raw,
            func_name=self.func_name,
            tensor=tensor,
            transform_kind=transform_kind,
            streaming_active=streaming_active,
        )

    def _validate_train_mode_transform_output(
        self,
        raw_tensor: torch.Tensor,
        output: Any,
        *,
        transform_kind: str,
    ) -> None:
        """Validate differentiability requirements for train-mode transform outputs."""

        trace = self._source_trace
        validate_train_mode_transform_output(
            raw_tensor=raw_tensor,
            transformed_tensor=output,
            transform_kind=transform_kind,
            backward_ready=getattr(trace, "backward_ready", False),
            label=self._streaming_label,
        )

    def _validate_streaming_transform_output(
        self,
        output: Any,
        *,
        transform_kind: str,
        streaming_active: bool,
    ) -> None:
        """Validate transformed tensors before streaming bundle finalization.

        Parameters
        ----------
        output:
            Value returned by the user transform.
        transform_kind:
            Transform kind, either ``"out"`` or ``"grad"``.
        streaming_active:
            Whether a streaming bundle writer is active for this trace.

        Returns
        -------
        None
            Raises if streaming cannot serialize the transformed value.

        Raises
        ------
        TorchLensIOError
            If streaming is active and the transform returns a non-tensor or sparse tensor.
        """

        try:
            validate_streaming_transform_output(
                transformed_tensor=output,
                transform_kind=transform_kind,
                streaming_active=streaming_active,
                label=self._streaming_label,
            )
        except TorchLensIOError as exc:
            self._abort_streaming_writer(str(exc))
            raise

    def _abort_streaming_writer(self, message: str) -> None:
        """Abort the active streaming writer when one is attached.

        Parameters
        ----------
        message:
            Reason written to the partial bundle marker.

        Returns
        -------
        None
            Mutates the writer state if present.
        """

        trace = self._source_trace
        writer = getattr(trace, "_out_writer", None) if trace is not None else None
        if writer is not None:
            writer.abort(message)

    def _stream_tensor_blob(
        self,
        writer: Any,
        *,
        tensor_field: str,
        pending_field: str,
        kind: str,
    ) -> None:
        """Stream one tensor field when present."""

        tensor = getattr(self, tensor_field)
        if tensor is None:
            return
        if not isinstance(tensor, torch.Tensor):
            if kind == "transformed_out":
                message = (
                    "Streaming save requires activation_transform outputs to be torch.Tensor "
                    f"instances, but layer {self._streaming_label} produced "
                    f"{type(tensor).__name__}."
                )
            elif kind == "transformed_grad":
                message = (
                    "Streaming save requires grad_transform outputs to be torch.Tensor "
                    f"instances, but layer {self._streaming_label} produced "
                    f"{type(tensor).__name__}."
                )
            else:
                message = (
                    f"{tensor_field} expected a tensor for streaming, got {type(tensor).__name__}."
                )
            writer.abort(message)
            raise TorchLensIOError(message)
        blob_id = writer.next_blob_id()
        setattr(self, pending_field, blob_id)
        writer.submit_blob(
            blob_id,
            tensor,
            kind=kind,
            label=self._streaming_label,
        )

    # ********************************************
    # ************* Fetcher Functions ************
    # ********************************************

    def _resolve_relation_record(self, label: str) -> "Op | None":
        """Resolve one relation label through the mainline, then orphans.

        The 62aba742 orphan tolerance stopped at the label aggregates: on a
        ``keep_orphans=True`` trace the aggregates include an orphan's label
        while the OBJECT-resolving surfaces crashed on the bare mainline
        lookup (b3 R05-N2). Mirror the per-op ``siblings`` behavior: fold
        the ``orphans`` fallback, skip what neither surface resolves.
        """

        trace = self.source_trace
        try:
            return cast("Op", trace[label])
        except (KeyError, ValueError):
            try:
                return cast("Op", trace.orphans[label])
            except KeyError:
                return None

    def get_children(self) -> list["Op"]:
        """Return child Op objects for this pass.

        Returns
        -------
        list[Op]
            Child ops resolved through the owning model log; orphan-relation
            labels resolve through ``trace.orphans`` (unresolvable skipped).
        """
        resolved = (self._resolve_relation_record(label) for label in self.children)
        return [record for record in resolved if record is not None]

    def get_parents(self) -> list["Op"]:
        """Return parent Op objects for this pass.

        Returns
        -------
        list[Op]
            Parent ops resolved through the owning model log; orphan-relation
            labels resolve through ``trace.orphans`` (unresolvable skipped).
        """
        resolved = (self._resolve_relation_record(label) for label in self.parents)
        return [record for record in resolved if record is not None]

    def show(
        self,
        method: Literal["auto", "heatmap", "channels", "rgb", "hist"] = "auto",
        **kwargs: Any,
    ) -> Any:
        """Display this pass's saved out.

        Parameters
        ----------
        method:
            Display method. ``"auto"`` chooses from tensor shape.
        **kwargs:
            Forwarded to the tensor display helper.

        Returns
        -------
        Any
            Matplotlib figure when plotting is available, otherwise a text
            fallback explaining why no plot was produced.
        """

        from ..viz._tensor_display import show_tensor

        return show_tensor(self, method=method, **kwargs)

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

        row = {field_name: getattr(self, field_name) for field_name in LAYER_PASS_LOG_FIELD_ORDER}
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
        """Return a human-readable operation summary.

        Data-model contract: never raises. An Op detached from its Trace
        (collected, standalone-pickled, or husked by cleanup) degrades to a
        one-line placeholder instead of silently printing an unknown
        denominator (``operation 1/?``) or propagating the typed relation
        refusal out of ``repr()``/``print()`` (r5 b7-opus R52-B, matching
        the Layer degradation).
        """

        trace = self._source_trace_or_none()
        trace_finished = trace is not None and trace._tracing_finished
        if self._tracing_finished or trace_finished:
            try:
                return self._str_after_pass()
            except RecordBindingError:
                label = self.layer_label or self._label_raw or "<unbound>"
                return f"<Op {label}: detached from its Trace>"
        return self._str_during_pass()

    def _str_during_pass(self) -> str:
        """Return a human-readable summary of this tensor entry while the forward pass is still in progress."""
        s = f"Tensor {self._label_raw} (layer {self._layer_label_raw}) (PASS NOT FINISHED):"
        s += f"\n\tPass: {self.pass_index}"
        s += f"\n\tTensor info: shape {self.shape}, dtype {self.dtype}"
        s += f"\n\tComputed from params: {self.uses_params}"
        s += f"\n\tComputed in modules: {self.modules}"
        s += f"\n\tOutput of modules: {self.output_of_module_calls}"
        if self.is_atomic_module:
            s += " (bottom-level submodule output)"
        else:
            s += " (not bottom-level submodule output)"
        s += "\n\tFamily info:"
        s += f"\n\t\tParents: {self.parents}"
        s += f"\n\t\tChildren: {self.children}"
        s += f"\n\t\tSpouses: {self.co_parents}"
        s += f"\n\t\tSiblings: {self.siblings}"
        s += (
            f"\n\t\tOriginal Ancestors: {self.root_ancestors} "
            f"(min dist {self.min_distance_from_input} nodes, max dist {self.max_distance_from_input} nodes)"
        )
        s += f"\n\t\tInput Ancestors: {self.input_ancestors}"
        s += f"\n\t\tInternal Ancestors: {self.internal_source_ancestors}"
        s += (
            f"\n\t\tOutput Descendents: {self.output_descendants} "
            f"(min dist {self.min_distance_to_output} nodes, max dist {self.max_distance_to_output} nodes)"
        )
        if self.out is not None:
            s += f"\n\tTensor contents: \n{print_override(self.out, '__str__')}"
        return s

    def _str_after_pass(self) -> str:
        """Return a human-readable summary of this tensor entry after the forward pass has completed."""
        if self.num_passes > 1:
            pass_str = f" (pass {self.pass_index}/{self.num_passes}), "
        else:
            pass_str = ", "
        # Raises RecordBindingError when detached; __str__ degrades it to the
        # explicit placeholder instead of printing an unknown denominator.
        num_ops = self.source_trace.num_ops
        s = f"Layer {self.layer_label}{pass_str}operation {self.step_index}/{num_ops}:"
        s += f"\n\tOutput tensor: shape={self.shape}, dtype={self.dtype}, size={self.activation_memory}"
        if not self.has_saved_activation:
            s += " (not saved)"
        s += self._tensor_contents_str_helper()
        s += self._tensor_family_str_helper()
        if len(self.param_shapes) > 0:
            params_shapes_str = format_shape_list(self.param_shapes)
            s += (
                f"\n\tParams: Computed from params with shape {params_shapes_str}; "
                f"{self.num_params} params total ({self.param_memory})"
            )
        else:
            s += "\n\tParams: no params used"
        if self.module is None:
            module_str = "\n\tComputed inside module: not computed inside a module"
        else:
            module_str = f"\n\tComputed inside module: {self.module}"
        if not self.is_input:
            s += f"\n\tFunction: {self.func_name} (grad_fn_handle: {self.grad_fn_class_name}) {module_str}"
            if self.func_config:
                config_str = format_config_items(self.func_config)
                s += f"\n\tConfig: {config_str}"
            s += f"\n\tTime elapsed: {self.func_duration: .3E}s"
        if len(self.output_of_modules) > 0:
            output_of_modules_str = ", ".join(self.output_of_modules)
            s += f"\n\tOutput of modules: {output_of_modules_str}"
        else:
            s += "\n\tOutput of modules: none"
        if self.is_atomic_module:
            s += f"\n\tOutput of bottom-level module: {self.atomic_module_call}"
        lookup_keys_str = ", ".join([str(key) for key in self.lookup_keys])
        s += f"\n\tLookup keys: {lookup_keys_str}"

        return s

    def _tensor_contents_str_helper(self) -> str:
        """Returns short, readable string for the tensor contents."""
        try:
            out = self.out
        except PayloadUnavailableError:
            # A predicate save refuses payload reads for unselected ops; the
            # repr must degrade (the "(not saved)" marker already prints),
            # never propagate the refusal out of __repr__/__str__ (b1 R01).
            return ""
        if out is None:
            return ""
        else:
            s = ""
            s += f"\n\t\t{tensor_stats_summary(out)}"
            if not isinstance(out, torch.Tensor):
                # Preview-backend (non-torch) saved activation, e.g. MLX/tinygrad/
                # TF/JAX/Paddle. The slice-then-clone preview below relies on
                # torch-only methods (.detach(), .requires_grad, .clone()); the
                # stats-summary line above already reports shape/dtype safely, so
                # skip the raw-content preview rather than duck-typing torch-only
                # calls across every preview backend's array type.
                return s
            tensor_size_shown = 8
            # Use logged shape, not live tensor shape (#45)
            saved_shape = self.shape if self.shape is not None else out.shape
            # Slice first, then clone only the small slice (#73)
            if len(saved_shape) == 0:
                tensor_slice = out.detach().clone()
            elif len(saved_shape) == 1:
                num_dims = min(tensor_size_shown, saved_shape[0])
                tensor_slice = out[0:num_dims].detach().clone()
            elif len(saved_shape) == 2:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = out[0:num_dims, 0:num_dims].detach().clone()
            else:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = out.data
                for _ in range(len(saved_shape) - 2):
                    tensor_slice = tensor_slice[0]
                tensor_slice = tensor_slice[0:num_dims, 0:num_dims].detach().clone()
            tensor_slice.requires_grad = False
            s += f"\n\t\t{str(tensor_slice)}"
            if (len(saved_shape) > 0) and (max(saved_shape) > tensor_size_shown):
                s += "..."
        return s

    def _tensor_family_str_helper(self) -> str:
        """Return a formatted string summarising parent, child, sibling, spouse, and ancestor relationships."""
        s = "\n\tRelated Layers:"
        if len(self.parents) > 0:
            s += "\n\t\t- parent layers: " + ", ".join(self.parents)
        else:
            s += "\n\t\t- no parent layers"

        if len(self.children) > 0:
            s += "\n\t\t- child layers: " + ", ".join(self.children)
        else:
            s += "\n\t\t- no child layers"

        if len(self.siblings) > 0:
            s += "\n\t\t- shares parents with layers: " + ", ".join(self.siblings)
        else:
            s += "\n\t\t- shares parents with no other layers"

        if len(self.co_parents) > 0:
            s += "\n\t\t- shares children with layers: " + ", ".join(self.co_parents)
        else:
            s += "\n\t\t- shares children with no other layers"

        if self.has_input_ancestor:
            s += "\n\t\t- descendent of input layers: " + ", ".join(self.input_ancestors)
        else:
            s += "\n\t\t- tensor was created de novo inside the model (not computed from input)"

        if self.has_output_descendant:
            s += "\n\t\t- ancestor of output layers: " + ", ".join(self.output_descendants)
        else:
            s += "\n\t\t- tensor is not an ancestor of the model output; it terminates within the model"

        return s

    def __repr__(self) -> str:
        """Return the developer representation for this operation."""

        return self.__str__()


# ---------------------------------------------------------------------------
# The M5 facade: generated per-field data descriptors over (_core, _row)
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

        A ``PooledCell`` (the M14 duplicate/empty-container pooling) hydrates
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
    """Descriptor for one immutable-view relation field (M6, JMT-FORK-1).

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
    """Descriptor for the two group-membership fields (M7, JMT-FORK-1).

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
    """Descriptor for one shared-fact field (M7 FunctionCall/ParamAlias blocks).

    A ``_FACT`` cell means the value lives in the store's shared fact block:
    the first read hydrates the field's exact public container type for THIS
    row and caches it back, so identity is stable across reads, per-row
    in-place mutation stays isolated (a fresh container per row, exactly the
    pre-M7 semantics), and uninspected rows retain no per-row container. A
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
    (``_DataflowField``), and the shared-fact fields hydrate from the M7
    fact blocks (``_FactField``).
    """

    tuple_view_names = frozenset(OP_TUPLE_VIEW_FIELDS)
    frozenset_view_names = frozenset(OP_FROZENSET_VIEW_FIELDS + OP_BITSET_VIEW_FIELDS)
    dataflow_names = frozenset(OP_DATAFLOW_FIELDS)
    existing = vars(Op)
    for fid, name in enumerate(_OP_SLOT_NAMES):
        if name in existing:
            raise RuntimeError(f"Op facade collision: {name!r} is already defined on Op")
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
        setattr(Op, name, descriptor)


_install_op_field_descriptors()


def _compact_store_rows(store: Any, pool: dict[Any, Any]) -> None:
    """Pool repeated immutable metadata across a whole building-phase store.

    Column-major equivalent of ``Op._compact_metadata`` (same class ladder,
    same ``_UNPOOLED_SLOTS`` skips, same container-member handling) operating
    directly on the row cells, so a core-backed trace pools without paying
    the per-attribute descriptor protocol.

    Parameters
    ----------
    store:
        Building-phase ``OpRowStore`` (frozen stores are left untouched --
        pooling always precedes the physical freeze).
    pool:
        Pass-local ``pool key -> canonical instance`` table shared with any
        remaining per-op walks of the same trace.
    """

    rows = store.rows_building()
    if rows is None:
        return
    pooled_classes = _POOLED_CLASSES
    pool_get = pool.get
    for fid, name in enumerate(store.layout.names):
        if name in _UNPOOLED_SLOTS:
            continue
        for row_cells in rows:
            value = row_cells[fid]
            if value is None or value is _MISSING:
                continue
            cls = value.__class__
            if cls in pooled_classes:
                key = (cls, value.hex()) if cls is Duration else (cls, value)
                pooled = pool_get(key)
                if pooled is None:
                    pool[key] = value
                elif pooled is not value:
                    row_cells[fid] = pooled
            elif cls is list or cls is set or cls is dict:
                _pool_container_members(value, pool, 0)
            elif cls is tuple or cls is frozenset:
                pooled = _pool_value(value, pool)
                if pooled is not value:
                    row_cells[fid] = pooled
            elif isinstance(value, dict):
                _pool_container_members(value, pool, 0)


# Backward-compatible alias: TensorLog was the original name for
# Op before the Layer aggregate class was introduced in PR #92.
TensorLog = Op


# The tlspec v8 coordinated bump retired this class's S3 pre-release
# registrations (site_key, edge_substitutions, edge_replacement_stamps);
# their persisting policies are declared directly in FIELD_POLICY above.
