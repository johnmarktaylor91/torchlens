"""Log tensors produced by decorated torch operations.

This module creates Op entries, emits capture events, applies predicate
interventions, and saves or streams activation payloads for torch captures.
"""

# ruff: noqa: E402, F401

import contextlib
import copy
import dataclasses
import time
import warnings
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch.utils.weak import WeakIdKeyDictionary

from ... import _state as _st
from ..._capture.arg_positions import (
    DYNAMIC_SPEC_UNCACHEABLE,
    FUNC_ARG_SPECS,
    VARIADIC_TENSOR_ARG_FUNCS,
    ArgSpec,
    _cache_dynamic_spec,
    _normalize_func_name,
    dynamic_spec_covers_call,
    extract_tensors_and_params,
)
from ..._capture.flops import compute_backward_flops, compute_forward_flops
from ..._capture.plan import EnrichmentLevel
from ..._capture.predicates import (
    _evaluate_intervene_op,
    _evaluate_keep_op,
    _is_halt_only_capture,
    build_op_record_context,
)
from ..._capture.projections import (
    LiveOpView,
    append_projected_event,
    commit_op,
    get_active_recording_state,
)
from ..._capture.salient_args import extract_salient_args
from ..._capture.session import capture_session_for
from ..._capture.stop import evaluate_halt_stop, stop_directive_for_trace
from ..._capture_state_helpers import _is_uninitialized_param
from ..._data_classes._nonfinite import record_op_nonfinite
from ..._data_classes.internal_types import FuncExecutionContext
from ..._data_classes.op import (
    Op,
    _dedup_cached_identity_out,
    _dedup_saved_activation_out,
    _dtype_or_none,
    _effective_activation_save_mode,
    _memory_or_none,
    _recursive_safe_copy,
    _shape_or_none,
    _stamp_reference_out,
    apply_transform,
    register_relation_cell_encoding,
    register_row_release_property,
    train_mode_tripwire_armed,
    validate_streaming_transform_output,
    validate_train_mode_transform_output,
)
from ..._errors import UserTransformError
from ..._io import BlobRef
from ..._ir.container import (
    _SAFE_DEFAULT_FACTORIES,
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
from ..._ir.container_registry import (
    OUTPUT_TREE_MAX_DEPTH,
    ContainerLeafOccurrence,
    FuncSite,
    Phase,
    Role,
    walk_container,
)
from ..._ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
    OutputRef,
    OutputVersionEvent,
    ParentEdge,
)
from ..._ir.intervention import FireResult, FunctionEventInput

# Several imports below are re-exported: the ``_ops_*`` split modules import
# them back from this module at their bottoms (that is why this file carries a
# module-level ``ruff: noqa: F401``). Verify with the producer-parity suite
# before removing any import from this module.
from ..._ir.op_record import amend_lookback_retention
from ..._ir.predicate import RetroactiveCaptureDecision
from ..._ir.refs import ParamRef, TensorRef
from ..._ir.semantics import BackendSemantics, CapturePolicy
from ..._robustness import UnsupportedTensorVariantError
from ..._selection.selectors import (
    BaseSelector,
    label as make_label_selector,
)
from ..._state import pause_logging
from ..._training_validation import TrainingModeConfigError
from ..._utils._call_site import (
    CALL_SITE_CLASS_MARKER,  # noqa: F401 (re-exported to the split modules)
)
from ..._utils._callable_safety import _PURE_TENSOR_PROPERTY_NAMES
from ..._utils._mutation_names import (
    _INPLACE_AUGMENTED_ASSIGNMENT_DUNDER_EXCLUSIONS as _INPLACE_AUGMENTED_ASSIGNMENT_DUNDER_EXCLUSIONS,
    _SETTER_MUTATION_FUNC_NAMES as _SETTER_MUTATION_FUNC_NAMES,
)
from ..._utils._torch_compat import (
    saved_tensors_default_hooks_active,
    tensor_version_or_none,
    torch_structseq_field_names,
)
from ..._utils.collections import ensure_iterable, index_nested
from ..._utils.display import _timed_phase
from ..._utils.introspection import (
    _get_code_context,
    _get_tensors_and_params_from_obj,
    get_arg_tensors_for_resolution,
    get_vars_of_type_from_obj,
)
from ..._utils.tensor_utils import (
    fp8_widen_for_numeric_ops,
    get_memory_amount_from_metadata,
    is_functorch_wrapped_tensor,
    safe_copy,
    safe_to,
    tensor_nanequal,
)
from ...intervention.hooks import make_live_site_proxy, normalize_hook_plan
from ...intervention.runtime import active_intervention_context
from ...intervention.types import (
    ArgComponent,
    CapturedArgTemplate,
    EdgeUseRecord,
    FunctionRegistryKey,
    InterventionDecision,
    LiteralTensor,
    LiteralValue,
    ParentRef,
    TargetSpec,
    Unsupported,
)
from ...recording._halt import StopSignal
from ...recording._storage_resolver import _resolve_storage
from ...recording.exceptions import PredicateError
from ...recording.types import (
    ActivationRecord,
    CaptureSpec,
    ModuleStackFrame,
    RecordContext,
    StorageIntent,
)
from . import module_stack as _mstack
from ._completeness_escape_state import record_alias_mutation_candidate
from ._completeness_origins import internal_scalar_read
from ._param_fingerprint import param_fingerprint
from ._same_object_handoff import pop_same_object_mutation, set_same_object_mutation
from ._tl import (
    active_label_session_token,
    get_label_list,
    get_live_label_list,
    get_live_tensor_label,
    get_param_meta,
    get_tensor_label,
    get_tensor_meta,
    is_tensor_data_alias,
    mark_detached_saved_activation,
    mutated_parameter_label,
    promote_mutated_parameters,
    session_label_storage_intact,
    session_meta_is_anchored,
    set_tensor_label,
)
from .aliasing import (
    detect_torch_alias_contract,
    detect_torch_output_alias_contract,
    get_parent_contents_for_contract_position,
    parent_label_has_alias_contract,
)
from .buffer_writes import resolve_registered_buffer_address, session_validated_buffer_address
from .sources import log_source_tensor
from .tensor_tracking import (
    _add_tensor_backward_hook,
    _append_module_suffix_to_equivalence_class,
    _get_ancestors_from_parents,
    _get_equivalence_class,
    _locate_parent_tensors_in_args,
    _make_raw_param_group_barcode,
    _process_parent_param_ops,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ..._ir.op_record import OpRecord


class _AncestorBitset:
    """Compact storage for one finished Op ancestor closure.

    One instance per distinct closure per trace (interned by
    ``_compact_ancestor_sets``), so the lazily cached frozen view is also
    one-per-closure: the first public read of any member materializes the
    ``frozenset`` once and every sibling read shares it (M6 immutable-view
    contract, FORK-1).

    Parameters
    ----------
    labels
        Trace-local dense-index-to-label table shared by every ancestor bitmap.
    bits
        Integer bitmap whose set bits select entries from ``labels``.
    """

    __slots__ = ("labels", "bits", "_frozen_view")

    def __init__(self, labels: tuple[str, ...], bits: int) -> None:
        """Bind the shared label table and this closure's bitmap."""

        self.labels = labels
        self.bits = bits
        self._frozen_view: frozenset[str] | None = None

    def materialize(self) -> set[str]:
        """Return the closure as a fresh mutable set (internal use).

        Returns
        -------
        set[str]
            Mutable set of ancestor labels.
        """

        labels = self.labels
        bits = self.bits
        result: set[str] = set()
        while bits:
            lowest_bit = bits & -bits
            result.add(labels[lowest_bit.bit_length() - 1])
            bits ^= lowest_bit
        return result

    def frozen_view(self) -> frozenset[str]:
        """Return the public immutable view, cached per closure."""

        view = self._frozen_view
        if view is None:
            view = frozenset(self.materialize())
            self._frozen_view = view
        return view


# All four label-set closure fields intern into bitmaps at the freeze
# (r8 R60-2 widened the historical root/internal pair): ``input_ancestors``
# and ``output_descendants`` grow O(N*L) on deep graphs too, and leaving
# them as per-op frozensets kept the dominant retained-memory term.
_ANCESTOR_FIELD_NAMES = (
    "root_ancestors",
    "internal_source_ancestors",
    "input_ancestors",
    "output_descendants",
)
_ANCESTOR_SLOT_DESCRIPTORS = {
    field_name: vars(Op)[field_name] for field_name in _ANCESTOR_FIELD_NAMES
}

# The bitset is a sanctioned finished-cell encoding: reads through the
# overlay materialize its cached frozenset view, and a refresh re-run may
# assign it onto a detached-backed op (which the closed finished-store
# assignment check would otherwise refuse).
register_relation_cell_encoding(_AncestorBitset)


def _get_input_ancestors(op: Op) -> "set[str] | frozenset[str]":
    """Return ``op.input_ancestors``: staging set or frozen closure view."""

    return _get_ancestor_field(op, "input_ancestors")


def _set_input_ancestors(op: Op, value: set[str]) -> None:
    """Assign ``op.input_ancestors`` through its preserved slot descriptor."""

    _set_ancestor_field(op, "input_ancestors", value)


def _delete_input_ancestors(op: Op) -> None:
    """Delete ``op.input_ancestors`` through its preserved slot descriptor."""

    _delete_ancestor_field(op, "input_ancestors")


def _get_output_descendants(op: Op) -> "set[str] | frozenset[str]":
    """Return ``op.output_descendants``: staging set or frozen closure view."""

    return _get_ancestor_field(op, "output_descendants")


def _set_output_descendants(op: Op, value: set[str]) -> None:
    """Assign ``op.output_descendants`` through its preserved slot descriptor."""

    _set_ancestor_field(op, "output_descendants", value)


def _delete_output_descendants(op: Op) -> None:
    """Delete ``op.output_descendants`` through its preserved slot descriptor."""

    _delete_ancestor_field(op, "output_descendants")


setattr(
    Op,
    "input_ancestors",
    property(_get_input_ancestors, _set_input_ancestors, _delete_input_ancestors),
)
setattr(
    Op,
    "output_descendants",
    property(_get_output_descendants, _set_output_descendants, _delete_output_descendants),
)


_BULK_ALIAS_FREE_TORCH_FUNCTIONS = frozenset(
    {
        "__add__",
        "__eq__",
        "__iadd__",
        "__mul__",
        "__ne__",
        "__radd__",
        "__rmul__",
        "__sub__",
        "adaptive_avg_pool2d",
        "addmm",
        "all",
        "arange",
        "batch_norm",
        "conv2d",
        "cumsum",
        "diff",
        "embedding",
        "layer_norm",
        "linear",
        "max_pool2d",
        "pow",
        "relu_",
        "scaled_dot_product_attention",
        "tanh",
    }
)
"""Built-ins with exact empty alias/mutation contracts in the bulk default-RAM projection.

The two in-place names are accepted only when the wrapper supplied live arguments as their own
"copies"; the wrapper has already cloned their same-object return for logging in that regime.
"""


_LABEL_VERSION_SNAPSHOT: WeakIdKeyDictionary = WeakIdKeyDictionary()
"""Per-tensor ``_version`` at the moment TorchLens last labeled that tensor as an op output.

An in-place op returns the SAME tensor object as one of its inputs, so the output already
carries a capture label; a NON-mutating identity return (``x.cpu()`` on CPU, ``x.contiguous()``
on a contiguous tensor) ALSO returns its receiver and so ALSO carries a label. The two are
distinguished ONLY by whether the aliased tensor's version counter actually bumped. This weak,
identity-keyed side table records each labeled op output's version so a later op consuming that
tensor can tell a genuine mutation (version bumped since it was labeled) from an identity return
(version unchanged). It is capture-transient state, kept off the Trace field schema and dropped
automatically on tensor GC.

Entries are ``(label_session_token, version)`` pairs. A tensor that survives across captures
(a reused ``out=`` buffer, a cached activation the user held onto) keeps its module-lifetime
weak entry, so a baseline recorded by an EARLIER session must never be consulted by a LATER
one: the tensor may have been mutated between captures, and a stale baseline would falsely
classify a non-mutating identity return as in-place (W3 audit F7). Reads go through
:func:`_label_version_baseline`, which discards any entry whose session token is not the
active label session; writes go through :func:`_record_label_version_snapshot`.
"""

_VIEW_MUTATION_RELABELS: WeakIdKeyDictionary = WeakIdKeyDictionary()
"""Storage aliases whose label advanced because an in-place op wrote through a view.

Entries are ``(label_session_token, prior_label, mutation_label, id(view))``, written by
:func:`_record_label_version_snapshot` (``mutated_via=``) and read by
:func:`_bind_writeback_receiver_label`, which honours an entry only in its own label
session, only while the alias still carries ``mutation_label`` and only for that view.
Capture-transient, kept off the Trace field schema, dropped on tensor GC.
"""


CaptureProducerMode = Literal["exhaustive", "predicate"]

# The producer-unification dual-path env switch died
# with the legacy producer in P7: every torch capture freezes decomposed
# ``OpRecord`` rows. Preview backends keep emitting compat ``OpEvent``s until
# S15 and adapt at the one ingest boundary (``op_record_from_event``).


@dataclass(frozen=True, slots=True)
class CaptureProducerPolicy:
    """Precomputed producer routing for one capture mode.

    Parameters
    ----------
    mode
        Capture mode represented by this policy.
    emit
        Callable that emits operation events for the mode.
    """

    mode: CaptureProducerMode
    emit: Callable[
        [
            "Trace",
            Callable[..., Any],
            str,
            tuple[Any, ...],
            dict[str, Any],
            tuple[Any, ...],
            dict[str, Any],
            Any,
            FuncExecutionContext,
            bool,
            int,
        ],
        None,
    ]


_CAPTURE_PRODUCER_POLICIES: dict[CaptureProducerMode, CaptureProducerPolicy] = {}

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
    #: Saved-band decomposition (observe items 7-8): saved_parameter /
    #: saved_buffer / saved_activation / newly_saved_bytes sub-counters
    #: beside the gross band, or None when no grad_fn walk ran.
    autograd_band: dict[str, int] | None = None


#: FunctionCallRef fields the per-output logging path genuinely rewrites
#: (``_log_output_tensor_info``): FLOPs derive from the output shape and
#: ``is_inplace`` from the output tensor's version. Everything else on the
#: ref is a call-level fact, so sibling outputs of one wrapped call share
#: ONE frozen ref (M7); a sibling whose per-output facts differ gets a
#: ``dataclasses.replace`` derivative that still shares every container
#: field by reference.
_FUNCTION_REF_PER_OUTPUT_FIELDS = ("flops_forward", "flops_backward", "is_inplace")


@dataclass(slots=True)
class ExhaustiveOpDraft:
    """Exhaustive-pipeline draft (the three ``_make_layer_log_entry`` sites)."""

    trace: Any
    fields_dict: dict[str, Any]
    tensor: torch.Tensor
    fire_results: tuple[FireResult, ...]
    module_stack: tuple[ModuleFrame, ...] | None
    call_ref_box: list[FunctionCallRef] | None

    pipeline: str = dataclasses.field(default="exhaustive", init=False)

    @property
    def grad_fn_handle(self) -> Any:
        """The independently carried selected autograd handle (index stage)."""

        return self.fields_dict["grad_fn_handle"]

    def freeze(self) -> Any:
        """Construct the journal record ONCE from the final draft state."""

        return _op_record_from_log(
            self.trace,
            self.fields_dict,
            self.tensor,
            self.fire_results,
            module_stack=self.module_stack,
            call_ref_box=self.call_ref_box,
        )


# Canonical set lives in ``torchlens._utils._callable_safety`` so this capture-side
# keyer, the load-side resolver, and the security gate's recognized-operator predicate
# can never drift apart on the safe pure-read property surface. The set is STRUCTURALLY
# computed there (r45) by probing every live ``TensorBase`` getset descriptor for a
# storage-sharing, autograd-preserving, non-mutating view -- it admits
# ``{T, mT, H, mH, real, imag}`` and denies ``data`` (autograd-detaching lvalue alias).
# ``.H`` / ``.mH`` are keyed here as ``("torch.Tensor", <name>, "method")`` exactly like
# ``.T`` / ``.mT`` rather than falling through to an unresolvable custom
# ``getset_descriptor.__get__`` key (the r44 corr1_1 / secF_1 over-deny).
_SAFE_TENSOR_PROPERTY_NAMES: frozenset[str] = _PURE_TENSOR_PROPERTY_NAMES


# Private implementation slices; public names remain owned by this module.

# Late imports (after this module's own definitions): the split modules import this
# module's classes and constants back, so they must exist before the splits load.
from ._ops_capture_records import (  # noqa: E402
    _delete_ancestor_field,
    _delete_internal_source_ancestors,
    _delete_root_ancestors,
    _get_ancestor_field,
    _get_internal_source_ancestors,
    _get_root_ancestors,
    _set_ancestor_field,
    _set_internal_source_ancestors,
    _set_root_ancestors,
)

setattr(
    Op,
    "root_ancestors",
    property(_get_root_ancestors, _set_root_ancestors, _delete_root_ancestors),
)
setattr(
    Op,
    "internal_source_ancestors",
    property(
        _get_internal_source_ancestors,
        _set_internal_source_ancestors,
        _delete_internal_source_ancestors,
    ),
)
# Each overlay reads through its preserved slot descriptor (raising on an
# unset cell) and deletes only that cell, so op removal may release rows
# directly. Registration reads the live class attribute, so it runs after
# all four overlays are installed.
for _ancestor_field_name in _ANCESTOR_FIELD_NAMES:
    register_row_release_property(_ancestor_field_name)
del _ancestor_field_name
from ._ops_container_base import (  # noqa: E402
    _op_record_from_log,
)
