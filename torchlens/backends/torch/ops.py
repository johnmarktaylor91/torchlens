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
from ..._capture_state_helpers import _is_uninitialized_param
from ..._errors import TorchLensPostfuncError
from ..._io import BlobRef
from ..._robustness import UnsupportedTensorVariantError
from ..._state import pause_logging
from ..._training_validation import TrainingModeConfigError
from ...capture.arg_positions import (
    DYNAMIC_SPEC_UNCACHEABLE,
    FUNC_ARG_SPECS,
    VARIADIC_TENSOR_ARG_FUNCS,
    ArgSpec,
    _cache_dynamic_spec,
    _normalize_func_name,
    dynamic_spec_covers_call,
    extract_tensors_and_params,
)
from ...capture.flops import compute_backward_flops, compute_forward_flops
from ...capture.plan import EnrichmentLevel
from ...capture.predicates import (
    _evaluate_intervene_op,
    _evaluate_keep_op,
    _is_halt_only_capture,
    build_op_record_context,
)
from ...capture.projections import (
    LiveOpView,
    append_projected_event,
    commit_op,
    get_active_recording_state,
)
from ...capture.salient_args import extract_salient_args
from ...capture.session import capture_session_for
from ...capture.stop import evaluate_halt_stop, stop_directive_for_trace
from ...data_classes._nonfinite import record_op_nonfinite
from ...data_classes.internal_types import FuncExecutionContext
from ...data_classes.op import (
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
    train_mode_tripwire_armed,
    validate_streaming_transform_output,
    validate_train_mode_transform_output,
)
from ...fastlog._halt import HaltSignal
from ...fastlog._storage_resolver import _resolve_storage
from ...fastlog.exceptions import PredicateError
from ...fastlog.types import (
    ActivationRecord,
    CaptureSpec,
    ModuleStackFrame,
    RecordContext,
    StorageIntent,
)
from ...intervention.hooks import make_live_site_proxy, normalize_hook_plan
from ...intervention.runtime import active_intervention_context
from ...intervention.selectors import (
    BaseSelector,
    label as make_label_selector,
)
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
from ...ir.container import (
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
from ...ir.container_registry import (
    OUTPUT_TREE_MAX_DEPTH,
    ContainerLeafOccurrence,
    FuncSite,
    Phase,
    Role,
    walk_container,
)
from ...ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
    OutputRef,
    OutputVersionEvent,
    ParentEdge,
)
from ...ir.intervention import FireResult, FunctionEventInput

# NOT unused (and NOT dead, despite reading that way to a scanner -- finding
# B1-23c proposed deleting it and three producer-parity tests died with
# NameError): the ``_ops_*`` slices below are REBOUND into this module's
# ``globals()``, so every name their bodies reference must resolve HERE.
# ``amend_lookback_retention`` is used by the rebound
# ``_ops_retention.amend_lookback_retention`` caller. That is also why this file
# carries a module-level ``ruff: noqa: F401``: rebind-required imports look
# unused to static analysis. Verify with the producer-parity suite before
# removing any import from this module.
from ...ir.op_record import amend_lookback_retention
from ...ir.predicate import RetroactiveCaptureDecision
from ...ir.refs import ParamRef, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...utils._callable_safety import _PURE_TENSOR_PROPERTY_NAMES
from ...utils._torch_compat import (
    saved_tensors_default_hooks_active,
    tensor_version_or_none,
    torch_structseq_field_names,
)
from ...utils.collections import ensure_iterable, index_nested
from ...utils.display import _timed_phase
from ...utils.introspection import (
    _get_code_context,
    _get_tensors_and_params_from_obj,
    get_arg_tensors_for_resolution,
    get_vars_of_type_from_obj,
)
from ...utils.tensor_utils import (
    fp8_widen_for_numeric_ops,
    get_memory_amount_from_metadata,
    is_functorch_wrapped_tensor,
    safe_copy,
    safe_to,
    tensor_nanequal,
)
from . import module_stack as _mstack
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
    pop_same_object_mutation,
    promote_mutated_parameters,
    session_label_storage_intact,
    session_meta_is_anchored,
    set_same_object_mutation,
    set_tensor_label,
)
from .aliasing import (
    detect_torch_alias_contract,
    detect_torch_output_alias_contract,
    get_parent_contents_for_contract_position,
    parent_label_has_alias_contract,
)
from .buffer_writes import resolve_registered_buffer_address, session_validated_buffer_address
from .completeness_witness import internal_scalar_read, record_alias_mutation_candidate
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
    from ...data_classes.trace import Trace
    from ...ir.op_record import OpRecord


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


def _get_root_ancestors(op: Op) -> "set[str] | frozenset[str]":
    """Return ``op.root_ancestors``: staging set or frozen closure view."""

    return _get_ancestor_field(op, "root_ancestors")


def _set_root_ancestors(op: Op, value: set[str]) -> None:
    """Assign ``op.root_ancestors`` through its preserved slot descriptor."""

    _set_ancestor_field(op, "root_ancestors", value)


def _delete_root_ancestors(op: Op) -> None:
    """Delete ``op.root_ancestors`` through its preserved slot descriptor."""

    _delete_ancestor_field(op, "root_ancestors")


def _get_internal_source_ancestors(op: Op) -> "set[str] | frozenset[str]":
    """Return ``op.internal_source_ancestors``: staging set or frozen closure view."""

    return _get_ancestor_field(op, "internal_source_ancestors")


def _set_internal_source_ancestors(op: Op, value: set[str]) -> None:
    """Assign ``op.internal_source_ancestors`` through its preserved slot descriptor."""

    _set_ancestor_field(op, "internal_source_ancestors", value)


def _delete_internal_source_ancestors(op: Op) -> None:
    """Delete ``op.internal_source_ancestors`` through its preserved slot descriptor."""

    _delete_ancestor_field(op, "internal_source_ancestors")


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


_SETTER_MUTATION_FUNC_NAMES = frozenset({"__setitem__", "__delitem__"})
"""Setter-style in-place ops whose names do NOT end in ``_`` (used only as the no-baseline
is_inplace fallback). They mutate their target but log a reconstructed output tensor, so the
version-baseline signal is unavailable and the operator name is the honest mutation signature.
"""

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

_INPLACE_AUGMENTED_ASSIGNMENT_DUNDER_EXCLUSIONS = frozenset(
    {"__index__", "__init__", "__init_subclass__", "__int__", "__invert__", "__iter__"}
)
"""Non-mutating ``__i*`` dunders excluded from augmented-assignment classification.

Unary dunders such as ``__invert__`` intentionally stay out of this set.
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

_EMIT_BY_MODE: dict[CaptureProducerMode, str] = {
    "exhaustive": "_emit_exhaustive_operation_events",
    "predicate": "_emit_predicate_operation_events",
}


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


# Canonical set lives in ``torchlens.utils._callable_safety`` so this capture-side
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
from ..._split_rebind import rebind_function as _rebind_function
from . import (
    _ops_activations as _ops_activations,
    _ops_arguments as _ops_arguments,
    _ops_autograd as _ops_autograd,
    _ops_capture_records as _ops_capture_records,
    _ops_container_base as _ops_container_base,
    _ops_containers as _ops_containers,
    _ops_emission as _ops_emission,
    _ops_exhaustive as _ops_exhaustive,
    _ops_finalize as _ops_finalize,
    _ops_interventions as _ops_interventions,
    _ops_predicate_events as _ops_predicate_events,
    _ops_predicates as _ops_predicates,
    _ops_retention as _ops_retention,
    _ops_shared_fields as _ops_shared_fields,
)

_get_ancestor_field = _rebind_function(_ops_capture_records._get_ancestor_field, globals())
_set_ancestor_field = _rebind_function(_ops_capture_records._set_ancestor_field, globals())
_delete_ancestor_field = _rebind_function(_ops_capture_records._delete_ancestor_field, globals())
_compact_ancestor_sets = _rebind_function(_ops_capture_records._compact_ancestor_sets, globals())
_is_inplace_augmented_assignment_dunder = _rebind_function(
    _ops_capture_records._is_inplace_augmented_assignment_dunder, globals()
)
_record_label_version_snapshot = _rebind_function(
    _ops_capture_records._record_label_version_snapshot, globals()
)
_label_version_baseline = _rebind_function(_ops_capture_records._label_version_baseline, globals())
_stamp_same_object_mutation = _rebind_function(
    _ops_capture_records._stamp_same_object_mutation, globals()
)
get_capture_producer_policy = _rebind_function(
    _ops_capture_records.get_capture_producer_policy, globals()
)
set_capture_producer_policy = _rebind_function(
    _ops_capture_records.set_capture_producer_policy, globals()
)
_should_keep_alias_mutation_contract = _rebind_function(
    _ops_capture_records._should_keep_alias_mutation_contract, globals()
)
_snapshot_exhaustive_module_stack = _rebind_function(
    _ops_capture_records._snapshot_exhaustive_module_stack, globals()
)
_tensor_ref_from_fields = _rebind_function(_ops_capture_records._tensor_ref_from_fields, globals())
_module_frames_from_fields = _rebind_function(
    _ops_capture_records._module_frames_from_fields, globals()
)
_param_refs_from_fields = _rebind_function(_ops_capture_records._param_refs_from_fields, globals())
_parent_edges_from_fields = _rebind_function(
    _ops_capture_records._parent_edges_from_fields, globals()
)
_function_call_ref_from_fields = _rebind_function(
    _ops_capture_records._function_call_ref_from_fields, globals()
)
_resolve_call_function_ref = _rebind_function(
    _ops_capture_records._resolve_call_function_ref, globals()
)
_exhaustive_freeze_refs = _rebind_function(_ops_capture_records._exhaustive_freeze_refs, globals())
_exhaustive_output_ref = _rebind_function(_ops_capture_records._exhaustive_output_ref, globals())
_exhaustive_capture_policy = _rebind_function(
    _ops_capture_records._exhaustive_capture_policy, globals()
)
_op_record_from_log = _rebind_function(_ops_container_base._op_record_from_log, globals())
_is_namedtuple_instance = _rebind_function(_ops_container_base._is_namedtuple_instance, globals())
_torch_return_type_fields = _rebind_function(
    _ops_container_base._torch_return_type_fields, globals()
)
_non_iterable_type_error = _rebind_function(_ops_container_base._non_iterable_type_error, globals())
_iter_sequence_items = _rebind_function(_ops_container_base._iter_sequence_items, globals())
_try_build_container_spec = _rebind_function(
    _ops_container_base._try_build_container_spec, globals()
)
_fallback_address_to_path = _rebind_function(
    _ops_container_base._fallback_address_to_path, globals()
)
_is_hf_model_output = _rebind_function(_ops_container_base._is_hf_model_output, globals())
_container_type_ref = _rebind_function(_ops_container_base._container_type_ref, globals())
_safe_default_factory_name = _rebind_function(
    _ops_container_base._safe_default_factory_name, globals()
)
_mapping_reconstruction = _rebind_function(_ops_container_base._mapping_reconstruction, globals())
_object_holds_tensor = _rebind_function(_ops_container_base._object_holds_tensor, globals())
_leaf_is_reconstructable = _rebind_function(_ops_container_base._leaf_is_reconstructable, globals())
_build_container_spec = _rebind_function(_ops_containers._build_container_spec, globals())
_build_container_spec_unguarded = _rebind_function(
    _ops_containers._build_container_spec_unguarded, globals()
)
_known_output_container_children = _rebind_function(
    _ops_containers._known_output_container_children, globals()
)
_walk_supported_output_container = _rebind_function(
    _ops_containers._walk_supported_output_container, globals()
)
_prove_runnable_output_lossless = _rebind_function(
    _ops_containers._prove_runnable_output_lossless, globals()
)
_prove_runnable_output_lossless_unguarded = _rebind_function(
    _ops_containers._prove_runnable_output_lossless_unguarded, globals()
)
runnable_output_losslessness = _rebind_function(
    _ops_containers.runnable_output_losslessness, globals()
)
_walk_output_tensors_with_paths = _rebind_function(
    _ops_containers._walk_output_tensors_with_paths, globals()
)
_function_registry_key = _rebind_function(_ops_arguments._function_registry_key, globals())
_literal_value_supported = _rebind_function(_ops_arguments._literal_value_supported, globals())
_classify_arg_component = _rebind_function(_ops_arguments._classify_arg_component, globals())
_build_args_template = _rebind_function(_ops_arguments._build_args_template, globals())
_arg_location_to_path = _rebind_function(_ops_arguments._arg_location_to_path, globals())
_build_edge_use_records = _rebind_function(_ops_arguments._build_edge_use_records, globals())
_session_validated_parameter = _rebind_function(
    _ops_arguments._session_validated_parameter, globals()
)
_tensor_has_known_provenance = _rebind_function(
    _ops_arguments._tensor_has_known_provenance, globals()
)
_unattributed_tensor_arg_positions = _rebind_function(
    _ops_emission._unattributed_tensor_arg_positions, globals()
)
log_function_output_tensors = _rebind_function(_ops_emission.log_function_output_tensors, globals())
_emit_operation_events = _rebind_function(_ops_emission._emit_operation_events, globals())
apply_live_hooks_to_outputs = _rebind_function(_ops_emission.apply_live_hooks_to_outputs, globals())
_apply_live_hooks_to_outputs_legacy = _rebind_function(
    _ops_interventions._apply_live_hooks_to_outputs_legacy, globals()
)
_apply_predicate_mode_interventions_to_outputs = _rebind_function(
    _ops_interventions._apply_predicate_mode_interventions_to_outputs, globals()
)
_trace_intervene_options = _rebind_function(_ops_interventions._trace_intervene_options, globals())
_predicate_hook_metadata = _rebind_function(_ops_interventions._predicate_hook_metadata, globals())
_record_predicate_intervention_spec = _rebind_function(
    _ops_interventions._record_predicate_intervention_spec, globals()
)
_live_output_index = _rebind_function(_ops_interventions._live_output_index, globals())
_apply_predicate_intervention = _rebind_function(
    _ops_interventions._apply_predicate_intervention, globals()
)
_iter_loggable_live_outputs = _rebind_function(
    _ops_interventions._iter_loggable_live_outputs, globals()
)
_replace_output_tensors_by_path = _rebind_function(
    _ops_interventions._replace_output_tensors_by_path, globals()
)
_set_tensor_live_fire_results = _rebind_function(
    _ops_interventions._set_tensor_live_fire_results, globals()
)
_pop_tensor_live_fire_results = _rebind_function(
    _ops_interventions._pop_tensor_live_fire_results, globals()
)
_replace_output_value = _rebind_function(_ops_predicates._replace_output_value, globals())
_record_predicate_output = _rebind_function(_ops_predicates._record_predicate_output, globals())
_is_default_ram_payload = _rebind_function(_ops_predicates._is_default_ram_payload, globals())
_predicate_function_ref = _rebind_function(_ops_predicates._predicate_function_ref, globals())
_predicate_backend_semantics = _rebind_function(
    _ops_predicates._predicate_backend_semantics, globals()
)
_has_proven_alias_free_output = _rebind_function(
    _ops_predicates._has_proven_alias_free_output, globals()
)
_has_only_builtin_tensor_leaves = _rebind_function(
    _ops_predicates._has_only_builtin_tensor_leaves, globals()
)
_alias_free_backend_semantics = _rebind_function(
    _ops_predicates._alias_free_backend_semantics, globals()
)
_emit_predicate_operation_events = _rebind_function(
    _ops_predicate_events._emit_predicate_operation_events, globals()
)
_build_graph_relationship_fields = _rebind_function(
    _ops_predicate_events._build_graph_relationship_fields, globals()
)
_extract_arg_tensors_and_params = _rebind_function(
    _ops_predicate_events._extract_arg_tensors_and_params, globals()
)
_build_param_fields = _rebind_function(_ops_predicate_events._build_param_fields, globals())
_build_module_context_fields = _rebind_function(
    _ops_predicate_events._build_module_context_fields, globals()
)
_build_shared_fields_dict = _rebind_function(
    _ops_shared_fields._build_shared_fields_dict, globals()
)
_classify_new_tensor_in_trace = _rebind_function(
    _ops_shared_fields._classify_new_tensor_in_trace, globals()
)
_tag_tensor_and_track_variations = _rebind_function(
    _ops_shared_fields._tag_tensor_and_track_variations, globals()
)
_get_parent_output_version_snapshot = _rebind_function(
    _ops_shared_fields._get_parent_output_version_snapshot, globals()
)
_emit_exhaustive_operation_events = _rebind_function(
    _ops_exhaustive._emit_exhaustive_operation_events, globals()
)
_project_foreach_member_parent_fields = _rebind_function(
    _ops_exhaustive._project_foreach_member_parent_fields, globals()
)
_get_parent_contents = _rebind_function(_ops_exhaustive._get_parent_contents, globals())
_output_should_be_logged = _rebind_function(_ops_exhaustive._output_should_be_logged, globals())
_check_if_tensor_arg = _rebind_function(_ops_exhaustive._check_if_tensor_arg, globals())
_iter_autograd_saved_candidates = _rebind_function(
    _ops_exhaustive._iter_autograd_saved_candidates, globals()
)
_collect_tensor_values = _rebind_function(_ops_exhaustive._collect_tensor_values, globals())
_add_autograd_saved_tensor = _rebind_function(_ops_autograd._add_autograd_saved_tensor, globals())
_classify_saved_storage = _rebind_function(_ops_autograd._classify_saved_storage, globals())
_get_autograd_saved_stats_by_output = _rebind_function(
    _ops_autograd._get_autograd_saved_stats_by_output, globals()
)
_partition_output_entries_with_autograd_stats = _rebind_function(
    _ops_autograd._partition_output_entries_with_autograd_stats, globals()
)
_register_call_output_container_snapshot = _rebind_function(
    _ops_autograd._register_call_output_container_snapshot, globals()
)
register_call_input_container_snapshots = _rebind_function(
    _ops_autograd.register_call_input_container_snapshots, globals()
)
_container_leaf_occurrences_from_entries = _rebind_function(
    _ops_autograd._container_leaf_occurrences_from_entries, globals()
)
_get_autograd_saved_stats_for_tensor = _rebind_function(
    _ops_autograd._get_autograd_saved_stats_for_tensor, globals()
)
# Rebound helper the shape-metadata reader needs in the caller's namespace.
_metadata_shape = _rebind_function(_ops_activations._metadata_shape, globals())
_log_output_tensor_info = _rebind_function(_ops_activations._log_output_tensor_info, globals())
_save_activation_fields = _rebind_function(_ops_activations._save_activation_fields, globals())
_stream_activation_fields = _rebind_function(_ops_activations._stream_activation_fields, globals())
_retention_device = _rebind_function(_ops_activations._retention_device, globals())
_admit_save_budget = _rebind_function(_ops_retention._admit_save_budget, globals())
_commit_save_budget = _rebind_function(_ops_retention._commit_save_budget, globals())
_charge_saved_args_budget = _rebind_function(_ops_retention._charge_saved_args_budget, globals())
_admit_saved_args_budget = _rebind_function(_ops_retention._admit_saved_args_budget, globals())
_commit_saved_args_budget = _rebind_function(_ops_retention._commit_saved_args_budget, globals())
_iter_tree_tensors = _ops_retention._iter_tree_tensors
_save_predicate_activation_fields = _rebind_function(
    _ops_retention._save_predicate_activation_fields, globals()
)
_stream_predicate_payloads = _rebind_function(_ops_retention._stream_predicate_payloads, globals())
_module_stack_frames_from_fields = _rebind_function(
    _ops_retention._module_stack_frames_from_fields, globals()
)
_append_trace_predicate_context = _rebind_function(
    _ops_retention._append_trace_predicate_context, globals()
)
_trace_followed_by_candidate_selector = _rebind_function(
    _ops_retention._trace_followed_by_candidate_selector, globals()
)
_retain_lookback_candidate = _rebind_function(_ops_retention._retain_lookback_candidate, globals())
_copy_lookback_payload = _rebind_function(_ops_retention._copy_lookback_payload, globals())
_apply_retroactive_decision = _rebind_function(
    _ops_retention._apply_retroactive_decision, globals()
)
_replace_event_with_retained_payload = _rebind_function(
    _ops_retention._replace_event_with_retained_payload, globals()
)
_build_trace_predicate_context = _rebind_function(
    _ops_retention._build_trace_predicate_context, globals()
)
_trace_predicate_context_key = _rebind_function(
    _ops_finalize._trace_predicate_context_key, globals()
)
_cache_trace_predicate_context = _rebind_function(
    _ops_finalize._cache_trace_predicate_context, globals()
)
_pop_trace_predicate_context = _rebind_function(
    _ops_finalize._pop_trace_predicate_context, globals()
)
_evaluate_trace_save_predicate = _rebind_function(
    _ops_finalize._evaluate_trace_save_predicate, globals()
)
_module_filter_namespace = _rebind_function(_ops_finalize._module_filter_namespace, globals())
_note_module_filter_suppression = _rebind_function(
    _ops_finalize._note_module_filter_suppression, globals()
)
_make_layer_log_entry = _rebind_function(_ops_finalize._make_layer_log_entry, globals())
_emit_op_echo = _rebind_function(_ops_finalize._emit_op_echo, globals())
_raise_if_nonfinite_requested = _rebind_function(
    _ops_finalize._raise_if_nonfinite_requested, globals()
)
_record_nonfinite_if_requested = _rebind_function(
    _ops_finalize._record_nonfinite_if_requested, globals()
)

_split_namespace = {name: value for name, value in globals().items() if not name.startswith("__")}
for _split_module in (
    _ops_capture_records,
    _ops_container_base,
    _ops_containers,
    _ops_arguments,
    _ops_emission,
    _ops_interventions,
    _ops_predicates,
    _ops_predicate_events,
    _ops_shared_fields,
    _ops_exhaustive,
    _ops_autograd,
    _ops_activations,
    _ops_retention,
    _ops_finalize,
):
    _split_module.__dict__.update(_split_namespace)
