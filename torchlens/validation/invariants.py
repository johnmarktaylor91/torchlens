"""Metadata invariant checks for ``Trace`` and its sub-objects.

Single entry point: ``check_metadata_invariants(trace)`` runs all checks
and raises ``MetadataInvariantError`` on the first failure.

**Phase 1 -- Structural invariants:**
  A. Trace self-consistency (counts, timing, label uniqueness)
  B. Special layer lists match per-layer boolean flags
  C. Graph topology (parent-child bidirectionality, boolean flag consistency)
  D. Op field consistency (shape, dtype, pass numbering, nesting)
  E. Recurrence / loop invariants (is_recurrent, pass dicts)
  F. Branching invariants (is_branching)
  F2. Conditional metadata invariants (15 conditional consistency checks)
  G. Op <-> Layer cross-references (pass numbering, back-pointers)
  H. Module <-> Layer containment (layers, module pass layers, reverse check)
  I. Module hierarchy (address parent-child bidirectionality, pass consistency)
  J. Param cross-references (Param -> layer, uses_params flag)
  K. Buffer cross-references (buffer_layers list, Buffer module references)
  L. Equivalence group symmetry (op_equivalence_classes labels are valid)

**Phase 2 -- Semantic invariants:**
  M. Graph ordering (raw_index uniqueness/monotonicity, topological order, no raw labels)
  N. Loop detection invariants (recurrent_ops symmetry, func identity, param sharing, pass numbering)
  O. Distance / reachability (min <= max, input/output layer distances == 0, ancestor/descendent consistency)
  P. Graph connectivity (non-input non-buffer layers have parents, orphans removed)
  Q. Module containment logic (address acyclicity, depth consistency, nested path ordering)
  R. Lookup key bidirectionality (forward/reverse dicts, raw/final label maps)
"""

from __future__ import annotations

import re

# The imports below marked ``noqa: F401`` are LOAD-BEARING despite having no
# reference in this file: ``_rebind_function`` rebinds every split-child
# function onto THIS module's globals, so the rebound bodies resolve these
# names here at call time (deleting them raises NameError inside the checks
# -- verified live). Per-line noqa keeps F401 armed for genuinely dead code
# in the rest of the file (b9 R43-3; the former file-wide blanket is gone).
from collections import Counter, defaultdict  # noqa: F401 (rebound-child globals)
from collections.abc import Callable, Iterable, Mapping  # noqa: F401 (rebound-child globals)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast  # noqa: F401 (rebound-child globals)

from .._errors import _ActionableErrorMixin
from .._split_rebind import rebind_function as _rebind_function
from ..errors._base import ValidationError
from ..ir.container import (  # noqa: F401 (rebound-child globals)
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    TupleIndex,
)
from . import (
    _invariants_backward_domain as _invariants_backward_domain,
    _invariants_backward_flow as _invariants_backward_flow,
    _invariants_backward_graph as _invariants_backward_graph,
    _invariants_buffers as _invariants_buffers,
    _invariants_conditional_base as _invariants_conditional_base,
    _invariants_conditional_modules as _invariants_conditional_modules,
    _invariants_conditionals as _invariants_conditionals,
    _invariants_connectivity as _invariants_connectivity,
    _invariants_entry as _invariants_entry,
    _invariants_equivalence as _invariants_equivalence,
    _invariants_modules_params as _invariants_modules_params,
    _invariants_payloads as _invariants_payloads,
    _invariants_primitive_ops as _invariants_primitive_ops,
    _invariants_topology as _invariants_topology,
)
from .status import (  # noqa: F401 (rebound-child globals)
    has_importer_region_provenance,
    is_region_replay_annotation,
)

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

InvariantApplicability = Literal["torch", "non_torch", "all"]
MetadataInvariantFunc = Callable[["Trace"], object]


class MetadataInvariantError(_ActionableErrorMixin, ValidationError, ValueError):
    """Raised when a metadata invariant check fails.

    Embeds the check name (e.g., ``"graph_topology"``) in the message prefix
    and stores it as an attribute for programmatic inspection in tests.
    ``_ActionableErrorMixin`` supplies ``__reduce__`` so the strict two-argument
    constructor survives pickle/deepcopy/process boundaries (R64-F1) instead of
    degrading to a bare ``TypeError`` when ``cls(*args)`` is replayed.
    """

    def __init__(self, check_name: str, message: str) -> None:
        """Initialize a metadata invariant failure.

        Parameters
        ----------
        check_name:
            Invariant check group name.
        message:
            Human-readable failure detail.
        """

        super().__init__(f"[{check_name}] {message}")
        self.check_name = check_name


@dataclass(frozen=True)
class InvariantResult:
    """Structured result for an individual metadata invariant.

    Attributes
    ----------
    name:
        Invariant check name.
    passed:
        Whether the invariant passed.
    message:
        Optional diagnostic message.
    """

    name: str
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class MetadataInvariantContract:
    """Backend applicability contract for one metadata invariant check.

    Attributes
    ----------
    name:
        Stable check name used for dispatch introspection.
    check:
        Callable that runs the invariant.
    applies_to:
        Backend family this check currently runs on.
    requires_capability:
        Optional backend capability flag that must be truthy for the check to run.
    """

    name: str
    check: MetadataInvariantFunc
    applies_to: InvariantApplicability
    requires_capability: str | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A. Trace self-consistency
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# B. Special layer lists ↔ Op flags
# ---------------------------------------------------------------------------

_SPECIAL_LIST_FLAG_PAIRS = [
    ("input_layers", "is_input", "layer"),
    ("output_layers", "is_output", "layer"),
    ("buffer_layers", "is_buffer", "layer"),
    ("internal_source_ops", "is_internal_source", "op"),
    ("internal_sink_ops", "is_internal_sink", "op"),
]


# ---------------------------------------------------------------------------
# C. Graph topology
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# D. Op field consistency
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# E. Recurrence / loop invariants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# F. Branching invariants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# F2. Conditional metadata invariants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# G. Op ↔ Layer cross-references
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# H. Module ↔ Layer containment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# I. Module hierarchy consistency
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# J. Param ↔ Layer ↔ Module cross-references
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# K. Buffer cross-references
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# L. Equivalence group symmetry
# ---------------------------------------------------------------------------


# Sentinel for "no canonical `equivalent_ops` object could be read". Distinct from
# `None`, which is a legitimate stored value that the checks below must still reject.
_EQUIVALENT_OPS_UNAVAILABLE = object()


# ---------------------------------------------------------------------------
# M. Graph ordering invariants
# ---------------------------------------------------------------------------

# Raw labels are the internal identifiers assigned during capture; every one must be
# replaced by its final human-readable label during postprocessing.
#
# The pattern used to be ``^l_\d+$``, a spelling TorchLens no longer emits: real raw
# labels are ``{func}_{n}_{m}_raw`` (``add_1_4_raw``, ``output_1_raw``,
# ``buffer_2_raw``), so the check could never fire on anything. A working version would
# have caught the dangling ``buffer_source`` raw label by construction, which is exactly
# what it does now.
_RAW_LABEL_PATTERN = re.compile(r"_raw$")

# Op fields that may hold raw labels and are therefore scanned by the survival check.
# Label SEQUENCES/SETS and the SCALAR label field both count: scanning only
# ``layer_labels`` left every relation and scalar label surface unguarded.
_RAW_LABEL_BEARING_LIST_FIELDS = (
    "parents",
    "children",
    "input_ancestors",
    "output_descendants",
    "root_ancestors",
    "internal_source_ancestors",
    "internal_source_parents",
    "conditional_entry_children",
    "conditional_then_children",
    "conditional_else_children",
    "equivalent_ops",
    "recurrent_ops",
)
# ``site_key`` joins the sweep roster; its COMPONENT-level scan (a raw label
# embedded before the ordinal tail is invisible to this whole-string pattern)
# lives in ``_invariants_sites._check_site_key_invariants``.
_RAW_LABEL_BEARING_SCALAR_FIELDS = ("buffer_source", "module", "atomic_module_call", "site_key")

# Dict-shaped relation metadata carrying labels on either axis:
# ``conditional_elif_children`` holds label LISTS as values (int keys), and
# ``parent_arg_positions`` holds labels as KEYS. Both are persisted KEEP and
# relabeled, but the list-field scan iterates dict keys only and neither was
# rostered, so they were structurally unscanned -- the exact class the roster
# was built to stop (p2 #13 / R08). The scan checks keys AND value strings.
_RAW_LABEL_BEARING_DICT_FIELDS = ("conditional_elif_children", "parent_arg_positions")


# ---------------------------------------------------------------------------
# N. Layer equivalence / loop detection invariants
# ---------------------------------------------------------------------------


# Note: subgraph-level adjacency (Rule 3) is verified during BFS in
# loop_detection.py and cannot be reconstructed from post-hoc data.
# Non-param multi-pass groups may be the ONLY multi-pass group in a model
# (e.g., param-free loops like repeated addition), so we cannot require
# connection to other multi-pass layers.  The checks above (func identity,
# equiv type, pass numbering, symmetry, param sharing) are sufficient.


# ---------------------------------------------------------------------------
# O. Distance / reachability invariants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P. Graph connectivity invariants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Q. Module containment logical consistency
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# R. Lookup key bidirectionality
# ---------------------------------------------------------------------------


# Private implementation slices; public names remain owned by this module.
check_metadata_invariants = _rebind_function(_invariants_entry.check_metadata_invariants, globals())
_check_receptive_field_metadata_invariants = _rebind_function(
    _invariants_entry._check_receptive_field_metadata_invariants, globals()
)
_check_backend_neutral_module_mode_invariants = _rebind_function(
    _invariants_entry._check_backend_neutral_module_mode_invariants, globals()
)
_check_region_replay_provenance = _rebind_function(
    _invariants_entry._check_region_replay_provenance, globals()
)
_check_function_root_module_invariants = _rebind_function(
    _invariants_entry._check_function_root_module_invariants, globals()
)
_check_compute_op_module_attribution = _rebind_function(
    _invariants_entry._check_compute_op_module_attribution, globals()
)
_compute_ops = _rebind_function(_invariants_entry._compute_ops, globals())
_module_claims = _rebind_function(_invariants_entry._module_claims, globals())
_module_claim_address = _rebind_function(_invariants_entry._module_claim_address, globals())
_check_backend_identity_invariants = _rebind_function(
    _invariants_entry._check_backend_identity_invariants, globals()
)
_check_non_torch_backward_inert = _rebind_function(
    _invariants_entry._check_non_torch_backward_inert, globals()
)
_check_backend_neutral_accessor_refs = _rebind_function(
    _invariants_entry._check_backend_neutral_accessor_refs, globals()
)
_record_has_backend_neutral_accessor_metadata = _rebind_function(
    _invariants_entry._record_has_backend_neutral_accessor_metadata, globals()
)
check_func_call_id_invariant = _rebind_function(
    _invariants_entry.check_func_call_id_invariant, globals()
)
_check_backward_graph_invariants = _rebind_function(
    _invariants_backward_graph._check_backward_graph_invariants, globals()
)
_refuse_backward_evidence_without_registry = _rebind_function(
    _invariants_backward_graph._refuse_backward_evidence_without_registry, globals()
)
_check_backward_grad_fn_registry = _rebind_function(
    _invariants_backward_graph._check_backward_grad_fn_registry, globals()
)
_check_backward_grad_fn_handle_records = _rebind_function(
    _invariants_backward_graph._check_backward_grad_fn_handle_records, globals()
)
_check_backward_layer_backpointers = _rebind_function(
    _invariants_backward_graph._check_backward_layer_backpointers, globals()
)
_check_backward_saved_grad_records = _rebind_function(
    _invariants_backward_graph._check_backward_saved_grad_records, globals()
)
_check_backward_pass_record_consistency = _rebind_function(
    _invariants_backward_graph._check_backward_pass_record_consistency, globals()
)
_check_backward_pass_index_density = _rebind_function(
    _invariants_backward_graph._check_backward_pass_index_density, globals()
)
_check_grad_fn_topology_invariants = _rebind_function(
    _invariants_backward_graph._check_grad_fn_topology_invariants, globals()
)
_check_grad_fn_relation_list = _rebind_function(
    _invariants_backward_graph._check_grad_fn_relation_list, globals()
)
_check_backward_pass_domain_invariants = _rebind_function(
    _invariants_backward_domain._check_backward_pass_domain_invariants, globals()
)
_layer_feeds_recorded_backward_roots = _rebind_function(
    _invariants_backward_domain._layer_feeds_recorded_backward_roots, globals()
)
_layer_postdates_all_backward_triggers = _rebind_function(
    _invariants_backward_domain._layer_postdates_all_backward_triggers, globals()
)
_backward_trigger_forward_positions = _rebind_function(
    _invariants_backward_domain._backward_trigger_forward_positions, globals()
)
_backward_pass_root_forward_position = _rebind_function(
    _invariants_backward_domain._backward_pass_root_forward_position, globals()
)
_backward_pass_observed_forward_position = _rebind_function(
    _invariants_backward_domain._backward_pass_observed_forward_position, globals()
)
_resolve_op_grad_event_label = _rebind_function(
    _invariants_backward_domain._resolve_op_grad_event_label, globals()
)
_check_journal_seq_invariants = _rebind_function(
    _invariants_backward_domain._check_journal_seq_invariants, globals()
)
_check_backward_event_flow_invariants = _rebind_function(
    _invariants_backward_flow._check_backward_event_flow_invariants, globals()
)
_intervention_spec_is_armed = _rebind_function(
    _invariants_backward_flow._intervention_spec_is_armed, globals()
)
_persisted_replacement_stamp = _rebind_function(
    _invariants_backward_flow._persisted_replacement_stamp, globals()
)
op_has_genuine_replacement_evidence = _rebind_function(
    _invariants_backward_flow.op_has_genuine_replacement_evidence, globals()
)
_is_func_call_id_exempt = _rebind_function(
    _invariants_backward_flow._is_func_call_id_exempt, globals()
)
_plain_func_call_group_signature = _rebind_function(
    _invariants_backward_flow._plain_func_call_group_signature, globals()
)
_check_trace_self_consistency = _rebind_function(
    _invariants_topology._check_trace_self_consistency, globals()
)
_retained_orphan_computational_count = _rebind_function(
    _invariants_topology._retained_orphan_computational_count, globals()
)
_retained_orphan_op_labels = _rebind_function(
    _invariants_topology._retained_orphan_op_labels, globals()
)
_retained_orphan_layer_labels = _rebind_function(
    _invariants_topology._retained_orphan_layer_labels, globals()
)
_check_special_layer_lists = _rebind_function(
    _invariants_topology._check_special_layer_lists, globals()
)
_check_capture_edge_survival = _rebind_function(
    _invariants_topology._check_capture_edge_survival, globals()
)
_check_graph_topology = _rebind_function(_invariants_topology._check_graph_topology, globals())
_check_sibling_relation_derivation = _rebind_function(
    _invariants_topology._check_sibling_relation_derivation, globals()
)
_check_backend_neutral_graph_topology = _rebind_function(
    _invariants_payloads._check_backend_neutral_graph_topology, globals()
)
_check_edge_use_parent_arg_invariants = _rebind_function(
    _invariants_payloads._check_edge_use_parent_arg_invariants, globals()
)
_check_op_log_fields = _rebind_function(_invariants_payloads._check_op_log_fields, globals())
_check_primitive_op_invariants = _invariants_primitive_ops._check_primitive_op_invariants
_check_non_torch_primitive_op_inert = _invariants_primitive_ops._check_non_torch_primitive_op_inert
_check_payload_metadata_invariants = _rebind_function(
    _invariants_payloads._check_payload_metadata_invariants, globals()
)
_live_payload_value = _rebind_function(_invariants_payloads._live_payload_value, globals())
_check_live_payload_metadata = _rebind_function(
    _invariants_payloads._check_live_payload_metadata, globals()
)
_payload_shape = _rebind_function(_invariants_payloads._payload_shape, globals())
_payload_dtype = _rebind_function(_invariants_payloads._payload_dtype, globals())
_payload_memory = _rebind_function(_invariants_payloads._payload_memory, globals())
_dtype_values_match = _rebind_function(_invariants_conditional_base._dtype_values_match, globals())
_check_recurrence_invariants = _rebind_function(
    _invariants_conditional_base._check_recurrence_invariants, globals()
)
_check_branching_invariants = _rebind_function(
    _invariants_conditional_base._check_branching_invariants, globals()
)
_fail_conditional_invariant = _rebind_function(
    _invariants_conditional_base._fail_conditional_invariant, globals()
)
_strip_pass_suffix = _rebind_function(_invariants_conditional_base._strip_pass_suffix, globals())
_append_unique = _rebind_function(_invariants_conditional_base._append_unique, globals())
_is_prefix_stack = _rebind_function(_invariants_conditional_base._is_prefix_stack, globals())
_expected_layer_pass_child_views = _rebind_function(
    _invariants_conditional_base._expected_layer_pass_child_views, globals()
)
_expected_layer_log_child_views = _rebind_function(
    _invariants_conditional_base._expected_layer_log_child_views, globals()
)
_expected_layer_log_child_union = _rebind_function(
    _invariants_conditional_base._expected_layer_log_child_union, globals()
)
_valid_conditional_child_labels = _rebind_function(
    _invariants_conditional_base._valid_conditional_child_labels, globals()
)
_check_conditional_invariants = _rebind_function(
    _invariants_conditional_base._check_conditional_invariants, globals()
)
_check_conditional_arm_entry_child_symmetry = _rebind_function(
    _invariants_conditional_base._check_conditional_arm_entry_child_symmetry, globals()
)
_check_conditional_derived_child_views = _rebind_function(
    _invariants_conditional_base._check_conditional_derived_child_views, globals()
)
_check_conditional_child_labels_resolve = _rebind_function(
    _invariants_conditionals._check_conditional_child_labels_resolve, globals()
)
_check_conditional_bool_classification = _rebind_function(
    _invariants_conditionals._check_conditional_bool_classification, globals()
)
_check_conditional_event_references = _rebind_function(
    _invariants_conditionals._check_conditional_event_references, globals()
)
_check_conditional_branch_stack_monotonicity = _rebind_function(
    _invariants_conditionals._check_conditional_branch_stack_monotonicity, globals()
)
_check_conditional_elif_key_contiguity = _rebind_function(
    _invariants_conditionals._check_conditional_elif_key_contiguity, globals()
)
_check_conditional_bool_event_backrefs = _rebind_function(
    _invariants_conditionals._check_conditional_bool_event_backrefs, globals()
)
_check_conditional_layer_aggregate_views = _rebind_function(
    _invariants_conditionals._check_conditional_layer_aggregate_views, globals()
)
_check_conditional_rolled_edge_call_indices = _rebind_function(
    _invariants_conditionals._check_conditional_rolled_edge_call_indices, globals()
)
_check_conditional_transient_bool_keys_removed = _rebind_function(
    _invariants_conditionals._check_conditional_transient_bool_keys_removed, globals()
)
_check_conditional_arm_child_pass_union = _rebind_function(
    _invariants_conditionals._check_conditional_arm_child_pass_union, globals()
)
_check_conditional_branch_entry_edges = _rebind_function(
    _invariants_conditionals._check_conditional_branch_entry_edges, globals()
)
_check_conditional_arm_edges_match_graph = _rebind_function(
    _invariants_conditional_modules._check_conditional_arm_edges_match_graph, globals()
)
_check_conditional_branch_membership_records = _rebind_function(
    _invariants_conditional_modules._check_conditional_branch_membership_records, globals()
)
_check_conditional_public_accessor_summary = _rebind_function(
    _invariants_conditional_modules._check_conditional_public_accessor_summary, globals()
)
_check_layer_pass_to_layer_log_xrefs = _rebind_function(
    _invariants_conditional_modules._check_layer_pass_to_layer_log_xrefs, globals()
)
_check_pass_count_consistency = _rebind_function(
    _invariants_conditional_modules._check_pass_count_consistency, globals()
)
_check_module_layer_containment = _rebind_function(
    _invariants_conditional_modules._check_module_layer_containment, globals()
)
_check_module_hierarchy = _rebind_function(
    _invariants_conditional_modules._check_module_hierarchy, globals()
)
_check_module_call_boundary_and_tree = _rebind_function(
    _invariants_modules_params._check_module_call_boundary_and_tree, globals()
)
_check_module_call_tree_links = _rebind_function(
    _invariants_modules_params._check_module_call_tree_links, globals()
)
_check_module_call_output_structure_paths = _rebind_function(
    _invariants_modules_params._check_module_call_output_structure_paths, globals()
)
_container_tensor_leaf_paths = _rebind_function(
    _invariants_modules_params._container_tensor_leaf_paths, globals()
)
_container_tensor_components = _rebind_function(
    _invariants_modules_params._container_tensor_components, globals()
)
_check_param_xrefs = _rebind_function(_invariants_modules_params._check_param_xrefs, globals())
_check_param_usage_reciprocal_links = _rebind_function(
    _invariants_modules_params._check_param_usage_reciprocal_links, globals()
)
_check_param_co_parent_links = _rebind_function(
    _invariants_modules_params._check_param_co_parent_links, globals()
)
_check_layers_with_params_matches_param_usage = _rebind_function(
    _invariants_modules_params._check_layers_with_params_matches_param_usage, globals()
)
_deduped_layers_with_params = _rebind_function(
    _invariants_modules_params._deduped_layers_with_params, globals()
)
_check_layer_param_aggregate_dedup = _rebind_function(
    _invariants_modules_params._check_layer_param_aggregate_dedup, globals()
)
_param_log_list_contains_param = _rebind_function(
    _invariants_buffers._param_log_list_contains_param, globals()
)
_param_address_index = _rebind_function(_invariants_buffers._param_address_index, globals())
_check_buffer_xrefs = _rebind_function(_invariants_buffers._check_buffer_xrefs, globals())
_check_buffer_static_versions = _rebind_function(
    _invariants_buffers._check_buffer_static_versions, globals()
)
_buffer_address_has_module_ancestor = _rebind_function(
    _invariants_buffers._buffer_address_has_module_ancestor, globals()
)
_check_buffer_semantic_ownership = _rebind_function(
    _invariants_buffers._check_buffer_semantic_ownership, globals()
)
_owner_module_address_for_buffer = _rebind_function(
    _invariants_buffers._owner_module_address_for_buffer, globals()
)
_module_addresses_for_buffer_version = _rebind_function(
    _invariants_buffers._module_addresses_for_buffer_version, globals()
)
_active_buffer_consumer_module_addresses = _rebind_function(
    _invariants_buffers._active_buffer_consumer_module_addresses, globals()
)
_check_buffer_write_versions = _rebind_function(
    _invariants_buffers._check_buffer_write_versions, globals()
)
_resolve_trace_label = _rebind_function(_invariants_buffers._resolve_trace_label, globals())
_check_buffer_replay_validated_versions = _rebind_function(
    _invariants_buffers._check_buffer_replay_validated_versions, globals()
)
_canonical_equivalent_ops = _rebind_function(
    _invariants_equivalence._canonical_equivalent_ops, globals()
)
_check_equivalence_symmetry = _rebind_function(
    _invariants_equivalence._check_equivalence_symmetry, globals()
)
_check_graph_ordering = _rebind_function(_invariants_equivalence._check_graph_ordering, globals())
_check_loop_detection_invariants = _rebind_function(
    _invariants_equivalence._check_loop_detection_invariants, globals()
)
from ._invariants_sites import _check_site_key_invariants  # noqa: E402

_check_distance_invariants = _rebind_function(
    _invariants_connectivity._check_distance_invariants, globals()
)
_check_ancestry_closure = _rebind_function(
    _invariants_connectivity._check_ancestry_closure, globals()
)
_pass_qualified_label = _rebind_function(_invariants_connectivity._pass_qualified_label, globals())
_check_one_ancestry_record = _rebind_function(
    _invariants_connectivity._check_one_ancestry_record, globals()
)
_check_distance_closure = _rebind_function(
    _invariants_connectivity._check_distance_closure, globals()
)
_op_follows_recorded_backward_trigger = _rebind_function(
    _invariants_connectivity._op_follows_recorded_backward_trigger, globals()
)
_consumed_unattributed_data_operand = _rebind_function(
    _invariants_connectivity._consumed_unattributed_data_operand, globals()
)
_check_graph_connectivity = _rebind_function(
    _invariants_connectivity._check_graph_connectivity, globals()
)
_check_module_containment_logic = _rebind_function(
    _invariants_connectivity._check_module_containment_logic, globals()
)
_check_lookup_key_consistency = _rebind_function(
    _invariants_connectivity._check_lookup_key_consistency, globals()
)
_metadata_invariant_contracts_for_trace = _rebind_function(
    _invariants_connectivity._metadata_invariant_contracts_for_trace, globals()
)
_metadata_invariant_contracts_for_backend = _rebind_function(
    _invariants_connectivity._metadata_invariant_contracts_for_backend, globals()
)
_metadata_invariant_applies = _rebind_function(
    _invariants_connectivity._metadata_invariant_applies, globals()
)

_split_namespace = {name: value for name, value in globals().items() if not name.startswith("__")}
for _split_module in (
    _invariants_entry,
    _invariants_backward_graph,
    _invariants_backward_domain,
    _invariants_backward_flow,
    _invariants_topology,
    _invariants_payloads,
    _invariants_primitive_ops,
    _invariants_conditional_base,
    _invariants_conditionals,
    _invariants_conditional_modules,
    _invariants_modules_params,
    _invariants_buffers,
    _invariants_equivalence,
    _invariants_connectivity,
):
    _split_module.__dict__.update(_split_namespace)

METADATA_INVARIANT_CONTRACTS: tuple[MetadataInvariantContract, ...] = (
    # Setup checks.
    MetadataInvariantContract(
        "backend_identity_invariants",
        _check_backend_identity_invariants,
        "all",
    ),
    # --- Phase 1: structural invariants (A-L) ---
    MetadataInvariantContract("trace_self_consistency", _check_trace_self_consistency, "all"),
    MetadataInvariantContract(
        "region_replay_provenance",
        _check_region_replay_provenance,
        "all",
    ),
    MetadataInvariantContract(
        "backward_graph_invariants",
        _check_backward_graph_invariants,
        "torch",
    ),
    MetadataInvariantContract(
        "non_torch_backward_inert",
        _check_non_torch_backward_inert,
        "non_torch",
    ),
    MetadataInvariantContract(
        "non_torch_primitive_op_inert",
        _check_non_torch_primitive_op_inert,
        "non_torch",
    ),
    MetadataInvariantContract(
        "backend_neutral_accessor_refs",
        _check_backend_neutral_accessor_refs,
        "all",
    ),
    MetadataInvariantContract(
        "receptive_field_metadata",
        _check_receptive_field_metadata_invariants,
        "all",
    ),
    MetadataInvariantContract(
        "backend_neutral_module_mode_invariants",
        _check_backend_neutral_module_mode_invariants,
        "non_torch",
    ),
    MetadataInvariantContract("special_layer_lists", _check_special_layer_lists, "torch"),
    # Recurrence metadata controls whether Layer accessors may resolve a
    # single pass. Validate it before topology and other checks invoke those
    # accessors, so recurrence corruption is reported by its owning contract
    # rather than leaking a ValueError or producing a secondary topology red.
    MetadataInvariantContract(
        "loop_detection_invariants",
        _check_loop_detection_invariants,
        "torch",
    ),
    # Site-key tripwires I-S1/I-S2/I-S3' (L1 grouping core). All backends:
    # every producer mints keys; legacy keyless artifacts are out of the
    # declared domain (presence guard at the invariant's birth).
    MetadataInvariantContract(
        "site_key_invariants",
        _check_site_key_invariants,
        "all",
    ),
    MetadataInvariantContract("graph_topology", _check_graph_topology, "torch"),
    MetadataInvariantContract(
        "backend_neutral_graph_topology",
        _check_backend_neutral_graph_topology,
        "non_torch",
    ),
    MetadataInvariantContract(
        "edge_use_parent_arg_consistency",
        _check_edge_use_parent_arg_invariants,
        "torch",
    ),
    MetadataInvariantContract(
        "capture_edge_survival",
        _check_capture_edge_survival,
        "torch",
    ),
    MetadataInvariantContract("op_log_fields", _check_op_log_fields, "torch"),
    MetadataInvariantContract(
        "primitive_op_invariants",
        _check_primitive_op_invariants,
        "torch",
    ),
    MetadataInvariantContract(
        "payload_metadata_invariants",
        _check_payload_metadata_invariants,
        "torch",
    ),
    MetadataInvariantContract("recurrence_invariants", _check_recurrence_invariants, "torch"),
    MetadataInvariantContract("branching_invariants", _check_branching_invariants, "torch"),
    MetadataInvariantContract("conditional_invariants", _check_conditional_invariants, "torch"),
    MetadataInvariantContract(
        "layer_pass_layer_log_xrefs",
        _check_layer_pass_to_layer_log_xrefs,
        "torch",
    ),
    MetadataInvariantContract(
        "module_layer_containment",
        _check_module_layer_containment,
        "torch",
    ),
    MetadataInvariantContract("module_hierarchy", _check_module_hierarchy, "torch"),
    MetadataInvariantContract("param_xrefs", _check_param_xrefs, "torch"),
    MetadataInvariantContract("buffer_xrefs", _check_buffer_xrefs, "torch"),
    MetadataInvariantContract(
        "equivalence_symmetry",
        _check_equivalence_symmetry,
        "torch",
    ),
    # --- Phase 2: semantic invariants (M-R) ---
    MetadataInvariantContract("graph_ordering", _check_graph_ordering, "all"),
    MetadataInvariantContract(
        "pass_count_consistency",
        _check_pass_count_consistency,
        "torch",
    ),
    MetadataInvariantContract("distance_invariants", _check_distance_invariants, "torch"),
    MetadataInvariantContract("graph_connectivity", _check_graph_connectivity, "torch"),
    # AFTER graph_connectivity on purpose: a dropped op leaves its consumer with no
    # parents, which is connectivity's dangling-node finding to report. The closure
    # check would also fire on it (the consumer's ancestry no longer matches its edges),
    # and an invariant must not steal the owning contract's finding.
    MetadataInvariantContract("ancestry_closure", _check_ancestry_closure, "torch"),
    MetadataInvariantContract(
        "module_containment_logic",
        _check_module_containment_logic,
        "torch",
    ),
    MetadataInvariantContract(
        "lookup_key_consistency",
        _check_lookup_key_consistency,
        "all",
    ),
    MetadataInvariantContract(
        "func_call_id_consistency",
        check_func_call_id_invariant,
        "torch",
    ),
)
