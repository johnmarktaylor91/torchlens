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
  J. Param cross-references (Param -> layer, has_params flag)
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .._errors import _ActionableErrorMixin
from ..errors._base import ValidationError

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

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

    def __init__(self, check_name: str, message: str, *, code: str | None = None) -> None:
        """Initialize a metadata invariant failure.

        Parameters
        ----------
        check_name:
            Invariant check group name.
        message:
            Human-readable failure detail.
        code:
            Optional stable refusal code from the error-refusal contract,
            carried on ``fields["code"]``.
        """

        text = f"[{check_name}] {message}"
        if code is None:
            super().__init__(text)
        else:
            super().__init__(text, code=code)
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


# Late imports (after this module's own definitions): the split modules import this
# module's classes and constants back, so they must exist before the splits load.
from ._invariants_backward_graph import (  # noqa: E402
    _check_backward_graph_invariants,
)
from ._invariants_buffers import (  # noqa: E402
    _check_buffer_xrefs,
)
from ._invariants_conditional_base import (  # noqa: E402
    _check_branching_invariants,
    _check_conditional_invariants,
    _check_recurrence_invariants,
)
from ._invariants_conditional_modules import (  # noqa: E402
    _check_layer_pass_to_layer_log_xrefs,
    _check_module_hierarchy,
    _check_module_layer_containment,
    _check_pass_count_consistency,
)
from ._invariants_connectivity import (  # noqa: E402
    _check_ancestry_closure,
    _check_distance_invariants,
    _check_graph_connectivity,
    _check_lookup_key_consistency,
    _check_module_containment_logic,
)
from ._invariants_entry import (  # noqa: E402
    _check_backend_identity_invariants,
    _check_backend_neutral_accessor_refs,
    _check_backend_neutral_module_mode_invariants,
    _check_non_torch_backward_inert,
    _check_receptive_field_metadata_invariants,
    _check_region_replay_provenance,
    check_func_call_id_invariant,
    check_metadata_invariants as check_metadata_invariants,
)
from ._invariants_equivalence import (  # noqa: E402
    _check_equivalence_symmetry,
    _check_graph_ordering,
    _check_loop_detection_invariants,
)
from ._invariants_modules_params import (  # noqa: E402
    _check_param_xrefs,
)
from ._invariants_payloads import (  # noqa: E402
    _check_backend_neutral_graph_topology,
    _check_edge_use_parent_arg_invariants,
    _check_op_log_fields,
    _check_payload_metadata_invariants,
)
from ._invariants_primitive_ops import (  # noqa: E402
    _check_non_torch_primitive_op_inert,
    _check_primitive_op_invariants,
)
from ._invariants_sites import _check_site_key_invariants  # noqa: E402
from ._invariants_topology import (  # noqa: E402
    _check_capture_edge_survival,
    _check_graph_topology,
    _check_special_layer_lists,
    _check_trace_self_consistency,
)

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
