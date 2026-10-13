"""Metadata invariants for the DROP-gated primitive-operation profile."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from ..constants import PRIMITIVE_OP_FIELD_ORDER
from ..data_classes.aten_op import AtenOp, OpRef, _ModePausedInteriorGap

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .invariants import MetadataInvariantError

_CAPTURE_PHASES = frozenset({"forward", "backward", "setup"})
_OWNER_STATUSES = frozenset({"forward_op", "backward_grad_fn_call", "orphan", "unresolved"})
_MUTATION_KINDS = frozenset({"none", "in_place", "out_variant", "metadata_only", "unknown"})
_VIEW_COPY_KINDS = frozenset({"view", "copy", "alias", "unknown"})
_GRAD_LINK_STATUSES = frozenset({"linked", "unlinked", "conflict", "not_applicable"})
_GRAD_LINK_PROVENANCE = frozenset({"exact_via_aten", "heuristic"})
_FLOP_STATUSES = frozenset({"formula_exact", "estimated", "unsupported"})
_OUTCOMES = frozenset({"returned", "raised"})


def _primitive_failure(message: str) -> None:
    """Raise the owning primitive-profile invariant error.

    Parameters
    ----------
    message
        Human-readable invariant failure.

    Raises
    ------
    MetadataInvariantError
        Always.
    """

    from .invariants import MetadataInvariantError

    raise MetadataInvariantError("primitive_op_invariants", message)


def _check_op_ref(ref: Any, ops: list[Any], *, row_label: str) -> None:
    """Validate one redundant Op foreign key against the dense Op table.

    Parameters
    ----------
    ref
        Candidate ``OpRef``.
    ops
        Dense final Op sequence.
    row_label
        Primitive label used in diagnostics.
    """

    if not isinstance(ref, OpRef):
        _primitive_failure(f"{row_label} parent reference is not an OpRef")
    if ref.op_row_index < 0 or ref.op_row_index >= len(ops):
        _primitive_failure(
            f"{row_label} parent op_row_index {ref.op_row_index} is outside 0..{len(ops) - 1}"
        )
    op = ops[ref.op_row_index]
    actual_label = getattr(op, "layer_label", None)
    actual_func_call_id = getattr(op, "func_call_id", None)
    if actual_label != ref.op_label or actual_func_call_id != ref.func_call_id:
        _primitive_failure(
            f"{row_label} parent OpRef witnesses disagree with dense row {ref.op_row_index}"
        )
    if "/" in ref.op_label:
        _primitive_failure(f"{row_label} parent Op label contains reserved '/' character")


def _check_primitive_row(
    row: Any,
    ops: list[Any],
    owner_evidence: dict[int, int | None],
    grad_fn_labels: set[str],
) -> None:
    """Validate one primitive row's schema, vocabulary, and foreign keys.

    Parameters
    ----------
    row
        Candidate primitive row.
    ops
        Dense final Op sequence.
    owner_evidence
        Immutable sequence-to-observation owner ledger.
    grad_fn_labels
        Materialized GradFn labels available for FK resolution.
    """

    _check_row_schema(row)
    _check_row_grad_fn_linkage(row, grad_fn_labels)
    _check_row_metrics(row)
    if owner_evidence.get(row.sequence, object()) != row.owner_func_call_id:
        _primitive_failure(f"{row.label} disagrees with its observation-time owner evidence")
    _check_row_parent_links(row, ops)


def _check_row_schema(row: Any) -> None:
    """Row type, declared-field presence, label grammar, and vocabularies."""

    if not isinstance(row, AtenOp):
        _primitive_failure("primitive profile contains a non-AtenOp row")
    missing = [name for name in PRIMITIVE_OP_FIELD_ORDER if not hasattr(row, name)]
    if missing:
        _primitive_failure(f"{row!r} is missing declared fields {missing!r}")
    if not isinstance(row.sequence, int) or row.sequence < 1:
        _primitive_failure(f"{row.label!r} has invalid sequence {row.sequence!r}")
    if row.label != f"aten_{row.sequence}" or "/" in row.label:
        _primitive_failure(f"{row.label!r} violates the wave-0 opaque label grammar")
    if row.capture_phase not in _CAPTURE_PHASES:
        _primitive_failure(f"{row.label} has invalid capture_phase {row.capture_phase!r}")
    if row.owner_status not in _OWNER_STATUSES:
        _primitive_failure(f"{row.label} has invalid owner_status {row.owner_status!r}")
    if row.mutation_kind not in _MUTATION_KINDS:
        _primitive_failure(f"{row.label} has invalid mutation_kind {row.mutation_kind!r}")
    if row.view_copy_kind not in _VIEW_COPY_KINDS:
        _primitive_failure(f"{row.label} has invalid view_copy_kind {row.view_copy_kind!r}")


def _check_row_grad_fn_linkage(row: Any, grad_fn_labels: set[str]) -> None:
    """GradFn link status/provenance vocabulary and FK coherence."""

    if row.grad_fn_link_status not in _GRAD_LINK_STATUSES:
        _primitive_failure(
            f"{row.label} has invalid grad_fn_link_status {row.grad_fn_link_status!r}"
        )
    if (
        row.grad_fn_link_provenance is not None
        and row.grad_fn_link_provenance not in _GRAD_LINK_PROVENANCE
    ):
        _primitive_failure(
            f"{row.label} has invalid grad_fn_link_provenance {row.grad_fn_link_provenance!r}"
        )
    if row.grad_fn_link_status == "linked":
        if row.grad_fn_ref not in grad_fn_labels or row.grad_fn_link_provenance is None:
            _primitive_failure(f"{row.label} has a dangling or unproven GradFn FK")
    elif row.grad_fn_ref is not None or row.grad_fn_link_provenance is not None:
        _primitive_failure(f"{row.label} has GradFn evidence without linked status")
    if row.grad_fn_link_status == "conflict":
        _primitive_failure(f"{row.label} has conflicting GradFn linkage")


def _check_row_metrics(row: Any) -> None:
    """FLOP/outcome/slot vocabularies and phase-index coherence."""

    if row.flop_status not in _FLOP_STATUSES:
        _primitive_failure(f"{row.label} has invalid flop_status {row.flop_status!r}")
    if row.algorithmic_flops is not None and (
        not isinstance(row.algorithmic_flops, int) or row.algorithmic_flops < 0
    ):
        _primitive_failure(f"{row.label} has invalid algorithmic_flops")
    if row.outcome not in _OUTCOMES:
        _primitive_failure(f"{row.label} has invalid outcome {row.outcome!r}")
    if not isinstance(row.decomposition_slot, int) or row.decomposition_slot < 0:
        _primitive_failure(f"{row.label} has invalid decomposition_slot")
    if row.capture_phase == "forward" and (
        row.forward_pass_index is None or row.backward_epoch_index is not None
    ):
        _primitive_failure(f"{row.label} has incoherent forward phase indices")
    if row.capture_phase == "backward" and (
        row.forward_pass_index is not None or row.backward_epoch_index is None
    ):
        _primitive_failure(f"{row.label} has incoherent backward phase indices")


def _check_row_parent_links(row: Any, ops: list[Any]) -> None:
    """Parent Op FKs, wrapper-fire agreement, and ownership coherence."""

    refs = tuple(row.parent_op_refs)
    for ref in refs:
        _check_op_ref(ref, ops, row_label=row.label)
    ref_call_ids = {ref.func_call_id for ref in refs}
    if len(ref_call_ids) > 1:
        _primitive_failure(f"{row.label} parent refs name more than one wrapper fire")
    if refs and row.owner_func_call_id not in ref_call_ids:
        _primitive_failure(f"{row.label} owner_func_call_id disagrees with parent refs")
    if row.owner_status == "forward_op" and (row.capture_phase != "forward" or not refs):
        _primitive_failure(f"{row.label} claims forward ownership without a forward Op FK")
    if row.owner_status == "backward_grad_fn_call" and row.parent_grad_fn_call_ref is None:
        _primitive_failure(f"{row.label} claims backward ownership without a GradFnCall FK")
    if row.owner_status == "orphan" and (refs or row.parent_grad_fn_call_ref is not None):
        _primitive_failure(f"{row.label} is orphaned but retains a parent FK")


def _check_gap(gap: Any, ops: list[Any]) -> None:
    """Validate one lower-bound observer-gap disclosure.

    Parameters
    ----------
    gap
        Candidate gap row.
    ops
        Dense final Op sequence.
    """

    if not isinstance(gap, _ModePausedInteriorGap):
        _primitive_failure("mode_paused_interior contains a non-gap row")
    if gap.kind != "mode_paused_interior" or gap.reason != "strict_subclass_constructor":
        _primitive_failure("mode_paused_interior gap has invalid kind or reason")
    if gap.capture_phase not in _CAPTURE_PHASES:
        _primitive_failure("mode_paused_interior gap has invalid capture phase")
    if gap.sequence_before < 0 or gap.sequence_after != gap.sequence_before + 1:
        _primitive_failure("mode_paused_interior gap has invalid boundary sequence")
    for ref in gap.parent_op_refs:
        _check_op_ref(ref, ops, row_label="mode_paused_interior")
    if gap.parent_op_refs and gap.owner_func_call_id not in {
        ref.func_call_id for ref in gap.parent_op_refs
    }:
        _primitive_failure("mode_paused_interior parent refs disagree with owner")


def _check_profile_rows(trace: Trace, profile: Any) -> list[Any]:
    """Shared evidence/row/partition/gap checks over one primitive profile.

    Returns the row list so callers can run their own tail checks. Check
    order is part of the contract: evidence duplication, evidence coverage,
    per-row checks with uniqueness and partitioning, partition contiguity,
    then gap disclosures.
    """

    rows = list(profile.primitive_ops)
    evidence_rows = tuple(profile._event_owner_evidence)
    evidence = dict(evidence_rows)
    if len(evidence) != len(evidence_rows):
        _primitive_failure("event-owner evidence contains duplicate sequences")
    if set(evidence) != {row.sequence for row in rows}:
        _primitive_failure("event-owner evidence does not exactly cover primitive rows")
    ops = list(trace.ops)
    grad_fn_labels = {
        str(grad_fn.label) for grad_fn in (getattr(trace, "grad_fn_logs", {}) or {}).values()
    }
    labels: set[str] = set()
    sequences: set[int] = set()
    partitions: dict[tuple[str, object], list[int]] = defaultdict(list)
    for row in rows:
        _check_primitive_row(row, ops, evidence, grad_fn_labels)
        if row.label in labels or row.sequence in sequences:
            _primitive_failure("primitive labels and sequences must be unique")
        labels.add(row.label)
        sequences.add(row.sequence)
        if row.owner_status == "forward_op":
            partitions[("forward", row.owner_func_call_id)].append(row.decomposition_slot)
        elif row.owner_status == "backward_grad_fn_call":
            partitions[("backward", row.parent_grad_fn_call_ref)].append(row.decomposition_slot)
    for key, slots in partitions.items():
        if sorted(slots) != list(range(len(slots))):
            _primitive_failure(f"primitive partition {key!r} has non-contiguous slots {slots!r}")
    for gap in profile.mode_paused_interior:
        _check_gap(gap, ops)
    return rows


def _check_unresolved_ownership(trace: Trace, rows: list[Any]) -> None:
    """A completed trace may not retain unresolved primitive ownership."""

    if getattr(trace, "_tracing_finished", False) and any(
        row.owner_status == "unresolved" for row in rows
    ):
        _primitive_failure("completed trace retains unresolved primitive ownership")


def _check_primitive_op_invariants(trace: Trace) -> None:
    """Validate primitive-profile rows, partitions, disclosures, and stores.

    Parameters
    ----------
    trace
        Postprocessed torch Trace.

    Raises
    ------
    MetadataInvariantError
        If any primitive-profile contract is inconsistent.
    """

    profile = getattr(trace, "_primitive_op_profile", None)
    core = trace.__dict__.get("_trace_core")
    forward_store = None if core is None else core.kind_rows.get("primitive_op")
    if profile is None:
        if forward_store is not None:
            _primitive_failure("recording-off trace retains a primitive_op store")
        return
    rows = _check_profile_rows(trace, profile)
    forward_rows = [row for row in rows if row.capture_phase == "forward"]
    if forward_rows:
        if forward_store is None or len(forward_store) != len(forward_rows):
            _primitive_failure("forward primitive store does not match profile rows")
    elif forward_store is not None and len(forward_store):
        _primitive_failure("present-empty forward profile has non-empty primitive store")
    _check_unresolved_ownership(trace, rows)


def _check_non_torch_primitive_op_inert(trace: Trace) -> None:
    """Prove non-torch traces expose no primitive profile or primitive store.

    Parameters
    ----------
    trace
        Postprocessed non-torch Trace.

    Raises
    ------
    MetadataInvariantError
        If primitive state appears on a non-torch backend.
    """

    name = "non_torch_primitive_op_inert"
    if getattr(trace, "_primitive_op_profile", None) is not None:
        raise MetadataInvariantError(name, "non-torch trace must not carry a primitive profile")
    core = trace.__dict__.get("_trace_core")
    if core is None:
        return
    if "primitive_op" in core.kind_rows:
        raise MetadataInvariantError(name, "non-torch trace must not carry a primitive_op store")
    if any("primitive_op" in epoch.stores for epoch in core.backward_epochs):
        raise MetadataInvariantError(
            name, "non-torch trace must not carry a backward primitive_op store"
        )


def validate_loaded_primitive_profile(trace: Trace) -> None:
    """Fail closed on a switched pre-release primitive profile before exposure.

    Parameters
    ----------
    trace
        Newly restored Trace whose primitive rows are still detached.

    Raises
    ------
    TorchLensIOError
        With ``primitive_op_schema_invalid`` or ``primitive_op_fk_invalid``.
    """

    from .._io import TorchLensIOError
    from .invariants import MetadataInvariantError

    profile = getattr(trace, "_primitive_op_profile", None)
    if profile is None:
        return
    try:
        rows = _check_profile_rows(trace, profile)
        _check_unresolved_ownership(trace, rows)
    except (AttributeError, KeyError, TypeError, ValueError, MetadataInvariantError) as exc:
        detail = str(exc)
        fk_markers = (
            "OpRef",
            "parent",
            "owner evidence",
            "owner_func_call_id",
            "partition",
            "GradFnCall FK",
        )
        code = (
            "primitive_op_fk_invalid"
            if any(marker in detail for marker in fk_markers)
            else "primitive_op_schema_invalid"
        )
        raise TorchLensIOError(
            f"Primitive-operation profile failed pre-release validation: {detail}",
            code=code,
            remedy="re-capture and re-save the trace; do not edit primitive profile metadata",
        ) from exc


__all__ = [
    "_check_non_torch_primitive_op_inert",
    "_check_primitive_op_invariants",
    "validate_loaded_primitive_profile",
]
