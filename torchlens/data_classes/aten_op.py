"""Primitive-operation records for the ATen profile (persisted, tlspec v8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from .._io import TLSPEC_VERSION, FieldPolicy, read_tlspec_version
from .._trace_core.record_rows import install_record_facade as _install_record_facade
from ..constants import PRIMITIVE_OP_FIELD_ORDER
from .field_policy import build_record_field_policy_table, portable_state_spec_from_policy


@dataclass(frozen=True, slots=True, kw_only=True)
class OpRef:
    """Redundant foreign key from an ATen row to a final Op row."""

    op_row_index: int
    op_label: str
    func_call_id: int

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "op_row_index": FieldPolicy.KEEP,
        "op_label": FieldPolicy.KEEP,
        "func_call_id": FieldPolicy.KEEP,
    }


@dataclass(frozen=True, slots=True)
class _ModePausedInteriorGap:
    """Typed lower-bound disclosure for a paused dispatcher interior."""

    kind: str
    capture_phase: str
    sequence_before: int
    sequence_after: int
    owner_func_call_id: int | None
    parent_op_refs: tuple[OpRef, ...]
    reason: str

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "kind": FieldPolicy.KEEP,
        "capture_phase": FieldPolicy.KEEP,
        "sequence_before": FieldPolicy.KEEP,
        "sequence_after": FieldPolicy.KEEP,
        "owner_func_call_id": FieldPolicy.KEEP,
        "parent_op_refs": FieldPolicy.KEEP,
        "reason": FieldPolicy.KEEP,
    }


@dataclass
class _PrimitiveOpProfile:
    """Trace section carrying primitive rows and disclosure gaps."""

    primitive_ops: list[AtenOp] = field(default_factory=list)
    mode_paused_interior: list[_ModePausedInteriorGap] = field(default_factory=list)
    aten_event_watermark: int = 0
    _event_owner_evidence: tuple[tuple[int, int | None], ...] = ()

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "primitive_ops": FieldPolicy.KEEP,
        "mode_paused_interior": FieldPolicy.KEEP,
        "aten_event_watermark": FieldPolicy.KEEP,
        "_event_owner_evidence": FieldPolicy.KEEP,
    }


@dataclass(eq=False, kw_only=True)
class AtenOp:
    """One value-free ATen dispatcher call observed during capture."""

    label: str
    sequence: int
    capture_phase: str
    forward_pass_index: int | None = None
    backward_epoch_index: int | None = None
    owner_func_call_id: int | None = None
    parent_op_refs: tuple[OpRef, ...] = ()
    parent_grad_fn_call_ref: tuple[str, int, int] | None = None
    owner_status: str = "unresolved"
    decomposition_slot: int = 0
    namespace: str = "aten"
    operator: str = "unknown"
    overload: str = "default"
    schema: str | None = None
    schema_fingerprint: str | None = None
    module_call_stack: tuple[tuple[str, int], ...] = ()
    input_tensor_facts: tuple[Any, ...] = ()
    output_tensor_facts: tuple[Any, ...] = ()
    mutation_kind: str = "unknown"
    view_copy_kind: str = "unknown"
    autocast_context: tuple[tuple[str, bool, str], ...] = ()
    dispatch_key_context: str | None = None
    grad_fn_ref: str | None = None
    grad_fn_link_status: str = "not_applicable"
    grad_fn_link_provenance: str | None = None
    algorithmic_flops: int | None = None
    flop_status: str = "unsupported"
    flop_formula_source: str | None = None
    flop_formula_version: str | None = None
    outcome: str = "returned"
    exception_type: str | None = None
    execution_context: Any = None

    _PORTABLE_STATE_POLICY: ClassVar[dict[str, FieldPolicy]] = dict.fromkeys(
        PRIMITIVE_OP_FIELD_ORDER, FieldPolicy.KEEP
    )
    FIELD_POLICY = build_record_field_policy_table(
        PRIMITIVE_OP_FIELD_ORDER,
        _PORTABLE_STATE_POLICY,
        schema_key="primitive_op",
    )
    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = portable_state_spec_from_policy(
        FIELD_POLICY
    )

    def __tl_state_items__(self) -> Any:
        """Yield live state pairs from the backing primitive row."""

        from .._trace_core.record_rows import record_state_items

        return record_state_items(self)

    def __tl_state_restore__(self, mapping: dict[str, Any]) -> None:
        """Restore primitive state through the row-cell descriptors.

        Parameters
        ----------
        mapping
            Scrubbed primitive row state.
        """

        from .._trace_core.record_rows import record_state_restore

        record_state_restore(self, mapping)

    def __getstate__(self) -> dict[str, Any]:
        """Return primitive row state without its backing-store handle."""

        from ._state_adapter import state_items

        state = dict(state_items(self))
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a primitive row through its detached facade store.

        Parameters
        ----------
        state
            Serialized primitive row state.
        """

        read_tlspec_version(state, cls_name=type(self).__name__, cls=type(self))
        self.__tl_state_restore__(state)


_PRIMITIVE_OP_STORE_LAYOUT = _install_record_facade(AtenOp, tuple(PRIMITIVE_OP_FIELD_ORDER))


# The tlspec v8 coordinated bump retired the primitive-profile S3 pre-release
# registrations: every profile record class above declares FieldPolicy.KEEP
# directly, and loaded profiles validate through
# validate_loaded_primitive_profile whenever present.


__all__ = ["AtenOp", "OpRef", "PRIMITIVE_OP_FIELD_ORDER"]
