"""Primitive-operation records for the ATen profile (persisted, tlspec v8)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from .._constants import PRIMITIVE_OP_FIELD_ORDER
from .._io import TLSPEC_VERSION, FieldPolicy, read_tlspec_version
from .._trace_core.record_rows import install_record_facade as _install_record_facade
from .field_policy import build_record_field_policy_table, portable_state_spec_from_policy

if TYPE_CHECKING:
    from .._kernel_telemetry import KernelLaunch

#: The detachable kernel-telemetry lane. ``gpu_kernels`` resolves it by name on
#: read, so the class always carries the view while neither this layer nor the
#: package imports the lane (``torchlens/_kernel_telemetry.py``).
_KERNEL_TELEMETRY_MODULE = "torchlens._kernel_telemetry"


@dataclass(frozen=True, slots=True, kw_only=True)
class OpRef:
    """Redundant foreign key from an ATen row to a final Op row.

    Examples
    --------
    >>> trace = tl.trace(model, x, capture=tl.options.CaptureOptions(aten_ops=True))
    >>> ref = trace["relu_1_2"].aten_ops[0].parent_op_refs[0]
    >>> type(ref).__name__, ref.op_label
    ('OpRef', 'relu_1_2')
    """

    op_row_index: int
    op_label: str
    func_call_id: int

    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
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

    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
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

    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "primitive_ops": FieldPolicy.KEEP,
        "mode_paused_interior": FieldPolicy.KEEP,
        "aten_event_watermark": FieldPolicy.KEEP,
        "_event_owner_evidence": FieldPolicy.KEEP,
    }


@dataclass(eq=False, kw_only=True)
class AtenOp:
    """One ATen dispatcher call observed during capture.

    Rows are value-free unless the capture asked for values
    (``CaptureOptions(aten_ops="values")``), in which case ``out`` holds the
    call's returned tensor(s) under the same save rules as ``Op.out``.

    Examples
    --------
    >>> trace = tl.trace(model, x, capture=tl.options.CaptureOptions(aten_ops=True))
    >>> row = trace["relu_1_2"].aten_ops[0]
    >>> type(row).__name__, row.capture_phase
    ('AtenOp', 'forward')
    """

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
    out: Any = None
    """The tensor(s) this call returned, mirroring ``Op.out``: a tensor for a
    single-tensor result, a tuple of tensors (one per ``output_tensor_facts``
    entry, same order) for a container result, ``None`` when no value was kept
    (values off, a raised call, a non-tensor result, an owner whose ``save=``
    selection dropped it, or an artifact saved without outs)."""

    _PORTABLE_STATE_POLICY: ClassVar[dict[str, FieldPolicy]] = {
        **dict.fromkeys(PRIMITIVE_OP_FIELD_ORDER, FieldPolicy.KEEP),
        "out": FieldPolicy.BLOB_RECURSIVE,
    }
    _FIELD_POLICY = build_record_field_policy_table(
        PRIMITIVE_OP_FIELD_ORDER,
        _PORTABLE_STATE_POLICY,
        schema_key="primitive_op",
    )
    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = portable_state_spec_from_policy(
        _FIELD_POLICY
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
        # ``include_outs=False`` omits the out column; the loader's true-up sets
        # it to None after this hook, so an absent cell is read as None here.
        out = getattr(self, "out", None)
        if isinstance(out, list):
            # A container result round-trips through JSON as a list; the
            # user-facing shape is the capture-time tuple.
            self.out = tuple(out)

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

    @property
    def gpu_kernels(self) -> tuple[KernelLaunch, ...]:
        """The GPU kernel launches this ATen call made, matched by the profiler's correlation id.

        Each ``KernelLaunch`` row is a kernel the profiler saw launched while this
        ATen call ran on the device. When kernel telemetry was off for the trace,
        the value is one row with ``attribution_status="unavailable"``, never an
        empty tuple. Reading it needs the optional kernel-telemetry adapter.

        Returns
        -------
        tuple[KernelLaunch, ...]
            The matched kernel launches, or one ``unavailable`` row.
        """

        telemetry = importlib.import_module(_KERNEL_TELEMETRY_MODULE)
        return telemetry._aten_gpu_kernels(self)


_PRIMITIVE_OP_STORE_LAYOUT = _install_record_facade(AtenOp, tuple(PRIMITIVE_OP_FIELD_ORDER))


# The tlspec v8 coordinated bump retired the primitive-profile S3 pre-release
# registrations: every profile record class above declares FieldPolicy.KEEP
# directly, and loaded profiles validate through
# validate_loaded_primitive_profile whenever present.


__all__ = ["AtenOp", "OpRef", "PRIMITIVE_OP_FIELD_ORDER"]
