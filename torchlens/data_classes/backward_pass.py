"""Per-invocation backward pass records and accessors."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from .._io import (
    TLSPEC_VERSION,
    FieldPolicy,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from ..constants import BACKWARD_PASS_FIELD_ORDER
from ..quantities import Duration
from ._accessor_base import Accessor
from .field_policy import build_record_field_policy_table, portable_state_spec_from_policy

if TYPE_CHECKING:
    import pandas as pd

# Typed container defaults for every non-Optional container field
# BackwardPass declares. Same defect class as
# `Op._LAYER_PASS_LOG_CONTAINER_DEFAULTS`/`Trace._MODEL_LOG_CONTAINER_DEFAULTS`:
# without this, `coerce_container_typed_state` cannot repair a
# present-but-wrong-typed legacy value, and an absent field crashes instead of
# restoring an empty typed container. Plain builtin types are used
# deliberately.
_BACKWARD_PASS_CONTAINER_DEFAULTS: dict[str, Any] = {
    "root_grad_fn_ids": [],
    "root_meta": (),
    "inputs_subset": (),
    "grad_fn_calls": [],
}


# BackwardPass fields deliberately omitted from ``BackwardPass.to_pandas()``
# columns. Every field in ``BACKWARD_PASS_FIELD_ORDER`` must either appear as
# a dataframe column or be listed here -- ``tests/test_to_pandas_field_coverage.py``
# enforces this so new BackwardPass fields can never silently fail to reach
# the table again (same regression class as TO-PANDAS-NEW-FIELDS).
_TO_PANDAS_EXCLUDED_BACKWARD_PASS_FIELDS: frozenset[str] = frozenset(
    {
        # List of live GradFnCall child records, not a scalar table cell --
        # summarized instead as the derived "num_grad_fn_calls" column (same
        # "list of child records" exclusion category as Buffer.versions /
        # GradFn.calls / Module.ops).
        "grad_fn_calls",
        # Nested per-pass count and attribution record (closed dict), read
        # through ``flops_attribution`` and ``tl.report.backward_status``.
        "executed_flops_record",
    }
)


@dataclass
class BackwardPass:
    """Projected metadata for one backward engine invocation.

    Parameters
    ----------
    pass_index:
        One-based global backward invocation number.
    trigger:
        Entry point that opened this backward pass.
    implicit:
        Whether this pass was inferred from orphan tensor-hook emissions.
    outer_context:
        Optional outer TorchLens trigger context when this pass was nested.
    executed_flops_record:
        The pass's counted executed backward and its attribution as a small
        closed record, written when the trace is saved so a loaded trace
        keeps the counted answer; ``None`` on a live pass (read the census
        through ``flops_attribution``) and on artifacts saved without it.
    """

    _PORTABLE_STATE_POLICY: ClassVar[dict[str, FieldPolicy]] = {
        "pass_index": FieldPolicy.KEEP,
        "trigger": FieldPolicy.KEEP,
        "implicit": FieldPolicy.KEEP,
        "outer_context": FieldPolicy.KEEP,
        "backward_call_context": FieldPolicy.KEEP,
        "root_grad_fn_ids": FieldPolicy.KEEP,
        "root_meta": FieldPolicy.KEEP,
        "root_grad_arguments": FieldPolicy.KEEP,
        "inputs_subset": FieldPolicy.KEEP,
        "order": FieldPolicy.KEEP,
        "origin_backward_pass": FieldPolicy.KEEP,
        "engine_flags": FieldPolicy.KEEP,
        "save_grads_policy": FieldPolicy.KEEP,
        "duration": FieldPolicy.KEEP,
        "peak_memory": FieldPolicy.KEEP,
        "status": FieldPolicy.KEEP,
        "order_attribution_coverage": FieldPolicy.KEEP,
        "grad_fn_calls": FieldPolicy.KEEP,
        "executed_flops_record": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
    }
    FIELD_POLICY = build_record_field_policy_table(
        BACKWARD_PASS_FIELD_ORDER,
        _PORTABLE_STATE_POLICY,
        schema_key="backward_pass",
    )
    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = portable_state_spec_from_policy(
        FIELD_POLICY
    )

    pass_index: int
    trigger: str
    implicit: bool
    outer_context: str | None = None
    backward_call_context: object | None = None
    root_grad_fn_ids: list[int] = field(default_factory=list)
    root_meta: tuple[object, ...] = field(default_factory=tuple)
    root_grad_arguments: object | None = None
    inputs_subset: tuple[object, ...] = field(default_factory=tuple)
    order: int | None = None
    origin_backward_pass: int | None = None
    engine_flags: dict[str, object] | None = None
    save_grads_policy: str | None = None
    duration: Duration | None = None
    peak_memory: int | None = None
    status: str = "ok"
    order_attribution_coverage: float | None = None
    grad_fn_calls: list[Any] = field(default_factory=list)
    executed_flops_record: dict[str, Any] | None = None
    _source_trace_ref: Any = None

    def __tl_state_items__(self) -> Any:
        """Yield live state pairs from the backing row (M9 facade hook).

        A live pass with a recorded census contributes its counted executed
        backward as ``executed_flops_record``, so every saved copy carries it.
        """

        from .._trace_core.record_rows import record_state_items

        live = self._census_record()
        seen = False
        for name, value in record_state_items(self):
            if name == "executed_flops_record":
                seen = True
                if value is None:
                    value = live
            yield name, value
        if not seen and live is not None:
            yield "executed_flops_record", live

    def _census_record(self) -> dict[str, Any] | None:
        """The persisted form of this pass's census attribution, if it has one."""

        trace = self.source_trace
        ops = self.dispatched_ops
        if trace is None or ops is None:
            return None
        from ..report._backward_attribution import attribute_census, executed_flops_record

        return executed_flops_record(attribute_census(trace, self, ops))

    def __tl_state_restore__(self, mapping: dict[str, Any]) -> None:
        """Install a state mapping through the cell descriptors (M9 hook)."""

        from .._trace_core.record_rows import record_state_restore

        record_state_restore(self, mapping)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with an IO format marker."""

        from ._state_adapter import state_items

        state = dict(state_items(self))
        state["_source_trace_ref"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state and fill fields added in newer versions."""

        read_tlspec_version(state, cls_name=type(self).__name__, cls=type(self))
        backward_pass_setstate_defaults: dict[str, Any] = {
            **_BACKWARD_PASS_CONTAINER_DEFAULTS,
            "outer_context": None,
            "backward_call_context": None,
            "root_grad_arguments": None,
            "order": None,
            "origin_backward_pass": None,
            "engine_flags": None,
            "save_grads_policy": None,
            "duration": None,
            "peak_memory": None,
            "status": "ok",
            "order_attribution_coverage": None,
            "executed_flops_record": None,
            "_source_trace_ref": None,
        }
        default_fill_state(state, defaults=backward_pass_setstate_defaults)
        # Repair present-but-wrong-typed container fields from legacy states.
        # `default_fill_state` only fills absent keys; this closes the same
        # gap `Trace`/`Op` already close for their own fields.
        coerce_container_typed_state(state, backward_pass_setstate_defaults)
        state.pop("call_context", None)
        from .._io.state_keys import refuse_callable_shadowing_state_keys

        refuse_callable_shadowing_state_keys(type(self), state)
        from .._trace_core.record_rows import record_state_restore

        record_state_restore(self, state)

    @property
    def source_trace(self) -> Any:
        """Return the owning Trace if it is still alive."""

        ref = self._source_trace_ref
        return None if ref is None else ref()

    @source_trace.setter
    def source_trace(self, value: Any) -> None:
        """Set the owning Trace weakref."""

        self._source_trace_ref = weakref.ref(value) if value is not None else None

    @property
    def trace(self) -> Any:
        """Alias for the owning Trace."""

        return self.source_trace

    @property
    def dispatched_ops(self) -> tuple[Any, ...] | None:
        """The ATen calls this backward pass dispatched (DOCUMENTED-UNSTABLE).

        One value-free ``DispatchedOp`` per call (operator, shapes, dtypes,
        the GradFn call it ran under, its counted FLOPs), recorded while the
        pass ran through a capture that requested a backward. Session-time:
        ``None`` when this object holds no record for the pass (a loaded
        trace, an implicit pass opened by tensor hooks, another backend).
        """
        trace = self.source_trace
        if trace is None:
            return None
        from ..backends.torch._backward_dispatch import recorded_ops

        return recorded_ops(trace, self.pass_index)

    @property
    def flops_attribution(self) -> Any:
        """This pass's executed backward FLOPs by GradFn call and forward op (DOCUMENTED-UNSTABLE).

        A ``BackwardFlopsAttribution``: one row per GradFn call (its forward
        op, counted FLOPs, recompute share) plus an ``unattributed`` bucket
        naming every call that ran under no recorded node; the rows plus the
        bucket equal the pass's executed count exactly. ``None`` when the
        pass has no recorded census on this object and was not saved with
        its count (a loaded trace reads the saved record).
        """
        trace = self.source_trace
        ops = self.dispatched_ops
        from ..report._backward_attribution import attribute_census, attribution_from_record

        if trace is not None and ops is not None:
            return attribute_census(trace, self, ops)
        record = self.executed_flops_record
        if record is None:
            return None
        return attribution_from_record(trace, self.pass_index, record)

    @property
    def ordinal_index(self) -> int:
        """Return this pass's 0-based position in ``trace.backward_passes``."""

        return self.pass_index - 1

    def to_pandas(self) -> pd.DataFrame:
        """Export this backward pass as a one-row DataFrame.

        Driven by ``BACKWARD_PASS_FIELD_ORDER``: every field is exported
        except ``grad_fn_calls`` (a list of live ``GradFnCall`` child records,
        not a scalar table cell -- summarized instead as ``num_grad_fn_calls``,
        the same "list of child records" exclusion category as
        ``Buffer.versions``/``GradFn.calls``/``Module.ops``). This used to
        hand-roll a 12-field subset that silently dropped
        ``root_grad_fn_ids``/``root_meta``/``root_grad_arguments``/
        ``inputs_subset``/``engine_flags``/``save_grads_policy``.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        row = {
            field_name: getattr(self, field_name)
            for field_name in BACKWARD_PASS_FIELD_ORDER
            if field_name not in _TO_PANDAS_EXCLUDED_BACKWARD_PASS_FIELDS
        }
        row["num_grad_fn_calls"] = len(self.grad_fn_calls)
        return pd.DataFrame([row], columns=list(row))


# The M9 facade: the declared stored fields become row-cell descriptors
# (dataclass defaults are baked into the generated __init__). Rows adopt
# into the owning backward epoch's stores.
_BACKWARD_PASS_STORED_FIELDS: tuple[str, ...] = (
    "pass_index",
    "trigger",
    "implicit",
    "outer_context",
    "backward_call_context",
    "root_grad_fn_ids",
    "root_meta",
    "root_grad_arguments",
    "inputs_subset",
    "order",
    "origin_backward_pass",
    "engine_flags",
    "save_grads_policy",
    "duration",
    "peak_memory",
    "status",
    "order_attribution_coverage",
    "grad_fn_calls",
    "executed_flops_record",
    "_source_trace_ref",
)


def _install_backward_pass_facade() -> None:
    """Install the BackwardPass row-cell descriptors (import-time)."""

    from .._trace_core.record_rows import install_record_facade

    install_record_facade(BackwardPass, _BACKWARD_PASS_STORED_FIELDS)


_install_backward_pass_facade()


class BackwardPassAccessor(Accessor[BackwardPass]):
    """Dict-like accessor for projected backward pass records."""

    def __init__(self, backward_passes: dict[int, BackwardPass]) -> None:
        """Initialize the accessor from pass-index keyed records."""

        items = [backward_passes[index] for index in sorted(backward_passes)]
        super().__init__(
            {str(pass_record.pass_index): pass_record for pass_record in items},
            item_list=items,
        )

    def for_pass(self, pass_index: int) -> BackwardPass:
        """Return a pass by its one-based global pass number."""

        key = str(pass_index)
        if key not in self._dict:
            available = [pass_record.pass_index for pass_record in self._list]
            raise KeyError(f"Backward pass {pass_index} not found. Available passes: {available}.")
        return self._dict[key]
