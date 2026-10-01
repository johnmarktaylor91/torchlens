"""Per-pass runtime data for autograd grad_fn_handle nodes."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import pandas as pd

from .._io import (
    TLSPEC_VERSION,
    FieldPolicy,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from ..constants import GRAD_FN_PASS_LOG_FIELD_ORDER
from ..quantities import Duration
from .field_policy import build_record_field_policy_table, portable_state_spec_from_policy

# GradFnCall currently declares no plain-container (list/dict/tuple/set)
# fields -- every field is scalar, `Any`-typed blob, or a weakref. This empty
# mapping documents that and keeps `GradFnCall.__setstate__` calling
# `coerce_container_typed_state` for consistency with every sibling record
# class (`Op`/`Layer`/`Param`/`GradFn`/...), so a future container field added
# to this dataclass is automatically covered instead of silently missed.
_GRAD_FN_CALL_CONTAINER_DEFAULTS: dict[str, Any] = {}

_ORDINAL_POSITIONS_CACHE: weakref.WeakKeyDictionary[Any, tuple[Any, dict[int, int]]] = (
    weakref.WeakKeyDictionary()
)
"""Per-trace ``id(call) -> position`` map for :attr:`GradFnCall.ordinal_index`.

Weak-keyed on the owning Trace and validated against the trace's backward
projection revision, so a full ordinal sweep over N calls is O(N) instead of
the historical O(N^2) (``list(trace.grad_fn_calls).index(self)`` rebuilt the
accessor AND ran an equality scan per access; measured exponent 2.02).
Identity keys are sound because GradFnCall facades are identity-stable while
referenced (the M9 weak-valued facade cache), and a stale/missing id triggers
one rebuild rather than a wrong answer.
"""


def _grad_fn_payload_equal(left: Any, right: Any, _depth: int = 0) -> bool:
    """Tensor-safe structural equality for saved gradient payloads.

    ``torch.Tensor.__eq__`` is elementwise, so the dataclass-generated
    ``GradFnCall.__eq__`` raised an untyped torch ``RuntimeError`` ("Boolean
    value of Tensor ... is ambiguous") whenever two like-labeled calls both
    carried saved multi-element gradients. Tensors compare by exact value via
    ``torch.equal`` (shape/dtype mismatches are plain ``False``), containers
    recurse, and everything else uses ordinary ``==``.
    """

    if left is right:
        return True
    if _depth > 50:
        # Saved gradient payloads are shallow (tuples/dicts of tensors); an
        # over-deep or self-referential payload compares unequal rather than
        # recursing without bound.
        return False
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)):
            return False
        try:
            return bool(torch.equal(left, right))
        except (RuntimeError, TypeError, ValueError):
            return False
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(
            _grad_fn_payload_equal(a, b, _depth + 1) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(
            _grad_fn_payload_equal(item, right[key], _depth + 1) for key, item in left.items()
        )
    try:
        return bool(left == right)
    except (RuntimeError, TypeError, ValueError):
        return False


@dataclass
class GradFnCall:
    """Runtime data for one execution of an autograd ``grad_fn_handle`` node."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "call_index": FieldPolicy.KEEP,
        "ordinal": FieldPolicy.KEEP,
        "backward_pass_index": FieldPolicy.KEEP,
        "label": FieldPolicy.KEEP,
        "grad_inputs": FieldPolicy.BLOB_RECURSIVE,
        "grad_outputs": FieldPolicy.BLOB_RECURSIVE,
        "intervention_fire_ref": FieldPolicy.KEEP,
        "timestamp": FieldPolicy.KEEP,
        "_time_started": FieldPolicy.KEEP,
        "_time_finished": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
    }
    FIELD_POLICY = build_record_field_policy_table(
        GRAD_FN_PASS_LOG_FIELD_ORDER,
        {
            **PORTABLE_STATE_SPEC,
            "call_label": PORTABLE_STATE_SPEC["label"],
        },
        schema_key="grad_fn_call",
    )
    PORTABLE_STATE_SPEC = portable_state_spec_from_policy(FIELD_POLICY)

    call_index: int
    ordinal: int | None = None
    backward_pass_index: int | None = None
    label: str = ""
    grad_inputs: Any = None
    grad_outputs: Any = None
    intervention_fire_ref: object | None = None
    timestamp: float | None = None
    _time_started: float | None = None
    _time_finished: float | None = None
    _source_trace_ref: Any = None

    def __post_init__(self) -> None:
        """Fill pass-projection defaults derived from legacy call fields."""

        if self.ordinal is None:
            self.ordinal = self.call_index
        if self.timestamp is None:
            self.timestamp = self._time_finished

    def __tl_state_items__(self) -> Any:
        """Yield live state pairs from the backing row (M9 facade hook)."""

        from .._trace_core.record_rows import record_state_items

        return record_state_items(self)

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
        grad_fn_call_setstate_defaults: dict[str, Any] = {
            **_GRAD_FN_CALL_CONTAINER_DEFAULTS,
            "label": "",
            "ordinal": state.get("call_index"),
            "backward_pass_index": None,
            "grad_inputs": None,
            "grad_outputs": None,
            "intervention_fire_ref": None,
            "timestamp": None,
            "_time_started": None,
            "_time_finished": None,
            "_source_trace_ref": None,
        }
        default_fill_state(state, defaults=grad_fn_call_setstate_defaults)
        # No-op today (no container fields declared), but keeps this class
        # consistent with every sibling record class and automatically covers
        # any container field added to this dataclass in the future.
        coerce_container_typed_state(state, grad_fn_call_setstate_defaults)
        if "duration" in state and "_time_started" not in state and "_time_finished" not in state:
            state["_time_started"] = 0.0
            state["_time_finished"] = float(state.pop("duration"))
        from .._io.state_keys import refuse_callable_shadowing_state_keys

        refuse_callable_shadowing_state_keys(type(self), state)
        from .._trace_core.record_rows import record_state_restore

        record_state_restore(self, state)
        if self.ordinal is None:
            self.ordinal = self.call_index

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

    def __eq__(self, other: object) -> bool:
        """Typed, tensor-safe value equality (never a torch ``RuntimeError``).

        The dataclass-generated ``__eq__`` compared the raw field tuples, so
        two like-labeled calls with saved multi-element gradients raised an
        untyped elementwise-tensor ``RuntimeError`` (and made ``list.index``
        unusable). Scalar identity fields compare by value, saved gradient
        payloads compare tensor-safely, and the ``_source_trace_ref`` weakref
        is excluded (pickle already nulls it).
        """

        if self is other:
            return True
        if not isinstance(other, GradFnCall):
            return NotImplemented
        if (
            self.call_index,
            self.ordinal,
            self.backward_pass_index,
            self.label,
            self.timestamp,
            self._time_started,
            self._time_finished,
        ) != (
            other.call_index,
            other.ordinal,
            other.backward_pass_index,
            other.label,
            other.timestamp,
            other._time_started,
            other._time_finished,
        ):
            return False
        return (
            _grad_fn_payload_equal(self.grad_inputs, other.grad_inputs)
            and _grad_fn_payload_equal(self.grad_outputs, other.grad_outputs)
            and _grad_fn_payload_equal(self.intervention_fire_ref, other.intervention_fire_ref)
        )

    def __hash__(self) -> int:
        """Hash on the immutable scalar identity fields (eq-consistent)."""

        return hash((type(self).__name__, self.call_index, self.backward_pass_index, self.label))

    @property
    def ordinal_index(self) -> int:
        """Return this GradFnCall's 0-based position in ``trace.grad_fn_calls``.

        Amortized O(1) per access through a per-trace identity-position map
        (see :data:`_ORDINAL_POSITIONS_CACHE`); a call not present in its
        trace's projection returns ``-1`` like a trace-less call.
        """

        trace = self.source_trace
        if trace is None:
            return -1
        revision = getattr(trace, "_backward_projection_revision", None)
        cached = _ORDINAL_POSITIONS_CACHE.get(trace)
        if cached is None or cached[0] != revision or id(self) not in cached[1]:
            calls = trace.grad_fn_calls  # syncs the lazy backward projection
            revision = getattr(trace, "_backward_projection_revision", None)
            positions = {id(call): index for index, call in enumerate(calls.values())}
            cached = (revision, positions)
            try:
                _ORDINAL_POSITIONS_CACHE[trace] = cached
            except TypeError:
                pass  # non-weakref-able trace stand-ins: fall through uncached
        return cached[1].get(id(self), -1)

    @property
    def call_label(self) -> str:
        """Return the pass-qualified GradFnCall label.

        Returns
        -------
        str
            GradFn label with the 1-based call index suffix.
        """

        return f"{self.label}:{self.call_index}" if self.label else str(self.call_index)

    @property
    def is_saved(self) -> bool:
        """Return whether this GradFnCall has saved gradient payloads.

        Returns
        -------
        bool
            ``True`` when gradient inputs or outputs were retained by the
            owning Trace's gradient-save selection.
        """

        return self.grad_inputs is not None or self.grad_outputs is not None

    @property
    def backward_duration(self) -> Duration | None:
        """Return the measured backward duration for this call.

        Returns
        -------
        Duration | None
            Seconds elapsed between ``_time_started`` and ``_time_finished``
            (the per-fire ``perf_counter`` pair as of tlspec v8, discriminated
            by ``Trace.grad_fn_timing_provenance``), or ``None`` for an
            untimed fire -- never a false zero.
        """

        if self._time_started is None or self._time_finished is None:
            return None
        return Duration(max(0.0, self._time_finished - self._time_started))

    def to_pandas(self) -> pd.DataFrame:
        """Export this pass as a one-row DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``GRAD_FN_PASS_LOG_FIELD_ORDER``.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        row = {field_name: getattr(self, field_name) for field_name in GRAD_FN_PASS_LOG_FIELD_ORDER}
        return pd.DataFrame([row], columns=GRAD_FN_PASS_LOG_FIELD_ORDER)


# The M9 facade: the declared stored fields become row-cell descriptors
# (dataclass defaults are baked into the generated __init__, so replacing
# the class-attribute defaults is behavior-preserving); the instance dict
# keeps only the store binding + user extras. Rows adopt into the owning
# backward epoch's stores.
_GRAD_FN_CALL_STORED_FIELDS: tuple[str, ...] = (
    "call_index",
    "ordinal",
    "backward_pass_index",
    "label",
    "grad_inputs",
    "grad_outputs",
    "intervention_fire_ref",
    "timestamp",
    "_time_started",
    "_time_finished",
    "_source_trace_ref",
)


def _install_grad_fn_call_facade() -> None:
    """Install the GradFnCall row-cell descriptors (import-time)."""

    from .._trace_core.record_rows import install_record_facade

    install_record_facade(GradFnCall, _GRAD_FN_CALL_STORED_FIELDS)


_install_grad_fn_call_facade()
