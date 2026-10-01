"""Param and ParamAccessor: per-parameter metadata and dict-like accessor for model parameters.

Param stores static metadata (address, shape, dtype, trainability) plus
lazy grad information.  It does NOT store the parameter tensor itself --
only a weak-ish reference (``_param_ref``) used solely for lazy grad
access via ``_check_param_grad()``.

**GC concern with _param_ref**: ``_param_ref`` holds a direct reference to
the ``nn.Parameter`` object.  This prevents the parameter from being garbage
collected as long as the Param (and thus the Trace) is alive.  This
is acceptable because the Trace's lifetime is typically shorter than or
equal to the model's lifetime.  The ``cleanup()`` method on Trace
deletes all Param references.

**Lazy grad properties**: Gradient metadata (has_grad, grad_shape, grad_dtype,
gradient_memory) is computed lazily on first access via ``_check_param_grad()``.
This allows grads computed after ``trace()`` returns (e.g.
after a ``loss.backward()`` call) to be reflected without re-logging.
The check is one-shot: once ``_has_grad`` is True, no further checks are made.
"""

import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._errors import AmbiguousOpLookupError, PostTraceParamUnavailable
from .._io import (
    TLSPEC_VERSION,
    FieldPolicy,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from ..constants import PARAM_LOG_FIELD_ORDER
from ..ir.refs import DeviceRef, DtypeRef
from ..quantities import Bytes
from ._accessor_base import Accessor
from ._runtime_handles import source_model_from_trace
from .field_policy import build_record_field_policy_table, portable_state_spec_from_policy
from .op import GradientRecord, GradientRecordAccessor

#: Per-accessor id -> position maps for ``Param.ordinal_index`` (weakly keyed
#: so a dropped accessor releases its map). Entries are verified by identity
#: against the accessor's current ``_list`` before use, so a stale map can
#: only trigger a rebuild, never a wrong answer.
_ORDINAL_INDEX_CACHE: "weakref.WeakKeyDictionary[Any, dict[int, int]]" = weakref.WeakKeyDictionary()

if TYPE_CHECKING:
    import pandas as pd


def _param_log_to_row(param_log: "Param") -> dict[str, Any]:
    """Convert a Param into one DataFrame row.

    Parameters
    ----------
    param_log:
        Parameter metadata entry to export.

    Returns
    -------
    Dict[str, Any]
        Mapping from canonical field name to exported value.
    """
    return {field: getattr(param_log, field) for field in PARAM_LOG_FIELD_ORDER}


# Typed container defaults for every non-Optional container field Param
# stores directly. Same defect class as
# `Op._LAYER_PASS_LOG_CONTAINER_DEFAULTS`/`Trace._MODEL_LOG_CONTAINER_DEFAULTS`:
# without this, `coerce_container_typed_state` cannot repair a
# present-but-wrong-typed legacy value (e.g. `co_parent_params` serialized as
# a `set` where a `list` is now declared), and an absent field crashes instead
# of restoring an empty typed container. Plain builtin types are used
# deliberately.
_PARAM_CONTAINER_DEFAULTS: dict[str, Any] = {
    "all_addresses": [],
    "all_module_addresses": [],
    "used_by_ops": [],
    "used_by_layers": [],
    "co_parent_params": [],
}


#: Closed vocabulary for the derived parameter value basis. ``snapshot`` is
#: declared but unreachable until capture-time parameter snapshots (R8(b))
#: land and own their carriage; only ``snapshot`` counts as immutable
#: capture-time parameter evidence.
_PARAM_VALUE_BASES = frozenset({"live_ref", "absent", "snapshot"})


@dataclass(frozen=True)
class ParamValueBasis:
    """Derived, read-time disclosure of where a parameter value comes from.

    TorchLens records WHICH parameter a run used, not its bytes:
    ``Param.value`` resolves through a live handle to the source model and
    returns TODAY'S value, which may have moved since capture. This basis is
    computed at read time and persisted NOWHERE -- it is a disclosure beside
    the documented live read, never a change to it.

    Values (closed vocabulary; spellings DOCUMENTED-UNSTABLE pending the
    naming session):

    - ``live_ref``: the live-model handle resolves; the value may have moved
      since capture.
    - ``absent`` with reason ``not_persisted``: the trace was deserialized
      and parameter bytes were never persisted, so there is no value to
      serve (this replaces the historical untyped bare ``None``).
    - ``snapshot``: reserved for capture-time parameter snapshots (R8(b));
      unreachable today.
    """

    basis: str
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the closed basis vocabulary."""

        if self.basis not in _PARAM_VALUE_BASES:
            raise ValueError(
                f"parameter value basis must be one of {sorted(_PARAM_VALUE_BASES)}, "
                f"got {self.basis!r}"
            )

    @property
    def is_immutable(self) -> bool:
        """Return whether this basis is immutable capture-time evidence.

        Only ``snapshot`` (R8(b), future) qualifies; a live handle or an
        absent value never proves what the bytes were at capture time.
        """

        return self.basis == "snapshot"

    @property
    def description(self) -> str:
        """Return the one-sentence teaching gloss for this basis."""

        if self.basis == "live_ref":
            return (
                "The value is read through a live handle to the source model; "
                "it may have moved since capture."
            )
        if self.basis == "snapshot":
            return "The value is an immutable capture-time parameter snapshot."
        return (
            "The trace was deserialized and parameter bytes were never "
            "persisted; there is no value to serve."
        )

    def __str__(self) -> str:
        """Return the compact ``basis`` or ``basis(reason)`` spelling."""

        return self.basis if self.reason is None else f"{self.basis}({self.reason})"


class Param:
    """Metadata about a single model parameter (weight or bias).

    Captures static parameter identity (address, shape, dtype, trainability)
    and links to the module that owns it.  Does NOT store the parameter tensor
    itself -- only a ``_param_ref`` reference for lazy grad access.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "module_address": FieldPolicy.KEEP,
        "name": FieldPolicy.KEEP,
        "shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "dtype_ref": FieldPolicy.KEEP,
        "device_ref": FieldPolicy.KEEP,
        "backend_address": FieldPolicy.KEEP,
        "resolver_status": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "param_memory": FieldPolicy.KEEP,
        "is_trainable": FieldPolicy.KEEP,
        "address": FieldPolicy.KEEP,
        "all_addresses": FieldPolicy.KEEP,
        "all_module_addresses": FieldPolicy.KEEP,
        "barcode": FieldPolicy.KEEP,
        "has_optimizer": FieldPolicy.KEEP,
        "_param_ref": FieldPolicy.DROP,
        "_param_ref_released": FieldPolicy.DROP,
        "_source_trace_ref": FieldPolicy.DROP,
        # Transient postprocess bookkeeping (set True at prep for a param
        # still pending as an UninitializedParameter, flipped back to False
        # by step 15's `_finalize_lazy_param_geometry` once the captured
        # forward materializes it in place). Never meaningful after a Trace
        # finishes construction; DROP, same as the other internal flags here.
        "_lazy_at_prep": FieldPolicy.DROP,
        "num_calls": FieldPolicy.KEEP,
        "used_by_ops": FieldPolicy.KEEP,
        "used_by_layers": FieldPolicy.KEEP,
        "co_parent_params": FieldPolicy.KEEP,
        "_has_grad": FieldPolicy.KEEP,
        "_grad_shape": FieldPolicy.KEEP,
        "_grad_dtype": FieldPolicy.KEEP,
        "_grad_memory": FieldPolicy.KEEP,
        "_grad_records": FieldPolicy.DROP,
        "_derived_grad_payload": FieldPolicy.KEEP,
        "_derived_grad_record_path": FieldPolicy.KEEP,
    }
    FIELD_POLICY = build_record_field_policy_table(
        PARAM_LOG_FIELD_ORDER, PORTABLE_STATE_SPEC, schema_key="param"
    )
    PORTABLE_STATE_SPEC = portable_state_spec_from_policy(FIELD_POLICY)

    def __init__(
        self,
        module_address: str,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        num_params: int,
        param_memory: int,
        trainable: bool,
        address: str,
        barcode: str,
        has_optimizer: bool | None = None,
    ) -> None:
        """Initialize persistent metadata for one model parameter.

        Parameters
        ----------
        module_address:
            Address of the module that owns the parameter.
        name:
            Parameter name relative to the owning module.
        shape:
            Parameter tensor shape.
        dtype:
            Parameter tensor dtype.
        num_params:
            Number of scalar elements in the parameter.
        param_memory:
            Parameter memory in bytes.
        trainable:
            Whether the parameter requires gradients.
        address:
            Fully-qualified parameter address.
        barcode:
            Stable identity token used during graph attribution.
        has_optimizer:
            Whether optimizer state was detected for the parameter.
        """

        self.address = address  # e.g. "features.0.weight"
        self.name = name  # short name, e.g. "weight"
        self.shape = shape
        self.dtype = dtype
        self.dtype_ref: DtypeRef | None = DtypeRef.from_value(dtype)
        self.device_ref: DeviceRef | None = None
        self.backend_address: str | None = address
        self.resolver_status: str = "resolved"
        self.num_params = num_params
        self.param_memory = Bytes(param_memory)
        self.is_trainable = trainable
        self.module_address = module_address
        self.all_addresses = [address]
        self.all_module_addresses = [module_address]
        self.barcode = barcode
        self.has_optimizer = has_optimizer

        # Direct reference to the actual nn.Parameter for lazy grad access.
        # Prevents GC of the parameter while this Param is alive (acceptable
        # because Trace lifetime <= model lifetime; cleanup() clears it).
        self._param_ref: torch.nn.Parameter | None = None
        self._param_ref_released: bool = False
        self._source_trace_ref: Any = None

        # Populated during postprocessing:
        self.num_calls: int = 1  # how many forward ops used this param
        self.used_by_ops: list[str] = []  # op labels that used this param
        self.used_by_layers: list[str] = []  # layer labels that used this param
        self.co_parent_params: list[str] = []  # other param addresses used by the same op
        self._has_grad: bool = False  # one-shot flag: once True, no further checks
        self._grad_shape: tuple[int, ...] | None = None
        self._grad_dtype: torch.dtype | None = None
        self._grad_memory: Bytes = Bytes(0)
        self._grad_records: list[GradientRecord] = []
        self._derived_grad_payload: Any | None = None
        self._derived_grad_record_path: str | None = None

    @property
    def is_quantized(self) -> bool:
        """Whether this parameter uses a quantized dtype (qint8, quint8, etc.)."""
        _QUANTIZED_DTYPES = {
            torch.qint8,
            torch.quint8,
            torch.qint32,
            torch.quint4x2,
            torch.quint2x4,
        }
        return self.dtype in _QUANTIZED_DTYPES

    @property
    def has_multiple_addresses(self) -> bool:
        """Return whether this parameter is registered at multiple addresses.

        Returns
        -------
        bool
            Whether multiple parameter addresses share this tensor.
        """

        return len(self.all_addresses) > 1

    @property
    def num_uses_by_ops(self) -> int:
        """Return the number of distinct Op usages.

        Returns
        -------
        int
            Count of pass-qualified Op labels in ``used_by_ops``.
        """

        return len(self.used_by_ops)

    @property
    def num_uses_by_layers(self) -> int:
        """Return the number of distinct Layer usages.

        Returns
        -------
        int
            Count of Layer labels in ``used_by_layers``.
        """

        return len(self.used_by_layers)

    @property
    def source_trace(self) -> Any:
        """Owning Trace, if still alive."""

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
    def ordinal_index(self) -> int:
        """Return this Param's 0-based position in ``trace.params``.

        Amortized O(1): the historical ``list(trace.params).index(self)``
        materialized the accessor (through its ref-rehydrating ``__iter__``)
        and identity-scanned it on EVERY read, so a full-table sweep measured
        a ~2.1 scaling exponent. The id-keyed position map is cached per
        accessor and verified by identity before use (stale caches rebuild),
        so results are exactly the historical identity semantics.
        """

        trace = self.source_trace
        if trace is None:
            return -1
        accessor = trace.params
        items = getattr(accessor, "_list", None)
        if not isinstance(items, list):
            return list(accessor).index(self)
        cache = _ORDINAL_INDEX_CACHE.get(accessor)
        if cache is not None and len(cache) == len(items):
            index = cache.get(id(self))
            if index is not None and items[index] is self:
                return index
        cache = {id(param): position for position, param in enumerate(items)}
        _ORDINAL_INDEX_CACHE[accessor] = cache
        index = cache.get(id(self))
        if index is None:
            raise ValueError(f"{self.address!r} is not in trace.params")
        return index

    @property
    def module(self) -> Any:
        """Primary owning Module."""

        trace = self.source_trace
        if trace is None:
            return None
        return trace.modules[self.module_address]

    @property
    def module_name(self) -> str:
        """Return the bare local name of the owning module.

        Returns
        -------
        str
            Final dotted segment of ``module_address``.
        """

        return "" if self.module_address == "self" else self.module_address.rsplit(".", 1)[-1]

    @property
    def module_cls(self) -> type[Any] | None:
        """Return the live class object for the owning module when available.

        Returns
        -------
        type[Any] | None
            Owning module class, or ``None`` if the model/module is unavailable.
        """

        module = self.module
        return None if module is None else module.cls

    @property
    def modules(self) -> list[Any]:
        """All owning ModuleLogs."""

        trace = self.source_trace
        if trace is None:
            return []
        return [trace.modules[address] for address in self.all_module_addresses]

    def _module_display_name(self) -> str:
        """Return the owning module class name for display output.

        Returns
        -------
        str
            The owning module class name, or an empty string when unavailable.
        """

        module = self.module
        return "" if module is None else str(getattr(module, "class_name", ""))

    @property
    def grad(self) -> Any | None:
        """Return the live gradient tensor for this parameter.

        Returns
        -------
        Any | None
            Live ``nn.Parameter.grad`` value, or a backend-derived gradient
            payload for non-torch pytree parameters when available.
        """

        if self._derived_grad_payload is not None:
            return self._derived_grad_payload
        param = self._resolve_live_param()
        return None if param is None else param.grad

    @property
    def value(self) -> torch.nn.Parameter | None:
        """Return the live model parameter when the source model is available.

        Returns
        -------
        torch.nn.Parameter | None
            Live parameter object, or ``None`` for deserialized traces that have
            no source-model weakref.
        """

        return self._resolve_live_param()

    @property
    def value_basis(self) -> "ParamValueBasis":
        """Return the derived, read-time basis for this parameter's value.

        Computed at read time and persisted nowhere (no schema act): the
        basis is a disclosure BESIDE the documented live-handle read
        (``value``/``handle``), never a change to it. ``live_ref`` means the
        live-model handle resolves and the value may have moved since
        capture; ``absent(not_persisted)`` means the trace was deserialized
        and parameter bytes were never persisted (replacing the historical
        untyped bare ``None``); ``snapshot`` arrives only with capture-time
        parameter snapshots (R8(b)). Spelling DOCUMENTED-UNSTABLE pending the
        naming session.

        Returns
        -------
        ParamValueBasis
            The derived value basis, never ``None``.
        """

        try:
            param = self._peek_live_param(cache=False)
        except PostTraceParamUnavailable:
            return ParamValueBasis("absent", "not_persisted")
        if param is None:
            return ParamValueBasis("absent", "not_persisted")
        return ParamValueBasis("live_ref")

    @property
    def handle(self) -> torch.nn.Parameter | None:
        """Return the live model parameter when reachable without caching it.

        Returns
        -------
        torch.nn.Parameter | None
            Live parameter object, or ``None`` when the source model or
            parameter address is unavailable. This computed runtime handle is
            not portable and does not repopulate ``_param_ref``.
        """

        try:
            return self._peek_live_param(cache=False)
        except PostTraceParamUnavailable:
            return None

    @property
    def grads(self) -> GradientRecordAccessor:
        """Per-pass accumulating gradient increments observed for this parameter."""

        return GradientRecordAccessor(self._grad_records)

    def _append_gradient_record(
        self,
        *,
        backward_pass_index: int,
        grad: Any | None,
        shape: tuple[int, ...] | None,
        dtype: str | None,
        memory: int | None,
        timestamp: float,
    ) -> "GradientRecord":
        """Append one projected AccumulateGrad increment for this parameter.

        Called only by the backward projection fold: the ``ParamGradObserved``
        event stream is the single authoritative source for these records, so
        the live AccumulateGrad hook never writes them directly.

        Parameters
        ----------
        backward_pass_index:
            One-based global backward pass number.
        grad:
            Retained gradient payload from the event, already detached.
        shape:
            Observed gradient shape from the event.
        dtype:
            Observed gradient dtype string from the event.
        memory:
            Observed gradient memory in bytes from the event.
        timestamp:
            Event timestamp.

        Returns
        -------
        GradientRecord
            The appended record.
        """

        record = GradientRecord(
            owner=self,
            ordinal=len(self._grad_records) + 1,
            backward_pass_index=backward_pass_index,
            grad=grad,
            transformed_grad=None,
            shape=shape,
            dtype=dtype,
            memory=memory,
            timestamp=timestamp,
        )
        self._grad_records.append(record)
        return record

    def _clear_gradient_records(self) -> None:
        """Reset projected gradient records ahead of a full projection rebuild."""

        self._grad_records = []

    def _check_param_grad(self) -> None:
        """Lazily check if the parameter has a grad and cache the result.

        Called by each grad property on first access.  Once a grad is
        found, all grad metadata is cached and no further checks are made
        (``_has_grad`` acts as a one-shot flag).
        """
        if self._derived_grad_payload is not None:
            self._has_grad = True
            return
        try:
            param = self._resolve_live_param()
        except PostTraceParamUnavailable:
            return
        if not self._has_grad and param is not None and param.grad is not None:
            grad = param.grad
            self._has_grad = True
            self._grad_shape = tuple(grad.shape)
            self._grad_dtype = grad.dtype
            self._grad_memory = Bytes(grad.nelement() * grad.element_size())

    def _peek_live_param(self, *, cache: bool) -> torch.nn.Parameter | None:
        """Resolve the live parameter after post-trace reference release.

        Parameters
        ----------
        cache:
            Whether to cache a source-model read-through in ``_param_ref``.

        Returns
        -------
        torch.nn.Parameter | None
            Live parameter if available; ``None`` when no source trace/model
            weakref exists, such as after portable deserialization.

        Raises
        ------
        PostTraceParamUnavailable
            If this Param released its direct reference and the source model
            weakref is now dead, or the parameter address is no longer present.
        """

        if self._param_ref is not None:
            return self._param_ref

        trace = self.source_trace
        if getattr(trace, "_source_model_ref", None) is None:
            return None

        model = source_model_from_trace(trace)
        if model is None:
            if self._param_ref_released:
                raise PostTraceParamUnavailable(
                    f"Parameter '{self.address}' is unavailable because the source model "
                    "has been garbage-collected after TorchLens released its direct "
                    "parameter reference."
                )
            return None

        try:
            param = model.get_parameter(self.address)
        except AttributeError as exc:
            raise PostTraceParamUnavailable(
                f"Source model for parameter '{self.address}' does not support get_parameter()."
            ) from exc
        except KeyError as exc:
            raise PostTraceParamUnavailable(
                f"Parameter '{self.address}' is no longer registered on the source model."
            ) from exc

        if cache:
            self._param_ref = param
        return param

    def _resolve_live_param(self) -> torch.nn.Parameter | None:
        """Resolve and cache the live parameter when reachable.

        Returns
        -------
        torch.nn.Parameter | None
            Live parameter if available; ``None`` when no source trace/model
            weakref exists, such as after portable deserialization.

        Raises
        ------
        PostTraceParamUnavailable
            If this Param released its direct reference and the source model
            weakref is now dead, or the parameter address is no longer present.
        """

        return self._peek_live_param(cache=True)

    @property
    def has_grad(self) -> bool:
        """Return whether this parameter currently has a grad stored.

        Returns
        -------
        bool
            ``True`` when the referenced parameter has a grad.
        """
        self._check_param_grad()
        return self._has_grad

    @has_grad.setter
    def has_grad(self, value: bool) -> None:
        """Set cached grad-presence status.

        Parameters
        ----------
        value:
            Cached grad-presence status.
        """
        self._has_grad = value

    @property
    def grad_shape(self) -> tuple[int, ...] | None:
        """Return the grad tensor shape.

        Returns
        -------
        Optional[Tuple[int, ...]]
            Shape of the grad tensor, or ``None`` if no grad exists.
        """
        self._check_param_grad()
        return self._grad_shape

    @grad_shape.setter
    def grad_shape(self, value: tuple[int, ...] | None) -> None:
        """Set cached grad tensor shape.

        Parameters
        ----------
        value:
            Cached grad tensor shape, or ``None`` when absent.
        """
        self._grad_shape = value

    @property
    def grad_dtype(self) -> torch.dtype | None:
        """Return the grad tensor dtype.

        Returns
        -------
        Optional[torch.dtype]
            Dtype of the grad tensor, or ``None`` if no grad exists.
        """
        self._check_param_grad()
        return self._grad_dtype

    @grad_dtype.setter
    def grad_dtype(self, value: torch.dtype | None) -> None:
        """Set cached grad tensor dtype.

        Parameters
        ----------
        value:
            Cached grad tensor dtype, or ``None`` when absent.
        """
        self._grad_dtype = value

    @property
    def gradient_memory(self) -> Bytes:
        """Return the grad tensor size in bytes.

        Returns
        -------
        Bytes
            Size of the grad tensor in bytes.
        """
        self._check_param_grad()
        return self._grad_memory

    @gradient_memory.setter
    def gradient_memory(self, value: int) -> None:
        """Set cached grad tensor size in bytes.

        Parameters
        ----------
        value:
            Cached grad memory amount in bytes.
        """
        self._grad_memory = Bytes(value)

    def __repr__(self) -> str:
        """One live/versioned envelope+core line (F10; lovely D30).

        Params are live records: the core is computed from the live
        ``nn.Parameter`` and version-checked per render; a released ref
        degrades to metadata with the explicit basis. Never raises.
        """

        from ..utils.fail_open import fail_open
        from ._value_repr import param_repr_line

        return fail_open(
            lambda: param_repr_line(self),
            lambda _error: f"<Param {getattr(self, 'address', '<unbound>')}: repr degraded>",
        )

    def __str__(self) -> str:
        """Bounded Param card with tie disclosure (F10; lovely matrix).

        Line 1 is the repr; the body adds module home, consuming sites
        (tied params NAMED -- real gpt2's ``wte.weight`` feeds two sites),
        grad/optimizer facts. Never raises.
        """

        from ..utils.fail_open import fail_open
        from ._value_repr import param_card

        return fail_open(lambda: param_card(self), lambda _error: self.__repr__())

    def release_param_ref(self) -> None:
        """Cache grad info, then null _param_ref to allow param GC."""
        if self._param_ref is None and self._param_ref_released:
            return
        self._check_param_grad()
        self._param_ref = None
        self._param_ref_released = True

    def to_pandas(self) -> "pd.DataFrame":
        """Export this Param as a one-row pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``PARAM_LOG_FIELD_ORDER``.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        return pd.DataFrame([_param_log_to_row(self)], columns=PARAM_LOG_FIELD_ORDER)

    def __len__(self) -> int:
        """Return the number of scalar elements in this parameter."""
        return self.num_params

    def __tl_state_items__(self) -> Any:
        """Yield live state pairs from the backing row (M8 facade hook)."""

        from .._trace_core.record_rows import record_state_items

        return record_state_items(self)

    def __tl_state_restore__(self, mapping: dict[str, Any]) -> None:
        """Install a state mapping through the cell descriptors (M8 hook)."""

        from .._trace_core.record_rows import record_state_restore

        record_state_restore(self, mapping)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with live parameter references stripped."""
        from ._state_adapter import state_items

        state = dict(state_items(self))
        state["_param_ref"] = None
        state["_param_ref_released"] = False
        state["_source_trace_ref"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state without reviving live parameter references."""
        read_tlspec_version(state, cls_name=type(self).__name__, cls=type(self))
        for removed_field in ("module_class_name", "module_class_qualname", "module_type"):
            state.pop(removed_field, None)
        if "param_memory" not in state and "memory" in state:
            state["param_memory"] = state.pop("memory")
        if "is_trainable" not in state and "trainable" in state:
            state["is_trainable"] = state.pop("trainable")
        param_setstate_defaults: dict[str, Any] = {
            **_PARAM_CONTAINER_DEFAULTS,
            "_param_ref": None,
            "_param_ref_released": False,
            "_source_trace_ref": None,
            "dtype_ref": DtypeRef.from_value(state.get("dtype")),
            "device_ref": None,
            "backend_address": state.get("address"),
            "resolver_status": "resolved",
            "_derived_grad_payload": None,
            "_derived_grad_record_path": None,
        }
        default_fill_state(state, defaults=param_setstate_defaults)
        # Repair present-but-wrong-typed container fields from legacy states
        # (e.g. `co_parent_params` serialized as a `set` where a `list` is now
        # declared). `default_fill_state` only fills absent keys; this closes
        # the same gap `Trace`/`Op` already close for their own fields.
        coerce_container_typed_state(state, param_setstate_defaults)
        if state.get("dtype_ref") is None:
            state["dtype_ref"] = DtypeRef.from_value(state.get("dtype"))
        if state.get("backend_address") is None:
            state["backend_address"] = state.get("address")
        if state.get("resolver_status") is None:
            state["resolver_status"] = "resolved"
        state["param_memory"] = Bytes(state.get("param_memory", 0) or 0)
        state["_grad_memory"] = Bytes(state.get("_grad_memory", 0) or 0)
        from .._io.state_keys import refuse_callable_shadowing_state_keys

        refuse_callable_shadowing_state_keys(type(self), state)
        from .._trace_core.record_rows import record_state_restore

        record_state_restore(self, state)


# The M8 facade: every declared stored field becomes a row-cell descriptor
# (the literal ``PORTABLE_STATE_SPEC`` keys above, in declared order); the
# instance ``__dict__`` keeps only the store binding and user extras.
_PARAM_STORED_FIELDS: tuple[str, ...] = (
    "module_address",
    "name",
    "shape",
    "dtype",
    "dtype_ref",
    "device_ref",
    "backend_address",
    "resolver_status",
    "num_params",
    "param_memory",
    "is_trainable",
    "address",
    "all_addresses",
    "all_module_addresses",
    "barcode",
    "has_optimizer",
    "_param_ref",
    "_param_ref_released",
    "_source_trace_ref",
    "num_calls",
    "used_by_ops",
    "used_by_layers",
    "co_parent_params",
    "_has_grad",
    "_grad_shape",
    "_grad_dtype",
    "_grad_memory",
    "_grad_records",
    "_derived_grad_payload",
    "_derived_grad_record_path",
)


def _install_param_facade() -> None:
    """Install the Param row-cell descriptors (import-time, collision-safe)."""

    from .._trace_core.record_rows import install_record_facade

    install_record_facade(Param, _PARAM_STORED_FIELDS)


_install_param_facade()


class ParamAccessor(Accessor["Param"]):
    """Dict-like accessor for Param objects.

    Supports indexing by:
    * **full address** (str) -- e.g. ``"features.0.weight"``.
    * **short name** (str) -- e.g. ``"weight"`` (must be unambiguous).
    * **ordinal position** (int) -- index into insertion-order list.

    Available as ``trace.params``, ``layer_log.params``, ``module_log.params``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_rehydrate_on_iter": FieldPolicy.DROP,
    }

    def __init__(self, param_logs: dict[str, "Param"]) -> None:
        """Initialize an accessor over parameter logs.

        Parameters
        ----------
        param_logs:
            Mapping from parameter addresses to ``Param`` logs.
        """

        super().__init__(param_logs)
        self._rehydrate_on_iter = False

    def __iter__(self) -> Iterator["Param"]:
        """Iterate over params, restoring live refs when the source model is available."""

        for param_log in self._list:
            if self._rehydrate_on_iter:
                try:
                    param_log._resolve_live_param()
                except PostTraceParamUnavailable:
                    pass
            yield param_log

    def _resolve_substring(self, key: str) -> "Param | None":
        """Resolve an unambiguous parameter short name."""
        for param_log in self._list:
            if key in param_log.all_addresses:
                return param_log
        # Fallback: match by short name (e.g. 'weight', 'bias')
        matches = [pl for pl in self._list if pl.name == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Name the bounded candidate set (R65) like the merged-presenter
            # sibling, instead of telling the user to guess the full address.
            candidates = tuple(address for match in matches for address in match.all_addresses)
            raise AmbiguousOpLookupError(
                f"Ambiguous short name '{key}' -- use a full address: {', '.join(candidates)}",
                candidates=candidates,
            )
        return None

    def _resolve_pass_qualified(self, key: str) -> "Param | None":
        """Resolve pass-qualified notation to the parent Param."""

        base, _, pass_str = key.rpartition(":")
        try:
            int(pass_str)
        except ValueError:
            return None
        if base in self._dict:
            return self._dict[base]
        return self._resolve_substring(base)

    def __contains__(self, key: object) -> bool:
        """Check membership by full address, short name, or integer index (#84)."""
        try:
            self[key]  # type: ignore[index]
        except (KeyError, TypeError, IndexError, ValueError):
            return False
        return True

    def _composition_note(self) -> str | None:
        """Composition breakdown for the one-line card (F10; never a dump)."""

        trainable = sum(1 for pl in self._list if pl.is_trainable)
        frozen = len(self._list) - trainable
        if frozen == 0:
            return None
        return f"{trainable} trainable, {frozen} frozen"

    def to_pandas(self) -> "pd.DataFrame":
        """Export parameter metadata as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per parameter, ordered by ``PARAM_LOG_FIELD_ORDER``.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = [_param_log_to_row(param_log) for param_log in self._list]
        return pd.DataFrame(rows, columns=PARAM_LOG_FIELD_ORDER)
