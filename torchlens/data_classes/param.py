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

from collections.abc import Iterator
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from .._io import FieldPolicy, TLSPEC_VERSION, default_fill_state, read_tlspec_version
from .._errors import PostTraceParamUnavailable
from ..constants import PARAM_LOG_FIELD_ORDER
from ..ir.refs import DeviceRef, DtypeRef
from ..quantities import Bytes
from ._accessor_base import Accessor
from .op import GradientRecord, GradientRecordAccessor

if TYPE_CHECKING:
    import pandas as pd


def _param_log_to_row(param_log: "Param") -> Dict[str, Any]:
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

    def __init__(
        self,
        module_address: str,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_params: int,
        param_memory: int,
        trainable: bool,
        address: str,
        barcode: str,
        has_optimizer: Optional[bool] = None,
    ) -> None:
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
        self._param_ref: Optional[torch.nn.Parameter] = None
        self._param_ref_released: bool = False
        self._source_trace_ref: Any = None

        # Populated during postprocessing:
        self.num_calls: int = 1  # how many forward ops used this param
        self.used_by_ops: List[str] = []  # op labels that used this param
        self.used_by_layers: List[str] = []  # layer labels that used this param
        self.co_parent_params: List[str] = []  # other param addresses sharing the same tensor
        self._has_grad: bool = False  # one-shot flag: once True, no further checks
        self._grad_shape: Optional[Tuple[int, ...]] = None
        self._grad_dtype: Optional[torch.dtype] = None
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

        return len(self.all_addresses) > 1 or bool(self.co_parent_params)

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
        """Return this Param's 0-based position in ``trace.params``."""

        trace = self.source_trace
        if trace is None:
            return -1
        return list(trace.params).index(self)

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
    def grads(self) -> GradientRecordAccessor:
        """Per-pass accumulating gradient increments observed for this parameter."""

        return GradientRecordAccessor(self._grad_records)

    def _record_gradient_increment(
        self,
        *,
        backward_pass_index: int,
        grad: torch.Tensor,
        timestamp: float,
    ) -> None:
        """Append one AccumulateGrad increment for this parameter.

        Parameters
        ----------
        backward_pass_index:
            One-based global backward pass number.
        grad:
            Incoming gradient increment.
        timestamp:
            Event timestamp.
        """

        saved = grad.detach().clone()
        memory = int(saved.nelement() * saved.element_size())
        self._grad_records.append(
            GradientRecord(
                owner=self,
                ordinal=len(self._grad_records) + 1,
                backward_pass_index=backward_pass_index,
                grad=saved,
                transformed_grad=None,
                shape=tuple(saved.shape),
                dtype=str(saved.dtype),
                memory=memory,
                timestamp=timestamp,
            )
        )

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

    def _resolve_live_param(self) -> torch.nn.Parameter | None:
        """Resolve the live parameter after post-trace reference release.

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
        source_ref = getattr(trace, "_source_model_ref", None) if trace is not None else None
        if source_ref is None:
            return None

        model = source_ref()
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

        self._param_ref = param
        return param

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
    def grad_shape(self) -> Optional[Tuple[int, ...]]:
        """Return the grad tensor shape.

        Returns
        -------
        Optional[Tuple[int, ...]]
            Shape of the grad tensor, or ``None`` if no grad exists.
        """
        self._check_param_grad()
        return self._grad_shape

    @grad_shape.setter
    def grad_shape(self, value: Optional[Tuple[int, ...]]) -> None:
        """Set cached grad tensor shape.

        Parameters
        ----------
        value:
            Cached grad tensor shape, or ``None`` when absent.
        """
        self._grad_shape = value

    @property
    def grad_dtype(self) -> Optional[torch.dtype]:
        """Return the grad tensor dtype.

        Returns
        -------
        Optional[torch.dtype]
            Dtype of the grad tensor, or ``None`` if no grad exists.
        """
        self._check_param_grad()
        return self._grad_dtype

    @grad_dtype.setter
    def grad_dtype(self, value: Optional[torch.dtype]) -> None:
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
        """Multi-line summary showing address, shape, dtype, trainability, and usage."""
        status = "trainable" if self.is_trainable else "frozen"
        lines = [
            f"Param: {self.address}",
            f"  shape: {self.shape}",
            f"  dtype: {self.dtype}",
            f"  size: {self.param_memory}",
            f"  {status}",
            f"  has_grad: {self.has_grad}",
            f"  module: {self.module_address} ({self._module_display_name()})",
        ]
        if self.used_by_layers:
            lines.append(f"  used by: {', '.join(self.used_by_layers)}")
        if self.co_parent_params:
            lines.append(f"  linked: {', '.join(self.co_parent_params)}")
        if self.has_optimizer is not None:
            lines.append(f"  has_optimizer: {self.has_optimizer}")
        if self.num_calls > 1:
            lines.append(f"  num_calls: {self.num_calls}")
        return "\n".join(lines)

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

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with live parameter references stripped."""
        state = self.__dict__.copy()
        state["_param_ref"] = None
        state["_param_ref_released"] = False
        state["_source_trace_ref"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state without reviving live parameter references."""
        read_tlspec_version(state, cls_name=type(self).__name__)
        for removed_field in ("module_class_name", "module_class_qualname", "module_type"):
            state.pop(removed_field, None)
        if "param_memory" not in state and "memory" in state:
            state["param_memory"] = state.pop("memory")
        if "is_trainable" not in state and "trainable" in state:
            state["is_trainable"] = state.pop("trainable")
        default_fill_state(
            state,
            defaults={
                "_param_ref": None,
                "_param_ref_released": False,
                "_source_trace_ref": None,
                "dtype_ref": DtypeRef.from_value(state.get("dtype")),
                "device_ref": None,
                "backend_address": state.get("address"),
                "resolver_status": "resolved",
                "_derived_grad_payload": None,
                "_derived_grad_record_path": None,
            },
        )
        if state.get("dtype_ref") is None:
            state["dtype_ref"] = DtypeRef.from_value(state.get("dtype"))
        if state.get("backend_address") is None:
            state["backend_address"] = state.get("address")
        if state.get("resolver_status") is None:
            state["resolver_status"] = "resolved"
        state["param_memory"] = Bytes(state.get("param_memory", 0) or 0)
        state["_grad_memory"] = Bytes(state.get("_grad_memory", 0) or 0)
        self.__dict__.update(state)


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

    def __init__(self, param_logs: Dict[str, "Param"]) -> None:
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
        # Fallback: match by short name (e.g. 'weight', 'bias')
        matches = [pl for pl in self._list if pl.name == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"Ambiguous short name '{key}' — use full address")
        return None

    def __contains__(self, key: object) -> bool:
        """Check membership by full address, short name, or integer index (#84)."""
        try:
            return super().__contains__(key)
        except KeyError:
            return True

    def __repr__(self) -> str:
        """Format as a dict-like string of parameter addresses with shapes and status."""
        if len(self) == 0:
            return "{}"
        items = []
        for pl in self._list:
            status = "trainable" if pl.is_trainable else "frozen"
            items.append(f"'{pl.address}': Param {pl.shape} {pl.dtype} {status}")
        inner = ",\n ".join(items)
        return "{" + inner + "}"

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


setattr(Param, "co_parent_params", [])
