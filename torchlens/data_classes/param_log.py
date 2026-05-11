"""ParamLog and ParamAccessor: per-parameter metadata and dict-like accessor for model parameters.

ParamLog stores static metadata (address, shape, dtype, trainability) plus
lazy grad information.  It does NOT store the parameter tensor itself --
only a weak-ish reference (``_param_ref``) used solely for lazy grad
access via ``_check_param_grad()``.

**GC concern with _param_ref**: ``_param_ref`` holds a direct reference to
the ``nn.Parameter`` object.  This prevents the parameter from being garbage
collected as long as the ParamLog (and thus the Trace) is alive.  This
is acceptable because the Trace's lifetime is typically shorter than or
equal to the model's lifetime.  The ``cleanup()`` method on Trace
deletes all ParamLog references.

**Lazy grad properties**: Gradient metadata (has_grad, grad_shape, grad_dtype,
grad_memory) is computed lazily on first access via ``_check_param_grad()``.
This allows grads computed after ``trace()`` returns (e.g.
after a ``loss.backward()`` call) to be reflected without re-logging.
The check is one-shot: once ``_has_grad`` is True, no further checks are made.
"""

from os import PathLike
from collections.abc import Iterator
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import torch

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..constants import PARAM_LOG_FIELD_ORDER
from ..utils.display import human_readable_size
from ._accessor_base import Accessor

if TYPE_CHECKING:
    import pandas as pd


def _param_log_to_row(param_log: "ParamLog") -> Dict[str, Any]:
    """Convert a ParamLog into one DataFrame row.

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


class ParamLog:
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
        "num_params": FieldPolicy.KEEP,
        "memory": FieldPolicy.KEEP,
        "trainable": FieldPolicy.KEEP,
        "address": FieldPolicy.KEEP,
        "all_addresses": FieldPolicy.KEEP,
        "all_module_addresses": FieldPolicy.KEEP,
        "module_class_name": FieldPolicy.KEEP,
        "module_class_qualname": FieldPolicy.KEEP,
        "module_type": FieldPolicy.KEEP,
        "barcode": FieldPolicy.KEEP,
        "has_optimizer": FieldPolicy.KEEP,
        "_param_ref": FieldPolicy.DROP,
        "_source_trace_ref": FieldPolicy.DROP,
        "num_calls": FieldPolicy.KEEP,
        "used_by_layers": FieldPolicy.KEEP,
        "co_parent_params": FieldPolicy.KEEP,
        "_has_grad": FieldPolicy.KEEP,
        "_grad_shape": FieldPolicy.KEEP,
        "_grad_dtype": FieldPolicy.KEEP,
        "_grad_memory": FieldPolicy.KEEP,
        "_grad_memory_str": FieldPolicy.KEEP,
    }

    def __init__(
        self,
        module_address: str,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_params: int,
        memory: int,
        trainable: bool,
        address: str,
        module_type: str,
        barcode: str,
        has_optimizer: Optional[bool] = None,
    ) -> None:
        self.address = address  # e.g. "features.0.weight"
        self.name = name  # short name, e.g. "weight"
        self.shape = shape
        self.dtype = dtype
        self.num_params = num_params
        self.memory = memory
        self.trainable = trainable
        self.module_address = module_address
        self.all_addresses = [address]
        self.all_module_addresses = [module_address]
        self.module_class_name = module_type
        self.module_class_qualname = module_type
        self.module_type = module_type
        self.barcode = barcode
        self.has_optimizer = has_optimizer

        # Direct reference to the actual nn.Parameter for lazy grad access.
        # Prevents GC of the parameter while this ParamLog is alive (acceptable
        # because Trace lifetime <= model lifetime; cleanup() clears it).
        self._param_ref: Optional[torch.nn.Parameter] = None
        self._source_trace_ref: Any = None

        # Populated during postprocessing:
        self.num_calls: int = 1  # how many forward ops used this param
        self.used_by_layers: List[str] = []  # layer labels that used this param
        self.co_parent_params: List[str] = []  # other param addresses sharing the same tensor
        self._has_grad: bool = False  # one-shot flag: once True, no further checks
        self._grad_shape: Optional[Tuple[int, ...]] = None
        self._grad_dtype: Optional[torch.dtype] = None
        self._grad_memory: int = 0
        self._grad_memory_str: str = human_readable_size(0)

    @property
    def memory_str(self) -> str:
        """Return parameter size in human-readable units.

        Returns
        -------
        str
            Human-readable parameter memory amount.
        """
        return human_readable_size(self.memory)

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
    def source_trace(self) -> Any:
        """Owning Trace, if still alive."""

        ref = self._source_trace_ref
        return None if ref is None else ref()

    @source_trace.setter
    def source_trace(self, value: Any) -> None:
        """Set the owning Trace weakref."""

        self._source_trace_ref = weakref.ref(value) if value is not None else None

    @property
    def module(self) -> Any:
        """Primary owning ModuleLog."""

        trace = self.source_trace
        if trace is None:
            return None
        return trace.modules[self.module_address]

    @property
    def modules(self) -> list[Any]:
        """All owning ModuleLogs."""

        trace = self.source_trace
        if trace is None:
            return []
        return [trace.modules[address] for address in self.all_module_addresses]

    @property
    def grad(self) -> torch.Tensor | None:
        """Return the live gradient tensor for this parameter.

        Returns
        -------
        torch.Tensor | None
            Live ``nn.Parameter.grad`` value, if available.
        """

        return None if self._param_ref is None else self._param_ref.grad

    def _check_param_grad(self) -> None:
        """Lazily check if the parameter has a grad and cache the result.

        Called by each grad property on first access.  Once a grad is
        found, all grad metadata is cached and no further checks are made
        (``_has_grad`` acts as a one-shot flag).
        """
        if not self._has_grad and self._param_ref is not None and self._param_ref.grad is not None:
            grad = self._param_ref.grad
            self._has_grad = True
            self._grad_shape = tuple(grad.shape)
            self._grad_dtype = grad.dtype
            self._grad_memory = grad.nelement() * grad.element_size()
            self._grad_memory_str = human_readable_size(self._grad_memory)

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
    def grad_memory(self) -> int:
        """Return the grad tensor size in bytes.

        Returns
        -------
        int
            Size of the grad tensor in bytes.
        """
        self._check_param_grad()
        return self._grad_memory

    @grad_memory.setter
    def grad_memory(self, value: int) -> None:
        """Set cached grad tensor size in bytes.

        Parameters
        ----------
        value:
            Cached grad memory amount in bytes.
        """
        self._grad_memory = value

    @property
    def grad_memory_str(self) -> str:
        """Return grad tensor size in human-readable units.

        Returns
        -------
        str
            Human-readable grad memory amount.
        """
        self._check_param_grad()
        return self._grad_memory_str

    @grad_memory_str.setter
    def grad_memory_str(self, value: str) -> None:
        """Set cached grad tensor size in human-readable units.

        Parameters
        ----------
        value:
            Human-readable grad memory amount.
        """
        self._grad_memory_str = value

    def __repr__(self) -> str:
        """Multi-line summary showing address, shape, dtype, trainability, and usage."""
        status = "trainable" if self.trainable else "frozen"
        lines = [
            f"ParamLog: {self.address}",
            f"  shape: {self.shape}",
            f"  dtype: {self.dtype}",
            f"  size: {self.memory_str}",
            f"  {status}",
            f"  has_grad: {self.has_grad}",
            f"  module: {self.address} ({self.module_type})",
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
        self._check_param_grad()
        self._param_ref = None

    def __len__(self) -> int:
        """Return the number of scalar elements in this parameter."""
        return self.num_params

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with live parameter references stripped."""
        state = self.__dict__.copy()
        state["_param_ref"] = None
        state["_source_trace_ref"] = None
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state without reviving live parameter references."""
        read_io_format_version(state, cls_name=type(self).__name__)
        default_fill_state(state, defaults={"_param_ref": None, "_source_trace_ref": None})
        self.__dict__.update(state)


class ParamAccessor(Accessor["ParamLog"]):
    """Dict-like accessor for ParamLog objects.

    Supports indexing by:
    * **full address** (str) -- e.g. ``"features.0.weight"``.
    * **short name** (str) -- e.g. ``"weight"`` (must be unambiguous).
    * **ordinal position** (int) -- index into insertion-order list.

    Available as ``trace.params``, ``layer_log.params``, ``module_log.params``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
    }

    def __init__(self, param_logs: Dict[str, "ParamLog"]) -> None:
        super().__init__(param_logs)

    def _resolve_substring(self, key: str) -> "ParamLog | None":
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
            status = "trainable" if pl.trainable else "frozen"
            items.append(f"'{pl.address}': ParamLog {pl.shape} {pl.dtype} {status}")
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

    def to_csv(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the parameter table to CSV.

        Parameters
        ----------
        filepath:
            Output CSV path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_csv``.
        """
        self.to_pandas().to_csv(filepath, index=False, **kwargs)

    def to_parquet(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the parameter table to Parquet.

        Parameters
        ----------
        filepath:
            Output Parquet path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_parquet``.

        Raises
        ------
        ImportError
            If ``pyarrow`` is unavailable.
        """
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "to_parquet requires pyarrow. Install with: pip install torchlens[io]"
            ) from exc
        from ..export import _parquet_safe_dataframe

        _parquet_safe_dataframe(self.to_pandas()).to_parquet(filepath, **kwargs)

    def to_json(
        self,
        filepath: str | PathLike[str],
        *,
        orient: Literal["split", "records", "index", "columns", "values", "table"] = "records",
        **kwargs: Any,
    ) -> None:
        """Write the parameter table to JSON.

        Parameters
        ----------
        filepath:
            Output JSON path.
        orient:
            JSON orientation passed to ``DataFrame.to_json``.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_json``.
        """
        self.to_pandas().to_json(filepath, orient=orient, **kwargs)
