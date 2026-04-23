"""ParamLog and ParamAccessor: per-parameter metadata and dict-like accessor for model parameters.

ParamLog stores static metadata (address, shape, dtype, trainability) plus
lazy gradient information.  It does NOT store the parameter tensor itself --
only a weak-ish reference (``_param_ref``) used solely for lazy gradient
access via ``_check_param_grad()``.

**GC concern with _param_ref**: ``_param_ref`` holds a direct reference to
the ``nn.Parameter`` object.  This prevents the parameter from being garbage
collected as long as the ParamLog (and thus the ModelLog) is alive.  This
is acceptable because the ModelLog's lifetime is typically shorter than or
equal to the model's lifetime.  The ``cleanup()`` method on ModelLog
deletes all ParamLog references.

**Lazy grad properties**: Gradient metadata (has_grad, grad_shape, grad_dtype,
grad_memory) is computed lazily on first access via ``_check_param_grad()``.
This allows gradients computed after ``log_forward_pass()`` returns (e.g.
after a ``loss.backward()`` call) to be reflected without re-logging.
The check is one-shot: once ``_has_grad`` is True, no further checks are made.
"""

from os import PathLike
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import torch

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..constants import PARAM_LOG_FIELD_ORDER
from ..utils.display import human_readable_size


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
    itself -- only a ``_param_ref`` reference for lazy gradient access.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "address": FieldPolicy.KEEP,
        "name": FieldPolicy.KEEP,
        "shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "memory": FieldPolicy.KEEP,
        "trainable": FieldPolicy.KEEP,
        "module_address": FieldPolicy.KEEP,
        "module_type": FieldPolicy.KEEP,
        "barcode": FieldPolicy.KEEP,
        "has_optimizer": FieldPolicy.KEEP,
        "_param_ref": FieldPolicy.DROP,
        "num_passes": FieldPolicy.KEEP,
        "used_by_layers": FieldPolicy.KEEP,
        "linked_params": FieldPolicy.KEEP,
        "_has_grad": FieldPolicy.KEEP,
        "_grad_shape": FieldPolicy.KEEP,
        "_grad_dtype": FieldPolicy.KEEP,
        "_grad_memory": FieldPolicy.KEEP,
        "_grad_memory_str": FieldPolicy.KEEP,
    }

    def __init__(
        self,
        address: str,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_params: int,
        memory: int,
        trainable: bool,
        module_address: str,
        module_type: str,
        barcode: str,
        has_optimizer: Optional[bool] = None,
    ):
        self.address = address  # e.g. "features.0.weight"
        self.name = name  # short name, e.g. "weight"
        self.shape = shape
        self.dtype = dtype
        self.num_params = num_params
        self.memory = memory
        self.trainable = trainable
        self.module_address = module_address
        self.module_type = module_type
        self.barcode = barcode
        self.has_optimizer = has_optimizer

        # Direct reference to the actual nn.Parameter for lazy gradient access.
        # Prevents GC of the parameter while this ParamLog is alive (acceptable
        # because ModelLog lifetime <= model lifetime; cleanup() clears it).
        self._param_ref: Optional[torch.nn.Parameter] = None

        # Populated during postprocessing:
        self.num_passes: int = 1  # how many forward passes used this param
        self.used_by_layers: List[str] = []  # layer labels that used this param
        self.linked_params: List[str] = []  # other param addresses sharing the same tensor
        self._has_grad: bool = False  # one-shot flag: once True, no further checks
        self._grad_shape: Optional[Tuple[int, ...]] = None
        self._grad_dtype: Optional[torch.dtype] = None
        self._grad_memory: int = 0
        self._grad_memory_str: str = human_readable_size(0)

    @property
    def memory_str(self) -> str:
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

    def _check_param_grad(self):
        """Lazily check if the parameter has a gradient and cache the result.

        Called by each grad property on first access.  Once a gradient is
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
        """Whether this parameter currently has a gradient stored."""
        self._check_param_grad()
        return self._has_grad

    @has_grad.setter
    def has_grad(self, value: bool) -> None:
        self._has_grad = value

    @property
    def grad_shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of the gradient tensor, or None if no gradient exists."""
        self._check_param_grad()
        return self._grad_shape

    @grad_shape.setter
    def grad_shape(self, value: Optional[Tuple[int, ...]]) -> None:
        self._grad_shape = value

    @property
    def grad_dtype(self) -> Optional[torch.dtype]:
        """Dtype of the gradient tensor, or None if no gradient exists."""
        self._check_param_grad()
        return self._grad_dtype

    @grad_dtype.setter
    def grad_dtype(self, value: Optional[torch.dtype]) -> None:
        self._grad_dtype = value

    @property
    def grad_memory(self) -> int:
        """Size of the gradient tensor in bytes."""
        self._check_param_grad()
        return self._grad_memory

    @grad_memory.setter
    def grad_memory(self, value: int) -> None:
        self._grad_memory = value

    @property
    def grad_memory_str(self) -> str:
        """Human-readable size of the gradient tensor (e.g. '4.0 KB')."""
        self._check_param_grad()
        return self._grad_memory_str

    @grad_memory_str.setter
    def grad_memory_str(self, value: str) -> None:
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
            f"  module: {self.module_address} ({self.module_type})",
        ]
        if self.used_by_layers:
            lines.append(f"  used by: {', '.join(self.used_by_layers)}")
        if self.linked_params:
            lines.append(f"  linked: {', '.join(self.linked_params)}")
        if self.has_optimizer is not None:
            lines.append(f"  has_optimizer: {self.has_optimizer}")
        if self.num_passes > 1:
            lines.append(f"  num_passes: {self.num_passes}")
        return "\n".join(lines)

    def release_param_ref(self):
        """Cache gradient info, then null _param_ref to allow param GC."""
        self._check_param_grad()
        self._param_ref = None

    def __len__(self) -> int:
        """Return the number of scalar elements in this parameter."""
        return self.num_params

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with live parameter references stripped."""
        state = self.__dict__.copy()
        state["_param_ref"] = None
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state without reviving live parameter references."""
        read_io_format_version(state, cls_name=type(self).__name__)
        default_fill_state(state, defaults={"_param_ref": None})
        self.__dict__.update(state)


class ParamAccessor:
    """Dict-like accessor for ParamLog objects.

    Supports indexing by:
    * **full address** (str) -- e.g. ``"features.0.weight"``.
    * **short name** (str) -- e.g. ``"weight"`` (must be unambiguous).
    * **ordinal position** (int) -- index into insertion-order list.

    Available as ``model_log.params``, ``layer_log.params``, ``module_log.params``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
    }

    def __init__(self, param_logs: Dict[str, "ParamLog"]) -> None:
        self._dict = param_logs  # address -> ParamLog
        self._list = list(param_logs.values())  # insertion-order list

    def __getitem__(self, key: Union[int, str]) -> "ParamLog":
        """Retrieve a parameter by integer index, full address, or short name (e.g. 'weight')."""
        if isinstance(key, int):
            return self._list[key]
        if key in self._dict:
            return self._dict[key]
        # Fallback: match by short name (e.g. 'weight', 'bias')
        matches = [pl for pl in self._list if pl.name == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"Ambiguous short name '{key}' — use full address")
        raise KeyError(key)

    def __contains__(self, key) -> bool:
        """Check membership by full address, short name, or integer index (#84)."""
        if isinstance(key, int):
            return 0 <= key < len(self._list)
        if isinstance(key, str):
            if key in self._dict:
                return True
            # Also check short name match
            return any(pl.name == key for pl in self._list)
        return False

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._dict)

    def __iter__(self):
        """Iterate over ParamLog objects in insertion order."""
        return iter(self._list)

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
        self.to_pandas().to_parquet(filepath, **kwargs)

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
