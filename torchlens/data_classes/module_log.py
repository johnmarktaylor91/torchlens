"""Structured per-module metadata: ModuleCallLog, ModuleLog, ModuleAccessor.

Three-level hierarchy:

* **ModuleCallLog** -- one invocation of one module (lightweight container).
  Stores ``layers`` as pass-qualified labels (e.g. ``"conv2d_1_1:1"``),
  pointing to individual OpLog entries.

* **ModuleLog** -- aggregate metadata for one ``nn.Module`` across all its
  invocations.  Stores ``layers`` as no-pass labels (e.g.
  ``"conv2d_1_1"``), pointing to aggregate LayerLog entries.
  For single-pass modules, per-pass fields are delegated to ``ops[1]``
  via ``_single_pass_or_error()``.

* **ModuleAccessor** -- dict-like accessor returned by ``trace.modules``.
  Supports lookup by module address, alias (shared modules), pass label,
  or ordinal index.

ModuleLog vs ModuleCallLog label convention:
  - ModuleLog.layers stores **no-pass** labels -> LayerLog
  - ModuleCallLog.layers stores **pass-qualified** labels -> OpLog
This matches each accessor's natural granularity.
"""

import weakref
from collections.abc import Iterator
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast

import torch

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..constants import MODULE_PASS_LOG_FIELD_ORDER
from ..utils.display import human_readable_size

if TYPE_CHECKING:
    import pandas as pd

    from .buffer_log import BufferAccessor
    from .model_log import Trace
    from .param_log import ParamAccessor


def _module_call_log_to_row(module_call_log: "ModuleCallLog") -> Dict[str, Any]:
    """Convert a ModuleCallLog into one DataFrame row.

    Parameters
    ----------
    module_call_log:
        Module-pass metadata entry to export.

    Returns
    -------
    Dict[str, Any]
        Mapping from canonical field name to exported value.
    """
    return {field: getattr(module_call_log, field) for field in MODULE_PASS_LOG_FIELD_ORDER}


class ModuleCallLog:
    """Per-(module, call_index) data for one invocation of a module.

    Lightweight container holding the list of layers computed during
    this particular invocation, the captured forward arguments, and
    the call-graph edges (parent/children in the module invocation tree).

    ``layers`` stores **pass-qualified** labels (e.g. ``"conv2d_1_1:1"``)
    that resolve to individual OpLog entries.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "address": FieldPolicy.KEEP,
        "call_index": FieldPolicy.KEEP,
        "call_label": FieldPolicy.KEEP,
        "layers": FieldPolicy.KEEP,
        "input_layers": FieldPolicy.KEEP,
        "output_layers": FieldPolicy.KEEP,
        "forward_args": FieldPolicy.BLOB_RECURSIVE,
        "forward_kwargs": FieldPolicy.BLOB_RECURSIVE,
        "forward_args_summary": FieldPolicy.KEEP,
        "forward_kwargs_summary": FieldPolicy.KEEP,
        "call_parent": FieldPolicy.KEEP,
        "call_children": FieldPolicy.KEEP,
        "all_addresses": FieldPolicy.KEEP,
    }

    def __init__(
        self,
        address: str,
        call_index: int,
        call_label: str,
        layers: List[str],
        input_layers: List[str],
        output_layers: List[str],
        forward_args: tuple[Any, ...] | None = None,
        forward_kwargs: dict[str, Any] | None = None,
        call_parent: Optional[str] = None,
        call_children: Optional[List[str]] = None,
        all_addresses: Optional[List[str]] = None,
    ) -> None:
        self.address = address
        self.call_index = call_index
        self.call_label = call_label  # e.g. "features.0:1"
        self.layers = layers  # pass-qualified layer labels
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.forward_args_summary = ""
        self.forward_kwargs_summary = ""
        self.call_parent = call_parent
        self.call_children = call_children if call_children is not None else []
        self.all_addresses = all_addresses if all_addresses is not None else [address]

    @property
    def num_layers(self) -> int:
        """Number of layers in this pass."""
        return len(self.layers)

    @property
    def is_shared(self) -> bool:
        """Whether this module appears at multiple addresses."""
        return len(self.all_addresses) > 1

    @property
    def inputs(self) -> List[str]:
        """Module-pass input layer labels."""

        return self.input_layers

    @property
    def outputs(self) -> List[str]:
        """Module-pass output layer labels."""

        return self.output_layers

    def __repr__(self) -> str:
        """Show pass label, layer count, and children."""
        lines = [
            f"ModuleCallLog: {self.call_label}",
            f"  layers: {self.num_layers}",
        ]
        if self.input_layers:
            lines.append(f"  input_layers: {self.input_layers}")
        if self.output_layers:
            lines.append(f"  output_layers: {self.output_layers}")
        if self.call_children:
            lines.append(f"  call_children: {self.call_children}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the number of layers in this pass."""
        return self.num_layers

    def to_pandas(self) -> "pd.DataFrame":
        """Export this module pass as a one-row pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame ordered by ``MODULE_PASS_LOG_FIELD_ORDER``.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        row = _module_call_log_to_row(self)
        return pd.DataFrame([row], columns=MODULE_PASS_LOG_FIELD_ORDER)

    def to_csv(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write this module-pass row to CSV.

        Parameters
        ----------
        filepath:
            Output CSV path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_csv``.
        """
        self.to_pandas().to_csv(filepath, index=False, **kwargs)

    def to_parquet(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write this module-pass row to Parquet.

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
        """Write this module-pass row to JSON.

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

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state annotated with the current I/O format version."""
        state = self.__dict__.copy()
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state with version-aware default filling."""
        read_io_format_version(state, cls_name=type(self).__name__)
        default_fill_state(
            state,
            defaults={
                "all_addresses": [state["address"]],
                "forward_args_summary": "",
                "forward_kwargs_summary": "",
            },
        )
        self.__dict__.update(state)


class ModuleLog:
    """Aggregate metadata for one nn.Module across all its invocations.

    The primary user-facing class for module inspection.  Provides both
    static metadata (source file, class name, hierarchy) and dynamic
    data (layers computed, parameter usage, pass-level detail).

    ``layers`` stores **no-pass** labels (e.g. ``"conv2d_1_1"``),
    pointing to aggregate LayerLog entries.  For per-pass detail, access
    ``self.ops[call_index].layers`` which stores pass-qualified labels.

    For single-pass modules, per-pass fields (``layers``, ``input_layers``,
    ``output_layers``, ``forward_args``, ``forward_kwargs``) are accessible
    directly via ``_single_pass_or_error()`` delegation to ``ops[1]``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "address": FieldPolicy.KEEP,
        "all_addresses": FieldPolicy.KEEP,
        "name": FieldPolicy.KEEP,
        "class_name": FieldPolicy.KEEP,
        "source_file": FieldPolicy.KEEP,
        "source_line": FieldPolicy.KEEP,
        "class_docstring": FieldPolicy.KEEP,
        "init_signature": FieldPolicy.KEEP,
        "init_docstring": FieldPolicy.KEEP,
        "forward_signature": FieldPolicy.KEEP,
        "forward_docstring": FieldPolicy.KEEP,
        "address_parent": FieldPolicy.KEEP,
        "address_children": FieldPolicy.KEEP,
        "address_depth": FieldPolicy.KEEP,
        "call_parent": FieldPolicy.KEEP,
        "call_children": FieldPolicy.KEEP,
        "call_depth": FieldPolicy.KEEP,
        "num_calls": FieldPolicy.KEEP,
        "ops": FieldPolicy.KEEP,
        "call_labels": FieldPolicy.KEEP,
        "layers": FieldPolicy.KEEP,
        "params": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "param_memory": FieldPolicy.KEEP,
        "has_trainable_params": FieldPolicy.KEEP,
        "buffer_layers": FieldPolicy.KEEP,
        "_buffer_accessor": FieldPolicy.DROP,
        "is_train_mode": FieldPolicy.KEEP,
        "has_forward_hooks": FieldPolicy.KEEP,
        "has_backward_hooks": FieldPolicy.KEEP,
        "custom_attributes": FieldPolicy.BLOB_RECURSIVE,
        "custom_methods": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(
        self,
        # Identity
        address: str,
        all_addresses: Optional[List[str]] = None,
        name: str = "",
        class_name: str = "",
        # Source info
        source_file: Optional[str] = None,
        source_line: Optional[int] = None,
        class_docstring: Optional[str] = None,
        init_signature: Optional[str] = None,
        init_docstring: Optional[str] = None,
        forward_signature: Optional[str] = None,
        forward_docstring: Optional[str] = None,
        # Hierarchy — address-based (static)
        address_parent: Optional[str] = None,
        address_children: Optional[List[str]] = None,
        address_depth: int = 0,
        # Hierarchy — call-based (dynamic)
        call_parent: Optional[str] = None,
        call_children: Optional[List[str]] = None,
        call_depth: int = 0,
        # Pass info
        num_calls: int = 1,
        ops: Optional[Dict[int, "ModuleCallLog"]] = None,
        call_labels: Optional[List[str]] = None,
        # Layers (aggregate)
        layers: Optional[List[str]] = None,
        # Parameters
        params: Optional["ParamAccessor"] = None,
        num_params: int = 0,
        num_params_trainable: int = 0,
        num_params_frozen: int = 0,
        param_memory: int = 0,
        has_trainable_params: bool = False,
        # Buffers
        buffer_layers: Optional[List[str]] = None,
        # Module state
        is_train_mode: bool = True,
        has_forward_hooks: bool = False,
        has_backward_hooks: bool = False,
        custom_attributes: Optional[Dict[str, Any]] = None,
        custom_methods: Optional[List[str]] = None,
        # Back-reference
        _source_trace: "Trace | None" = None,
    ) -> None:
        self.address = address
        self.all_addresses = all_addresses if all_addresses is not None else [address]
        self.name = name
        self.class_name = class_name

        self.source_file = source_file
        self.source_line = source_line
        self.class_docstring = class_docstring
        self.init_signature = init_signature
        self.init_docstring = init_docstring
        self.forward_signature = forward_signature
        self.forward_docstring = forward_docstring

        self.address_parent = address_parent
        self.address_children = address_children if address_children is not None else []
        self.address_depth = address_depth

        self.call_parent = call_parent
        self.call_children = call_children if call_children is not None else []
        self.call_depth = call_depth

        self.num_calls = num_calls
        self.ops = ops if ops is not None else {}
        self.call_labels = call_labels if call_labels is not None else []

        # layers stores NO-PASS labels (e.g. "conv2d_1_1") -> LayerLog.
        # Contrast with ModuleCallLog.layers which stores pass-qualified labels.
        self.layers = layers if layers is not None else []

        from .param_log import ParamAccessor

        self.params = params if params is not None else ParamAccessor({})
        self.num_params = num_params
        self.num_params_trainable = num_params_trainable
        self.num_params_frozen = num_params_frozen
        self.param_memory = param_memory
        self.has_trainable_params = has_trainable_params

        self.buffer_layers = buffer_layers if buffer_layers is not None else []
        self._buffer_accessor: Any = None  # populated by _build_module_logs

        self.is_train_mode = is_train_mode
        self.has_forward_hooks = has_forward_hooks
        self.has_backward_hooks = has_backward_hooks
        self.custom_attributes = custom_attributes if custom_attributes is not None else {}
        self.custom_methods = custom_methods if custom_methods is not None else []

        # Store as weakref to break circular reference (Trace -> _module_logs -> ModuleLog -> Trace).
        self._source_trace_ref = weakref.ref(_source_trace) if _source_trace is not None else None

    @property
    def is_shared(self) -> bool:
        """Whether this module appears at multiple addresses."""
        return len(self.all_addresses) > 1

    @property
    def num_layers(self) -> int:
        """Number of unique layers in this module."""
        return len(self.layers)

    @property
    def param_memory_str(self) -> str:
        """Return parameter tensor size in human-readable units.

        Returns
        -------
        str
            Human-readable parameter memory amount.
        """
        return human_readable_size(self.param_memory)

    @property
    def _source_trace(self) -> "Trace | None":
        """Back-reference to the owning Trace (stored as weakref)."""
        ref = self.__dict__.get("_source_trace_ref")
        if ref is None:
            return None
        return cast("Trace | None", ref())

    @_source_trace.setter
    def _source_trace(self, value: "Trace | None") -> None:
        self._source_trace_ref = weakref.ref(value) if value is not None else None

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with weakrefs stripped."""
        state = self.__dict__.copy()
        state["_source_trace_ref"] = None
        state["_buffer_accessor"] = None
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state without touching disk."""
        read_io_format_version(state, cls_name=type(self).__name__)
        default_fill_state(state, defaults={"_buffer_accessor": None, "_source_trace_ref": None})
        self.__dict__.update(state)

    # --- Per-call delegating properties ---
    # Mirror the LayerLog delegation pattern: single-pass modules transparently
    # expose per-pass fields; multi-pass modules raise with guidance.

    def _single_pass_or_error(self, field_name: str) -> Any:
        """Return a field from the single pass, or raise if the module has multiple ops.

        For modules invoked once, transparently delegates to ops[1].
        For multi-pass modules, raises AttributeError directing the user to
        access the field on a specific pass.
        """
        if self.num_calls > 1:
            raise AttributeError(
                f"Module '{self.address}' has {self.num_calls} ops. "
                f"Access '{field_name}' on a specific pass: "
                f"module.ops[1].{field_name}, module.ops[2].{field_name}, etc."
            )
        if 1 not in self.ops:
            return None
        return getattr(self.ops[1], field_name)

    @property
    def input_layers(self) -> List[str]:
        """Aggregate module input layer labels.

        Returns
        -------
        List[str]
            No-pass input layer labels across module calls.
        """
        labels: list[str] = []
        for call in self.ops.values():
            labels.extend(label.split(":", 1)[0] for label in call.input_layers)
        return list(dict.fromkeys(labels))

    @property
    def output_layers(self) -> List[str]:
        """Aggregate module output layer labels.

        Returns
        -------
        List[str]
            No-pass output layer labels across module calls.
        """
        labels: list[str] = []
        for call in self.ops.values():
            labels.extend(label.split(":", 1)[0] for label in call.output_layers)
        return list(dict.fromkeys(labels))

    @property
    def forward_args(self) -> tuple[Any, ...] | None:
        """Return captured forward positional arguments for one module pass.

        Returns
        -------
        tuple[Any, ...] | None
            Captured positional arguments, or ``None`` when unavailable.
        """
        return cast("tuple[Any, ...] | None", self._single_pass_or_error("forward_args"))

    @property
    def forward_kwargs(self) -> dict[str, Any] | None:
        """Return captured forward keyword arguments for one module pass.

        Returns
        -------
        dict[str, Any] | None
            Captured keyword arguments, or ``None`` when unavailable.
        """
        return cast("dict[str, Any] | None", self._single_pass_or_error("forward_kwargs"))

    @property
    def buffers(self) -> "BufferAccessor":
        """Scoped BufferAccessor for buffers belonging to this module."""
        if self._buffer_accessor is not None:
            return cast("BufferAccessor", self._buffer_accessor)
        # Build on first access from the source Trace
        from .buffer_log import BufferAccessor

        if self._source_trace is None or self._source_trace._buffer_accessor is None:
            self._buffer_accessor = BufferAccessor({})
            return cast("BufferAccessor", self._buffer_accessor)
        parent_accessor = self._source_trace._buffer_accessor
        scoped = {
            addr: bl for addr, bl in parent_accessor._dict.items() if bl.address == self.address
        }
        self._buffer_accessor = BufferAccessor(scoped, source_trace=self._source_trace)
        return cast("BufferAccessor", self._buffer_accessor)

    def _sum_layer_field(self, field: str) -> int:
        """Sum a numeric field across all layers in this module (skipping None)."""
        if self._source_trace is None:
            return 0
        total = 0
        for label in self.layers:
            entry = self._source_trace[label]
            val = getattr(entry, field, None)
            if val is not None:
                total += val
        return total

    @property
    def flops_forward(self) -> int:
        """Total forward FLOPs across all layers in this module."""
        return self._sum_layer_field("flops_forward")

    @property
    def flops_backward(self) -> int:
        """Total backward FLOPs across all layers in this module."""
        return self._sum_layer_field("flops_backward")

    @property
    def flops(self) -> int:
        """Total FLOPs (forward + backward) for this module."""
        return self.flops_forward + self.flops_backward

    @property
    def macs_forward(self) -> int:
        """Total forward MACs for this module. 1 MAC = 2 FLOPs."""
        return self.flops_forward // 2

    @property
    def macs_backward(self) -> int:
        """Total backward MACs for this module. 1 MAC = 2 FLOPs."""
        return self.flops_backward // 2

    @property
    def macs(self) -> int:
        """Total MACs (forward + backward) for this module."""
        return self.flops // 2

    @property
    def grad(self) -> torch.Tensor | List[torch.Tensor] | None:
        """Aggregate saved grads across layers in this module.

        Returns
        -------
        torch.Tensor | List[torch.Tensor] | None
            A stacked tensor when layer grad shapes match, a list when
            shapes differ, or ``None`` when no layer grads were saved.
        """
        if self._source_trace is None:
            return None
        grads = [
            self._source_trace[layer_label].grad
            for layer_label in self.layers
            if getattr(self._source_trace[layer_label], "has_grad", False)
        ]
        if not grads:
            return None
        first_shape = grads[0].shape
        if all(grad.shape == first_shape for grad in grads):
            return torch.stack(grads)
        return grads

    def __repr__(self) -> str:
        """Show address, class, depth, param count, layer count, and pass count."""
        lines = [
            f"ModuleLog: {self.address} ({self.class_name})",
            f"  call_depth: {self.call_depth}, address_depth: {self.address_depth}",
            f"  num_params: {self.num_params}",
            f"  num_layers: {self.num_layers}",
            f"  num_calls: {self.num_calls}",
        ]
        if self.is_shared:
            lines.append(f"  aliases: {self.all_addresses}")
        if self.address_children:
            lines.append(f"  children: {self.address_children}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the total number of layers across all ops of this module."""
        return self.num_layers

    def __getitem__(self, ix: int | str) -> Any:
        """Return the OpLog at position ix within this module's layer list.

        Supports int indexing and string label lookup (#120).
        """
        if self._source_trace is None:
            raise RuntimeError("No source Trace reference; cannot index into layers.")
        if isinstance(ix, str):
            # String label lookup within this module's layers
            if ix in self.layers:
                return self._source_trace[ix]
            # Try substring match within module layers
            matches = [lbl for lbl in self.layers if ix in lbl]
            if len(matches) == 1:
                return self._source_trace[matches[0]]
            elif len(matches) > 1:
                raise ValueError(
                    f"Ambiguous lookup: '{ix}' matches {len(matches)} layers in module "
                    f"'{self.address}': {', '.join(matches[:5])}"
                )
            raise KeyError(f"'{ix}' not found in module '{self.address}' layers")
        return self._source_trace[self.layers[ix]]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over OpLog entries for all layers in this module."""
        if self._source_trace is None:
            return iter(self.layers)
        return iter(self._source_trace[label] for label in self.layers)

    def show_graph(self, **kwargs: Any) -> str:
        """Render this module's focused graph.

        Parameters
        ----------
        **kwargs:
            Keyword arguments forwarded to ``Trace.render_graph``.

        Returns
        -------
        str
            Graphviz DOT source string.

        Raises
        ------
        RuntimeError
            If this ModuleLog is not bound to a Trace.
        """

        trace: Trace | None = self._source_trace
        if trace is None:
            raise RuntimeError("ModuleLog not bound to a Trace")
        return cast(str, trace.render_graph(module=self, **kwargs))

    def to_pandas(self) -> "pd.DataFrame":
        """Export this module's layers as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per layer belonging to this module.
        """
        if self._source_trace is None:
            raise RuntimeError("No source Trace reference; cannot build DataFrame.")
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = []
        for label in self.layers:
            entry = self._source_trace[label]
            rows.append(
                {
                    "layer_label": entry.layer_label,
                    "layer_type": entry.layer_type,
                    "shape": entry.shape,
                    "dtype": entry.dtype,
                    "call_index": entry.call_index,
                    "func_name": entry.func_name,
                }
            )
        return pd.DataFrame(rows)

    def to_csv(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write this module's layer table to CSV.

        Parameters
        ----------
        filepath:
            Output CSV path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_csv``.
        """
        self.to_pandas().to_csv(filepath, index=False, **kwargs)

    def to_parquet(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write this module's layer table to Parquet.

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
        """Write this module's layer table to JSON.

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


class ModuleAccessor:
    """Dict-like accessor for ModuleLog objects.

    Supports indexing by:
    * **module address** (str) -- e.g. ``"features.0"``, ``"self"``.
    * **alias** (str) -- for shared modules (same nn.Module registered at
      multiple addresses), any alias resolves to the same ModuleLog.
    * **pass label** (str) -- e.g. ``"features.0:2"`` returns a ModuleCallLog.
    * **ordinal index** (int) -- position in address order.

    Available as ``trace.modules``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_pass_dict": FieldPolicy.KEEP,
        "_alias_dict": FieldPolicy.KEEP,
    }

    def __init__(
        self,
        module_dict: Dict[str, "ModuleLog"],
        module_list: Optional[List["ModuleLog"]] = None,
        pass_dict: Optional[Dict[str, "ModuleCallLog"]] = None,
    ) -> None:
        self._dict = module_dict  # primary address -> ModuleLog
        self._list = module_list if module_list is not None else list(module_dict.values())
        self._pass_dict = pass_dict if pass_dict is not None else {}  # pass label -> ModuleCallLog
        # Alias map: for shared modules (same nn.Module at multiple addresses),
        # every non-primary address also resolves to the same ModuleLog.
        self._alias_dict: Dict[str, "ModuleLog"] = {}
        for ml in self._dict.values():
            for alias in ml.all_addresses:
                if alias not in self._dict:
                    self._alias_dict[alias] = ml

    def __getitem__(self, key: Union[int, str]) -> Union["ModuleLog", "ModuleCallLog"]:
        """Return a ModuleLog by address string or ordinal index, or a ModuleCallLog by pass label.

        Shared modules (same nn.Module registered under multiple addresses) can
        be looked up by any of their aliases.
        """
        if isinstance(key, int):
            return self._list[key]
        if key == "":
            key = "self"
        if key in self._dict:
            return self._dict[key]
        if key in self._alias_dict:
            return self._alias_dict[key]
        if key in self._pass_dict:
            return self._pass_dict[key]
        raise KeyError(f"Module '{key}' not found.")

    def __contains__(self, key: object) -> bool:
        """Return True if key is a known module address, alias, or pass label."""
        if key == "":
            key = "self"
        return key in self._dict or key in self._alias_dict or key in self._pass_dict

    def __len__(self) -> int:
        """Return the number of modules in this accessor."""
        return len(self._dict)

    def __dir__(self) -> List[str]:
        """Return Python attributes plus module addresses for tab completion.

        Returns
        -------
        List[str]
            Attribute names and valid module addresses.
        """

        keys = set(self._dict.keys()) | set(self._alias_dict.keys()) | set(self._pass_dict.keys())
        return sorted(set(super().__dir__()) | keys)

    def _ipython_key_completions_(self) -> List[str]:
        """Return module addresses for IPython ``obj[...]`` completion.

        Returns
        -------
        List[str]
            Valid module addresses and pass labels.
        """

        return (
            list(self._dict.keys()) + list(self._alias_dict.keys()) + list(self._pass_dict.keys())
        )

    def __iter__(self) -> Iterator["ModuleLog"]:
        """Iterate over ModuleLog objects in address order."""
        return iter(self._list)

    def __repr__(self) -> str:
        """Show a table of all modules with class, depth, param, layer, and pass counts."""
        if len(self) == 0:
            return "ModuleAccessor({})"
        items = []
        for ml in self._list:
            items.append(
                f"  '{ml.address}': {ml.class_name} "
                f"(depth={ml.call_depth}, params={ml.num_params}, "
                f"layers={ml.num_layers}, ops={ml.num_calls})"
            )
        inner = "\n".join(items)
        return f"ModuleAccessor({len(self)} modules):\n{inner}"

    def to_pandas(self) -> "pd.DataFrame":
        """Export module metadata as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per module in address order.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = []
        for ml in self._list:
            rows.append(
                {
                    "address": ml.address,
                    "class_name": ml.class_name,
                    "call_depth": ml.call_depth,
                    "address_depth": ml.address_depth,
                    "num_params": ml.num_params,
                    "num_layers": ml.num_layers,
                    "num_calls": ml.num_calls,
                }
            )
        return pd.DataFrame(rows)

    def to_csv(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the module table to CSV.

        Parameters
        ----------
        filepath:
            Output CSV path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_csv``.
        """
        self.to_pandas().to_csv(filepath, index=False, **kwargs)

    def to_parquet(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the module table to Parquet.

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
        """Write the module table to JSON.

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

    def summary(self) -> str:
        """Return a compact text table of all modules.

        Returns
        -------
        str
            Summary table with module address, class, depth, params, layers, and ops.
        """
        if len(self) == 0:
            return "No modules."
        lines = [
            f"{'Address':<40} {'Class':<20} {'Depth':>5} {'Params':>10} {'Layers':>7} {'Passes':>7}"
        ]
        lines.append("-" * 92)
        for ml in self._list:
            lines.append(
                f"{ml.address:<40} {ml.class_name:<20} "
                f"{ml.call_depth:>5} {ml.num_params:>10} "
                f"{ml.num_layers:>7} {ml.num_calls:>7}"
            )
        return "\n".join(lines)
