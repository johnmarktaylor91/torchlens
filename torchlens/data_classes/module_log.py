"""Structured per-module metadata: ModuleCall, Module, ModuleAccessor.

Three-level hierarchy:

* **ModuleCall** -- one invocation of one module (lightweight container).
  Stores ``ops`` as pass-qualified labels (e.g. ``"conv2d_1_1:1"``),
  pointing to individual Op entries.

* **Module** -- aggregate metadata for one ``nn.Module`` across all its
  invocations.  Stores ``layer_labels`` as no-pass labels (e.g.
  ``"conv2d_1_1"``), pointing to aggregate Layer entries.
  For single-pass modules, per-pass fields are delegated to ``ops[0]``
  via ``_single_pass_or_error()``.

* **ModuleAccessor** -- dict-like accessor returned by ``trace.modules``.
  Supports lookup by module address, alias (shared modules), pass label,
  or ordinal index.

Module vs ModuleCall label convention:
  - Module.layer_labels stores **no-pass** labels -> Layer
  - ModuleCall.ops stores **pass-qualified** labels -> Op
This matches each accessor's natural granularity.
"""

import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union, cast

import torch

from .._io import FieldPolicy, TLSPEC_VERSION, default_fill_state, read_tlspec_version
from ..constants import MODULE_PASS_LOG_FIELD_ORDER
from ..utils.display import human_readable_size
from ._accessor_base import Accessor

if TYPE_CHECKING:
    import pandas as pd

    from .buffer_log import BufferAccessor
    from .func_call_location import FuncCallLocation
    from .model_log import Trace
    from .op_log import Op
    from .param_log import ParamAccessor
    from ..intervention.types import ContainerSpec


class ModuleCallAccessor(Accessor["ModuleCall"]):
    """Scoped dict-like accessor for ModuleCall entries owned by a Module."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(self, calls: Dict[int, "ModuleCall"] | None = None) -> None:
        """Initialize the accessor.

        Parameters
        ----------
        calls:
            Mapping from 1-based call index to ModuleCall.
        """

        calls = calls or {}
        super().__init__(calls, item_list=[call for _, call in sorted(calls.items())])

    def __getitem__(self, key: int | str) -> "ModuleCall":
        """Return a ModuleCall by 0-based position or call label."""

        if isinstance(key, int):
            return self._list[key]
        resolved = self._resolve_substring(key)
        if resolved is not None:
            return resolved
        raise KeyError(f"Module call '{key}' not found in scoped calls.")

    def __setitem__(self, key: int, value: "ModuleCall") -> None:
        """Set a ModuleCall by 1-based call index."""

        self._dict[key] = value
        self._list = [call for _, call in sorted(self._dict.items())]

    def __contains__(self, key: object) -> bool:
        """Return whether key resolves to a ModuleCall."""

        if isinstance(key, int):
            return -len(self._list) <= key < len(self._list)
        if isinstance(key, str):
            try:
                self[key]
            except (KeyError, ValueError):
                return False
            return True
        return False

    # This scoped accessor intentionally iterates call-index keys, unlike the generic
    # Trace-level accessors that iterate log values.
    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        """Iterate call-index keys."""

        return iter(self._dict)

    def get(self, key: int, default: "ModuleCall | None" = None) -> "ModuleCall | None":
        """Return a ModuleCall by call index, or default."""

        return self._dict.get(key, default)

    def _resolve_substring(self, key: str) -> "ModuleCall | None":
        """Resolve a ModuleCall by exact call label or unique parent address."""
        if len(self._dict) == 1:
            only_call = next(iter(self._dict.values()))
            if key == only_call.address:
                return only_call
        for call in self._dict.values():
            if key == call.call_label:
                return call
        parent_matches = [call for call in self._dict.values() if key == call.address]
        if len(parent_matches) > 1:
            raise ValueError(
                f"Module '{key}' has {len(parent_matches)} calls. Use a 0-based integer "
                f"position or a call-qualified label like '{key}:1'."
            )
        return None


@dataclass
class HookInfo:
    """Metadata for one PyTorch module hook."""

    name: str = ""
    qualname: str = ""
    source_location: "FuncCallLocation | None" = None


def _duration_str(duration: float) -> str:
    """Return a human-readable duration string.

    Parameters
    ----------
    duration:
        Duration in seconds.

    Returns
    -------
    str
        Duration formatted in milliseconds.
    """

    return f"{duration * 1000:.3f} ms"


def _module_call_log_to_row(module_call_log: "ModuleCall") -> Dict[str, Any]:
    """Convert a ModuleCall into one DataFrame row.

    Parameters
    ----------
    module_call_log:
        Module-pass metadata entry to export.

    Returns
    -------
    Dict[str, Any]
        Mapping from canonical field name to exported value.
    """
    row = {field: getattr(module_call_log, field) for field in MODULE_PASS_LOG_FIELD_ORDER}
    row["output_ops"] = list(module_call_log.output_ops)
    row["output_structure"] = (
        repr(module_call_log.output_structure)
        if module_call_log.output_structure is not None
        else None
    )
    return row


@dataclass(init=False)
class ModuleCall:
    """Per-(module, call_index) data for one invocation of a module.

    Lightweight container holding the list of layers computed during
    this particular invocation, the captured forward arguments, and
    the call-graph edges (parent/children in the module invocation tree).

    ``ops`` stores **pass-qualified** labels (e.g. ``"conv2d_1_1:1"``)
    that resolve to individual Op entries.
    """

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "address": FieldPolicy.KEEP,
        "all_addresses": FieldPolicy.KEEP,
        "class_name": FieldPolicy.KEEP,
        "class_qualname": FieldPolicy.KEEP,
        "cls": FieldPolicy.DROP,
        "call_index": FieldPolicy.KEEP,
        "call_label": FieldPolicy.KEEP,
        "ordinal_index": FieldPolicy.KEEP,
        "ops": FieldPolicy.KEEP,
        "input_ops": FieldPolicy.KEEP,
        "input_layers": FieldPolicy.KEEP,
        "output_ops": FieldPolicy.KEEP,
        "output_layers": FieldPolicy.KEEP,
        "output_structure": FieldPolicy.KEEP,
        "forward_args": FieldPolicy.BLOB_RECURSIVE,
        "forward_kwargs": FieldPolicy.BLOB_RECURSIVE,
        "forward_arg_names": FieldPolicy.KEEP,
        "num_forward_args_total": FieldPolicy.KEEP,
        "num_forward_pos_args": FieldPolicy.KEEP,
        "num_forward_kwargs": FieldPolicy.KEEP,
        "forward_args_summary": FieldPolicy.KEEP,
        "forward_kwargs_summary": FieldPolicy.KEEP,
        "forward_duration": FieldPolicy.KEEP,
        "code_context": FieldPolicy.KEEP,
        "module_call_stack": FieldPolicy.KEEP,
        "call_parent": FieldPolicy.KEEP,
        "call_children": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
    }

    address: str
    all_addresses: List[str]
    cls: type[Any] | None
    class_name: str
    class_qualname: str
    call_index: int
    call_label: str
    ordinal_index: int
    ops: List[str]
    input_ops: List[str]
    input_layers: List[str]
    output_ops: List[str]
    output_layers: List[str]
    output_structure: "ContainerSpec | None"
    forward_args: tuple[Any, ...] | None
    forward_kwargs: dict[str, Any] | None
    forward_arg_names: List[str]
    num_forward_args_total: int
    num_forward_pos_args: int
    num_forward_kwargs: int
    forward_args_summary: str
    forward_kwargs_summary: str
    forward_duration: float
    code_context: List["FuncCallLocation"]
    module_call_stack: List[str]
    call_parent: Optional[str]
    call_children: List[str]

    def __init__(
        self,
        address: str,
        call_index: int,
        call_label: str,
        ops: List[str],
        input_layers: List[str],
        output_layers: List[str],
        output_ops: List[str] | None = None,
        output_structure: "ContainerSpec | None" = None,
        forward_args: tuple[Any, ...] | None = None,
        forward_kwargs: dict[str, Any] | None = None,
        forward_arg_names: List[str] | None = None,
        forward_duration: float = 0.0,
        code_context: List["FuncCallLocation"] | None = None,
        module_call_stack: List[str] | None = None,
        call_parent: Optional[str] = None,
        call_children: Optional[List[str]] = None,
        all_addresses: Optional[List[str]] = None,
        cls: type[Any] | None = None,
        class_name: str = "",
        class_qualname: str = "",
        ordinal_index: int = 0,
        _source_trace: "Trace | None" = None,
    ) -> None:
        self.address = address
        self.all_addresses = all_addresses if all_addresses is not None else [address]
        self.cls = cls
        self.class_name = class_name
        self.class_qualname = class_qualname
        self.call_index = call_index
        self.call_label = call_label  # e.g. "features.0:1"
        self.ordinal_index = ordinal_index
        self.ops = ops  # pass-qualified Op labels
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.input_ops = list(input_layers)
        self.output_ops = output_ops if output_ops is not None else list(output_layers)
        self.output_structure = output_structure
        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.forward_arg_names = forward_arg_names if forward_arg_names is not None else []
        self.num_forward_pos_args = len(forward_args) if forward_args is not None else 0
        self.num_forward_kwargs = len(forward_kwargs) if forward_kwargs is not None else 0
        self.num_forward_args_total = self.num_forward_pos_args + self.num_forward_kwargs
        self.forward_args_summary = ""
        self.forward_kwargs_summary = ""
        self.forward_duration = forward_duration
        self.code_context = code_context if code_context is not None else []
        self.module_call_stack = module_call_stack if module_call_stack is not None else []
        self.call_parent = call_parent
        self.call_children = call_children if call_children is not None else []
        self._source_trace_ref = weakref.ref(_source_trace) if _source_trace is not None else None

    @property
    def module(self) -> "Module":
        """Return the Module record invoked by this call."""

        trace = self._source_trace
        if trace is None:
            raise RuntimeError("ModuleCall not bound to a Trace")
        return cast("Module", trace.modules[self.address])

    @property
    def num_layers(self) -> int:
        """Number of layers in this pass."""
        return len(self.ops)

    @property
    def num_ops(self) -> int:
        """Number of Ops in this module call."""

        return len(self.ops)

    @property
    def name(self) -> str:
        """Return the final segment of the primary module-call address.

        Returns
        -------
        str
            Module name within its parent module.
        """

        return "" if self.address == "self" else self.address.rsplit(".", 1)[-1]

    @property
    def has_multiple_addresses(self) -> bool:
        """Whether this module appears at multiple addresses."""
        return len(self.all_addresses) > 1

    @property
    def has_saved_forward_args(self) -> bool:
        """Whether forward arguments were captured for this module call."""

        return self.forward_args is not None or self.forward_kwargs is not None

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors owned by the called Module."""

        return self.module.num_param_tensors

    @property
    def num_param_tensors_trainable(self) -> int:
        """Number of trainable parameter tensors owned by the called Module."""

        return self.module.num_param_tensors_trainable

    @property
    def num_param_tensors_frozen(self) -> int:
        """Number of frozen parameter tensors owned by the called Module."""

        return self.module.num_param_tensors_frozen

    @property
    def has_trainable_params(self) -> bool:
        """Whether the called Module owns trainable parameters."""

        return self.module.has_trainable_params

    @property
    def has_frozen_params(self) -> bool:
        """Whether the called Module owns frozen parameters."""

        return self.module.has_frozen_params

    @property
    def func_calls_duration(self) -> float:
        """Inclusive duration of torch function calls inside this ModuleCall."""

        trace = self._source_trace
        if trace is None:
            return 0.0
        return sum(getattr(trace.ops[label], "func_duration", 0.0) or 0.0 for label in self.ops)

    @property
    def func_calls_duration_str(self) -> str:
        """Human-readable inclusive torch function duration."""

        return _duration_str(self.func_calls_duration)

    @property
    def forward_duration_str(self) -> str:
        """Human-readable wall-clock forward duration."""

        return _duration_str(self.forward_duration)

    @property
    def _source_trace(self) -> "Trace | None":
        """Owning Trace, if still alive."""

        ref = self.__dict__.get("_source_trace_ref")
        if ref is None:
            return None
        return cast("Trace | None", ref())

    @_source_trace.setter
    def _source_trace(self, value: "Trace | None") -> None:
        self._source_trace_ref = weakref.ref(value) if value is not None else None

    def _output_values(self, field_name: str) -> list[Any]:
        """Return one field from all output layers."""

        trace = self._source_trace
        if trace is None:
            return []
        return [getattr(trace.ops[op_label], field_name) for op_label in self.output_ops]

    def _single_output_value(self, field_name: str) -> Any:
        """Return one output field, requiring exactly one output."""

        values = self._output_values(field_name)
        if len(values) != 1:
            from ..intervention.errors import MultiOutputModuleError

            raise MultiOutputModuleError(
                f"ModuleCall '{self.call_label}' has {len(values)} outputs; "
                "use .outs[i] or resolve .output_ops[i]."
            )
        return values[0]

    @property
    def outs(self) -> list[Any]:
        """Saved output tensors for this module call."""

        return self._output_values("out")

    @property
    def out(self) -> Any:
        """Saved output tensor for a single-output module call."""

        return self._single_output_value("out")

    @property
    def out_shapes(self) -> list[Any]:
        """Output shapes for this module call."""

        return self._output_values("shape")

    @property
    def out_shape(self) -> Any:
        """Output shape for a single-output module call."""

        return self._single_output_value("shape")

    @property
    def out_dtypes(self) -> list[Any]:
        """Output dtypes for this module call."""

        return self._output_values("dtype")

    @property
    def out_dtype(self) -> Any:
        """Output dtype for a single-output module call."""

        return self._single_output_value("dtype")

    @property
    def out_memories(self) -> list[Any]:
        """Output memories for this module call."""

        return self._output_values("memory")

    @property
    def out_memory(self) -> Any:
        """Output memory for a single-output module call."""

        return self._single_output_value("memory")

    @property
    def out_memories_str(self) -> list[str]:
        """Human-readable output memories for this module call."""

        return [human_readable_size(memory or 0) for memory in self.out_memories]

    @property
    def out_memory_str(self) -> str:
        """Human-readable output memory for a single-output module call."""

        return human_readable_size(self.out_memory or 0)

    @property
    def grads(self) -> list[Any]:
        """Saved output gradients for this module call."""

        return self._output_values("grad")

    @property
    def grad(self) -> Any:
        """Saved output gradient for a single-output module call."""

        return self._single_output_value("grad")

    def __repr__(self) -> str:
        """Show pass label, layer count, and children."""
        lines = [
            f"ModuleCall: {self.call_label}",
            f"  ops: {self.num_ops}",
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
        from ..export import _parquet_safe_dataframe

        _parquet_safe_dataframe(self.to_pandas()).to_json(filepath, orient=orient, **kwargs)

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state annotated with the current I/O format version."""
        state = self.__dict__.copy()
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state with version-aware default filling."""
        read_tlspec_version(state, cls_name=type(self).__name__)
        if "address" not in state and "module_address" in state:
            state["address"] = state.pop("module_address")
        if "call_index" not in state and "pass_num" in state:
            state["call_index"] = state.pop("pass_num")
        if "call_label" not in state and "pass_label" in state:
            state["call_label"] = state.pop("pass_label")
        if "all_addresses" not in state and "all_module_addresses" in state:
            state["all_addresses"] = state.pop("all_module_addresses")
        if "ops" not in state and "layers" in state:
            state["ops"] = state.pop("layers")
        if "input_ops" not in state:
            state["input_ops"] = list(state.get("input_layers", []))
        if "output_ops" not in state:
            if "output_labels" in state:
                state["output_ops"] = state.pop("output_labels")
            else:
                state["output_ops"] = list(state.get("output_layers", []))
        default_fill_state(
            state,
            defaults={
                "all_addresses": [state["address"]],
                "cls": None,
                "class_name": "",
                "class_qualname": "",
                "ordinal_index": 0,
                "forward_arg_names": [],
                "num_forward_args_total": 0,
                "num_forward_pos_args": 0,
                "num_forward_kwargs": 0,
                "forward_args_summary": "",
                "forward_kwargs_summary": "",
                "forward_duration": 0.0,
                "code_context": [],
                "module_call_stack": [],
                "_source_trace_ref": None,
                "output_structure": None,
            },
        )
        self.__dict__.update(state)


@dataclass(init=False)
class Module:
    """Aggregate metadata for one nn.Module across all its invocations.

    The primary user-facing class for module inspection.  Provides both
    static metadata (source file, class name, hierarchy) and dynamic
    data (layers computed, parameter usage, pass-level detail).

    ``layer_labels`` stores **no-pass** labels (e.g. ``"conv2d_1_1"``),
    pointing to aggregate Layer entries.  For per-pass detail, access
    ``self.ops[call_index].ops`` which stores pass-qualified labels.

    For single-pass modules, per-pass fields (``input_layers``,
    ``output_layers``, ``forward_args``, ``forward_kwargs``) are accessible
    directly via ``_single_pass_or_error()`` delegation to ``ops[0]``.
    """

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "address": FieldPolicy.KEEP,
        "all_addresses": FieldPolicy.KEEP,
        "cls": FieldPolicy.DROP,
        "class_name": FieldPolicy.KEEP,
        "class_qualname": FieldPolicy.KEEP,
        "class_source_file": FieldPolicy.KEEP,
        "class_source_line": FieldPolicy.KEEP,
        "init_source_file": FieldPolicy.KEEP,
        "init_source_line": FieldPolicy.KEEP,
        "forward_source_file": FieldPolicy.KEEP,
        "forward_source_line": FieldPolicy.KEEP,
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
        "layer_labels": FieldPolicy.KEEP,
        "params": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "param_memory": FieldPolicy.KEEP,
        "buffer_layers": FieldPolicy.KEEP,
        "_buffer_accessor": FieldPolicy.DROP,
        "training": FieldPolicy.KEEP,
        "forward_pre_hooks": FieldPolicy.KEEP,
        "forward_hooks": FieldPolicy.KEEP,
        "backward_pre_hooks": FieldPolicy.KEEP,
        "backward_hooks": FieldPolicy.KEEP,
        "full_backward_pre_hooks": FieldPolicy.KEEP,
        "full_backward_hooks": FieldPolicy.KEEP,
        "custom_attributes": FieldPolicy.BLOB_RECURSIVE,
        "custom_methods": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
    }

    address: str
    all_addresses: List[str]
    cls: type[Any] | None
    class_name: str
    class_qualname: str
    class_source_file: Optional[str]
    class_source_line: Optional[int]
    init_source_file: Optional[str]
    init_source_line: Optional[int]
    forward_source_file: Optional[str]
    forward_source_line: Optional[int]
    class_docstring: Optional[str]
    init_signature: Optional[str]
    init_docstring: Optional[str]
    forward_signature: Optional[str]
    forward_docstring: Optional[str]
    address_parent: Optional[str]
    address_children: List[str]
    address_depth: int
    call_parent: Optional[str]
    call_children: List[str]
    call_depth: int
    num_calls: int
    ops: ModuleCallAccessor
    call_labels: List[str]
    layer_labels: List[str]
    input_ops: List[str]
    input_layers: List[str]
    output_ops: List[str]
    output_layers: List[str]
    output_structure: "ContainerSpec | None"
    params: "ParamAccessor"
    num_params: int
    num_params_trainable: int
    num_params_frozen: int
    num_param_tensors: int
    num_param_tensors_trainable: int
    num_param_tensors_frozen: int
    param_memory: int
    has_trainable_params: bool
    has_frozen_params: bool
    buffer_layers: List[str]
    training: bool
    forward_pre_hooks: List[HookInfo]
    forward_hooks: List[HookInfo]
    backward_pre_hooks: List[HookInfo]
    backward_hooks: List[HookInfo]
    full_backward_pre_hooks: List[HookInfo]
    full_backward_hooks: List[HookInfo]
    custom_attributes: Dict[str, Any]
    custom_methods: List[str]
    forward_args_summary: str
    forward_kwargs_summary: str
    forward_duration: float
    total_forward_duration: float
    func_calls_duration: float
    total_func_calls_duration: float

    def __init__(
        self,
        # Identity
        address: str,
        all_addresses: Optional[List[str]] = None,
        name: str = "",
        cls: type[Any] | None = None,
        class_name: str = "",
        class_qualname: str = "",
        # Source info
        class_source_file: Optional[str] = None,
        class_source_line: Optional[int] = None,
        init_source_file: Optional[str] = None,
        init_source_line: Optional[int] = None,
        forward_source_file: Optional[str] = None,
        forward_source_line: Optional[int] = None,
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
        ops: Optional[Dict[int, "ModuleCall"]] = None,
        call_labels: Optional[List[str]] = None,
        # Layers (aggregate)
        layer_labels: Optional[List[str]] = None,
        # Parameters
        params: Optional["ParamAccessor"] = None,
        num_params: int = 0,
        num_params_trainable: int = 0,
        num_params_frozen: int = 0,
        param_memory: int = 0,
        # Buffers
        buffer_layers: Optional[List[str]] = None,
        # Module state
        training: bool = True,
        forward_pre_hooks: List[HookInfo] | None = None,
        forward_hooks: List[HookInfo] | None = None,
        backward_pre_hooks: List[HookInfo] | None = None,
        backward_hooks: List[HookInfo] | None = None,
        full_backward_pre_hooks: List[HookInfo] | None = None,
        full_backward_hooks: List[HookInfo] | None = None,
        custom_attributes: Optional[Dict[str, Any]] = None,
        custom_methods: Optional[List[str]] = None,
        # Back-reference
        _source_trace: "Trace | None" = None,
    ) -> None:
        self.address = address
        self.all_addresses = all_addresses if all_addresses is not None else [address]
        self.cls = cls
        self.class_name = class_name
        self.class_qualname = class_qualname

        self.class_source_file = class_source_file
        self.class_source_line = class_source_line
        self.init_source_file = init_source_file
        self.init_source_line = init_source_line
        self.forward_source_file = forward_source_file
        self.forward_source_line = forward_source_line
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
        self.ops = ModuleCallAccessor(ops)
        self.call_labels = call_labels if call_labels is not None else []

        # layer_labels stores NO-PASS labels (e.g. "conv2d_1_1") -> Layer.
        # Contrast with ModuleCall.ops which stores pass-qualified labels.
        self.layer_labels = layer_labels if layer_labels is not None else []

        from .param_log import ParamAccessor

        self.params = params if params is not None else ParamAccessor({})
        self.num_params = num_params
        self.num_params_trainable = num_params_trainable
        self.num_params_frozen = num_params_frozen
        self.param_memory = param_memory
        self.buffer_layers = buffer_layers if buffer_layers is not None else []
        self._buffer_accessor: Any = None  # populated by _build_module_logs

        self.training = training
        self.forward_pre_hooks = forward_pre_hooks if forward_pre_hooks is not None else []
        self.forward_hooks = forward_hooks if forward_hooks is not None else []
        self.backward_pre_hooks = backward_pre_hooks if backward_pre_hooks is not None else []
        self.backward_hooks = backward_hooks if backward_hooks is not None else []
        self.full_backward_pre_hooks = (
            full_backward_pre_hooks if full_backward_pre_hooks is not None else []
        )
        self.full_backward_hooks = full_backward_hooks if full_backward_hooks is not None else []
        self.custom_attributes = custom_attributes if custom_attributes is not None else {}
        self.custom_methods = custom_methods if custom_methods is not None else []

        # Store as weakref to break circular reference (Trace -> _module_logs -> Module -> Trace).
        self._source_trace_ref = weakref.ref(_source_trace) if _source_trace is not None else None

    @property
    def name(self) -> str:
        """Return the final segment of the primary module address.

        Returns
        -------
        str
            Module name within its parent module.
        """

        return "" if self.address == "self" else self.address.rsplit(".", 1)[-1]

    @property
    def has_multiple_addresses(self) -> bool:
        """Whether this module appears at multiple addresses."""
        return len(self.all_addresses) > 1

    @property
    def num_layers(self) -> int:
        """Number of unique layers in this module."""
        return len(self.layer_labels)

    @property
    def layers(self) -> list[Any]:
        """Layer records belonging to this Module."""

        trace = self._source_trace
        if trace is None:
            return list(self.layer_labels)
        return [trace.layers[label] for label in self.layer_labels]

    @property
    def call_parent_module(self) -> "Module | None":
        """Parent Module in the dynamic call tree, if any."""

        if self.call_parent is None or self._source_trace is None:
            return None
        return cast("Module", self._source_trace.modules[self.call_parent])

    @property
    def call_children_modules(self) -> list["Module"]:
        """Child Modules in the dynamic call tree."""

        trace = self._source_trace
        if trace is None:
            return []
        return [cast("Module", trace.modules[address]) for address in self.call_children]

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
    def has_forward_hooks(self) -> bool:
        """Whether any forward hook registry is nonempty."""

        return bool(self.forward_pre_hooks or self.forward_hooks)

    @property
    def has_backward_hooks(self) -> bool:
        """Whether any backward hook registry is nonempty."""

        return bool(
            self.backward_pre_hooks
            or self.backward_hooks
            or self.full_backward_pre_hooks
            or self.full_backward_hooks
        )

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
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state without touching disk."""
        read_tlspec_version(state, cls_name=type(self).__name__)
        default_fill_state(
            state,
            defaults={
                "_buffer_accessor": None,
                "_source_trace_ref": None,
                "class_source_file": None,
                "class_source_line": None,
                "init_source_file": None,
                "init_source_line": None,
                "forward_source_file": None,
                "forward_source_line": None,
                "training": True,
                "forward_pre_hooks": [],
                "forward_hooks": [],
                "backward_pre_hooks": [],
                "backward_hooks": [],
                "full_backward_pre_hooks": [],
                "full_backward_hooks": [],
            },
        )
        self.__dict__.update(state)

    # --- Per-call delegating properties ---
    # Mirror the Layer delegation pattern: single-pass modules transparently
    # expose per-pass fields; multi-pass modules raise with guidance.

    def _single_pass_or_error(self, field_name: str) -> Any:
        """Return a field from the single pass, or raise if the module has multiple ops.

        For modules invoked once, transparently delegates to ops[0].
        For multi-pass modules, raises AttributeError directing the user to
        access the field on a specific pass.
        """
        if self.num_calls > 1:
            raise AttributeError(
                f"Module '{self.address}' has {self.num_calls} ops. "
                f"Access '{field_name}' on a specific pass: "
                f"module.ops[0].{field_name}, module.ops[1].{field_name}, etc."
            )
        if len(self.ops) == 0:
            return None
        return getattr(self.ops[0], field_name)

    @property
    def output_structure(self) -> "ContainerSpec | None":
        """Return the single call's structured output shape when unambiguous.

        Returns
        -------
        ContainerSpec | None
            Output container specification for single-call modules. Multi-call
            aggregate modules return ``None`` to avoid inventing an additional
            call-index container dimension.
        """

        if len(self.ops._dict) != 1:
            return None
        return next(iter(self.ops._dict.values())).output_structure

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
    def forward_args_summary(self) -> str:
        """Human-readable forward positional arguments for a single-call Module."""

        return cast(str, self._single_pass_or_error("forward_args_summary"))

    @property
    def forward_kwargs_summary(self) -> str:
        """Human-readable forward keyword arguments for a single-call Module."""

        return cast(str, self._single_pass_or_error("forward_kwargs_summary"))

    @property
    def forward_duration(self) -> float:
        """Wall-clock duration for this Module's single forward call."""

        return cast(float, self._single_pass_or_error("forward_duration"))

    @property
    def forward_duration_str(self) -> str:
        """Human-readable single-call forward duration."""

        return _duration_str(self.forward_duration)

    @property
    def total_forward_duration(self) -> float:
        """Sum of wall-clock forward durations across all calls."""

        return sum(call.forward_duration for call in self.ops.values())

    @property
    def total_forward_duration_str(self) -> str:
        """Human-readable total forward duration."""

        return _duration_str(self.total_forward_duration)

    @property
    def func_calls_duration(self) -> float:
        """Inclusive torch function duration for this Module's single call."""

        return cast(float, self._single_pass_or_error("func_calls_duration"))

    @property
    def func_calls_duration_str(self) -> str:
        """Human-readable single-call torch function duration."""

        return _duration_str(self.func_calls_duration)

    @property
    def total_func_calls_duration(self) -> float:
        """Sum of inclusive torch function durations across all calls."""

        return sum(call.func_calls_duration for call in self.ops.values())

    @property
    def total_func_calls_duration_str(self) -> str:
        """Human-readable total torch function duration."""

        return _duration_str(self.total_func_calls_duration)

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors owned by this Module."""

        return len(self.params)

    @property
    def num_param_tensors_trainable(self) -> int:
        """Number of trainable parameter tensors owned by this Module."""

        return sum(1 for param in self.params if param.trainable)

    @property
    def num_param_tensors_frozen(self) -> int:
        """Number of frozen parameter tensors owned by this Module."""

        return sum(1 for param in self.params if not param.trainable)

    @property
    def has_trainable_params(self) -> bool:
        """Whether this Module owns at least one trainable parameter."""

        return self.num_params_trainable > 0

    @property
    def has_frozen_params(self) -> bool:
        """Whether this Module owns at least one frozen parameter."""

        return self.num_params_frozen > 0

    @property
    def class_source_location(self) -> str | None:
        """Combined class source location as ``file:line``."""

        if self.class_source_file is None or self.class_source_line is None:
            return None
        return f"{self.class_source_file}:{self.class_source_line}"

    @property
    def init_source_location(self) -> str | None:
        """Combined ``__init__`` source location as ``file:line``."""

        if self.init_source_file is None or self.init_source_line is None:
            return None
        return f"{self.init_source_file}:{self.init_source_line}"

    @property
    def forward_source_location(self) -> str | None:
        """Combined ``forward`` source location as ``file:line``."""

        if self.forward_source_file is None or self.forward_source_line is None:
            return None
        return f"{self.forward_source_file}:{self.forward_source_line}"

    def _single_call_or_error(self) -> ModuleCall:
        """Return the only ModuleCall or raise for multi-call modules."""

        if self.num_calls != 1 or len(self.ops) != 1:
            raise ValueError(
                f"Module '{self.address}' has {self.num_calls} calls; use module.calls[N]."
            )
        return self.ops[0]

    @property
    def calls(self) -> ModuleCallAccessor:
        """Scoped module-call collection."""

        return self.ops

    @property
    def outs(self) -> list[Any]:
        """Saved output tensors for a single-call module."""

        return self._single_call_or_error().outs

    @property
    def out(self) -> Any:
        """Saved output tensor for a single-call, single-output module."""

        return self._single_call_or_error().out

    @property
    def out_shapes(self) -> list[Any]:
        """Output shapes for a single-call module."""

        return self._single_call_or_error().out_shapes

    @property
    def out_shape(self) -> Any:
        """Output shape for a single-call, single-output module."""

        return self._single_call_or_error().out_shape

    @property
    def grads(self) -> list[Any]:
        """Saved output gradients for a single-call module."""

        return self._single_call_or_error().grads

    @property
    def grad(self) -> Any:
        """Saved output gradient for a single-call, single-output module."""

        return self._single_call_or_error().grad

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
        for label in self.layer_labels:
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

    def __repr__(self) -> str:
        """Show address, class, depth, param count, layer count, and pass count."""
        lines = [
            f"Module: {self.address} ({self.class_name})",
            f"  call_depth: {self.call_depth}, address_depth: {self.address_depth}",
            f"  num_params: {self.num_params}",
            f"  num_layers: {self.num_layers}",
            f"  num_calls: {self.num_calls}",
        ]
        if self.has_multiple_addresses:
            lines.append(f"  aliases: {self.all_addresses}")
        if self.address_children:
            lines.append(f"  children: {self.address_children}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the total number of layers across all ops of this module."""
        return self.num_layers

    def __getitem__(self, ix: int | str) -> Any:
        """Return the Op at position ix within this module's layer list.

        Supports int indexing and string label lookup (#120).
        """
        if self._source_trace is None:
            raise RuntimeError("No source Trace reference; cannot index into layers.")
        if isinstance(ix, str):
            # String label lookup within this module's layers
            if ix in self.layer_labels:
                return self._source_trace[ix]
            # Try substring match within module layers
            matches = [lbl for lbl in self.layer_labels if ix in lbl]
            if len(matches) == 1:
                return self._source_trace[matches[0]]
            elif len(matches) > 1:
                raise ValueError(
                    f"Ambiguous lookup: '{ix}' matches {len(matches)} layers in module "
                    f"'{self.address}': {', '.join(matches[:5])}"
                )
            raise KeyError(f"'{ix}' not found in module '{self.address}' layers")
        return self._source_trace[self.layer_labels[ix]]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over Op entries for all layers in this module."""
        if self._source_trace is None:
            return iter(self.layer_labels)
        return iter(self._source_trace[label] for label in self.layer_labels)

    def draw(self, **kwargs: Any) -> str:
        """Render this module's focused graph.

        Parameters
        ----------
        **kwargs:
            Keyword arguments forwarded to ``Trace.draw``.

        Returns
        -------
        str
            Graphviz DOT source string.

        Raises
        ------
        RuntimeError
            If this Module is not bound to a Trace.
        """

        trace: Trace | None = self._source_trace
        if trace is None:
            raise RuntimeError("Module not bound to a Trace")
        return cast(str, trace.draw(module=self, **kwargs))

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
        for label in self.layer_labels:
            entry = self._source_trace[label]
            rows.append(
                {
                    "layer_label": entry.layer_label,
                    "layer_type": entry.layer_type,
                    "shape": entry.shape,
                    "dtype": entry.dtype,
                    "num_ops": getattr(entry, "num_ops", 1),
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


class ModuleAccessor(Accessor["Module"]):
    """Dict-like accessor for Module objects.

    Supports indexing by:
    * **module address** (str) -- e.g. ``"features.0"``, ``"self"``.
    * **alias** (str) -- for shared modules (same nn.Module registered at
      multiple addresses), any alias resolves to the same Module.
    * **pass label** (str) -- e.g. ``"features.0:2"`` returns a ModuleCall.
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
        module_dict: Dict[str, "Module"],
        module_list: Optional[List["Module"]] = None,
        pass_dict: Optional[Dict[str, "ModuleCall"]] = None,
    ) -> None:
        super().__init__(module_dict, item_list=module_list)
        self._pass_dict = pass_dict if pass_dict is not None else {}  # pass label -> ModuleCall
        # Alias map: for shared modules (same nn.Module at multiple addresses),
        # every non-primary address also resolves to the same Module.
        self._alias_dict: Dict[str, "Module"] = {}
        for ml in self._dict.values():
            for alias in ml.all_addresses:
                if alias not in self._dict:
                    self._alias_dict[alias] = ml

    def __getitem__(  # type: ignore[override]
        self, key: Union[int, str]
    ) -> "Module":
        """Return a Module by module-specific lookup rules."""
        if key == "":
            key = "self"
        if isinstance(key, str) and key in self._alias_dict:
            return self._alias_dict[key]
        if isinstance(key, str) and key in self._pass_dict:
            key = self._pass_dict[key].address
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        """Return True if key is a known module address, alias, or pass label."""
        if key == "":
            key = "self"
        return super().__contains__(key) or key in self._alias_dict or key in self._pass_dict

    def __dir__(self) -> list[str]:
        """Return Python attributes plus module addresses for tab completion.

        Returns
        -------
        list[str]
            Attribute names and valid module addresses.
        """

        keys = set(self._dict.keys()) | set(self._alias_dict.keys()) | set(self._pass_dict.keys())
        return sorted(set(super().__dir__()) | keys)

    def _ipython_key_completions_(self) -> list[str]:
        """Return module addresses for IPython ``obj[...]`` completion.

        Returns
        -------
        list[str]
            Valid module addresses and pass labels.
        """

        return (
            list(self._dict.keys()) + list(self._alias_dict.keys()) + list(self._pass_dict.keys())
        )

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
