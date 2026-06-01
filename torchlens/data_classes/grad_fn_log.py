"""Autograd grad_fn_handle metadata and lookup accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, Union
import weakref

import torch

from .._io import FieldPolicy, TLSPEC_VERSION, default_fill_state, read_tlspec_version
from ..constants import GRAD_FN_LOG_FIELD_ORDER
from ._accessor_base import Accessor
from ._tabular_export import TabularExportMixin
from .grad_fn_call_log import GradFnCall

if TYPE_CHECKING:
    import pandas as pd

    from .layer_log import Layer


class GradFnCallAccessor(Accessor[GradFnCall]):
    """Scoped dict-like accessor for GradFnCall entries owned by a GradFn."""

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
        "_label": FieldPolicy.KEEP,
    }

    def __init__(self, calls: dict[int, GradFnCall] | None = None, label: str = "") -> None:
        """Initialize the accessor.

        Parameters
        ----------
        calls:
            Mapping from 1-based call index to GradFnCall.
        label:
            Owning GradFn label used for pass-qualified lookup.
        """

        calls = calls or {}
        super().__init__(calls, item_list=[call for _, call in sorted(calls.items())])
        self._label = label

    def __getitem__(self, key: int | str) -> GradFnCall:
        """Return a GradFnCall by 0-based position or pass-qualified label."""

        if isinstance(key, int):
            return self._list[key]
        resolved = self._resolve_pass_qualified(key)
        if resolved is not None:
            return resolved
        raise KeyError(f"GradFn call '{key}' not found in scoped calls.")

    def __setitem__(self, key: int, value: GradFnCall) -> None:
        """Set a GradFnCall by 1-based call index."""

        self._dict[key] = value
        self._list = [call for _, call in sorted(self._dict.items())]

    def __contains__(self, key: object) -> bool:
        """Return whether key resolves to a GradFnCall."""

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

    def _resolve_pass_qualified(self, key: str) -> GradFnCall | None:
        """Resolve ``label:pass`` notation to a GradFnCall."""
        base, _, call_index_str = key.rpartition(":")
        if base == self._label:
            try:
                return self._dict[int(call_index_str)]
            except (KeyError, ValueError):
                return None
        return None

    def _resolve_substring(self, key: str) -> GradFnCall | None:
        """Resolve a unique bare GradFn label to its only call."""
        if key != self._label:
            return None
        if len(self._dict) == 1:
            return next(iter(self._dict.values()))
        raise ValueError(
            f"GradFn '{key}' has {len(self._dict)} calls. Use a 0-based integer "
            f"position or a call-qualified label like '{key}:1'."
        )


def _clone_grad_value(value: Any) -> Any:
    """Detach and clone tensors in a nested autograd hook payload.

    Parameters
    ----------
    value:
        Hook payload value, often a tuple containing tensors or ``None``.

    Returns
    -------
    Any
        Payload with tensors detached and cloned.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_grad_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_grad_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_grad_value(item) for key, item in value.items()}
    return value


@dataclass
class GradFn(TabularExportMixin):
    """Metadata and runtime ops for one autograd ``grad_fn_handle`` node."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "grad_fn_object_id": FieldPolicy.KEEP,
        "class_name": FieldPolicy.KEEP,
        "class_qualname": FieldPolicy.KEEP,
        "is_custom": FieldPolicy.KEEP,
        "label": FieldPolicy.KEEP,
        "type": FieldPolicy.KEEP,
        "type_index": FieldPolicy.KEEP,
        "ordinal_index": FieldPolicy.KEEP,
        "step_index": FieldPolicy.KEEP,
        "has_op": FieldPolicy.KEEP,
        "op_label": FieldPolicy.KEEP,
        "next_grad_fn_ids": FieldPolicy.KEEP,
        "parents": FieldPolicy.KEEP,
        "children": FieldPolicy.KEEP,
        "siblings": FieldPolicy.KEEP,
        "co_parents": FieldPolicy.KEEP,
        "ops": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
        "class_source_file": FieldPolicy.KEEP,
        "class_source_line": FieldPolicy.KEEP,
        "class_docstring": FieldPolicy.KEEP,
        "init_source_file": FieldPolicy.KEEP,
        "init_source_line": FieldPolicy.KEEP,
        "init_signature": FieldPolicy.KEEP,
        "init_docstring": FieldPolicy.KEEP,
        "forward_source_file": FieldPolicy.KEEP,
        "forward_source_line": FieldPolicy.KEEP,
        "forward_signature": FieldPolicy.KEEP,
        "forward_docstring": FieldPolicy.KEEP,
        "backward_source_file": FieldPolicy.KEEP,
        "backward_source_line": FieldPolicy.KEEP,
        "backward_signature": FieldPolicy.KEEP,
        "backward_docstring": FieldPolicy.KEEP,
    }

    grad_fn_object_id: int
    class_name: str
    class_qualname: str
    is_custom: bool
    label: str
    type: str
    type_index: int
    ordinal_index: int
    step_index: int = 0
    has_op: bool = False
    op_label: str | None = None
    next_grad_fn_ids: list[int] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    siblings: list[str] = field(default_factory=list)
    co_parents: list[str] = field(default_factory=list)
    ops: dict[int, GradFnCall] | GradFnCallAccessor = field(default_factory=dict)
    _source_trace_ref: Any = None
    class_source_file: str | None = None
    class_source_line: int | None = None
    class_docstring: str | None = None
    init_source_file: str | None = None
    init_source_line: int | None = None
    init_signature: str | None = None
    init_docstring: str | None = None
    forward_source_file: str | None = None
    forward_source_line: int | None = None
    forward_signature: str | None = None
    forward_docstring: str | None = None
    backward_source_file: str | None = None
    backward_source_line: int | None = None
    backward_signature: str | None = None
    backward_docstring: str | None = None

    def __post_init__(self) -> None:
        """Promote call storage to a scoped accessor."""

        if not isinstance(self.ops, GradFnCallAccessor):
            self.ops = GradFnCallAccessor(self.ops, self.label)  # type: ignore[assignment]
        else:
            self.ops._label = self.label
        for call in self.ops.values():
            call.label = self.label
            call.source_trace = self.source_trace

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with an IO format marker."""

        state = self.__dict__.copy()
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state and fill fields added in newer versions."""

        read_tlspec_version(state, cls_name=type(self).__name__)
        default_fill_state(
            state,
            defaults={
                "step_index": state.get("trace_" + "index", state.get("overall_" + "index", 0)),
                "has_op": not bool(state.get("is_" + "intervening", True)),
                "op_label": None,
                "next_grad_fn_ids": [],
                "parents": [],
                "children": [],
                "siblings": [],
                "co_parents": [],
                "ops": {},
                "_source_trace_ref": None,
                "class_source_file": None,
                "class_source_line": None,
                "class_docstring": None,
                "init_source_file": None,
                "init_source_line": None,
                "init_signature": None,
                "init_docstring": None,
                "forward_source_file": None,
                "forward_source_line": None,
                "forward_signature": None,
                "forward_docstring": None,
                "backward_source_file": None,
                "backward_source_line": None,
                "backward_signature": None,
                "backward_docstring": None,
            },
        )
        legacy_op = state.pop("op", None)
        if legacy_op is not None and state.get("op_label") is None:
            state["op_label"] = getattr(legacy_op, "layer_label", None)
        legacy_intervening_key = "is_" + "intervening"
        if legacy_intervening_key in state:
            state["has_op"] = not bool(state.pop(legacy_intervening_key))
        state.pop("trace_" + "index", None)
        state.pop("overall_" + "index", None)
        self.__dict__.update(state)
        self.__post_init__()

    @property
    def source_trace(self) -> Any:
        """Return the owning Trace if it is still alive.

        Returns
        -------
        Any
            Owning Trace or ``None``.
        """

        ref = self._source_trace_ref
        return None if ref is None else ref()

    @source_trace.setter
    def source_trace(self, value: Any) -> None:
        """Set the owning Trace weakref.

        Parameters
        ----------
        value:
            Trace object or ``None``.
        """

        self._source_trace_ref = weakref.ref(value) if value is not None else None
        for call in self.ops.values():
            call.source_trace = value

    @property
    def trace(self) -> Any:
        """Alias for the owning Trace."""

        return self.source_trace

    @property
    def op(self) -> "Layer | None":
        """Return the forward Op or Layer associated with this GradFn.

        Returns
        -------
        Layer | None
            Resolved forward record, or ``None`` when no corresponding op was captured.
        """

        if self.op_label is None:
            return None
        trace = self.source_trace
        if trace is None:
            return None
        return trace[self.op_label]

    @property
    def num_calls(self) -> int:
        """Number of times this grad_fn_handle has executed during captured backward ops."""
        return len(self.ops)

    @property
    def has_parents(self) -> bool:
        """Return whether this GradFn has backward parents."""

        return bool(self.parents)

    @property
    def has_children(self) -> bool:
        """Return whether this GradFn has backward children."""

        return bool(self.children)

    @property
    def has_siblings(self) -> bool:
        """Return whether this GradFn has backward siblings."""

        return bool(self.siblings)

    @property
    def has_co_parents(self) -> bool:
        """Return whether this GradFn has backward co-parents."""

        return bool(self.co_parents)

    @property
    def has_saved_call(self) -> bool:
        """Return whether this GradFn has any recorded call data."""

        return any(
            call.grad_inputs is not None or call.grad_outputs is not None
            for call in self.ops.values()
        )

    @property
    def cls(self) -> type[Any] | None:
        """Return the runtime Python class for this grad_fn_handle when available.

        Returns
        -------
        type[Any] | None
            Runtime class object, if an op or autograd object exposes one.
        """

        return None

    @property
    def class_source_location(self) -> str | None:
        """Return the grad-fn class source location.

        Returns
        -------
        str | None
            ``file:line`` location, or ``None`` when unavailable.
        """

        if self.class_source_file is None or self.class_source_line is None:
            return None
        return f"{self.class_source_file}:{self.class_source_line}"

    @property
    def init_source_location(self) -> str | None:
        """Return the grad-fn ``__init__`` source location."""

        if self.init_source_file is None or self.init_source_line is None:
            return None
        return f"{self.init_source_file}:{self.init_source_line}"

    @property
    def forward_source_location(self) -> str | None:
        """Return the custom autograd ``forward`` source location."""

        if self.forward_source_file is None or self.forward_source_line is None:
            return None
        return f"{self.forward_source_file}:{self.forward_source_line}"

    @property
    def backward_source_location(self) -> str | None:
        """Return the custom autograd ``backward`` source location."""

        if self.backward_source_file is None or self.backward_source_line is None:
            return None
        return f"{self.backward_source_file}:{self.backward_source_line}"

    @property
    def call_labels(self) -> list[str]:
        """Pass-qualified labels for this grad_fn_handle."""
        return [f"{self.label}:{call_index}" for call_index in self.ops.keys()]

    @property
    def backward_duration(self) -> float:
        """Return this GradFn's single-call backward duration.

        Returns
        -------
        float
            Backward duration for the only call.

        Raises
        ------
        ValueError
            If this GradFn has multiple calls.
        """

        if len(self.ops) != 1:
            raise ValueError(
                f"GradFn '{self.label}' has {len(self.ops)} calls. Access "
                "backward_duration on a specific call or use total_backward_duration."
            )
        return next(iter(self.ops.values())).backward_duration

    @property
    def backward_duration_str(self) -> str:
        """Return this GradFn's single-call backward duration as text."""

        if len(self.ops) != 1:
            raise ValueError(
                f"GradFn '{self.label}' has {len(self.ops)} calls. Access "
                "backward_duration_str on a specific call or use total_backward_duration_str."
            )
        return next(iter(self.ops.values())).backward_duration_str

    @property
    def total_backward_duration(self) -> float:
        """Return total backward duration across all calls for this GradFn."""

        return sum(call.backward_duration for call in self.ops.values())

    @property
    def total_backward_duration_str(self) -> str:
        """Return total backward duration across all calls as text."""

        return f"{self.total_backward_duration * 1000:.3f} ms"

    def _log_call(self, grad_inputs: Any, grad_outputs: Any, timestamp: float) -> None:
        """Append one runtime hook firing to this grad_fn_handle log.

        Parameters
        ----------
        grad_inputs:
            Gradient inputs provided by the autograd hook.
        grad_outputs:
            Gradient outputs provided by the autograd hook.
        timestamp:
            Wall-clock time when the hook fired.
        """
        call_index = len(self.ops) + 1
        self.ops[call_index] = GradFnCall(
            call_index=call_index,
            label=self.label,
            grad_inputs=_clone_grad_value(grad_inputs),
            grad_outputs=_clone_grad_value(grad_outputs),
            _time_started=timestamp,
            _time_finished=timestamp,
        )

    def to_pandas(self) -> "pd.DataFrame":
        """Export this grad_fn_handle as a one-row DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``GRAD_FN_LOG_FIELD_ORDER``.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        row = {field_name: getattr(self, field_name) for field_name in GRAD_FN_LOG_FIELD_ORDER}
        return pd.DataFrame([row], columns=GRAD_FN_LOG_FIELD_ORDER)


class GradFnAccessor(Accessor[GradFn]):
    """Dict-like accessor for ``GradFn`` objects.

    Supports integer order, exact label, pass-qualified label, and first
    substring match against labels.
    """

    def __init__(self, grad_fn_logs: Dict[int, GradFn], grad_fn_order: list[int]) -> None:
        """Initialize an accessor from Trace's flat grad_fn_handle fields.

        Parameters
        ----------
        grad_fn_logs:
            Mapping from ``id(grad_fn_handle)`` to ``GradFn``.
        grad_fn_order:
            Discovery-order list of grad_fn_handle ids.
        """
        grad_fn_dict = {
            grad_fn_handle.label: grad_fn_handle for grad_fn_handle in grad_fn_logs.values()
        }
        grad_fn_list = [grad_fn_logs[grad_fn_object_id] for grad_fn_object_id in grad_fn_order]
        super().__init__(grad_fn_dict, item_list=grad_fn_list)

    def __getitem__(self, key: int | str) -> GradFn | GradFnCall:  # type: ignore[override]
        """Return a GradFn by grad-fn-specific lookup rules."""
        return super().__getitem__(key)

    def _resolve_pass_qualified(self, key: str) -> GradFn | None:
        """Resolve ``label:pass`` notation to the parent GradFn."""
        base, _, pass_str = key.rpartition(":")
        try:
            int(pass_str)
        except ValueError:
            return None
        if base in self._dict:
            return self._dict[base]
        return None

    def _resolve_substring(self, key: str) -> GradFn | None:
        """Resolve the first grad_fn_handle whose label contains ``key``."""
        matches = [grad_fn_handle for grad_fn_handle in self._list if key in grad_fn_handle.label]
        if matches:
            return matches[0]
        return None

    def _suggest(self, key: str) -> list[str]:
        """Return valid grad_fn_handle labels for error context."""
        return list(self._dict.keys())[:10]

    def __repr__(self) -> str:
        """Format the accessor as a compact label summary."""
        if len(self) == 0:
            return "GradFnAccessor({})"
        items = [
            f"  '{grad_fn_handle.label}': {grad_fn_handle.class_name} "
            f"(ops={grad_fn_handle.num_calls}, has_op={grad_fn_handle.has_op})"
            for grad_fn_handle in self._list
        ]
        return f"GradFnAccessor({len(self)} grad_fns):\n" + "\n".join(items)
