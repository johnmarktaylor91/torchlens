"""Autograd grad_fn metadata and lookup accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, Union
import warnings

import torch

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..constants import GRAD_FN_LOG_FIELD_ORDER
from ._accessor_base import Accessor
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

        super().__init__(calls or {})
        self._label = label

    def __getitem__(self, key: int | str) -> GradFnCall:
        """Return a GradFnCall by call index or pass-qualified label."""

        if isinstance(key, int):
            return self._dict[key]
        resolved = self._resolve_pass_qualified(key)
        if resolved is not None:
            return resolved
        raise KeyError(f"GradFn call '{key}' not found in scoped calls.")

    def __setitem__(self, key: int, value: GradFnCall) -> None:
        """Set a GradFnCall by call index."""

        self._dict[key] = value

    def __contains__(self, key: object) -> bool:
        """Return whether key resolves to a GradFnCall."""

        if isinstance(key, int):
            return key in self._dict
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
        """Resolve ``grad_fn_label:pass`` notation to a GradFnCall."""
        base, _, call_index_str = key.rpartition(":")
        if base == self._label:
            try:
                return self._dict[int(call_index_str)]
            except (KeyError, ValueError):
                return None
        return None


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
class GradFn:
    """Metadata and runtime ops for one autograd ``grad_fn`` node."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "grad_fn_id": FieldPolicy.KEEP,
        "class_name": FieldPolicy.KEEP,
        "class_qualname": FieldPolicy.KEEP,
        "is_custom": FieldPolicy.KEEP,
        "label": FieldPolicy.KEEP,
        "grad_fn_type": FieldPolicy.KEEP,
        "grad_fn_type_num": FieldPolicy.KEEP,
        "grad_fn_total_num": FieldPolicy.KEEP,
        "is_intervening": FieldPolicy.KEEP,
        "op": FieldPolicy.KEEP,
        "next_grad_fn_ids": FieldPolicy.KEEP,
        "ops": FieldPolicy.KEEP,
    }

    grad_fn_id: int
    class_name: str
    class_qualname: str
    is_custom: bool
    label: str
    grad_fn_type: str
    grad_fn_type_num: int
    grad_fn_total_num: int
    is_intervening: bool = True
    op: "Layer | None" = None
    next_grad_fn_ids: list[int] = field(default_factory=list)
    ops: dict[int, GradFnCall] | GradFnCallAccessor = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Promote call storage to a scoped accessor."""

        if not isinstance(self.ops, GradFnCallAccessor):
            self.ops = GradFnCallAccessor(self.ops, self.label)  # type: ignore[assignment]

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with an IO format marker."""

        state = self.__dict__.copy()
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state and fill fields added in newer versions."""

        read_io_format_version(state, cls_name=type(self).__name__)
        default_fill_state(
            state,
            defaults={
                "next_grad_fn_ids": [],
                "ops": {},
            },
        )
        if "has_op" in state:
            legacy_has_op = state.pop("has_op")
            if "is_intervening" not in state:
                state["is_intervening"] = bool(legacy_has_op)
        self.__dict__.update(state)
        self.__post_init__()

    @property
    def has_op(self) -> bool:
        """Return whether this grad_fn has a corresponding forward op.

        Returns
        -------
        bool
            True when ``op`` is not None.
        """

        warnings.warn(
            "GradFn.has_op is deprecated; use not grad_fn.is_intervening instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return not self.is_intervening

    @property
    def num_calls(self) -> int:
        """Number of times this grad_fn has executed during captured backward ops."""
        return len(self.ops)

    @property
    def cls(self) -> type[Any] | None:
        """Return the runtime Python class for this grad_fn when available.

        Returns
        -------
        type[Any] | None
            Runtime class object, if an op or autograd object exposes one.
        """

        return None

    @property
    def grad_fn_label(self) -> str:
        """Return the canonical grad_fn label.

        Returns
        -------
        str
            Grad-fn label.
        """

        return self.label

    @property
    def children(self) -> list[str]:
        """Return labels of child grad_fn nodes.

        Returns
        -------
        list[str]
            Child grad-fn labels when resolvable from stored ids.
        """

        return [str(grad_fn_id) for grad_fn_id in self.next_grad_fn_ids]

    @property
    def call_labels(self) -> list[str]:
        """Pass-qualified labels for this grad_fn."""
        return [f"{self.label}:{call_index}" for call_index in self.ops]

    def _log_call(self, grad_inputs: Any, grad_outputs: Any, timestamp: float) -> None:
        """Append one runtime hook firing to this grad_fn log.

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
            grad_inputs=_clone_grad_value(grad_inputs),
            grad_outputs=_clone_grad_value(grad_outputs),
            time_started=timestamp,
            time_finished=timestamp,
        )

    def to_pandas(self) -> "pd.DataFrame":
        """Export this grad_fn as a one-row DataFrame.

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
        """Initialize an accessor from Trace's flat grad_fn fields.

        Parameters
        ----------
        grad_fn_logs:
            Mapping from ``id(grad_fn)`` to ``GradFn``.
        grad_fn_order:
            Discovery-order list of grad_fn ids.
        """
        grad_fn_dict = {grad_fn.label: grad_fn for grad_fn in grad_fn_logs.values()}
        grad_fn_list = [grad_fn_logs[grad_fn_id] for grad_fn_id in grad_fn_order]
        super().__init__(grad_fn_dict, item_list=grad_fn_list)

    def __getitem__(self, key: int | str) -> GradFn | GradFnCall:  # type: ignore[override]
        """Return a GradFn or GradFnCall by grad-fn-specific lookup rules."""
        return super().__getitem__(key)

    def _resolve_pass_qualified(self, key: str) -> GradFnCall | None:  # type: ignore[override]
        """Resolve ``grad_fn_label:pass`` notation to a GradFnCall."""
        base, _, pass_str = key.rpartition(":")
        if base in self._dict:
            try:
                call_index = int(pass_str)
                return self._dict[base].ops[call_index]
            except (ValueError, KeyError):
                return None
        return None

    def _resolve_substring(self, key: str) -> GradFn | None:
        """Resolve the first grad_fn whose label contains ``key``."""
        matches = [grad_fn for grad_fn in self._list if key in grad_fn.label]
        if matches:
            return matches[0]
        return None

    def _suggest(self, key: str) -> list[str]:
        """Return valid grad_fn labels for error context."""
        return list(self._dict.keys())[:10]

    def __repr__(self) -> str:
        """Format the accessor as a compact label summary."""
        if len(self) == 0:
            return "GradFnAccessor({})"
        items = [
            f"  '{grad_fn.label}': {grad_fn.class_name} "
            f"(ops={grad_fn.num_calls}, intervening={grad_fn.is_intervening})"
            for grad_fn in self._list
        ]
        return f"GradFnAccessor({len(self)} grad_fns):\n" + "\n".join(items)
