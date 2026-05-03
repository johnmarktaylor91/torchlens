"""Autograd grad_fn metadata and lookup accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, Union

import torch

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..constants import GRAD_FN_LOG_FIELD_ORDER
from .grad_fn_call_log import GradFnCallLog

if TYPE_CHECKING:
    import pandas as pd

    from .layer_log import LayerLog


class GradFnCallAccessor:
    """Scoped dict-like accessor for GradFnCallLog entries owned by a GradFnLog."""

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_label": FieldPolicy.KEEP,
    }

    def __init__(self, calls: dict[int, GradFnCallLog] | None = None, label: str = "") -> None:
        """Initialize the accessor.

        Parameters
        ----------
        calls:
            Mapping from 1-based call index to GradFnCallLog.
        label:
            Owning GradFnLog label used for pass-qualified lookup.
        """

        self._dict: dict[int, GradFnCallLog] = dict(calls or {})
        self._label = label

    def __getitem__(self, key: int | str) -> GradFnCallLog:
        """Return a GradFnCallLog by call index or pass-qualified label."""

        if isinstance(key, int):
            return self._dict[key]
        if ":" in key:
            base, _, call_index_str = key.rpartition(":")
            if base == self._label:
                return self._dict[int(call_index_str)]
        raise KeyError(f"GradFn call '{key}' not found in scoped calls.")

    def __setitem__(self, key: int, value: GradFnCallLog) -> None:
        """Set a GradFnCallLog by call index."""

        self._dict[key] = value

    def __contains__(self, key: object) -> bool:
        """Return whether key resolves to a GradFnCallLog."""

        if isinstance(key, int):
            return key in self._dict
        if isinstance(key, str):
            try:
                self[key]
            except (KeyError, ValueError):
                return False
            return True
        return False

    def __iter__(self) -> Iterator[int]:
        """Iterate call-index keys."""

        return iter(self._dict)

    def __len__(self) -> int:
        """Return the number of scoped calls."""

        return len(self._dict)

    def keys(self) -> list[int]:
        """Return call-index keys."""

        return list(self._dict.keys())

    def values(self) -> list[GradFnCallLog]:
        """Return scoped GradFnCallLog values."""

        return list(self._dict.values())

    def items(self) -> list[tuple[int, GradFnCallLog]]:
        """Return ``(call_index, GradFnCallLog)`` pairs."""

        return list(self._dict.items())


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
class GradFnLog:
    """Metadata and runtime ops for one autograd ``grad_fn`` node."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "grad_fn_id": FieldPolicy.KEEP,
        "name": FieldPolicy.KEEP,
        "module_path": FieldPolicy.KEEP,
        "is_custom": FieldPolicy.KEEP,
        "label": FieldPolicy.KEEP,
        "grad_fn_type": FieldPolicy.KEEP,
        "grad_fn_type_num": FieldPolicy.KEEP,
        "grad_fn_total_num": FieldPolicy.KEEP,
        "has_op": FieldPolicy.KEEP,
        "op": FieldPolicy.KEEP,
        "next_grad_fn_ids": FieldPolicy.KEEP,
        "ops": FieldPolicy.KEEP,
    }

    grad_fn_id: int
    name: str
    module_path: str
    is_custom: bool
    label: str
    grad_fn_type: str
    grad_fn_type_num: int
    grad_fn_total_num: int
    has_op: bool = True
    op: "LayerLog | None" = None
    next_grad_fn_ids: list[int] = field(default_factory=list)
    ops: dict[int, GradFnCallLog] | GradFnCallAccessor = field(default_factory=dict)

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
        self.__dict__.update(state)
        self.__post_init__()

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
    def class_name(self) -> str:
        """Return the grad_fn class name.

        Returns
        -------
        str
            Grad-fn class name.
        """

        return self.name

    @property
    def class_qualname(self) -> str:
        """Return the best-known qualified grad_fn class name.

        Returns
        -------
        str
            Qualified class name when known, otherwise ``class_name``.
        """

        return self.name

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
        self.ops[call_index] = GradFnCallLog(
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


class GradFnAccessor:
    """Dict-like accessor for ``GradFnLog`` objects.

    Supports integer order, exact label, pass-qualified label, and first
    substring match against labels.
    """

    def __init__(self, grad_fn_logs: Dict[int, GradFnLog], grad_fn_order: list[int]) -> None:
        """Initialize an accessor from Trace's flat grad_fn fields.

        Parameters
        ----------
        grad_fn_logs:
            Mapping from ``id(grad_fn)`` to ``GradFnLog``.
        grad_fn_order:
            Discovery-order list of grad_fn ids.
        """
        self._dict = {grad_fn.label: grad_fn for grad_fn in grad_fn_logs.values()}
        self._list = [grad_fn_logs[grad_fn_id] for grad_fn_id in grad_fn_order]

    def __getitem__(self, key: Union[int, str]) -> Union[GradFnLog, GradFnCallLog]:
        """Return a grad_fn by index, label, pass label, or first substring match.

        Parameters
        ----------
        key:
            Integer ordinal, exact grad_fn label, pass-qualified label, or label substring.

        Returns
        -------
        Union[GradFnLog, GradFnCallLog]
            Matching grad_fn log or grad_fn pass log.
        """
        if isinstance(key, int):
            return self._list[key]
        if key in self._dict:
            return self._dict[key]
        if ":" in key:
            base, _, pass_str = key.rpartition(":")
            if base in self._dict:
                try:
                    call_index = int(pass_str)
                    return self._dict[base].ops[call_index]
                except (ValueError, KeyError):
                    pass
        matches = [grad_fn for grad_fn in self._list if key in grad_fn.label]
        if matches:
            return matches[0]
        raise KeyError(f"GradFn '{key}' not found. Valid labels: {list(self._dict.keys())[:10]}...")

    def __contains__(self, key: object) -> bool:
        """Return True if key resolves to a grad_fn or grad_fn pass."""
        if isinstance(key, int):
            return 0 <= key < len(self._list)
        if not isinstance(key, str):
            return False
        if key in self._dict:
            return True
        if ":" in key:
            base, _, pass_str = key.rpartition(":")
            if base in self._dict:
                try:
                    return int(pass_str) in self._dict[base].ops
                except ValueError:
                    return False
        return any(key in grad_fn.label for grad_fn in self._list)

    def __len__(self) -> int:
        """Return the number of grad_fn logs."""
        return len(self._list)

    def __iter__(self) -> Iterator[GradFnLog]:
        """Iterate over grad_fn logs in discovery order."""
        return iter(self._list)

    def __repr__(self) -> str:
        """Format the accessor as a compact label summary."""
        if len(self) == 0:
            return "GradFnAccessor({})"
        items = [
            f"  '{grad_fn.label}': {grad_fn.name} "
            f"(ops={grad_fn.num_calls}, intervening={grad_fn.has_op})"
            for grad_fn in self._list
        ]
        return f"GradFnAccessor({len(self)} grad_fns):\n" + "\n".join(items)
