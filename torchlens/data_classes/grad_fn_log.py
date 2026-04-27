"""Autograd grad_fn metadata and lookup accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, Union

import pandas as pd
import torch

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..constants import GRAD_FN_LOG_FIELD_ORDER
from .grad_fn_pass_log import GradFnPassLog

if TYPE_CHECKING:
    from .layer_log import LayerLog


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
    """Metadata and runtime passes for one autograd ``grad_fn`` node."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "grad_fn_id": FieldPolicy.KEEP,
        "name": FieldPolicy.KEEP,
        "module_path": FieldPolicy.KEEP,
        "is_custom": FieldPolicy.KEEP,
        "label": FieldPolicy.KEEP,
        "grad_fn_type": FieldPolicy.KEEP,
        "grad_fn_type_num": FieldPolicy.KEEP,
        "grad_fn_total_num": FieldPolicy.KEEP,
        "is_intervening": FieldPolicy.KEEP,
        "corresponding_layer": FieldPolicy.KEEP,
        "next_grad_fn_ids": FieldPolicy.KEEP,
        "passes": FieldPolicy.KEEP,
    }

    grad_fn_id: int
    name: str
    module_path: str
    is_custom: bool
    label: str
    grad_fn_type: str
    grad_fn_type_num: int
    grad_fn_total_num: int
    is_intervening: bool = True
    corresponding_layer: "LayerLog | None" = None
    next_grad_fn_ids: list[int] = field(default_factory=list)
    passes: dict[int, GradFnPassLog] = field(default_factory=dict)

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
                "passes": {},
            },
        )
        self.__dict__.update(state)

    @property
    def num_passes(self) -> int:
        """Number of times this grad_fn has executed during captured backward passes."""
        return len(self.passes)

    @property
    def pass_labels(self) -> list[str]:
        """Pass-qualified labels for this grad_fn."""
        return [f"{self.label}:{pass_num}" for pass_num in self.passes]

    def log_pass(self, grad_inputs: Any, grad_outputs: Any, timestamp: float) -> None:
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
        pass_num = len(self.passes) + 1
        self.passes[pass_num] = GradFnPassLog(
            pass_num=pass_num,
            grad_inputs=_clone_grad_value(grad_inputs),
            grad_outputs=_clone_grad_value(grad_outputs),
            time_started=timestamp,
            time_finished=timestamp,
        )

    def to_pandas(self) -> pd.DataFrame:
        """Export this grad_fn as a one-row DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``GRAD_FN_LOG_FIELD_ORDER``.
        """
        row = {field_name: getattr(self, field_name) for field_name in GRAD_FN_LOG_FIELD_ORDER}
        return pd.DataFrame([row], columns=GRAD_FN_LOG_FIELD_ORDER)


class GradFnAccessor:
    """Dict-like accessor for ``GradFnLog`` objects.

    Supports integer order, exact label, pass-qualified label, and first
    substring match against labels.
    """

    def __init__(self, grad_fn_logs: Dict[int, GradFnLog], grad_fn_order: list[int]) -> None:
        """Initialize an accessor from ModelLog's flat grad_fn fields.

        Parameters
        ----------
        grad_fn_logs:
            Mapping from ``id(grad_fn)`` to ``GradFnLog``.
        grad_fn_order:
            Discovery-order list of grad_fn ids.
        """
        self._dict = {grad_fn.label: grad_fn for grad_fn in grad_fn_logs.values()}
        self._list = [grad_fn_logs[grad_fn_id] for grad_fn_id in grad_fn_order]

    def __getitem__(self, key: Union[int, str]) -> Union[GradFnLog, GradFnPassLog]:
        """Return a grad_fn by index, label, pass label, or first substring match.

        Parameters
        ----------
        key:
            Integer ordinal, exact grad_fn label, pass-qualified label, or label substring.

        Returns
        -------
        Union[GradFnLog, GradFnPassLog]
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
                    pass_num = int(pass_str)
                    return self._dict[base].passes[pass_num]
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
                    return int(pass_str) in self._dict[base].passes
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
            f"(passes={grad_fn.num_passes}, intervening={grad_fn.is_intervening})"
            for grad_fn in self._list
        ]
        return f"GradFnAccessor({len(self)} grad_fns):\n" + "\n".join(items)
