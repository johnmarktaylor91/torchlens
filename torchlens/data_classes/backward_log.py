"""Data classes for first-class backward-pass capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from ..constants import (
    BACKWARD_LOG_FIELD_ORDER,
    GRAD_FN_LOG_FIELD_ORDER,
    GRAD_FN_PASS_LOG_FIELD_ORDER,
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
class GradFnPassLog:
    """Runtime data for one execution of an autograd ``grad_fn`` node."""

    pass_num: int
    grad_inputs: Any = None
    grad_outputs: Any = None
    time_started: float | None = None
    time_finished: float | None = None

    def to_pandas(self) -> pd.DataFrame:
        """Export this pass as a one-row DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``GRAD_FN_PASS_LOG_FIELD_ORDER``.
        """
        row = {field_name: getattr(self, field_name) for field_name in GRAD_FN_PASS_LOG_FIELD_ORDER}
        return pd.DataFrame([row], columns=GRAD_FN_PASS_LOG_FIELD_ORDER)


@dataclass
class GradFnLog:
    """Metadata and runtime passes for one autograd ``grad_fn`` node."""

    grad_fn_id: int
    name: str
    module_path: str
    is_custom: bool
    is_intervening: bool = True
    corresponding_layer: str | None = None
    next_grad_fn_ids: list[int] = field(default_factory=list)
    passes: dict[int, GradFnPassLog] = field(default_factory=dict)

    @property
    def num_passes(self) -> int:
        """Number of times this grad_fn has executed during captured backward passes."""
        return len(self.passes)

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


@dataclass
class BackwardLog:
    """Container for a ModelLog's captured backward graph and runtime data."""

    grad_fn_logs: dict[int, GradFnLog] = field(default_factory=dict)
    grad_fn_order: list[int] = field(default_factory=list)
    root_grad_fn_id: int | None = None
    num_passes: int = 0
    peak_memory_bytes: int = 0
    memory_backend: str = "unknown"

    @property
    def num_grad_fns(self) -> int:
        """Number of unique autograd grad_fn nodes discovered."""
        return len(self.grad_fn_logs)

    @property
    def num_intervening_grad_fns(self) -> int:
        """Number of grad_fn nodes without a corresponding forward LayerLog."""
        return sum(1 for grad_fn in self.grad_fn_logs.values() if grad_fn.is_intervening)

    def to_pandas(self) -> pd.DataFrame:
        """Export this backward log as a one-row DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``BACKWARD_LOG_FIELD_ORDER``.
        """
        row = {field_name: getattr(self, field_name) for field_name in BACKWARD_LOG_FIELD_ORDER}
        return pd.DataFrame([row], columns=BACKWARD_LOG_FIELD_ORDER)
