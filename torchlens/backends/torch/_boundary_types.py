"""Plain data records of one ``autograd.Function`` boundary call.

:mod:`._function_boundary` takes :class:`BoundaryMarks` before a call runs and logs a
finished call (a :class:`_BoundaryCall` with its :class:`BoundarySpec`) as one op. The
records live in this leaf module so modules that only build or read them need not
import the capture machinery.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..._data_classes.internal_types import FuncExecutionContext


@dataclass(frozen=True)
class BoundarySpec:
    """What a boundary op is: its callable, op type, census owner token and source."""

    replay: Callable[..., Any]
    func_name: str
    wrapper_name: str
    original: Any


@dataclass(frozen=True)
class _BoundaryCall:
    """One collapsed call, ready to be logged as its boundary op."""

    spec: BoundarySpec
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    out: Any
    exec_ctx: FuncExecutionContext
    func_call_id: int


@dataclass(frozen=True)
class BoundaryMarks:
    """Journal, census and context marks taken just before a boundary call runs."""

    op_events: list[Any] | None
    journal_mark: int
    witness_state: Any | None
    witness_mark: int
    capture_start_time: float
    rng_states: dict[str, Any]
    autocast_state: Any
    arg_labels: list[tuple[Any, str | None]]
