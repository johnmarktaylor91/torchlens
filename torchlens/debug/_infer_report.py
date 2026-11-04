"""The report :func:`torchlens.debug.guess_input_shape` returns, and its one-line repr.

Extracted from ``_infer_input_shape`` (file-cap ratchet) so the report's display
lives beside its fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._debug_report import FoundReport
from .._utils._display_card import REPR_COLS

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace


@dataclass(frozen=True)
class GuessInputShapeReport(FoundReport):
    """Result returned by :func:`guess_input_shape`.

    Parameters
    ----------
    found:
        Whether a verified input was found.
    shape:
        Single tensor input shape, when applicable.
    shapes:
        Multi-input shapes, when applicable.
    dtype:
        Input dtype for the primary tensor.
    value_range:
        Synthetic value recipe: ``("uniform", 0, 1)`` or ``("randint", 0, vocab)``.
    flexible_dims:
        Dimensions that are expected to tolerate other sizes.
    constraining_module:
        Module qualname that constrained the inferred shape, when known.
    constraining_op:
        Operation name that constrained the inferred shape, when known.
    source_line:
        Source line for the constraining op, when known.
    example_input:
        Ready-to-use input object.
    strategy:
        Strategy that produced the winning input.
    reason:
        Failure reason when ``found`` is false.
    attempts:
        Probe diary as ``(shape, outcome)`` tuples.
    trace:
        Final verification trace when the internal caller asked for one; `None` otherwise.
    message:
        Actionable human-readable summary.

    Examples
    --------
    >>> mlp = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
    >>> report = tl.debug.guess_input_shape(mlp)
    >>> isinstance(report, GuessInputShapeReport), report.found
    (True, True)
    >>> report.shape[-1]
    8
    """

    found: bool
    shape: tuple[int, ...] | None
    shapes: tuple[tuple[int, ...], ...] | None
    dtype: torch.dtype | None
    value_range: tuple[str, float, float] | None
    flexible_dims: tuple[int, ...]
    constraining_module: str | None
    constraining_op: str | None
    source_line: str | None
    example_input: Any | None
    strategy: str
    reason: str | None
    attempts: tuple[tuple[tuple[int, ...] | None, str], ...]
    trace: Trace | None
    message: str

    _subject_text = "input shape"

    def __repr__(self) -> str:
        """One line: found, shape, dtype, flexible dims and constraining module.

        Never prints tensor values (``example_input`` stays out); ``print(report)``
        shows the card with the message.
        """

        dtype = None if self.dtype is None else str(self.dtype).removeprefix("torch.")
        shape = f"shape={self.shape}" if self.shapes is None else f"shapes={self.shapes}"
        parts = [f"found={self.found}", shape, f"dtype={dtype}"]
        parts.append(f"flexible_dims={self.flexible_dims}")
        parts.append(f"constraining_module={self.constraining_module!r}")
        if not self.found:
            parts.append(f"reason={self.reason!r}")
        text = f"GuessInputShapeReport({', '.join(parts)})"
        return text if len(text) <= REPR_COLS else text[: REPR_COLS - 4] + "...)"
