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
        """One line: found, dtype and shape, flexible dims and constraining module.

        ``GuessInputShapeReport: found float32 [1, 3, 224, 224], flexible dims (2, 3),
        constrained by conv1``. Never prints tensor values (``example_input``
        stays out); ``print(report)`` shows the card with the message.
        """

        head = "GuessInputShapeReport: "
        if not self.found:
            return f"{head}not found ({self.reason})"
        shapes = (self.shape,) if self.shapes is None else self.shapes
        shape_text = ", ".join(
            "[" + ", ".join(str(size) for size in shape) + "]" for shape in shapes if shape
        )
        dtype = "" if self.dtype is None else str(self.dtype).removeprefix("torch.") + " "
        facts = [f"found {dtype}{shape_text or 'no shape'}"]
        flexible = f"flexible dims {self.flexible_dims}" if self.flexible_dims else None
        constrained = (
            f"constrained by {self.constraining_module}" if self.constraining_module else None
        )
        # Width ladder (house style A1): drop the flexible dims first, then cut.
        for optional in ((flexible, constrained), (constrained,)):
            text = head + ", ".join(facts + [part for part in optional if part])
            if len(text) <= REPR_COLS:
                return text
        return text[: REPR_COLS - 3] + "..."
