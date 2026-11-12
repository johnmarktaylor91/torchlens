"""Capture records the tinygrad backend keeps for validation and derived gradients.

Extracted from ``backend`` so that module stays under its size ceiling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..._ir.refs import DtypeRef


@dataclass(frozen=True)
class TinygradUOpCapture:
    """Captured tinygrad UOp metadata for validation.

    Parameters
    ----------
    label_raw
        Raw TorchLens label for the materialized op.
    uop
        Source tinygrad UOp snapshot.
    op_name
        UOp operation name.
    parent_labels
        Raw parent labels captured from UOp source edges.
    parent_arg_positions
        UOp source positions for each raw parent label.
    payload_snapshot
        Realized tinygrad tensor copy saved during capture.
    edit
        For an op a ``trace(intervene=...)`` edit replaced, the edit's lazy
        ``tensor -> tensor`` application: ``uop`` is the unedited op and the
        payload is the edit applied to its value. ``None`` for every other op.
    """

    label_raw: str
    uop: Any
    op_name: str
    parent_labels: tuple[str, ...]
    parent_arg_positions: tuple[tuple[int, str], ...]
    payload_snapshot: Any
    edit: Callable[[Any], Any] | None = None


@dataclass(frozen=True)
class TinygradIntermediateCandidate:
    """Live retained tinygrad tensor candidate for one op-level gradient.

    Parameters
    ----------
    signature
        Conservative structural signature used for trace-op matching.
    tensor
        Live tinygrad Tensor retained from the no-realize forward pass.
    aval
        Human-readable abstract-value description.
    dtype_ref
        Backend-neutral dtype reference for the tensor.
    """

    signature: tuple[Any, ...]
    tensor: Any
    aval: str
    dtype_ref: DtypeRef | None
