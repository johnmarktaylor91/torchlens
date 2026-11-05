"""Orphan-island selection and render-omission warnings for Graphviz renders."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member
from .._utils.display import user_stacklevel
from ..errors._base import TorchLensWarning

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _warn_render_omission(message: str) -> None:
    """Disclose that a requested render element is not in the output.

    Parameters
    ----------
    message:
        What was left out and how to get it back.
    """

    warnings.warn(message, TorchLensWarning, stacklevel=user_stacklevel())


def _orphan_island_logs(self: Trace) -> tuple[Any, ...]:
    """Return the orphan ops a ``show_orphans=True`` render draws.

    Warns when the capture kept only dropped orphan husks, which have nothing to
    draw (re-trace with ``keep_orphans=True``).

    Parameters
    ----------
    self:
        Trace whose ``_orphan_logs`` are inspected.

    Returns
    -------
    tuple[Any, ...]
        Labelled orphan ops, in capture order.
    """

    orphan_husks = getattr(self, "_orphan_logs", ())
    orphan_logs = tuple(
        op
        for op in orphan_husks
        if bool(record_member(op, "is_orphan", False))
        and bool(getattr(op, "label", "") or getattr(op, "_label_raw", ""))
    )
    if not orphan_logs and (orphan_husks or self._orphan_labels):  # dropped: labels, no husks
        _warn_render_omission(
            "orphans were dropped from this capture; re-trace with keep_orphans=True"
        )
    return orphan_logs
