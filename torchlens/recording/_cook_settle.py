"""Settle a Trace cooked from a Recording at the cook seam (settlement path 9)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._capture.outcome import stamp_cooked

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _settle_cooked_trace(self: Any, trace: Trace) -> None:
    """Stamp the cooked Trace's attested outcome from its Recording's.

    Parameters
    ----------
    self
        The Recording that was cooked (``is_stopped``, ``stop_reason``, ``outcome``).
    trace
        The postprocessed cooked Trace to stamp.
    """

    # Settle at the cook seam (settlement authority path 9): the cooked
    # Trace is a real product and must carry an attested outcome; a halted
    # cooked trace is HALTED (so the runnable/live-replay gates see it),
    # never a silently-blessed complete. The frontier label is read back
    # post-postprocess so it is the FINAL remapped label.
    cooked_frontier = None
    cooked_reason = None
    cooked_boundary = None
    cooked_boundary_kind = None
    if self.is_stopped:
        output_labels = list(getattr(trace, "output_layers", ()))
        cooked_frontier = str(output_labels[0]) if output_labels else None
        # R06: the halted postprocess remapped the persisted halt fields to
        # FINAL labels; the settled record mirrors them (settle_halted
        # parity) instead of stamping the Recording-space raw label into an
        # outcome whose frontier is final. boundary_kind rides the
        # Recording's own settled outcome; the raw stop_reason stays the
        # fallback when the remap did not run.
        remapped_reason = getattr(trace, "stop_reason", None)
        cooked_reason = remapped_reason if isinstance(remapped_reason, str) else self.stop_reason
        remapped_frontier = trace._graph_stop_frontier
        cooked_boundary = remapped_frontier if isinstance(remapped_frontier, str) else cooked_reason
        cooked_boundary_kind = getattr(self.outcome, "boundary_kind", None)
    stamp_cooked(
        trace,
        halted=self.is_stopped,
        reason=cooked_reason,
        boundary_kind=cooked_boundary_kind,
        boundary_label=cooked_boundary,
        frontier_label=cooked_frontier,
    )
