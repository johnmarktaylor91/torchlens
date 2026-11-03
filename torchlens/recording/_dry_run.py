"""The payload-free dry-run view of a predicate pass (``tl.recording.DryRun``)."""

from __future__ import annotations

__tl_layer__ = "L2"

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._ir.predicate import CaptureSpec, PredicateDecision, PredicateFailure, RecordContext

if TYPE_CHECKING:
    from ..viz._drawing import Drawing

__all__ = ["DryRun"]


@dataclass(frozen=True, slots=True)
class DryRun:
    """Predicate dry-run trace without retained tensor payloads."""

    contexts: tuple[RecordContext, ...]
    decisions: tuple[bool, ...] = ()
    predicate_failures: tuple[PredicateFailure, ...] = ()

    @property
    def events(self) -> tuple[RecordContext, ...]:
        """Return chronological dry-run events."""

        return self.contexts

    def print_tree(self) -> str:
        """Return a unicode-indented event tree for this trace."""

        from .._visualization.fastlog_live import print_tree

        return print_tree(self)

    def to_pandas(self) -> Any:
        """Return a pandas DataFrame representation of trace events."""

        from .._visualization.fastlog_live import to_pandas

        return to_pandas(self)

    def draw(self, **kwargs: Any) -> Drawing:
        """Render a flat Graphviz graph of trace operation events.

        This is the dry run's flat event view: it accepts only ``vis_outpath``,
        ``vis_save_only`` and ``vis_fileformat``. The full drawing options
        (``vis_mode``, nesting depth, rolled recurrent views) belong to
        ``Trace.draw``; record the model with ``tl.record`` and call
        ``Recording.to_trace().draw(...)`` for them.

        Returns
        -------
        Drawing
            The picture; ``str()`` of it is the DOT source.
        """

        from .._visualization.fastlog_live import draw

        return draw(self, **kwargs)

    def summary(self) -> str:
        """Return a concise human-readable dry-run summary."""

        from .._visualization.fastlog_live import summary

        return summary(self)

    def timeline_html(self) -> Any:
        """Return an IPython HTML timeline for this trace."""

        from .._visualization.fastlog_live import timeline_html

        return timeline_html(self)

    def filter(
        self,
        other_keep_op: Callable[[RecordContext], PredicateDecision] | None = None,
    ) -> DryRun:
        """Return a new trace with decisions from a new op predicate.

        Parameters
        ----------
        other_keep_op:
            Predicate for op, input, and buffer events. Module boundary
            events have no predicate slot and are never re-selected.

        Returns
        -------
        RecordingTrace
            New trace sharing the same event tuple and predicate failures.
        """

        from .._capture.predicates import _normalize_capture_decision

        decisions: list[bool] = []
        for ctx in self.contexts:
            predicate = None if ctx.kind in {"module_enter", "module_exit"} else other_keep_op
            result = predicate(ctx) if predicate is not None else False
            spec = _normalize_capture_decision(result, ctx, False)
            if not isinstance(spec, CaptureSpec):
                decisions.append(True)
                continue
            decisions.append(spec.save_out or spec.save_metadata)
        return DryRun(
            contexts=self.contexts,
            decisions=tuple(decisions),
            predicate_failures=self.predicate_failures,
        )
