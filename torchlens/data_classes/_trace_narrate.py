"""Post-hoc narration mixin for ``Trace`` (lane F28, snoop memo D5).

Lives beside the other Trace mixins; unlike them it is a LEAF (module-level
imports are typing-only, the renderer import is lazy inside the method --
``data_classes`` is L1 and must not eagerly import the L5 snoop package), so
``trace.py`` imports it unconditionally with no typing stand-in.
"""

from __future__ import annotations

from typing import Any


class TraceNarrateMixin:
    """Adds the post-hoc ``narrate`` renderer to :class:`Trace`."""

    def narrate(self, last: int | None = None, *, select: Any = None) -> str:
        """Render this trace's events in the live-narration grammar (snoop D5).

        Post-hoc narration works when ``echo=`` was never enabled: the same
        renderer serves ``Trace``, ``PartialTrace``, and ``Recording``. Final
        labels render natively here; ``trace.nonfinite_ops`` evidence (when
        available) flags the first nonfinite op in the footer.

        Parameters
        ----------
        last:
            Keep only the last ``last`` rows (the crash-tail idiom).
        select:
            Optional filter: a substring, or a callable over
            :class:`torchlens.snoop.NarrationEvent` rows.

        Returns
        -------
        str
            Rendered narration block (no trailing newline).
        """

        from ..snoop import narrate_trace

        return narrate_trace(self, last=last, select=select)
