"""Notebook-card ``_repr_html_`` for ``Op`` (housed apart from ``op.py`` for its line ceiling).

This is the one record-layer hook that reaches the notebook package; it stays out of
``_repr.py`` so the plain text formatting helpers there keep no upward edge.
"""

from __future__ import annotations


class OpCardHtmlMixin:
    """Notebook-card ``_repr_html_`` for ``Op`` (housed here for the R43 op.py ceiling)."""

    __slots__ = ()

    def _repr_html_(self) -> str:
        """Return the notebook Op card (treescope memo B2).

        Identity / value / context zones with the PROOF row; cost wrappers
        pass through ``str()`` untouched; the budgeted array view rides the
        ported truncation protocol. Never raises: any internal failure
        degrades to a one-line ``card unavailable`` fragment, and the
        treescope-bridge suppression sentinel replaces the card exactly
        when the bridge just rendered this object (memo 3.6).
        """

        from .._notebook.cards import op_repr_html

        return op_repr_html(self)
