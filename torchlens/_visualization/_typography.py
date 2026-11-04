"""The one typography record for graph rendering (vizmech D29, wave-2 item 14).

Before this module, font decisions were scattered literals: ``labelfontsize="8"``
in four edge builders, ``_EDGE_LABEL_FONT_SIZE = 8`` and
``_MULTIPLICITY_LABEL_FONT_SIZE = 8`` module constants, a ``POINT-SIZE="18"``
HTML wrapper, a ``fontsize="10"`` orphan-cluster caption, and a default theme
that pinned NO font family at all -- making it the only serif theme (Graphviz
falls back to Times) and moving measured label geometry by theme swap alone
(59 -> 68 violations in the memo's theme sweep).

This module is the single authority those channels consult:

- ``TypographyRecord`` carries the pinned family plus SEMANTIC size roles
  (base / annotation / secondary / emphasis) with the ratios derivable, so a
  future scaled theme changes one number instead of six literals.
- ``DEFAULT_TYPOGRAPHY`` pins the family to ``"Helvetica"`` -- the family the
  non-default presets already used. WHICH family ships long-term is maintainer fork
  FK3 ([UI-SPRINT]); THAT a family is pinned everywhere, including the default
  theme, is decided (D29) and implemented here.

Every spelling in this module is DOCUMENTED-UNSTABLE pending the naming
session (the memo's "one typography record" is a placeholder name).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "DEFAULT_TYPOGRAPHY",
    "EDIT_GAP_SIZE",
    "HIGH_CONTRAST_TYPOGRAPHY",
    "TypographyRecord",
    "active_edge_typography",
    "edge_annotation_pt",
    "edge_emphasis_pt",
    "edge_typography_for_draw",
    "format_pt",
    "resolve_edge_typography",
]


#: Height in points of the blank row between a node's label and its edit rows
#: (the F2 spec's 4 pt gap; it holds a space, so only its size matters).
EDIT_GAP_SIZE = 4.0


def format_pt(value: float) -> str:
    """Format a point size for DOT emission (``8`` not ``8.0``).

    Integral values format as integers so the emitted DOT stays byte-stable
    against the historical literal channels ("8", "10", "18").
    """

    if value == int(value):
        return str(int(value))
    return f"{value:g}"


@dataclass(frozen=True)
class TypographyRecord:
    """Font family plus semantic size roles for every label builder.

    Attributes
    ----------
    family:
        Pinned font family emitted on graph, node, and edge scopes. The
        family CHOICE is fork FK3; the pinning is not.
    base_size:
        Ordinary node-label size in points (Graphviz default 14).
    annotation_size:
        Edge annotation channels: argument labels, ``xN`` multiplicity,
        ``accum``/``bwd`` markers (the historical literal 8).
    secondary_size:
        De-emphasized captions such as the orphans cluster (the historical
        literal 10).
    emphasis_size:
        Emphasized standalone edge labels such as conditional-arm markers
        (the historical literal 18).
    """

    family: str
    base_size: float = 14.0
    annotation_size: float = 8.0
    secondary_size: float = 10.0
    emphasis_size: float = 18.0

    @property
    def annotation_ratio(self) -> float:
        """Annotation size as a fraction of the base size."""

        return self.annotation_size / self.base_size

    @property
    def secondary_ratio(self) -> float:
        """Secondary size as a fraction of the base size."""

        return self.secondary_size / self.base_size

    @property
    def emphasis_ratio(self) -> float:
        """Emphasis size as a fraction of the base size."""

        return self.emphasis_size / self.base_size

    @property
    def annotation_pt(self) -> str:
        """Annotation size formatted for DOT emission."""

        return format_pt(self.annotation_size)

    @property
    def secondary_pt(self) -> str:
        """Secondary size formatted for DOT emission."""

        return format_pt(self.secondary_size)

    @property
    def emphasis_pt(self) -> str:
        """Emphasis size formatted for DOT emission."""

        return format_pt(self.emphasis_size)

    def scaled(self, factor: float) -> TypographyRecord:
        """Return a copy with every size role multiplied by ``factor``.

        Ratios are preserved by construction; this is the one door a future
        large-print or compact theme goes through.
        """

        return replace(
            self,
            base_size=self.base_size * factor,
            annotation_size=self.annotation_size * factor,
            secondary_size=self.secondary_size * factor,
            emphasis_size=self.emphasis_size * factor,
        )


#: The pinned default record (family choice rides FK3; sizes are the
#: historical literals, now with one owner).
DEFAULT_TYPOGRAPHY = TypographyRecord(family="Helvetica")

#: The high-contrast preset keeps its bold family across every scope.
HIGH_CONTRAST_TYPOGRAPHY = TypographyRecord(family="Helvetica-Bold")


#: The edge typography of the forward draw in progress. ``None`` outside a draw,
#: where the edge-annotation builders fall back to :data:`DEFAULT_TYPOGRAPHY`.
_EDGE_TYPOGRAPHY: ContextVar[TypographyRecord | None] = ContextVar(
    "torchlens_edge_typography", default=None
)


def active_edge_typography() -> TypographyRecord:
    """Return the typography the edge-annotation builders size their text with."""

    record = _EDGE_TYPOGRAPHY.get()
    return DEFAULT_TYPOGRAPHY if record is None else record


def edge_annotation_pt() -> str:
    """Return the active edge-annotation size formatted for DOT emission."""

    return active_edge_typography().annotation_pt


def edge_emphasis_pt() -> str:
    """Return the active emphasized edge-label size formatted for DOT emission."""

    return active_edge_typography().emphasis_pt


def _static_point_size(value: object) -> float | None:
    """Return ``value`` as a positive point size, or ``None`` when it is not one.

    Callables (per-edge overrides), booleans and unparseable strings carry no
    draw-wide size; Graphviz still receives them verbatim as edge attributes.
    """

    if value is None or callable(value) or isinstance(value, bool):
        return None
    try:
        size = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def resolve_edge_typography(
    record: TypographyRecord,
    *,
    edge_overrides: Mapping[str, Any] | None = None,
    font_size: float | None = None,
    theme_edge: Mapping[str, Any] | None = None,
) -> TypographyRecord:
    """Resolve the edge-annotation sizes of one draw from its edge font setting.

    The edge font size is the first static size among ``edge_overrides["fontsize"]``,
    ``font_size`` and the theme's edge ``fontsize``. When one is set, edge
    annotations (pass ranges, argument labels, ``xN`` counts) render at that
    size and the emphasized branch labels (``IF``/``THEN``) keep their ratio to
    it; when none is set the theme's record applies unchanged, so the default
    drawing is byte-identical.
    """

    candidates = (
        (edge_overrides or {}).get("fontsize"),
        font_size,
        (theme_edge or {}).get("fontsize"),
    )
    for candidate in candidates:
        size = _static_point_size(candidate)
        if size is not None:
            return replace(record.scaled(size / record.base_size), annotation_size=size)
    return record


@contextmanager
def edge_typography_for_draw(theme: Any, request: Any) -> Iterator[TypographyRecord]:
    """Hold the edge typography of one forward draw for the builders it calls.

    ``theme`` is the resolved theme (``typography`` and ``edge``); ``request`` is
    the resolved render request (``edge_overrides`` and ``font_size``).
    """

    record = resolve_edge_typography(
        theme.typography,
        edge_overrides=getattr(request, "edge_overrides", None),
        font_size=getattr(request, "font_size", None),
        theme_edge=getattr(theme, "edge", None),
    )
    token = _EDGE_TYPOGRAPHY.set(record)
    try:
        yield record
    finally:
        _EDGE_TYPOGRAPHY.reset(token)
