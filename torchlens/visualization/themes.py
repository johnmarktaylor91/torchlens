"""Theme presets (cosmetic SKINS) for TorchLens graph rendering.

A skin styles the ink -- palette, ramps, typography -- and composes freely
with the semantic lens axis (``theme_registry``). Each skin carries a LIVE
semantic palette (N4: the rendered node-role colors resolve through the
active skin at :func:`apply_theme_to_spec`, replacing the dead
``legend_items`` data deleted in the same change), a 3-anchor sequential
ramp (low/mid/high, so a diverging map is later a mapping change, not a
schema change), and the declared NEUTRAL "aggregate, not encoded" fill
(N17) for collapsed boxes under an active channel.

FORK-2 (the palette flip to Okabe-Ito) is a maintainer fork: EVERY shipped skin keeps
its pixel-identical historical palette until the fork is resolved (the
render-identity golden pins this); the Okabe-Ito set ships as data with its
accessibility evidence attached, so the flip is a one-line palette swap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from ._typography import DEFAULT_TYPOGRAPHY, HIGH_CONTRAST_TYPOGRAPHY, TypographyRecord
from .node_spec import NodeSpec

#: The Okabe-Ito semantic role palette (memo FORK-2 branch A set). Shipped
#: as DATA only: the flip of any live skin onto this set is a maintainer fork
#: (FORK-2); the CVD accessibility gate measures it now so the fork decision
#: has its evidence attached.
OKABE_ITO_SEMANTIC_PALETTE: Mapping[str, str] = MappingProxyType(
    {
        "input": "#009E73",
        "output": "#D55E00",
        "params_generic": "#CFE5F5",
        "params_trainable": "#56B4E9",
        "params_frozen": "#2E86C1",
        "params_gradient": "#56B4E9:#2E86C1",
        "boolean": "#F0E442",
        "buffer": "#E69F00",
        "intervention": "#CC79A7",
    }
)

#: The historical semantic role palette: the identity mapping of today's
#: rendered constants. EVERY shipped skin keeps pixel-identical output until
#: FORK-2 is resolved (the render-identity golden pins this); the machinery
#: is live, the palettes are unchanged.
LEGACY_SEMANTIC_PALETTE: Mapping[str, str] = MappingProxyType(
    {
        "input": "#98FB98",
        "output": "#ff9999",
        "params_generic": "#E6E6E6",
        "params_trainable": "#D9D9D9",
        "params_frozen": "#B0B0B0",
        "params_gradient": "#D9D9D9:#B0B0B0",
        "boolean": "#F7D460",
        "buffer": "#E69F00",
        "intervention": "#CC79A7",
    }
)

#: Dark-skin semantic palette: byte-identical to the historical dark-theme
#: behavior (params remap to the dark grays; input/output/boolean pass
#: through unchanged, exactly as before the palette went live).
_DARK_SEMANTIC_PALETTE: Mapping[str, str] = MappingProxyType(
    {
        "input": "#98FB98",
        "output": "#ff9999",
        "params_generic": "#374151",
        "params_trainable": "#374151",
        "params_frozen": "#374151",
        "params_gradient": "#374151:#4B5563",
        "boolean": "#F7D460",
        "buffer": "#E69F00",
        "intervention": "#CC79A7",
    }
)


#: Theme fields deleted outright (lane F12; no alias kept).
_REMOVED_THEME_MEMBERS: dict[str, str] = {
    "legend_items": (
        "legends now derive from the active encoding channels; style them through "
        "VisualizationTheme.semantic_palette and ramp -- legend_items was deleted"
    ),
}


@dataclass(frozen=True)
class VisualizationTheme:
    """Resolved visual skin values for graph renderers.

    Parameters
    ----------
    name:
        Public skin preset name.
    graph:
        Graphviz graph attributes.
    node:
        Graphviz node attributes.
    edge:
        Graphviz edge attributes.
    default_fill:
        Default operation-node fill color.
    default_border:
        Default operation-node border color.
    default_font:
        Default operation-node font color.
    semantic_palette:
        LIVE role -> hex mapping consumed by :func:`apply_theme_to_spec`;
        the role vocabulary is the key set of
        :data:`LEGACY_SEMANTIC_PALETTE`.
    ramp:
        Sequential 3-anchor ramp (low, mid, high). The low anchor may never
        equal ``default_fill`` (the measured white-on-white absence defect:
        an encoded-lowest node indistinguishable from an unencoded one).
    neutral_aggregate_fill:
        The declared "aggregate, not encoded" fill for collapsed boxes and
        fold ellipses while a channel is active (N17) -- never a fill that
        already carries a semantic meaning elsewhere in the same picture.
    typography:
        The one typography record (vizmech D29): pinned family plus semantic
        size roles. Attribute builders inject the family into any scope the
        preset dicts leave unset, so no theme can regress to the engine's
        serif fallback.
    """

    name: str
    graph: dict[str, str]
    node: dict[str, str]
    edge: dict[str, str]
    default_fill: str
    default_border: str
    default_font: str
    # default_factory, not default: Python 3.11 dataclasses reject the unhashable
    # mappingproxy as a field default. The factory returns the one shared
    # read-only proxy, so the default stays identical and immutable.
    semantic_palette: Mapping[str, str] = field(default_factory=lambda: LEGACY_SEMANTIC_PALETTE)
    ramp: tuple[str, str, str] = ("#F2F2F2", "#79A8CC", "#0072B2")
    neutral_aggregate_fill: str = "#E8EEF2"
    typography: TypographyRecord = field(default=DEFAULT_TYPOGRAPHY)

    # Runtime only: under TYPE_CHECKING the hook would make every attribute
    # type-check as Any, hiding typos and the removed spellings from mypy.
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            """Name the replacement for a removed public member, else fail as usual."""

            from ..utils.facade import refuse_removed_member

            refuse_removed_member("VisualizationTheme", name, _REMOVED_THEME_MEMBERS)
            return object.__getattribute__(self, name)


# Every preset pins a font family on ALL THREE scopes (graph covers cluster
# captions and the graph caption). Before vizmech D29 the default preset
# pinned nothing, making it the only serif theme (Graphviz falls back to
# Times) and the theme sweep alone moved measured label geometry (59 -> 68
# violations). The family choice is fork FK3; the pinning is decided.
THEME_PRESETS: dict[str, VisualizationTheme] = {
    "torchlens": VisualizationTheme(
        name="torchlens",
        graph={"bgcolor": "white", "fontname": "Helvetica"},
        node={"fontname": "Helvetica"},
        edge={"fontname": "Helvetica"},
        default_fill="white",
        default_border="black",
        default_font="black",
        semantic_palette=LEGACY_SEMANTIC_PALETTE,
        ramp=("#F2F2F2", "#79A8CC", "#0072B2"),
        neutral_aggregate_fill="#E8EEF2",
    ),
    "paper": VisualizationTheme(
        name="paper",
        graph={"bgcolor": "white", "colorscheme": "paired12", "fontname": "Helvetica"},
        node={"fontname": "Helvetica"},
        edge={"fontname": "Helvetica"},
        default_fill="#F7F7F7",
        default_border="#222222",
        default_font="#111111",
        semantic_palette=LEGACY_SEMANTIC_PALETTE,
        # Grayscale ramp (review r3 working anchors): print-safe by design.
        ramp=("#F0F0F0", "#969696", "#252525"),
        neutral_aggregate_fill="#E8EEF2",
    ),
    "dark": VisualizationTheme(
        name="dark",
        graph={"bgcolor": "#111827", "fontname": "Helvetica"},
        node={"fontname": "Helvetica"},
        edge={"color": "#9CA3AF", "fontcolor": "#D1D5DB", "fontname": "Helvetica"},
        default_fill="#1F2937",
        default_border="#E5E7EB",
        default_font="#F9FAFB",
        semantic_palette=_DARK_SEMANTIC_PALETTE,
        ramp=("#2A3646", "#3F7CA6", "#56B4E9"),
        neutral_aggregate_fill="#2A3441",
    ),
    "colorblind": VisualizationTheme(
        name="colorblind",
        graph={"bgcolor": "white", "fontname": "Helvetica"},
        node={"fontname": "Helvetica"},
        edge={"fontname": "Helvetica"},
        default_fill="#F0F0F0",
        default_border="#0072B2",
        default_font="#111111",
        semantic_palette=LEGACY_SEMANTIC_PALETTE,
        ramp=("#F2F2F2", "#79A8CC", "#0072B2"),
        neutral_aggregate_fill="#E8EEF2",
    ),
    "high_contrast": VisualizationTheme(
        name="high_contrast",
        graph={"bgcolor": "white", "fontname": "Helvetica-Bold"},
        node={"fontname": "Helvetica-Bold"},
        edge={
            "color": "black",
            "fontcolor": "black",
            "penwidth": "2",
            "fontname": "Helvetica-Bold",
        },
        default_fill="white",
        default_border="black",
        default_font="black",
        semantic_palette=LEGACY_SEMANTIC_PALETTE,
        ramp=("#EBEBEB", "#767676", "#000000"),
        neutral_aggregate_fill="#E8EEF2",
        typography=HIGH_CONTRAST_TYPOGRAPHY,
    ),
}


#: FORK-5 two-step flip switch (megaplan; collapse memo item 10): the
#: kind-word title rows, K3 peripheries, and reuse accents ship as machinery
#: behind this flag until the legibility protocol (collapse memo item 11,
#: run by the themes lane's evaluator battery) ratifies the default flip.
#: The K4/K3 THEMING below is a bug fix (hardcoded grays ignored dark
#: themes) and ships unconditionally.
COLLAPSE_KIND_TOKENS_DEFAULT = False


@dataclass(frozen=True)
class CollapseTokens:
    """Per-kind visual tokens for collapsed units (collapse memo D10).

    ONE table holds every kind's marks and wording so the naming sprint
    renames in one place and a compact-label mode is a table change (memo
    plumbing note). The wording CONTENT is panel-fixed: kind, honest counts,
    and the honesty disclosures ("different", never a sameness claim across
    a segment; "separate instances", never "distinct weights"). The exact
    speech-friendly kind names are protocol-tested variants owned by the
    UI sprint.

    Parameters
    ----------
    segment_fill / segment_border / segment_font:
        K4/K5 capsule colors (rounded dashed family), theme-derived.
    ellipsis_fill / ellipsis_border / ellipsis_font:
        K3 elision-chip colors (dashed family, bound to its representative).
    reuse_glyph:
        The in-card circular-arrow reuse cue (K1/K1u thumbnail-surviving
        mark).
    word_reused / word_box / word_fold / word_segment / word_pattern:
        Title-row wording templates per kind (zero new label rows).
    label_rows:
        Per-kind EXTRA label-row budget (compact mode = lower the numbers).
    """

    segment_fill: str
    segment_border: str
    segment_font: str
    ellipsis_fill: str
    ellipsis_border: str
    ellipsis_font: str
    reuse_glyph: str = "\u21ba"
    word_reused: str = "reused x{n}"
    word_box: str = "{n} ops inside"
    word_fold: str = "1 of {n} shown"
    word_segment: str = "{n} different ops"
    word_pattern: str = "PATTERN '{name}' -- {n} ops"
    label_rows: tuple[tuple[str, int], ...] = (
        ("reused", 0),
        ("box", 0),
        ("fold", 0),
        ("segment", 0),
        ("pattern", 0),
    )


def collapse_tokens(theme: VisualizationTheme) -> CollapseTokens:
    """Return the theme-derived collapse tokens (fixes the hardcode bug).

    K3's ellipsis (#777777) and K4's fills (#f7f7f7 family) were hardcoded
    and ignored dark themes (collapse memo section 2); both now derive from
    the active theme.
    """

    if theme.name == "dark":
        return CollapseTokens(
            segment_fill="#1F2937",
            segment_border="#9CA3AF",
            segment_font="#F9FAFB",
            ellipsis_fill="#111827",
            ellipsis_border="#9CA3AF",
            ellipsis_font="#D1D5DB",
        )
    return CollapseTokens(
        segment_fill="#f7f7f7",
        segment_border="#666666",
        segment_font="#222222",
        ellipsis_fill="white",
        ellipsis_border="#777777",
        ellipsis_font="#555555",
    )


def resolve_theme(theme: str, *, for_paper: bool = False) -> VisualizationTheme:
    """Return a supported visualization theme preset.

    Parameters
    ----------
    theme:
        Theme preset name.
    for_paper:
        Whether to force the paper preset.

    Returns
    -------
    VisualizationTheme
        Resolved theme preset.

    Raises
    ------
    ValueError
        If the requested theme is unknown.
    """

    resolved_name = "paper" if for_paper else theme
    if resolved_name not in THEME_PRESETS:
        supported = ", ".join(sorted(THEME_PRESETS))
        raise InvalidArgumentError(
            f"Unsupported visualization theme {resolved_name!r}; choose one of {supported}",
            code="visualization_theme_invalid",
            remedy=f"pass one of the supported themes ({supported})",
            argument="theme",
        )
    return THEME_PRESETS[resolved_name]


#: Reverse map from the rendered module constants to semantic ROLES: the one
#: table that makes every skin's palette LIVE (N4). The constants stay the
#: renderer-side spelling; the skin owns the pixels.
_CONSTANT_ROLES: dict[str, str] = {
    "#98FB98": "input",
    "#ff9999": "output",
    "#E6E6E6": "params_generic",
    "#D9D9D9": "params_trainable",
    "#B0B0B0": "params_frozen",
    "#F7D460": "boolean",
}


def _skin_fill_for(spec_fill: str | None, theme: VisualizationTheme) -> str | None:
    """Translate a semantic-constant fill through the skin's live palette.

    Colon-list gradient fills translate segment-wise; unrecognized fills
    pass through untouched (a user's explicit color is never overridden).
    Returns ``None`` when no palette translation applies.
    """

    if spec_fill is None:
        return None
    fill_text = str(spec_fill)
    if fill_text == "#D9D9D9:#B0B0B0":
        # The trainable:frozen gradient translates as ONE role so a skin can
        # keep a two-tone gradient (the historical dark remap).
        return theme.semantic_palette.get("params_gradient", fill_text)
    segments = fill_text.split(":")
    roles = [_CONSTANT_ROLES.get(segment) for segment in segments]
    if not any(roles):
        return None
    translated = [
        theme.semantic_palette.get(role, segment) if role is not None else segment
        for role, segment in zip(roles, segments, strict=True)
    ]
    return ":".join(translated)


def apply_theme_to_spec(spec: NodeSpec, theme: VisualizationTheme) -> NodeSpec:
    """Apply theme defaults to a node spec without replacing explicit styles.

    Semantic role fills (input/output/params/boolean constants) resolve
    through the skin's LIVE palette here -- the N4 wiring: every rendered
    node spec passes through this seam, so a skin actually themes the
    colours that matter. The default skin's palette is the identity mapping
    of the historical constants (FORK-2, the flip, is a maintainer fork).

    Parameters
    ----------
    spec:
        Node spec to style.
    theme:
        Resolved theme preset.

    Returns
    -------
    NodeSpec
        Themed copy of ``spec``.
    """

    fillcolor: str | None
    palette_fill = _skin_fill_for(spec.fillcolor, theme)
    if palette_fill is not None:
        fillcolor = palette_fill
    else:
        fillcolor = theme.default_fill if spec.fillcolor in {None, "white"} else spec.fillcolor
    fontcolor = theme.default_font if spec.fontcolor in {None, "black"} else spec.fontcolor
    color = theme.default_border if spec.color in {None, "black"} else spec.color
    return spec.replace(fillcolor=fillcolor, fontcolor=fontcolor, color=color)


def theme_graph_attrs(
    theme: VisualizationTheme,
    *,
    font_size: int | None = None,
    dpi: int | None = None,
    fileformat: str | None = None,
) -> dict[str, str]:
    """Build graph-level attributes for a theme and convenience knobs.

    Parameters
    ----------
    theme:
        Resolved theme preset.
    font_size:
        Optional graph font size.
    dpi:
        Optional output DPI. RASTER-ONLY (vizmech D23): on vector formats
        graphviz's ``dpi`` attribute multiplies the coordinate space itself,
        so ``draw(dpi=300)`` produced a 36-inch PDF page and an SVG whose
        ``viewBox`` and declared size disagreed. When ``fileformat`` names a
        vector format the knob is dropped; raster pixels still scale
        linearly with it.
    fileformat:
        Output format the attributes will render to. ``None`` (unknown)
        conservatively treats the target as vector and drops ``dpi``.

    Returns
    -------
    dict[str, str]
        Graphviz graph attributes.
    """

    from .render_execution import is_raster_format

    attrs: dict[str, str] = dict(theme.graph)
    attrs.setdefault("fontname", theme.typography.family)
    if font_size is not None:
        attrs["fontsize"] = str(font_size)
    if dpi is not None and fileformat is not None and is_raster_format(fileformat):
        attrs["dpi"] = str(dpi)
    return attrs


def theme_node_attrs(theme: VisualizationTheme, *, font_size: int | None = None) -> dict[str, str]:
    """Build node-level attributes for a theme and font-size knob.

    Parameters
    ----------
    theme:
        Resolved theme preset.
    font_size:
        Optional node font size.

    Returns
    -------
    dict[str, str]
        Graphviz node attributes.
    """

    attrs: dict[str, str] = dict(theme.node)
    attrs.setdefault("fontname", theme.typography.family)
    if font_size is not None:
        attrs["fontsize"] = str(font_size)
    return attrs


def theme_edge_attrs(theme: VisualizationTheme, *, font_size: int | None = None) -> dict[str, str]:
    """Build edge-level attributes for a theme and font-size knob.

    Parameters
    ----------
    theme:
        Resolved theme preset.
    font_size:
        Optional edge font size.

    Returns
    -------
    dict[str, str]
        Graphviz edge attributes.
    """

    attrs: dict[str, str] = dict(theme.edge)
    attrs.setdefault("fontname", theme.typography.family)
    if font_size is not None:
        attrs["fontsize"] = str(font_size)
    return attrs
