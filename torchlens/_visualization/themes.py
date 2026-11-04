"""Theme presets (cosmetic SKINS) for TorchLens graph rendering.

A skin styles the ink -- palette, ramps, typography -- and composes freely
with the semantic lens axis (``theme_registry``). Each skin carries a LIVE
semantic palette (N4: the rendered node-role colors resolve through the
active skin at :func:`apply_theme_to_spec`, replacing the dead
``legend_items`` data deleted in the same change), a 3-anchor sequential
ramp (low/mid/high, so a diverging map is later a mapping change, not a
schema change), and the declared NEUTRAL "aggregate, not encoded" fill
(N17) for collapsed boxes under an active channel.

The default ``torchlens`` skin and the ``colorblind`` skin render the
Okabe-Ito role colours (input green, output vermillion, plain operations the
page colour) with grey weights (light grey when trainable, dark grey when
frozen), the set that passes every colour-vision gate. No skin draws blue:
weights are grey in every skin and the sequential ramps are warm. The ``paper``
and ``high_contrast`` skins keep the legacy TorchLens role colours (input light
green, output light red, weights light grey when trainable and dark grey when
frozen) and ``dark`` its dark remap.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from .._vocab.node_spec import NodeSpec
from ._typography import DEFAULT_TYPOGRAPHY, HIGH_CONTRAST_TYPOGRAPHY, TypographyRecord

#: The Okabe-Ito semantic role palette (memo FORK-2 branch A set) with the
#: ruled grey weights (no blue in any skin: trainable light grey, frozen dark
#: grey, generic a paler neutral grey). Each grey pair sits at least 40 apart
#: under every colour-vision simulation; the frozen grey keeps black ink
#: legible (5.3:1) so the trainable/frozen split fill reads with one ink; the
#: generic grey sits 13 dE from the white page. The default ``torchlens`` skin
#: and the ``colorblind`` skin render it; the CVD accessibility gate measures it.
OKABE_ITO_SEMANTIC_PALETTE: Mapping[str, str] = MappingProxyType(
    {
        "input": "#009E73",
        "output": "#D55E00",
        "params_generic": "#DADADA",
        "params_trainable": "#C0C0C0",
        "params_frozen": "#808080",
        "params_gradient": "#C0C0C0:#808080",
        "boolean": "#F0E442",
        "buffer": "#E69F00",
        "intervention": "#CC79A7",
    }
)

#: The historical semantic role palette: the identity mapping of the rendered
#: constants (the role vocabulary). The ``paper`` and ``high_contrast`` skins
#: render it; it stays the field default of :class:`VisualizationTheme`.
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

#: Dark-skin semantic palette: the historical dark-theme role colours for
#: input, output and boolean (passed through unchanged) with the ruled grey
#: weights: trainable light grey, frozen dark grey, as on the light page. The
#: trainable grey stays at least 40 from the light input green under every
#: colour-vision simulation, and the frozen grey keeps black ink legible so
#: the trainable/frozen split fill reads with one ink. The dark plain fill is
#: the page colour, so a plain operation and a parameterised one separate.
_DARK_SEMANTIC_PALETTE: Mapping[str, str] = MappingProxyType(
    {
        "input": "#98FB98",
        "output": "#ff9999",
        "params_generic": "#DADADA",
        "params_trainable": "#DADADA",
        "params_frozen": "#808080",
        "params_gradient": "#DADADA:#808080",
        "boolean": "#F7D460",
        "buffer": "#E69F00",
        "intervention": "#CC79A7",
    }
)


#: Warm sequential ramps (low, mid, high; ColorBrewer YlOrBr anchors): no blue
#: and no purple in any skin. Luminance falls monotonically on the light page
#: (243, 170, 77) and rises on the dark one (48, 121, 220), and each passes the
#: colour-vision gate.
LIGHT_WARM_RAMP: tuple[str, str, str] = ("#FFF7BC", "#FE9929", "#993404")
DARK_WARM_RAMP: tuple[str, str, str] = ("#3A2E22", "#C0661E", "#FED98E")
#: The dark skin's neutral near-black page (no blue tint).
DARK_PAGE = "#141414"
#: The declared "aggregate, not encoded" fills (N17): a pale warm stone on the
#: light skins and a dark warm stone on the dark one, never a role colour.
LIGHT_AGGREGATE_FILL = "#EDE7DE"
DARK_AGGREGATE_FILL = "#2E2A25"

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
    ramp: tuple[str, str, str] = LIGHT_WARM_RAMP
    neutral_aggregate_fill: str = LIGHT_AGGREGATE_FILL
    typography: TypographyRecord = field(default=DEFAULT_TYPOGRAPHY)

    # Runtime only: under TYPE_CHECKING the hook would make every attribute
    # type-check as Any, hiding typos and the removed spellings from mypy.
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            """Name the replacement for a removed public member, else fail as usual."""

            from .._utils.facade import refuse_removed_member

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
        semantic_palette=OKABE_ITO_SEMANTIC_PALETTE,
        ramp=LIGHT_WARM_RAMP,
        neutral_aggregate_fill=LIGHT_AGGREGATE_FILL,
    ),
    "paper": VisualizationTheme(
        name="paper",
        graph={"bgcolor": "white", "fontname": "Helvetica"},
        node={"fontname": "Helvetica"},
        edge={"fontname": "Helvetica"},
        default_fill="#F7F7F7",
        default_border="#222222",
        default_font="#111111",
        semantic_palette=LEGACY_SEMANTIC_PALETTE,
        # Grayscale ramp (review r3 working anchors): print-safe by design.
        ramp=("#F0F0F0", "#969696", "#252525"),
        neutral_aggregate_fill=LIGHT_AGGREGATE_FILL,
    ),
    "dark": VisualizationTheme(
        name="dark",
        # Neutral near-black page and neutral grey inks: no blue-tinted greys.
        graph={"bgcolor": DARK_PAGE, "fontname": "Helvetica"},
        node={"fontname": "Helvetica"},
        edge={"color": "#A3A3A3", "fontcolor": "#D4D4D4", "fontname": "Helvetica"},
        # The plain fill is the page, as in the default skin, so a plain
        # operation and a parameterised one separate.
        default_fill=DARK_PAGE,
        default_border="#E5E5E5",
        default_font="#FAFAFA",
        semantic_palette=_DARK_SEMANTIC_PALETTE,
        ramp=DARK_WARM_RAMP,
        neutral_aggregate_fill=DARK_AGGREGATE_FILL,
    ),
    "colorblind": VisualizationTheme(
        name="colorblind",
        graph={"bgcolor": "white", "fontname": "Helvetica"},
        node={"fontname": "Helvetica"},
        edge={"fontname": "Helvetica"},
        # The plain fill is the page, as in the default skin: ``#F0F0F0`` sat
        # 8 dE from the trainable-parameter gray, under the 12 dE floor.
        default_fill="white",
        default_border="#111111",
        default_font="#111111",
        semantic_palette=OKABE_ITO_SEMANTIC_PALETTE,
        ramp=LIGHT_WARM_RAMP,
        neutral_aggregate_fill=LIGHT_AGGREGATE_FILL,
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
        neutral_aggregate_fill=LIGHT_AGGREGATE_FILL,
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
    """Return the theme-derived collapse tokens.

    Only the ``segment_*`` tokens are read today: K4's segment fills derive
    from the active theme. The ``ellipsis_*``, ``reuse_glyph``, ``word_*``
    and ``label_rows`` tokens are carried but unused; the run-fold ellipsis
    still draws its fixed ``#777777`` font colour in every theme.
    """

    if theme.name == "dark":
        return CollapseTokens(
            segment_fill="#262626",
            segment_border="#A3A3A3",
            segment_font="#FAFAFA",
            ellipsis_fill=DARK_PAGE,
            ellipsis_border="#A3A3A3",
            ellipsis_font="#D4D4D4",
        )
    return CollapseTokens(
        segment_fill="#f7f7f7",
        segment_border="#666666",
        segment_font="#222222",
        ellipsis_fill="white",
        ellipsis_border="#777777",
        ellipsis_font="#555555",
    )


def resolve_theme(theme: str) -> VisualizationTheme:
    """Return a supported visualization theme preset.

    Parameters
    ----------
    theme:
        Theme preset name.

    Returns
    -------
    VisualizationTheme
        Resolved theme preset.

    Raises
    ------
    ValueError
        If the requested theme is unknown.
    """

    resolved_name = theme
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
    # A False test: the skin's boolean colour blended toward its page (no
    # palette key of its own; see ``_role_fill``).
    "#FBEAB0": "boolean_false",
    # Buffers fill with the palette's buffer role. They used to fill with the
    # plain-op default, so the legend's buffer swatch and a plain op looked
    # the same in every skin; the shape was the only cue.
    "#E69F00": "buffer",
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
    # A segment may carry a Graphviz weight (``#FBEAB0;0.67``): translate the
    # colour and keep the weight.
    segments = [segment.partition(";") for segment in fill_text.split(":")]
    roles = [_CONSTANT_ROLES.get(color) for color, _, _ in segments]
    if not any(roles):
        return None
    translated = [
        (_role_fill(theme, role, color) if role is not None else color) + sep + weight
        for role, (color, sep, weight) in zip(roles, segments, strict=True)
    ]
    return ":".join(translated)


#: Share of the boolean colour in a False test's fill (the rest is white).
_FALSE_TEST_SHARE = 0.45


def _role_fill(theme: VisualizationTheme, role: str, constant: str) -> str:
    """Return the skin's fill for one role, deriving the False-test pale.

    The pale is the boolean colour mixed toward white in every skin: mixing
    toward the dark skin's page gave a dark olive that black ink could not
    read on, and white keeps both halves of a mixed test light.
    """

    if role != "boolean_false":
        return theme.semantic_palette.get(role, constant)
    boolean = _hex_rgb(theme.semantic_palette.get("boolean", "#F7D460"))
    if boolean is None:
        return constant
    mixed = (round(255 + (boolean[i] - 255) * _FALSE_TEST_SHARE) for i in range(3))
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def _hex_rgb(color: str) -> tuple[int, int, int] | None:
    """Return ``#RRGGBB`` (or ``white``/``black``) as an RGB triple, else ``None``."""

    text = {"white": "#FFFFFF", "black": "#000000"}.get(color.strip().lower(), color.strip())
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        return None
    return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)


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
    fontcolor: str | None
    if spec.fontcolor in {None, "black"}:
        fontcolor = _legible_ink(fillcolor, theme.default_font)
    else:
        fontcolor = spec.fontcolor
    color = theme.default_border if spec.color in {None, "black"} else spec.color
    return spec.replace(fillcolor=fillcolor, fontcolor=fontcolor, color=color)


#: WCAG AA contrast floor for normal-size text.
_MIN_TEXT_CONTRAST = 4.5
#: Fallback inks when a skin's default text colour is unreadable on a fill.
_FALLBACK_INKS = ("#111111", "#FFFFFF")


def _relative_luminance(color: str) -> float | None:
    """Return the WCAG relative luminance of ``#RRGGBB`` or ``white``/``black``."""

    text = {"white": "#FFFFFF", "black": "#000000"}.get(color.strip().lower(), color.strip())
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        return None
    channels = []
    for index in (1, 3, 5):
        value = int(text[index : index + 2], 16) / 255.0
        channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(color_a: str, color_b: str) -> float | None:
    """Return the WCAG contrast ratio of two colours, or ``None`` if either is unparseable."""

    lum_a, lum_b = _relative_luminance(color_a), _relative_luminance(color_b)
    if lum_a is None or lum_b is None:
        return None
    high, low = max(lum_a, lum_b), min(lum_a, lum_b)
    return (high + 0.05) / (low + 0.05)


def _worst_contrast(ink: str, fill: str) -> float | None:
    """Return ``ink``'s lowest contrast over a fill's gradient segments, or ``None``."""

    segments = [segment.partition(";")[0] for segment in fill.split(":")]
    ratios = [_contrast_ratio(ink, segment) for segment in segments]
    known = [ratio for ratio in ratios if ratio is not None]
    if not known or len(known) != len(ratios):
        return None
    return min(known)


def _legible_ink(fill: str | None, ink: str) -> str:
    """Return ``ink``, or a fallback ink when ``ink`` is unreadable on ``fill``.

    A skin's default text colour suits its default fill; on a semantic fill
    of the opposite lightness (the dark skin's white text on the light input,
    output, boolean and buffer fills) it fell to about 1.2:1. Gradient fills
    are judged by their worst segment; an unparseable fill keeps ``ink``.
    """

    if fill is None:
        return ink
    current = _worst_contrast(ink, str(fill))
    if current is None or current >= _MIN_TEXT_CONTRAST:
        return ink
    best = max(_FALLBACK_INKS, key=lambda candidate: _worst_contrast(candidate, str(fill)) or 0.0)
    return best if (_worst_contrast(best, str(fill)) or 0.0) > current else ink


def _theme_line_ink(theme: VisualizationTheme, color: str | None) -> str | None:
    """Return the skin's edge colour for the renderer's default black edge ink."""

    if color in (None, "black"):
        return theme.edge.get("color", color)
    return color


def _theme_annotation_ink(theme: VisualizationTheme, color: str | None) -> str | None:
    """Return the skin's edge-label colour for the renderer's default black label ink."""

    if color in (None, "black"):
        return theme.edge.get("fontcolor", color)
    return color


def _edge_and_label_inks(theme: VisualizationTheme | None, node_color: str) -> tuple[str, str]:
    """Return one node's (edge ink, edge-label ink).

    Edges and their labels keep the renderer's black only on skins that
    leave edge ink unset; the dark skin's edge colours apply here.
    """

    if theme is None:
        return node_color, node_color
    edge_ink = _theme_line_ink(theme, node_color) or node_color
    label_ink = _theme_annotation_ink(theme, node_color) or node_color
    return edge_ink, label_ink


def _theme_cluster_attrs(theme: VisualizationTheme) -> dict[str, str]:
    """Return the module-box ink a skin implies: page fill, line border, label caption.

    Module boxes are drawn as white boxes with Graphviz's default black ink.
    That suits the white-page skins and was the dark skin's defect (white
    boxes and black borders on a near-black page). Only attributes that
    differ from that historical box are returned, so white-page skins emit
    the same DOT as before.
    """

    attrs: dict[str, str] = {}
    page = theme.graph.get("bgcolor", "white")
    if page.lower() != "white":
        attrs["fillcolor"] = page
    border = theme.edge.get("color")
    if border is not None and border.lower() != "black":
        attrs["color"] = border
    caption = theme.edge.get("fontcolor")
    if caption is not None and caption.lower() != "black":
        attrs["fontcolor"] = caption
    return attrs


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
