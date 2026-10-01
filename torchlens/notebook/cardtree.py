"""CardTree: presentation-only HTML card IR + native safe renderer (B1, lane C05).

The treescope-grade card substrate (treescope memo decision 3). CardTree is
presentation STRUCTURE only -- fold state, access expressions, child
budgets, style roles -- and must never become a second semantic fact model:
lovely owns facts/wording/order, themes owns palette, this module owns
folds, layout, budgets, access controls, HTML safety, and document
composition.

Hard rules implemented here (each earned by a measurement in the memo):

- **Typed-leaf escaping**: text is DATA, escaped exactly once at the leaf
  boundary; nothing above the leaf may inject markup.
- **Never-raise boundary**: :func:`safe_card_html` degrades any internal
  failure to a one-line ``card unavailable: <reason>`` -- a repr that raises
  paints a red traceback into the user's cell and can take down third-party
  renderers.
- **Budgeted collections**: a collection renders at most its budget of
  children and DISCLOSES the hidden count; truncation is never silent.
- **Versioned scoped CSS**: every style lives under the ``tl-card-v1`` root
  class, with colors consumed from the themes tokens -- no random global DOM
  ids, no unscoped selectors.
- **``<details>`` folding**: fold state is native HTML, readable with JS
  disabled.
- **Clipboard with click-to-select degradation**: copyable access
  expressions carry a stable ``data-lookup-key`` attribute and a clipboard
  handler; with JS disabled the CSS ``user-select: all`` makes one click
  select the whole expression.

Card GENERATION is stdlib-only by design: the notebook extra gates
integration helpers, never whether ``_repr_html_`` can return safe HTML
(the silent IPython gate is removed -- card tests assert CONTENT).

Every spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from html import escape
from typing import Any

__all__ = [
    "CARD_CSS_VERSION",
    "Card",
    "CardCollection",
    "CardHtml",
    "CardKey",
    "CardSection",
    "CardText",
    "card_css",
    "render_card_html",
    "safe_card_html",
    "trace_overview_card",
]

#: Versioned CSS scope root: bump when the card layout contract changes.
CARD_CSS_VERSION = "tl-card-v1"


@dataclass(frozen=True)
class CardText:
    """One typed text leaf.

    Attributes
    ----------
    text:
        Plain text DATA (labels, module names, exception text). Escaped at
        render; may never carry markup.
    role:
        Style role resolved through the scoped CSS (``"title"``, ``"badge"``,
        ``"fact"``, ``"notice"``, ``"muted"``). Unknown roles render with the
        base class only.
    """

    text: str
    role: str = "fact"


@dataclass(frozen=True)
class CardKey:
    """One copyable access expression.

    Attributes
    ----------
    expression:
        The copyable text (``log['relu_1_2']``), also the stable
        ``data-lookup-key`` value later viewers hydrate against.
    """

    expression: str


@dataclass(frozen=True)
class CardHtml:
    """One TRUSTED pre-rendered fragment from a TorchLens emitter (F16).

    The escaping contract moves, never weakens: the producing emitter
    (e.g. the native array grid) escapes its own data at ITS leaf
    boundary, and the fragment passes through here verbatim. User data
    may NEVER ride this node directly -- that is what :class:`CardText`
    is for.

    Attributes
    ----------
    fragment:
        Already-safe HTML produced by a TorchLens renderer.
    """

    fragment: str


@dataclass(frozen=True)
class CardSection:
    """A foldable titled group backed by native ``<details>``.

    Attributes
    ----------
    title:
        Summary line text (DATA, escaped).
    children:
        Nested card nodes.
    folded:
        Initial fold state; ``False`` renders the section open.
    """

    title: str
    children: tuple[CardNode, ...] = ()
    folded: bool = True


@dataclass(frozen=True)
class CardCollection:
    """A budgeted child sequence with mandatory truncation disclosure.

    Attributes
    ----------
    children:
        All candidate child nodes.
    budget:
        Maximum children rendered; the remainder is disclosed, never
        silently dropped.
    """

    children: tuple[CardNode, ...] = ()
    budget: int = 20


@dataclass(frozen=True)
class Card:
    """One card root: collapsed identity plus zoned content.

    Attributes
    ----------
    title:
        Card identity line (DATA, escaped).
    badge:
        Optional honesty badge text (capture outcome, partial/failure state).
        Honesty states are NEVER folded, so the badge renders in the always-
        visible header.
    children:
        Content zone nodes.
    kind:
        Card kind slug used as an extra CSS class (``"trace"``, ``"partial"``).
    """

    title: str
    badge: str | None = None
    children: tuple[CardNode, ...] = ()
    kind: str = "card"


CardNode = Card | CardCollection | CardHtml | CardKey | CardSection | CardText


@dataclass(frozen=True)
class _ThemeTokens:
    """Resolved cosmetic tokens the scoped CSS consumes."""

    fill: str = "#ffffff"
    border: str = "#d0d7de"
    font: str = "#1f2328"
    muted: str = "#57606a"
    notice: str = "#9a6700"


def _resolve_theme_tokens(theme: str) -> _ThemeTokens:
    """Return cosmetic tokens from the themes presets, defaulting safely."""

    try:
        from ..visualization.themes import resolve_theme

        resolved = resolve_theme(theme)
        return _ThemeTokens(
            fill=resolved.default_fill,
            border=resolved.default_border,
            font=resolved.default_font,
        )
    except Exception:  # noqa: BLE001 - cosmetic resolution may never break a card
        return _ThemeTokens()


def card_css(theme: str = "torchlens") -> str:
    """Return the versioned scoped stylesheet for one theme.

    Every selector is scoped under :data:`CARD_CSS_VERSION`; the palette
    comes from the themes tokens so a skin change is a token change.
    """

    tokens = _resolve_theme_tokens(theme)
    scope = f".{CARD_CSS_VERSION}"
    return (
        "<style>"
        f"{scope}{{border:1px solid {tokens.border};border-radius:8px;"
        f"padding:10px 12px;font-family:system-ui,sans-serif;max-width:640px;"
        f"background:{tokens.fill};color:{tokens.font};font-size:13px}}"
        f"{scope} .tl-card-title{{font-weight:700;margin-bottom:6px}}"
        f"{scope} .tl-card-badge{{display:inline-block;border:1px solid {tokens.border};"
        f"border-radius:6px;padding:0 6px;margin-left:8px;font-weight:600;"
        f"color:{tokens.notice}}}"
        f"{scope} .tl-card-fact{{margin:1px 0}}"
        f"{scope} .tl-card-notice{{color:{tokens.notice};font-weight:600}}"
        f"{scope} .tl-card-muted{{color:{tokens.muted}}}"
        f"{scope} .tl-card-key{{font-family:ui-monospace,monospace;cursor:pointer;"
        f"user-select:all;border:1px dashed {tokens.border};border-radius:4px;"
        "padding:0 4px}"
        f"{scope} details{{margin:4px 0}}"
        f"{scope} summary{{cursor:pointer;font-weight:600}}"
        # Native array grid + six-state motif styles (F16): motifs are
        # PATTERN glyphs, so these classes only set weight/color contrast.
        f"{scope} .tl-grid-table{{border-collapse:collapse;margin:4px 0}}"
        f"{scope} .tl-grid-cell{{width:10px;height:10px;font-size:7px;"
        "text-align:center;padding:0;line-height:10px}"
        f"{scope} .tl-grid-axes{{font-family:ui-monospace,monospace;"
        f"color:{tokens.muted};font-size:11px}}"
        f"{scope} .tl-grid-disclosure{{color:{tokens.muted};font-size:11px}}"
        f"{scope} .tl-motif-nan,{scope} .tl-motif-posinf,{scope} .tl-motif-neginf"
        f"{{font-weight:700;color:{tokens.font}}}"
        f"{scope} .tl-motif-masked,{scope} .tl-motif-unknown{{color:{tokens.muted}}}"
        f"{scope} .tl-motif-oor{{font-weight:700;color:{tokens.notice}}}"
        f"{scope} .tl-card-sentinel{{color:{tokens.muted};font-size:12px}}"
        "</style>"
    )


# Clipboard handler with click-to-select degradation: with JS enabled a
# click copies the stable lookup key; with JS disabled the CSS
# ``user-select: all`` makes the same click select the whole expression.
_CLIPBOARD_ONCLICK = (
    "if(navigator.clipboard){navigator.clipboard.writeText(this.dataset.lookupKey);}"
)


def render_card_html(node: CardNode, *, theme: str = "torchlens", include_css: bool = True) -> str:
    """Render one card tree to an HTML fragment.

    Parameters
    ----------
    node:
        Card tree root.
    theme:
        Themes preset whose tokens the scoped CSS consumes.
    include_css:
        Emit the versioned scoped stylesheet before the fragment. Repeated
        emission is idempotent in notebooks (identical scoped rules).
    """

    prefix = card_css(theme) if include_css else ""
    return f"{prefix}{_render_node(node)}"


def _render_node(node: CardNode) -> str:
    """Serialize one node; text escapes exactly once, here at the leaf."""

    if isinstance(node, CardText):
        role_class = f"tl-card-{node.role}" if node.role else "tl-card-fact"
        return f'<div class="{escape(role_class, quote=True)}">{escape(node.text)}</div>'
    if isinstance(node, CardHtml):
        # Trusted TorchLens-emitter fragment: escaped at ITS leaf boundary.
        return node.fragment
    if isinstance(node, CardKey):
        expression = escape(node.expression, quote=True)
        return (
            f'<code class="tl-card-key" data-lookup-key="{expression}" '
            f'onclick="{escape(_CLIPBOARD_ONCLICK, quote=True)}" '
            f'title="click to copy">{escape(node.expression)}</code>'
        )
    if isinstance(node, CardSection):
        open_attr = "" if node.folded else " open"
        body = "".join(_render_node(child) for child in node.children)
        return f"<details{open_attr}><summary>{escape(node.title)}</summary>{body}</details>"
    if isinstance(node, CardCollection):
        shown = node.children[: max(0, node.budget)]
        body = "".join(_render_node(child) for child in shown)
        hidden = len(node.children) - len(shown)
        if hidden > 0:
            # Truncation is disclosed, never silent.
            body += (
                f'<div class="tl-card-muted">... {hidden} more ({len(node.children)} total)</div>'
            )
        return f"<div>{body}</div>"
    if isinstance(node, Card):
        badge = f'<span class="tl-card-badge">{escape(node.badge)}</span>' if node.badge else ""
        body = "".join(_render_node(child) for child in node.children)
        kind_class = f"tl-card-{node.kind}" if node.kind else ""
        return (
            f'<div class="{CARD_CSS_VERSION} {escape(kind_class, quote=True)}">'
            f'<div class="tl-card-title">{escape(node.title)}{badge}</div>'
            f"{body}</div>"
        )
    # Unknown node types degrade to their repr AS DATA -- never markup.
    return f'<div class="tl-card-muted">{escape(repr(node))}</div>'


def safe_card_html(build: Callable[[], CardNode], *, theme: str = "torchlens") -> str:
    """Build and render a card behind the never-raise boundary.

    Any internal failure degrades to a one-line ``card unavailable``
    fragment naming the exception class -- exception TEXT can carry user
    data, so it is escaped like every other leaf.

    Parameters
    ----------
    build:
        Zero-argument card-tree builder.
    theme:
        Themes preset for the scoped CSS.
    """

    try:
        return render_card_html(build(), theme=theme)
    except Exception as error:  # noqa: BLE001 - the boundary IS the contract
        reason = f"{type(error).__name__}: {error}"
        return (
            f'{card_css("torchlens")}<div class="{CARD_CSS_VERSION}">'
            f'<div class="tl-card-muted">card unavailable: {escape(reason)}</div></div>'
        )


def trace_overview_card(trace: Any) -> Card:
    """Assemble the Trace overview Card from session-safe reads.

    Owned here (not on ``Trace``) so the R43-ledgered ``trace.py`` god file
    carries only the ``_repr_html_`` dispatch; every read stays
    ``getattr``-guarded because the card renders on unfinished and legacy
    logs too.
    """

    layers = len(getattr(trace, "layer_logs", {}) or {})
    ops = getattr(trace, "num_ops", 0)
    save_level = "all" if getattr(trace, "_layers_saved", False) else "selected"
    if getattr(trace, "num_saved_ops", 0) == 0:
        save_level = "metadata only"
    # Text links only: card leaves are DATA and escape at the leaf
    # boundary, so an HTML-formatted link would render as raw markup.
    nonfinite = trace.first_nonfinite(link_format="text")
    title = str(getattr(trace, "trace_label", None) or trace.model_label)
    state = str(getattr(getattr(trace, "state", None), "name", "UNKNOWN"))
    key_index = CardCollection(
        children=tuple(CardKey(f"log[{label!r}]") for label in list(trace.layer_logs or {})[:64]),
        budget=20,
    )
    return Card(
        title=f"TorchLens Trace: {title}",
        badge=state,
        kind="trace",
        children=(
            CardText(f"Layers: {layers}"),
            CardText(f"Ops: {ops}"),
            CardText(f"Save level: {save_level}"),
            CardText(f"NaN/Inf: {nonfinite}"),
            CardSection(title="lookup keys", children=(key_index,), folded=True),
        ),
    )
