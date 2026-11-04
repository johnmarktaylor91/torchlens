"""Notebook cards for traces, records and failed captures.

Op and Layer cards delegate to the same bounded facts as their text cards.
Trace and PartialTrace retain their capture-proof and failure-context zones.
Every HTML entry point uses the never-raise, escaping CardTree boundary.
"""

from __future__ import annotations

from typing import Any

from ..._data_classes._trace_missing_attrs import is_live_trace
from ._card_suppression import card_render_suppressed
from ._value_cards import _from_text, layer_value_card, op_value_card
from .cardtree import (
    Card,
    CardCollection,
    CardNode,
    CardSection,
    CardText,
    safe_card_html,
)

__all__ = [
    "BRIDGE_SENTINEL_HTML",
    "layer_card",
    "layer_repr_html",
    "op_card",
    "op_repr_html",
    "partial_trace_card",
    "partial_repr_html",
    "trace_card",
    "trace_repr_html",
]

#: One-line sentinel returned instead of the full card when the treescope
#: bridge just rendered this object (memo 3.6; wording is [UI-SPRINT]).
BRIDGE_SENTINEL_HTML = (
    '<div class="tl-card-sentinel">card omitted: rendered by the treescope bridge above</div>'
)


def _bridge_sentinel(obj: Any) -> str | None:
    """Return the suppression sentinel when the bridge claims this render.

    The treescope bridge registers its predicate in the card-suppression slot
    when it loads; the card never imports the bridge. Fails SAFE: no bridge
    loaded or a predicate fault returns ``None`` and the full card renders --
    a broken sniff may never suppress the user's only good rendering.
    """

    if card_render_suppressed(obj):
        return BRIDGE_SENTINEL_HTML
    return None


def _stats_line(tensor: Any) -> str | None:
    """Render the lovely core stats line for one resident tensor payload.

    The stats package resolves DYNAMICALLY: cards are reachable from
    ``Trace._repr_html_`` and the stats package's aggregate module reads
    the root facade, so a static edge here closes an import cycle that
    breaks type analysis of ``torchlens.stats._aggregate``.
    """

    try:
        import importlib

        stats = importlib.import_module("torchlens.stats")
        render = importlib.import_module("torchlens.stats._stats_render")
        return str(render.render_core_line(stats.tensor_stats(tensor)))
    except Exception:  # noqa: BLE001 - stats are cosmetic on a card; degrade
        return None


def _proof_section(trace_like: Any) -> CardSection:
    """Build the PROOF row: outcome, verification, coverage, save policy.

    Rendered OPEN (honesty is never folded).
    """

    facts: list[CardNode] = []
    outcome = getattr(trace_like, "outcome", None)
    status = getattr(getattr(outcome, "status", None), "name", None)
    facts.append(CardText(f"capture outcome: {status or 'UNKNOWN'}"))
    verified = trace_like._provenance_verified if is_live_trace(trace_like) else None
    if verified is False:
        reason = (
            trace_like._provenance_verification_reason if is_live_trace(trace_like) else None
        ) or "unverified"
        facts.append(CardText(f"capture NOT verified: {reason}", role="notice"))
    elif verified is True:
        facts.append(CardText("capture verified"))
    coverage = getattr(trace_like, "nonfinite_coverage", None)
    if coverage is not None:
        facts.append(CardText(f"nonfinite coverage: {coverage}"))
    total, saved = _save_counts(trace_like)
    if total is not None and saved is not None:
        facts.append(CardText(f"save policy: {saved}/{total} op records saved"))
    return CardSection(title="capture proof", children=tuple(facts), folded=False)


def _save_counts(trace_like: Any) -> tuple[int | None, int | None]:
    """(total op records, saved op records) on ONE consistent basis.

    ``len(trace.ops)`` counts only real computation ops while ``len(trace.saved_ops)``
    counts every saved record including input/output boundaries -- mixing
    them printed ``saved 4/2``. The record count is the honest denominator.
    """

    try:
        total: int | None = len(tuple(trace_like.op_labels or ()))
    except Exception:  # noqa: BLE001 - unfinished/legacy logs may lack labels
        total = None
    saved = len(trace_like.saved_ops) if hasattr(trace_like, "saved_ops") else None
    return total, saved


def op_card(op: Any, *, root: str | None = None, kind: str = "op") -> Card:
    """Render the same identity and rows as the bounded text Op card."""

    return op_value_card(op, kind=kind)


def layer_card(layer: Any, *, root: str | None = None) -> Card:
    """Render the same per-pass facts as the bounded text Layer card."""

    return layer_value_card(layer)


def trace_card(trace: Any, *, root: str | None = None) -> Card:
    """The same title and fact rows as ``print(trace)``, without analysis or a tree."""

    return _from_text(str(trace), "trace")


def _first_nonfinite_text(trace: Any) -> str | None:
    """Text-format first-nonfinite disclosure, degrade-safe."""

    try:
        return str(trace.first_nonfinite(link_format="text"))
    except Exception:  # noqa: BLE001 - the card renders on unfinished logs too
        return None


def _module_index(trace: Any) -> tuple[CardNode, ...]:
    """Budgeted mini module table (top collapse-order rows)."""

    try:
        order = list(getattr(trace, "module_collapse_order", ()) or ())[:10]
    except Exception:  # noqa: BLE001 - collapse metadata is optional
        order = []
    if not order:
        return ()
    rows = tuple(CardText(str(address), role="muted") for address in order)
    return (CardSection(title="modules", children=(CardCollection(rows, budget=10),)),)


def partial_trace_card(partial: Any) -> Card:
    """Assemble the failure-first PartialTrace card.

    The card treescope structurally cannot have: it renders an execution
    that did not finish -- persistent failure banner, phase/reason,
    escaped exception, committed prefix and frontier, known/unknown
    evidence, and what remains safe to inspect.
    """

    outcome = getattr(partial, "outcome", None)
    phase = getattr(getattr(outcome, "phase", None), "name", None)
    facts: list[CardNode] = []
    if phase:
        facts.append(CardText(f"failed during: {phase}", role="notice"))
    error = getattr(partial, "original_exception", None)
    if error is not None:
        facts.append(CardText(f"error: {type(error).__name__}: {error}", role="notice"))
    raw_layers = list(getattr(partial, "raw_layers", ()) or ())
    facts.append(CardText(f"committed prefix: {len(raw_layers)} raw layers"))
    frontier = [_frontier_label(record) for record in raw_layers[-3:]]
    if frontier:
        facts.append(CardText(f"frontier (last committed): {', '.join(frontier)}"))
    facts.append(CardText(_partial_nonfinite(partial)))
    facts.append(
        CardText(
            "safe to inspect: raw_layers, audit(), draw(); "
            "replay/validation refuse on failed captures",
            role="muted",
        )
    )
    facts.append(_proof_section(partial))
    return Card(
        title="TorchLens PartialTrace",
        badge="FAILED CAPTURE",
        kind="partial",
        children=tuple(facts),
    )


def _frontier_label(record: Any) -> str:
    """Return one committed record's display label, never its full dump.

    Uses the partial package's own display rule (canonical label when
    finalized, raw identity otherwise); a record's ``str()`` prints its
    tensor contents, which a card may never inline.
    """

    try:
        from ..._partial import _display_label

        return str(_display_label(record))
    except Exception:  # noqa: BLE001 - a display read may never raise
        return type(record).__name__


def _partial_nonfinite(partial: Any) -> str:
    """Nonfinite evidence line for a partial capture, degrade-safe."""

    try:
        return f"NaN/Inf: {partial.first_nonfinite()}"
    except Exception:  # noqa: BLE001 - evidence may be unavailable mid-failure
        return "NaN/Inf: unknown (capture did not finish)"


def trace_repr_html(trace: Any) -> str:
    """``Trace._repr_html_`` body: sniff-aware, never-raise."""

    sentinel = _bridge_sentinel(trace)
    if sentinel is not None:
        return sentinel
    return safe_card_html(lambda: trace_card(trace))


def layer_repr_html(layer: Any) -> str:
    """``Layer._repr_html_`` body: sniff-aware, never-raise."""

    sentinel = _bridge_sentinel(layer)
    if sentinel is not None:
        return sentinel
    return safe_card_html(lambda: layer_card(layer))


def op_repr_html(op: Any) -> str:
    """``Op._repr_html_`` body: sniff-aware, never-raise."""

    sentinel = _bridge_sentinel(op)
    if sentinel is not None:
        return sentinel
    return safe_card_html(lambda: op_card(op))


def partial_repr_html(partial: Any) -> str:
    """``PartialTrace._repr_html_`` body: sniff-aware, never-raise."""

    sentinel = _bridge_sentinel(partial)
    if sentinel is not None:
        return sentinel
    return safe_card_html(lambda: partial_trace_card(partial))
