"""Rendered-edge multiplicity disclosure (r19): dedupe registry helpers.

Split out of ``_render_edges.py`` under the R43 file-size ratchet. When the
visual dedupe merges genuinely distinct dataflow edges, the emitted edge
gains an ``xN`` label instead of silently collapsing them.
"""

from typing import Any

import graphviz

from ._render_common import _EDGE_LABEL_FONT_SIZE
from ._render_utils import html_escape
from ._typography import DEFAULT_TYPOGRAPHY

__all__ = [
    "_ARG_LABEL_MIDPOINT_FANIN",
    "_bump_deduped_edge_multiplicity",
    "_html_argument_edge_label",
    "_merge_parallel_arg_midpoint",
    "_register_arg_midpoint_edge",
    "_register_deduped_edge",
    "_set_argument_edge_label",
]

# ``xN`` multiplicity labels are midpoint text on their own spline; unpadded
# plaintext there collides with the spline and neighbours. Padding 2 through
# the same one-cell-table idiom as the head/tail builders measured the gpt2
# depth-1 audit entry 11 -> 0 violations at +0.0% width (vizmech D9).
_MULTIPLICITY_LABEL_PAD = 2
# Sourced from the one typography record (vizmech D29).
_MULTIPLICITY_LABEL_FONT_SIZE = int(DEFAULT_TYPOGRAPHY.annotation_size)


def _html_multiplicity_edge_label(text: str) -> str:
    """Return the padded HTML midpoint label for an ``xN`` disclosure."""

    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" '
        f'CELLPADDING="{_MULTIPLICITY_LABEL_PAD}">'
        f'<TR><TD><FONT POINT-SIZE="{_MULTIPLICITY_LABEL_FONT_SIZE}">'
        f"{html_escape(text)}</FONT></TD></TR></TABLE>>"
    )


def _register_deduped_edge(
    registry: dict[tuple[Any, ...], dict[str, Any]] | None,
    visual_dedupe_key: tuple[Any, ...],
    raw_edge_identity: tuple[str, str],
    edge_dict: dict[str, Any],
    body_index: int | None,
) -> None:
    """Record one emitted rendered edge for later multiplicity disclosure.

    Parameters
    ----------
    registry:
        Cross-call registry owned by the render entrypoint, or ``None``.
    visual_dedupe_key:
        The rendered-edge visual identity the edge was emitted under.
    raw_edge_identity:
        ``(source_layer_label, target_layer_label)`` of the underlying
        dataflow edge.
    edge_dict:
        The emitted edge attributes (queued dicts stay mutable in
        ``module_edge_dict``).
    body_index:
        Index of the emitted statement in ``graphviz_graph.body`` for
        directly-emitted edges, or ``None`` for cluster-queued edges.
    """

    if registry is None:
        return
    registry[visual_dedupe_key] = {
        "identities": {raw_edge_identity},
        "edge_dict": edge_dict,
        "body_index": body_index,
        "base_label": edge_dict.get("label"),
    }


def _bump_deduped_edge_multiplicity(
    registry: dict[tuple[Any, ...], dict[str, Any]] | None,
    visual_dedupe_key: tuple[Any, ...],
    raw_edge_identity: tuple[str, str],
    graphviz_graph: graphviz.Digraph,
) -> None:
    """Disclose multiplicity when distinct dataflow edges merge in a render.

    r19 (b6-fable carried LOW, 3rd round): a collapsed module returning TWO
    tensors both consumed by one exterior op rendered as ONE unlabeled edge
    — the visual dedupe silently merged genuinely distinct dataflow edges
    with no multiplicity disclosure. When a deduped edge's underlying
    ``(source, target)`` identity is NEW (not a repeat occurrence of the
    same logical edge), the emitted edge gains an ``xN`` label. Queued
    cluster edges are updated in place; directly-emitted edges are rewritten
    in ``graphviz_graph.body`` at their recorded index, so statement order —
    and therefore DOT byte determinism for undisclosed renders — is
    unchanged.
    """

    if registry is None:
        return
    entry = registry.get(visual_dedupe_key)
    if entry is None or raw_edge_identity in entry["identities"]:
        return
    entry["identities"].add(raw_edge_identity)
    count = len(entry["identities"])
    base_label = entry["base_label"]
    if base_label and str(base_label).startswith("<"):
        # A pre-existing HTML label keeps the historical concatenation.
        entry["edge_dict"]["label"] = f"{base_label} x{count}"
    else:
        text = f"{base_label} x{count}" if base_label else f"x{count}"
        entry["edge_dict"]["label"] = _html_multiplicity_edge_label(text)
    body_index = entry["body_index"]
    if body_index is None:
        return
    _rewrite_emitted_edge(graphviz_graph, body_index, entry["edge_dict"])


# The argument-label channel routes through the same tuned one-cell-table
# builder as every other head/tail label (vizmech D9): head/tail labels
# reserve NO layout space in graphviz, so an unpadded bold 10-pt label sits
# inside the node it annotates and gets speared by arrowheads on essentially
# every multi-input op. Padding 6 / 8 pt / non-bold measured the corpus label
# audit 433 -> 139 violations with zero regressions and +0.0% width.
_ARG_EDGE_LABEL_PAD = 6
_ARG_EDGE_LABEL_PREFIX = (
    f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="{_ARG_EDGE_LABEL_PAD}">'
    f'<TR><TD><FONT POINT-SIZE="{_EDGE_LABEL_FONT_SIZE}">'
)
_ARG_EDGE_LABEL_SUFFIX = "</FONT></TD></TR></TABLE>>"

#: Visible fan-in at/above which a child op's argument labels relocate from
#: ``headlabel`` to a midpoint ``label`` (FIXD03-F13, D03-R4). Graphviz
#: reserves layout space ONLY for midpoint labels; every head label of a
#: node is painted post-layout at one default radius, so they smear into
#: one band as fan-in grows. Sweep provenance (toy N-way cat ladder, dot
#: 2.43, hard geometry violations): fan-in 3/4/5/6/8/12/24 measured
#: 0/1/0/3/11/15/75 with headlabels and 0 at EVERY degree after midpoint
#: relocation (+5-10% canvas area; densenet121 collapse=auto: 302
#: arg-label violations -> 0). Per-edge ``labeldistance``/``labelangle``
#: staggering was swept FIRST and is dominated by the unset baseline at
#: every grid point (place_portlabel trap, playbook s1): 15 -> 20..38 and
#: 75 -> 86..214. PORTABILITY: the threshold conditions on an emit-time
#: structural fact (visible parent count), never on font metrics.
_ARG_LABEL_MIDPOINT_FANIN = 4

#: Registry-key marker for the same-pair parallel arg-edge merge; namespaced
#: so entries can share the render entrypoint's ``deduped_edge_registry``
#: without colliding with visual-dedupe keys (those are
#: ``(tail, head, signature)`` tuples of edge attrs, never this literal).
_ARG_MERGE_KEY_MARKER = "__tl_arg_midpoint_merge__"


def _arg_midpoint_registry_key(tail_name: str, head_name: str) -> tuple[str, str, str]:
    """Registry key for one rendered (tail, head) pair's merged arg edge."""

    return (_ARG_MERGE_KEY_MARKER, tail_name, head_name)


def _html_argument_edge_label(rows: list[str]) -> str:
    """Return the padded HTML label for argument-position rows.

    Parameters
    ----------
    rows:
        Plain-text argument rows (``"arg 0"``, ``"kwarg mask"``); escaped here.

    Returns
    -------
    str
        Graphviz HTML label string with an even transparent margin.
    """

    text = "<BR/>".join(html_escape(row) for row in rows)
    return f"{_ARG_EDGE_LABEL_PREFIX}{text}{_ARG_EDGE_LABEL_SUFFIX}"


def _argument_edge_label_rows_text(arg_label: str) -> str | None:
    """Return the inner row text of a builder-shaped argument label, else ``None``."""

    if arg_label.startswith(_ARG_EDGE_LABEL_PREFIX) and arg_label.endswith(_ARG_EDGE_LABEL_SUFFIX):
        return arg_label[len(_ARG_EDGE_LABEL_PREFIX) : -len(_ARG_EDGE_LABEL_SUFFIX)]
    return None


def _set_argument_edge_label(
    edge_dict: dict[str, Any], arg_label: str, *, prefer_midpoint: bool = False
) -> str:
    """Attach an argument-position label without overwriting semantic edge labels.

    Args:
        edge_dict:
            Mutable Graphviz edge attribute dict.
        arg_label:
            HTML label string describing edge argument positions.
        prefer_midpoint:
            High-fan-in mode (D03-R4): route the label to a midpoint
            ``label`` -- which graphviz reserves real layout space for --
            when no semantic midpoint label occupies the slot; a taken slot
            falls back to the historical head/xlabel chain.

    Returns:
        The channel the label landed on: ``"label"``, ``"headlabel"``, or
        ``"xlabel"``.
    """
    if prefer_midpoint and "label" not in edge_dict:
        edge_dict["label"] = arg_label
        return "label"
    if "headlabel" not in edge_dict:
        edge_dict["headlabel"] = arg_label
        return "headlabel"
    if "xlabel" not in edge_dict:
        edge_dict["xlabel"] = arg_label
        return "xlabel"
    if edge_dict["xlabel"] == arg_label:
        return "xlabel"
    # Two builder-shaped labels merge into ONE table (two sibling tables under
    # one HTML root are not a valid graphviz label); anything else keeps the
    # historical tag surgery for foreign label shapes.
    existing_rows = _argument_edge_label_rows_text(edge_dict["xlabel"])
    new_rows = _argument_edge_label_rows_text(arg_label)
    if existing_rows is not None and new_rows is not None:
        edge_dict["xlabel"] = (
            f"{_ARG_EDGE_LABEL_PREFIX}{existing_rows}<BR/>{new_rows}{_ARG_EDGE_LABEL_SUFFIX}"
        )
        return "xlabel"
    edge_dict["xlabel"] = edge_dict["xlabel"][:-1] + "<br/>" + arg_label[1:]
    return "xlabel"


def _rewrite_emitted_edge(
    graphviz_graph: graphviz.Digraph, body_index: int, edge_dict: dict[str, Any]
) -> None:
    """Rewrite one already-emitted edge statement in place (label updates).

    Mirrors the multiplicity-bump rewrite: statement order -- and therefore
    DOT byte determinism for untouched renders -- is unchanged.
    """

    calls = getattr(graphviz_graph, "calls", None)
    if calls is not None:
        from .render_ir import RenderIRDotStatement

        calls[body_index] = RenderIRDotStatement("edge", (), tuple(edge_dict.items()))
    else:
        rewrite = graphviz.Digraph()
        rewrite.edge(**edge_dict)
        graphviz_graph.body[body_index] = rewrite.body[-1]


def _register_arg_midpoint_edge(
    registry: dict[tuple[Any, ...], dict[str, Any]] | None,
    tail_name: str,
    head_name: str,
    edge_dict: dict[str, Any],
    body_index: int | None,
) -> None:
    """Record one emitted midpoint-arg-labeled edge for same-pair merging.

    Parameters mirror :func:`_register_deduped_edge`; ``body_index`` is
    ``None`` for cluster-queued edges (their dicts stay mutable until
    serialization).
    """

    if registry is None:
        return
    registry[_arg_midpoint_registry_key(tail_name, head_name)] = {
        "edge_dict": edge_dict,
        "body_index": body_index,
    }


def _merge_parallel_arg_midpoint(
    registry: dict[tuple[Any, ...], dict[str, Any]] | None,
    tail_name: str,
    head_name: str,
    arg_label: str,
    graphviz_graph: graphviz.Digraph,
) -> bool:
    """Fold a same-pair parallel arg edge into the pair's merged edge.

    D03-R4, high-fan-in mode: when a collapsed source feeds one child at N
    argument slots, N visually identical parallel edges each carrying one
    ``arg (0, k)`` head label rendered as an unreadable smear -- and the
    per-edge indices carried no distinguishing information, because every
    edge connected the SAME two rendered boxes. Under midpoint relocation
    the pair renders as ONE edge whose reserved-space midpoint label lists
    every argument row (arrival order, deduplicated), so the slot inventory
    stays exactly recoverable; this follows the run-fold ellipsis
    precedent (one summary edge over N parallel occurrence edges).

    Returns ``True`` when the occurrence was merged (the caller skips
    emitting it), ``False`` when no merged edge exists yet or either label
    is not builder-shaped.
    """

    if registry is None:
        return False
    entry = registry.get(_arg_midpoint_registry_key(tail_name, head_name))
    if entry is None:
        return False
    existing_label = entry["edge_dict"].get("label", "")
    existing_rows = _argument_edge_label_rows_text(existing_label)
    new_rows = _argument_edge_label_rows_text(arg_label)
    if existing_rows is None or new_rows is None:
        return False
    seen_rows = existing_rows.split("<BR/>")
    add_rows = [row for row in new_rows.split("<BR/>") if row not in seen_rows]
    if add_rows:
        merged = "<BR/>".join(seen_rows + add_rows)
        entry["edge_dict"]["label"] = f"{_ARG_EDGE_LABEL_PREFIX}{merged}{_ARG_EDGE_LABEL_SUFFIX}"
        if entry["body_index"] is not None:
            _rewrite_emitted_edge(graphviz_graph, entry["body_index"], entry["edge_dict"])
    return True
