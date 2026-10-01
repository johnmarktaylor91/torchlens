"""Rendered-edge multiplicity disclosure (r19): dedupe registry helpers.

Split out of ``_render_edges.py`` under the R43 file-size ratchet. When the
visual dedupe merges genuinely distinct dataflow edges, the emitted edge
gains an ``xN`` label instead of silently collapsing them.
"""

from typing import Any

import graphviz

from ._render_common import _EDGE_LABEL_FONT_SIZE
from ._render_utils import html_escape

__all__ = [
    "_bump_deduped_edge_multiplicity",
    "_html_argument_edge_label",
    "_register_deduped_edge",
    "_set_argument_edge_label",
]

# ``xN`` multiplicity labels are midpoint text on their own spline; unpadded
# plaintext there collides with the spline and neighbours. Padding 2 through
# the same one-cell-table idiom as the head/tail builders measured the gpt2
# depth-1 audit entry 11 -> 0 violations at +0.0% width (vizmech D9).
_MULTIPLICITY_LABEL_PAD = 2
_MULTIPLICITY_LABEL_FONT_SIZE = 8


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
    calls = getattr(graphviz_graph, "calls", None)
    if calls is not None:
        from .render_ir import RenderIRDotStatement

        calls[body_index] = RenderIRDotStatement("edge", (), tuple(entry["edge_dict"].items()))
    else:
        rewrite = graphviz.Digraph()
        rewrite.edge(**entry["edge_dict"])
        graphviz_graph.body[body_index] = rewrite.body[-1]


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


def _set_argument_edge_label(edge_dict: dict[str, Any], arg_label: str) -> None:
    """Attach an argument-position label without overwriting semantic edge labels.

    Args:
        edge_dict:
            Mutable Graphviz edge attribute dict.
        arg_label:
            HTML label string describing edge argument positions.
    """
    if "headlabel" not in edge_dict:
        edge_dict["headlabel"] = arg_label
        return
    if "xlabel" not in edge_dict:
        edge_dict["xlabel"] = arg_label
        return
    if edge_dict["xlabel"] == arg_label:
        return
    # Two builder-shaped labels merge into ONE table (two sibling tables under
    # one HTML root are not a valid graphviz label); anything else keeps the
    # historical tag surgery for foreign label shapes.
    existing_rows = _argument_edge_label_rows_text(edge_dict["xlabel"])
    new_rows = _argument_edge_label_rows_text(arg_label)
    if existing_rows is not None and new_rows is not None:
        edge_dict["xlabel"] = (
            f"{_ARG_EDGE_LABEL_PREFIX}{existing_rows}<BR/>{new_rows}{_ARG_EDGE_LABEL_SUFFIX}"
        )
        return
    edge_dict["xlabel"] = edge_dict["xlabel"][:-1] + "<br/>" + arg_label[1:]
