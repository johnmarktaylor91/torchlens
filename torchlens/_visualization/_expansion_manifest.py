"""Expansion manifest for over-ceiling overviews.

FK4 ruled the over-ceiling residual -- a graph still too large for the
readable band after the coarse fallback grouping -- toward a bounded overview
plus an expansion manifest rather than a typed refusal. The overview always
renders; the manifest, written beside the picture as
``<outpath>.expansion.json``, lists every collapsed unit of that overview with
a stable expansion id (the pass-qualified module call, or the member list for
a fold or segment) and the draw spelling that opens it (``module=`` focus).
The manifest is the future interactive door; today it is a file a user or a
tool reads to choose the next, smaller render.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from .collapse_plan import (
    ChildSegment,
    CollapsePlan,
    EllipsisNode,
    ModuleBox,
    OpSegment,
    RepeatFold,
    count,
)

__all__ = [
    "EXPANSION_MANIFEST_SCHEMA",
    "ExpansionEntry",
    "build_expansion_manifest",
    "write_expansion_manifest",
]

#: Schema id stamped on every written manifest.
EXPANSION_MANIFEST_SCHEMA = "torchlens_expansion_manifest_v1"


@dataclass(frozen=True)
class ExpansionEntry:
    """One collapsed unit of an overview and how to open it.

    Attributes
    ----------
    expansion_id:
        Stable id: the pass-qualified module call for a box, ``fold:<first>``
        for a repeat fold, ``segment:<first>`` for a condensed segment.
    kind:
        ``"module"``, ``"repeat_fold"``, ``"child_segment"`` or ``"op_segment"``.
    module:
        Module address the ``module=`` focus takes, or ``None`` for an
        operation segment (which opens with ``collapse="none"`` on its owner).
    members:
        Folded or segmented member addresses (empty for a single box).
    num_ops:
        Hidden operation count when the segment records it, else ``None``.
    expand:
        Keyword arguments for the ``trace.draw`` call that opens the unit.
    """

    expansion_id: str
    kind: str
    module: str | None
    members: tuple[str, ...]
    num_ops: int | None
    expand: dict[str, Any]


def _module_address(call: str) -> str:
    """Return the module address of a pass-qualified call (``a.b:2`` -> ``a.b``)."""

    return call.rsplit(":", 1)[0] if ":" in call else call


def _entry_for(node: Any) -> ExpansionEntry | None:
    """Return the manifest entry for one plan node, or ``None`` for exposed nodes."""

    if isinstance(node, ModuleBox):
        address = _module_address(node.call)
        return ExpansionEntry(node.call, "module", address, (), None, {"module": address})
    if isinstance(node, RepeatFold):
        first = node.members[0] if node.members else _module_address(node.rep.call)
        return ExpansionEntry(
            f"fold:{first}",
            "repeat_fold",
            first,
            tuple(node.members),
            None,
            {"module": first, "fold_repeats": False},
        )
    if isinstance(node, ChildSegment):
        first = node.members[0] if node.members else ""
        return ExpansionEntry(
            f"segment:{first}",
            "child_segment",
            first or None,
            tuple(node.members),
            None,
            {"module": first} if first else {"collapse": "none"},
        )
    if isinstance(node, OpSegment):
        first = node.ops[0] if node.ops else ""
        return ExpansionEntry(
            f"segment:{first}",
            "op_segment",
            None,
            tuple(node.ops),
            len(node.ops),
            {"collapse": "none"},
        )
    if isinstance(node, EllipsisNode):
        return None
    return None


def build_expansion_manifest(
    plan: CollapsePlan,
    *,
    planner: str,
    reason: str | None,
    band_high: int,
) -> dict[str, Any]:
    """Return the manifest document for an overview plan.

    Parameters
    ----------
    plan:
        The plan the overview rendered.
    planner:
        Which tier produced it (``"floor_fallback"``, ``"overview_fallback"``).
    reason:
        Why the overview is over the band (the planner's typed reason).
    band_high:
        Upper edge of the readable band the overview could not reach.

    Returns
    -------
    dict[str, Any]
        JSON-ready manifest with one entry per collapsed unit.
    """

    entries = [entry for entry in (_entry_for(node) for node in plan.nodes) if entry is not None]
    return {
        "schema": EXPANSION_MANIFEST_SCHEMA,
        "planner": planner,
        "reason": reason,
        "visible_count": count(plan),
        "readable_band_high": band_high,
        "entries": [asdict(entry) for entry in entries],
    }


def write_expansion_manifest(outpath: str, manifest: dict[str, Any]) -> str:
    """Write ``manifest`` beside the rendered file; return the manifest path.

    Parameters
    ----------
    outpath:
        Render output path without its extension.
    manifest:
        Document from :func:`build_expansion_manifest`.

    Returns
    -------
    str
        ``<outpath>.expansion.json``.
    """

    path = f"{outpath}.expansion.json"
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)
    return path
