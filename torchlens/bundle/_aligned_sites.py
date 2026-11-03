"""Cross-member site pairing for Bundle site reads.

Bundle site reads (``rank_changes``, ``compare``) pair each reference op with
its counterpart in every member through the supergraph's structural alignment,
never by literal label: a capture-time module edit inserts an
``interventionreplacement`` op and renumbers every later op, so a label join
silently dropped exactly the ops the edit changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from .._selection.resolver import _resolve_sites
from .._selection.selectors import label as _exact_label
from ..intervention._super.super_op import SuperOp
from ..intervention.errors import BundleMemberError

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from ..intervention._topology.topology import Supergraph
    from . import Bundle


def aligned_site_labels(supergraph: Supergraph, reference_name: str) -> dict[str, dict[str, str]]:
    """Map each reference layer label to its structurally aligned labels.

    Parameters
    ----------
    supergraph:
        The bundle's supergraph.
    reference_name:
        Member whose layer labels key the map.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{reference_label: {member_name: member_label}}`` for every
        supergraph node the reference member traverses.
    """

    aligned: dict[str, dict[str, str]] = {}
    for node in supergraph.nodes.values():
        reference_layer = node.layer_refs.get(reference_name)
        if reference_layer is None:
            continue
        aligned[str(reference_layer.layer_label)] = {
            name: str(layer.layer_label) for name, layer in node.layer_refs.items()
        }
    return aligned


def site_row_key(site: Any) -> str:
    """Return the row key of one site in ``compare`` and ``rank_changes``.

    A multi-pass op has one row per pass, keyed by the pass-qualified label
    (``tanh_1_1:2``); a single-pass op keeps its bare layer label.

    Parameters
    ----------
    site:
        An op of the reference member.

    Returns
    -------
    str
        The row key.
    """

    layer_label = str(site.layer_label)
    if int(getattr(site, "num_passes", 1) or 1) > 1:
        return f"{layer_label}:{site.pass_number}"
    return layer_label


def _split_pass_suffix(label: str) -> tuple[str, str]:
    """Split ``tanh_1_1:2`` into ``("tanh_1_1", ":2")``; a bare label keeps an empty suffix."""

    head, sep, tail = label.rpartition(":")
    if sep and tail.isdigit():
        return head, f":{tail}"
    return label, ""


def resolve_aligned_site(
    members: Mapping[str, Trace],
    label: str,
    aligned: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    """Resolve one reference site in every member at its aligned label.

    Parameters
    ----------
    members:
        Member name to trace.
    label:
        Reference member's row key (:func:`site_row_key`): a layer label, or
        the pass-qualified label of one pass of a multi-pass op. The
        alignment pairs layer labels; the pass suffix carries over unchanged.
    aligned:
        Map from :func:`aligned_site_labels`. A member the alignment does not
        pair is looked up by the literal label.

    Returns
    -------
    dict[str, Any] | None
        Member name to resolved op, or ``None`` when any member lacks the site
        (it is then not a common site).
    """

    base, suffix = _split_pass_suffix(label)
    counterparts = aligned.get(base, {})
    resolved: dict[str, Any] = {}
    for name, log in members.items():
        # Exact label: a bare string is a substring selector post hoc, so
        # ``t_1_1`` would also match ``cat_1_10`` and drop the site as ambiguous.
        try:
            target = _exact_label(counterparts.get(name, base) + suffix)
            table = _resolve_sites(log, target, max_fanout=1)
            resolved[name] = table.first()
        except Exception:  # noqa: BLE001 - missing sites are not common sites
            return None
    return resolved


def site_view(self: Bundle, site: Any) -> SuperOp:
    """Resolve a selector-like site to one op per member (the model-floor gated read)."""

    names = list(self._members)
    pairs = [(left, right) for index, left in enumerate(names) for right in names[index + 1 :]]
    self._require_comparable("at", pairs=pairs)
    self._ensure_supergraph()
    layer_members: dict[str, Op] = {}
    failures: dict[str, str] = {}
    for name in names:
        try:
            table = _resolve_sites(self._members[name], site, max_fanout=1)
            layer_members[name] = cast("Op", table.first())
        except Exception as exc:  # noqa: BLE001 - rewrapped with member context
            failures[name] = str(exc)
    if failures:
        detail = "; ".join(f"{name}: {message}" for name, message in failures.items())
        raise BundleMemberError(f"site {site!r} failed to resolve for bundle members: {detail}")
    return SuperOp.from_members(site, layer_members)
