"""Node-id minting and component escaping (memo D1 escaping, D2 id rule).

ONE id rule for every graph product: ``site_key`` + ``"|"`` + the 1-based
graph-local occurrence ordinal, appended ALWAYS (including ordinal 1). Every
alternative spelling was measured to lose nodes silently -- Model Explorer's
worker keeps the first duplicate id and silently drops the second with its
edges, namespace, and overlay binding. The separator ``|`` lies inside
site_key's escape set (escaped inside every component), so split-on-last-
``|`` is exact; ``#`` is forbidden (legal in ModuleDict keys). Ids contain
raw ``/`` and ``|`` and are URL-encoded at the URL layer, never here.
"""

from __future__ import annotations

from typing import Any

__tl_layer__ = "L8"

#: Reserved prefix for nodes without a mintable site key (memo D2): visibly
#: prefixed, stamped ``id_fidelity=legacy`` in graph attrs, and the ONLY
#: nodes (with guarded non-identical-structure joins) that ever appear in
#: sync ``mappingEntries``.
LEGACY_ID_PREFIX = "legacy|"

#: Reserved prefix for step-boundary proxy nodes (memo D12): can never
#: collide with a real id in their own or the adjacent pane.
PROXY_ID_PREFIX = "proxy|"

_ESCAPE_CHARS = {"%", "/", "|"}


def escape_component(component: str) -> str:
    """Percent-encode ``%``, ``/``, ``|``, and control bytes in one component.

    Reversible per memo D1 (upstream slash-escaping PR #537 died unmerged,
    so the payload self-defends). Applied to namespace path components and
    legacy id payloads; captured site keys arrive already escaped.
    """

    out: list[str] = []
    for char in component:
        if char in _ESCAPE_CHARS or ord(char) < 0x20 or ord(char) == 0x7F:
            out.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
        else:
            out.append(char)
    return "".join(out)


def mint_node_ids(entries: list[Any]) -> tuple[list[str], int]:
    """Mint the universal node id for each entry, in graph order.

    Parameters
    ----------
    entries:
        Layer-pass entries in the order the exporter consumes
        (``_iter_layers``); the ordinal is minted over exactly this order,
        pinned by test, never implicit.

    Returns
    -------
    tuple[list[str], int]
        Ids aligned with ``entries``, and the count of legacy-id nodes
        (``id_fidelity=legacy`` when nonzero).
    """

    occurrence_counts: dict[str, int] = {}
    ids: list[str] = []
    legacy_count = 0
    for entry in entries:
        base = _id_base(entry)
        if base.startswith(LEGACY_ID_PREFIX):
            legacy_count += 1
        ordinal = occurrence_counts.get(base, 0) + 1
        occurrence_counts[base] = ordinal
        ids.append(f"{base}|{ordinal}")
    return ids, legacy_count


def legacy_v2_spelling(entry: Any) -> str:
    """Return the v2 exporter's node spelling for one entry.

    Preserved as the payload of a legacy id (and as the ``legacy_id`` attr)
    so pre-site-key artifacts keep a stable, recognizable identity.
    """

    layer_label = str(getattr(entry, "layer_label", "") or "")
    if int(getattr(entry, "num_passes", 1) or 1) > 1:
        return str(getattr(entry, "label", layer_label) or layer_label)
    return layer_label


def _id_base(entry: Any) -> str:
    """Return the pre-ordinal id base: the site key, or the legacy spelling."""

    site_key = getattr(entry, "site_key", None)
    if isinstance(site_key, str) and site_key:
        return site_key
    return LEGACY_ID_PREFIX + escape_component(legacy_v2_spelling(entry))
