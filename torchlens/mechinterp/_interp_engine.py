"""This trace's sites as interp-engine points: the points file and its writer.

interp-engine (a capture engine with eager and vLLM backends) names a
site by a canonical address: a point name, then the layer, then any extra
coordinate (``resid_post.5``, ``embeddings``, ``resid_streams.5.stream-2``).
Its ``capture(token_ids, points)`` accepts those strings as they are, so the
points file is plain JSON: every point resolved against THIS trace through the
``interp-engine`` generation of ``alias_report``, the list to hand interp-engine
for the sites torchlens has (``capture_points``), the engine's points with no
torchlens site, and this trace's tensor sites with no engine point -- each
absence with its reason, never a guess.

The engine's registry and address grammar are pinned in
``interp_engine_points.json`` (version and commit inside); a record's address
fields are exactly the engine's ``Address`` fields.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ._aliases import AliasResolution, _package_json, alias_report, translation_table
from ._heads import _facet_keys_or_none

_GENERATION = "interp-engine"
_SCHEMA = "torchlens_interp_engine_points_v1"
_REGISTRY_FILE = "interp_engine_points.json"
#: Statuses whose site interp-engine can be asked for (torchlens has the site).
_HAS_SITE = ("real", "reconstructed_read_only", "needs_capture")


def _address_fields(text: str, grammar: dict[str, Any]) -> dict[str, Any]:
    """Split a canonical address into the engine's ``Address`` fields.

    The points this generation emits carry a name and at most a layer; the
    grammar is the pinned one, so a malformed row fails the round-trip test.
    """

    head, _, rest = text.partition(grammar["separator"])
    fields: dict[str, Any] = dict.fromkeys(grammar["fields"])
    fields["name"] = head
    if rest and re.match(grammar["layer_pattern"], rest):
        fields["layer"] = int(rest)
    return fields


def _site(entry: AliasResolution) -> dict[str, Any]:
    """The torchlens side of one resolved point."""

    return {
        "status": entry.status,
        "facet": entry.facet,
        "module_address": entry.module_address,
        "site_key": entry.site_key,
        "pass_index": entry.pass_index,
        "remedy": entry.remedy,
    }


def _site_facets(registry: dict[str, Any]) -> set[str]:
    """The tensor facets the export accounts for: mapped by a row, or reasoned absent."""

    mapped = {str(row["facet"]) for row in translation_table() if row["generation"] == _GENERATION}
    return mapped | set(registry["facets_without_point"])


def _sites_without_point(
    trace: Any, registry: dict[str, Any], served: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    """This trace's tensor sites no interp-engine point resolves to, with the reason."""

    reasons = registry["facets_without_point"]
    accounted = _site_facets(registry)
    rows = []
    for module in trace.modules:
        keys = _facet_keys_or_none(module) or set()
        address = str(module.address)
        for facet in sorted(keys & accounted):
            if (address, facet) in served:
                continue
            reason = reasons.get(facet) or (
                f"no interp-engine point resolves to this module's {facet!r}; the points "
                "naming that facet resolve to other modules (listed under points)"
            )
            rows.append({"module_address": address, "facet": facet, "reason": reason})
    return rows


def _interp_engine_points(trace: Any) -> dict[str, Any]:
    """Build the points file for ``trace`` (see the module docstring)."""

    registry = _package_json(_REGISTRY_FILE)
    grammar = registry["address"]
    points, absent, served = [], [], set()
    for entry in alias_report(trace, generation=_GENERATION):
        if entry.facet is None:
            absent.append({"point": entry.query, "reason": entry.remedy})
            continue
        record = _address_fields(entry.query, grammar)
        record["address"] = entry.query
        record["torchlens"] = _site(entry)
        points.append(record)
        if entry.status in _HAS_SITE and entry.module_address is not None:
            served.add((entry.module_address, entry.facet))
    source = registry["source"]
    return {
        "schema": _SCHEMA,
        "interp_engine": {
            "repository": source["repository"],
            "version": source["version"],
            "commit": source["commit"],
            "address_fields": list(grammar["fields"]),
        },
        "capture_points": [p["address"] for p in points if p["torchlens"]["status"] in _HAS_SITE],
        "points": points,
        "no_torchlens_site": absent,
        "no_interp_engine_point": _sites_without_point(trace, registry, served),
    }


def _write_interp_engine_points(trace: Any, path: str | Path) -> Path:
    """Write ``trace``'s interp-engine points file as JSON; return the path.

    interp-engine reads it with ``json.load``: ``capture_points`` goes to
    ``model.capture(token_ids, points)`` as is, and each record's
    ``name``/``layer``/``stream`` are ``interp_engine.Address`` fields.
    """

    target = Path(path)
    payload = json.dumps(_interp_engine_points(trace), indent=2, sort_keys=False)
    target.write_text(payload + "\n", encoding="utf-8")
    return target
