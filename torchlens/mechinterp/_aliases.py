"""The alias layer: a RESOLVER, not a dict (mikit D16).

Trace-scoped resolution from TransformerLens spellings to native torchlens
addresses, independent of the process-wide compatibility flag. Resolution
returns the native facet, module address, site key, and pass index plus an
HONEST status (``real`` / ``reconstructed_read_only`` / ``needs_capture`` /
``structurally_absent`` / ``ambiguous``) with a remedy -- a TLens user's
first contact with fused reality is a teaching message, not a KeyError.

Accepted forms: full 2.x names (``blocks.3.attn.hook_z``), 3.x bridge-cache
names (the two generations have DIFFERENT vocabularies -- measured),
tuple forms (``("z", 3)``), and compact forms (``"k6"``). Layer indices
resolve through the EXECUTED attention/block order, so models with no
``blocks.N`` naming still resolve; no parser manufactures paths.

The data lives in ``translation.json`` keyed by (TLens name, TLens
generation); ``alias_report`` GENERATES the coverage matrix for the model in
front of the user -- it is never hand-written.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

from .._io import _json
from ._errors import refuse
from ._heads import _attention_modules

__all__ = ["AliasResolution", "alias_report", "resolve_alias", "translation_table"]

#: Resolver status vocabulary (mikit D16).
_STATUSES = ("real", "reconstructed_read_only", "needs_capture", "structurally_absent", "ambiguous")

_COMPACT_PATTERN = re.compile(
    r"^(q|k|v|z|pattern|scores|attn_out|mlp_out|resid_pre|resid_mid|resid_post)(\d+)$"
)


@dataclass(frozen=True)
class AliasResolution:
    """One resolved alias: native coordinates + honest capability status.

    Parameters
    ----------
    query:
        The TLens spelling as the user gave it.
    status:
        ``real`` (captured op, read -- and write where proven),
        ``reconstructed_read_only`` (validated fused reconstruction; edits
        need eager recapture or a lowered counterfactual),
        ``needs_capture`` (structurally present, payload not saved),
        ``structurally_absent`` (this architecture never produces it),
        ``ambiguous`` (more than one native site matches).
    facet:
        Native facet name, when the query maps to one.
    module_address:
        Native module address serving the facet.
    site_key:
        Live structural site key of the serving op, when derivable.
    pass_index:
        Pass qualifier of the serving op, when derivable.
    remedy:
        The action that unblocks the caller (always present off ``real``).
    generation:
        Which TLens generation vocabulary matched (``2.x`` / ``3.x-bridge``).
    """

    query: str
    status: str
    facet: str | None = None
    module_address: str | None = None
    site_key: str | None = None
    pass_index: int | None = None
    remedy: str | None = None
    generation: str | None = None


def translation_table() -> list[dict[str, Any]]:
    """Return the (TLens name, generation)-keyed translation rows."""

    with (
        resources.files("torchlens.mechinterp")
        .joinpath("translation.json")
        .open("r", encoding="utf-8")
    ) as handle:
        data = _json.load_bounded(handle)
    return list(data["rows"])


def _parse_query(name: Any) -> tuple[str, str | None, int | None, str | None]:
    """Normalize a query to (display, facet-or-None, layer-or-None, generation).

    Full names match translation rows (with ``{L}`` extraction); tuples are
    ``(facet, layer)``; compact ``"k6"`` is facet + layer. An unmatched full
    name returns (display, None, None, None) and resolves structurally
    absent with the table as the remedy.
    """

    if isinstance(name, tuple):
        facet, layer = name
        return f"({facet!r}, {layer})", str(facet), int(layer), None
    text = str(name)
    compact = _COMPACT_PATTERN.match(text)
    if compact:
        return text, compact.group(1), int(compact.group(2)), None
    for row in translation_table():
        pattern = str(row["tlens"])
        regex = "^" + re.escape(pattern).replace(r"\{L\}", r"(\d+)") + "$"
        match = re.match(regex, text)
        if match:
            layer = int(match.group(1)) if match.groups() else None
            return text, str(row["facet"]), layer, str(row["generation"])
    return text, None, None, None


def _facet_home_modules(trace: Any, facet: str) -> list[Any]:
    """Return modules exposing ``facet`` in executed order."""

    from ._heads import _facet_keys_or_none

    homes = []
    for module in trace.modules:
        keys = _facet_keys_or_none(module)
        if keys is not None and facet in keys:
            homes.append(module)
    return homes


def _status_for(module: Any, facet: str) -> tuple[str, str | None]:
    """Return (status, remedy) for one facet on one module."""

    menu = module.facets.menu()
    item = menu.get(facet)
    if item is None:
        return (
            "structurally_absent",
            "this architecture's recipes never declare the facet; see "
            "tl.facets.facet_coverage(trace)",
        )
    if item.status == "available_now":
        value = module.facets.get(facet)
        flags = getattr(getattr(value, "spec", None), "capability_flags", None)
        if flags is not None and getattr(flags, "reconstructed", False):
            return (
                "reconstructed_read_only",
                "the value is a validated fused-kernel reconstruction: reads are exact, "
                "writes need an eager recapture or a lowered_counterfactual",
            )
        return "real", None
    if item.status == "needs_capture":
        return (
            "needs_capture",
            item.save_hint or "recapture with the facet's home op saved",
        )
    return (
        "structurally_absent",
        item.detail or "the recipe declares this facet structurally absent here",
    )


def _site_coordinates(module: Any, facet: str) -> tuple[str | None, int | None]:
    """Best-effort (site_key, pass_index) for the facet's home op."""

    value = module.facets.get(facet)
    spec = getattr(value, "spec", None)
    home = getattr(spec, "home", None)
    return getattr(home, "site_key", None), getattr(home, "pass_index", None)


def resolve_alias(trace: Any, name: Any) -> AliasResolution:
    """Resolve one TLens spelling against a trace (mikit D16).

    Parameters
    ----------
    trace:
        A finished torchlens trace.
    name:
        Full TLens name, tuple form ``(facet, layer)``, or compact ``"k6"``.

    Returns
    -------
    AliasResolution
        Never raises for an unresolvable name -- the status IS the answer.
    """

    display, facet, layer, generation = _parse_query(name)
    if facet is None:
        return AliasResolution(
            query=display,
            status="structurally_absent",
            remedy="no translation row matches this spelling; see "
            "tl.mechinterp.translation_table() for the known (name, generation) rows",
        )
    homes = _facet_home_modules(trace, facet)
    if not homes and facet in ("resid_pre", "resid_mid", "resid_post", "mlp_out", "attn_out"):
        homes = _facet_home_modules(trace, facet)
    if layer is not None:
        ordered = homes if len(homes) > 1 else _attention_modules(trace)
        candidates = list(homes) or ordered
        if layer >= len(candidates):
            return AliasResolution(
                query=display,
                status="structurally_absent",
                facet=facet,
                remedy=f"layer {layer} outside the {len(candidates)} executed modules "
                f"exposing {facet!r}",
                generation=generation,
            )
        module = candidates[layer]
        status, remedy = _status_for(module, facet)
        site_key, pass_index = (
            _site_coordinates(module, facet)
            if status.startswith(("real", "recon"))
            else (None, None)
        )
        return AliasResolution(
            query=display,
            status=status,
            facet=facet,
            module_address=str(module.address),
            site_key=site_key,
            pass_index=pass_index,
            remedy=remedy,
            generation=generation,
        )
    if not homes:
        return AliasResolution(
            query=display,
            status="structurally_absent",
            facet=facet,
            remedy=f"no module in this trace exposes {facet!r}; see "
            "tl.facets.facet_coverage(trace)",
            generation=generation,
        )
    if len(homes) > 1:
        return AliasResolution(
            query=display,
            status="ambiguous",
            facet=facet,
            remedy=f"{len(homes)} modules expose {facet!r}; qualify with a layer index "
            f"(e.g. ('{facet}', 0)) or a module address",
            generation=generation,
        )
    status, remedy = _status_for(homes[0], facet)
    site_key, pass_index = _site_coordinates(homes[0], facet)
    return AliasResolution(
        query=display,
        status=status,
        facet=facet,
        module_address=str(homes[0].address),
        site_key=site_key,
        pass_index=pass_index,
        remedy=remedy,
        generation=generation,
    )


def alias_report(trace: Any, *, generation: str = "2.x") -> list[AliasResolution]:
    """GENERATE the per-trace TLens-name coverage matrix (never hand-written).

    Every translation row of the requested generation is instantiated for
    every executed layer and resolved against THIS trace; the result is the
    coverage matrix for the model in front of the user (published per R0
    family by CI).
    """

    rows = [row for row in translation_table() if row["generation"] == generation]
    if not rows:
        refuse(
            code="mi_translation_generation_unknown",
            message=f"No translation rows carry generation {generation!r}.",
            remedy='pass generation="2.x" or "3.x-bridge"',
            generation=generation,
        )
    n_layers = len(_attention_modules(trace))
    report = []
    for row in rows:
        pattern = str(row["tlens"])
        if "{L}" in pattern:
            for layer in range(n_layers):
                report.append(resolve_alias(trace, pattern.replace("{L}", str(layer))))
        else:
            report.append(resolve_alias(trace, pattern))
    return report
