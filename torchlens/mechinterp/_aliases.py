"""The alias layer: a RESOLVER, not a dict (mikit D16).

Trace-scoped resolution from TransformerLens spellings to native torchlens
addresses, independent of the process-wide compatibility flag. Resolution
returns the native facet, module address, site key, and pass index plus an
HONEST status (``real`` / ``reconstructed_read_only`` / ``needs_capture`` /
``structurally_absent`` / ``ambiguous``) with a remedy -- a TLens user's
first contact with fused reality is a teaching message, not a KeyError.

Accepted forms: full 2.x names (``blocks.3.attn.hook_z``), 3.x bridge-cache
names (the two generations have DIFFERENT vocabularies -- measured), the
4.x ``TransformerBridge`` canonical names (``blocks.3.attn.o.hook_in``,
``blocks.3.ln2.hook_in``), tuple forms (``("z", 3)``), and compact forms
(``"k6"``). Rows may carry a ``scope`` (attention / mlp modules only), an
embedding ``role`` (token / position, decided by table size against the
unembedding vocabulary), and an ``only_if_facet`` / ``unless_facet``
condition where one TLens name means different tensors by architecture. Layer indices
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
        Which TLens generation vocabulary matched (``2.x`` / ``3.x-bridge`` /
        ``4.x``).
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


#: Facets that identify a module's scope when a translation row names one.
_SCOPE_KEYS: dict[str, frozenset[str]] = {
    "attention": frozenset({"n_q_heads", "d_head"}),
    "mlp": frozenset({"down_out"}),
}


@dataclass(frozen=True)
class _Query:
    """A parsed alias query: the matched translation row (if any) and its layer."""

    display: str
    facet: str | None
    layer: int | None
    generation: str | None
    row: dict[str, Any] | None = None


def _match_row(text: str, row: dict[str, Any]) -> tuple[bool, int | None]:
    """Return (matched, layer) for one translation row's ``{L}`` pattern."""

    pattern = str(row["tlens"])
    regex = "^" + re.escape(pattern).replace(r"\{L\}", r"(\d+)") + "$"
    match = re.match(regex, text)
    if not match:
        return False, None
    return True, (int(match.group(1)) if match.groups() else None)


def _parse_query(name: Any, prefer: str | None = None) -> _Query:
    """Normalize a query to its translation row, facet, layer and generation.

    Full names match translation rows (with ``{L}`` extraction; a name that
    appears in several generations resolves through the ``prefer``
    generation's row when given, else the first); tuples are ``(facet,
    layer)``; compact ``"k6"`` is facet + layer. An unmatched full name
    carries no facet and resolves structurally absent with the table as
    the remedy.
    """

    if isinstance(name, tuple):
        facet, layer = name
        return _Query(f"({facet!r}, {layer})", str(facet), int(layer), None)
    text = str(name)
    compact = _COMPACT_PATTERN.match(text)
    if compact:
        return _Query(text, compact.group(1), int(compact.group(2)), None)
    matches = []
    for row in translation_table():
        matched, layer = _match_row(text, row)
        if matched:
            matches.append((row, layer))
    if not matches:
        return _Query(text, None, None, None)
    chosen = next((m for m in matches if m[0]["generation"] == prefer), matches[0])
    row, layer = chosen
    return _Query(text, str(row["facet"]), layer, str(row["generation"]), row)


def _facet_home_modules(trace: Any, facet: str, scope: str | None = None) -> list[Any]:
    """Return modules exposing ``facet`` in executed order, filtered by row scope."""

    from ._heads import _facet_keys_or_none

    required = {facet} | set(_SCOPE_KEYS.get(scope or "", frozenset()))
    homes = []
    for module in trace.modules:
        keys = _facet_keys_or_none(module)
        if keys is not None and required <= keys:
            homes.append(module)
    return homes


def _param_rows(facet: Any) -> int | None:
    """Return a 2D parameter facet's row count from its recorded shape.

    Reads the parameter record's shape METADATA (``spec.home.shape``), never
    the live value, so resolution still answers after the source model was
    garbage-collected.
    """

    home = getattr(getattr(facet, "spec", None), "home", None)
    shape = tuple(getattr(home, "shape", ()) or ())
    return int(shape[0]) if len(shape) == 2 else None


def _vocab_size(trace: Any) -> int | None:
    """Return the unembedding's vocabulary size, when a head is recorded."""

    for module in _facet_home_modules(trace, "unembed_weight"):
        rows = _param_rows(module.facets.get("unembed_weight"))
        if rows is not None:
            return rows
    return None


def _embedding_rows(module: Any) -> int | None:
    """Return an embedding module's table size (``num_embeddings``)."""

    return _param_rows(module.facets.get("weight"))


def _embedding_homes(trace: Any, role: str) -> list[Any]:
    """Return the embedding modules serving ``role`` (``token`` / ``position``).

    The token embedding is the lookup whose table size equals the
    unembedding's vocabulary (a structural fact, never a module-name
    guess); with no readable head, a lone embedding is the token embedding.
    Every other lookup is a position-side candidate, so BERT-style token-type
    tables make ``position`` ambiguous rather than silently wrong.
    """

    embeddings = _facet_home_modules(trace, "lookup")
    vocab = _vocab_size(trace)
    if vocab is None:
        token = embeddings[:1] if len(embeddings) == 1 else []
    else:
        token = [m for m in embeddings if _embedding_rows(m) == vocab]
    if role == "token":
        return token
    return [m for m in embeddings if m not in token]


def _homes_for(trace: Any, query: _Query, facet: str) -> list[Any]:
    """Return candidate home modules for a parsed query."""

    row = query.row or {}
    role = row.get("role")
    if role is not None:
        return _embedding_homes(trace, str(role))
    return _facet_home_modules(trace, facet, row.get("scope"))


def _layer_candidates(trace: Any, query: _Query, homes: list[Any]) -> list[Any]:
    """Return the executed-order modules a layer index counts over.

    A scoped or role-bearing row counts only its own homes; an unscoped
    query with no home falls back to the attention modules (the historical
    order, so ``("pattern", 1)`` still finds layer 1).
    """

    if homes:
        return homes
    row = query.row or {}
    if row.get("role") is not None or row.get("scope") in _SCOPE_KEYS:
        return []
    return _attention_modules(trace)


def _condition_absence(module: Any, row: dict[str, Any] | None) -> str | None:
    """Return the row's absence remedy when its facet condition fails, else None.

    ``only_if_facet`` / ``unless_facet`` express where a TransformerLens name
    means a DIFFERENT tensor by architecture (4.x ``mlp.hook_pre`` is the
    gate projection's raw output on gated MLPs), so the resolver answers
    honestly absent instead of serving the wrong tensor.
    """

    if not row:
        return None
    from ._heads import _facet_keys_or_none

    keys = _facet_keys_or_none(module) or set()
    only_if = row.get("only_if_facet")
    unless = row.get("unless_facet")
    if (only_if is not None and only_if not in keys) or (unless is not None and unless in keys):
        return str(row.get("absent_remedy") or "this name means a different tensor here")
    return None


def _status_for(module: Any, facet: str) -> tuple[str, str | None]:
    """Return (status, remedy) for one facet on one module."""

    menu = module.facets.declared()
    item = menu.get(facet)
    if item is None:
        return (
            "structurally_absent",
            "this architecture's recipes never declare the facet; see tl.facets.coverage(trace)",
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


def _resolve_on(module: Any, query: _Query, facet: str) -> AliasResolution:
    """Resolve a parsed query against its chosen home module."""

    absent = _condition_absence(module, query.row)
    if absent is not None:
        return AliasResolution(
            query=query.display,
            status="structurally_absent",
            facet=facet,
            module_address=str(module.address),
            remedy=absent,
            generation=query.generation,
        )
    status, remedy = _status_for(module, facet)
    site_key, pass_index = (
        _site_coordinates(module, facet) if status.startswith(("real", "recon")) else (None, None)
    )
    return AliasResolution(
        query=query.display,
        status=status,
        facet=facet,
        module_address=str(module.address),
        site_key=site_key,
        pass_index=pass_index,
        remedy=remedy,
        generation=query.generation,
    )


def resolve_alias(trace: Any, name: Any) -> AliasResolution:
    """Resolve one TLens spelling against a trace (mikit D16).

    Parameters
    ----------
    trace:
        A finished torchlens trace.
    name:
        Full TLens name (2.x, 3.x-bridge or 4.x canonical), tuple form
        ``(facet, layer)``, or compact ``"k6"``.

    Returns
    -------
    AliasResolution
        Never raises for an unresolvable name -- the status IS the answer.
    """

    return _resolve(trace, _parse_query(name))


def _resolve(trace: Any, query: _Query) -> AliasResolution:
    """Resolve a parsed query (shared by ``resolve_alias`` and ``alias_report``)."""

    if query.facet is None:
        return AliasResolution(
            query=query.display,
            status="structurally_absent",
            remedy="no translation row matches this spelling; see "
            "torchlens.mechinterp.translation_table() for the known (name, generation) rows",
        )
    facet = query.facet
    homes = _homes_for(trace, query, facet)
    if query.layer is not None:
        candidates = _layer_candidates(trace, query, homes)
        if query.layer >= len(candidates):
            return AliasResolution(
                query=query.display,
                status="structurally_absent",
                facet=facet,
                remedy=f"layer {query.layer} outside the {len(candidates)} executed modules "
                f"exposing {facet!r}",
                generation=query.generation,
            )
        return _resolve_on(candidates[query.layer], query, facet)
    if not homes:
        role = (query.row or {}).get("role")
        detail = (
            f"no {role} embedding in this trace (a rotary or position-free model has no "
            "learned position table)"
            if role
            else f"no module in this trace exposes {facet!r}"
        )
        return AliasResolution(
            query=query.display,
            status="structurally_absent",
            facet=facet,
            remedy=f"{detail}; see tl.facets.coverage(trace)",
            generation=query.generation,
        )
    if len(homes) > 1:
        return AliasResolution(
            query=query.display,
            status="ambiguous",
            facet=facet,
            remedy=f"{len(homes)} modules expose {facet!r}; qualify with a layer index "
            f"(e.g. ('{facet}', 0)) or a module address",
            generation=query.generation,
        )
    return _resolve_on(homes[0], query, facet)


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
            remedy='pass generation="2.x", "3.x-bridge" or "4.x"',
            generation=generation,
        )
    n_layers = len(_attention_modules(trace))
    report = []
    for row in rows:
        pattern = str(row["tlens"])
        if "{L}" in pattern:
            for layer in range(n_layers):
                name = pattern.replace("{L}", str(layer))
                report.append(_resolve(trace, _parse_query(name, prefer=generation)))
        else:
            report.append(_resolve(trace, _parse_query(pattern, prefer=generation)))
    return report
