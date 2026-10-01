"""Retention plans (mikit D15): printable, priceable, composable save plans.

Every kit function REFUSES with the complete missing-site set when payloads
are absent -- never a partial table, never a silent widening to
``save="all"``. The plan object closes the loop: built metadata-first from
any trace (structure-only captures included), it names the sites each
analysis needs, prices them, and exposes a ``.predicate`` that composes
with ordinary ``save=`` for the selective RECAPTURE (the generic discovery
route is two forwards, disclosed).

The predicate's match basis is (func_name, nearest-module-address) pairs
plus whole-module subtrees -- a disclosed SUPERSET of the exact site set
(capture-time events predate site-key minting; the exact-key predicate is
the shared-plumbing follow-up). A superset save is disclosed, never silent.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ._errors import refuse
from ._walk import walk_spine

__all__ = ["RetentionPlan", "retention_plan"]

_ANALYSES = (
    "accumulation",
    "decomposition",
    "dla",
    "heads",
    "head_scores",
    "prompts",
)


@dataclass(frozen=True)
class PlanSite:
    """One planned retention site.

    Parameters
    ----------
    op_label:
        Pass-qualified captured op label (from the planning trace).
    site_key:
        Live structural site key, when minted.
    func_name:
        Backend function name (a predicate match basis).
    module_address:
        Nearest module address (the other match basis).
    estimated_bytes:
        Payload size estimate from recorded shape/dtype.
    analysis:
        Which analysis wants the site.
    """

    op_label: str
    site_key: str | None
    func_name: str
    module_address: str | None
    estimated_bytes: int
    analysis: str


@dataclass(frozen=True)
class RetentionPlan:
    """A priced, composable retention plan (mikit D15).

    ``predicate`` composes with ordinary ``save=``; ``module_subtrees`` are
    whole-module saves (attention interiors for head analyses). The
    ``basis`` string disclosing the superset match is part of the contract.
    """

    analyses: tuple[str, ...]
    sites: tuple[PlanSite, ...]
    module_subtrees: tuple[str, ...]
    basis: str = (
        "func_name + nearest-module-address pairs, plus whole module subtrees: a "
        "DISCLOSED superset of the exact site set (all passes of matched ops save)"
    )
    _op_pairs: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    @property
    def estimated_bytes(self) -> int:
        """Return the summed payload estimate for the named sites."""

        return sum(site.estimated_bytes for site in self.sites)

    @property
    def predicate(self) -> Callable[[Any], bool]:
        """Return a ``save=``-composable predicate matching the plan."""

        pairs = self._op_pairs
        subtrees = self.module_subtrees

        def _matches(ctx: Any) -> bool:
            """Match one capture event against the plan (superset basis)."""

            address = str(getattr(ctx, "address", None) or "")
            for subtree in subtrees:
                if address == subtree or address.startswith(subtree + "."):
                    return True
            func_name = str(getattr(ctx, "func_name", None) or "")
            return (func_name, address) in pairs

        return _matches

    def __repr__(self) -> str:
        """Return the printable plan."""

        megabytes = self.estimated_bytes / 1e6
        return (
            f"RetentionPlan(analyses={list(self.analyses)}, sites={len(self.sites)}, "
            f"module_subtrees={list(self.module_subtrees)}, "
            f"~{megabytes:.1f} MB; basis: {self.basis})"
        )

    def describe(self) -> str:
        """Return the full per-site table as text."""

        lines = [repr(self)]
        for site in self.sites:
            lines.append(
                f"  {site.analysis:14s} {site.op_label:24s} "
                f"{site.module_address or '-':32s} ~{site.estimated_bytes / 1e6:.2f} MB"
            )
        return "\n".join(lines)


def _estimated_bytes(op: Any) -> int:
    """Estimate one op's payload size from recorded shape and dtype."""

    shape = getattr(op, "shape", None) or ()
    try:
        numel = 1
        for dim in shape:
            numel *= int(dim)
    except (TypeError, ValueError):
        return 0
    dtype = str(getattr(op, "dtype", "") or "")
    width = 2 if ("16" in dtype) else (8 if "64" in dtype else 4)
    return numel * width


def _site(trace: Any, label: str, analysis: str) -> PlanSite:
    """Build one plan site from a captured op label."""

    op = trace.ops[label]
    modules = tuple(getattr(op, "modules", ()) or ())
    address = None
    if modules:
        base, sep, tail = str(modules[-1]).rpartition(":")
        address = base if sep and tail.isdigit() else str(modules[-1])
    return PlanSite(
        op_label=str(getattr(op, "label", label)),
        site_key=getattr(op, "site_key", None),
        func_name=str(getattr(op, "func_name", "")),
        module_address=address,
        estimated_bytes=_estimated_bytes(op),
        analysis=analysis,
    )


def _spine_sites(trace: Any, analysis: str) -> list[PlanSite]:
    """Sites for the residual family: spine states + writers + norm + logits."""

    from ._anchors import resolve_lm_head

    anchor = resolve_lm_head(trace)
    walk = walk_spine(trace, anchor.target_op, structure_only=True)
    labels: list[str] = []
    for node in walk.nodes:
        labels.append(str(getattr(node.op, "label", "")))
        labels.extend(node.writer_labels)
    sites = [_site(trace, label, analysis) for label in labels]
    norm_outputs = list(getattr(anchor.norm, "output_ops", ()) or ())
    if norm_outputs:
        sites.append(_site(trace, str(norm_outputs[0]), analysis))
    head_outputs = list(getattr(anchor.head, "output_ops", ()) or ())
    if head_outputs:
        sites.append(_site(trace, str(head_outputs[0]), analysis))
    return sites


def _attention_subtrees(trace: Any) -> list[str]:
    """Attention module addresses (whole-subtree saves for head analyses)."""

    from ._heads import _attention_modules

    return [str(getattr(m, "address", "")) for m in _attention_modules(trace)]


def retention_plan(trace: Any, analyses: Any = ("decomposition", "dla")) -> RetentionPlan:
    """Build the retention plan for the named analyses (mikit D15).

    Parameters
    ----------
    trace:
        Any finished trace of the model -- a cheap metadata-first capture
        works; the plan is structural (the two-forward discovery route:
        capture metadata, plan, recapture selectively with
        ``save=plan.predicate``).
    analyses:
        Analysis names among ``accumulation`` / ``decomposition`` / ``dla`` /
        ``heads`` / ``head_scores`` / ``prompts``.

    Returns
    -------
    RetentionPlan
        Printable, priced, ``save=``-composable.
    """

    wanted = [analyses] if isinstance(analyses, str) else list(analyses)
    unknown = [name for name in wanted if name not in _ANALYSES]
    if unknown:
        refuse(
            code="mi_analysis_unknown",
            message=f"Unknown analyses {unknown!r}.",
            remedy=f"choose among {list(_ANALYSES)}",
            unknown=unknown,
        )
    sites: list[PlanSite] = []
    subtrees: list[str] = []
    for name in wanted:
        if name in ("accumulation", "decomposition", "dla", "prompts"):
            sites.extend(_spine_sites(trace, name))
        if name in ("heads", "head_scores"):
            subtrees.extend(_attention_subtrees(trace))
    deduped: dict[str, PlanSite] = {}
    for site in sites:
        deduped.setdefault(site.op_label, site)
    pairs = frozenset((site.func_name, site.module_address or "") for site in deduped.values())
    return RetentionPlan(
        analyses=tuple(wanted),
        sites=tuple(deduped.values()),
        module_subtrees=tuple(dict.fromkeys(subtrees)),
        _op_pairs=pairs,
    )
