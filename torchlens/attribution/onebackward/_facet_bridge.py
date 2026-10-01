"""Read-side facet -> site bridge (M(reads) item 3b, decision D7).

The frozen resolver is SEMANTIC-EVIDENCE-ONLY: it consumes the facet system's
recipe-matched module views (``trace.modules_with_facet("output")`` -- the
"output" facet is declared only by the MLP recipe family) and each facet's
op-home spec. Name substrings are FORBIDDEN here: substring heuristics failed
three measured times in four panel rounds on real gpt2 (a gelu substring
found 0 MLP sites; an "mlp" path filter selected 48 sites where the true
answer is 12) -- the guard pin rides the acceptance tests.

The resolver is a policy REGISTRY so the R10 frozen-LN lane adds
``'ln_scale'`` and a composite preset later without redesign; the v1
vocabulary is ``{'mlp_out'}`` only, and there is deliberately NO
``'attribution_graph'`` preset name until R10 lands (a preset that
under-delivers published semantics is an overclaim hazard).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._accessor import ReadEdgeIndex, SiteEdge
from ._errors import ReadError

__all__ = [
    "FrozenResolution",
    "resolve_frozen_policy",
    "register_frozen_policy",
    "FROZEN_POLICY_NAMES",
]


@dataclass(frozen=True)
class FrozenResolution:
    """Resolved freeze-site set for one policy on one trace.

    Attributes
    ----------
    policy:
        Registry policy name (``'mlp_out'`` in v1).
    sites:
        Resolved freeze sites, label -> :class:`SiteEdge`.
    attention_bearing:
        Whether the trace carries semantically-evidenced attention blocks
        (modules with a ``q`` facet) -- the disclosure detector's input.
    disclosures:
        Human-readable notes (empty-set disclosure on MLP-free
        architectures, fused-grain notes).
    """

    policy: str
    sites: dict[str, SiteEdge]
    attention_bearing: bool
    disclosures: tuple[str, ...]


def _facet_available(module: Any, name: str) -> bool:
    """Return whether a module's facet view serves ``name``, guarded.

    A module whose facet view refuses evaluation (for example a
    twice-invoked module with call-scoped facets) contributes no evidence
    rather than crashing the resolver.
    """

    try:
        return bool(module.facets.has(name))
    except Exception:  # noqa: BLE001 - absence of evidence, not an error
        return False


def _attention_bearing(trace: Any) -> bool:
    """Return whether the trace has semantically-evidenced attention blocks."""

    return any(_facet_available(module, "q") for module in getattr(trace, "modules", ()) or ())


def _resolve_mlp_out(trace: Any, index: ReadEdgeIndex) -> FrozenResolution:
    """Resolve the global MLP-output stop set (the D5 default policy).

    A module qualifies through semantic facet evidence only: its recipe view
    exposes an ``"output"`` facet whose home is a whole op output
    (``home_kind='op'``, empty ``output_path``, no transforms) and whose home
    op is addressable in the edge index. On attention-bearing graphs whose
    MLP homes are not resolvable, this REFUSES typed
    (``default_frozen_sites_underivable``) naming the modules -- never
    warn-and-proceed, never substring guessing. On architectures with no
    applicable MLPs (ResNet) the resolution is the disclosed empty set.
    """

    attention = _attention_bearing(trace)
    sites: dict[str, SiteEdge] = {}
    untrustworthy: list[str] = []
    candidates = [
        module
        for module in getattr(trace, "modules", ()) or ()
        if _facet_available(module, "output")
    ]
    for module in candidates:
        address = getattr(module, "address", repr(module))
        try:
            facet = module.facets["output"]
            spec = facet.spec
        except Exception:  # noqa: BLE001 - any resolution failure is evidence
            untrustworthy.append(str(address))
            continue
        home_label = getattr(spec, "home_label", None)
        if (
            getattr(spec, "home_kind", None) != "op"
            or not home_label
            or tuple(getattr(spec, "output_path", ()) or ()) != ()
            or tuple(getattr(spec, "transforms", ()) or ()) != ()
        ):
            untrustworthy.append(str(address))
            continue
        edge = index.edges.get(home_label)
        if edge is None:
            untrustworthy.append(str(address))
            continue
        sites[edge.label] = edge
    if attention and (untrustworthy or not sites):
        raise ReadError(
            "The default linearization freezes transformer MLP-output sites, "
            "but this attention-bearing capture has no semantically "
            f"resolvable MLP-output homes ({len(sites)} resolved; "
            f"unresolvable modules: {untrustworthy or 'none found'}). The "
            "resolver never guesses from names. Remedy: pass an explicit "
            "frozen= Selection of stop sites, or frozen=None for the total "
            "derivative",
            code="default_frozen_sites_underivable",
            resolved=sorted(sites),
            unresolvable_modules=untrustworthy,
        )
    disclosures: tuple[str, ...] = ()
    if not attention and not sites:
        disclosures = (
            "default frozen policy 'mlp_out' resolved an empty set on this "
            "architecture (no transformer MLP facet homes); the read is the "
            "total derivative",
        )
    return FrozenResolution(
        policy="mlp_out",
        sites=sites,
        attention_bearing=attention,
        disclosures=disclosures,
    )


FrozenPolicyResolver = Callable[[Any, ReadEdgeIndex], FrozenResolution]

_FROZEN_POLICIES: dict[str, FrozenPolicyResolver] = {"mlp_out": _resolve_mlp_out}

FROZEN_POLICY_NAMES: tuple[str, ...] = ("mlp_out",)


def register_frozen_policy(name: str, resolver: FrozenPolicyResolver) -> None:
    """Register a frozen-policy resolver (internal seam for the R10 lane).

    Parameters
    ----------
    name:
        Policy vocabulary token (for example ``'ln_scale'``).
    resolver:
        ``(trace, edge_index) -> FrozenResolution``.

    Raises
    ------
    ReadError
        Code ``read_frozen_policy_invalid`` on a duplicate registration --
        policies are declared once, never silently replaced.
    """

    if name in _FROZEN_POLICIES:
        raise ReadError(
            f"Frozen policy {name!r} is already registered. Remedy: pick an "
            "unregistered policy name",
            code="read_frozen_policy_invalid",
            policy=name,
        )
    _FROZEN_POLICIES[name] = resolver
    global FROZEN_POLICY_NAMES
    FROZEN_POLICY_NAMES = tuple(_FROZEN_POLICIES)


def resolve_frozen_policy(trace: Any, index: ReadEdgeIndex, name: str) -> FrozenResolution:
    """Resolve a registered frozen policy against a trace.

    Parameters
    ----------
    trace:
        The resolution trace.
    index:
        Its addressable-edge index.
    name:
        Registered policy name (v1: ``'mlp_out'``).

    Raises
    ------
    ReadError
        Code ``read_frozen_policy_invalid`` for unknown names.
    """

    resolver = _FROZEN_POLICIES.get(name)
    if resolver is None:
        raise ReadError(
            f"Unknown frozen policy {name!r}; registered policies: "
            f"{sorted(_FROZEN_POLICIES)}. Remedy: pick a registered policy "
            "or pass an explicit frozen= Selection",
            code="read_frozen_policy_invalid",
            policy=name,
            registered=sorted(_FROZEN_POLICIES),
        )
    return resolver(trace, index)
