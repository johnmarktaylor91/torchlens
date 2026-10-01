"""Target normalization: two families, exactly-stated requirements (D4).

Edge-seeded targets are PRIMARY: a site plus scalar coordinates, or a site
plus an explicit cotangent, normalizes to ``(target edge, cotangent)`` --
needing the target site's registry node only: no payload, no retention, no
graph-connected tensor. The tensor-expression family (a graph-connected
finite real scalar Tensor, or a pure ``Trace -> Tensor`` callable) is the
power-user door with real retention requirements; it refuses teachably,
naming the edge-seeded spelling, because the natural payload spelling is
clone-insulated from the graph on every default capture.

Cotangent shape validation reads recorded ``op.shape`` metadata, which
survives even unsaved ops -- the wrong-shape mistake names the site's
recorded shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._accessor import ReadEdgeIndex, SiteEdge
from ._errors import ReadError

__all__ = ["SeedTarget", "seed", "NormalizedTarget", "normalize_targets"]


@dataclass(frozen=True)
class SeedTarget:
    """An edge-seeded target request: a site plus coordinates or a cotangent.

    Attributes
    ----------
    site:
        Site address: a pass-qualified op label, bare layer label
        (single-pass layers only), Op, or Layer.
    index:
        Complete output-element coordinates (one-hot cotangent), mutually
        exclusive with ``cotangent``.
    cotangent:
        Explicit dense cotangent/projection over the site's full output
        shape, mutually exclusive with ``index``.
    """

    site: Any
    index: tuple[int, ...] | None = None
    cotangent: Any = None


def seed(site: Any, *, index: Any = None, cotangent: Any = None) -> SeedTarget:
    """Build an edge-seeded target (the primary target spelling).

    Parameters
    ----------
    site:
        Site address (pass-qualified label, Op, or Layer).
    index:
        Output-element coordinates for a one-hot cotangent.
    cotangent:
        Explicit cotangent tensor over the site's output shape.

    Returns
    -------
    SeedTarget
        Frozen request; validation happens at read time against the trace.
    """

    normalized_index = tuple(int(part) for part in index) if index is not None else None
    return SeedTarget(site=site, index=normalized_index, cotangent=cotangent)


@dataclass(frozen=True)
class NormalizedTarget:
    """One engine-ready target.

    Attributes
    ----------
    target_id:
        Stable order-preserving id (``t0``, ``t1``, ...; callable 1-D
        expansions append ``[j]``).
    family:
        ``'edge'`` (edge-seeded) or ``'tensor'`` (tensor-expression).
    edge:
        The seed :class:`SiteEdge` for edge-seeded targets, else ``None``.
    cotangent:
        Dense cotangent at the seed site (edge family) or over the
        expression result (tensor family).
    tensor:
        The graph-connected expression result (tensor family), else
        ``None``.
    cone_key:
        Chunk-grouping key: targets may share a batched VJP chunk ONLY when
        their cone keys are equal, because a batched call returns ZERO rows
        (not ``None``) for a (target, site) pair outside the target's cone
        -- fabricated zeros, the exact defect class D3 exists to prevent.
        Same seed site => same cone; same expression tensor => same cone.
    source_repr:
        Human disclosure of the request, for provenance.
    """

    target_id: str
    family: str
    edge: SiteEdge | None
    cotangent: torch.Tensor | None
    tensor: torch.Tensor | None
    cone_key: tuple[Any, ...]
    source_repr: str


def _site_label(site: Any) -> str | None:
    """Extract a label string from a site spec (label, Op, or Layer)."""

    if isinstance(site, str):
        return site
    label = getattr(site, "label", None)
    if isinstance(label, str):
        return label
    layer_label = getattr(site, "layer_label", None)
    if isinstance(layer_label, str):
        return layer_label
    return None


def resolve_site_edge(index: ReadEdgeIndex, site: Any, *, role: str) -> SiteEdge:
    """Resolve a site spec against the addressable-edge index, teachably.

    A bare layer label resolves iff exactly one pass exists; multi-pass
    layers require the pass-qualified spelling (consistent with the
    pass-qualified replay boundary).

    Parameters
    ----------
    index:
        The trace's addressable-edge index.
    site:
        Label / Op / Layer site spec.
    role:
        ``'target'`` or ``'population'`` -- used in refusal text.

    Raises
    ------
    ReadError
        Code ``read_site_unaddressable`` when the site has no usable
        ``(node, slot)``; code ``read_target_invalid`` for unparseable specs
        or ambiguous bare labels.
    """

    label = _site_label(site)
    if label is None:
        raise ReadError(
            f"Cannot interpret {type(site).__name__!r} as a {role} site. "
            "Remedy: pass a pass-qualified op label (e.g. 'mlp_1_2:1'), an "
            "Op, or a Layer",
            code="read_target_invalid",
            role=role,
        )
    if label in index.edges:
        return index.edges[label]
    matches = [edge for edge in index.edges.values() if edge.layer_label == label]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ReadError(
            f"Bare label {label!r} names a multi-pass layer "
            f"({len(matches)} passes). Remedy: use a pass-qualified "
            f"spelling such as {matches[0].label!r}",
            code="read_target_invalid",
            role=role,
            label=label,
            passes=[edge.label for edge in matches],
        )
    unaddressable_key = label if label in index.unaddressable else None
    if unaddressable_key is None:
        bare_matches = [key for key in index.unaddressable if key.rsplit(":", 1)[0] == label]
        if len(bare_matches) == 1:
            unaddressable_key = bare_matches[0]
    if unaddressable_key is not None:
        raise ReadError(
            f"Site {unaddressable_key!r} has no usable autograd (node, slot): "
            f"{index.unaddressable[unaddressable_key]}. Remedy: pick a "
            "differentiable op site (graph inputs and non-differentiable "
            "outputs have no producer node)",
            code="read_site_unaddressable",
            role=role,
            label=unaddressable_key,
            reason=index.unaddressable[unaddressable_key],
        )
    raise ReadError(
        f"Site {label!r} is not an op site of this trace. Remedy: pick a "
        "pass-qualified op label from the trace",
        code="read_target_invalid",
        role=role,
        label=label,
    )


def _one_hot_cotangent(edge: SiteEdge, index: tuple[int, ...]) -> torch.Tensor:
    """Build the one-hot cotangent for coordinates at a site, from metadata.

    Negative coordinates wrap Python-style; out-of-range coordinates refuse
    naming the recorded shape (the message carries the exact geometry so the
    user never needs the payload to fix the call).
    """

    shape = edge.shape
    if shape is None:
        raise ReadError(
            f"Site {edge.label!r} recorded no output shape; an index-seeded "
            "target cannot be validated. Remedy: pass an explicit cotangent=",
            code="read_target_invalid",
            label=edge.label,
        )
    if len(index) != len(shape):
        raise ReadError(
            f"Target index {index!r} has {len(index)} coordinates but site "
            f"{edge.label!r} recorded shape {shape!r} "
            f"({len(shape)} axes). Remedy: pass one coordinate per output "
            "axis",
            code="read_target_invalid",
            label=edge.label,
            recorded_shape=list(shape),
        )
    wrapped: list[int] = []
    for axis, (coordinate, extent) in enumerate(zip(index, shape, strict=True)):
        adjusted = coordinate + extent if coordinate < 0 else coordinate
        if not (0 <= adjusted < extent):
            raise ReadError(
                f"Target index {index!r} is out of range on axis {axis} for "
                f"site {edge.label!r} with recorded shape {shape!r}. "
                "Remedy: keep every coordinate within the recorded shape",
                code="read_target_invalid",
                label=edge.label,
                recorded_shape=list(shape),
                axis=axis,
            )
        wrapped.append(adjusted)
    dtype = edge.dtype if isinstance(edge.dtype, torch.dtype) else torch.float32
    if not dtype.is_floating_point:
        raise ReadError(
            f"Site {edge.label!r} has non-floating dtype {dtype}; it cannot "
            "seed a gradient target. Remedy: seed at a floating-point site",
            code="read_target_invalid",
            label=edge.label,
        )
    cotangent = torch.zeros(shape, dtype=dtype)
    cotangent[tuple(wrapped)] = 1.0
    return cotangent


def _validate_cotangent(edge: SiteEdge, cotangent: Any) -> torch.Tensor:
    """Validate an explicit user cotangent against recorded site metadata."""

    if not isinstance(cotangent, torch.Tensor):
        raise ReadError(
            f"cotangent for site {edge.label!r} must be a Tensor; got "
            f"{type(cotangent).__name__}. Remedy: pass a dense tensor over "
            f"the site's recorded shape {edge.shape!r}",
            code="read_target_invalid",
            label=edge.label,
        )
    if edge.shape is not None and tuple(cotangent.shape) != edge.shape:
        raise ReadError(
            f"cotangent shape {tuple(cotangent.shape)!r} does not match site "
            f"{edge.label!r} recorded shape {edge.shape!r}. Remedy: match "
            "the site's recorded output shape exactly",
            code="read_target_invalid",
            label=edge.label,
            recorded_shape=list(edge.shape),
            cotangent_shape=list(cotangent.shape),
        )
    if cotangent.is_complex():
        raise ReadError(
            f"cotangent for site {edge.label!r} is complex; targets must be "
            "real. Remedy: pass a real-dtype cotangent",
            code="read_target_invalid",
            label=edge.label,
        )
    if not bool(torch.isfinite(cotangent).all()):
        raise ReadError(
            f"cotangent for site {edge.label!r} carries non-finite values. "
            "Remedy: pass a finite cotangent",
            code="read_target_invalid",
            label=edge.label,
        )
    return cotangent.detach()


def _validate_expression_tensor(value: Any, *, source_repr: str) -> torch.Tensor:
    """Validate a tensor-expression target (graph-connected, finite, real)."""

    if not isinstance(value, torch.Tensor):
        raise ReadError(
            f"Target {source_repr} produced {type(value).__name__}, not a "
            "Tensor. Remedy: return a scalar or 1-D tensor computed from "
            "graph-connected values",
            code="read_target_invalid",
            target=source_repr,
        )
    if value.grad_fn is None:
        raise ReadError(
            f"Target {source_repr} is not connected to the autograd graph "
            "(saved payloads are clone-insulated on default captures, so "
            "spellings like trace[label].out[...] cannot seed a backward). "
            "Remedy: use the edge-seeded spelling seed(site, index=...) or "
            "seed(site, cotangent=...), which needs no payload at all; for "
            "tensor expressions, capture with "
            "CaptureOptions(backward_ready=True)",
            code="read_target_invalid",
            target=source_repr,
        )
    if value.is_complex():
        raise ReadError(
            f"Target {source_repr} is complex; targets must be real. "
            "Remedy: reduce to a real scalar first",
            code="read_target_invalid",
            target=source_repr,
        )
    if value.dim() > 1:
        raise ReadError(
            f"Target {source_repr} has shape {tuple(value.shape)!r}; only a "
            "scalar or a 1-D target batch is accepted -- a multi-dim result "
            "is never silently summed. Remedy: reduce it explicitly or "
            "flatten to 1-D",
            code="read_target_invalid",
            target=source_repr,
        )
    if not bool(torch.isfinite(value).all()):
        raise ReadError(
            f"Target {source_repr} carries non-finite values. Remedy: fix the target expression",
            code="read_target_invalid",
            target=source_repr,
        )
    return value


def _normalize_seed(index: ReadEdgeIndex, request: SeedTarget, target_id: str) -> NormalizedTarget:
    """Normalize one edge-seeded request (the primary target family)."""

    edge = resolve_site_edge(index, request.site, role="target")
    has_index = request.index is not None
    has_cotangent = request.cotangent is not None
    if has_index == has_cotangent:
        raise ReadError(
            f"seed(site={edge.label!r}) needs exactly one of index= "
            "(one-hot coordinates) or cotangent= (explicit projection); got "
            f"{'both' if has_index else 'neither'}. Remedy: pass exactly one",
            code="read_target_invalid",
            label=edge.label,
        )
    if has_index:
        cotangent = _one_hot_cotangent(edge, request.index or ())
        source_repr = f"seed({edge.label!r}, index={request.index!r})"
    else:
        cotangent = _validate_cotangent(edge, request.cotangent)
        source_repr = f"seed({edge.label!r}, cotangent=<{tuple(cotangent.shape)}>)"
    return NormalizedTarget(
        target_id=target_id,
        family="edge",
        edge=edge,
        cotangent=cotangent,
        tensor=None,
        cone_key=("site", edge.alias_key),
        source_repr=source_repr,
    )


def normalize_targets(
    trace: Any,
    index: ReadEdgeIndex,
    target: Any,
) -> tuple[NormalizedTarget, ...]:
    """Normalize the user ``target=`` into engine-ready targets, order kept.

    Accepted forms: a :class:`SeedTarget` (or ``seed(...)`` result), a
    graph-connected scalar Tensor, a pure ``Trace -> Tensor`` callable
    (scalar, or 1-D = target batch, never silently summed), or a
    list/tuple of these.

    Parameters
    ----------
    trace:
        The resolution trace (passed to callable targets exactly once).
    index:
        Addressable-edge index of the trace.
    target:
        The user target request.

    Returns
    -------
    tuple[NormalizedTarget, ...]
        One entry per resolved target, order preserved.
    """

    requests = list(target) if isinstance(target, (list, tuple)) else [target]
    if not requests:
        raise ReadError(
            "target= is empty. Remedy: pass at least one target",
            code="read_target_invalid",
        )
    normalized: list[NormalizedTarget] = []
    for position, request in enumerate(requests):
        target_id = f"t{position}"
        if isinstance(request, SeedTarget):
            normalized.append(_normalize_seed(index, request, target_id))
            continue
        if isinstance(request, torch.Tensor):
            value = _validate_expression_tensor(request, source_repr=f"targets[{position}]")
            if value.dim() != 0 and value.numel() != 1:
                raise ReadError(
                    f"A bare Tensor target must be scalar; targets[{position}] "
                    f"has shape {tuple(value.shape)!r}. Remedy: use a "
                    "callable target returning 1-D for a target batch",
                    code="read_target_invalid",
                )
            normalized.append(
                NormalizedTarget(
                    target_id=target_id,
                    family="tensor",
                    edge=None,
                    cotangent=torch.ones_like(value).detach(),
                    tensor=value,
                    cone_key=("expr", id(value)),
                    source_repr=f"tensor targets[{position}]",
                )
            )
            continue
        if callable(request):
            produced = request(trace)
            value = _validate_expression_tensor(
                produced, source_repr=getattr(request, "__name__", f"targets[{position}]")
            )
            if value.dim() == 0:
                normalized.append(
                    NormalizedTarget(
                        target_id=target_id,
                        family="tensor",
                        edge=None,
                        cotangent=torch.ones_like(value).detach(),
                        tensor=value,
                        cone_key=("expr", id(value)),
                        source_repr=getattr(request, "__name__", f"targets[{position}]"),
                    )
                )
                continue
            for element in range(value.shape[0]):
                cotangent = torch.zeros_like(value).detach()
                cotangent[element] = 1.0
                normalized.append(
                    NormalizedTarget(
                        target_id=f"{target_id}[{element}]",
                        family="tensor",
                        edge=None,
                        cotangent=cotangent,
                        tensor=value,
                        cone_key=("expr", id(value)),
                        source_repr=(
                            f"{getattr(request, '__name__', f'targets[{position}]')}[{element}]"
                        ),
                    )
                )
            continue
        raise ReadError(
            f"Cannot interpret targets[{position}] "
            f"({type(request).__name__!r}) as a target. Remedy: pass "
            "seed(site, index=...), seed(site, cotangent=...), a "
            "graph-connected scalar Tensor, or a Trace -> Tensor callable",
            code="read_target_invalid",
            position=position,
        )
    return tuple(normalized)
