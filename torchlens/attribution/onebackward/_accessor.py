"""Site -> ``(autograd node, output slot)`` accessor (M(reads) item 0, D1).

The substrate already exists on every finished torch trace: each op records
its real autograd producer (``op.grad_fn_handle``, live), a durable object id
(``op.grad_fn_object_id``, portable), and its output slot
(``op.multi_output_index`` when ``op.in_multi_output``). This module is the
XS-S accessor over those EXISTING fields -- no new recording, no arming flag,
no capture option: any graph-alive trace under any save mode is addressable
with zero capture-time preparation.

The id rejoin is a FALLBACK, not a co-primary source: a suppressed read never
consumes the handles, so the rejoin exists for the one real case of a user
``log_backward`` before the read (which nulls per-op handles while the
trace's strong-ref list keeps the objects alive).

Alias identity is ``(id(node), slot)`` (D11): distinct slots of one split
node are NOT aliases (q/k/v must never merge); distinct ops producing the
same ``(node, slot)`` are true aliases sharing one gradient.
"""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import dataclass
from typing import Any

from ...utils._torch_compat import HAS_GRADIENT_EDGE
from ._errors import ReadError

__all__ = [
    "SiteEdge",
    "ReadEdgeIndex",
    "read_edge_index",
    "require_gradient_edge_support",
]

# The v1 read is live-PyTorch-only (D14); the named constant keeps the gate
# greppable without a hard-coded backend literal branch (the registry lint).
_SUPPORTED_BACKEND: str = "torch"

# Closed vocabulary for trace-level addressing refusals (fields["reason"]).
TRACE_REFUSAL_REASONS: tuple[str, ...] = (
    "cleaned",
    "inference_only",
    "chunked_forward",
    "structure_only",
    "detached",
    "not_live",
    "backend_unsupported",
)

# Closed vocabulary for per-site unaddressability (index.unaddressable values).
SITE_REFUSAL_REASONS: tuple[str, ...] = (
    "no_grad_fn",
    "handle_dead",
)


def require_gradient_edge_support() -> None:
    """Refuse typed when the running torch lacks the GradientEdge surface.

    Raises
    ------
    ReadError
        With code ``onebackward_torch_unsupported`` when
        ``torch.autograd.graph.GradientEdge`` is unavailable (torch older
        than the read's supported band).
    """

    if not HAS_GRADIENT_EDGE:
        raise ReadError(
            "One-backward reads require torch's GradientEdge surface "
            "(torch.autograd.graph.GradientEdge), which this torch build "
            "does not provide. Remedy: upgrade torch to a version providing "
            "torch.autograd.graph.GradientEdge (2.4+)",
            code="onebackward_torch_unsupported",
        )


@dataclass(frozen=True)
class SiteEdge:
    """One addressable read site: a live autograd node plus its output slot.

    Attributes
    ----------
    label:
        Pass-qualified op label (``Op.label``, the ``layer_label:pass``
        spelling) -- the session address.
    pass_index:
        1-based pass index parsed from the label.
    layer_label:
        Bare layer label (label without the pass suffix).
    site_key:
        Portable L1 structural site key, or ``None`` on keyless records.
        Carried as a SEPARATE field, never overloaded onto the address.
    node:
        Live autograd node (the registry object, not a payload walk).
    slot:
        Output slot on ``node``: ``multi_output_index`` when the op is part
        of a multi-output call, else ``0`` (the D1 slot rule).
    via_rejoin:
        ``True`` when the node was recovered through the object-id rejoin
        fallback rather than the per-op live handle.
    shape:
        Recorded output shape metadata (survives even unsaved ops); used for
        cotangent validation without touching payloads.
    dtype:
        Recorded output dtype, or ``None`` when unrecorded.
    """

    label: str
    pass_index: int
    layer_label: str
    site_key: str | None
    node: Any
    slot: int
    via_rejoin: bool
    shape: tuple[int, ...] | None
    dtype: Any

    @property
    def alias_key(self) -> tuple[int, int]:
        """Identity key for engine dedup: ``(id(node), slot)`` (D11)."""

        return (id(self.node), self.slot)

    def gradient_edge(self) -> Any:
        """Return the ``torch.autograd.graph.GradientEdge`` for this site."""

        from torch.autograd.graph import GradientEdge

        return GradientEdge(self.node, self.slot)


@dataclass(frozen=True)
class ReadEdgeIndex:
    """Addressable-site index for one trace at one validity token.

    Attributes
    ----------
    edges:
        Pass-qualified label -> :class:`SiteEdge` for every addressable op.
    unaddressable:
        Pass-qualified label -> closed reason for ops without a resolvable
        ``(node, slot)`` (graph inputs, non-differentiable outputs).
    alias_groups:
        ``(id(node), slot)`` -> tuple of member labels, for every key with
        two or more members (true aliases sharing one gradient).
    token:
        Cache-validity token captured at build time.
    """

    edges: dict[str, SiteEdge]
    unaddressable: dict[str, str]
    alias_groups: dict[tuple[int, int], tuple[str, ...]]
    token: tuple[Any, ...]


_INDEX_CACHE: weakref.WeakKeyDictionary[Any, ReadEdgeIndex] = weakref.WeakKeyDictionary()


def _validity_token(trace: Any) -> tuple[Any, ...]:
    """Return the cheap invalidation token for a trace's edge index.

    A user ``log_backward`` advances ``num_backward_passes`` (and nulls
    per-op handles -> the rejoin path), cleanup flips ``_tl_cleaned_up``, and
    both must invalidate a cached index. Suppressed reads change neither.
    """

    refs = trace.__dict__.get("_backward_gradfn_refs")
    return (
        int(getattr(trace, "num_backward_passes", 0)),
        None if refs is None else len(refs),
        bool(trace.__dict__.get("_tl_cleaned_up", False)),
    )


def _refuse_trace(trace: Any, reason: str, detail: str, remedy: str) -> None:
    """Raise the typed trace-level addressing refusal.

    Parameters
    ----------
    trace:
        The refused trace.
    reason:
        Member of ``TRACE_REFUSAL_REASONS``.
    detail:
        One-sentence explanation of what makes the trace unaddressable.
    remedy:
        Exact recapture or ordering remedy, without the ``Remedy:`` prefix.
    """

    raise ReadError(
        f"One-backward reads need a live autograd registry, and this trace "
        f"cannot serve one: {detail} Remedy: {remedy}",
        code="read_addressing_unavailable",
        reason=reason,
        trace_backend=getattr(trace, "backend", None),
    )


def _check_trace_liveness(trace: Any) -> None:
    """Run the closed liveness gate (D1/D14) before any index build.

    Raises
    ------
    ReadError
        Code ``read_addressing_unavailable`` with ``fields["reason"]`` from
        ``TRACE_REFUSAL_REASONS`` on loaded / cleaned / inference-only /
        chunked / structure-only / detached / non-torch traces.
    """

    backend = getattr(trace, "backend", None)
    if backend is not None and backend != _SUPPORTED_BACKEND:
        _refuse_trace(
            trace,
            "backend_unsupported",
            f"the v1 read is live-PyTorch-only and this trace's backend is {backend!r}.",
            "capture with the torch backend",
        )
    if trace.__dict__.get("_tl_cleaned_up", False):
        _refuse_trace(
            trace,
            "cleaned",
            "Trace.cleanup() released its autograd registry.",
            "re-capture the model and read before cleanup",
        )
    if getattr(trace, "structure_only", False):
        _refuse_trace(
            trace,
            "structure_only",
            "structure-only captures retain no values and no autograd graph.",
            "re-capture without CaptureOptions(structure_only=True)",
        )
    if getattr(trace, "inference_only", False):
        _refuse_trace(
            trace,
            "inference_only",
            "inference-only captures discard the autograd graph.",
            "re-capture without inference_only=True",
        )
    if getattr(trace, "chunked_forward", False):
        _refuse_trace(
            trace,
            "chunked_forward",
            "chunk-assembled traces hold no single live autograd graph.",
            "re-capture without forward chunking",
        )
    if getattr(trace, "detach_saved_activations", False):
        _refuse_trace(
            trace,
            "detached",
            "the capture detached saved activations from the autograd graph.",
            "re-capture without detach_saved_activations=True",
        )


def _resolve_node(op: Any, rejoin_map: dict[int, Any]) -> tuple[Any, bool] | None:
    """Resolve the live autograd node for one op, primary then rejoin.

    Returns
    -------
    tuple[Any, bool] | None
        ``(node, via_rejoin)`` or ``None`` when the op has no resolvable
        producer node.
    """

    handle = getattr(op, "grad_fn_handle", None)
    if handle is not None:
        return handle, False
    object_id = getattr(op, "grad_fn_object_id", None)
    if object_id is not None:
        node = rejoin_map.get(object_id)
        if node is not None:
            return node, True
    return None


def _build_index(trace: Any, token: tuple[Any, ...]) -> ReadEdgeIndex:
    """Build the edge index for a liveness-checked trace."""

    refs = trace.__dict__.get("_backward_gradfn_refs") or ()
    rejoin_map = {id(node): node for node in refs}
    edges: dict[str, SiteEdge] = {}
    unaddressable: dict[str, str] = {}
    alias_members: dict[tuple[int, int], list[str]] = {}
    for op in trace.ops:
        label = op.label
        resolved = _resolve_node(op, rejoin_map)
        if resolved is None:
            has_id = getattr(op, "grad_fn_object_id", None) is not None
            unaddressable[label] = "handle_dead" if has_id else "no_grad_fn"
            continue
        node, via_rejoin = resolved
        slot_raw = getattr(op, "multi_output_index", None)
        slot = (
            int(slot_raw) if getattr(op, "in_multi_output", False) and slot_raw is not None else 0
        )
        shape = getattr(op, "shape", None)
        edge = SiteEdge(
            label=label,
            pass_index=int(getattr(op, "pass_index", 1) or 1),
            layer_label=getattr(op, "layer_label", label.rsplit(":", 1)[0]),
            site_key=getattr(op, "site_key", None),
            node=node,
            slot=slot,
            via_rejoin=via_rejoin,
            shape=tuple(shape) if shape is not None else None,
            dtype=getattr(op, "dtype", None),
        )
        edges[label] = edge
        alias_members.setdefault(edge.alias_key, []).append(label)
    if not edges:
        _refuse_trace(
            trace,
            "not_live",
            "no op carries a live or rejoinable autograd node (a loaded or "
            "fully detached artifact records which nodes produced each value "
            "but not the nodes themselves).",
            "re-capture the model in this process and read the live trace",
        )
    alias_groups = {
        key: tuple(members) for key, members in alias_members.items() if len(members) > 1
    }
    return ReadEdgeIndex(
        edges=edges, unaddressable=unaddressable, alias_groups=alias_groups, token=token
    )


def read_edge_index(trace: Any) -> ReadEdgeIndex:
    """Return the (cached) addressable-site index for a live trace.

    The cache invalidates on the cheap validity token (backward-pass count,
    pinned-ref count, cleanup flag), so a user ``log_backward`` between reads
    transparently rebuilds through the id-rejoin fallback while suppressed
    reads reuse the cached index.

    Parameters
    ----------
    trace:
        Finished live torch trace.

    Returns
    -------
    ReadEdgeIndex
        Addressable sites, per-site refusal reasons, and alias groups.

    Raises
    ------
    ReadError
        Code ``onebackward_torch_unsupported`` without GradientEdge support;
        code ``read_addressing_unavailable`` on the closed liveness gate.
    """

    require_gradient_edge_support()
    _check_trace_liveness(trace)
    token = _validity_token(trace)
    cached = _INDEX_CACHE.get(trace)
    if cached is not None and cached.token == token:
        return cached
    index = _build_index(trace, token)
    # Non-weakref-able trace stand-ins (tests) simply skip the cache.
    with contextlib.suppress(TypeError):
        _INDEX_CACHE[trace] = index
    return index
