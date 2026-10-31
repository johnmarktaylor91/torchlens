"""Layer-level site accessors: ``site_key``, ``site_peers``, ``shape_summary``.

Implementation home for the thin ``Layer`` properties (extracted per the
file-size ratchet). The Layer surface exposes the op-granular
``site_key_v1`` bridging relation (:mod:`..postprocess._site_key`) with the
L1 design memo's typed refusals:

* ``Layer.site_key`` returns the single key iff every op in the layer
  shares exactly one; a site-SPANNING layer (within-call-instance
  recurrence, root loops) refuses typed -- a silent single-key read on a
  spanning layer is exactly the failure the tripwire exists to catch
  (invariant I-S3').
* ``Layer.site_peers`` indexes the trace live (recomputed per call, never
  persisted -- storing it would be a coherence liability) and refuses typed
  when this layer carries no valid keys: a legacy artifact must never
  collapse into a ``None``-key peer-of-everything.
* ``Layer.shape_summary`` is DERIVED, never persisted (memo RQ4): a plain
  data string summarizing output shapes ACROSS PASSES of one layer (the
  internal ``ModuleRepeatFold.shape_summary`` summarizes across a repeated
  module RUN -- a different axis). Rendering/escaping is L5 territory (S5):
  the value may legitimately contain ``->`` and must be HTML-escaped at
  render, never asserted absent.

Refusal codes here are PROVISIONAL pending the S2 vocabulary amendment
(design memo 4.3): ``layer_site_ambiguous``, ``site_key_unavailable``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from .layer import Layer

#: Characters legal in a shape_summary string (the S5 handoff character-class
#: pin): digits, the axis separator, the monotone arrow, the range dash.
SHAPE_SUMMARY_CHARACTER_CLASS = frozenset("0123456789x->-")


def _pass_ordered_ops(layer: Layer) -> list[Any]:
    """Return the layer's ops in pass order.

    ``Layer.ops`` is keyed by 1-based pass index (positional ints are a
    separate 0-based access path); sorting the items yields pass order.
    """

    return [op for _, op in sorted(layer.ops.items())]


def layer_site_key(layer: Layer) -> str:
    """Return the layer's single site key, or refuse typed (I-S3').

    Raises
    ------
    InvalidArgumentError
        ``site_key_unavailable`` when no op carries a key (legacy artifact
        written before site keys existed); ``layer_site_ambiguous`` when the
        layer's ops span multiple sites -- read the exact per-pass key via
        ``.ops[k].site_key``.
    """

    keys = {op.site_key for op in _pass_ordered_ops(layer)}
    if keys == {None}:
        raise InvalidArgumentError(
            f"Layer '{layer.layer_label}' carries no site keys: this artifact "
            "predates the site_key_v1 grouping surface. Site keys are minted "
            "at capture time and their persisted row is pre-release-gated "
            "under tlspec v7, so loaded artifacts (any tlspec v6 artifact, "
            "and v7 artifacts written before the coordinated persistence "
            "bump) read keyless.",
            code="site_key_unavailable",
            remedy=(
                "re-capture the model with a current TorchLens and read site_key on the live trace"
            ),
            layer_label=layer.layer_label,
        )
    keys.discard(None)
    if len(keys) > 1:
        raise InvalidArgumentError(
            f"Layer '{layer.layer_label}' spans {len(keys)} structural sites "
            "(within-call-instance recurrence or a root-context loop); a "
            "single site_key read would be silently wrong.",
            code="layer_site_ambiguous",
            remedy="read the per-pass key via .ops[k].site_key",
            layer_label=layer.layer_label,
        )
    return next(iter(keys))


def layer_site_peers(layer: Layer) -> tuple[Layer, ...]:
    """Return Layers sharing ANY of this layer's site keys within the trace.

    The index is computed live from the trace's layers on every call
    (recomputable, never persisted). Ops without keys never join the index,
    so a legacy layer refuses instead of matching everything.

    Raises
    ------
    InvalidArgumentError
        ``site_key_unavailable`` when this layer has no valid site key or
        the owning trace is unavailable.
    """

    own_keys = {op.site_key for op in _pass_ordered_ops(layer)}
    own_keys.discard(None)
    trace = getattr(layer, "source_trace", None)
    if not own_keys or trace is None:
        raise InvalidArgumentError(
            f"Layer '{layer.layer_label}' has no valid site key to index "
            "peers by: either the layer is detached from its trace, or this "
            "artifact predates the site_key_v1 grouping surface (site keys "
            "are minted at capture time; their persisted row is "
            "pre-release-gated under tlspec v7, so loaded artifacts read "
            "keyless).",
            code="site_key_unavailable",
            remedy=(
                "re-capture the model with a current TorchLens and read "
                "site_peers on the live trace"
            ),
            layer_label=layer.layer_label,
        )
    peers: list[Layer] = []
    for candidate_label in trace.layer_labels:
        candidate = trace.layer_logs.get(candidate_label)
        if candidate is None or candidate is layer:
            continue
        candidate_keys = {op.site_key for op in candidate.ops.values()}
        candidate_keys.discard(None)
        if candidate_keys & own_keys:
            peers.append(candidate)
    return tuple(peers)


def layer_shape_summary(layer: Layer) -> str | None:
    """Return the across-pass output-shape summary string, or ``None``.

    ``None`` for single-pass layers and for layers whose passes share one
    shape (or carry no usable shapes). One varying axis at uniform rank
    renders that axis alone: ``"A->B"`` when the axis sequence is monotone
    across passes, ``"A-B"`` (min-max) otherwise. Multi-axis or
    rank-varying layers render first-to-last full shapes,
    ``"2x64x8x8->2x512x4x4"``. Plain data, no markup; the character class
    is pinned (:data:`SHAPE_SUMMARY_CHARACTER_CLASS`).
    """

    ops = _pass_ordered_ops(layer)
    if len(ops) <= 1:
        return None
    shapes: list[tuple[int, ...]] = []
    for op in ops:
        shape = getattr(op, "shape", None)
        if shape is None:
            return None
        shapes.append(tuple(int(dim) for dim in shape))
    if len(set(shapes)) <= 1:
        return None
    ranks = {len(shape) for shape in shapes}
    if len(ranks) == 1:
        varying_axes = [
            axis for axis in range(next(iter(ranks))) if len({shape[axis] for shape in shapes}) > 1
        ]
        if len(varying_axes) == 1:
            axis_values = [shape[varying_axes[0]] for shape in shapes]
            pairs = list(zip(axis_values, axis_values[1:], strict=False))
            ascending = all(a <= b for a, b in pairs)
            descending = all(a >= b for a, b in pairs)
            if ascending or descending:
                return f"{axis_values[0]}->{axis_values[-1]}"
            return f"{min(axis_values)}-{max(axis_values)}"
    first = "x".join(str(dim) for dim in shapes[0])
    last = "x".join(str(dim) for dim in shapes[-1])
    return f"{first}->{last}"
