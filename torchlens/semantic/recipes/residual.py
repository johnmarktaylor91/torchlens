"""Built-in semantic recipes for transformer residual stream facets."""

from __future__ import annotations

from typing import Any

from .._hops import BRANCH_GRAMMAR, HopWalk, walk_upstream
from ..facets import AbsenceReason, FacetSpec, register
from ._helpers import (
    add_if_present,
    module_output_spec,
    needs_capture,
    op_output_readable,
    structural,
)

_RESIDUAL_FACETS = ("resid_pre", "resid_mid", "resid_post")
_ATTENTION_CHILD_NAMES = ("attn", "attention", "self_attn", "self_attention")
_MLP_CHILD_NAMES = ("mlp", "feed_forward", "ffn", "intermediate", "output")
#: Unfused feed-forward evidence: blocks (OPT's ``OPTDecoderLayer``, and its
#: relatives) that hold the two MLP projections as DIRECT children instead of
#: wrapping them in an ``mlp`` submodule. Both must be present.
_MLP_SPLIT_CHILD_NAMES = ("fc1", "fc2")


def _is_transformer_block(module: Any) -> bool:
    """Return whether a module record is a genuine transformer block.

    A block qualifies only on STRUCTURAL evidence: it must contain both an
    attention child and feed-forward evidence -- either an MLP/feed-forward
    child module, or the unfused ``fc1`` + ``fc2`` direct-projection pair
    (the OPT family holds its MLP as two direct Linear children, so requiring
    a wrapped MLP child silently produced NO residual facets there). A
    class-name marker alone (``*Block*`` / ``*Layer*``) is neither necessary
    nor sufficient -- plain non-transformer modules routinely carry those
    names (a scaling ``*Layer*``, a conv ``*Block*``), so matching on the
    name fabricated resid_pre/mid/post facets on modules that have no
    residual stream at all.

    Parameters
    ----------
    module:
        Candidate module record.

    Returns
    -------
    bool
        Whether the recipe should attempt residual stream facets.
    """

    children = set(getattr(module, "address_children", ()) or ())
    local_children = {str(child).rsplit(".", maxsplit=1)[-1] for child in children}
    has_attention = any(name in local_children for name in _ATTENTION_CHILD_NAMES)
    has_mlp = any(name in local_children for name in _MLP_CHILD_NAMES) or all(
        name in local_children for name in _MLP_SPLIT_CHILD_NAMES
    )
    return has_attention and has_mlp


@register(predicate=_is_transformer_block, target_scope="module", facets=_RESIDUAL_FACETS)
def transformer_residuals(module: Any) -> dict[str, Any]:
    """Return residual stream facets for transformer-like block modules.

    Parameters
    ----------
    module:
        TorchLens module record.

    Returns
    -------
    dict[str, Any]
        Residual facets anchored to captured ops where available.
    """

    result: dict[str, Any] = {}
    add_if_present(result, "resid_pre", _resid_pre_spec(module))
    add_if_present(result, "resid_mid", _resid_mid_spec(module))
    add_if_present(result, "resid_post", module_output_spec(module, "transformer_residuals"))
    return result


def _resid_pre_spec(module: Any) -> FacetSpec | AbsenceReason:
    """Return the block's residual-stream input, selected by dataflow + shape.

    The previous implementation took ``call.input_ops[0]`` blind, and on real
    HF blocks that op is routinely the token ids or the attention mask (both
    enter every block alongside the hidden states), so ``resid_pre`` returned
    an ids/mask-shaped tensor with NO refusal -- a silent wrong read. The
    residual stream input is now REQUIRED to earn its anchor:

    - SHAPE: its recorded shape equals the block's single output shape (the
      residual stream is shape-preserved through a transformer block) and its
      dtype is floating.
    - DATAFLOW: it is an upstream ancestor of the block's output op.

    Exactly one input op may satisfy both; zero or several refuse with a
    typed absence, never a guess.

    Parameters
    ----------
    module:
        TorchLens module record.

    Returns
    -------
    FacetSpec | AbsenceReason
        Op-anchored residual input spec, or a typed absence reason.
    """

    trace = getattr(module, "trace", None)
    if trace is None:
        return structural("module trace is unavailable")
    anchored = _block_shape_anchor(trace, module)
    if isinstance(anchored, AbsenceReason):
        return anchored
    input_ops, out_op, out_shape = anchored
    candidates, unresolved = _residual_candidates(trace, module, input_ops, out_op, out_shape)
    if not candidates:
        detail = (
            "no block input matches the residual stream by dataflow + shape: no "
            f"floating input op with the block's output shape {out_shape} feeds the "
            "block output. resid_pre is never guessed from input order."
        )
        if unresolved:
            detail += f" (unresolvable input labels skipped: {tuple(unresolved)})"
        return structural(detail)
    if len(candidates) > 1:
        labels = tuple(str(getattr(op, "label", "?")) for op in candidates)
        return structural(
            f"residual stream input is ambiguous: block inputs {labels} all match the "
            f"output shape {out_shape} and feed the block output. resid_pre refuses "
            "rather than guess; read the candidate ops directly."
        )
    op = candidates[0]
    if not op_output_readable(op):
        return needs_capture(
            f"residual input op {getattr(op, 'label', '<unknown>')!r} was not saved",
            f"save=... including {getattr(op, 'label', 'the residual input')!r}",
        )
    return FacetSpec.from_home(op, home_kind="op", recipe_id="transformer_residuals")


def _block_shape_anchor(
    trace: Any, module: Any
) -> tuple[list[str], Any, tuple[int, ...]] | AbsenceReason:
    """Return (input labels, single output op, output shape) or a typed absence."""

    try:
        call = module._single_call_or_error()
        input_ops = list(getattr(call, "input_ops", ()) or ())
        output_ops = list(getattr(call, "output_ops", ()) or ())
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return structural("module call records are unavailable")
    if not input_ops:
        return structural("module has no dataflow input op")
    if len(output_ops) != 1:
        return structural("module has ambiguous outputs; the residual shape anchor needs one")
    try:
        out_op = trace.ops[output_ops[0]]
    except (KeyError, ValueError):
        return structural("module output op is unavailable")
    out_shape = _shape_of(out_op)
    if out_shape is None:
        return structural("module output shape is unavailable")
    return input_ops, out_op, out_shape


def _residual_candidates(
    trace: Any, module: Any, input_ops: list[str], out_op: Any, out_shape: tuple[int, ...]
) -> tuple[list[Any], list[str]]:
    """Return (matching input ops, unresolvable labels) for the resid_pre anchor."""

    member_labels = _member_op_labels(trace, module)
    candidates: list[Any] = []
    unresolved: list[str] = []
    for label in input_ops:
        op = _resolve_input_op(trace, label, member_labels)
        if op is None:
            unresolved.append(str(label))
            continue
        if _shape_of(op) != out_shape or not _is_floating(op):
            continue
        if not _feeds(trace, op, out_op):
            continue
        candidates.append(op)
    return candidates, unresolved


def _member_op_labels(trace: Any, module: Any) -> set[str]:
    """Return canonical (pass-qualified) labels of the ops inside a module."""

    labels: set[str] = set()
    for label in _module_op_labels(module):
        try:
            labels.add(str(trace.ops[label].label))
        except (KeyError, TypeError, ValueError):
            continue
    return labels


def _resolve_input_op(trace: Any, label: Any, member_labels: set[str]) -> Any | None:
    """Resolve a module-call input label to ONE op record, pass-exactly.

    Call-level ``input_ops`` may hold the BARE label of a multi-pass op (on
    real GPT-2, every deep block's residual input is the previous block's
    output add, whose label serves both of that block's adds), and a bare
    lookup refuses as ambiguous. The pass is disambiguated by CONSUMPTION:
    the input op is the pass with at least one child op inside this module
    call. Zero or several qualifying passes stay unresolved -- never a
    last-pass default (mikit D4).
    """

    try:
        return trace.ops[label]
    except KeyError:
        return None
    except ValueError:
        pass
    hits = [
        op for op in _layer_passes(trace, str(label)) if _has_child_in(trace, op, member_labels)
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _layer_passes(trace: Any, label: str) -> list[Any]:
    """Return every pass-qualified op of a multi-pass layer, or ``[]``."""

    try:
        num_passes = int(getattr(trace[label], "num_passes", 0) or 0)
    except (KeyError, TypeError, ValueError):
        return []
    passes = []
    for index in range(1, num_passes + 1):
        try:
            passes.append(trace.ops[f"{label}:{index}"])
        except (KeyError, ValueError):
            return []
    return passes


def _has_child_in(trace: Any, op: Any, member_labels: set[str]) -> bool:
    """Return whether one of ``op``'s children lies inside the member set."""

    for child_label in getattr(op, "children", ()) or ():
        try:
            canonical = str(trace.ops[str(child_label)].label)
        except (KeyError, ValueError):
            continue
        if canonical in member_labels:
            return True
    return False


def _shape_of(op: Any) -> tuple[int, ...] | None:
    """Return an op's recorded output shape as a plain tuple."""

    shape = getattr(op, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _is_floating(op: Any) -> bool:
    """Return whether an op's recorded dtype is floating point."""

    dtype = getattr(op, "dtype", None)
    return bool(getattr(dtype, "is_floating_point", False))


def _feeds(trace: Any, source: Any, sink: Any, max_visits: int = 2000) -> bool:
    """Return whether ``source`` is a dataflow ancestor of ``sink``.

    Bounded upstream breadth-first search over recorded parent edges, with
    labels canonicalized through the trace so bare and pass-qualified
    spellings compare equal. Fails closed at the visit bound.
    """

    target = str(getattr(source, "label", ""))
    queue = [sink]
    seen: set[str] = set()
    while queue and len(seen) < max_visits:
        current = queue.pop()
        label = str(getattr(current, "label", ""))
        if label in seen:
            continue
        seen.add(label)
        if label == target:
            return True
        for parent_label in getattr(current, "parents", ()) or ():
            try:
                queue.append(trace.ops[str(parent_label)])
            except (KeyError, ValueError):
                continue
    return False


def _resid_mid_spec(module: Any) -> FacetSpec | AbsenceReason | None:
    """Return a spec for the post-attention residual add inside a block.

    The residual midpoint is only well defined when an add op genuinely
    consumes the block's attention branch. Direct parenthood is not required:
    real blocks route the attention output through value-transporting ops
    before the add (OPT applies dropout between ``self_attn`` and the
    residual add), so the branch is identified through the hop rule's
    BRANCH grammar -- a bounded structural walk, since the tensor returned is
    the add op's own captured output and never crosses the hops. The block's
    own output add is excluded: on single-add blocks that add is the block
    output, so a midpoint claim would be degenerate. Absence (``None``)
    is returned rather than a fabricated midpoint.

    Parameters
    ----------
    module:
        TorchLens module record.

    Returns
    -------
    FacetSpec | AbsenceReason | None
        Op-anchored spec for the real post-attention add, a needs-capture
        reason when that add was not saved, or ``None`` when no such add exists.
    """

    trace = getattr(module, "trace", None)
    if trace is None:
        return None
    attention_outputs = _attention_output_labels(module)
    if not attention_outputs:
        return None
    try:
        output_ops = {
            str(trace.ops[label].label)
            for label in getattr(module._single_call_or_error(), "output_ops", ()) or ()
        }
    except (AttributeError, KeyError, RuntimeError, ValueError):
        output_ops = set()
    for label in _module_op_labels(module):
        try:
            op = trace.ops[label]
        except (KeyError, TypeError, ValueError):
            continue
        if str(getattr(op, "func_name", "")) not in {"add", "__add__", "add_"}:
            continue
        if str(getattr(op, "label", "")) in output_ops:
            continue
        if not _consumes_attention_branch(trace, op, attention_outputs):
            continue
        if not op_output_readable(op):
            return needs_capture(
                f"residual midpoint op {getattr(op, 'label', '<unknown>')!r} was not saved",
                f"save=... including {getattr(op, 'label', 'the residual midpoint')!r}",
            )
        return FacetSpec.from_home(op, home_kind="op", recipe_id="transformer_residuals")
    return None


def _consumes_attention_branch(trace: Any, add_op: Any, attention_outputs: set[str]) -> bool:
    """Return whether one of ``add_op``'s parents carries the attention output.

    A parent matches directly, or through a short upstream walk across
    BRANCH-grammar hops (dropout/cast/view) ending at an attention output op.
    """

    def _is_attention_output(candidate: Any) -> bool:
        """Anchor test: the candidate is one of the attention child's outputs."""

        return str(getattr(candidate, "label", "")) in attention_outputs

    for parent_label in getattr(add_op, "parents", ()) or ():
        try:
            parent = trace.ops[str(parent_label)]
        except (KeyError, ValueError):
            continue
        walk = walk_upstream(
            trace,
            parent,
            _is_attention_output,
            grammar=BRANCH_GRAMMAR,
            structure_only=True,
            max_hops=4,
        )
        if isinstance(walk, HopWalk):
            return True
    return False


def _module_op_labels(module: Any) -> list[str]:
    """Return op labels contained by a module record."""

    try:
        return list(module._op_labels())
    except (AttributeError, TypeError, ValueError):
        return list(getattr(module, "output_ops", ()) or ())


def _attention_output_labels(module: Any) -> set[str]:
    """Return canonical output op labels for direct attention children."""

    trace = getattr(module, "trace", None)
    if trace is None:
        return set()
    labels: set[str] = set()
    for child_address in getattr(module, "address_children", ()) or ():
        local_name = str(child_address).rsplit(".", maxsplit=1)[-1]
        if local_name not in _ATTENTION_CHILD_NAMES:
            continue
        try:
            child = trace.modules[child_address]
        except (KeyError, ValueError):
            continue
        for label in getattr(child, "output_ops", ()) or ():
            try:
                labels.add(str(trace.ops[str(label)].label))
            except (KeyError, ValueError):
                continue
    return labels
