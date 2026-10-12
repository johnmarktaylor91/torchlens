"""Built-in semantic recipes for vision-model output heads.

Two head shapes give a vision trace a default logit lens when no
language-model head exists:

- ``classification_head``: a final norm plus a ``classifier`` linear layer
  reading ONE token of the last block's sequence (ViT's ``[CLS]`` read,
  ``sequence_output[:, 0, :]``).
- ``contrastive_image_head``: CLIP's image path into its image-text
  similarity: the ``[CLS]`` read, ``post_layernorm``, ``visual_projection``,
  L2 normalization, and the scaled cosine against the captured text
  features of the prompts the capture ran (``logits_per_image``).

Both recipes produce the final-norm facet vocabulary of
``language_model_head`` plus the head's own projection facets, so the
``logit_lens`` appliance stays generic math over facets. The single token
read is a dimension-dropping subscript the shared hop walker does not fold
into a position map; these recipes parse exactly that one grammar (batch
kept, one integer sequence position, the feature axis kept), verify it by
payload identity, and disclose the position as ``pooled_position``. Any
other pooling (mean pooling, attention pooling, a pooler MLP) leaves the
norm unanchored, and the lens refuses instead of guessing.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .._hops import HopRefusal, HopWalk, walk_upstream
from ..facets import AbsenceReason, FacetSpec, register
from ._helpers import (
    add_if_present,
    child_module,
    child_output_spec,
    config_value,
    needs_capture,
    op_output_readable,
    parameter_spec,
    structural,
)
from .lm_head import (
    _external_parents,
    _has_lm_head,
    _innermost_norm_record,
    _norm_input_spec,
    _norm_kind_for_class,
)

_FINAL_NORM_FACETS = (
    "final_norm_kind",
    "final_norm_eps",
    "final_norm_gamma",
    "final_norm_beta",
    "final_norm_input",
)

_CLASSIFIER_HEAD_FACETS = (
    "logits",
    "classifier_weight",
    "classifier_bias",
    "pooled_position",
    "tower_address",
    *_FINAL_NORM_FACETS,
)

_CONTRASTIVE_HEAD_FACETS = (
    "logits",
    "image_projection_weight",
    "text_features",
    "logit_scale",
    "pooled_position",
    "tower_address",
    *_FINAL_NORM_FACETS,
)

#: Subscript ops that can realize a single-token read.
_INDEX_FUNCS = frozenset({"__getitem__", "getitem"})

#: The classifier child name ViT-style ``*ForImageClassification`` heads use.
_CLASSIFIER_CHILD = "classifier"


@dataclass(frozen=True)
class _PooledNorm:
    """A head's final norm, anchored by dataflow, plus its one-token read."""

    norm: Any
    position: int
    verified: bool


def _full_slice(entry: Any) -> bool:
    """Return whether a subscript entry is the untouched ``:`` slice."""

    return isinstance(entry, slice) and entry == slice(None, None, None)


def _single_token_position(op: Any) -> int | None:
    """Return the sequence position a ``[:, p]`` / ``[:, p, :]`` read selects.

    Only the exact grammar counts: batch axis kept whole, one non-negative
    integer sequence position, the feature axis kept whole. Anything else
    (slices, negative indices, tensor indices) returns ``None``.
    """

    if str(getattr(op, "func_name", "")) not in _INDEX_FUNCS:
        return None
    pos = list(getattr(op, "non_tensor_pos_args", ()) or ())
    if len(pos) != 1 or not isinstance(pos[0], tuple):
        return None
    entries = pos[0]
    if len(entries) not in (2, 3) or not _full_slice(entries[0]):
        return None
    if len(entries) == 3 and not _full_slice(entries[2]):
        return None
    position = entries[1]
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        return None
    return position


def _payload(op: Any) -> torch.Tensor | None:
    """Return an op's saved output tensor, or ``None`` when unreadable."""

    if not op_output_readable(op):
        return None
    value = getattr(op, "out", None)
    return value if isinstance(value, torch.Tensor) else None


def _read_is_verified(trace: Any, read_op: Any, position: int) -> bool:
    """Replay the one-token read on its parent payload and compare bitwise."""

    parents = tuple(getattr(read_op, "parents", ()) or ())
    if len(parents) != 1:
        return False
    try:
        parent = trace.ops[str(parents[0])]
    except (KeyError, ValueError):
        return False
    source = _payload(parent)
    result = _payload(read_op)
    if source is None or result is None or source.ndim != 3:
        return False
    replayed = source[:, position, :]
    return replayed.shape == result.shape and torch.equal(replayed, result)


def _input_op(module: Any) -> Any | AbsenceReason:
    """Return a module's single dataflow input op record."""

    trace = getattr(module, "trace", None)
    if trace is None:
        return structural("module trace is unavailable")
    try:
        call = module._single_call_or_error()
        input_ops = list(getattr(call, "input_ops", ()) or ())
        if not input_ops:
            return structural("head has no dataflow input op")
        return trace.ops[input_ops[0]]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return structural("head input op is unavailable")


def _walk_to_norm(trace: Any, start: Any) -> tuple[Any, HopWalk] | AbsenceReason:
    """Walk upstream from ``start`` to the op a classified norm produced."""

    def _is_norm_output(candidate: Any) -> bool:
        """Anchor test: the candidate was produced inside a classified norm."""

        return _innermost_norm_record(trace, candidate) is not None

    walk = walk_upstream(trace, start, _is_norm_output)
    if isinstance(walk, HopRefusal):
        return structural(
            "no classified normalization module feeds the head by dataflow: "
            f"{walk.reason}" + (f" (at {walk.at_label!r})" if walk.at_label else "")
        )
    norm = _innermost_norm_record(trace, walk.anchor)
    if norm is None:
        return structural("final-norm anchor op lost its module record")
    if walk.index_map.derivation != "identity":
        return structural("the walk to the head's final norm crosses a position subset")
    return norm, walk


def _norm_then_read(trace: Any, start: Any, position: int) -> _PooledNorm | AbsenceReason:
    """ViT order: the head reads one token of the norm's full-sequence output."""

    parents = tuple(getattr(start, "parents", ()) or ())
    if len(parents) != 1:
        return structural("the head's one-token read has no single tensor parent")
    try:
        parent = trace.ops[str(parents[0])]
    except (KeyError, ValueError):
        return structural("the head's one-token read parent is unavailable")
    anchored = _walk_to_norm(trace, parent)
    if isinstance(anchored, AbsenceReason):
        return anchored
    norm, walk = anchored
    verified = walk.verification == "payload_identity" and _read_is_verified(trace, start, position)
    return _PooledNorm(norm=norm, position=position, verified=verified)


def _read_then_norm(trace: Any, start: Any) -> _PooledNorm | AbsenceReason:
    """CLIP order: the norm normalizes the one token the tower's output read."""

    anchored = _walk_to_norm(trace, start)
    if isinstance(anchored, AbsenceReason):
        return anchored
    norm, walk = anchored
    externals = _external_parents(trace, norm)
    if isinstance(externals, AbsenceReason):
        return externals
    if len(externals) != 1:
        return structural(
            f"norm input is ambiguous: {len(externals)} distinct external ops feed the norm"
        )
    read_op = next(iter(externals.values()))
    position = _single_token_position(read_op)
    if position is None:
        return structural(
            "the head pools its input some other way than one token read "
            "(e.g. mean pooling or a pooler layer); the lens does not reconstruct it"
        )
    verified = walk.verification == "payload_identity" and _read_is_verified(
        trace, read_op, position
    )
    return _PooledNorm(norm=norm, position=position, verified=verified)


def _pooled_norm(head: Any) -> _PooledNorm | AbsenceReason:
    """Anchor a vision head's final norm and its one-token read by dataflow."""

    trace = getattr(head, "trace", None)
    if trace is None:
        return structural("module trace is unavailable")
    start = _input_op(head)
    if isinstance(start, AbsenceReason):
        return start
    position = _single_token_position(start)
    if position is not None:
        return _norm_then_read(trace, start, position)
    return _read_then_norm(trace, start)


def _tower_address(norm: Any) -> str:
    """Return the address of the module that owns the head's final norm."""

    address = str(getattr(norm, "address", ""))
    parent, sep, _ = address.rpartition(".")
    return parent if sep else "self"


def _add_norm_facets(result: dict[str, Any], pooled: _PooledNorm | AbsenceReason) -> None:
    """Attach the final-norm, read-position, and tower facets."""

    if isinstance(pooled, AbsenceReason):
        for name in ("pooled_position", "tower_address", *_FINAL_NORM_FACETS):
            add_if_present(result, name, pooled)
        return
    norm = pooled.norm
    kind = _norm_kind_for_class(str(getattr(norm, "class_name", "")))
    add_if_present(result, "final_norm_kind", kind)
    add_if_present(result, "pooled_position", pooled.position)
    add_if_present(result, "tower_address", _tower_address(norm))
    if pooled.verified:
        add_if_present(result, "final_norm_input", _norm_input_spec(norm))
        add_if_present(result, "final_norm_gamma", parameter_spec(norm, "weight", "vision_head"))
        beta: Any
        if kind == "rms_norm":
            beta = structural("RMSNorm has no beta/bias parameter")
        else:
            beta = parameter_spec(norm, "bias", "vision_head")
        add_if_present(result, "final_norm_beta", beta)
    else:
        unverified = needs_capture(
            "the head's final norm and token read are proposed by dataflow but "
            "UNVERIFIED: their payloads were not saved, and tensor-bearing norm facets "
            "are only handed out over a payload-verified anchor",
            "save=... including the head input and the final norm output",
        )
        for name in ("final_norm_input", "final_norm_gamma", "final_norm_beta"):
            add_if_present(result, name, unverified)
    eps = config_value(norm, "eps", "variance_epsilon")
    if isinstance(eps, AbsenceReason):
        eps = structural("normalization epsilon metadata is absent")
    add_if_present(result, "final_norm_eps", eps)


def _has_classifier_head(module: Any) -> bool:
    """Return whether a module owns a linear ``classifier`` child and no LM head."""

    if _has_lm_head(module):
        return False
    for child in getattr(module, "address_children", ()) or ():
        if str(child).rsplit(".", maxsplit=1)[-1] != _CLASSIFIER_CHILD:
            continue
        record = child_module(module, _CLASSIFIER_CHILD)
        return record is not None and str(getattr(record, "class_name", "")) == "Linear"
    return False


@register(predicate=_has_classifier_head, target_scope="module", facets=_CLASSIFIER_HEAD_FACETS)
def classification_head(module: Any) -> dict[str, Any]:
    """Return classifier-head facets for vision classification models.

    Parameters
    ----------
    module:
        TorchLens module record owning a linear ``classifier`` child.

    Returns
    -------
    dict[str, Any]
        Classifier and final-norm facets anchored to captured records.
    """

    result: dict[str, Any] = {}
    head = child_module(module, _CLASSIFIER_CHILD)
    if head is None:
        absent = structural("classifier child record is unavailable")
        for name in _CLASSIFIER_HEAD_FACETS:
            add_if_present(result, name, absent)
        return result
    add_if_present(result, "logits", child_output_spec(module, _CLASSIFIER_CHILD, "vision_head"))
    add_if_present(result, "classifier_weight", parameter_spec(head, "weight", "vision_head"))
    add_if_present(result, "classifier_bias", parameter_spec(head, "bias", "vision_head"))
    _add_norm_facets(result, _pooled_norm(head))
    return result


def _output_leaf_spec(module: Any, role: str) -> FacetSpec | AbsenceReason:
    """Return the op-anchored spec of one named leaf of a module's output."""

    try:
        call = module._single_call_or_error()
        matches = [
            label
            for label in call.output_ops
            if getattr(module.trace.ops[label], "multi_output_name", None) == role
        ]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return structural(f"module output leaf {role!r} is unavailable")
    if len(matches) != 1:
        return structural(f"module output leaf {role!r} did not resolve to exactly one op")
    op = module.trace.ops[matches[0]]
    if not op_output_readable(op):
        return needs_capture(
            f"module output leaf {role!r} op {getattr(op, 'label', '<unknown>')!r} was not saved",
            f"save=... including {getattr(op, 'label', role)!r}",
        )
    return FacetSpec.from_home(op, home_kind="op", recipe_id="vision_head")


@register(class_name="CLIPModel", target_scope="module", facets=_CONTRASTIVE_HEAD_FACETS)
def contrastive_image_head(module: Any) -> dict[str, Any]:
    """Return CLIP's image-side similarity facets.

    The text side is whatever prompts the capture ran: ``text_features`` is
    the captured ``text_projection`` output (one row per prompt, before L2
    normalization), and ``logits`` is the captured ``logits_per_image``.

    Parameters
    ----------
    module:
        TorchLens module record of a ``CLIPModel``.

    Returns
    -------
    dict[str, Any]
        Similarity-head and final-norm facets anchored to captured records.
    """

    result: dict[str, Any] = {}
    add_if_present(result, "logits", _output_leaf_spec(module, "logits_per_image"))
    add_if_present(
        result, "text_features", child_output_spec(module, "text_projection", "vision_head")
    )
    add_if_present(result, "logit_scale", parameter_spec(module, "logit_scale", "vision_head"))
    projection = child_module(module, "visual_projection")
    if projection is None:
        absent = structural("visual_projection child record is unavailable")
        for name in ("image_projection_weight", "pooled_position", "tower_address"):
            add_if_present(result, name, absent)
        for name in _FINAL_NORM_FACETS:
            add_if_present(result, name, absent)
        return result
    add_if_present(
        result, "image_projection_weight", parameter_spec(projection, "weight", "vision_head")
    )
    _add_norm_facets(result, _pooled_norm(projection))
    return result
