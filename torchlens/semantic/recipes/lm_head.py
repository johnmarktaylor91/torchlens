"""Built-in semantic recipes for language-model unembedding heads.

The ``language_model_head`` recipe anchors every architecture-specific fact a
logit-lens style projection needs -- which child module is the unembedding
head, which normalization module feeds it, and that norm's kind/parameters --
so downstream appliances (``torchlens.semantic.logit_lens``) stay generic math
over these facets. New architectures extend coverage by registering another
recipe that produces the same facet names; the appliance never changes.

The final norm is anchored by DATAFLOW through the hop rule
(``torchlens.semantic._hops``): the head's captured input op is walked
upstream across value-transporting hops (views, casts, recorded index
subsets such as the transformers-5.x ``logits_to_keep`` slice) to the op a
classified norm module produced, and the walk's payload-identity
verification gates every tensor-bearing facet. The recorded index subset is
disclosed as the ``logits_position_map`` facet -- an explicit position
mapping between the logits and the captured sequence -- never hopped
silently.
"""

from __future__ import annotations

from typing import Any

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

_LM_HEAD_FACETS = (
    "logits",
    "unembed_weight",
    "unembed_bias",
    "final_norm_kind",
    "final_norm_eps",
    "final_norm_gamma",
    "final_norm_beta",
    "final_norm_input",
    "logits_position_map",
)

#: Facets that hand out tensors under the final-norm anchor claim. They are
#: gated on the hop walk's payload-identity verification (hop rule: identity
#: is MANDATORY before any tensor is returned through the walk); the
#: string/scalar classification facets ride the proposer alone.
_NORM_TENSOR_FACETS = ("final_norm_gamma", "final_norm_beta", "final_norm_input")

#: Conventional direct-child names for the unembedding projection. A model
#: whose head is named differently is covered by registering a user recipe
#: producing the same facet names, never by widening this list speculatively.
_HEAD_CHILD_NAMES = ("lm_head", "embed_out", "output_projection")

_LAYER_NORM_CLASS_NAMES = frozenset({"LayerNorm"})


def _norm_kind_for_class(class_name: str) -> str | None:
    """Classify a module class name into a closed normalization-kind vocabulary.

    The ``*RMSNorm`` suffix match is deliberately broad (``LlamaRMSNorm``,
    ``Qwen2RMSNorm``, ...) and is safe ONLY because every consumer of
    ``final_norm_kind`` must numerically validate its reconstruction against
    the captured norm-input -> logits pair before trusting it (the
    ``logit_lens`` appliance refuses on mismatch). A nonstandard family member
    (e.g. a ``(1 + weight)``-scaled RMSNorm) therefore fails loudly downstream
    instead of being silently mislabelled here.

    Parameters
    ----------
    class_name:
        Module class name.

    Returns
    -------
    str | None
        ``"layer_norm"``, ``"rms_norm"``, or ``None`` when unclassified.
    """

    if class_name in _LAYER_NORM_CLASS_NAMES:
        return "layer_norm"
    if class_name == "RMSNorm" or class_name.endswith("RMSNorm"):
        return "rms_norm"
    return None


def _head_child_name(module: Any) -> str | None:
    """Return the conventional unembedding-head child name when present.

    Parameters
    ----------
    module:
        Candidate module record.

    Returns
    -------
    str | None
        Matched local child name, or ``None`` when no head child exists.
    """

    for child in getattr(module, "address_children", ()) or ():
        local = str(child).rsplit(".", maxsplit=1)[-1]
        if local in _HEAD_CHILD_NAMES:
            return local
    return None


def _has_lm_head(module: Any) -> bool:
    """Return whether a module record has a conventional unembedding child."""

    return _head_child_name(module) is not None


def _strip_pass(label: str) -> str:
    """Return a pass-qualified module label with its ``:<pass>`` suffix removed."""

    base, sep, tail = label.rpartition(":")
    if sep and tail.isdigit():
        return base
    return label


def _innermost_norm_record(trace: Any, op: Any) -> Any | None:
    """Return the classified norm module that PRODUCED ``op``, if any.

    Only the op's innermost containing module counts: the anchor claim is
    "this value is the norm's output", so a norm merely higher up the call
    stack must not qualify.

    Parameters
    ----------
    trace:
        Trace holding the module records.
    op:
        Candidate anchor op.

    Returns
    -------
    Any | None
        Norm module record, or ``None`` when the op was not produced inside a
        classified norm module.
    """

    stack = tuple(getattr(op, "modules", ()) or ())
    if not stack:
        return None
    address = _strip_pass(str(stack[-1]))
    try:
        record = trace.modules[address]
    except (KeyError, ValueError):
        return None
    if _norm_kind_for_class(str(getattr(record, "class_name", ""))) is None:
        return None
    return record


def _final_norm_walk(head: Any) -> tuple[Any, HopWalk] | AbsenceReason:
    """Anchor the norm module feeding the head's input, by dataflow.

    The head's captured input op is walked upstream across hop-rule proposals
    until an op produced inside a classified norm module is reached. The old
    implementation inspected only the immediate input op's module stack, and
    on every current HF causal LM the norm sits hops upstream (a reshape plus
    the ``logits_to_keep`` index subset), so no norm was ever found.

    Parameters
    ----------
    head:
        Unembedding-head module record.

    Returns
    -------
    tuple[Any, HopWalk] | AbsenceReason
        The norm module record and the completed walk, or a typed absence.
    """

    trace = getattr(head, "trace", None)
    if trace is None:
        return structural("module trace is unavailable")
    try:
        call = head._single_call_or_error()
        input_ops = list(getattr(call, "input_ops", ()) or ())
        if not input_ops:
            return structural("unembedding head has no dataflow input op")
        start = trace.ops[input_ops[0]]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return structural("unembedding head input op is unavailable")

    def _is_norm_output(candidate: Any) -> bool:
        """Anchor test: the candidate was produced inside a classified norm."""

        return _innermost_norm_record(trace, candidate) is not None

    walk = walk_upstream(trace, start, _is_norm_output)
    if isinstance(walk, HopRefusal):
        return structural(
            "no classified normalization module feeds the unembedding head input "
            f"by dataflow: {walk.reason}" + (f" (at {walk.at_label!r})" if walk.at_label else "")
        )
    norm = _innermost_norm_record(trace, walk.anchor)
    if norm is None:
        return structural("final-norm anchor op lost its module record")
    return norm, walk


def _norm_input_spec(norm: Any) -> FacetSpec | AbsenceReason:
    """Return the op-anchored value ENTERING the norm module, pass-exactly.

    The module-call record's ``input_ops`` may hold a BARE label of a
    multi-pass op (the final norm consumes the last block's residual add,
    whose label serves every pass), and a bare lookup refuses as ambiguous --
    so this derivation uses OP-LEVEL parent edges, which are pass-qualified:
    the norm's input is the unique op OUTSIDE the norm that its interior ops
    consume. More than one distinct external parent refuses rather than
    guesses (mikit D4: pass-qualified identity, never a last-pass default).

    Parameters
    ----------
    norm:
        Norm module record.

    Returns
    -------
    FacetSpec | AbsenceReason
        Op-anchored input spec, or a typed absence reason.
    """

    trace = getattr(norm, "trace", None)
    if trace is None:
        return structural("norm module trace is unavailable")
    externals = _external_parents(trace, norm)
    if isinstance(externals, AbsenceReason):
        return externals
    if len(externals) != 1:
        return structural(
            f"norm input is ambiguous: {len(externals)} distinct external ops feed the "
            "norm interior"
        )
    op = next(iter(externals.values()))
    if not op_output_readable(op):
        return needs_capture(
            f"norm input op {getattr(op, 'label', '<unknown>')!r} was not saved",
            f"save=... including {getattr(op, 'label', 'the norm input')!r}",
        )
    return FacetSpec.from_home(op, home_kind="op", recipe_id="language_model_head")


def _external_parents(trace: Any, norm: Any) -> dict[str, Any] | AbsenceReason:
    """Return the ops OUTSIDE the norm that its interior ops consume."""

    try:
        interior_labels = list(norm._op_labels())
    except (AttributeError, TypeError, ValueError):
        interior_labels = list(getattr(norm, "output_ops", ()) or ())
    interior: set[str] = set()
    records: list[Any] = []
    for label in interior_labels:
        try:
            op = trace.ops[label]
        except (KeyError, ValueError):
            continue
        interior.add(str(op.label))
        records.append(op)
    externals: dict[str, Any] = {}
    for op in records:
        for parent_label in getattr(op, "parents", ()) or ():
            try:
                parent = trace.ops[str(parent_label)]
            except (KeyError, ValueError):
                return structural(
                    f"norm interior parent {parent_label!r} did not resolve to one op; "
                    "the norm input would be a guess"
                )
            canonical = str(parent.label)
            if canonical not in interior:
                externals[canonical] = parent
    return externals


@register(predicate=_has_lm_head, target_scope="module", facets=_LM_HEAD_FACETS)
def language_model_head(module: Any) -> dict[str, Any]:
    """Return unembedding-head facets for language-model modules.

    Parameters
    ----------
    module:
        TorchLens module record with a conventional unembedding child.

    Returns
    -------
    dict[str, Any]
        Head and final-norm facets anchored to captured records.
    """

    result: dict[str, Any] = {}
    head_name = _head_child_name(module)
    head = child_module(module, head_name) if head_name is not None else None
    if head_name is None or head is None:
        absent = structural(f"unembedding child {head_name!r} record is unavailable")
        for name in _LM_HEAD_FACETS:
            add_if_present(result, name, absent)
        return result
    add_if_present(result, "logits", child_output_spec(module, head_name, "language_model_head"))
    add_if_present(result, "unembed_weight", parameter_spec(head, "weight", "language_model_head"))
    add_if_present(result, "unembed_bias", parameter_spec(head, "bias", "language_model_head"))
    anchored = _final_norm_walk(head)
    if isinstance(anchored, AbsenceReason):
        for name in _LM_HEAD_FACETS[3:]:
            add_if_present(result, name, anchored)
        return result
    norm, walk = anchored
    kind = _norm_kind_for_class(str(getattr(norm, "class_name", "")))
    add_if_present(result, "final_norm_kind", kind)
    add_if_present(result, "logits_position_map", walk.index_map)
    if walk.verification == "payload_identity":
        add_if_present(result, "final_norm_input", _norm_input_spec(norm))
        add_if_present(
            result, "final_norm_gamma", parameter_spec(norm, "weight", "language_model_head")
        )
        beta: Any
        if kind == "rms_norm":
            beta = structural("RMSNorm has no beta/bias parameter")
        else:
            beta = parameter_spec(norm, "bias", "language_model_head")
        add_if_present(result, "final_norm_beta", beta)
    else:
        unverified = needs_capture(
            "final-norm anchor is proposed by dataflow but UNVERIFIED: the walk's "
            "endpoint payloads were not saved, and tensor-bearing norm facets are "
            "only handed out over a payload-verified anchor (hop rule)",
            "save=... including the unembedding head input and the norm output",
        )
        for name in _NORM_TENSOR_FACETS:
            if name == "final_norm_beta" and kind == "rms_norm":
                add_if_present(result, name, structural("RMSNorm has no beta/bias parameter"))
                continue
            add_if_present(result, name, unverified)
    eps = config_value(norm, "eps", "variance_epsilon")
    if isinstance(eps, AbsenceReason):
        eps = structural("normalization epsilon metadata is absent")
    add_if_present(result, "final_norm_eps", eps)
    return result
