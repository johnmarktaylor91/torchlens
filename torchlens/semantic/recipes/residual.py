"""Built-in semantic recipes for transformer residual stream facets."""

from __future__ import annotations

from typing import Any

from ..facets import AbsenceReason, FacetSpec, register
from ._helpers import (
    add_if_present,
    module_input_op_spec,
    module_output_spec,
    needs_capture,
    op_output_readable,
)

_RESIDUAL_FACETS = ("resid_pre", "resid_mid", "resid_post")
_BLOCK_NAME_MARKERS = (
    "Block",
    "Layer",
    "DecoderLayer",
    "EncoderLayer",
    "TransformerLayer",
)
_ATTENTION_CHILD_NAMES = ("attn", "attention", "self_attn", "self_attention")
_MLP_CHILD_NAMES = ("mlp", "feed_forward", "ffn", "intermediate", "output")


def _is_transformer_block(module: Any) -> bool:
    """Return whether a module record looks like a transformer block.

    Parameters
    ----------
    module:
        Candidate module record.

    Returns
    -------
    bool
        Whether the recipe should attempt residual stream facets.
    """

    class_name = str(getattr(module, "class_name", ""))
    if any(marker in class_name for marker in _BLOCK_NAME_MARKERS):
        return True
    children = set(getattr(module, "address_children", ()) or ())
    local_children = {str(child).rsplit(".", maxsplit=1)[-1] for child in children}
    has_attention = any(name in local_children for name in _ATTENTION_CHILD_NAMES)
    has_mlp = any(name in local_children for name in _MLP_CHILD_NAMES)
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
    add_if_present(result, "resid_pre", module_input_op_spec(module, "transformer_residuals"))
    add_if_present(result, "resid_mid", _resid_mid_spec(module))
    add_if_present(result, "resid_post", module_output_spec(module, "transformer_residuals"))
    return result


def _resid_mid_spec(module: Any) -> FacetSpec | AbsenceReason | None:
    """Return a spec for the post-attention residual add inside a block.

    Parameters
    ----------
    module:
        TorchLens module record.

    Returns
    -------
    FacetSpec | None
        Op-anchored spec for a real add op, when identifiable.
    """

    trace = getattr(module, "trace", None)
    if trace is None:
        return None
    labels = _module_op_labels(module)
    attention_outputs = _attention_output_labels(module)
    add_candidates: list[Any] = []
    for label in labels:
        try:
            op = trace.ops[label]
        except (KeyError, TypeError):
            continue
        if str(getattr(op, "func_name", "")) in {"add", "__add__", "add_"}:
            add_candidates.append(op)
    for op in add_candidates:
        parents = set(getattr(op, "parents", ()) or ())
        if parents.intersection(attention_outputs):
            if not op_output_readable(op):
                return needs_capture(
                    f"residual midpoint op {getattr(op, 'label', '<unknown>')!r} was not saved",
                    f"save=... including {getattr(op, 'label', 'the residual midpoint')!r}",
                )
            return FacetSpec.from_home(op, home_kind="op", recipe_id="transformer_residuals")
    if add_candidates:
        if not op_output_readable(add_candidates[0]):
            return needs_capture(
                f"residual midpoint op {getattr(add_candidates[0], 'label', '<unknown>')!r} "
                "was not saved",
                f"save=... including {getattr(add_candidates[0], 'label', 'the residual midpoint')!r}",
            )
        return FacetSpec.from_home(
            add_candidates[0], home_kind="op", recipe_id="transformer_residuals"
        )
    return None


def _module_op_labels(module: Any) -> list[str]:
    """Return op labels contained by a module record."""

    try:
        return list(module._op_labels())
    except (AttributeError, TypeError, ValueError):
        return list(getattr(module, "output_ops", ()) or ())


def _attention_output_labels(module: Any) -> set[str]:
    """Return output op labels for direct attention children."""

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
        labels.update(str(label) for label in getattr(child, "output_ops", ()) or ())
    return labels
