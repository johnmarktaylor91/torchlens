"""Transformer lens facts: attention subject, per-layer role rows, focus block.

Every fact here comes from the module tree's own names and classes plus op
types; nothing classifies the model as a whole.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..._errors import InvalidArgumentError
from ..._record_type import record_type

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

#: Attention role rows: module-address suffix -> per-layer fact wording.
#: These are per-LAYER facts (the module tree's own names plus op types),
#: never trace classification -- a hybrid gets rows exactly where the facts
#: are.
_ATTENTION_ROLE_SUFFIXES = {
    "q_proj": "attention role: query projection",
    "k_proj": "attention role: key projection",
    "v_proj": "attention role: value projection",
    "out_proj": "attention role: output projection",
    "o_proj": "attention role: output projection",
    "qkv": "attention role: fused q/k/v projection",
    # Hugging Face spellings: BERT/ViT-style self-attention and GPT-2's
    # fused Conv1D projection.
    "query": "attention role: query projection",
    "key": "attention role: key projection",
    "value": "attention role: value projection",
    "c_attn": "attention role: fused q/k/v projection",
}

#: Suffixes that name an attention role only inside an attention module
#: (GPT-2's ``attn.c_proj`` and BERT/ViT's ``attention.output.dense``; the
#: same names occur in the MLP).
_SCOPED_ATTENTION_ROLE_SUFFIXES = {
    "c_proj": ("attn", "attention role: output projection"),
    "dense": ("attention", "attention role: output projection"),
}

#: Op types that are themselves attention facts.
_ATTENTION_OP_TYPES = frozenset(
    {
        "scaled_dot_product_attention",
        "scaleddotproductattention",
        "sdpa",
        "multi_head_attention_forward",
        "multiheadattentionforward",
    }
)


def _module_facets_say_attention(trace: Trace) -> bool:
    """The facet leg of the attention-subject probe.

    ``Module.facets`` is single-call and refuses ``module_call_ambiguous``
    on any reused module; that per-module accessor limitation is never the
    TRACE-level fact and must not surface as the user's refusal
    -- the op scan in :func:`_has_attention_subject` still probes those
    modules' calls.
    """

    try:
        for module in trace.modules:
            try:
                if module.facets.has("q"):
                    return True
            except InvalidArgumentError as error:
                if error.fields.get("code") != "module_call_ambiguous":
                    raise
    except (AttributeError, TypeError):
        pass
    return False


def has_attention_subject(trace: Trace) -> bool:
    """Return whether the trace factually contains attention structure."""

    if _module_facets_say_attention(trace):
        return True
    for op in trace._layer_list:
        if str(record_type(op, "")) in _ATTENTION_OP_TYPES:
            return True
        for address in getattr(op, "modules", ()) or ():
            tail = str(address).split(":", 1)[0].rsplit(".", 1)[-1].lower()
            if tail in _ATTENTION_ROLE_SUFFIXES:
                return True
    return False


def _attention_role_row(address: str) -> str | None:
    """Return the attention-role row a module address names, if any."""

    parts = address.lower().split(".")
    row = _ATTENTION_ROLE_SUFFIXES.get(parts[-1])
    if row is not None:
        return row
    scoped = _SCOPED_ATTENTION_ROLE_SUFFIXES.get(parts[-1])
    if scoped is not None and any(scoped[0] in part for part in parts[:-1]):
        return scoped[1]
    return None


def attention_role_spec_fn() -> Any:
    """Return the transformer preset fn attaching per-layer role rows."""

    def _apply(layer: Any, spec: Any) -> Any:
        """Append the observed attention-role row for recognized layers."""

        layer_type = str(record_type(layer, ""))
        if layer_type in _ATTENTION_OP_TYPES:
            spec.lines.append("attention: scaled dot-product (observed)")
            return spec
        for address in getattr(layer, "modules", ()) or ():
            row = _attention_role_row(str(address).split(":", 1)[0])
            if row is not None:
                spec.lines.append(row)
                break
        return spec

    return _apply


def _bare_address(address: object) -> str:
    """Return a module address without its ``:call`` suffix."""

    return str(address).split(":", 1)[0]


def _attention_start(trace: Trace) -> str | None:
    """Return the module holding the first attention fact, or ``None``."""

    for op in trace._layer_list:
        addresses = [_bare_address(a) for a in getattr(op, "modules", ()) or ()]
        if str(record_type(op, "")) in _ATTENTION_OP_TYPES and addresses:
            return max(addresses, key=lambda address: address.count("."))
        for address in addresses:
            if _attention_role_row(address) is not None and "." in address:
                return address.rsplit(".", 1)[0]
    return None


def first_attention_block(trace: Trace) -> tuple[str, str, int] | None:
    """Return the first repeated block that holds attention, if there is one.

    Walks up from the module holding the first attention fact to the nearest
    module whose parent holds two or more children of its class (GPT-2's
    ``transformer.h.0``, ViT's ``vit.encoder.layer.0``).

    Parameters
    ----------
    trace:
        The trace being drawn.

    Returns
    -------
    tuple[str, str, int] | None
        ``(address, class name, number of sibling blocks of that class)``,
        or ``None`` when attention sits in no repeated block.

    Examples
    --------
    >>> first_attention_block(trace)  # doctest: +SKIP
    ('transformer.h.0', 'GPT2Block', 12)
    """

    start = _attention_start(trace)
    if start is None:
        return None
    try:
        by_address = {str(module.address): module for module in trace.modules}
    except (AttributeError, TypeError):
        return None
    module = by_address.get(start)
    while module is not None:
        parent_address = getattr(module, "address_parent", None)
        if parent_address is None:
            return None
        count = sum(
            1
            for other in by_address.values()
            if getattr(other, "address_parent", None) == parent_address
            and other.class_name == module.class_name
        )
        if count >= 2:
            return str(module.address), str(module.class_name), count
        module = by_address.get(str(parent_address))
    return None
