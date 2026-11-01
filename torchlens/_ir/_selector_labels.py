"""Label-spelling and module-address helpers for selector evaluation."""

from __future__ import annotations

from typing import Any

#: String-label attribute universes per lifecycle, in match order.
_STRING_LABEL_ATTRS: dict[str, tuple[str, ...]] = {
    "capture": ("label", "raw_label", "label_raw", "layer_label", "layer_label_short"),
    "site": ("layer_label", "label", "layer_label_short", "label_short", "_layer_label_raw"),
    "live": ("_layer_label_raw", "_label_raw", "layer_label"),
}


def module_address_matches(module_pass: Any, address: str) -> bool:
    """Return whether a module candidate belongs to an address.

    Parameters
    ----------
    module_pass:
        Pass-qualified label, ``(address, call_index)`` tuple, or tuple repr.
    address:
        Module address with or without pass qualification. A bare address
        matches every call of the module; a pass label (``"block:2"``) matches
        only that call, whichever spelling the candidate uses.

    Returns
    -------
    bool
        Whether the module candidate belongs to the requested address.
    """

    if isinstance(module_pass, tuple) and module_pass and isinstance(module_pass[0], str):
        if module_pass[0] == address:
            return True
        # Live boundary sites carry ``(address, call_index)`` tuples; a pass
        # label must compare against the same "address:call" spelling the
        # string branch below (and bind's boundary hook) uses.
        return len(module_pass) > 1 and f"{module_pass[0]}:{module_pass[1]}" == address
    module_label = str(module_pass)
    if module_label.startswith("("):
        base, sep, call = address.rpartition(":")
        if sep and call.isdigit():
            return f"'{base}', {call})" in module_label or f'"{base}", {call})' in module_label
        return f"'{address}'" in module_label or f'"{address}"' in module_label
    module_address = module_label.rsplit(":", 1)[0]
    return module_label == address or module_address == address


def _string_labels(
    subject: Any,
    lifecycle: str,
    attrs: dict[str, tuple[str, ...]] = _STRING_LABEL_ATTRS,
) -> tuple[str, ...]:
    """Return the label-string universe for one subject.

    Parameters
    ----------
    subject:
        Lifecycle subject (context, site, or live proxy).
    lifecycle:
        Active lifecycle key.
    attrs:
        Attribute-universe table; substring/regex matching passes the
        narrower :data:`_SUBSTRING_LABEL_ATTRS`.

    Returns
    -------
    tuple[str, ...]
        Non-``None`` label spellings, deduplicated in match order.
    """

    labels: list[str] = []
    seen: set[str] = set()
    for attr in attrs[lifecycle]:
        value = getattr(subject, attr, None)
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            labels.append(text)
    return tuple(labels)


def _is_module_scope_alias(kind: str, subject: Any, lifecycle: str) -> bool:
    """Return whether a module-scoped selector must skip ``subject`` as an alias.

    ``tl.module`` / ``tl.in_module`` address the ops a module call produces or
    contains. Two kinds of subject carry that module stamp without being such
    an op, and matching them would apply a value edit twice:

    - post hoc, the synthetic ``output_N`` node (``is_output``). Postprocess
      mints one per returned value; it executes nothing and re-carries its
      producer's module calls in ``output_of_module_calls`` for display only,
      and replay recomputes it from the producer;
    - live, for ``tl.in_module`` only, a module-boundary leaf whose tensor an
      op INSIDE that call produced (``_tl_boundary_inner_alias``): that op
      already matched at the op door. ``tl.module`` keeps firing at the
      boundary, where op doors never match it.

    Parameters
    ----------
    kind:
        Selector kind (``module`` or ``in_module``).
    subject:
        Site candidate.
    lifecycle:
        Active lifecycle key.

    Returns
    -------
    bool
        Whether the module-scoped selector must not match ``subject``.
    """

    if lifecycle == "site":
        return bool(getattr(subject, "is_output", False))
    if lifecycle == "live" and kind == "in_module":
        return bool(getattr(subject, "_tl_module_boundary", False)) and bool(
            getattr(subject, "_tl_boundary_inner_alias", False)
        )
    return False
