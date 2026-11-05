"""Recover a fused SDPA call's exact arguments from what a capture recorded.

A ``save_args`` / ``reconstruction_ready`` capture stores the call's arguments
verbatim (``saved_args`` / ``saved_kwargs``). A default capture stores no
argument copies, but it does record everything needed to rebuild them exactly:
the replay argument template (``args_template``: a ``ParentRef`` per graph
tensor, a ``LiteralTensor`` copy per unlabeled tensor, a ``LiteralValue`` per
scalar such as ``scale`` / ``is_causal`` / ``dropout_p``), the slot-to-parent map
(``parent_arg_positions``), and every parent op's saved output (with the exact
version this call read in ``out_versions_by_child`` when a later in-place op
moved it). This module rebuilds the arguments from those records only, and
names the first missing piece instead of guessing: a slot whose producer output
was not saved, an unsupported template leaf, or a label that does not resolve to
exactly one op all fail closed.
"""

from __future__ import annotations

from typing import Any

import torch

from ..intervention.types import CapturedArgTemplate, LiteralTensor, LiteralValue, ParentRef

#: Missing-prerequisite text for a capture that recorded neither saved arguments
#: nor a replay template (``structure_only`` captures).
NO_ARGUMENT_RECORD = "the SDPA call's arguments (no saved args and no replay argument template)"


class _Unrecoverable(Exception):
    """One argument slot cannot be rebuilt exactly from the recorded capture."""


def sdpa_call_arguments(op: Any, trace: Any) -> tuple[list[Any], dict[str, Any]] | str:
    """Return the SDPA op's positional and keyword arguments, or what is missing.

    Parameters
    ----------
    op:
        Fused ``scaled_dot_product_attention`` op record.
    trace:
        Trace owning ``op``; parent labels resolve against its op table.

    Returns
    -------
    tuple[list[Any], dict[str, Any]] | str
        ``(args, kwargs)`` exactly as the call received them, or a short
        description of the first prerequisite the capture did not record.
    """

    saved_args = getattr(op, "saved_args", None)
    saved_kwargs = getattr(op, "saved_kwargs", None)
    if saved_args is not None and saved_kwargs is not None:
        return list(saved_args), dict(saved_kwargs)
    template = getattr(op, "args_template", None)
    if not isinstance(template, CapturedArgTemplate):
        return NO_ARGUMENT_RECORD
    positions = getattr(op, "parent_arg_positions", None) or {}
    try:
        args = [
            _resolve(component, op, trace, positions.get("args", {}), index)
            for index, component in enumerate(template.args)
        ]
        kwargs = {
            key: _resolve(component, op, trace, positions.get("kwargs", {}), key)
            for key, component in template.kwargs
        }
    except _Unrecoverable as missing:
        return str(missing)
    return args, kwargs


def _resolve(component: Any, op: Any, trace: Any, slot_parents: dict[Any, str], key: Any) -> Any:
    """Return one argument slot's exact value from its template component."""

    if isinstance(component, ParentRef):
        label = slot_parents.get(key)
        if not isinstance(label, str):
            raise _Unrecoverable(f"the producer of SDPA argument {key!r} (no recorded parent slot)")
        return _parent_value(op, trace, label, key)
    if isinstance(component, LiteralTensor):
        if not isinstance(component.value, torch.Tensor):
            raise _Unrecoverable(f"the recorded tensor literal for SDPA argument {key!r}")
        return component.value
    if isinstance(component, LiteralValue):
        return component.value
    raise _Unrecoverable(
        f"SDPA argument {key!r}: replay template holds {type(component).__name__}, "
        "not a recoverable value"
    )


def _parent_value(op: Any, trace: Any, label: str, key: Any) -> torch.Tensor:
    """Return the version of parent ``label`` that ``op`` read at capture time."""

    by_key = getattr(trace, "_layer_dict_all_keys", None) or {}
    parent = by_key.get(label) if hasattr(by_key, "get") else None
    if parent is None:
        raise _Unrecoverable(f"the producer {label!r} of SDPA argument {key!r} (label unresolved)")
    versions = getattr(parent, "out_versions_by_child", None) or {}
    for spelling in (getattr(op, "label", None), getattr(op, "layer_label", None)):
        if spelling in versions and isinstance(versions[spelling], torch.Tensor):
            return versions[spelling]
    value = saved_output(parent)
    if value is None:
        raise _Unrecoverable(
            f"the saved output of {label!r}, which feeds SDPA argument {key!r} "
            "(that op's activation was not saved)"
        )
    return value


def saved_output(op: Any) -> torch.Tensor | None:
    """Return an op's saved output tensor, or ``None`` when the capture did not keep it.

    Reads through ``Op.out`` so a lazily loaded bundle materializes its payload;
    an unsaved row's typed refusal (a ``ValueError``) reads as ``None``.
    """

    try:
        value = op.out
    except (AttributeError, RuntimeError, ValueError):
        return None
    return value if isinstance(value, torch.Tensor) else None
