"""Resolve the forward pass one GradFn differentiates."""

from __future__ import annotations

from typing import Any

from .._utils._record_group_table import record_member


def _pass_op_for_grad_fn(record: Any, grad_fn_object_id: int) -> Any:
    """Return the forward pass one GradFn differentiates, from per-pass evidence.

    ``GradFn.op_label`` is the paired LAYER label, so on a multi-pass layer it
    resolves to the whole ``Layer``, whose per-pass fields (``label``) refuse by
    design. A GradFn belongs to exactly one pass: the pass whose recorded
    ``grad_fn_object_id`` is this GradFn's autograd identity.

    Parameters
    ----------
    record:
        The Op or Layer the GradFn's ``op_label`` resolves to, or ``None``.
    grad_fn_object_id:
        The GradFn's autograd node identity.

    Returns
    -------
    Any
        ``record`` itself when it is already one pass (an Op or a single-pass
        Layer), the matching pass of a multi-pass Layer, or ``None`` when no
        pass of a multi-pass Layer carries this autograd identity.
    """

    if record is None:
        return None
    passes = tuple(getattr(record, "ops", None) or ())
    if len(passes) <= 1:
        return record
    for pass_op in passes:
        if record_member(pass_op, "grad_fn_object_id", None) == grad_fn_object_id:
            return pass_op
    return None
