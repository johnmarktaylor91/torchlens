"""Clear recorded op and parameter gradient projections before a differentiable replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._quantities import Bytes

if TYPE_CHECKING:
    from .._data_classes.op import Op


def _clear_op_gradient_projection(site: Op) -> None:
    """Clear inherited gradient fields from one replay-fork op.

    Parameters
    ----------
    site:
        Operation record copied onto the replay fork.
    """

    site._clear_gradient_records()
    site._internal_set("grad", None)
    site._internal_set("transformed_grad", None)
    site.has_grad = False
    site._autograd_grad_shape = None
    site._autograd_grad_dtype = None
    site.gradient_memory = Bytes(0)
    site._payload_transformed_grad_shape = None
    site._payload_transformed_grad_dtype = None
    site.transformed_gradient_memory = Bytes(0)


def _clear_param_gradient_projection(param_log: Any) -> None:
    """Clear inherited captured-gradient state from one replay-fork Param.

    The captured AccumulateGrad records, their cached metadata, AND any
    backend-derived gradient payload are reset: ``_check_param_grad`` treats a
    surviving ``_derived_grad_payload`` as proof of a gradient, so leaving it
    would resurrect the exact stale ``has_grad = True`` this reset exists to
    kill. The lazy live-model read-through remains a deliberately distinct
    view and repopulates from the live parameter.
    """

    param_log._grad_records = []
    param_log._derived_grad_payload = None
    param_log._has_grad = False
    param_log._grad_shape = None
    param_log._grad_dtype = None
    param_log._grad_memory = Bytes(0)
