"""Transient handoffs from the op wrapper to activation logging for same-object returns.

A call that returns its own receiver (``x.contiguous()`` on a contiguous tensor,
eval-mode dropout, ``x.type(x.type())``, an in-place ``add_``) is logged against a
fresh copy, so labelling the op never clobbers the receiver's label. The wrapper
stamps two facts on that copy and activation logging pops them at once:

* the call's mutation verdict (``set_same_object_mutation`` /
  ``pop_same_object_mutation``);
* for a NON-mutating return, the live object the caller actually receives
  (``set_same_object_live_value`` / ``reference_payload_source``). The copy is
  a dead end in the autograd graph: downstream ops consume the live object, so a
  payload saved by reference must be that object, or gradients taken with the
  op as their source come back empty. The two are the same value, since the call
  changed nothing.
"""

from __future__ import annotations

from typing import Any

from ._tl import _ensure_tensor_meta, get_tensor_meta

__all__ = [
    "pop_same_object_mutation",
    "reference_payload_source",
    "set_same_object_live_value",
    "set_same_object_mutation",
]


def set_same_object_mutation(t: Any, verdict: bool) -> None:
    """Stamp a same-object return's mutation verdict on the tensor that gets logged.

    Parameters
    ----------
    t : Any
        Fresh copy of the same-object return that activation logging will see.
    verdict : bool
        Whether the wrapped call mutated its receiver.

    Examples
    --------
    >>> import torch
    >>> copy = torch.zeros(2)
    >>> set_same_object_mutation(copy, False)
    >>> pop_same_object_mutation(copy)
    False
    """

    _ensure_tensor_meta(t).same_object_mutation = verdict


def pop_same_object_mutation(t: Any) -> bool | None:
    """Return and clear a stamped same-object mutation verdict.

    Parameters
    ----------
    t : Any
        Tensor being logged.

    Returns
    -------
    bool | None
        The stamped verdict, or ``None`` when no verdict was stamped.

    Examples
    --------
    >>> import torch
    >>> pop_same_object_mutation(torch.zeros(2)) is None
    True
    """

    meta = get_tensor_meta(t)
    if meta is None:
        return None
    verdict = meta.same_object_mutation
    meta.same_object_mutation = None
    return verdict


def set_same_object_live_value(t: Any, live: Any) -> None:
    """Stamp the live object a non-mutating same-object return hands its caller.

    Parameters
    ----------
    t : Any
        Fresh copy of the same-object return that activation logging will see.
    live : Any
        The receiver object the caller receives and downstream ops consume.

    Examples
    --------
    >>> import torch
    >>> live, copy = torch.ones(2), torch.ones(2)
    >>> set_same_object_live_value(copy, live)
    >>> reference_payload_source(copy, "reference") is live
    True
    """

    _ensure_tensor_meta(t).same_object_live_value = live


def reference_payload_source(t: Any, save_mode: str) -> Any:
    """Return the tensor a payload is saved from, clearing any live-value stamp.

    Parameters
    ----------
    t : Any
        Tensor being logged (for a same-object return, the logging copy).
    save_mode : str
        Effective activation save mode for this op.

    Returns
    -------
    Any
        The stamped live object when ``save_mode`` is ``"reference"`` and one was
        stamped, otherwise ``t``. Any other mode copies the value, and the copy's
        value equals the live object's.

    Examples
    --------
    >>> import torch
    >>> live, copy = torch.ones(2), torch.ones(2)
    >>> set_same_object_live_value(copy, live)
    >>> reference_payload_source(copy, "copy") is copy
    True
    """

    meta = get_tensor_meta(t)
    if meta is None:
        return t
    live = meta.same_object_live_value
    meta.same_object_live_value = None
    if live is None or save_mode != "reference":
        return t
    return live
