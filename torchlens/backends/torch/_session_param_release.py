"""Session-end release of the parameter state a capture's preparation stamped."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ..._utils._model_walk import model_parameters
from ..._utils._tensor_label import clear_param_meta, restore_param_requires_grad

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


def restore_session_param_state(trace: Trace, model: nn.Module) -> None:
    """Restore parameter grad flags and remove session-scoped parameter metadata.

    This cleanup is deliberately independent of the exception that ended the
    capture. Callers invoke it from teardown paths that may be handling any
    ``BaseException`` raised by user code.

    The AUTHORITATIVE clear iterates the RECORDED prep inventory
    (``trace._session_param_inventory``), never only the live model tree -- a
    parameter popped from ``_parameters`` mid-forward escapes a
    ``model.parameters()`` re-traversal, and its surviving prep stamp would let
    a LATER capture accept stale provenance (false VERIFIED / wrong-bind). The
    live-tree walk is kept as belt-and-suspenders for parameters the inventory
    does not hold. A parameter the inventory pass just released is skipped
    there: releasing is idempotent (``clear_meta`` pops with a default,
    ``restore_param_requires_grad`` no-ops once the stamp is gone), so the
    second release of the same object did nothing but cost calls.

    Parameters
    ----------
    trace
        Trace whose session prep recorded the stamped-parameter inventory.
    model
        Model whose parameters were prepared for the capture session.
    """

    inventory = getattr(trace, "_session_param_inventory", None) or ()
    for param in inventory:
        restore_param_requires_grad(param)
        clear_param_meta(param)
    # ``inventory`` stays referenced until this function returns, so every id
    # below still names the parameter object that was released above.
    released = set(map(id, inventory))
    if inventory:
        trace._session_param_inventory = []
    for param in model_parameters(model):
        if id(param) not in released:
            restore_param_requires_grad(param)
            clear_param_meta(param)
