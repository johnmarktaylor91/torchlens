"""The closure-constant source op as validation recognises it (SPEC 2 rules 1 and 3).

Torch records a tensor the forward closes over as a parentless source op with
``func_name="constant"``, an internal source with no callable and no call id,
whose value is its saved payload. Validation treats it as a leaf, like a
buffer. The predicate is narrow on purpose: an op that merely carries the
``constant`` name but has parents, a callable or a call id is not one, so a
renamed computational op still fails every check it failed before.
"""

from typing import Any

from .._utils._record_group_table import record_member

#: The ``func_name`` and layer type of a closure-constant source op.
CONSTANT_FUNC_NAME = "constant"


def is_closure_constant_source(layer: Any) -> bool:
    """Return whether ``layer`` is a closure-constant source op.

    Parameters
    ----------
    layer:
        An op (live or loaded).

    Returns
    -------
    bool
        True only for a parentless internal source named ``constant`` with no
        callable and no recorded call.
    """

    return (
        getattr(layer, "func_name", None) == CONSTANT_FUNC_NAME
        and getattr(layer, "type", CONSTANT_FUNC_NAME) == CONSTANT_FUNC_NAME
        and bool(record_member(layer, "is_internal_source", False))
        and getattr(layer, "func", None) is None
        and record_member(layer, "func_call_id", None) is None
        and not tuple(getattr(layer, "parents", ()) or ())
    )
