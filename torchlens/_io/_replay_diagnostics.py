"""Producer diagnostics for captured values a runnable replay cannot recompute.

``intervention_replay_diagnostics`` names the ops a sparse runnable artifact
must refuse: values patched from another run (``tl.patch_from``) and other
user-intervention replacements, neither of which is a function of this run's
inputs. A leaf module so ``runnable`` can import it without importing the
save-entry gates, which themselves import ``runnable``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

__tl_layer__ = "L3"


_PATCH_SOURCE_BOUND = (
    "This capture carries {count} op(s) patched from another run (tl.patch_from); the "
    "patched values belong to the donor run's inputs, not to a function of this run's "
    "inputs, so no runnable artifact can recompute them. Runnable save is refused; keep "
    "the patched fork with the analysis save, trace.save(path) at level 'portable' (the "
    "default) or 'audit', which stores the patched activations as data."
)
_USER_INTERVENTION_REPLACED = (
    "This capture carries {count} user-intervention-replaced op(s); the replacement value "
    "has no traceable function, so a sparse runnable replay would recompute the "
    "UN-intervened computation -- a different function than the captured provenance. "
    "Runnable save is refused; save the intervention spec separately and re-apply it at "
    "capture time, or use an analysis-level save for the intervened artifact."
)


def intervention_replay_diagnostics(
    ops: Iterable[Any], *, diagnostic: Callable[..., Any], code: Any
) -> list[Any]:
    """Producer diagnostics for captured values a runnable replay cannot recompute.

    Ops patched from another run (``tl.patch_from``) are refused first, naming the
    analysis save that keeps the patched fork as data; every other user-intervention
    replacement is refused as a value with no traceable function. ``diagnostic`` is
    the producer's structured-diagnostic constructor and ``code`` the error code both
    refusals carry.
    """

    ops = list(ops)
    patched = tuple(str(op.label) for op in ops if _op_carries_patch_from(op))
    replaced = tuple(
        str(op.label)
        for op in ops
        if bool(getattr(op, "is_edited", False)) and str(op.label) not in patched
    )
    found: list[Any] = []
    if patched:
        found.append(
            diagnostic(
                code,
                _PATCH_SOURCE_BOUND.format(count=len(patched)),
                affected_ops=patched,
                detection_stage="producer_patch_source_bound",
            )
        )
    if replaced:
        found.append(
            diagnostic(
                code,
                _USER_INTERVENTION_REPLACED.format(count=len(replaced)),
                affected_ops=replaced,
                detection_stage="producer_user_intervention",
            )
        )
    return found


def _op_carries_patch_from(op: Any) -> bool:
    """Return whether an edited op's fire records include a ``patch_from`` edit."""

    if not getattr(op, "is_edited", False):
        return False
    for record in getattr(op, "edits", ()) or ():
        names = str(getattr(record, "edit_name", None) or "").split(",")
        if "patch_from" in names:
            return True
        edit = getattr(record, "edit", None)
        if getattr(edit, "name", None) == "patch_from":
            return True
    return False
