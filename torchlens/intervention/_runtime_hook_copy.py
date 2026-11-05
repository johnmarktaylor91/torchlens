"""Copy helpers for a validated live hook result: reused tensors and replacement attributes."""

from __future__ import annotations

from contextlib import suppress

import torch

from .._state import pause_logging
from ..backends.torch._tl import copy_replacement_meta


def _copy_reused_live_hook_result(out: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """Copy a hook result that is already a labeled live capture object.

    The commit path OVERWRITES the result's live label metadata in place
    (:func:`_copy_tl_replacement_attrs` -> ``copy_replacement_meta``). When a
    hook returns a REUSED tensor object -- another live op's current-session
    output, or one shared object fired at 2+ matched sites -- each fire stamps
    its own site label on the same object and the LAST fire steals it:
    consumers executing afterwards record the last site as parent, the earlier
    site's children vanish (byte-identical payloads make the misattributed
    graph validate clean), and chained edits at the orphaned site become
    silent no-ops. A labeled foreign object is therefore CLONED (autograd
    graph preserved) so the site label lands on a distinct object, mirroring
    the raw-module-hook rewire guard. The site's own pass-through result is
    exempt (the in-place carve-out).
    """

    if result is out:
        return result
    from torchlens.backends.torch import _tl as _tl_meta

    if _tl_meta.get(result) is None:
        return result
    # The copy is TorchLens-internal bookkeeping the user's program never
    # executed; it must not enter the captured graph as a spurious clone op
    # (this runs OUTSIDE _execute_hook's paused window).
    with pause_logging():
        return result.clone()


def _copy_tl_replacement_attrs(source: torch.Tensor, replacement: torch.Tensor) -> None:
    """Copy TorchLens tensor metadata from an original out to a replacement.

    Parameters
    ----------
    source:
        Original activation supplied to a user hook.
    replacement:
        Tensor returned by the hook.

    Returns
    -------
    None
        The replacement tensor is annotated in place when PyTorch permits
        dynamic tensor attributes.
    """

    if replacement is source:
        return
    with suppress(Exception):
        copy_replacement_meta(source, replacement)
