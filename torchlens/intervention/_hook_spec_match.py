"""Removal matching for sticky hook specs, used by ``InterventionSpec.remove_hook``.

Split from :mod:`torchlens.intervention._types` so that module stays under its size cap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._selection.target import TargetSpec
    from ._types import HookSpec


def _hook_spec_matches(
    hook_spec: HookSpec,
    *,
    site_target: TargetSpec | None,
    handle: str | None,
) -> bool:
    """Return whether a hook spec matches a removal request.

    Parameters
    ----------
    hook_spec:
        Sticky hook spec to inspect.
    site_target:
        Optional target spec to match.
    handle:
        Optional handle to match.

    Returns
    -------
    bool
        ``True`` when the hook spec should be removed.
    """

    if handle is not None and hook_spec.handle != handle:
        return False
    if site_target is not None and hook_spec.site_target != site_target:
        return False
    return handle is not None or site_target is not None
