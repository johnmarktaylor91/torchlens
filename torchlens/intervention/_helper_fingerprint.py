"""Content fingerprints of the tensors a staged helper closes over (F9).

``tl.steer``, ``tl.mean_ablate``, ``tl.project_onto``, ``tl.project_off`` and
``tl.swap_with`` keep the caller's tensor by reference (no copy, so a large
steering bank costs nothing and a loop can update a direction in place). The
reference also means a later rerun or recipe save would silently use whatever
the tensor holds THEN, which is not what produced the trace. Each staged entry
therefore records a full-content digest of those tensors when it is staged
(``tl.hash.content``: every byte, so ``.data`` and NumPy-view writes that leave
the version counter alone are seen too), and reruns and ``save_intervention``
refuse ``helper_tensor_changed_since_capture`` when it moved. Re-staging takes
the new value deliberately.

A bound executor has no recorded artifact, so it stays live: it reads the
tensor at each call, and its report records the tensors' version counters
(``helper_versions``) so an in-place change is visible after the fact.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .._state import pause_logging
from .types import HelperSpec

#: HookSpec metadata key holding the staged helper's tensor digests.
DIGEST_KEY = "helper_tensor_digests"

_REMEDY = (
    "Remedy: re-stage the edit so it takes the tensor's new value on purpose "
    "(detach and attach_hooks() again, or capture fresh with tl.trace(model, x, "
    "intervene=...)), or "
    "restore the tensor's captured value"
)


def _collect(value: Any, found: list[torch.Tensor]) -> None:
    """Append every tensor held directly in a helper argument tree."""

    if isinstance(value, torch.Tensor):
        found.append(value)
    elif type(value) in (tuple, list):
        for item in value:
            _collect(item, found)
    elif type(value) is dict:
        for item in value.values():
            _collect(item, found)


def helper_of(entry_hook: Any, helper: Any = None) -> HelperSpec | None:
    """Return the helper spec behind a hook entry, if it came from a helper.

    Parameters
    ----------
    entry_hook:
        The entry's hook payload.
    helper:
        The entry's explicit helper spec, when recorded.

    Returns
    -------
    HelperSpec | None
        The helper spec, or ``None`` for a plain callable.
    """

    if isinstance(helper, HelperSpec):
        return helper
    return entry_hook if isinstance(entry_hook, HelperSpec) else None


def helper_digests(helper: HelperSpec | None) -> tuple[str, ...] | None:
    """Return the content digests of the tensors a helper closes over.

    Parameters
    ----------
    helper:
        Helper spec, or ``None``.

    Returns
    -------
    tuple[str, ...] | None
        One digest per tensor in the helper's args and kwargs, in order;
        ``None`` when the helper holds no tensor.
    """

    if helper is None:
        return None
    tensors: list[torch.Tensor] = []
    _collect(helper.args, tensors)
    _collect(helper.kwargs, tensors)
    if not tensors:
        return None
    from ..hash import content

    # Staging can happen mid-capture (the predicate door stages at fire time);
    # the digest's copy ops are TorchLens-internal and must not be logged.
    with pause_logging():
        return tuple(content(tensor) for tensor in tensors)


def helper_versions(helper: HelperSpec | None) -> tuple[int, ...] | None:
    """Return the version counters of the tensors a helper closes over.

    Parameters
    ----------
    helper:
        Helper spec, or ``None``.

    Returns
    -------
    tuple[int, ...] | None
        One ``_version`` per tensor in the helper's args and kwargs, in order;
        ``None`` when the helper holds no tensor. Reading the counter copies
        nothing and never synchronizes a device.
    """

    if helper is None:
        return None
    tensors: list[torch.Tensor] = []
    _collect(helper.args, tensors)
    _collect(helper.kwargs, tensors)
    return tuple(tensor._version for tensor in tensors) if tensors else None


def stamp_metadata(
    metadata: dict[str, Any] | None, hook: Any, helper: Any = None
) -> dict[str, Any]:
    """Return entry metadata carrying the helper's tensor digests.

    Parameters
    ----------
    metadata:
        Caller-supplied metadata, or ``None``.
    hook:
        The entry's hook payload.
    helper:
        The entry's explicit helper spec, when recorded.

    Returns
    -------
    dict[str, Any]
        ``metadata`` (a new dict when stamped); an existing digest is kept, so
        a copied entry keeps the value it was staged with.
    """

    metadata = metadata or {}
    if DIGEST_KEY in metadata:
        return metadata
    digests = helper_digests(helper_of(hook, helper))
    if digests is None:
        return metadata
    return {**metadata, DIGEST_KEY: digests}


def changed_helpers(entries: Iterable[tuple[Any, Any, Any]]) -> list[str]:
    """Return the names of helpers whose tensors changed since their digest.

    Parameters
    ----------
    entries:
        ``(helper, recorded_digests, site_description)`` triples; entries with
        no recorded digest are skipped.

    Returns
    -------
    list[str]
        ``"<helper> at <site>"`` for every changed entry.
    """

    changed = []
    for helper, recorded, site in entries:
        if recorded is None or helper is None:
            continue
        if helper_digests(helper) != tuple(recorded):
            changed.append(f"{helper.helper_name} at {site}")
    return changed


def staged_changed_helpers(spec: Any) -> list[str]:
    """Return the staged helper entries of ``spec`` whose tensors changed."""

    return changed_helpers(
        (
            helper_of(hook_spec.hook, hook_spec.helper),
            (hook_spec.metadata or {}).get(DIGEST_KEY),
            f"{hook_spec.site_target.selector_kind}:{hook_spec.site_target.selector_value}",
        )
        for hook_spec in getattr(spec, "hook_specs", None) or ()
    )


def changed_message(door: str, changed: list[str]) -> str:
    """Return the refusal message for helpers whose tensors changed."""

    return (
        f"{door} refused: the tensor held by {', '.join(changed)} changed since the "
        "edit was staged or bound. Helpers keep the caller's tensor by reference, so "
        "running now would silently apply a different edit than the one that "
        f"produced this result. {_REMEDY}"
    )


def refuse_changed_staged_helpers(spec: Any, *, door: str) -> None:
    """Refuse a door that re-reads staged helper tensors which changed.

    Parameters
    ----------
    spec:
        A trace's staged ``InterventionSpec`` (or ``None``).
    door:
        Door name for the message (``"rerun"``, ``"save_intervention"``).

    Raises
    ------
    SpecMutationError
        ``helper_tensor_changed_since_capture`` naming each changed entry.
    """

    changed = staged_changed_helpers(spec)
    if not changed:
        return
    from .errors import SpecMutationError

    raise SpecMutationError(
        changed_message(door, changed),
        code="helper_tensor_changed_since_capture",
        changed_helpers=tuple(changed),
    )
