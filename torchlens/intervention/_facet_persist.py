"""Facet-write rules of a saved recipe, rebound to the trace they load with.

A facet write (``tl.head(i, "v").in_module(m)`` plus a built-in edit) fires through a
scatter wrapper bound to the captured trace, so a recipe persists the request instead:
the built-in edit, the narrowed facet selector as the rule's site, the resolved home
label and facet name, and the write mask with its sha256. Loading a trace re-runs
``expand_facet_hook_entries`` on that selector against the loaded trace and keeps the
rebuilt wrapper only when the re-resolved home label, facet name and mask digest equal
the stored ones; otherwise the load refuses typed. A recipe loaded without a trace keeps
the request unbound, and the lost-wrapper check refuses to fire it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .._io.format_errors import ArtifactError
from .errors import SiteResolutionError
from .hooks import (
    NormalizedHookEntry,
    expand_facet_hook_entries,
    is_facet_target,
    normalize_hook_plan,
)
from .masked_edit import _mask_digest
from .save import load_intervention_spec
from .types import Edit, HookSpec, InterventionSpec, TargetSpec

#: Error code of a stored facet write the loaded trace no longer reproduces.
_NOT_REPRODUCIBLE = "facet_write_not_reproducible"


def load_trace_recipe_spec(path: str | Path, trace: Any = None) -> InterventionSpec:
    """Return the staged spec of a saved ``intervention.json``, facet writes rebound.

    Parameters
    ----------
    path:
        Artifact directory holding ``intervention.json``.
    trace:
        The loaded trace the spec attaches to; ``None`` leaves facet-write
        requests unbound (they refuse to fire until re-attached on a trace).

    Returns
    -------
    InterventionSpec
        The spec, with ``metadata["recipe_placeholders"]`` and
        ``metadata["executable"]`` set.

    Raises
    ------
    ArtifactError
        ``facet_write_not_reproducible`` when a stored facet write does not
        re-resolve to the same home label, facet name and write mask.
    """

    spec = load_intervention_spec(path)
    if trace is not None:
        spec.hook_specs = [
            _rebound(trace, hook_spec, Path(path)) if _is_facet_request(hook_spec) else hook_spec
            for hook_spec in spec.hook_specs
        ]
    return spec


def _is_facet_request(hook_spec: HookSpec) -> bool:
    """Return whether a loaded rule is a persisted facet-write request."""

    return (
        bool(hook_spec.metadata.get("facet_write"))
        and isinstance(hook_spec.helper, Edit)
        and is_facet_target(hook_spec.site_target)
    )


def _rebound(trace: Any, hook_spec: HookSpec, path: Path) -> HookSpec:
    """Return the rule with its scatter wrapper rebuilt on ``trace``, or refuse."""

    entry, problem = _reproduced_entry(trace, hook_spec)
    if entry is None:
        metadata = hook_spec.metadata
        raise ArtifactError(
            f"The facet write on {metadata.get('facet_home_label')!r} (facet "
            f"{metadata.get('facet_name')!r}) saved in {path} does not reproduce on the "
            f"loaded trace: {problem}.",
            code=_NOT_REPRODUCIBLE,
            remedy="re-attach the facet intervention on a live trace and save again; "
            "do not hand-edit the recipe",
        )
    return HookSpec(
        site_target=TargetSpec("label", entry.metadata["facet_home_label"]),
        hook=entry.normalized_callable,
        helper=hook_spec.helper,
        handle=hook_spec.handle,
        metadata=dict(hook_spec.metadata),
    )


def _reproduced_entry(trace: Any, hook_spec: HookSpec) -> tuple[NormalizedHookEntry | None, str]:
    """Re-expand the stored selector on ``trace``; return the matching entry or the problem."""

    metadata = hook_spec.metadata
    stored_mask = metadata.get("facet_write_mask")
    digest = metadata.get("facet_write_mask_sha256")
    if not isinstance(stored_mask, torch.Tensor) or _mask_digest(stored_mask) != digest:
        return None, "the stored write mask is missing or does not match its sha256"
    try:
        entries = expand_facet_hook_entries(
            trace,
            normalize_hook_plan(
                hook_spec.site_target, hook_spec.helper, allow_replay_only_site_targets=True
            ),
        )
    except SiteResolutionError as exc:
        return None, f"the facet selector no longer resolves ({exc})"
    wanted = (metadata.get("facet_home_label"), metadata.get("facet_name"))
    for entry in entries:
        if (entry.metadata["facet_home_label"], entry.metadata["facet_name"]) != wanted:
            continue
        if entry.metadata["facet_write_mask_sha256"] != digest:
            return None, "the re-resolved write mask differs from the stored one"
        return entry, ""
    return None, "the facet selector resolves to another home op or facet"


__all__ = ["load_trace_recipe_spec"]
