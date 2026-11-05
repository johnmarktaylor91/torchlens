"""Runnable torch alias resolution: successors for recorded torch callable paths.

Sparse runnable reattachment records the dotted torch path of each callable. When
that path no longer resolves in the running torch, :func:`resolve_runnable_torch_alias`
walks the monotonic move table in ``_runnable_torch_aliases`` and returns the live
successor, probing that the target exists (and, past a row's measured upper bound,
that source and target are still one object). It reads only public torch bindings
through ``torch_attr`` and lives apart from the ``_torch_compat`` chokepoint, which
holds the private-API probes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ._torch_symbols import torch_attr

if TYPE_CHECKING:
    from ._runnable_torch_aliases import RunnableTorchAlias

__all__ = ("resolve_runnable_torch_alias",)


def _runtime_alias_target_exists(target_namespace: str, target_qualname: str) -> bool:
    """Return whether an alias target resolves in the running torch.

    Capability probe (feature detection, not a ``torch.__version__`` parse): the
    resolver only offers an alias whose recorded target actually exists in the
    current runtime, honoring :class:`RunnableTorchAlias`'s target-existence
    contract. The first hop off the top-level ``torch`` module reads
    ``torch.__dict__`` via :func:`torch_attr` (an identifier-only read that
    avoids the PEP-562 lazy-submodule-import hazard); deeper hops are off
    submodule/class roots that carry no lazy-import hazard, so they use
    ``getattr``.

    Parameters
    ----------
    target_namespace:
        Public namespace of the alias target (always rooted at ``torch``).
    target_qualname:
        Attribute path of the alias target within ``target_namespace``.

    Returns
    -------
    bool
        True only when every path segment resolves to a non-``None`` attribute.
    """

    parts = f"{target_namespace}.{target_qualname}".split(".")
    if parts[0] != "torch":
        return False
    current: Any = torch
    for part in parts[1:]:
        if current is torch:
            current = torch_attr(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return False
    return True


def _runtime_alias_identity_holds(
    source_qualname: str, target_namespace: str, target_qualname: str
) -> bool:
    """Return whether a recorded source path and its alias target are one object now.

    Capability probe (feature detection, not a ``torch.__version__`` parse) for
    an alias whose recorded upper bound the running torch has passed: the move
    the alias records is monotonic, so when both paths still resolve and the
    source *is* the target in the running torch (``torch.nn.functional.linear``
    is the ``torch._C._nn.linear`` binding on every torch this probe has met),
    the alias stays valid beyond the version the table last measured. A path
    that no longer resolves, or two distinct objects, answers ``False`` and the
    recorded bound stands.

    ``source_qualname`` may omit its ``torch.`` prefix. Identical paths (a drift
    row such as ``Tensor.add``) prove nothing and answer ``False``.
    """

    source_path = (
        source_qualname if source_qualname.startswith("torch.") else f"torch.{source_qualname}"
    )
    target_path = f"{target_namespace}.{target_qualname}"
    if source_path == target_path:
        return False
    source = _resolve_runtime_torch_path(source_path)
    target = _resolve_runtime_torch_path(target_path)
    return source is not None and source is target


def _resolve_runtime_torch_path(dotted: str) -> Any:
    """Return the object at a ``torch``-rooted dotted path, or ``None`` when any hop is missing."""

    parts = dotted.split(".")
    if parts[0] != "torch":
        return None
    current: Any = torch
    for part in parts[1:]:
        current = torch_attr(part) if current is torch else getattr(current, part, None)
        if current is None:
            return None
    return current


def _alias_target_qualname(
    alias: RunnableTorchAlias, source_qualname: str
) -> tuple[bool, str | None]:
    """Return ``(matched, target_qualname)`` for one alias row against a recorded path.

    A wildcard row (``source`` ending in ``.*``) matches any path under its prefix
    and maps the remainder, minus ``strip_target_prefix``; an exact row matches
    its own path and maps to its recorded target qualname. ``matched`` False
    means the row does not apply; ``matched`` True with an empty qualname means
    the path named the namespace itself (no callable to alias).
    """

    if alias.source.endswith(".*"):
        prefix = alias.source[:-1]
        if not source_qualname.startswith(prefix):
            return False, None
        target_qualname: str | None = source_qualname[len(prefix) :]
    elif alias.source == source_qualname:
        target_qualname = alias.target_qualname
    else:
        return False, None
    if (
        target_qualname
        and alias.strip_target_prefix
        and target_qualname.startswith(alias.strip_target_prefix)
    ):
        target_qualname = target_qualname[len(alias.strip_target_prefix) :]
    return True, target_qualname


def resolve_runnable_torch_alias(
    source_qualname: str,
    recorded_version: str | None = None,
) -> tuple[str, str, str] | None:
    """Return an explicit current-runtime successor for a recorded torch path.

    Parameters
    ----------
    source_qualname:
        Fully qualified recorded callable path.
    recorded_version:
        Torch version that produced the registry key. Aliases may be bounded to
        supported minor releases; monotonic moves with an open upper bound rely on
        the live target-existence capability probe, and a version past an alias's
        recorded upper bound is accepted only when the live identity probe
        (:func:`_runtime_alias_identity_holds`) shows the move still holds. Unknown
        versions retain compatibility behavior for descriptors that predate this
        metadata.

    Returns
    -------
    tuple[str, str, str] | None
        Target namespace, target qualname, and stable provenance, or ``None``
        when the path has no explicit monotonic alias.
    """

    from ._producer_version import _torch_minor_version  # deferred: vocabulary closure
    from ._runnable_torch_aliases import _RUNNABLE_TORCH_ALIASES

    parsed_version = _torch_minor_version(recorded_version)
    for alias in _RUNNABLE_TORCH_ALIASES:
        if parsed_version is not None and parsed_version < alias.recorded_min_version:
            continue
        beyond_recorded_max = (
            parsed_version is not None
            and alias.recorded_max_version is not None
            and parsed_version > alias.recorded_max_version
        )
        matched, target_qualname = _alias_target_qualname(alias, source_qualname)
        if not matched:
            continue
        if not target_qualname:
            return None
        if not _runtime_alias_target_exists(alias.target_namespace, target_qualname):
            # Capability gate: never manufacture an alias whose recorded target
            # is absent from the running torch (the RunnableTorchAlias contract).
            # This also closes the unbounded/legacy-version bypass: even when the
            # version bounds are skipped, a matched entry can only resolve when
            # its target genuinely exists. A later entry may still match with a
            # live target, so keep scanning.
            continue
        if beyond_recorded_max and not _runtime_alias_identity_holds(
            source_qualname, alias.target_namespace, target_qualname
        ):
            # Past the measured upper bound the table is silent; only a live
            # identity (source is target in this torch) extends the alias.
            continue
        return alias.target_namespace, target_qualname, alias.provenance
    return None
