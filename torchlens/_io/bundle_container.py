"""Bundle-container loading for ``tl.load``: ``bundle.json`` and its members.

Split out of :mod:`torchlens._io.bundle` to respect its size ceiling
(split-never-raise). ``tl.load`` dispatches a unified ``kind="bundle"``
manifest here; each member loads through ``tl.load`` itself with the same
``lazy`` / ``map_location`` / ``materialize_nested`` options, so
``tl.load(bundle_path, lazy=True)`` leaves every member's tensors lazy. The
manifest's ``tlspec_version`` reaches the relation parser, which refuses a
relation kind the stamped version does not admit.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from ..data_classes.trace import Trace
from . import TorchLensIOError, _json, bundle as _bundle_module
from .bundle import _reject_symlink_path, _RenameAwareUnpickler
from .bundle_metadata import (
    _load_gated_member_relations,
    _load_lineage_sections,
    _partition_bundle_sections,
)

if TYPE_CHECKING:
    from ..bundle import Bundle

#: Split out of ``_io.bundle`` and loading through it, so it sits in the same
#: stratum (the layer map's ``_io.bundle`` row).
__tl_layer__ = "L3"


class BundleLoadContext(NamedTuple):
    """What ``tl.load`` threads into a bundle container besides its path.

    ``member_load_options`` (``lazy`` / ``map_location`` /
    ``materialize_nested``) reach each member's own ``tl.load``; ``None``
    loads members eagerly on CPU. ``tlspec_version`` is the bundle manifest's
    stamped version: relation kinds that version does not admit refuse.
    """

    member_load_options: dict[str, Any] | None = None
    tlspec_version: Any = None


# SECURITY (secF-2). Hard cap on nested-bundle recursion depth. Nested bundles are
# shallow by construction; a deep chain is a hand-edited attacker artifact. The
# resolved-path visited set closes self-reference and mutual cycles exactly; this
# cap is the belt-and-suspenders bound on any deep acyclic chain that would still
# exhaust the Python stack (and open FDs / re-parse JSON per level) before a
# ``RecursionError``.
_MAX_BUNDLE_NESTING_DEPTH = 32


def _load_unified_bundle(
    bundle_path: Path,
    *,
    bundle_visited: frozenset[Path] | None = None,
    unknown_relations: Literal["opaque", "refuse"] = "opaque",
    context: BundleLoadContext | None = None,
) -> Bundle:
    """Load a unified ``Bundle`` payload.

    Parameters
    ----------
    bundle_path:
        Directory containing a unified bundle manifest and metadata payload.
    bundle_visited:
        Internal set of already-in-progress bundle-root real paths, threaded to
        detect self-referential / mutually-recursive nested-bundle members
        (secF-2). ``None`` at the top level.
    unknown_relations:
        Loader-doctrine policy for unknown NAMESPACED member-relation kinds:
        ``"opaque"`` (default) preserves them as opaque rows, ``"refuse"``
        rejects the artifact typed. Bare unknown kinds always refuse.
    context:
        Member load options and the stamped tlspec version
        (:class:`BundleLoadContext`); ``None`` is the default context.

    Returns
    -------
    Bundle
        Loaded bundle.

    Raises
    ------
    TorchLensIOError
        If the bundle payload cannot be loaded or has the wrong type.
    """

    metadata_path = bundle_path / "bundle.json"
    _reject_symlink_path(metadata_path, context="bundle metadata")
    if metadata_path.exists():
        return _load_unified_bundle_directory(
            bundle_path,
            metadata_path,
            bundle_visited=bundle_visited,
            unknown_relations=unknown_relations,
            context=context,
        )

    legacy_pickle_path = bundle_path / "metadata.pkl"
    _reject_symlink_path(legacy_pickle_path, context="bundle metadata")
    try:
        with legacy_pickle_path.open("rb") as handle:
            # Same coarse allocation guard as the trace metadata path (B8-16):
            # the legacy kind=bundle branch fed the unpickler an uncapped file,
            # so an absurd on-disk pickle was an alloc/time DoS at tl.load even
            # though the SafeBundleUnpickler still blocked code execution.
            metadata_size = os.fstat(handle.fileno()).st_size
            if metadata_size > _bundle_module._MAX_METADATA_PKL_BYTES:
                raise TorchLensIOError(
                    f"Bundle metadata {legacy_pickle_path} is {metadata_size} bytes, above "
                    f"the {_bundle_module._MAX_METADATA_PKL_BYTES}-byte ceiling; refusing to load a "
                    "structurally implausible artifact."
                )
            if metadata_size > _bundle_module._METADATA_PKL_PRESCAN_BYTES:
                _bundle_module._prescan_metadata_pickle_opcodes(handle, legacy_pickle_path)
            bundle = _RenameAwareUnpickler(handle).load()
    except (
        pickle.UnpicklingError,
        EOFError,
        OSError,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
    ) as exc:
        raise TorchLensIOError(
            f"Failed to load bundle metadata from {legacy_pickle_path}."
        ) from exc

    from ..bundle import Bundle

    if not isinstance(bundle, Bundle):
        raise TorchLensIOError(f"Unified bundle payload at {legacy_pickle_path} is not a Bundle.")
    return bundle


def _resolve_bundle_member_path(bundle_path: Path, relative_path: str) -> Path:
    """Resolve a ``bundle.json`` member path under the bundle root, rejecting escapes.

    Mirrors the anti-traversal protection of ``resolve_bundle_blob_path`` (which is
    scoped to ``<bundle>/blobs``) for nested bundle members, which are written directly
    under the bundle root (``members/NNNN.tlspec``). A member ``path`` is portable,
    attacker-influenceable data: without this guard an absolute or ``..`` member path
    would load (and thus reconstruct) an arbitrary filesystem location outside the
    bundle. The resolved path is required to stay inside the bundle root.

    Parameters
    ----------
    bundle_path:
        Bundle root directory.
    relative_path:
        Manifest-provided member path relative to the bundle root.

    Returns
    -------
    Path
        Absolute resolved member path guaranteed to lie inside ``bundle_path``.

    Raises
    ------
    TorchLensIOError
        If the path is absolute, contains ``".."``, or resolves outside the bundle root.
    """

    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        raise TorchLensIOError(f"Bundle rejected absolute member path {relative_path!r}.")
    if ".." in candidate_path.parts:
        raise TorchLensIOError(
            f"Bundle rejected parent traversal in member path {relative_path!r}."
        )
    candidate = (bundle_path / candidate_path).resolve()
    allowed_root = bundle_path.resolve()
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise TorchLensIOError(
            f"Bundle rejected member path traversal outside bundle root: {relative_path!r}."
        ) from exc
    # SECURITY (secF-2). A member path that resolves to the bundle root itself
    # (``"."`` / ``""`` / any path collapsing onto the root) is a self-reference
    # that would re-enter this same directory and recurse without bound. The
    # containment check above passes for it (it stays inside the root), so it must
    # be rejected explicitly. Mutual / deeper cycles are additionally closed by the
    # visited-set + depth cap in ``_load_unified_bundle_directory``.
    if candidate == allowed_root:
        raise TorchLensIOError(
            f"Bundle rejected self-referential member path {relative_path!r} "
            "(resolves to the bundle root)."
        )
    return candidate


def _load_bundle_member_traces(
    bundle_path: Path,
    raw_members: list[Any],
    next_visited: frozenset[Path],
    member_load_options: dict[str, Any] | None = None,
) -> dict[str, Trace]:
    """Load each nested member spec as a Trace (cycle-checked per member).

    ``member_load_options`` (``lazy``, ``map_location``,
    ``materialize_nested``) reach each member's ``tl.load``.
    """

    members: dict[str, Trace] = {}
    for index, entry in enumerate(raw_members):
        if not isinstance(entry, dict):
            raise TorchLensIOError(f"Unified bundle member {index} must be an object.")
        name = entry.get("name")
        relative_path = entry.get("path")
        if not isinstance(name, str) or not isinstance(relative_path, str):
            raise TorchLensIOError(f"Unified bundle member {index} has invalid name/path.")
        member_path = _resolve_bundle_member_path(bundle_path, relative_path)
        if member_path.resolve() in next_visited:
            raise TorchLensIOError(
                f"Unified bundle member {name!r} forms a load cycle: its path "
                f"{relative_path!r} re-enters an in-progress bundle directory."
            )
        loaded = _bundle_module.load(
            member_path, _bundle_visited=next_visited, **(member_load_options or {})
        )
        if not isinstance(loaded, Trace):
            raise TorchLensIOError(f"Unified bundle member {name!r} did not load as a Trace.")
        members[name] = loaded
    return members


def _load_unified_bundle_directory(
    bundle_path: Path,
    metadata_path: Path,
    *,
    bundle_visited: frozenset[Path] | None = None,
    unknown_relations: Literal["opaque", "refuse"] = "opaque",
    context: BundleLoadContext | None = None,
) -> Bundle:
    """Load a unified bundle container from nested member specs.

    Parameters
    ----------
    bundle_path:
        Bundle root directory.
    metadata_path:
        ``bundle.json`` metadata path.
    bundle_visited:
        Internal set of already-in-progress bundle-root real paths, threaded down
        the recursive load chain to detect a member that re-enters this or an
        ancestor bundle (secF-2 self-reference / mutual recursion). ``None`` at the
        top level.
    unknown_relations:
        Loader-doctrine policy for unknown NAMESPACED member-relation kinds:
        ``"opaque"`` (default) preserves them as opaque rows, ``"refuse"``
        rejects the artifact typed. Bare unknown kinds always refuse.
    context:
        Member load options and the stamped tlspec version
        (:class:`BundleLoadContext`), threaded to the members and to the
        relation parser.

    Returns
    -------
    Bundle
        Reconstructed bundle.

    Raises
    ------
    TorchLensIOError
        If the nested bundle metadata or members are invalid, or a member forms a
        load cycle or exceeds the nesting-depth cap.
    """

    # SECURITY (secF-2). Record THIS bundle root before loading any member so a
    # member that points back at us (or at an ancestor already being loaded) is
    # caught as a cycle instead of recursing forever. The depth cap bounds any deep
    # acyclic chain that would still blow the stack before a ``RecursionError``.
    visited = frozenset() if bundle_visited is None else bundle_visited
    current_root = bundle_path.resolve()
    if len(visited) >= _MAX_BUNDLE_NESTING_DEPTH:
        raise TorchLensIOError(
            "Unified bundle nesting exceeds the maximum depth of "
            f"{_MAX_BUNDLE_NESTING_DEPTH}; refusing to recurse further."
        )
    next_visited = visited | {current_root}

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = _json.load_bounded(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TorchLensIOError(f"Failed to read bundle metadata from {metadata_path}.") from exc
    if not isinstance(metadata, dict):
        raise TorchLensIOError("Unified bundle metadata must be a JSON object.")
    raw_members = metadata.get("members")
    if not isinstance(raw_members, list):
        raise TorchLensIOError("Unified bundle metadata must include a members list.")

    context = context or BundleLoadContext()
    members = _load_bundle_member_traces(
        bundle_path, raw_members, next_visited, context.member_load_options
    )

    from ..bundle import Bundle

    baseline_name = metadata.get("baseline_name")
    if baseline_name is not None and not isinstance(baseline_name, str):
        raise TorchLensIOError("Unified bundle baseline_name must be a string or null.")
    preserved_sections = _partition_bundle_sections(metadata)
    member_relations = _load_gated_member_relations(
        metadata, unknown_relations=unknown_relations, tlspec_version=context.tlspec_version
    )
    lineage = _load_lineage_sections(metadata, member_names=list(members))
    loaded_bundle = Bundle(
        members,
        baseline=baseline_name,
        member_relations=member_relations,
        preserved_sections=preserved_sections,
        bundle_id=lineage["bundle_id"],
    )
    loaded_bundle._forked_from_bundle_id = lineage["forked_from_bundle_id"]
    # Anchors restore verbatim when persisted; a pre-F03 artifact carries no
    # construction evidence, which reads honestly as origin="loaded".
    anchors = lineage["member_construction"]
    if anchors is None:
        anchors = {name: {"origin": "loaded"} for name in members}
    loaded_bundle._member_construction = anchors
    loaded_bundle._operations = list(lineage["operations"])
    loaded_bundle._effect_tables = dict(lineage["member_effect_tables"])
    return loaded_bundle
