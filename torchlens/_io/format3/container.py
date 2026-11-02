"""The 3.0 bundle container: members as envelope references, relations as JSON.

Contract: ``docs/maintainers/saved_format_3.md``, section "Bundle". A bundle
artifact is a directory with its own manifest and a ``bundle.json`` index; each
member trace is a complete artifact of its own under ``members/NNNN`` and the
index references it by path plus the sha256 of the member's manifest, so a
member swapped or edited after the save refuses at load. The member relation
table (``bundle_relation_kinds``), lineage sections and preserved namespaced
sections are plain JSON objects, written by the same section writers the
tlspec 11 container used and parsed by the same closed-vocabulary parsers in
``torchlens._io.bundle_metadata``. Nothing in the container is pickled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn

from ..._utils.hashing import sha256_file
from ..format_errors import ArtifactError
from .contract import MANIFEST_FILE, REFUSAL_CODES

#: Index file of a bundle artifact, at its root.
BUNDLE_INDEX_FILE = "bundle.json"
#: Directory holding one member artifact per trace.
MEMBERS_DIR = "members"
#: Keys of one member reference. ``manifest_sha256`` names the member envelope.
MEMBER_REFERENCE_KEYS: tuple[str, ...] = ("name", "path", "manifest_sha256")

#: A member writer: ``(trace, member_dir, level) -> None``.
MemberWriter = Callable[[Any, Path, str], None]


def _refuse(situation: str, message: str, remedy: str) -> NoReturn:
    """Raise the contract's typed refusal for one situation."""

    raise ArtifactError(message, code=REFUSAL_CODES[situation], remedy=remedy)


def member_relative_path(index: int, suffix: str = ".tlspec") -> str:
    """Return the artifact-relative directory of member ``index``.

    Parameters
    ----------
    index:
        Zero-based member position.
    suffix:
        Directory suffix the member writer uses.
    """

    return f"{MEMBERS_DIR}/{index:04d}{suffix}"


def member_reference(bundle_root: Path, name: str, relative_path: str) -> dict[str, str]:
    """Build the index reference of one written member envelope.

    Parameters
    ----------
    bundle_root:
        Directory the bundle is being written into.
    name:
        Member name in the bundle.
    relative_path:
        Member directory relative to ``bundle_root``.

    Returns
    -------
    dict[str, str]
        ``name``, ``path`` and ``manifest_sha256`` of the member.
    """

    manifest = bundle_root / relative_path / MANIFEST_FILE
    if not manifest.is_file():
        _refuse(
            "writer_not_built",
            f"Bundle member {name!r} wrote no {MANIFEST_FILE} at {relative_path!r}.",
            "save each member trace on its own first to see why its save failed",
        )
    return {"name": name, "path": relative_path, "manifest_sha256": sha256_file(manifest)}


def write_members(
    bundle: Any, bundle_root: Path, *, level: str, write_member: MemberWriter
) -> list[dict[str, str]]:
    """Write every member trace as its own envelope and return the references.

    Parameters
    ----------
    bundle:
        Live bundle whose ``members`` maps names to traces.
    bundle_root:
        Directory the bundle is being written into (it exists).
    level:
        Save level applied to every member.
    write_member:
        Writes one member artifact (``Trace.save`` through the active writer).

    Returns
    -------
    list[dict[str, str]]
        One member reference per member, in bundle order.
    """

    members = getattr(bundle, "members", None)
    if not isinstance(members, Mapping):
        _refuse(
            "writer_not_built",
            "The bundle writer expected a mapping of member traces.",
            "build the bundle with tl.Bundle({name: trace, ...})",
        )
    (bundle_root / MEMBERS_DIR).mkdir()
    references: list[dict[str, str]] = []
    for index, (name, trace) in enumerate(members.items()):
        relative_path = member_relative_path(index)
        write_member(trace, bundle_root / relative_path, level)
        references.append(member_reference(bundle_root, str(name), relative_path))
    return references


def verify_member_reference(member_root: Path, entry: Mapping[str, Any], *, required: bool) -> None:
    """Check a member reference against the member envelope on disk.

    Parameters
    ----------
    member_root:
        Resolved member directory (already contained in the bundle root).
    entry:
        The member reference from ``bundle.json``.
    required:
        Whether the reference must carry ``manifest_sha256`` (every 3.0
        bundle does; the tlspec 11 container wrote none).

    Raises
    ------
    ArtifactError
        ``manifest_schema_invalid`` for an unknown key or a missing or
        mistyped hash; ``bundle_metadata_integrity_refused`` when the member
        manifest is absent or its bytes differ from the recorded hash.
    """

    unknown = sorted(set(entry) - set(MEMBER_REFERENCE_KEYS))
    expected = entry.get("manifest_sha256")
    if unknown or (expected is None and required) or not isinstance(expected, (str, type(None))):
        _refuse(
            "manifest_invalid",
            f"Bundle member reference {entry.get('name')!r} must hold exactly "
            f"{list(MEMBER_REFERENCE_KEYS)}; found {sorted(entry)}.",
            "save the bundle again; its index was altered",
        )
    if expected is None:
        return
    manifest = member_root / MANIFEST_FILE
    if manifest.is_symlink():
        _refuse(
            "symlink_path",
            f"Bundle member manifest {manifest} is a symlink.",
            "copy the bundle with its member directories as plain files",
        )
    if not manifest.is_file() or sha256_file(manifest) != expected:
        _refuse(
            "record_file_corrupt",
            f"Bundle member {entry.get('name')!r} does not match the envelope its bundle "
            "recorded: the member was replaced or edited after the bundle was saved.",
            "save the bundle again from the live traces",
        )
