"""The 3.0 bundle container: members as envelope references, relations as JSON.

Contract: ``docs/maintainers/saved_format_3.md``, section 1 "Directory layout".
A bundle artifact is a directory with its own ``manifest.json`` (``kind:
"bundle"``), one complete trace artifact per member under
``members/<index>-<name>/`` and the member relation table in
``relations.json``. The bundle's member list references each member by name,
path and the sha256 of the member's ``manifest.json``, so a member swapped or
edited after the save refuses at load. Relation rows are the closed
``bundle_relation_kinds`` rows, parsed by the same closed-vocabulary parser in
``torchlens._io.bundle_metadata`` the tlspec 11 container used. Nothing in the
container is pickled.

The tlspec 11 container (``bundle.json``) uses the member helpers today; the
3.0 bundle writer and reader doors use all of them.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from ..._utils.hashing import sha256_file
from .._json import read_bounded
from ..bundle_metadata import _load_gated_member_relations
from ..paths import reject_symlink_path
from .contract import MANIFEST_FILE, REFUSAL_CODES
from .runnable_families import refuse as _refuse
from .validate import integrity_error, read_json_file, symlink_guard

#: Index file of a bundle artifact, at its root.
BUNDLE_INDEX_FILE = "relations.json"
#: Directory holding one member artifact per trace.
MEMBERS_DIR = "members"
#: The member relation table of a 3.0 bundle, at its root.
RELATIONS_FILE = "relations.json"
#: Longest member name kept in a member directory name.
MEMBER_NAME_CHARS = 64
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
#: Keys of one member reference. ``manifest_sha256`` names the member envelope.
MEMBER_REFERENCE_KEYS: tuple[str, ...] = ("name", "path", "manifest_sha256")

#: A member writer: ``(trace, member_dir, level) -> None``.
MemberWriter = Callable[[Any, Path, str], None]


def member_relative_path(index: int, name: str) -> str:
    """Return the artifact-relative directory of one member: ``members/<index>-<name>``.

    The name part keeps letters, digits, ``.``, ``_`` and ``-`` (anything else
    becomes ``_``) and at most ``MEMBER_NAME_CHARS`` characters; the index
    prefix keeps every directory unique and never a ``.`` or ``..`` component.
    The member's real name is the reference's ``name``.

    Parameters
    ----------
    index:
        Zero-based member position.
    name:
        Member name in the bundle.
    """

    safe = _UNSAFE_NAME_CHARS.sub("_", name)[:MEMBER_NAME_CHARS]
    return f"{MEMBERS_DIR}/{index:04d}-{safe}"


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
    (bundle_root / MEMBERS_DIR).mkdir(mode=0o700)
    references: list[dict[str, str]] = []
    for index, (name, trace) in enumerate(members.items()):
        relative_path = member_relative_path(index, str(name))
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
        The member reference from ``relations.json``.
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


def write_relations(bundle: Any, bundle_root: Path) -> dict[str, Any] | None:
    """Write ``relations.json`` when the bundle has relation rows.

    Parameters
    ----------
    bundle:
        Live bundle; its member relation table supplies the rows.
    bundle_root:
        Directory the bundle is being written into.

    Returns
    -------
    dict[str, Any] | None
        The manifest ``files`` row (``sha256``, ``bytes``) of the file, or
        ``None`` when the bundle has no relation rows and no file was written.
    """

    table = getattr(bundle, "_member_relations", None)
    if table is None or len(table.rows) == 0:
        return None
    target = bundle_root / RELATIONS_FILE
    text = json.dumps(table.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    target.write_text(text + "\n", encoding="utf-8")
    os.chmod(target, 0o600)
    return {"sha256": sha256_file(target), "bytes": target.stat().st_size}


def _verify_relations_file(target: Path, row: Any) -> None:
    """Refuse a symlinked relation table or one that differs from its ``files`` row."""

    reject_symlink_path(target, context="bundle relations", code=REFUSAL_CODES["symlink_path"])
    if (
        not isinstance(row, Mapping)
        or not target.is_file()
        or target.stat().st_size != row.get("bytes")
        or sha256_file(target) != row.get("sha256")
    ):
        _refuse(
            "record_file_corrupt",
            f"The bundle relation table {target} does not match its sha256 and byte length.",
            "save the bundle again from the live traces",
        )


def read_relations(
    bundle_root: Path,
    files: Mapping[str, Any],
    *,
    unknown_relations: Literal["opaque", "refuse"] = "opaque",
    tlspec_version: Any = None,
) -> tuple[Any, ...] | None:
    """Read and validate ``relations.json``; ``None`` when the bundle lists none.

    Parameters
    ----------
    bundle_root:
        Bundle artifact directory.
    files:
        The bundle manifest's ``files`` rows; the relation file is read only
        when listed there, and only after its sha256 and byte length match.
    unknown_relations:
        The loader's policy for well-formed rows of unknown namespaced kinds.
    tlspec_version:
        The bundle manifest's record contract version.

    Returns
    -------
    tuple | None
        Parsed relation rows for the ``Bundle`` constructor, or ``None``.
    """

    row = files.get(RELATIONS_FILE)
    target = bundle_root / RELATIONS_FILE
    if row is None:
        # Contract section 1: a file the manifest does not list is never opened or read.
        return None
    _verify_relations_file(target, row)
    try:
        payload = read_bounded(target)
    except (OSError, ValueError) as exc:
        _refuse(
            "record_file_corrupt",
            f"The bundle relation table {target} is not readable JSON ({type(exc).__name__}).",
            "save the bundle again from the live traces",
        )
    return _load_gated_member_relations(
        {"member_relations": payload},
        unknown_relations=unknown_relations,
        tlspec_version=tlspec_version,
    )


_MEMBER_DIR = re.compile(r"^members/(\d+)-([A-Za-z0-9_.-]+)$")
_CORRUPT_CODE = REFUSAL_CODES["record_file_corrupt"]


def read_member_index(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Read ``relations.json``: an object whose ``members`` list names each member's path."""

    target = root / "relations.json"
    if "relations.json" not in manifest["files"]:
        raise integrity_error(target, "a bundle must list relations.json", code=_CORRUPT_CODE)
    data, _ = read_json_file(target, code=_CORRUPT_CODE, what="relations")
    members = data.get("members") if isinstance(data, dict) else None
    if not isinstance(members, list) or not all(
        isinstance(member, dict)
        and isinstance(member.get("name"), str)
        and isinstance(member.get("path"), str)
        for member in members
    ):
        raise integrity_error(
            target, "relations.json must be {members: [{name, path}, ...], ...}", code=_CORRUPT_CODE
        )
    return data


def iter_members(
    root: Path, manifest: Mapping[str, Any], metadata: Mapping[str, Any]
) -> Iterator[tuple[str, Path]]:
    """Members in ``relations.json`` order; every ``members/`` file must belong to one."""

    declared: dict[str, str] = {}
    for member in metadata["members"]:
        path, name = member["path"], member["name"]
        if _MEMBER_DIR.match(path) is None or path in declared or name in declared.values():
            raise integrity_error(
                root / path,
                "bundle members must be distinct members/<index>-<name>/ directories",
                code=_CORRUPT_CODE,
            )
        declared[path] = name
    for relpath in manifest["files"]:
        parts = relpath.split("/")
        if parts[0] == "members" and "/".join(parts[:2]) not in declared:
            raise integrity_error(
                root / relpath, "member files must belong to a declared member", code=_CORRUPT_CODE
            )
    for member in metadata["members"]:
        member_root = root / member["path"]
        symlink_guard(member_root, "bundle member")
        verify_member_reference(member_root, member, required=True)
        yield member["name"], member_root
