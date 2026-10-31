"""``merged-directory`` artifact: canonical descriptor, tree hashes, load rederivation.

Layout (design-merge-ranks-c v5, 4.1)::

    <root>/
      manifest.json            # root integrity record (merged-directory)
      merge/descriptor.json    # canonical-JSON derivation CACHE
      members/rank_NNNN.tlspec # ordinary rank-core bundles, byte-identical
                               # to standalone saves (P1)

The descriptor is a CACHE; the rank cores are the authority (4.3). Every load
verifies integrity (descriptor bytes vs root-manifest checksum, member tree
hashes), then INDEPENDENTLY rederives the merge from the rank cores' boundary
records and requires EXACT equality with the cache -- any inequality is a
typed tamper refusal, never a gap. Presence expectations derive from the group
memberships recorded INSIDE the surviving rank cores, never from the editable
descriptor input list (which can only widen them).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .._io import _json
from .._io._durability import fsync_dir, fsync_tree
from ._engine import derive_merge
from ._enums import (
    MERGED_BUNDLE_FORMAT,
    MERGED_DESCRIPTOR_KIND,
    MERGED_DESCRIPTOR_SCHEMA_VERSION,
    MERGED_TLSPEC_VERSION,
    MergedErrorCode,
)
from ._errors import MergedArtifactError, MergeInputError
from ._evidence import extract_rank_evidence
from ._presenter import MergedTrace, _RankHandle

__all__ = ["canonical_json_bytes", "load_merged", "save_merged", "tree_hash"]

_TREE_HASH_CHUNK_BYTES = 1 << 20

_EXC_TEXT_LIMIT = 300


def _bounded_exc(exc: BaseException) -> str:
    """Bounded ``TypeName: text`` rendering of a chained cause (R65-14).

    Artifact-controlled exception strings are interpolated into refusal
    messages; without a cap a hostile artifact chooses the message size.
    """

    text = f"{type(exc).__name__}: {exc}"
    if len(text) > _EXC_TEXT_LIMIT:
        text = text[: _EXC_TEXT_LIMIT - 3] + "..."
    return text


CANONICAL_ENCODING = "torchlens-canonical-json-v1"
"""UTF-8, sorted keys, EXACT-``str`` keys only, no NaN/Infinity, no lone
surrogates, LF, no insignificant whitespace."""


def _reject_noncanonical(obj: Any, path: str = "$") -> None:
    """Refuse payload shapes ``json.dumps`` would coerce or die on (R73).

    The output feeds SHA-256 integrity digests and byte-equality derivation
    oracles, so silent coercion is a canonicalization COLLISION: ``{1: "a"}``
    and ``{"1": "a"}`` produced identical bytes (likewise ``True``/``"true"``),
    ``{"1": a, 1: b}`` would emit DUPLICATE keys, and int-keyed maps sort
    numerically while their coerced forms sort lexicographically — letting two
    structurally distinct payloads share one attested digest. Keys must be
    exactly ``str`` (subclasses can carry hidden state past a digest).
    Non-finite floats, lone surrogates, and unserializable leaves raised raw
    ``ValueError``/``UnicodeEncodeError``/``TypeError`` past the module's
    otherwise-typed refusal surface. Reject all of it typed at the boundary.
    """

    if isinstance(obj, dict):
        for key, value in obj.items():
            if type(key) is not str:
                raise _schema_refusal(
                    f"canonical JSON requires string mapping keys; {path} carries a "
                    f"{type(key).__name__} key (silent coercion would collide digests)"
                )
            _reject_noncanonical(value, f"{path}.{key}")
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _reject_noncanonical(value, f"{path}[{index}]")
        return
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        raise _schema_refusal(f"canonical JSON forbids non-finite floats; {path} is {obj!r}")
    if isinstance(obj, str):
        try:
            obj.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise _schema_refusal(
                f"canonical JSON requires UTF-8-encodable strings; {path} carries a lone surrogate"
            ) from exc


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize ``obj`` under the canonical encoding declared above."""

    _reject_noncanonical(obj)
    try:
        text = json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise _schema_refusal(
            f"payload is not canonically serializable ({_bounded_exc(exc)})"
        ) from exc
    return (text + "\n").encode("utf-8")


def tree_hash(root: Path) -> str:
    """Canonical tree hash of a rank-core directory (4.1).

    Sorted POSIX-relative paths; per entry ``path_len path size sha256(bytes)``;
    the tree hash is the SHA-256 of the concatenated entries. Symlinks are
    REJECTED at hash time, matching the loader guards.
    """

    entries: list[bytes] = []
    for candidate in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if candidate.is_symlink():
            raise MergedArtifactError(
                f"Symlink {candidate} inside a rank core; merged artifacts "
                "reject symlinks at hash time.",
                code=MergedErrorCode.MERGED_SCHEMA_INVALID,
                remedy="remove the symlink from the rank core and re-save",
            )
        if not candidate.is_file():
            continue
        # Chunked: a rank core's safetensors blobs are legitimately multi-GiB, so
        # ``read_bytes()`` materialized the whole file just to hash it.
        size = 0
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            while True:
                chunk = handle.read(_TREE_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        entry = _tree_hash_entry(
            candidate.relative_to(root).as_posix(),
            size,
            digest.hexdigest(),
        )
        entries.append(entry)
    return hashlib.sha256(b"\n".join(entries)).hexdigest()


def _tree_hash_entry(relative_path: str, size: int, digest: str) -> bytes:
    """Frame one merged tree-hash entry without delimiter ambiguity.

    Parameters
    ----------
    relative_path:
        POSIX-relative member path.
    size:
        File size in bytes.
    digest:
        Hexadecimal SHA-256 of the file body.

    Returns
    -------
    bytes
        Length-prefixed canonical entry bytes.
    """

    path_bytes = relative_path.encode("utf-8")
    return (
        len(path_bytes).to_bytes(8, "big")
        + path_bytes
        + size.to_bytes(8, "big")
        + bytes.fromhex(digest)
    )


def _member_dirname(rank: int) -> str:
    """Canonical per-member directory name for one rank."""

    return f"rank_{rank:04d}.tlspec"


_MEMBER_ENTRY_KEYS = frozenset({"rank", "path", "tree_sha256"})
"""Closed key set for one descriptor ``members`` entry (unknown keys refuse)."""


_TAMPER_REMEDY = (
    "The artifact's integrity records do not match its contents; it was edited "
    "or corrupted in transit. Re-generate it from the original rank cores with "
    "tl.merge_ranks([...]).save(path)."
)
_SCHEMA_REMEDY = (
    "The artifact does not match the merged-directory schema this runtime "
    "supports. Re-generate it with tl.merge_ranks([...]).save(path) using a "
    "matching torchlens release."
)


# Bundle-load failures that are INTEGRITY signals (tamper), not environmental
# degradation. A guarded-unpickler denylist refusal / corrupt pickle stream is
# raised as ``pickle.UnpicklingError``/``EOFError`` and, at the ``bundle.load``
# boundary, wrapped into a ``TorchLensIOError`` tagged with this code (A-R58-1).
_BUNDLE_INTEGRITY_CODE = "bundle_metadata_integrity_refused"


def _is_bundle_integrity_refusal(exc: BaseException) -> bool:
    """Return True if a member bundle-load failure is a tamper signal, not drift.

    A guarded-unpickler denylist refusal or a corrupt/truncated pickle stream is
    a bundle-INTEGRITY failure: it means the member core's bytes are inconsistent
    with an honest capture, which is tamper. It is distinguished from genuine
    environmental degradation (torch/codec import drift, a missing optional
    dependency) both by the stable ``code`` the ``bundle.load`` boundary stamps
    on it AND by walking the ``__cause__``/``__context__`` chain for the raw
    ``pickle.UnpicklingError``/``EOFError`` (belt-and-braces: a refusal that
    reaches here through any other wrapper is still caught).
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (pickle.UnpicklingError, EOFError)):
            return True
        fields = getattr(current, "fields", None)
        if isinstance(fields, dict) and fields.get("code") == _BUNDLE_INTEGRITY_CODE:
            return True
        current = current.__cause__ or current.__context__
    return False


def _tamper(detail: str, *, remedy: str = _TAMPER_REMEDY, **payload: Any) -> MergedArtifactError:
    """Build the typed tamper refusal (integrity failure, never a presence gap).

    Every merged refusal now carries a ``fields["remedy"]`` (R65: the tamper
    refusals shipped with none, even where the correct sentence existed verbatim
    elsewhere in this module).
    """

    return MergedArtifactError(
        f"Merged artifact integrity failure: {detail}",
        code=MergedErrorCode.MERGED_DESCRIPTOR_TAMPER,
        remedy=remedy,
        **payload,
    )


def _schema_refusal(
    detail: str, *, remedy: str = _SCHEMA_REMEDY, **payload: Any
) -> MergedArtifactError:
    """Build the typed merged-artifact schema refusal (carries a remedy, R65)."""

    return MergedArtifactError(
        f"Merged artifact schema refusal: {detail}",
        code=MergedErrorCode.MERGED_SCHEMA_INVALID,
        remedy=remedy,
        **payload,
    )


def _restrict_mode(path: Path, mode: int) -> None:
    """Best-effort tighten a merged-artifact path's permissions (POSIX only).

    ``mkdir``/``write_bytes`` honor the ambient umask, so under the common
    umask 002 the merged root, its ``merge/`` subdir, and the descriptor /
    manifest sidecars were left group-writable/readable even though the copied
    rank cores inside stay 0600/0700 (B8-10 parity, missed for the merged
    writer). Best-effort: a filesystem that ignores mode bits is not a save
    failure.

    Parameters
    ----------
    path:
        Merged-artifact directory or file to tighten.
    mode:
        Target permission bits (``0o700`` for directories, ``0o600`` for
        files).
    """

    if os.name != "posix":
        return
    try:
        path.chmod(mode)
    except OSError:
        pass


def save_merged(merged: MergedTrace, path: str | Path, *, overwrite: bool = False) -> None:
    """Write a ``merged-directory`` artifact.

    Parameters
    ----------
    merged:
        The merged trace to persist.
    path:
        Output directory path.
    overwrite:
        Whether an existing directory may be replaced.

    Raises
    ------
    MergedArtifactError
        If the target exists (without ``overwrite``) or a member cannot be
        persisted.
    """

    root = Path(path)
    # ``reason`` disambiguates the frozen MERGE_INPUT_INVALID code (R65-4):
    # bad merge EVIDENCE and save-target FILESYSTEM conditions shared the one
    # code with nothing but message text to tell them apart.
    if root.is_symlink():
        raise MergedArtifactError(
            f"Refusing symlinked merged artifact target: {root}.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="save_target_symlink",
            remedy="pass the resolved real path as the save target",
        )
    if root.exists() and not overwrite:
        raise MergedArtifactError(
            f"{root} already exists; pass overwrite=True to replace it.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="save_target_exists",
            remedy="pass overwrite=True or choose a fresh target path",
        )
    # Re-verify derivation-vs-members BEFORE writing anything (deep-hunt F12):
    # a live input trace whose distributed annotations were mutated between
    # merge_ranks and save() would otherwise produce an artifact whose members
    # never rederive to the cached descriptor -- every future load refuses as
    # merged_descriptor_tamper, a permanent false tamper accusation for an
    # honest sequence. The refusal belongs at save time, where it is fixable.
    reverified = derive_merge(
        {
            rank: extract_rank_evidence(merged._handles[rank].trace, f"save-reverify[{rank}]")
            for rank in merged.rank_ids
        },
        merged._derivation.expected_ranks,
    )
    if canonical_json_bytes(reverified.to_payload()) != canonical_json_bytes(
        merged._derivation.to_payload()
    ):
        raise MergedArtifactError(
            "The merge inputs no longer rederive this MergedTrace's derivation "
            "(their distributed evidence changed after merge_ranks, or this is "
            "a degraded load whose lost members cannot be re-attested). Saving "
            "would produce an artifact every future load refuses as tampered; "
            "re-merge the current inputs and save that result instead.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="derivation_drift",
            remedy="re-merge the current inputs and save that result",
        )

    staging_root = root.parent / f"{root.name}.tmp.{uuid.uuid4().hex}"
    backup_root: Path | None = None
    try:
        members_dir = staging_root / "members"
        members_dir.mkdir(parents=True)
        _restrict_mode(staging_root, 0o700)
        _restrict_mode(members_dir, 0o700)
        (staging_root / "merge").mkdir()
        _restrict_mode(staging_root / "merge", 0o700)

        members_payload: list[dict[str, Any]] = []
        for rank in merged.rank_ids:
            handle = merged._handles[rank]
            member_path = members_dir / _member_dirname(rank)
            if handle.path is not None:
                source = Path(handle.path)
                if source.is_dir():
                    # Copy is the only link mode: member bytes stay byte-identical
                    # to the standalone rank-core save (P1). ``symlinks=True``
                    # copies a symlink AS a symlink instead of DEREFERENCING it
                    # (b4:R38-5): the default ``symlinks=False`` followed a symlink
                    # inside the source rank core and copied its TARGET's bytes
                    # (e.g. a private file outside the core) INTO the shareable
                    # merged artifact -- an exfiltration channel. Preserved
                    # symlinks are then refused by the ``tree_hash`` guard below
                    # (merged artifacts reject symlinks at hash time), so a
                    # symlinked rank core fails the save instead of leaking.
                    shutil.copytree(source, member_path, symlinks=True)
                else:
                    raise MergedArtifactError(
                        f"Rank {rank} core path {source} is not a bundle directory.",
                        code=MergedErrorCode.MERGE_INPUT_INVALID,
                        reason="member_path_not_directory",
                        remedy="point the rank handle at its .tlspec bundle directory",
                    )
            else:
                from .._io.bundle import save as save_bundle

                save_bundle(handle.trace, member_path)
            members_payload.append(
                {
                    "rank": rank,
                    "path": f"members/{_member_dirname(rank)}",
                    "tree_sha256": tree_hash(member_path),
                }
            )

        descriptor = {
            "descriptor_kind": MERGED_DESCRIPTOR_KIND,
            "schema_version": MERGED_DESCRIPTOR_SCHEMA_VERSION,
            "encoding": CANONICAL_ENCODING,
            "members": members_payload,
            "derivation": merged._derivation.to_payload(),
        }
        descriptor_bytes = canonical_json_bytes(descriptor)
        (staging_root / "merge" / "descriptor.json").write_bytes(descriptor_bytes)
        _restrict_mode(staging_root / "merge" / "descriptor.json", 0o600)

        import platform as platform_module
        from datetime import datetime, timezone

        import torch

        from .. import __version__ as torchlens_version

        manifest = {
            "tlspec_version": MERGED_TLSPEC_VERSION,
            "bundle_format": MERGED_BUNDLE_FORMAT,
            "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
            "members": {str(entry["rank"]): entry["tree_sha256"] for entry in members_payload},
            "torchlens_version": str(torchlens_version),
            "torch_version": str(torch.__version__),
            "python_version": platform_module.python_version(),
            # Coarse system-machine form, matching the core bundle writer's
            # deliberate choice: the full platform.platform() string leaks the
            # kernel build, libc, and cloud image tag into a shareable
            # artifact (B8-22).
            "platform": f"{platform_module.system().lower()}-{platform_module.machine().lower()}",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _restrict_mode(staging_root / "manifest.json", 0o600)

        # Crash-durability before publish: fsync every written blob/sidecar and
        # the staged directories so a power/OS crash after the rename below
        # cannot leave a "successfully saved" artifact holding zero-length or
        # partial files. load_merged requires EXACT descriptor-sha + per-member
        # tree-hash equality, so a torn publish reads back as a PERMANENT false
        # merged_descriptor_tamper accusation on an honest save (deep-hunt F12);
        # the previous version is already renamed aside. Mirrors the .tlspec
        # writers (_io/bundle.py, _io/tlspec.py).
        fsync_tree(staging_root)

        # Re-check the target existence/overwrite policy at publish time, not
        # only at entry: the staging copytree/save_bundle above can run for
        # minutes, and a path created at ``root`` in that window would otherwise
        # be silently renamed aside and destroyed even under overwrite=False
        # (TOCTOU). Mirrors _io/bundle.py's publish-time FileExistsError.
        if root.exists():
            if not overwrite:
                raise MergedArtifactError(
                    f"{root} was created by another writer during the save; "
                    "pass overwrite=True to replace it.",
                    code=MergedErrorCode.MERGE_INPUT_INVALID,
                    reason="save_target_exists",
                    remedy="pass overwrite=True or choose a fresh target path",
                )
            backup_root = root.parent / f"{root.name}.bak.{uuid.uuid4().hex}"
            root.rename(backup_root)
        staging_root.rename(root)
        # Make the rename itself durable before declaring the save complete.
        fsync_dir(root.parent)
        if backup_root is not None:
            shutil.rmtree(backup_root, ignore_errors=True)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        if backup_root is not None and not root.exists() and backup_root.exists():
            try:
                backup_root.rename(root)
            except OSError:
                # Double fault: the save failed AND the restore failed. The
                # prior artifact is gone from its canonical path but still
                # exists under the backup name -- disclose it instead of
                # stranding it under a hidden name the error never mentions
                # (twin of the _io/bundle.py _restore_backup disclosure).
                import warnings

                from ..errors._base import TorchLensWarning

                # Categorized so operators can filter it with
                # filterwarnings(category=TorchLensWarning); a bare UserWarning
                # cannot be targeted (R66 F7).
                warnings.warn(
                    f"Failed to restore the previous merged artifact after a "
                    f"failed overwrite; it remains recoverable at {backup_root}",
                    TorchLensWarning,
                    stacklevel=2,
                )
        raise


def _cached_verdict(cached: dict[str, Any], key: str, enum_type: Any) -> Any:
    """Convert one cached verdict field typed; missing/foreign values refuse.

    The degraded-load branch is the ONE consumer of cache fields that exact
    rederivation equality has not already proven well-formed, so a missing key
    or a value outside the closed vocabulary previously escaped as a raw
    KeyError/ValueError instead of the documented typed refusal (R18-5).
    """

    if key not in cached:
        raise _schema_refusal(f"descriptor derivation cache is missing {key!r}")
    value = cached[key]
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _schema_refusal(
            f"descriptor derivation cache {key}={value!r} is outside the closed vocabulary"
        ) from exc


def _cached_join_refs(join: Any, index: int) -> tuple[dict[str, Any], dict[int, Any]]:
    """Typed parse of one cached join row into engine-comparable pieces.

    The degraded branch is the one consumer of cache rows that exact
    rederivation equality has not proven well-formed, so every field a
    verdict recomputation reads is validated here (typed schema refusal,
    never a raw KeyError/TypeError).
    """

    from ._engine import PerRankRef

    def refuse(detail: str) -> MergedArtifactError:
        """Build the typed schema refusal for this join row, naming its index."""

        return _schema_refusal(f"descriptor cache join {index} {detail}")

    if not isinstance(join, dict):
        raise refuse("is not a JSON object")
    key = join.get("key")
    if not isinstance(key, list) or len(key) != 4:
        raise refuse("has a malformed key")
    if not isinstance(join.get("kind"), str):
        raise refuse("has no kind")
    backend = join.get("backend")
    if backend is not None and not isinstance(backend, str):
        raise refuse("backend is not a string or null")
    for name in ("membership", "presence", "missing"):
        value = join.get(name)
        if not isinstance(value, list) or any(
            isinstance(rank, bool) or not isinstance(rank, int) for rank in value
        ):
            raise refuse(f"{name} is not a list of integers")
    per_rank_payload = join.get("per_rank")
    if not isinstance(per_rank_payload, dict):
        raise refuse("per_rank is not a JSON object")
    per_rank: dict[int, Any] = {}
    for rank_key, ref in per_rank_payload.items():
        if not isinstance(rank_key, str) or not rank_key.isdigit():
            raise refuse(f"per_rank key {rank_key!r} is not a rank string")
        if not isinstance(ref, dict):
            raise refuse(f"per_rank entry {rank_key} is not a JSON object")
        digests: dict[str, tuple[str, ...] | None] = {}
        for field in ("contribution_digests", "destination_digests"):
            value = ref.get(field)
            if value is not None and (
                not isinstance(value, list) or any(not isinstance(d, str) for d in value)
            ):
                raise refuse(f"per_rank entry {rank_key} {field} is not a list of strings or null")
            digests[field] = None if value is None else tuple(value)
        for field in ("rank", "boundary_index", "seq_abs"):
            value = ref.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise refuse(f"per_rank entry {rank_key} {field} is not an integer")
        group_rank = ref.get("group_rank")
        if group_rank is not None and (
            isinstance(group_rank, bool) or not isinstance(group_rank, int)
        ):
            raise refuse(f"per_rank entry {rank_key} group_rank is not an integer or null")
        for field in ("n_contribution_roles", "n_destination_roles"):
            value = ref.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise refuse(f"per_rank entry {rank_key} {field} is not a non-negative integer")
        op_labels = ref.get("op_labels_raw")
        if not isinstance(op_labels, list) or any(
            not isinstance(label, str) for label in op_labels
        ):
            raise refuse(f"per_rank entry {rank_key} op_labels_raw is not a list of strings")
        c10d_group_seq = ref.get("c10d_group_seq")
        if c10d_group_seq is not None and (
            isinstance(c10d_group_seq, bool) or not isinstance(c10d_group_seq, int)
        ):
            raise refuse(f"per_rank entry {rank_key} c10d_group_seq is not an integer or null")
        if not isinstance(ref.get("witness_policy"), str):
            raise refuse(f"per_rank entry {rank_key} witness_policy is not a string")
        per_rank[int(rank_key)] = PerRankRef(
            rank=ref["rank"],
            boundary_index=ref["boundary_index"],
            seq_abs=ref["seq_abs"],
            op_labels_raw=tuple(op_labels),
            group_rank=group_rank,
            witness_policy=ref["witness_policy"],
            n_contribution_roles=ref["n_contribution_roles"],
            n_destination_roles=ref["n_destination_roles"],
            contribution_digests=digests["contribution_digests"],
            destination_digests=digests["destination_digests"],
            c10d_group_seq=c10d_group_seq,
        )
    if not set(join["presence"]).issubset(per_rank):
        raise refuse("presence names a rank with no per_rank row")
    return join, per_rank


def _check_degraded_cache_coherence(cached: dict[str, Any], rederived: Any) -> None:
    """Lower-bound the cached verdicts against what could have been derived.

    Deep-hunt F1: with a member unparseable, the exact rederivation-equality
    tamper oracle is off, and the substitute monotone checks only refused
    structural conflicts and a DIVERGENT rederivation under a non-DIVERGENT
    cache -- they never checked the ATTESTATION direction, so forging the
    cached ``stored_value_status`` to ``attested_complete`` and corrupting one
    member upgraded an honest UNWITNESSED merge to attested-at-face-value.

    Three checks, all satisfied by construction for an honest cache:

    1. Verdict recomputation: the cached joins' consistencies, the merge value
       status, and the alignment are recomputed from the cached rows with the
       same engine derivations that produced them; any disagreement means the
       verdict fields were edited independently of the evidence rows.
    2. Witness recomputation per join runs on the cached per-rank digest rows,
       so an honest ``attested_*`` cache requires digests from every member of
       every attested join -- survivors included (the sharper lower bound: a
       cache claiming attestation over digestless survivor rows is impossible).
    3. Survivor grounding: every join rederived from the surviving cores must
       appear in the cache with byte-equal per-rank rows for the surviving
       ranks (references to lost ranks stay unprovable and untouched).
    """

    from ._engine import _STRUCTURAL_KINDS, _witness_consistency
    from ._enums import BoundaryConsistency, MergeAlignment, MergeValueStatus

    cached_joins = cached.get("joins")
    if not isinstance(cached_joins, list):
        raise _schema_refusal("descriptor derivation cache joins table is not a list")
    cached_findings = cached.get("findings")
    if not isinstance(cached_findings, list):
        raise _schema_refusal("descriptor derivation cache findings table is not a list")

    stored_alignment = _cached_verdict(cached, "stored_alignment", MergeAlignment)
    stored_value_status = _cached_verdict(cached, "stored_value_status", MergeValueStatus)

    # 1 + 2. Recompute every cached join's witness consistency from its own
    # recorded per-rank rows, then the merge-level verdicts from those.
    joins_by_key: dict[tuple[Any, ...], tuple[dict[str, Any], dict[int, Any]]] = {}
    consistencies: list[BoundaryConsistency] = []
    for index, join_payload in enumerate(cached_joins):
        join, per_rank = _cached_join_refs(join_payload, index)
        key = tuple(join["key"])
        if key in joins_by_key:
            raise _schema_refusal(f"descriptor cache join key {key} is duplicated")
        joins_by_key[key] = (join, per_rank)
        try:
            recorded = BoundaryConsistency(join.get("consistency"))
        except ValueError as exc:
            raise _schema_refusal(
                f"descriptor cache join {index} consistency "
                f"{join.get('consistency')!r} is outside the closed vocabulary"
            ) from exc
        recomputed = _witness_consistency(
            join["kind"],
            join["backend"],
            tuple(join["membership"]),
            tuple(join["presence"]),
            per_rank,
        )
        if recomputed is not recorded:
            raise _tamper(
                f"descriptor cache join {index} records consistency "
                f"{recorded.value!r} but its own per-rank witness rows derive "
                f"{recomputed.value!r}; the verdict was edited independently "
                "of the evidence rows"
            )
        consistencies.append(recorded)

    applicable = [c for c in consistencies if c is not BoundaryConsistency.NOT_APPLICABLE]
    attested = [c for c in applicable if c is BoundaryConsistency.ATTESTED]
    mismatched = [c for c in applicable if c is BoundaryConsistency.MISMATCHED]
    if mismatched:
        recomputed_status = MergeValueStatus.DIVERGENT
    elif applicable and len(attested) == len(applicable):
        recomputed_status = MergeValueStatus.ATTESTED_COMPLETE
    elif attested:
        recomputed_status = MergeValueStatus.ATTESTED_PARTIAL
    else:
        recomputed_status = MergeValueStatus.UNWITNESSED
    if recomputed_status is not stored_value_status:
        raise _tamper(
            f"descriptor cache claims stored_value_status "
            f"{stored_value_status.value!r} but its own join rows derive "
            f"{recomputed_status.value!r}; witness evidence is demote-only, so "
            "the cache was edited"
        )

    kinds: list[str] = []
    for index, finding in enumerate(cached_findings):
        if not isinstance(finding, dict) or not isinstance(finding.get("kind"), str):
            raise _schema_refusal(f"descriptor cache finding {index} has no kind")
        kinds.append(finding["kind"])
    if any(kind in _STRUCTURAL_KINDS for kind in kinds):
        recomputed_alignment = MergeAlignment.CONFLICTED
    elif any(kind == "presence_gap" for kind in kinds):
        recomputed_alignment = MergeAlignment.PARTIAL
    else:
        recomputed_alignment = MergeAlignment.ALIGNED
    if recomputed_alignment is not stored_alignment:
        raise _tamper(
            f"descriptor cache claims stored_alignment {stored_alignment.value!r} "
            f"but its own findings ledger derives {recomputed_alignment.value!r}"
        )

    # 3. Survivor grounding: the cache must contain every join the surviving
    # cores rederive, with byte-equal per-rank rows for the surviving ranks.
    for join in rederived.joins:
        cached_entry = joins_by_key.get(tuple(join.key))
        if cached_entry is None:
            raise _tamper(
                f"the surviving rank cores rederive join {join.key} but the "
                "descriptor cache never recorded it"
            )
        cached_join, cached_per_rank = cached_entry
        if (
            cached_join["kind"] != join.kind
            or tuple(cached_join["membership"]) != join.membership
            or cached_join["backend"] != join.backend
            or cached_join.get("reduce_op") != join.reduce_op
        ):
            raise _tamper(
                f"the descriptor cache disagrees with the surviving rank cores "
                f"about join {join.key} (kind/membership/backend/reduce_op)"
            )
        for rank, ref in join.per_rank.items():
            cached_ref = cached_per_rank.get(rank)
            if cached_ref is None:
                raise _tamper(
                    f"surviving rank {rank} presents join {join.key} but the "
                    "descriptor cache records no per-rank row for it"
                )
            if cached_ref != ref:
                raise _tamper(
                    f"the descriptor cache per-rank row for surviving rank {rank} "
                    f"at join {join.key} does not equal the row rederived from "
                    "its own core"
                )


def _resolve_member_path(root: Path, relative: str) -> Path:
    """Resolve a descriptor member path under the root, rejecting escapes."""

    candidate_path = Path(relative)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise _schema_refusal(f"member path {relative!r} escapes the artifact root")
    resolved = (root / candidate_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise _schema_refusal(f"member path {relative!r} escapes the artifact root") from exc
    if resolved == root.resolve():
        raise _schema_refusal(f"member path {relative!r} is self-referential")
    return resolved


def load_merged(path: str | Path) -> MergedTrace:
    """Load and REDERIVE a ``merged-directory`` artifact (4.3).

    Parameters
    ----------
    path:
        Artifact root directory.

    Returns
    -------
    MergedTrace
        The merged presenter, with ``load_degradations`` recording any rank
        core that no longer parses on this runtime (capping the effective
        alignment at ``partial``).

    Raises
    ------
    MergedArtifactError
        Typed schema refusal (closed vocabularies, unknown versions) or
        typed tamper refusal (checksum mismatch, tree-hash mismatch, or any
        inequality between the descriptor cache and the independent
        rederivation from the rank cores). Tamper is never a gap.
    """

    root = Path(path)
    manifest_path = root / "manifest.json"
    # Reject a symlinked manifest/descriptor before reading it: the tree-hash
    # and member guards already refuse symlinks, but the two root sidecars were
    # read through symlinks (arbitrary-path read / DoS), unlike the sibling
    # .tlspec loaders. A symlink here is never a legitimate merged artifact.
    if manifest_path.is_symlink():
        raise _schema_refusal(f"{root} manifest.json is a symlink")
    if not manifest_path.is_file():
        raise _schema_refusal(f"{root} has no manifest.json")
    try:
        # Bounded PATH read: ``read_text`` allocated the whole attacker-sized file
        # before the ceiling applied, so the byte limit was advisory only here.
        manifest = _json.read_bounded(manifest_path)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _schema_refusal(f"root manifest does not parse ({_bounded_exc(exc)})") from exc
    if not isinstance(manifest, dict):
        raise _schema_refusal("root manifest is not a JSON object")
    if manifest.get("bundle_format") != MERGED_BUNDLE_FORMAT:
        raise _schema_refusal(
            f"bundle_format {manifest.get('bundle_format')!r} is not {MERGED_BUNDLE_FORMAT!r}"
        )
    if manifest.get("tlspec_version") != MERGED_TLSPEC_VERSION:
        raise _schema_refusal(
            f"merged root tlspec_version {manifest.get('tlspec_version')!r} is "
            f"not supported by this runtime (expected {MERGED_TLSPEC_VERSION})"
        )

    descriptor_path = root / "merge" / "descriptor.json"
    if descriptor_path.is_symlink():
        raise _schema_refusal("merge/descriptor.json is a symlink")
    if not descriptor_path.is_file():
        raise _tamper("merge/descriptor.json is missing")
    # The descriptor's EXACT on-disk bytes are the checksum subject, so they must be
    # read raw -- but under the same ceiling, not via an unbounded ``read_bytes``.
    try:
        descriptor_bytes = _json.read_bytes_bounded(descriptor_path)
    except json.JSONDecodeError as exc:
        raise _schema_refusal(f"descriptor does not parse ({_bounded_exc(exc)})") from exc
    recorded_sha = manifest.get("descriptor_sha256")
    if hashlib.sha256(descriptor_bytes).hexdigest() != recorded_sha:
        raise _tamper("descriptor bytes do not match the root-manifest checksum")
    try:
        descriptor = _json.loads_bounded(descriptor_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _schema_refusal(f"descriptor does not parse ({_bounded_exc(exc)})") from exc
    if not isinstance(descriptor, dict):
        raise _schema_refusal("descriptor is not a JSON object")
    if descriptor.get("descriptor_kind") != MERGED_DESCRIPTOR_KIND:
        raise _schema_refusal(
            f"descriptor_kind {descriptor.get('descriptor_kind')!r} is not "
            f"{MERGED_DESCRIPTOR_KIND!r}"
        )
    if descriptor.get("schema_version") != MERGED_DESCRIPTOR_SCHEMA_VERSION:
        raise _schema_refusal(
            f"descriptor schema_version {descriptor.get('schema_version')!r} is "
            f"not supported (expected {MERGED_DESCRIPTOR_SCHEMA_VERSION})"
        )
    members = descriptor.get("members")
    if not isinstance(members, list) or not members:
        raise _schema_refusal("descriptor members table is absent or empty")

    manifest_members = manifest.get("members")
    if not isinstance(manifest_members, dict):
        raise _schema_refusal("root manifest members table is absent")

    # 1a. Schema: every member entry is validated against closed keys and exact
    # types BEFORE any value is used: a validly hashed descriptor with
    # `"members": [{}]` (or a non-mapping entry, a boolean rank, ...)
    # previously escaped the documented typed refusal as a raw
    # KeyError/TypeError (p2 R58 sol-R58-1).
    validated_members: dict[int, tuple[str, str]] = {}
    seen_paths: set[str] = set()
    for entry in members:
        if not isinstance(entry, dict):
            raise _schema_refusal("descriptor member entry is not a JSON object")
        if set(entry) != _MEMBER_ENTRY_KEYS:
            raise _schema_refusal(
                "descriptor member entry keys "
                f"{sorted(str(key) for key in entry)} are not the closed set "
                f"{sorted(_MEMBER_ENTRY_KEYS)}"
            )
        rank = entry["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise _schema_refusal(f"descriptor member rank {rank!r} is not a non-negative integer")
        if rank in validated_members:
            raise _schema_refusal(f"descriptor member rank {rank} is duplicated")
        if not isinstance(entry["path"], str):
            raise _schema_refusal(f"rank {rank} member path is not a string")
        # Defense in depth for the rank-identity binding below: two ranks that
        # name the SAME member directory would each hash the one honest core.
        # The load-time evidence.rank check catches the swap, but a duplicated
        # path is itself incoherent -- every rank core is its own directory.
        if entry["path"] in seen_paths:
            raise _schema_refusal(
                f"descriptor member path {entry['path']!r} is shared by more than "
                "one rank; every rank core is a distinct member directory"
            )
        seen_paths.add(entry["path"])
        recorded = entry["tree_sha256"]
        if (
            not isinstance(recorded, str)
            or len(recorded) != 64
            or any(char not in "0123456789abcdef" for char in recorded)
        ):
            raise _schema_refusal(f"rank {rank} tree_sha256 is not a lowercase hex SHA-256 digest")
        validated_members[rank] = (entry["path"], recorded)

    # 1b. Integrity: every member's canonical tree hash must match BOTH records.
    member_paths: dict[int, Path] = {}
    for rank, (relative, recorded) in validated_members.items():
        member_path = _resolve_member_path(root, relative)
        if not member_path.is_dir():
            raise _tamper(f"rank {rank} core {relative!r} is missing")
        manifest_recorded = manifest_members.get(str(rank))
        if manifest_recorded != recorded:
            raise _tamper(f"rank {rank} tree hash disagrees between descriptor and manifest")
        actual = tree_hash(member_path)
        if actual != recorded:
            raise _tamper(f"rank {rank} core bytes do not match the recorded tree hash")
        member_paths[rank] = member_path

    # 2. Independent rederivation from the parsable rank cores.
    from .._io.bundle import load as load_bundle

    evidence = {}
    handles: dict[int, _RankHandle] = {}
    load_degradations: list[str] = []
    for declared_rank, member_path in sorted(member_paths.items()):
        try:
            trace = load_bundle(member_path)
        except (MemoryError, OSError):
            # ENOMEM/EIO during member load is an ENVIRONMENT failure, not
            # evidence about the artifact: the member's bytes were just read
            # successfully by tree_hash above, so "no longer parses on this
            # runtime" would be an actively misleading description and would
            # silently cap the merge at partial (B2R4-17, the resource half of
            # A-R58-1). Resource errors surface to the caller raw; they never
            # enter the degradation channel.
            raise
        except Exception as exc:
            # A member bundle whose failure is a bundle-INTEGRITY signal -- a
            # guarded-unpickler denylist refusal or a corrupt/truncated pickle
            # stream -- is a tampered artifact, not environmental drift, and must
            # refuse typed exactly like the parses-but-invalid class below
            # (A-R58-1: the bare handler laundered a `pickle.UnpicklingError`
            # denylist refusal into the "no longer parses on this runtime"
            # channel, silently capping the merge at partial off the remaining
            # members while the adjacent handler refused the same tamper class).
            if _is_bundle_integrity_refusal(exc):
                raise _tamper(
                    f"rank {declared_rank} member core {str(member_path)!r} failed "
                    f"a bundle-integrity check ({_bounded_exc(exc)}); a guarded-unpickler denylist "
                    "refusal or corrupt pickle stream is a tampered artifact, never "
                    "a runtime degradation to be laundered into a partial merge"
                ) from exc
            # A member bundle that no longer LOADS on this runtime is a genuine
            # environmental degradation (torch/codec drift). It caps the
            # effective alignment at partial but is never a tamper.
            load_degradations.append(
                f"rank {declared_rank} core {str(member_path)!r} no longer parses on "
                f"this runtime: {_bounded_exc(exc)}"
            )
            continue
        # A member that loads as a bundle but is NOT a valid rank core (no
        # distributed evidence, malformed boundary journal) is a tampered
        # artifact -- a non-rank-core bundle dropped into a member slot -- not
        # an environmental degradation. Laundering it into the runtime-parse
        # channel silently caps the merge at partial instead of refusing
        # (b3-opus). Refuse typed.
        try:
            rank_evidence = extract_rank_evidence(trace, str(member_path))
        except MergeInputError as exc:
            raise _tamper(
                f"rank {declared_rank} member core loaded but carries no coherent "
                f"rank-core evidence ({_bounded_exc(exc)}); a member that parses yet is not a "
                "valid rank capture is a tampered artifact, never a runtime "
                "degradation"
            ) from exc
        # Rank-IDENTITY binding (b3-opus HIGH): the descriptor's member table
        # merely LABELS which rank each slot holds, but each core proves its OWN
        # global rank from its boundary records. Nothing tied the two, so one
        # honest core byte-duplicated across N member slots read back as N
        # distinct attesting ranks -- making ATTESTED_COMPLETE structurally
        # guaranteed and bypassing the R18-1 digest fix. The core's self-proven
        # rank is the authority; a slot/label disagreement is tamper.
        if rank_evidence.rank != declared_rank:
            raise _tamper(
                f"descriptor labels a member slot as rank {declared_rank} but the "
                f"core's own boundary records prove it is rank {rank_evidence.rank}; "
                "a core placed at the wrong slot (or one core duplicated across "
                "slots) cannot attest as multiple ranks"
            )
        evidence[declared_rank] = rank_evidence
        handles[declared_rank] = _RankHandle(declared_rank, trace=trace, path=str(member_path))

    if not evidence:
        raise _schema_refusal(
            "no rank core of this artifact parses on this runtime; nothing can "
            "be rederived. Load degradations: " + "; ".join(load_degradations)
        )

    cached = descriptor.get("derivation")
    if not isinstance(cached, dict):
        raise _schema_refusal("descriptor derivation cache is absent")
    # ``expected_ranks`` feeds straight into ``derive_merge`` (``int(r)`` over
    # every element); a forged non-iterable or non-integer element escaped as a
    # raw TypeError/ValueError instead of the documented typed refusal. Validate
    # against the closed shape (null or a list of non-negative ints) first.
    expected = cached.get("expected_ranks")
    if expected is not None and (
        not isinstance(expected, list)
        or any(isinstance(r, bool) or not isinstance(r, int) or r < 0 for r in expected)
    ):
        raise _schema_refusal(
            "descriptor derivation cache expected_ranks is not null or a list of "
            "non-negative integers"
        )
    rederived = derive_merge(evidence, expected)

    if not load_degradations:
        # 3. EXACT equality with the cache is required (any inequality = tamper).
        if canonical_json_bytes(rederived.to_payload()) != canonical_json_bytes(cached):
            raise _tamper(
                "the merge rederived from the rank cores does not equal the "
                "descriptor cache; the descriptor was edited or the cores were "
                "coherently rewritten"
            )
        derivation = rederived
    else:
        # Degraded environment: rederivation covers the parsable subset; the
        # STORED verdicts stay visible while the effective alignment is capped
        # at partial by the degradation ledger (3.2).
        from ._enums import MergeAlignment, MergeValueStatus

        stored_alignment = _cached_verdict(cached, "stored_alignment", MergeAlignment)
        stored_value_status = _cached_verdict(cached, "stored_value_status", MergeValueStatus)
        # Monotone coherence (R18-5): evidence is demote-only, so the cached
        # full-set verdicts can never be BETTER than what the surviving cores
        # prove. A byte-exact digest mismatch among survivors cannot have been
        # attested with more ranks present, and survivors that structurally
        # contradict each other could never have merged at all. Without this
        # cross-check, corrupting ONE member unparseable let an edited cache
        # present ATTESTED_COMPLETE over cores that rederive DIVERGENT.
        if rederived.structural_findings:
            raise _tamper(
                "the surviving rank cores structurally conflict with each "
                "other; no honest merge could have produced this artifact"
            )
        if (
            rederived.stored_value_status is MergeValueStatus.DIVERGENT
            and stored_value_status is not MergeValueStatus.DIVERGENT
        ):
            raise _tamper(
                "the surviving rank cores rederive a DIVERGENT value status "
                f"but the descriptor cache claims {stored_value_status.value!r}; "
                "witness evidence is demote-only, so the cache was edited"
            )
        # Attestation lower bound (deep-hunt F1): the two checks above only
        # guard the DIVERGENT/structural direction, so a forged cache could
        # still upgrade an honest UNWITNESSED merge to attested_* once one
        # member was made unparseable. Recompute the cached verdicts from the
        # cache's own evidence rows and ground them in the surviving cores.
        _check_degraded_cache_coherence(cached, rederived)
        derivation = replace(
            rederived,
            stored_alignment=stored_alignment,
            stored_value_status=stored_value_status,
        )

    merged = MergedTrace(derivation, handles, tuple(load_degradations))
    merged._source_path = str(root)
    return merged
