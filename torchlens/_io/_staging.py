"""Sibling staging folders of a saved artifact: discovery, failure settlement, stale-load refusal.

Every saver stages into ``<name>.tmp.<hex>`` beside its target and renames the folder into
place. A staging folder still beside the target afterwards is a save that failed (marked
``PARTIAL``, with a ``REASON.txt``) or that was killed or is still writing (unmarked).
:func:`refuse_stale_load` makes ``tl.load`` refuse a target whose staging folder is newer
than the installed copy: the copy on disk is then the older one, and serving it as a complete
artifact would hide the failed save.

A staging folder stays on the machine that ran the save. So before an overwrite stages,
:func:`mark_superseding_save` also writes a small record (:data:`SUPERSEDED_RECORD`) INSIDE
the installed copy. A completed overwrite replaces the copy and the record with it; a failed
one restores the older copy with the record still in it (rewritten with the scrubbed failure
type); a killed process leaves it in place. ``rsync`` or ``shutil.copytree`` carries the record
with the data, so ``tl.load`` refuses the copy on any machine, whatever its mtimes, until
``tl.io.cleanup_tmp(path)`` removes it. No manifest lists the record, and no digest covers it.

The installed copy's age is the target directory's own mtime. Every saver installs and
restores by a same-parent ``rename``, which keeps the directory's mtime (the time its last
entry was written) and bumps only its ctime. ctime would therefore make a pre-overwrite copy
restored after a failed save look newer than that save; a manifest ``created_at`` is
format-specific and may be absent or normalized for deterministic saves.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import stat
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from ..errors._base import TorchLensWarning
from ._durability import fsync_dir
from ._json import loads_bounded
from .format_errors import ArtifactError

PARTIAL_SENTINEL = "PARTIAL"
REASON_SENTINEL = "REASON.txt"
#: Belt bound on the persisted PARTIAL failure reason, and on the reason quoted at load.
MAX_PARTIAL_REASON_CHARS = 200

_ACCEPT_OLDER = "Re-save, or run tl.io.cleanup_tmp(path) to accept the older copy"

#: The record a superseding save leaves inside the installed copy it is replacing.
SUPERSEDED_RECORD = "NEWER_SAVE_UNFINISHED.json"
_RECORD_KIND = "torchlens.newer_save_unfinished"
#: Belt bound on the bytes read from a record at load.
_MAX_RECORD_BYTES = 4096


def is_superseded_record_name(name: str) -> bool:
    """Return whether ``name`` is the record or a temp file left while writing it."""

    return name == SUPERSEDED_RECORD or name.startswith(f"{SUPERSEDED_RECORD}.tmp.")


def _record_path(target: Path) -> Path:
    """Return the record path inside the installed copy ``target``."""

    return target / SUPERSEDED_RECORD


def _write_record(target: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as the record inside ``target``: temp file, fsync, rename, fsync."""

    temp = target / f"{SUPERSEDED_RECORD}.tmp.{uuid.uuid4().hex}"
    data = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, _record_path(target))
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    fsync_dir(target)


def mark_superseding_save(target: Path, *, overwrite: bool) -> None:
    """Record inside the installed copy ``target`` that a save replacing it has started.

    Call it before the overwrite stages anything. Writing a record is the save's first
    step, so a failure here (read-only copy, full disk) fails the save before anything
    else changed; the older copy stays loadable and unmarked.

    Parameters
    ----------
    target:
        The save's target path.
    overwrite:
        The save's ``overwrite`` argument; nothing is written unless it is set and an
        installed copy (a directory) is at ``target``.
    """

    if not overwrite or target.is_symlink() or not target.is_dir():
        return
    started_at = datetime.now(timezone.utc).isoformat()
    _write_record(target, {"kind": _RECORD_KIND, "state": "started", "started_at": started_at})


def begin_staging(target: Path, staging: Path, *, overwrite: bool) -> None:
    """Record an overwrite inside the installed copy, then create ``staging`` (and parents).

    The record (:func:`mark_superseding_save`) is durable before the first staging folder
    exists, so no crash point leaves a staged overwrite whose older copy is unmarked.
    """

    mark_superseding_save(target, overwrite=overwrite)
    staging.mkdir(parents=True)


def record_failed_save(target: Path, reason: str) -> None:
    """Best-effort rewrite the record in ``target`` as failed, keeping its start time.

    Only a record the save itself wrote is rewritten; a target without one (a new copy
    installed before the failure, or no record written) is left alone.
    """

    if target.is_symlink() or not target.is_dir():
        return
    record = _read_record(target)
    if record is None:
        return
    started_at = record.get("started_at")
    payload: dict[str, Any] = {
        "kind": _RECORD_KIND,
        "state": "failed",
        "started_at": started_at if isinstance(started_at, str) else None,
        "reason": reason[:MAX_PARTIAL_REASON_CHARS],
    }
    try:
        _write_record(target, payload)
    except OSError:
        return


def _read_record(target: Path) -> dict[str, Any] | None:
    """Return the record in ``target`` (``{}`` when unreadable), or ``None`` when absent.

    Any entry under the record's name counts, so a damaged, emptied or replaced record
    still refuses; only a regular file is opened, and only its first bytes are read.
    """

    record_path = _record_path(target)
    try:
        status = os.lstat(record_path)
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    if not stat.S_ISREG(status.st_mode):
        return {}
    try:
        with record_path.open("rb") as handle:
            raw = handle.read(_MAX_RECORD_BYTES)
        data = loads_bounded(raw.decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def sweep_staging_dirs(target: Path, *, force: bool) -> list[Path]:
    """Remove the ``PARTIAL`` staging folders beside ``target`` (every one with ``force``).

    A folder without ``PARTIAL`` may be a save still writing, so it stays (with a warning)
    unless ``force`` is set. Returns the removed folders.
    """

    removed: list[Path] = []
    for candidate in sibling_staging_dirs(target):
        if not candidate.is_dir():
            continue
        if force or (candidate / PARTIAL_SENTINEL).exists():
            shutil.rmtree(candidate)
            removed.append(candidate)
            continue
        warnings.warn(
            f"Leaving non-partial temp directory {candidate} in place; pass force=True to remove it.",
            TorchLensWarning,
            stacklevel=3,
        )
    return removed


def clear_record_when_unstaged(target: Path) -> list[Path]:
    """Remove the record from ``target`` once no staging folder is left beside it.

    A kept non-``PARTIAL`` folder may be a save in progress, so the record stays with it.
    """

    if any(candidate.is_dir() for candidate in sibling_staging_dirs(target)):
        return []
    return clear_superseded_record(target)


def clear_superseded_record(target: Path) -> list[Path]:
    """Remove the record and any temp file left while writing it from ``target``.

    Parameters
    ----------
    target:
        The installed copy.

    Returns
    -------
    list[Path]
        The removed entries.
    """

    if target.is_symlink() or not target.is_dir():
        return []
    removed: list[Path] = []
    for entry in sorted(target.iterdir()):
        if is_superseded_record_name(entry.name) and not entry.is_dir():
            entry.unlink()
            removed.append(entry)
    if removed:
        fsync_dir(target)
    return removed


def sibling_staging_dirs(target: Path) -> list[Path]:
    """Return the ``<name>.tmp.*`` entries beside ``target``, refusing a symlinked one.

    The basename is data, not a pattern: escaping it stops a legal name such as ``job*``
    from matching the staging folder of a sibling artifact (``jobX.tmp.<hex>``).

    Parameters
    ----------
    target:
        The artifact path itself (what ``save``/``load`` receive), not its parent.

    Returns
    -------
    list[Path]
        Matching sibling entries in name order; non-directories are left to the caller.

    Raises
    ------
    ArtifactError
        If a matching entry is a symlink.
    """

    candidates = sorted(target.parent.glob(f"{glob.escape(target.name)}.tmp.*"))
    for candidate in candidates:
        if candidate.is_symlink():
            raise ArtifactError(
                f"Refusing to use symlink temp directory {candidate} beside {target}; "
                "remove it by hand."
            )
    return candidates


def mark_partial(tmp_path: Path, *, reason: str | None = None) -> None:
    """Best-effort mark a staging folder ``PARTIAL`` after a failed save.

    Parameters
    ----------
    tmp_path:
        Staging folder path.
    reason:
        Optional failure reason to persist beside the sentinel. Callers pass a scrubbed
        value (the exception TYPE name, not ``str(exc)``, whose message can carry object
        reprs, paths or values); it is length-bounded here as a belt.
    """

    try:
        if tmp_path.exists():
            (tmp_path / PARTIAL_SENTINEL).write_text("", encoding="utf-8")
            if reason is not None:
                bounded = reason[:MAX_PARTIAL_REASON_CHARS]
                (tmp_path / REASON_SENTINEL).write_text(bounded, encoding="utf-8")
    except OSError:
        return


def settle_failed_staging(
    tmp_path: Path, target: Path, *, overwrite: bool, primary: BaseException
) -> None:
    """Settle a failed save's staging folder: keep it ``PARTIAL`` when an older copy survives.

    Call it after any backup restore. An overwriting save that leaves an artifact at the
    target leaves the OLDER copy there, so the staging folder stays as the evidence
    :func:`refuse_stale_load` reads, and the record inside that copy (when the save wrote
    one) is rewritten with the failure type. Otherwise nothing at the target can be mistaken for the
    failed save's result (a failed first save, or a non-overwriting save that lost a race
    to another writer), and the staging folder is removed as before.

    Parameters
    ----------
    tmp_path:
        Staging folder of the failed save.
    target:
        The save's target path.
    overwrite:
        The save's ``overwrite`` argument.
    primary:
        The exception that failed the save; its type name becomes the ``REASON.txt``.
    """

    if overwrite and target.exists():
        mark_partial(tmp_path, reason=type(primary).__name__)
        record_failed_save(target, type(primary).__name__)
    elif tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


def _read_reason(staging: Path) -> str | None:
    """Return the bounded ``REASON.txt`` text of ``staging``, or ``None`` when unreadable."""

    try:
        with (staging / REASON_SENTINEL).open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(MAX_PARTIAL_REASON_CHARS + 1)
    except OSError:
        return None
    text = " ".join(text.split())
    if len(text) > MAX_PARTIAL_REASON_CHARS:
        text = text[:MAX_PARTIAL_REASON_CHARS] + "..."
    return text or None


def _record_clause(record: dict[str, Any]) -> tuple[str, bool, str | None]:
    """Describe ``record`` for the refusal: (clause, failed, bounded reason)."""

    failed = record.get("state") == "failed"
    raw_reason = record.get("reason")
    reason = None
    if failed and isinstance(raw_reason, str):
        reason = " ".join(raw_reason.split())[:MAX_PARTIAL_REASON_CHARS] or None
    state = (
        "the save failed"
        if failed
        else "the save did not finish: its process was killed or is still writing"
    )
    detail = f"; recorded failure: {reason}" if reason else ""
    return f"The copy carries a {SUPERSEDED_RECORD} record: {state}{detail}.", failed, reason


def _refuse_newer_save(target: Path, newer: list[Path], record: dict[str, Any] | None) -> NoReturn:
    """Raise ``newer_save_unfinished`` for the record in ``target`` and/or ``newer`` folders."""

    staging = newer[-1] if newer else None
    clauses: list[str] = []
    partial = False
    reason: str | None = None
    unmarked = False
    if record is not None:
        clause, partial, reason = _record_clause(record)
        clauses.append(clause)
    if staging is not None:
        staged_partial = (staging / PARTIAL_SENTINEL).exists()
        unmarked = not staged_partial
        staged_reason = _read_reason(staging) if staged_partial else None
        state = (
            "marked PARTIAL: the save failed"
            if staged_partial
            else "not marked PARTIAL: the saving process was killed or is still writing"
        )
        detail = f"; recorded failure: {staged_reason}" if staged_reason else ""
        others = f" ({len(newer) - 1} more staging folders beside it)" if len(newer) > 1 else ""
        clauses.append(f"Staging folder {staging} is {state}{detail}{others}.")
        partial = partial or staged_partial
        reason = reason or staged_reason
    cleanup = (
        "it is not marked PARTIAL, so tl.io.cleanup_tmp(path, force=True) is needed "
        "(first make sure no save to this path is still running)"
        if unmarked
        else "tl.io.cleanup_tmp(path) removes it"
    )
    raise ArtifactError(
        f"A newer save to this path failed or is still running; the file on disk is the "
        f"older copy. {' '.join(clauses)} {_ACCEPT_OLDER}: {cleanup}.",
        code="newer_save_unfinished",
        remedy=f"{_ACCEPT_OLDER} ({cleanup})",
        path=str(target),
        staging=None if staging is None else str(staging),
        record=None if record is None else str(_record_path(target)),
        partial=partial,
        reason=reason,
    )


def refuse_stale_load(target: Path, *, member: bool = False) -> None:
    """Refuse loading ``target`` while a newer save to it failed or has not finished.

    Runs once, on the top-level path a caller passed to a load entry; members inside a
    loaded artifact are not re-checked. A missing or non-directory target returns
    silently so the loader's own refusal (not found, symlinked path) stays unchanged.
    Two kinds of evidence refuse: the :data:`SUPERSEDED_RECORD` inside the copy (it
    travels with the copy, so it refuses on any machine and needs no mtimes), and a
    staging folder beside the copy at least as new as it. A staging folder older than
    the installed copy (an earlier failure followed by a successful save) does not
    refuse; it warns once with the cleanup remedy.

    Parameters
    ----------
    target:
        The artifact directory path passed to the load entry.
    member:
        Whether ``target`` is a member loaded from inside another artifact; members are
        not re-checked, so this returns at once.

    Raises
    ------
    ArtifactError
        ``newer_save_unfinished`` when the copy carries the record or a staging folder at
        least as new as the installed copy sits beside it; a symlinked staging entry also
        refuses.
    """

    if member or target.is_symlink() or not target.is_dir():
        return
    record = _read_record(target)
    installed_ns = target.stat().st_mtime_ns
    found: list[tuple[int, Path]] = []
    for staging in sibling_staging_dirs(target):
        try:
            if staging.is_dir():
                found.append((staging.stat().st_mtime_ns, staging))
        except OSError:  # renamed into place by a save finishing right now
            continue
    found.sort()
    # A tie counts as newer: refusing a current copy is recoverable, serving a stale one
    # as complete is not. Rewriting the record after a failure bumps the copy's mtime
    # past its staging folder, so with a record every staging folder belongs to it.
    newer = [staging for mtime, staging in found if record is not None or mtime >= installed_ns]
    older = [staging for mtime, staging in found if record is None and mtime < installed_ns]
    if newer or record is not None:
        _refuse_newer_save(target, newer, record)
    if older:
        warnings.warn(
            TorchLensWarning(
                f"Leftover staging folder {older[-1]} beside {target} is older than the copy "
                "being loaded (an earlier save failed before a later one succeeded). "
                "Remedy: run tl.io.cleanup_tmp(path) to remove it",
                code="older_staging_leftover",
                path=str(target),
                staging=str(older[-1]),
            ),
            stacklevel=3,
        )
