"""Sibling staging folders of a saved artifact: discovery, failure settlement, stale-load refusal.

Every saver stages into ``<name>.tmp.<hex>`` beside its target and renames the folder into
place. A staging folder still beside the target afterwards is a save that failed (marked
``PARTIAL``, with a ``REASON.txt``) or that was killed or is still writing (unmarked).
:func:`refuse_stale_load` makes ``tl.load`` refuse a target whose staging folder is newer
than the installed copy: the copy on disk is then the older one, and serving it as a complete
artifact would hide the failed save.

The installed copy's age is the target directory's own mtime. Every saver installs and
restores by a same-parent ``rename``, which keeps the directory's mtime (the time its last
entry was written) and bumps only its ctime. ctime would therefore make a pre-overwrite copy
restored after a failed save look newer than that save; a manifest ``created_at`` is
format-specific and may be absent or normalized for deterministic saves.
"""

from __future__ import annotations

import glob
import shutil
import warnings
from pathlib import Path

from ..errors._base import TorchLensWarning
from .format_errors import ArtifactError

PARTIAL_SENTINEL = "PARTIAL"
REASON_SENTINEL = "REASON.txt"
#: Belt bound on the persisted PARTIAL failure reason, and on the reason quoted at load.
MAX_PARTIAL_REASON_CHARS = 200

_ACCEPT_OLDER = "Re-save, or run tl.io.cleanup_tmp(path) to accept the older copy"


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
    target leaves the OLDER copy there, so the staging folder stays as the record
    :func:`refuse_stale_load` reads. Otherwise nothing at the target can be mistaken for the
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


def _newer_save_error(target: Path, newer: list[Path]) -> ArtifactError:
    """Build the ``newer_save_unfinished`` refusal for the newest staging folder in ``newer``."""

    staging = newer[-1]
    partial = (staging / PARTIAL_SENTINEL).exists()
    reason = _read_reason(staging) if partial else None
    state = (
        "marked PARTIAL: the save failed"
        if partial
        else "not marked PARTIAL: the saving process was killed or is still writing"
    )
    detail = f"; recorded failure: {reason}" if reason else ""
    others = f" ({len(newer) - 1} more newer staging folders beside it)" if len(newer) > 1 else ""
    cleanup = (
        "tl.io.cleanup_tmp(path) removes it"
        if partial
        else "it is not marked PARTIAL, so tl.io.cleanup_tmp(path, force=True) is needed "
        "(first make sure no save to this path is still running)"
    )
    return ArtifactError(
        f"A newer save to this path failed or is still running; the file on disk is the "
        f"older copy. Staging folder {staging} is {state}{detail}{others}. "
        f"{_ACCEPT_OLDER}: {cleanup}.",
        code="newer_save_unfinished",
        remedy=f"{_ACCEPT_OLDER} ({cleanup})",
        path=str(target),
        staging=str(staging),
        partial=partial,
        reason=reason,
    )


def refuse_stale_load(target: Path, *, member: bool = False) -> None:
    """Refuse loading ``target`` while a newer save to it failed or has not finished.

    Runs once, on the top-level path a caller passed to a load entry; members inside a
    loaded artifact are not re-checked. A missing or non-directory target returns
    silently so the loader's own refusal (not found, symlinked path) stays unchanged. A
    staging folder older than the installed copy (an earlier failure followed by a
    successful save) does not refuse; it warns once with the cleanup remedy.

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
        ``newer_save_unfinished`` when a staging folder at least as new as the installed
        copy sits beside it; a symlinked staging entry also refuses.
    """

    if member or target.is_symlink() or not target.is_dir():
        return
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
    # as complete is not.
    newer = [staging for mtime, staging in found if mtime >= installed_ns]
    older = [staging for mtime, staging in found if mtime < installed_ns]
    if newer:
        raise _newer_save_error(target, newer)
    if older:
        warnings.warn(
            f"Leftover staging folder {older[-1]} beside {target} is older than the copy "
            "being loaded (an earlier save failed before a later one succeeded); run "
            "tl.io.cleanup_tmp(path) to remove it.",
            TorchLensWarning,
            stacklevel=3,
        )
