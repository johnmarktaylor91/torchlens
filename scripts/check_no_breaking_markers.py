#!/usr/bin/env python3
"""
Hard-reject commit messages or push payloads that contain semantic-release
major-bump triggers (Conventional Commits ``!`` markers or ``BREAKING CHANGE:``
footers).

TorchLens stays on the 2.x family per the locked project release policy.
The PyPI 1.0.0 and 2.0.0 slots have already been burned by accidental major
bumps; the 3.0.0 slot was nearly burned a third time on 2026-05-01 (rescued
only by an unrelated workflow bug). This script makes the failure mode
mechanically impossible by scanning every commit message at commit time
(``commit-msg`` stage) and every outgoing commit at push time (``pre-push``
stage), and exiting non-zero if any major-bump trigger is present.

Override (use ONLY when the maintainer explicitly authorizes a major bump in this turn):

    TORCHLENS_ALLOW_MAJOR_BUMP=1 git commit ...
    TORCHLENS_ALLOW_MAJOR_BUMP=1 git push ...

Usage::

    # commit-msg stage (pre-commit framework passes the message-file path)
    scripts/check_no_breaking_markers.py --commit-msg <file>

    # pre-push stage (pre-commit framework passes "<remote> <url>" via argv,
    # then "<local_ref> <local_sha> <remote_ref> <remote_sha>" lines on stdin)
    scripts/check_no_breaking_markers.py --pre-push

Exit codes:
    0  -- no major-bump triggers found
    1  -- a major-bump trigger was found (commit/push blocked)
    2  -- usage / argument error
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

# Conventional-Commits major-bump trigger pattern: any conventional type
# (lowercase letters, optional scope) followed by a literal "!" before the
# colon. Examples that match:
#   feat!: ...
#   feat(api)!: ...
#   fix(scope-name)!: ...
#   chore!: ...
TYPE_BANG_RE = re.compile(r"^[a-z][a-z0-9_-]*(?:\([^)\n]+\))?!:", re.MULTILINE)

# Footer-style triggers. Conventional Commits accepts both "BREAKING CHANGE"
# and "BREAKING-CHANGE" in the footer (per spec); some tooling also recognises
# "BREAKING:" alone. Reject all three forms regardless of position.
FOOTER_RES = (
    re.compile(r"^BREAKING CHANGE:", re.MULTILINE),
    re.compile(r"^BREAKING-CHANGE:", re.MULTILINE),
    re.compile(r"^BREAKING:", re.MULTILINE),
)

OVERRIDE_ENV = "TORCHLENS_ALLOW_MAJOR_BUMP"

ERROR_BANNER = """
==============================================================================
 BLOCKED: semantic-release major-bump trigger detected
==============================================================================
 TorchLens stays on the 2.x family per the locked policy. Accidentally
 publishing a major bump permanently burns a PyPI version slot.

 Triggers found:
{triggers}

 Source location:
   {location}

 To fix:
   - Remove the "!" after the conventional-commit type
     (e.g. "feat!:" -> "feat:"), and
   - Remove any "BREAKING CHANGE:" / "BREAKING-CHANGE:" / "BREAKING:" footer.
   - Document the breaking change in human-readable prose in the commit body
     instead. semantic-release will not bump major.

 If you ACTUALLY mean to cut a major release this turn (extremely rare;
 requires the maintainer's explicit, in-turn authorisation), re-run the same git command
 with the override:
   {override}=1 <your git command>

 Background: see the project release-policy notes tracked by the maintainer.
==============================================================================
""".rstrip()


def _find_triggers(text: str) -> list[str]:
    """Return a list of human-readable trigger descriptions found in ``text``."""

    hits: list[str] = []
    for match in TYPE_BANG_RE.finditer(text):
        hits.append(f'  - Conventional-Commits "!" marker: {match.group().rstrip(":")}')
    for footer_re in FOOTER_RES:
        for match in footer_re.finditer(text):
            hits.append(f"  - Footer marker: {match.group().rstrip(':')}")
    return hits


def _override_active() -> bool:
    # Exact "1" only, matching the golden mutation-flag discipline (grind r5,
    # b10 R86 probe): the former truthy parse authorized a major bump on
    # OVERRIDE=FALSE, =no, or any templated junk — values a user sets to
    # DISABLE the override — while the notice claimed "=1 active".
    return os.environ.get(OVERRIDE_ENV, "").strip() == "1"


def _emit_block(triggers: Iterable[str], location: str) -> None:
    print(
        ERROR_BANNER.format(
            triggers="\n".join(triggers),
            location=location,
            override=OVERRIDE_ENV,
        ),
        file=sys.stderr,
    )


def _check_text(text: str, location: str) -> int:
    triggers = _find_triggers(text)
    if not triggers:
        return 0
    if _override_active():
        print(
            f"[check_no_breaking_markers] {OVERRIDE_ENV}=1 active; allowing "
            f"major-bump trigger in {location}.",
            file=sys.stderr,
        )
        for trigger in triggers:
            print(f"  - {trigger.lstrip(' -')}", file=sys.stderr)
        return 0
    _emit_block(triggers, location)
    return 1


def _check_commit_msg(path: str) -> int:
    msg_path = Path(path)
    try:
        text = msg_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        print(f"[check_no_breaking_markers] commit-msg file not found: {path}", file=sys.stderr)
        return 2
    # Strip comment lines (git uses "#" for comments in commit-msg files).
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return _check_text(body, location=str(msg_path))


def _push_records() -> list[list[str]]:
    """Collect "<local_ref> <local_sha> <remote_ref> <remote_sha>" records.

    Raw git hands the refspec lines on stdin. The pre-commit FRAMEWORK, however,
    consumes that stdin itself and re-executes hooks with EMPTY stdin, exporting
    ``PRE_COMMIT_TO_REF`` (local sha) / ``PRE_COMMIT_FROM_REF`` (remote sha)
    instead -- so an stdin-only reader silently scans NOTHING and passes under
    the framework (grind r3, R61/B27: the pre-push layer was green-but-inert).
    Read stdin first, then fall back to the framework's env contract.
    """

    records = [parts for line in sys.stdin if len(parts := line.strip().split()) == 4]
    if not records:
        to_ref = os.environ.get("PRE_COMMIT_TO_REF", "")
        from_ref = os.environ.get("PRE_COMMIT_FROM_REF", "")
        if to_ref and from_ref:
            records = [["(pre-commit framework push)", to_ref, "(remote)", from_ref]]
    if not records:
        # Ref-less remote (e.g. first push to an empty repository): the
        # framework exports the branch names but NO FROM_REF/TO_REF at all
        # (grind r4, b9-opus R70r4-F1: this branch used to return [] and the
        # hook printed "Passed" having scanned NOTHING -- a BREAKING CHANGE
        # commit landed). Synthesize a new-branch record from the local
        # branch so the merge-base scan path runs over the outgoing commits.
        local_branch = os.environ.get("PRE_COMMIT_LOCAL_BRANCH", "")
        if local_branch:
            records = [[local_branch, local_branch, "(remote)", "0" * 40]]
    return records


def _check_pre_push() -> int:
    """Scan every outgoing commit message in the push payload."""

    rc = 0
    records = _push_records()
    if not records:
        # FAIL CLOSED: raw git always hands refspec lines on stdin and the
        # framework always exports at least the local branch, so "no records"
        # means the push range could not be determined -- refusing to scan is
        # a usage error, never a pass (the fail-open half of R70r4-F1).
        print(
            "[check_no_breaking_markers] could not determine the push range: "
            "no refspecs on stdin and no PRE_COMMIT_FROM_REF/PRE_COMMIT_TO_REF/"
            "PRE_COMMIT_LOCAL_BRANCH in the environment. Refusing to pass "
            "without scanning any commits.",
            file=sys.stderr,
        )
        return 2
    for parts in records:
        local_ref, local_sha, _remote_ref, remote_sha = parts
        if local_sha == "0000000000000000000000000000000000000000":
            # Branch deletion -- nothing to scan.
            continue
        if remote_sha == "0000000000000000000000000000000000000000":
            # New branch on remote: scan all reachable commits not on any other
            # remote branch. Conservative: scan local_sha back to merge-base
            # with origin/main if possible, else just scan the tip.
            try:
                base = subprocess.check_output(
                    ["git", "merge-base", local_sha, "origin/main"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                rev_range = f"{base}..{local_sha}"
            except subprocess.CalledProcessError:
                rev_range = local_sha
        else:
            rev_range = f"{remote_sha}..{local_sha}"
        try:
            log = subprocess.check_output(
                ["git", "log", "--format=%H%x00%B%x00%x00", rev_range],
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"[check_no_breaking_markers] failed to enumerate commits in {rev_range}: {exc}",
                file=sys.stderr,
            )
            rc = max(rc, 2)
            continue
        for record in log.split("\x00\x00"):
            record = record.strip("\x00 \n")
            if not record:
                continue
            try:
                sha, body = record.split("\x00", 1)
            except ValueError:
                continue
            location = f"commit {sha[:12]} (ref {local_ref})"
            commit_rc = _check_text(body, location=location)
            rc = max(rc, commit_rc)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--commit-msg",
        metavar="FILE",
        help="Path to the commit message file (commit-msg stage).",
    )
    group.add_argument(
        "--pre-push",
        action="store_true",
        help="Read pre-push refspecs from stdin and scan outgoing commits.",
    )
    args = parser.parse_args(argv)

    if args.commit_msg:
        return _check_commit_msg(args.commit_msg)
    return _check_pre_push()


if __name__ == "__main__":
    sys.exit(main())
