"""Refuse a ``--base`` the surface diff cannot be trusted against.

A diff reports the names head adds since base. Two bases make it silently report nothing:

- a ref the repository does not have (a worker checkout shipped without remote refs has
  no ``origin/...``), which must fail instead of being guessed around;
- a commit inside the branch under review (an ancestor of head, but not of the branch's
  merge target), which hides every name added between the fork point and that commit.

So a base must be an ancestor of head and reachable from a merge target: ``main`` or a
``*-train`` branch in ``refs/heads`` or ``refs/remotes``, or the ref given as ``--target``.
``--target none`` skips the target check (and tolerates inventories without a commit).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

NONE = "none"
_TARGET_SUFFIX = "-train"
_TARGET_NAMES = ("main",)


class BaseError(Exception):
    """A base (or target) that the diff must not run against; the message says why."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in the repository, capturing text output."""

    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


def rev(repo: Path, ref: str) -> str | None:
    """Return the commit a ref names, or None when the repository lacks it.

    Parameters
    ----------
    repo
        Repository root.
    ref
        Any revision expression (``ref:`` prefix tolerated).

    Returns
    -------
    str | None
        The full commit SHA.
    """

    name = ref[4:] if ref.startswith("ref:") else ref
    done = _git(repo, "rev-parse", "--verify", "--quiet", f"{name}^{{commit}}")
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def resolve(repo: Path, ref: str, role: str = "base") -> str:
    """Return the commit a ref names, or raise with a hint.

    Parameters
    ----------
    repo
        Repository root.
    ref
        Revision expression.
    role
        Word for the error message (``base``, ``target``, ``head``).

    Returns
    -------
    str
        The full commit SHA.

    Raises
    ------
    BaseError
        When the repository does not have the ref.
    """

    commit = rev(repo, ref)
    if commit is None:
        hint = ""
        if "/" in ref:
            hint = (
                " (a checkout shipped without remote refs has no origin/ names: pass the"
                " commit SHA, or fetch the ref first)"
            )
        raise BaseError(f"{role} ref {ref!r} is unknown in {repo}{hint}")
    return commit


def is_ancestor(repo: Path, commit: str, descendant: str) -> bool:
    """Return whether ``commit`` is reachable from ``descendant``."""

    return _git(repo, "merge-base", "--is-ancestor", commit, descendant).returncode == 0


def target_refs(repo: Path) -> list[str]:
    """Return the merge-target refs the repository has (``main`` and ``*-train`` branches).

    Parameters
    ----------
    repo
        Repository root.

    Returns
    -------
    list[str]
        Short ref names (``main``, ``origin/wip/mp-train``), sorted.
    """

    done = _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes")
    found = []
    for name in done.stdout.split():
        leaf = name.rsplit("/", 1)[-1]
        if leaf in _TARGET_NAMES or leaf.endswith(_TARGET_SUFFIX):
            found.append(name)
    return sorted(found)


def _fork_points(repo: Path, head: str, targets: list[str]) -> list[str]:
    """Return ``<target>: <merge-base>`` lines for the error message."""

    lines = []
    for target in targets:
        done = _git(repo, "merge-base", head, target)
        if done.returncode == 0 and done.stdout.strip():
            lines.append(f"  {target}: {done.stdout.strip()[:12]}")
    return lines


def verify(repo: Path, base: str, head: str, target: str | None) -> str:
    """Check that a base is a trustworthy diff base for a head.

    Parameters
    ----------
    repo
        Repository root.
    base, head
        Full commit SHAs.
    target
        ``--target``: a ref the base must be reachable from, ``none`` to skip that check,
        or None to look for ``main`` and ``*-train`` refs.

    Returns
    -------
    str
        A one-line summary for the log (``base: <sha12> ...``).

    Raises
    ------
    BaseError
        When the base is not an ancestor of the head, is not reachable from any merge
        target, or no target can be found.
    """

    short = base[:12]
    if base != head and not is_ancestor(repo, base, head):
        raise BaseError(
            f"base {short} is not an ancestor of head {head[:12]}: diff it against the merge"
            " base (git merge-base <target> <head>)"
        )
    if target == NONE:
        return f"base: {short} is an ancestor of head (target check skipped)"
    if target is not None:
        tip = resolve(repo, target, "target")
        if not is_ancestor(repo, base, tip):
            raise BaseError(
                f"base {short} is not reachable from --target {target} ({tip[:12]}): it is a"
                " branch-internal commit, so names added before it would be invisible;"
                f" use the merge base {_merge_base(repo, head, tip)}"
            )
        return f"base: {short} is reachable from --target {target}"
    targets = target_refs(repo)
    if not targets:
        raise BaseError(
            "no merge-target ref (main or *-train) in this repository: pass --target <ref>"
            " naming the branch this head merges into"
        )
    reached = [name for name in targets if is_ancestor(repo, base, name)]
    if not reached:
        forks = "\n".join(_fork_points(repo, head, targets))
        raise BaseError(
            f"base {short} is not reachable from any merge target ({', '.join(targets)}):"
            " it is a branch-internal commit, so names added before it would be invisible."
            f" Fork points of head {head[:12]}:\n{forks}"
        )
    return f"base: {short} is reachable from {reached[0]}"


def _merge_base(repo: Path, head: str, tip: str) -> str:
    """Return the short merge base of two commits, or ``?``."""

    done = _git(repo, "merge-base", head, tip)
    return done.stdout.strip()[:12] if done.returncode == 0 else "?"
