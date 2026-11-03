"""Decide whether a preview backend's per-change CI leg runs for a diff.

The ``backend-protocol`` job in ``.github/workflows/pr.yml`` runs one leg per
preview backend (tf, jax, mlx, paddle, tinygrad). A leg runs when the change
touches that backend's own files or the shared backend surface; otherwise its
steps are skipped and the leg reports success, so untouched backends cost
nothing on a pull request. The nightly matrix still runs every backend
unconditionally.

A path triggers a leg when it is:

- shared: a top-level module of ``torchlens/backends/`` (the registry, the
  protocol, the shared finalize and constants rules), anything under
  ``torchlens/backends/_protocols/``, ``tests/conformance/``,
  ``torchlens/compat/`` or ``tests/backend_conformance/``, ``pyproject.toml``,
  the workflow itself, this script, or the executed-floor check;
- the backend's own: anything under ``torchlens/backends/<backend>/``, its
  page ``docs/backends/<backend>.md``, or a test path that names one of the
  backend's tokens as a whole word (``tests/test_tf_eager.py`` names ``tf``).

Usage::

    python scripts/backend_ci_changes.py --backend tf --base HEAD^1
    python scripts/backend_ci_changes.py --backend tf --all

It prints ``run=true`` or ``run=false`` (append it to ``$GITHUB_OUTPUT``) and
the reason on stderr. When the diff cannot be computed it prints
``run=true``: an unknown diff runs the leg rather than skipping it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable

#: Path tokens that name each preview backend in test file paths.
BACKEND_TOKENS: dict[str, tuple[str, ...]] = {
    "tf": ("tf", "tensorflow", "keras"),
    "jax": ("jax", "flax", "equinox"),
    "mlx": ("mlx",),
    "paddle": ("paddle",),
    "tinygrad": ("tinygrad",),
}

#: Paths whose change runs every backend leg (directory prefixes end in "/").
SHARED_PREFIXES: tuple[str, ...] = (
    "torchlens/backends/_protocols/",
    "tests/conformance/",
    "torchlens/compat/",
    "tests/backend_conformance/",
)
SHARED_FILES: frozenset[str] = frozenset(
    {
        "pyproject.toml",
        ".github/workflows/pr.yml",
        "scripts/backend_ci_changes.py",
        "scripts/check_ci_executed_tests.py",
    }
)
_BACKENDS_TOP_LEVEL = re.compile(r"^torchlens/backends/[^/]+\.py$")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def is_shared_path(path: str) -> bool:
    """Whether a change to ``path`` runs every backend leg.

    Parameters
    ----------
    path:
        Repository-relative path with forward slashes.

    Returns
    -------
    bool
        ``True`` for the shared backend surface.
    """

    if path in SHARED_FILES or _BACKENDS_TOP_LEVEL.match(path):
        return True
    return path.startswith(SHARED_PREFIXES)


def is_backend_path(path: str, backend: str) -> bool:
    """Whether ``path`` belongs to ``backend`` alone.

    Parameters
    ----------
    path:
        Repository-relative path with forward slashes.
    backend:
        A key of :data:`BACKEND_TOKENS`.

    Returns
    -------
    bool
        ``True`` for the backend's package, docs page or a test naming it.
    """

    if path.startswith(f"torchlens/backends/{backend}/"):
        return True
    if path == f"docs/backends/{backend}.md":
        return True
    if not path.startswith("tests/"):
        return False
    words = set(_TOKEN_SPLIT.split(path.lower()))
    return any(token in words for token in BACKEND_TOKENS[backend])


def leg_should_run(backend: str, paths: Iterable[str]) -> tuple[bool, str]:
    """Decide whether ``backend``'s leg runs for the changed ``paths``.

    Parameters
    ----------
    backend:
        A key of :data:`BACKEND_TOKENS`.
    paths:
        Changed repository-relative paths.

    Returns
    -------
    tuple[bool, str]
        The decision and a one-line reason naming the first triggering path.

    Raises
    ------
    KeyError
        If ``backend`` is not a preview backend.
    """

    if backend not in BACKEND_TOKENS:
        raise KeyError(
            f"unknown preview backend {backend!r}; expected one of {sorted(BACKEND_TOKENS)}"
        )
    for path in paths:
        if is_shared_path(path):
            return True, f"shared backend path changed: {path}"
        if is_backend_path(path, backend):
            return True, f"{backend} path changed: {path}"
    return False, f"no {backend} or shared backend path changed"


def changed_paths(base: str) -> list[str] | None:
    """Paths changed between ``base`` and ``HEAD``, or ``None`` when git fails.

    Parameters
    ----------
    base:
        Any git revision (``HEAD^1`` for a pull request's merge commit).

    Returns
    -------
    list[str] | None
        The changed paths; ``None`` when the diff cannot be computed.
    """

    result = subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    """Print the leg decision for one backend.

    Parameters
    ----------
    argv:
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Always 0; an unknown backend exits 2 through argparse.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", required=True, choices=sorted(BACKEND_TOKENS))
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--base", help="git revision to diff HEAD against")
    scope.add_argument("--all", action="store_true", help="run the leg unconditionally")
    args = parser.parse_args(argv)
    if args.all:
        run, reason = True, "unconditional run requested"
    else:
        paths = changed_paths(args.base)
        if paths is None:
            run, reason = True, f"could not diff against {args.base}; running the leg"
        else:
            run, reason = leg_should_run(args.backend, paths)
    print(f"run={'true' if run else 'false'}")
    print(f"{args.backend}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
