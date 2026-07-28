"""Normalize configured executable paths without destroying virtualenv semantics.

Two independent call sites — the strict doctor's wrapper probe and the campaign config's
clean-environment command resolution — each reached for ``Path.resolve()`` to turn a
configured executable into an absolute path. That is wrong for exactly one reason, and it
cost two separate live failures before the cause was identified:

A virtualenv's ``bin/python`` is a symlink chain ending at the base interpreter, and CPython
derives ``sys.prefix`` from *the path it was invoked by*, not from the resolved target. It
finds ``pyvenv.cfg`` next to the invoking path. Resolving the symlink therefore silently
swaps the configured venv for the base environment and its site-packages, and the failure
surfaces far away as a missing third-party import.

Both failures looked like something else:

* the doctor reported ``missing version receipts ['checker']`` for a correctly configured
  wrapper, because only that lane imports a module needing ``jsonschema`` at load time;
* the driver died mid-run with ``ModuleNotFoundError: No module named 'jsonschema'`` raised
  from inside the checker subprocess.

The operator runbook configures all three wrapper commands as
``<clone>/.venv-crawler/bin/python`` by design, so the configured interpreter is the one that
must be executed. This module exists so that rule lives in exactly one place.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def normalize_executable(value: str, *, cwd: Optional[Path] = None) -> Optional[Path]:
    """Return an absolute executable path with symlinks left intact.

    Parameters
    ----------
    value:
        Configured executable token: absolute, relative-with-parent, or a bare name to be
        looked up on ``PATH``.
    cwd:
        Base for a relative token carrying a parent component.

    Returns
    -------
    Path | None
        Absolute path to the executable exactly as configured, or ``None`` when it does not
        resolve to an existing file.

    Notes
    -----
    ``os.path.abspath`` normalizes ``.`` and ``..`` **without** following the final symlink,
    which is precisely the behaviour required: a venv interpreter keeps pointing at its own
    ``pyvenv.cfg`` rather than collapsing into the base installation.
    """

    token = Path(value).expanduser()
    if token.is_absolute():
        candidate = Path(os.path.abspath(token))
    elif token.parent != Path("."):
        base = Path.cwd() if cwd is None else cwd
        candidate = Path(os.path.abspath(base / token))
    else:
        found = shutil.which(value)
        if found is None:
            return None
        candidate = Path(os.path.abspath(found))
    return candidate if candidate.is_file() else None
