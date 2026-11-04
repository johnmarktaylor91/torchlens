"""Which Python frames belong to TorchLens itself.

Capture walks the live call stack and must skip its own frames: the code-context
and call-site chain of an op are the frames *between* the user's ``forward`` and
the wrapped torch function, nothing of TorchLens's wrappers and recorders. The
only identity a frame offers for that is ``f_code.co_filename``, whose spelling
is whatever path the import system joined when the module was loaded: a
``sys.path`` entry of ``"."`` or ``"/repo/."`` leaves ``/./`` inside every
filename of the package, a relative entry leaves a relative one. The package
directory this module computes is normalised, so a bare prefix comparison on the
raw filename failed in those processes and TorchLens's frames leaked into every
op's code context. Both sides are normalised here, once per filename.
"""

from __future__ import annotations

import inspect
import os
from types import CodeType

#: The package directory with a trailing separator, normalised the same way as
#: every filename checked against it. The separator keeps sibling installs
#: (``torchlens_contrib/...``) out.
TORCHLENS_PACKAGE_PREFIX = (
    os.path.normcase(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) + os.sep
)

# Verdict per filename. Keyed by the filename string, never by the code object:
# CPython compares code objects structurally and ignores ``co_filename``, so two
# files with identical code (two empty ``__init__.py``) would share one verdict.
# Distinct filenames in a process are few, and a string's hash is cached, so the
# per-frame cost on the capture hot path is one dictionary lookup. The code
# context walk (``introspection._get_code_context``) reads this dictionary
# inline on its hot loop and falls back to ``is_torchlens_code`` on a miss, so
# this stays the single verdict store and that function the single rule.
_VERDICTS: dict[str, bool] = {}


def normalized_filename(filename: str) -> str:
    """Return ``filename`` spelled the way :data:`TORCHLENS_PACKAGE_PREFIX` is."""

    return os.path.normcase(os.path.abspath(filename))


def is_torchlens_code(code: CodeType) -> bool:
    """Whether ``code`` was compiled from a file inside the TorchLens package.

    Spelling-independent: ``/repo/./torchlens/x.py``, ``/repo/torchlens/x.py``
    and (relative to the process directory) ``torchlens/x.py`` all answer the
    same. Pseudo-filenames (``<string>``, ``<frozen ...>``) answer False.
    """

    filename = code.co_filename
    verdict = _VERDICTS.get(filename)
    if verdict is None:
        verdict = normalized_filename(filename).startswith(TORCHLENS_PACKAGE_PREFIX)
        _VERDICTS[filename] = verdict
    return verdict


def external_warning_stacklevel() -> int:
    """Return the ``warnings.warn`` stacklevel that lands in the caller's user code.

    Counts frames from the function that calls this helper outward until the first
    frame compiled outside the TorchLens package, so a warning raised deep inside a
    door (``do`` -> staging -> expansion) still names the user's line.
    """

    level = 1
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None and is_torchlens_code(frame.f_code):
            level += 1
            frame = frame.f_back
    finally:
        del frame
    return level
