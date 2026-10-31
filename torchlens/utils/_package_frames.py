"""Which Python frames belong to TorchLens itself.

Capture walks the live call stack and must skip its own frames: the code-context
and call-site chain of an op are the frames *between* the user's ``forward`` and
the wrapped torch function, nothing of TorchLens's wrappers and recorders. The
only identity a frame offers for that is ``f_code.co_filename``, whose spelling
is whatever path the import system joined when the module was loaded: a
``sys.path`` entry of ``"."`` or ``"/repo/."`` leaves ``/./`` inside every
filename of the package, a relative entry leaves a relative one. The package
directory this module computes is normalised, so a bare prefix comparison on the
raw filename failed in those processes and TorchLens's frames leaked into the
chain; the leak then merged distinct call sites (the chain collapse on the
wrapper's repeated code object dropped the frames in between). Both sides are
normalised here, once per code object.
"""

from __future__ import annotations

import os
from types import CodeType
from weakref import WeakKeyDictionary

#: The package directory with a trailing separator, normalised the same way as
#: every filename checked against it. The separator keeps sibling installs
#: (``torchlens_contrib/...``) out.
TORCHLENS_PACKAGE_PREFIX = (
    os.path.normcase(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) + os.sep
)

# Verdict per code object. Code objects are immutable and weak-referenceable, so
# the memo never pins run-time code (``exec``, reloaded modules) and the per-frame
# cost on the capture hot path is one dictionary lookup.
_VERDICTS: WeakKeyDictionary[CodeType, bool] = WeakKeyDictionary()


def normalized_filename(filename: str) -> str:
    """Return ``filename`` spelled the way :data:`TORCHLENS_PACKAGE_PREFIX` is."""

    return os.path.normcase(os.path.abspath(filename))


def is_torchlens_code(code: CodeType) -> bool:
    """Whether ``code`` was compiled from a file inside the TorchLens package.

    Spelling-independent: ``/repo/./torchlens/x.py``, ``/repo/torchlens/x.py``
    and (relative to the process directory) ``torchlens/x.py`` all answer the
    same. Pseudo-filenames (``<string>``, ``<frozen ...>``) answer False.
    """

    verdict = _VERDICTS.get(code)
    if verdict is None:
        verdict = normalized_filename(code.co_filename).startswith(TORCHLENS_PACKAGE_PREFIX)
        _VERDICTS[code] = verdict
    return verdict
