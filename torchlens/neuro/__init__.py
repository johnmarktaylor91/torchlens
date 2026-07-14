"""Extras-gated neuroscience namespace with no public objects yet.

Import time stays inert: this module must NOT import ``rsatoolbox`` /
``brainscore_core`` (the ``neuro`` extra's foreign third-party dependencies) as
a side effect of ``import torchlens.neuro``. A bare package import can happen
incidentally -- e.g. a portable ``.tlspec`` bundle's metadata unpickler
resolving a pickled global whose module path names ``torchlens.neuro`` -- and
an eager import-time dependency check would run that foreign code with no
trust opt-in. The extras check is therefore deferred to first ATTRIBUTE ACCESS
via module ``__getattr__`` (PEP 562), preserving the original "clear ImportError
naming the missing extra" contract at first USE instead of at import time.
"""

from __future__ import annotations

import importlib
from typing import Any

_REQUIRED_DEPS = ("rsatoolbox", "brainscore_core")

__all__: list[str] = []


def _check_required_deps() -> None:
    """Raise a clear ``ImportError`` if the ``neuro`` extra's deps are missing.

    Raises
    ------
    ImportError
        Naming every missing dependency, with the install hint.
    """

    missing_deps: list[str] = []
    for dep in _REQUIRED_DEPS:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing_deps.append(dep)

    if missing_deps:
        missing = ", ".join(missing_deps)
        raise ImportError(
            "torchlens.neuro requires extra: install with "
            f"`pip install torchlens[neuro]`. Missing deps: {missing}"
        )


def __getattr__(name: str) -> Any:
    """Gate attribute access behind the deferred extras check.

    Parameters
    ----------
    name:
        Requested module attribute name.

    Returns
    -------
    Any
        Never returns; ``torchlens.neuro`` exports no public objects yet.

    Raises
    ------
    ImportError
        If the ``neuro`` extra's dependencies are not installed.
    AttributeError
        If the dependencies are installed but ``name`` is not a real attribute.
    """

    _check_required_deps()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
