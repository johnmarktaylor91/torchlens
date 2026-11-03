"""Hermetic deterministic counters for hot-path tests.

A counter test asserts that a code path performs N (often zero) calls of some
kind. Counting by patching a process-global hook (``builtins.__import__``, a
class method) also counts calls made by everything else that runs in the
window: finalizers that the cyclic garbage collector fires mid-walk for
garbage earlier tests left behind (an unclosed-file ``ResourceWarning`` alone
runs import statements), and other threads. Such a test then depends on what
ran before it in the same worker process.

The helpers here make the measured window independent of process history:

* :func:`quiet_gc` collects pending garbage before the window and keeps the
  cyclic collector off inside it (reference-count frees caused by the code
  under test still run, as they would in production);
* :func:`count_package_imports` counts only import statements executed on the
  current thread by a frame whose code lives in the ``torchlens`` package;
* :func:`thread_counting` wraps a patched method so only calls on the current
  thread count.
"""

from __future__ import annotations

import builtins
import gc
import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torchlens

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(torchlens.__file__)) + os.sep


@contextmanager
def quiet_gc() -> Iterator[None]:
    """Flush pending cyclic garbage, then keep the cyclic collector off."""

    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class PackageImportCounter:
    """Import statements run by ``torchlens`` frames on the owning thread."""

    def __init__(self) -> None:
        self.sites: list[tuple[str, int, str]] = []

    @property
    def calls(self) -> int:
        """Number of counted import statements."""

        return len(self.sites)


@contextmanager
def count_package_imports() -> Iterator[PackageImportCounter]:
    """Count import statements executed by package code on this thread.

    The window runs under :func:`quiet_gc`. Each counted call records the
    importing frame's file, line and module name, so a failure names the site.
    """

    counter = PackageImportCounter()
    owner = threading.get_ident()
    real_import = builtins.__import__

    def counting(name: str, *args: Any, **kwargs: Any) -> Any:
        if threading.get_ident() == owner:
            frame = sys._getframe(1)
            filename = frame.f_code.co_filename
            if filename.startswith(_PACKAGE_ROOT):
                counter.sites.append((filename[len(_PACKAGE_ROOT) :], frame.f_lineno, name))
        return real_import(name, *args, **kwargs)

    with quiet_gc():
        builtins.__import__ = counting
        try:
            yield counter
        finally:
            builtins.__import__ = real_import


def thread_counting(
    real: Callable[..., Any], counts: dict[str, int], key: str
) -> Callable[..., Any]:
    """Wrap ``real`` so calls on the creating thread increment ``counts[key]``."""

    owner = threading.get_ident()

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if threading.get_ident() == owner:
            counts[key] = counts.get(key, 0) + 1
        return real(*args, **kwargs)

    return wrapper
