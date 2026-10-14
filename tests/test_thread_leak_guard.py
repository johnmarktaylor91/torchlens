"""Regression guard for the thread-leak tripwire in ``tests/conftest.py``.

A thread a test leaves running can exit inside a later test's capture window,
where ``capture_escape_guard`` sees ``threading.active_count()`` change and marks
that unrelated capture ``owner_thread_tripwire_changed``. The autouse
``_threads_stopped_after_test`` fixture fails the leaking test instead. These
tests pin the pieces it relies on: start sites are recorded, pool workers are
attributed to the code that submitted to the pool, and a left-running thread is
reported until it is joined.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

_CONFTEST = Path(__file__).resolve().parent / "conftest.py"


def _root_conftest(request: pytest.FixtureRequest) -> ModuleType:
    for plugin in request.config.pluginmanager.get_plugins():
        if getattr(plugin, "__file__", None) == str(_CONFTEST):
            return plugin
    raise AssertionError("tests/conftest.py is not registered as a plugin")


def test_thread_start_site_is_recorded_for_test_threads() -> None:
    """A thread started here records this file as its start site."""

    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, name="leak-guard-plain", daemon=True)
    thread.start()
    try:
        assert thread._torchlens_test_start_site.startswith(f"{__file__}:")  # type: ignore[attr-defined]
    finally:
        stop.set()
        thread.join(timeout=30)


def test_pool_worker_is_attributed_to_the_submitter() -> None:
    """Stdlib frames (``submit`` -> ``_adjust_thread_count``) are skipped."""

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(int).result()
        (worker,) = pool._threads
        assert worker._torchlens_test_start_site.startswith(f"{__file__}:")  # type: ignore[attr-defined]


def test_left_running_thread_is_reported_until_joined(request: pytest.FixtureRequest) -> None:
    """``owned_threads_left`` names a live test-started thread and drops it once joined."""

    conftest = _root_conftest(request)
    before = set(threading.enumerate())
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, name="leak-guard-left", daemon=True)
    thread.start()
    try:
        assert conftest.owned_threads_left(before) == [thread]
    finally:
        stop.set()
        thread.join(timeout=30)
    assert conftest.owned_threads_left(before) == []
