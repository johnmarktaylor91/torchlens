"""Leak tripwires: capture slots, owned threads and default draw outpaths.

Registered by ``tests/conftest.py`` through ``pytest_plugins``.
"""

from __future__ import annotations

import inspect
import os
import sys
import sysconfig
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from torchlens import _state
from torchlens._data_classes.trace import Trace


def _global_hook_and_mode_depths() -> tuple[int, int, int]:
    """Return the global module-hook counts and the torch-function mode depth."""

    from torch.nn.modules import module as module_mod

    return (
        len(module_mod._global_forward_hooks),
        len(module_mod._global_forward_pre_hooks),
        torch._C._len_torch_function_stack(),
    )


def _leaked_capture_slots() -> list[str]:
    """Return the names of TorchLens process-global capture slots still held."""

    held = {
        "_active_trace": _state._active_trace is not None,
        "_active_hook_plan": _state._active_hook_plan is not None,
        "_active_intervention_spec": _state._active_intervention_spec is not None,
        "_capture_reserved_by": _state._capture_reserved_by is not None,
        "_logging_enabled": bool(_state._logging_enabled),
    }
    return [name for name, is_held in held.items() if is_held]


@pytest.fixture(autouse=True)
def _capture_state_released_after_test() -> Iterator[None]:
    """State-leak tripwire: every test must leave the capture globals as it found them.

    A finished (or failed) TorchLens call must release every capture slot and
    leave no global module hook or torch-function mode behind; a leak would
    silently arm the next test's model. The slots are reset after reporting so
    one leak fails one test instead of cascading.
    """

    before = _global_hook_and_mode_depths()
    yield
    leaked = _leaked_capture_slots()
    after = _global_hook_and_mode_depths()
    if leaked:
        _state._active_trace = None
        _state._active_hook_plan = None
        _state._active_intervention_spec = None
        _state._capture_reserved_by = None
        _state._logging_enabled = False
    problems = [f"capture slot still held: {name}" for name in leaked]
    if after != before:
        problems.append(
            "global forward hooks / pre-hooks / torch-function modes changed "
            f"from {before} to {after}"
        )
    if problems:
        pytest.fail("; ".join(problems), pytrace=False)


#: Source roots whose thread starts a test must stop before it ends. A thread
#: started by test code or by TorchLens and left running can exit inside a LATER
#: test's capture window; ``capture_escape_guard`` then sees
#: ``threading.active_count()`` change and marks that unrelated capture
#: ``owner_thread_tripwire_changed``. Threads a third-party library starts for
#: itself (tqdm's monitor, inductor's compile pool) are process-lifetime
#: singletons owned by that library, not by the test that first touched them.
_JOIN_OWNED_ROOTS: tuple[str, ...] = tuple(
    str(Path(__file__).resolve().parent.parent / name) + os.sep for name in ("tests", "torchlens")
)
_STDLIB_ROOTS: tuple[str, ...] = tuple(
    sorted({sysconfig.get_paths()[key] + os.sep for key in ("stdlib", "platstdlib")})
)
_ORIGINAL_THREAD_START = threading.Thread.start
_THREAD_EXIT_GRACE_S = 2.0  # total join wait for threads still alive at teardown


def thread_start_site(frame: Any) -> str | None:
    """Return ``path:line`` of the innermost non-stdlib frame from ``frame`` outward.

    The stdlib frames between the caller and ``Thread.start`` (a
    ``ThreadPoolExecutor.submit``, an asyncio executor) are skipped so a pool
    worker is attributed to whoever submitted to the pool.
    """

    while frame is not None:
        filename = frame.f_code.co_filename
        in_stdlib = filename.startswith("<") or (
            filename.startswith(_STDLIB_ROOTS) and "-packages" not in filename
        )
        if not in_stdlib:
            return f"{filename}:{frame.f_lineno}"
        frame = frame.f_back
    return None


def _recording_thread_start(self: threading.Thread) -> None:
    """``Thread.start`` that records where the thread was started from."""

    self._torchlens_test_start_site = thread_start_site(sys._getframe(1))  # type: ignore[attr-defined]
    _ORIGINAL_THREAD_START(self)


def owned_threads_left(before: set[threading.Thread]) -> list[threading.Thread]:
    """Return live threads absent from ``before`` that tests or TorchLens started."""

    return [
        thread
        for thread in threading.enumerate()
        if thread not in before
        and thread.is_alive()
        and (getattr(thread, "_torchlens_test_start_site", None) or "").startswith(
            _JOIN_OWNED_ROOTS
        )
    ]


@pytest.fixture(autouse=True, scope="session")
def _record_thread_start_sites() -> Iterator[None]:
    """Record every ``Thread.start`` call site for ``_threads_stopped_after_test``."""

    threading.Thread.start = _recording_thread_start  # type: ignore[method-assign]
    try:
        yield
    finally:
        threading.Thread.start = _ORIGINAL_THREAD_START  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _threads_stopped_after_test() -> Iterator[None]:
    """Thread-leak tripwire: a test stops every thread it or TorchLens started.

    Autouse fixtures tear down after the test's own fixtures, so a pool or
    loop fixture has already joined its threads here. A thread still alive
    would otherwise exit during some later test's capture and fail THAT test
    (``tests/test_w051_nondeterminism_foreign_join.py``'s flusher replay read
    ``unverifiable`` this way under xdist); failing here names the leaker.
    """

    before = set(threading.enumerate())
    yield
    leaked = owned_threads_left(before)
    if leaked:
        # A thread the test signalled to stop may still be returning; give it a short
        # shared grace so the tripwire does not race a thread that is already exiting.
        deadline = time.monotonic() + _THREAD_EXIT_GRACE_S
        for thread in leaked:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        leaked = owned_threads_left(before)
    if leaked:
        root = str(Path(__file__).resolve().parent.parent) + os.sep
        detail = "; ".join(
            f"{thread.name!r} started at {thread._torchlens_test_start_site.removeprefix(root)}"  # type: ignore[attr-defined]
            for thread in leaked
        )
        pytest.fail(
            f"test left {len(leaked)} thread(s) running: {detail}. Join or shut down every "
            "thread, pool and stream writer the test starts before it returns; a thread "
            "that exits during a later test's capture changes the thread count there and "
            "makes that capture unverified.",
            pytrace=False,
        )


#: ``(owner, method_name, default_basename)`` for every ``Trace`` draw
#: entrypoint whose ``vis_outpath``-equivalent parameter defaults to a bare
#: relative basename. Graphviz resolves a relative ``vis_outpath`` against
#: the process cwd, not any per-trace scratch directory, so every entry here
#: is a repo-root collision risk for a test that omits the argument.
_DRAW_DEFAULT_OUTPATH_TARGETS: tuple[tuple[type, str, str], ...] = (
    (Trace, "draw", "modelgraph"),
    (Trace, "draw_backward", "backward_modelgraph"),
    (Trace, "draw_combined", "combined_modelgraph"),
    (Trace, "render_dagua_graph", "graph.gv"),
)


def _defaults_with_replacement(func: Any, param_name: str, value: object) -> tuple[object, ...]:
    """Return ``func.__defaults__`` with one positional parameter's default replaced.

    Parameters
    ----------
    func:
        Plain function whose trailing positional parameters carry defaults.
    param_name:
        Name of the positional-or-keyword parameter to replace.
    value:
        Replacement default value.

    Returns
    -------
    tuple[object, ...]
        New ``__defaults__`` tuple, aligned the same way Python aligns it:
        against the trailing positional parameters.
    """

    positional = [
        p
        for p in inspect.signature(func).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    defaults = list(func.__defaults__ or ())
    first_defaulted = len(positional) - len(defaults)
    target_index = [p.name for p in positional].index(param_name) - first_defaulted
    defaults[target_index] = value
    return tuple(defaults)


@pytest.fixture(autouse=True)
def _draw_default_outpath_in_tmp_path(tmp_path: Path) -> Iterator[None]:
    """Redirect every ``vis_outpath``-defaulting draw entrypoint into ``tmp_path``.

    ``Trace.draw`` / ``draw_backward`` / ``draw_combined`` /
    ``render_dagua_graph`` default their output-path argument to a bare
    relative basename (``"modelgraph"``, ``"backward_modelgraph"``,
    ``"combined_modelgraph"``, ``"graph.gv"``). Graphviz resolves that
    relative path against the process's current working directory, which
    under xdist is the repo root shared by every worker: two tests that both
    omit the argument (about 21 files call ``draw(vis_save_only=True)`` with
    no ``vis_outpath=``) raced to write and read the same file ("dot: can't
    open .../modelgraph", nightly fast tier, 2026-10-03).

    A blanket ``monkeypatch.chdir`` would fix the same race but would also
    change how every OTHER relative path in a test resolves (fixture
    corpora, cache dirs), which is unproven safe suite-wide; patching only
    the draw-family defaults is the narrower fix with the same effect.
    Explicit ``vis_outpath=``/``container_path=`` callers are untouched --
    this only changes what a test gets when it supplies nothing.
    ``tests/test_draw_default_outpath_repo_root_guard.py`` is the regression
    test for this fixture.

    The defaults are swapped and restored by hand rather than through the
    shared ``monkeypatch`` fixture: requesting ``monkeypatch`` from an autouse
    fixture instantiates it before the later autouse fixtures, so a test's
    own ``monkeypatch`` undo would then run after their teardown (it made
    ``_restore_distributed_arming`` disarm a test's monkeypatched state).
    """

    saved: list[tuple[Any, tuple[object, ...] | None]] = []
    try:
        for owner, method_name, basename in _DRAW_DEFAULT_OUTPATH_TARGETS:
            func = getattr(owner, method_name)
            saved.append((func, func.__defaults__))
            func.__defaults__ = _defaults_with_replacement(func, "path", str(tmp_path / basename))
        yield
    finally:
        for func, defaults in reversed(saved):
            func.__defaults__ = defaults
