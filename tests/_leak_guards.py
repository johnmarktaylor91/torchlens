"""Leak tripwires: capture slots, owned threads, test patches and default draw outpaths.

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
from types import FunctionType, ModuleType
from typing import Any

import pytest
import torch

from torchlens import _state
from torchlens._data_classes.trace import Trace
from torchlens.bundle import Bundle


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


#: Functions whose code lives under this directory are test code. One installed
#: as a ``torchlens`` module attribute is a test patch of a TorchLens internal.
_TESTS_ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep

#: The internals test helpers wrap (pass doors, producer and ingest seams),
#: checked after every test; the whole-package scan runs after every module.
_WRAPPED_INTERNALS: tuple[tuple[str, str], ...] = (
    ("torchlens._capture.run_entry", "_run_resolved_request"),
    ("torchlens.intervention._replay", "_run_replay"),
    ("torchlens._postprocess", "materialize_from_events"),
    ("torchlens._postprocess._materialize", "materialize_from_events"),
    ("torchlens._postprocess._materialize", "ingest_op_records"),
    ("torchlens.backends.torch._ops_capture_records", "set_capture_producer_policy"),
    ("torchlens._ir.op_record_scatter", "scatter_record_to_cells"),
)


def _is_test_patch(value: object) -> bool:
    """Return whether ``value`` is a function defined in test code or a ``unittest.mock`` object."""

    kind = type(value)
    if kind is FunctionType:
        # The code's file, never ``__module__``: ``functools.wraps`` copies the
        # wrapped function's ``__module__`` onto the wrapper.
        filename = value.__code__.co_filename  # type: ignore[attr-defined]
        if not os.path.isabs(filename):
            filename = os.path.abspath(filename)
        return filename.startswith(_TESTS_ROOT)
    return kind.__module__ == "unittest.mock"


def _torchlens_modules() -> list[ModuleType]:
    """Return every loaded ``torchlens`` module."""

    return [
        module
        for name, module in list(sys.modules.items())
        if module is not None and (name == "torchlens" or name.startswith("torchlens."))
    ]


def leaked_internal_patches(
    names: tuple[tuple[str, str], ...] | None = None,
) -> list[tuple[ModuleType, str, object]]:
    """Return the ``torchlens`` module attributes currently bound to a test patch.

    ``names`` limits the check to those ``(module, attribute)`` pairs; without
    it every loaded ``torchlens`` module is scanned. A helper that wraps a
    TorchLens internal must restore it before its test ends: a wrapper left
    installed changes every later capture on the worker (an extra frame, a
    stale recorder) and makes results depend on test order.
    """

    if names is not None:
        pairs = [
            (sys.modules[module_name], attr)
            for module_name, attr in names
            if sys.modules.get(module_name) is not None
        ]
    else:
        pairs = [(module, attr) for module in _torchlens_modules() for attr in list(vars(module))]
    leaks = []
    for module, attr in pairs:
        value = vars(module).get(attr)
        if _is_test_patch(value):
            leaks.append((module, attr, value))
    return leaks


def _unwrapped(value: object) -> object | None:
    """Return the innermost non-test function under a ``functools.wraps`` chain, if any."""

    while _is_test_patch(value):
        value = getattr(value, "__wrapped__", None)
        if value is None:
            return None
    return value


def _fail_on_patches(leaks: list[tuple[ModuleType, str, object]]) -> None:
    """Unwrap what can be unwrapped (one leak fails one test, not the rest), then fail."""

    if not leaks:
        return
    for module, attr, value in leaks:
        original = _unwrapped(value)
        if original is not None:
            setattr(module, attr, original)
    named = ", ".join(f"{module.__name__}.{attr}" for module, attr, _ in leaks)
    pytest.fail(f"test code left TorchLens internals patched: {named}", pytrace=False)


@pytest.fixture(autouse=True)
def _internals_unpatched_after_test() -> Iterator[None]:
    """Patch tripwire: no test may leave a wrapped TorchLens internal behind.

    Patches made through ``monkeypatch`` are undone before this check runs.
    """

    yield
    _fail_on_patches(leaked_internal_patches(_WRAPPED_INTERNALS))


@pytest.fixture(autouse=True, scope="module")
def _torchlens_unpatched_after_module() -> Iterator[None]:
    """Patch tripwire over every loaded ``torchlens`` module, once per test module."""

    yield
    _fail_on_patches(leaked_internal_patches())


#: ``(owner, method_name, default_basename)`` for every ``Trace`` draw
#: entrypoint whose ``vis_outpath``-equivalent parameter defaults to a bare
#: relative basename. Graphviz resolves a relative ``vis_outpath`` against
#: the process cwd, not any per-trace scratch directory, so every entry here
#: is a repo-root collision risk for a test that omits the argument.
_DRAW_DEFAULT_OUTPATH_TARGETS: tuple[tuple[type, str, str], ...] = (
    (Trace, "draw", "modelgraph"),
    # A bare ``Bundle.draw()`` writes ``bundle_modelgraph.<format>`` into the cwd
    # (JMT drawing.3, Sun 1 Nov 2026), the same repo-root collision risk.
    (Bundle, "draw", "bundle_modelgraph"),
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

    ``Trace.draw`` (the one draw door; ``graph="backward"`` and
    ``graph="combined"`` default to ``backward_modelgraph`` and
    ``combined_modelgraph``) defaults its output-path argument to a bare
    relative basename (``"modelgraph"``). Graphviz resolves that
    relative path against the process's current working directory, which
    under xdist is the repo root shared by every worker: two tests that both
    omit the argument (about 21 files call ``draw()`` with
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
            # Unwrap decorators (Trace.draw's unknown-keyword teaching wrapper): the
            # defaults live on the function that binds the arguments.
            func = inspect.unwrap(getattr(owner, method_name))
            saved.append((func, func.__defaults__))
            func.__defaults__ = _defaults_with_replacement(func, "path", str(tmp_path / basename))
        yield
    finally:
        for func, defaults in reversed(saved):
            func.__defaults__ = defaults


_DISTRIBUTION_VALIDATION_AT_START = pytest.StashKey[bool]()


def pytest_configure(config: pytest.Config) -> None:
    """Record the ``torch.distributions`` validation default before collection."""

    from torch.distributions import Distribution

    config.stash[_DISTRIBUTION_VALIDATION_AT_START] = Distribution._validate_args


def pytest_collection_finish(session: pytest.Session) -> None:
    """Collection tripwire: importing test modules must not change distribution validation.

    ``torch.compile`` switches the process-wide default off, so a module that
    compiles at import silently changes the graph of every distribution-building
    model tested after it. The default is put back before failing so the report
    names the cause instead of a downstream oracle.
    """

    from torch.distributions import Distribution

    expected = session.config.stash.get(_DISTRIBUTION_VALIDATION_AT_START, None)
    if expected is None or Distribution._validate_args is expected:
        return
    Distribution.set_default_validate_args(expected)
    raise pytest.UsageError(
        "collecting the test modules changed the torch.distributions argument-validation "
        f"default from {expected} to {not expected}; a module probably calls torch.compile "
        "at import. Compile on first use inside a test instead."
    )
