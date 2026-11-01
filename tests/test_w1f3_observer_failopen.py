"""W1_F3: invisible host-escape observers must fail CLOSED (round7 B1/B2 class).

An observer that was never installed must never be mistaken for proof that no invisible
escape occurred. Two layers of coverage:

* FULL-INVENTORY structural tripwires over ``_observe_invisible_host_escapes``: every
  install-loop ``continue`` and every restore handler must mark
  ``_HOST_ESCAPE_OBSERVER_FAILED``, except a CLOSED two-arm allowlist of classified-benign
  skips (the storage row filter and the r67 feature-absent classification); the three
  scan-based families must fail closed on an empty scan.
* BEHAVIORAL version-drift fixtures: an absent observer member, a failed restore, an
  unwrappable wrap-required storage member, an empty storage scan, and an MRO-inherited
  (``torch._C.TensorBase``) descriptor each drive the runtime to the fail-closed mark --
  or, for the MRO case, to a genuinely installed observer.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterator
from typing import Any

import pytest
import torch

import torchlens.backends.torch._completeness_cross_thread
import torchlens.backends.torch._completeness_finalize
import torchlens.backends.torch._completeness_patches
import torchlens.backends.torch._completeness_storage
from torchlens.backends.torch import completeness_witness as cw

# The ONLY install-loop skip arms that are classified benign, keyed by the exact unparsed
# guard expression. Row filter: a non-wrap-required disposition row is not an observer at
# all. Feature-absent: an accessor that does not exist on this torch build cannot be called
# through this class (r67 C3 documented classification). Everything else must fail closed.
_BENIGN_INSTALL_SKIP_GUARDS = frozenset(
    {
        "disposition not in _STORAGE_WRAPPED_DISPOSITIONS or member == 'data_ptr'",
        "descriptor is None",
    }
)

# Scan-based observer families: an EMPTY enumeration is version-drift uncertainty, never
# proof that the op/handle surface disappeared.
_EMPTY_SCAN_GUARDS = frozenset(
    {
        "not _ops_call_classes",
        "not _private_c_callables",
        "not _STORAGE_RAW_POINTER_TARGETS()",
    }
)

_MIN_INSTALL_LOOPS = 12
_MIN_RESTORE_HANDLERS = 11


class _Trace:
    """Weakrefable trace stand-in for the fail-closed weak sets."""


def _fresh_state() -> tuple[_Trace, Any]:
    """Build an isolated witness state around a weakrefable dummy trace.

    Returns
    -------
    tuple[_Trace, Any]
        The dummy trace and its ``_WitnessState``.
    """

    trace = _Trace()
    return trace, cw._WitnessState(trace=trace, owner_thread_id=0, guard_pass_index=1)


def _observer_function_ast() -> ast.FunctionDef:
    """Parse ``_observe_invisible_host_escapes`` into an AST function definition.

    Returns
    -------
    ast.FunctionDef
        Parsed observer install/restore context manager.
    """

    source = textwrap.dedent(
        inspect.getsource(
            torchlens.backends.torch._completeness_patches._observe_invisible_host_escapes
        )
    )
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def _install_section_and_finally(
    function: ast.FunctionDef,
) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """Split the observer function into the install section and the restore body.

    The observer's R07 shape (27a5c986) is pinned structurally here: the
    install loops live in a fence ``try`` whose sole ``except BaseException``
    arm calls the shared ``_restore_installed`` closure and bare-re-raises,
    and the yield ``try`` unconditionally restores through that same closure
    in its ``finally``. The per-registry restore handlers live in the closure
    body.

    Parameters
    ----------
    function:
        Parsed observer function.

    Returns
    -------
    tuple[list[ast.stmt], list[ast.stmt]]
        The install-fence ``try`` body and the ``_restore_installed`` body.
    """

    restore_def = next(
        (
            statement
            for statement in function.body
            if isinstance(statement, ast.FunctionDef) and statement.name == "_restore_installed"
        ),
        None,
    )
    assert restore_def is not None, "observer lost its shared _restore_installed closure"
    tries = [statement for statement in function.body if isinstance(statement, ast.Try)]
    assert len(tries) == 2, "observer lost its install-fence + yield try pair"
    fence, yield_try = tries
    assert len(fence.handlers) == 1 and not fence.finalbody, (
        "install fence must be a single-handler try"
    )
    fence_handler = fence.handlers[0]
    assert fence_handler.type is not None and ast.unparse(fence_handler.type) == "BaseException", (
        "install fence must catch BaseException"
    )
    fence_actions = [ast.unparse(statement) for statement in fence_handler.body]
    assert "_restore_installed()" in fence_actions, "install fence lost its unwind call"
    assert any(
        isinstance(statement, ast.Raise) and statement.exc is None
        for statement in fence_handler.body
    ), "install fence must bare-re-raise"
    assert yield_try.finalbody, "observer yield try lost its unconditional finally"
    finally_actions = [ast.unparse(statement) for statement in yield_try.finalbody]
    assert "_restore_installed()" in finally_actions, "observer finally lost its restore call"
    return fence.body, restore_def.body


def _is_fail_closed_mark(statement: ast.stmt) -> bool:
    """Classify a statement as ``_HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)``.

    Parameters
    ----------
    statement:
        Statement to classify.

    Returns
    -------
    bool
        ``True`` for the exact fail-closed marking call.
    """

    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value) == "_HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)"
    )


def _blocks_with_guards(
    node: ast.AST, guard: str | None = None
) -> Iterator[tuple[list[ast.stmt], str | None]]:
    """Yield every statement block under ``node`` with its innermost ``if`` guard.

    Parameters
    ----------
    node:
        Root node to walk.
    guard:
        Unparsed test expression of the nearest enclosing ``if``, if any.

    Yields
    ------
    tuple[list[ast.stmt], str | None]
        Statement block plus the guard expression governing it.
    """

    for field_name in ("body", "orelse", "finalbody"):
        block = getattr(node, field_name, None)
        if isinstance(block, list) and block and all(isinstance(s, ast.stmt) for s in block):
            block_guard = ast.unparse(node.test) if isinstance(node, ast.If) else guard
            yield block, block_guard
            for statement in block:
                yield from _blocks_with_guards(statement, block_guard)
    for handler in getattr(node, "handlers", None) or []:
        if isinstance(handler, ast.ExceptHandler):
            yield handler.body, guard
            for statement in handler.body:
                yield from _blocks_with_guards(statement, guard)


def test_every_install_skip_is_fail_closed_or_classified_benign() -> None:
    """Every install-loop ``continue`` marks the trace failed, bar the closed allowlist.

    Returns
    -------
    None
        Asserts the full install inventory (>= ``_MIN_INSTALL_LOOPS`` loops) has no
        unclassified silent skip and that the benign allowlist is used exactly twice.
    """

    install_section, _ = _install_section_and_finally(_observer_function_ast())
    loops = [
        node
        for statement in install_section
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
    ]
    assert len(loops) >= _MIN_INSTALL_LOOPS, (
        f"observer install inventory shrank: {len(loops)} < {_MIN_INSTALL_LOOPS}"
    )
    benign_hits = 0
    seen_continue = 0
    checked_blocks: set[int] = set()
    for loop in loops:
        for block, guard in _blocks_with_guards(loop):
            if id(block) in checked_blocks:
                continue
            checked_blocks.add(id(block))
            for index, statement in enumerate(block):
                if not isinstance(statement, ast.Continue):
                    continue
                seen_continue += 1
                if index > 0 and _is_fail_closed_mark(block[index - 1]):
                    continue
                assert guard in _BENIGN_INSTALL_SKIP_GUARDS, (
                    f"unclassified fail-open install skip under guard {guard!r}"
                )
                benign_hits += 1
    assert seen_continue >= _MIN_INSTALL_LOOPS, "install loops lost their guarded skip arms"
    assert benign_hits == len(_BENIGN_INSTALL_SKIP_GUARDS), (
        f"benign install-skip allowlist drifted: {benign_hits} arms matched, expected "
        f"{len(_BENIGN_INSTALL_SKIP_GUARDS)}"
    )


def test_every_scan_family_fails_closed_on_empty_scan() -> None:
    """Each scan-enumerated observer family marks the trace failed on an empty scan.

    Returns
    -------
    None
        Asserts one fail-closed ``if not <scan>`` guard exists per scan family.
    """

    install_section, _ = _install_section_and_finally(_observer_function_ast())
    guarded: set[str] = set()
    for statement in install_section:
        for node in ast.walk(statement):
            if not isinstance(node, ast.If):
                continue
            test_expr = ast.unparse(node.test)
            if test_expr in _EMPTY_SCAN_GUARDS and any(
                _is_fail_closed_mark(body_stmt) for body_stmt in node.body
            ):
                guarded.add(test_expr)
    missing = _EMPTY_SCAN_GUARDS - guarded
    assert not missing, f"scan families with fail-open empty scans: {sorted(missing)}"


def test_every_restore_handler_is_fail_closed() -> None:
    """Every restore exception handler in the ``finally`` marks the trace failed.

    Returns
    -------
    None
        Asserts the full restore inventory (>= ``_MIN_RESTORE_HANDLERS`` handlers) has no
        ``pass``-swallowed restore failure.
    """

    _, finally_section = _install_section_and_finally(_observer_function_ast())
    handlers = [
        node
        for statement in finally_section
        for node in ast.walk(statement)
        if isinstance(node, ast.ExceptHandler)
    ]
    assert len(handlers) >= _MIN_RESTORE_HANDLERS, (
        f"observer restore inventory shrank: {len(handlers)} < {_MIN_RESTORE_HANDLERS}"
    )
    for handler in handlers:
        assert not any(isinstance(statement, ast.Pass) for statement in handler.body), (
            "restore handler swallows failure with pass"
        )
        assert any(_is_fail_closed_mark(statement) for statement in handler.body), (
            "restore handler is not fail-closed"
        )


def test_absent_observer_member_marks_trace_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A version-drift-absent tensor observer member downgrades the capture.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        Asserts the absent-member install arm marks ``_HOST_ESCAPE_OBSERVER_FAILED`` and
        that the public verdict predicate reports it.
    """

    widened = frozenset(cw.INVISIBLE_HOST_ESCAPE_FUNCS | {"tl_w1f3_absent_member"})
    # The defining module and every witness module that binds the name by import.
    for module in (
        cw,
        torchlens.backends.torch._completeness_patches,
        torchlens.backends.torch._completeness_storage,
    ):
        monkeypatch.setattr(module, "INVISIBLE_HOST_ESCAPE_FUNCS", widened)
    assert getattr(torch.Tensor, "tl_w1f3_absent_member", None) is None
    trace, state = _fresh_state()
    with torchlens.backends.torch._completeness_patches._observe_invisible_host_escapes(state):
        pass
    assert trace in cw._HOST_ESCAPE_OBSERVER_FAILED
    assert torchlens.backends.torch._completeness_cross_thread.host_escape_observer_install_failed(
        trace
    )


def test_failed_observer_restore_marks_trace_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A restore that raises must downgrade the capture, never be swallowed.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        Asserts a module-observer restore failure marks ``_HOST_ESCAPE_OBSERVER_FAILED``.
    """

    class _PoisonedModule:
        """Module stand-in whose second ``setattr`` (the restore) raises."""

        def __init__(self) -> None:
            object.__setattr__(self, "_sets", 0)
            object.__setattr__(self, "target", lambda tensor: tensor)

        def __setattr__(self, name: str, value: Any) -> None:
            count = object.__getattribute__(self, "_sets")
            if count >= 1:
                raise AttributeError("restore refused")
            object.__setattr__(self, "_sets", count + 1)
            object.__setattr__(self, name, value)

    poisoned = _PoisonedModule()
    # The home module and every module that binds the name by import.
    for module in (
        torchlens.backends.torch._completeness_finalize,
        torchlens.backends.torch._completeness_patches,
    ):
        monkeypatch.setattr(module, "_MODULE_ESCAPE_TARGETS", lambda: ((poisoned, "target"),))
    trace, state = _fresh_state()
    with torchlens.backends.torch._completeness_patches._observe_invisible_host_escapes(state):
        pass
    assert trace in cw._HOST_ESCAPE_OBSERVER_FAILED


def test_mro_inherited_descriptor_observer_installs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``torch._C.TensorBase``-inherited descriptor still gets a real observer.

    The pre-fix lookup ``type(torch.Tensor).__dict__.get(name) or
    torch.Tensor.__dict__.get(name)`` misses every descriptor inherited from the C base
    (and consults the metaclass with class-object semantics); the MRO-aware
    ``inspect.getattr_static`` resolution must install the observer instead of silently
    skipping it.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        Asserts the observer property is installed during the window with no failure mark.
    """

    name = "shape"  # getset descriptor on torch._C.TensorBase, not in torch.Tensor.__dict__
    if name in torch.Tensor.__dict__:  # pragma: no cover - torch layout drift escape hatch
        pytest.skip("torch build hoists shape onto torch.Tensor; fixture premise gone")
    legacy = type(torch.Tensor).__dict__.get(name) or torch.Tensor.__dict__.get(name)
    assert legacy is None, "fixture premise: the non-MRO lookup misses this descriptor"
    assert inspect.getattr_static(torch.Tensor, name, None) is not None
    # The defining module and every witness module that binds the name by import.
    for module in (cw, torchlens.backends.torch._completeness_patches):
        monkeypatch.setattr(module, "INVISIBLE_HOST_ESCAPE_PROPERTIES", frozenset({name}))
    trace, state = _fresh_state()
    try:
        with torchlens.backends.torch._completeness_patches._observe_invisible_host_escapes(state):
            installed = isinstance(torch.Tensor.__dict__.get(name), property)
    finally:
        if name in torch.Tensor.__dict__:  # drop the restore shadow: pristine class layout
            delattr(torch.Tensor, name)
    assert installed, "MRO-inherited descriptor observer was never installed"
    assert trace not in cw._HOST_ESCAPE_OBSERVER_FAILED


def test_unwrappable_wrap_required_storage_member_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrap-required storage member that cannot be wrapped downgrades the capture.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        Asserts the non-callable/non-descriptor install arm marks
        ``_HOST_ESCAPE_OBSERVER_FAILED`` instead of silently skipping.
    """

    member = "tl_w1f3_plain_attr"
    patched_rows = {
        name: dict(rows) for name, rows in cw.STORAGE_METADATA_ACCESSOR_DISPOSITIONS.items()
    }
    patched_rows.setdefault("UntypedStorage", {})[member] = (
        cw._STORAGE_ACCESSOR_VALUE_READ,
        "w1f3 fixture: wrap-required row whose member is a plain non-descriptor attribute",
    )
    # The defining module and every witness module that binds the name by import.
    for module in (cw, torchlens.backends.torch._completeness_patches):
        monkeypatch.setattr(module, "STORAGE_METADATA_ACCESSOR_DISPOSITIONS", patched_rows)
    monkeypatch.setattr(torch.UntypedStorage, member, 42, raising=False)
    trace, state = _fresh_state()
    with torchlens.backends.torch._completeness_patches._observe_invisible_host_escapes(state):
        pass
    assert trace in cw._HOST_ESCAPE_OBSERVER_FAILED


def test_empty_storage_scan_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty storage-class scan downgrades the capture like the sibling scans.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        Asserts an empty ``_STORAGE_RAW_POINTER_TARGETS()`` marks
        ``_HOST_ESCAPE_OBSERVER_FAILED``.
    """

    # The home module and every module that binds the name by import.
    for module in (
        torchlens.backends.torch._completeness_finalize,
        torchlens.backends.torch._completeness_patches,
    ):
        monkeypatch.setattr(module, "_STORAGE_RAW_POINTER_TARGETS", lambda: ())
    trace, state = _fresh_state()
    with torchlens.backends.torch._completeness_patches._observe_invisible_host_escapes(state):
        pass
    assert trace in cw._HOST_ESCAPE_OBSERVER_FAILED
