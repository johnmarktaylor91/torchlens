"""Closed ledger of sanctioned in-place op-lane writers (r7 R03-1).

The producer contract says the op lane is append-only with post-commit
knowledge riding the typed amendment lane -- but the fastlog ancestry-closure
backfill legitimately REPLACES op-lane cells in place on the ``_copy_for_replay``
projection a cook owns. That carve-out lived in exactly one function docstring:
no contract named it, and a subscript assignment (``op_events[i] = ...``) is
invisible to the name-based P4 AST guard that pins the amendment call sites. A
second in-place writer landing anywhere else -- especially one that forgets the
``_amended_fold_cache`` invalidation or runs on a journal that is NOT a
cook-owned projection -- would pass every producer gate silently.

This gate closes that ceiling: every mutation of an ``op_events`` container
anywhere in the shipped package (subscript/slice assignment, ``del``, and the
mutating list methods) must appear in the reason-bearing sanctioned ledger,
EXACTLY -- a new writer is red until it is consciously ledgered, and a retired
writer is red until it is removed here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _source_corpus import package_ast, package_files

import torchlens as tl

#: Mutating list-method names that rewrite op-lane contents in place.
_MUTATING_METHODS = frozenset(
    {"append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse", "__setitem__"}
)

#: The closed set of sanctioned in-place op-lane writers, each with its WHY.
#: (relative path, enclosing qualname, mutation kind).
SANCTIONED_INPLACE_OP_LANE_WRITERS = frozenset(
    {
        # The append chokepoint itself: seq-stamping single writer for the
        # forward op spine (refuses on sealed journals).
        ("_ir/capture_events.py", "CaptureEvents.append", "append"),
        # Working-projection teardown after Step 0 consumes the lanes.
        ("_ir/capture_events.py", "CaptureEvents.release_working_projection", "clear"),
        # THE cook-owned-projection carve-out (r7 R03-1): the fastlog
        # ancestry-closure backfill replaces op-lane cells in place, ONLY on
        # the ``_copy_for_replay`` projection a cook owns (never the sealed
        # Recording stream) and explicitly nulls ``_amended_fold_cache``
        # afterwards. Named in torchlens/CLAUDE.md ("Journal Producer").
        ("fastlog/types.py", "_backfill_cooked_ancestry", "subscript_write"),
    }
)


def _subscript_over_op_events(node: ast.expr) -> bool:
    """Return whether an expression is a subscript over an op_events container."""

    if not isinstance(node, ast.Subscript):
        return False
    base = node.value
    if isinstance(base, ast.Attribute):
        return base.attr == "op_events"
    return isinstance(base, ast.Name) and base.id == "op_events"


def _scan_source(source: str, relative: str) -> set[tuple[str, str, str]]:
    """Return every in-place op-lane mutation site in one module's source."""

    return _scan_tree(ast.parse(source), relative)


def _scan_tree(tree: ast.AST, relative: str) -> set[tuple[str, str, str]]:
    """Return every in-place op-lane mutation site in one parsed module."""

    found: set[tuple[str, str, str]] = set()

    def _visit(node: ast.AST, stack: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            child_stack = stack
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_stack = (*stack, child.name)
            qualname = ".".join(child_stack) or "<module>"
            if isinstance(child, (ast.Assign, ast.AugAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                if any(_subscript_over_op_events(target) for target in targets):
                    found.add((relative, qualname, "subscript_write"))
            if isinstance(child, ast.AugAssign) and (
                (isinstance(child.target, ast.Attribute) and child.target.attr == "op_events")
                or (isinstance(child.target, ast.Name) and child.target.id == "op_events")
            ):
                # ``events.op_events += [...]`` mutates the list in place.
                found.add((relative, qualname, "augmented_write"))
            if isinstance(child, ast.Delete) and any(
                _subscript_over_op_events(target) for target in child.targets
            ):
                found.add((relative, qualname, "delete"))
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                method = child.func.attr
                if method in _MUTATING_METHODS and (
                    (
                        isinstance(child.func.value, ast.Attribute)
                        and child.func.value.attr == "op_events"
                    )
                    or (
                        isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "op_events"
                    )
                ):
                    found.add((relative, qualname, method))
            _visit(child, child_stack)

    _visit(tree, ())
    return found


@pytest.mark.heavy
def test_inplace_op_lane_writers_match_the_sanctioned_ledger() -> None:
    """Package-wide scan: op-lane mutators == the reason-bearing ledger, exactly.

    ``heavy`` by measured cost at marking time (5.3s standalone package
    scan, 2026-08-17). The scan now reads the shared session corpus
    (tests/_source_corpus.py), so the per-test cost is the walk only; the
    tier stays conservative rather than tracking every infrastructure win.
    """

    package_root = Path(tl.__file__).parent
    observed: set[tuple[str, str, str]] = set()
    for source_path in package_files():
        relative = source_path.relative_to(package_root).as_posix()
        observed |= _scan_tree(package_ast(source_path), relative)

    unsanctioned = observed - SANCTIONED_INPLACE_OP_LANE_WRITERS
    assert not unsanctioned, (
        "UNSANCTIONED in-place op-lane writer(s): the op lane is append-only "
        "with post-commit knowledge riding the typed amendment lane "
        "(append_amendment). Route the change through an amendment, or -- for "
        "a genuinely cook-owned-projection rewrite that nulls "
        "_amended_fold_cache -- add a reason-bearing ledger row here: "
        f"{sorted(unsanctioned)}"
    )
    stale = SANCTIONED_INPLACE_OP_LANE_WRITERS - observed
    assert not stale, (
        f"stale sanctioned-writer ledger rows (writer moved or retired): {sorted(stale)}"
    )


def test_op_lane_writer_scan_is_red_capable() -> None:
    """The scanner sees every mutation shape a forged writer could take."""

    forged = "\n".join(
        (
            "def rogue(events):",
            "    events.op_events[3] = None",
            "    events.op_events[1:2] = []",
            "    events.op_events += [None]",
            "    del events.op_events[0]",
            "    events.op_events.pop()",
            "    op_events = events.op_events",
            "    op_events[0] = None",
            "    op_events.extend([None])",
        )
    )
    found = _scan_source(forged, "rogue.py")
    assert found == {
        ("rogue.py", "rogue", "subscript_write"),
        ("rogue.py", "rogue", "augmented_write"),
        ("rogue.py", "rogue", "delete"),
        ("rogue.py", "rogue", "pop"),
        ("rogue.py", "rogue", "extend"),
    }
    # Reads never trip the gate.
    benign = "def reader(events):\n    return events.op_events[-1]\n"
    assert _scan_source(benign, "reader.py") == set()
