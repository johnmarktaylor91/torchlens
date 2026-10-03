"""Instrumented consumer ledger for the producer migration (P0, four sources).

Sources (design-of-record section 7):

1. Static AST scan — every ``<expr>.<field>`` and ``getattr(<expr>, "<field>"
   [, default])`` site over ``torchlens/`` where ``<field>`` is an OpEvent
   field name. Over-approximates by name (conservative); records
   ``has_default`` so getattr default-reliers are explicit (review v4 note N10
   lineage: the strict protocol must raise an AttributeError subclass for
   these sites to keep today's semantics).
2. Runtime read instrumentation — ``OpEvent.__getattribute__`` recording over
   the scenario battery; the observed (field, caller) set must close against
   the static inventory. The full ``not slow`` instrumented sweep uses the
   same hook via ``TORCHLENS_LEDGER_RECORD=<path>``.
3. Planted-field sensitivity — lives in ``test_planted.py``.
4. Mutator/lifecycle inventory — the exact ``replace_op_event`` caller set,
   the six preview promotion sites, and the lifecycle surfaces; asserted
   exactly so any new post-commit mutation channel fails the gate.

Plus the step-0 trace-read recorder feeding the ``IngestInputs`` v1 freeze:
every ``trace.<attr>`` read during ``materialize_from_events`` is recorded;
the P1 gate asserts the set is covered by the frozen input enumeration.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from _source_corpus import PACKAGE_ROOT as _CORPUS_PACKAGE_ROOT, package_ast, package_files

from torchlens.ir.events import OpEvent


def _require_corpus_root(package_root: Path) -> None:
    """These scans read the shared session corpus, which covers exactly the
    repo package; any other root would silently scan the wrong tree."""

    resolved = package_root.resolve()
    assert resolved == _CORPUS_PACKAGE_ROOT, (
        f"scan root {resolved} != shared corpus root {_CORPUS_PACKAGE_ROOT}"
    )


OPEVENT_FIELDS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(OpEvent))

# Field names too generic to attribute to OpEvent by name alone; the static
# scan still records them but flags ambiguity so closure reads stay honest.
_AMBIGUOUS_FIELDS: frozenset[str] = frozenset(
    {
        "kind",
        "output",
        "label_raw",
        "params",
        "parents",
        "function",
        "policy",
        "modules",
        "seq",
        "address",
        "shape",
        "dtype",
    }
)


@dataclasses.dataclass(frozen=True)
class ReadSite:
    """One static consumer site."""

    file: str
    line: int
    field: str
    via_getattr: bool
    has_default: bool
    ambiguous: bool


def static_scan(package_root: Path) -> list[ReadSite]:
    """AST-scan the package for OpEvent-field read sites."""

    _require_corpus_root(package_root)
    sites: list[ReadSite] = []
    for path in package_files():
        rel = str(path.relative_to(package_root.parent))
        tree = package_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in OPEVENT_FIELDS:
                if isinstance(node.ctx, ast.Load):
                    sites.append(
                        ReadSite(
                            file=rel,
                            line=node.lineno,
                            field=node.attr,
                            via_getattr=False,
                            has_default=False,
                            ambiguous=node.attr in _AMBIGUOUS_FIELDS,
                        )
                    )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value in OPEVENT_FIELDS
            ):
                sites.append(
                    ReadSite(
                        file=rel,
                        line=node.lineno,
                        field=node.args[1].value,
                        via_getattr=True,
                        has_default=len(node.args) >= 3,
                        ambiguous=node.args[1].value in _AMBIGUOUS_FIELDS,
                    )
                )
    return sites


@contextlib.contextmanager
def runtime_read_recorder() -> Iterator[dict[str, set[str]]]:
    """Record every OpEvent attribute read as field -> {caller file:line}."""

    reads: dict[str, set[str]] = {}
    original = OpEvent.__getattribute__

    def recording(self: Any, name: str) -> Any:
        if name in OPEVENT_FIELDS:
            frame = sys._getframe(1)
            caller = f"{frame.f_code.co_filename}:{frame.f_lineno}"
            reads.setdefault(name, set()).add(caller)
        return original(self, name)

    OpEvent.__getattribute__ = recording  # type: ignore[method-assign]
    try:
        yield reads
    finally:
        OpEvent.__getattribute__ = original  # type: ignore[method-assign]


@contextlib.contextmanager
def step0_trace_read_recorder() -> Iterator[set[str]]:
    """Record trace attribute names read during step-0 materialization."""

    from torchlens.data_classes.trace import Trace

    postprocess_module = importlib.import_module("torchlens.postprocess")
    materialize_module = importlib.import_module("torchlens.postprocess._materialize")
    original_public = postprocess_module.materialize_from_events
    original_direct = materialize_module.materialize_from_events
    original_getattribute = Trace.__getattribute__

    observed: set[str] = set()
    recording_active = [False]

    def recording_getattribute(self: Any, name: str) -> Any:
        if recording_active[0] and not name.startswith("__"):
            observed.add(name)
        return original_getattribute(self, name)

    def observing(trace: Any, events: Any) -> None:
        recording_active[0] = True
        try:
            original_direct(trace, events)
        finally:
            recording_active[0] = False

    Trace.__getattribute__ = recording_getattribute  # type: ignore[method-assign]
    postprocess_module.materialize_from_events = observing
    materialize_module.materialize_from_events = observing
    try:
        yield observed
    finally:
        Trace.__getattribute__ = original_getattribute  # type: ignore[method-assign]
        postprocess_module.materialize_from_events = original_public
        materialize_module.materialize_from_events = original_direct


def mutator_inventory(package_root: Path) -> dict[str, list[str]]:
    """Exact post-commit mutation channels after the P4 migration.

    The ONE sanctioned channel is the typed amendment lane
    (``append_amendment`` callers); the legacy channels — ``replace_op_event``
    callers and in-place ``op_events[i] = ...`` list writes — are scanned so
    the ledger PROVES they stay at zero.
    """

    _require_corpus_root(package_root)
    replace_callers: list[str] = []
    amendment_callers: list[str] = []
    inplace_list_writes: list[str] = []
    for path in package_files():
        rel = str(path.relative_to(package_root.parent))
        tree = package_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if name == "replace_op_event":
                    replace_callers.append(f"{rel}:{node.lineno}")
                if name == "append_amendment":
                    amendment_callers.append(f"{rel}:{node.lineno}")
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Attribute)
                and node.targets[0].value.attr == "op_events"
            ):
                inplace_list_writes.append(f"{rel}:{node.lineno}")
    return {
        "replace_op_event_callers": replace_callers,
        "append_amendment_callers": amendment_callers,
        "op_events_inplace_writes": inplace_list_writes,
    }
