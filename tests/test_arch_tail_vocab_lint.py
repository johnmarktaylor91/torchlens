"""The generalized vocabulary lint: Rule V3 + closure census (F35; memo item 6).

C01 landed the ratified V2 relocations with per-move V3 spot checks
(tests/test_arch_spine_relocations.py). This suite is the GENERAL lint the
memo's VOCABULARY CLOSURE row demands, over every module that declares
``__tl_vocabulary__ = True``:

- **V3 (no smuggling), always** (memo 3.3): a declared vocabulary module may
  not eagerly import behavior, and its function bodies may not defer
  torchlens imports in (behavior smuggling). ``__tl_vocabulary__`` is a
  CLASSIFIER and relocation trigger, never an import-law waiver -- so the
  one declared module that is not V3-clean today (``fastlog/types``, the
  Recording product home, declared by C01 item 4 as a label ahead of its
  physical split) rides a reason-bearing residue ledger whose counts are
  SHRINK-ONLY. A new smuggling edge anywhere is RED.
- **Vocabulary is born at L0/L1**: every declared vocabulary module carries
  a BASIS or PRODUCT layer. New upward vocabulary edges are banned.
- **Closure census**: per declared module, the transitive eager
  torchlens-internal import closure is computed and printed, and the
  published closure sizes only shrink (memo test-plan row VOCABULARY
  CLOSURE).

Growing the vocabulary is legal and deliberate: declare
``__tl_vocabulary__ = True`` + ``__tl_layer__`` of L0/L1 in the module and
add it to ``DECLARED_VOCABULARY`` here in the same change; the new module
must be V3-clean (the residue ledger is closed to new entries).
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO / "torchlens"

#: The declared-vocabulary census (frozen; grow only with a V3-clean module).
DECLARED_VOCABULARY: frozenset[str] = frozenset(
    {
        "_vocab",
        "_vocab.trace_state",
        "_vocab.node_spec",
        "_io.format_errors",
        "fastlog.types",
    }
)

#: Vocabulary modules may eagerly import ONLY other vocabulary modules and
#: the L0 error strata (error classes are themselves frozen vocabulary).
_ALLOWED_EAGER_PREFIXES: tuple[str, ...] = ("_errors", "errors", "_vocab")

#: Reason-bearing V3 residue ledger -- the classifier-not-waiver posture in
#: test form. Counts are ceilings and only shrink; the ledger is CLOSED to
#: new modules. fastlog/types was DECLARED vocabulary by C01 (memo item 4:
#: "fastlog/types declared vocabulary") ahead of its physical vocabulary/
#: behavior split: the Recording product class still lives there, carrying
#: 3 distinct eager behavior targets (captured_run, ir.predicate,
#: utils.tensor_utils) and 14 distinct deferred import targets
#: (draw/summary/to_trace/log_backward delegations plus the sanctioned
#: ancestry-backfill writer). The relocation trigger is armed: burn these
#: down by splitting the vocabulary half out, never by widening the ledger.
V3_RESIDUE_CEILINGS: dict[str, tuple[int, int]] = {
    # module: (max distinct eager behavior targets, max distinct deferred targets)
    "fastlog.types": (3, 14),
}


@functools.lru_cache(maxsize=1)
def _iter_modules() -> tuple[tuple[str, Path], ...]:
    modules: list[tuple[str, Path]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE_ROOT)
        parts = list(relative.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        modules.append((".".join(parts), path))
    return tuple(modules)


@functools.lru_cache(maxsize=1)
def _module_sources() -> dict[str, str]:
    return {dotted: path.read_text() for dotted, path in _iter_modules()}


@functools.lru_cache(maxsize=1)
def _module_paths() -> dict[str, Path]:
    return dict(_iter_modules())


_VOCAB_PATTERN = re.compile(r"^__tl_vocabulary__\s*=\s*True", re.MULTILINE)
_TL_LAYER_PATTERN = re.compile(r'^__tl_layer__\s*=\s*"([A-Z0-9]+)"', re.MULTILINE)


@functools.lru_cache(maxsize=1)
def _declared_vocabulary_modules() -> frozenset[str]:
    return frozenset(
        dotted for dotted, source in _module_sources().items() if _VOCAB_PATTERN.search(source)
    )


def _resolve_relative(dotted: str, path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        if node.module and node.module.startswith("torchlens"):
            return node.module[len("torchlens") :].lstrip(".")
        return None
    is_package = path.name == "__init__.py"
    parts = dotted.split(".") if dotted else []
    up = node.level - (1 if is_package else 0)
    if up > len(parts):
        return None
    base = parts[: len(parts) - up] if up else parts
    return ".".join(base + (node.module.split(".") if node.module else []))


def _import_targets(dotted: str) -> tuple[list[str], list[str]]:
    """Return (eager, deferred) torchlens-internal import targets.

    Function bodies are deferred; ``TYPE_CHECKING`` conditionals are
    typing-only and skipped whole (same semantics as the layer lint).
    """

    path = _module_paths()[dotted]
    existing = frozenset(_module_paths())
    eager: list[str] = []
    deferred: list[str] = []

    def collect(node: ast.Import | ast.ImportFrom, sink: list[str]) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("torchlens."):
                    sink.append(alias.name[len("torchlens.") :])
            return
        target = _resolve_relative(dotted, path, node)
        if target is None:
            return
        for alias in node.names:
            candidate = f"{target}.{alias.name}" if target else alias.name
            if candidate in existing:
                sink.append(candidate)
            elif target and target in existing:
                sink.append(target)

    def walk(stmts: list[ast.stmt], in_function: bool) -> None:
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(node.body, True)
                continue
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                collect(node, deferred if in_function else eager)
                continue
            for field in ("body", "orelse", "finalbody"):
                children = getattr(node, field, None)
                if children:
                    walk(children, in_function)
            for handler in getattr(node, "handlers", None) or ():
                walk(handler.body, in_function)
            for case in getattr(node, "cases", None) or ():
                walk(case.body, in_function)

    walk(ast.parse(_module_sources()[dotted]).body, False)
    return eager, deferred


def _eager_closure(start: str) -> set[str]:
    seen: set[str] = set()
    frontier = [start]
    paths = _module_paths()
    while frontier:
        module = frontier.pop()
        if module in seen or module not in paths:
            continue
        seen.add(module)
        eager, _ = _import_targets(module)
        frontier.extend(target for target in eager if target not in seen)
    seen.discard(start)
    return seen


def test_vocabulary_census_is_pinned() -> None:
    """The declared set is frozen contract data; growth is a reviewed act."""

    live = _declared_vocabulary_modules()
    assert live == DECLARED_VOCABULARY, (
        f"declared-vocabulary census drifted: +{sorted(live - DECLARED_VOCABULARY)} "
        f"-{sorted(DECLARED_VOCABULARY - live)} -- a new vocabulary module must be "
        "born at L0/L1 and V3-clean; update DECLARED_VOCABULARY in the same change"
    )


def test_vocabulary_is_born_at_basis_or_product() -> None:
    """Memo 3.3: NEW vocabulary shared downward is born at L0/L1."""

    misplaced = []
    for module in sorted(_declared_vocabulary_modules()):
        match = _TL_LAYER_PATTERN.search(_module_sources()[module])
        layer = match.group(1) if match else None
        if layer not in {"L0", "L1"}:
            misplaced.append((module, layer))
    assert misplaced == [], (
        f"vocabulary modules outside L0/L1: {misplaced} -- frozen vocabularies are "
        "BASIS/PRODUCT strata; relocate the vocabulary down (Rule V2) or undeclare"
    )


def test_v3_no_eager_behavior_imports() -> None:
    """V3: a vocabulary module may not eagerly import behavior."""

    offenders: dict[str, list[str]] = {}
    for module in sorted(_declared_vocabulary_modules()):
        eager, _ = _import_targets(module)
        behavior = sorted(
            {
                target
                for target in eager
                if not target.startswith(_ALLOWED_EAGER_PREFIXES)
                and target not in _declared_vocabulary_modules()
            }
        )
        ceiling = V3_RESIDUE_CEILINGS.get(module, (0, 0))[0]
        if len(behavior) > ceiling:
            offenders[module] = behavior
    assert offenders == {}, (
        f"V3 violations (eager behavior imports from vocabulary modules): {offenders} "
        "-- vocabulary imports only vocabulary/errors; move the behavior out or "
        "relocate the vocabulary down. The residue ledger is closed and shrink-only."
    )


def test_v3_no_deferred_behavior_smuggling() -> None:
    """V3: function bodies in vocabulary modules may not defer torchlens imports."""

    offenders: dict[str, int] = {}
    for module in sorted(_declared_vocabulary_modules()):
        _, deferred = _import_targets(module)
        distinct = len(set(deferred))
        ceiling = V3_RESIDUE_CEILINGS.get(module, (0, 0))[1]
        if distinct > ceiling:
            offenders[module] = distinct
    assert offenders == {}, (
        f"V3 violations (deferred behavior imports, module -> count): {offenders} -- "
        "a vocabulary module defines names, never functions that call into "
        "behavior; the fastlog.types ceiling only shrinks (split the Recording "
        "behavior out rather than widening it)"
    )


#: Published closure sizes (memo: "published closure sizes only shrink").
_CLOSURE_BASELINE: dict[str, int] = {
    "_vocab": 0,
    "_vocab.trace_state": 0,
    "_vocab.node_spec": 0,
    "_io.format_errors": 1,  # errors._base (L0 error vocabulary)
    "fastlog.types": 25,  # the declared residue; splits burn this down
}


def test_vocabulary_closure_sizes_only_shrink() -> None:
    lines = []
    regressions = {}
    for module in sorted(_declared_vocabulary_modules()):
        closure = _eager_closure(module)
        lines.append(f"{module}: closure {len(closure)}")
        baseline = _CLOSURE_BASELINE.get(module)
        if baseline is None or len(closure) > baseline:
            regressions[module] = (baseline, len(closure))
    print("\nVOCABULARY CLOSURE: " + "; ".join(lines))
    assert regressions == {}, (
        f"vocabulary closures grew (module: published -> live): {regressions} -- "
        "the vocabulary graph is itself a cross-layer DAG; publish the new closure "
        "size by shrinking it, never by silently dragging more modules"
    )
