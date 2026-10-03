"""The layer lint: eager-upward imports never increase (C01 items 1 + 12).

Module-granular AST audit under the memo's tiering (torchlens/_architecture.py
is the declared layer map; a module-level ``__tl_layer__`` overrides its
package row). Eager upward imports are the inversion class; deferred
(function-body) and typing-only (TYPE_CHECKING) imports are legal; declared
FACADE-role modules are the one legal upward door (Rule F) and are excluded
from the denominator exactly as the memo's consensus count excluded the
eight facade modules.

The baseline (tests/test_arch_spine_layer_baseline.tsv) was recomputed under
this tiering AFTER the C01 moves landed, then FROZEN: per-file counts may
only shrink, and a NEW file may not introduce inversions. Regenerate
deliberately with TORCHLENS_REGEN_LAYER_BASELINE=1 and review the diff.

The same walk enforces the torch-privates license (item 12): every package
touching ``torch._*`` or the ``_torch_compat`` chokepoint carries a row in
``TORCH_PRIVATE_LICENSED_PACKAGES``; the count is printed and shrink-only.
"""

from __future__ import annotations

import ast
import functools
import os
import re
from pathlib import Path

import pytest

from torchlens._architecture import (
    LAYER_ORDER,
    PACKAGE_LAYERS,
    TORCH_PRIVATE_LICENSED_PACKAGES,
    layer_for_module,
)

REPO = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO / "torchlens"
BASELINE_PATH = Path(__file__).with_name("test_arch_spine_layer_baseline.tsv")

#: Root modules not individually mapped resolve through these startswith
#: families before falling back to the package root's FACADE row.
_ROOT_FAMILY_LAYERS: tuple[tuple[str, str], ...] = (
    ("_runnable", "L3"),
    ("_trace_", "L1"),
)


# Session-scoped caches: the tree is frozen for the duration of a test run,
# and this file's tests each walk the same 650+ modules. Rescanning the
# package root and re-reading every source per test (and per pass inside the
# inversion audit) was the dominant avoidable cost that pushed the audit over
# the smoke duration budget once the tree grew (T50 fix cycle).
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
    """Read every module's source exactly once per session."""

    return {dotted: path.read_text() for dotted, path in _iter_modules()}


_TL_LAYER_PATTERN = re.compile(r'^__tl_layer__\s*=\s*"([A-Z0-9]+)"', re.MULTILINE)


def _declared_layer(dotted: str, source: str) -> str | None:
    match = _TL_LAYER_PATTERN.search(source)
    if match:
        return match.group(1)
    for prefix, layer in _ROOT_FAMILY_LAYERS:
        if "." not in dotted and dotted.startswith(prefix) and dotted not in PACKAGE_LAYERS:
            return layer
    return layer_for_module(dotted)


def _resolve_relative(dotted: str, path: Path, node: ast.ImportFrom) -> str | None:
    """Resolve a relative import to a torchlens-relative dotted path."""

    if node.level == 0:
        if node.module and node.module.startswith("torchlens"):
            trimmed = node.module[len("torchlens") :].lstrip(".")
            return trimmed
        return None
    is_package = path.name == "__init__.py"
    parts = dotted.split(".") if dotted else []
    # For a module, level 1 = its own package; for a package __init__,
    # level 1 = itself.
    up = node.level - (1 if is_package else 0)
    if up > len(parts):
        return None
    base = parts[: len(parts) - up] if up else parts
    target = ".".join(base + (node.module.split(".") if node.module else []))
    return target


@functools.lru_cache(maxsize=1)
def _existing_targets() -> frozenset[str]:
    """Every dotted name resolvable as a module or package under torchlens/.

    ``_iter_modules`` yields exactly the dotted spelling of every ``.py`` file
    (and, via ``__init__.py``, every package), so set membership is equivalent
    to the former two-``stat()``-per-alias filesystem probe.
    """

    return frozenset(dotted for dotted, _ in _iter_modules())


def _target_exists(target: str) -> bool:
    return target in _existing_targets()


#: Statement fields that can carry (transitively) eager import statements.
_STMT_LIST_FIELDS = ("body", "orelse", "finalbody")


def _eager_import_targets(tree: ast.Module, dotted: str, path: Path) -> list[str]:
    """Collect eager torchlens-internal import targets (skip deferred/typing).

    Imports are statements, so the walk stays on statement lists instead of
    descending into every expression node (the former ``NodeVisitor`` visited
    the full tree; on 650+ modules that traversal alone was a large share of
    the audit's budget). Semantics are pinned identical: function bodies are
    deferred, a ``TYPE_CHECKING`` conditional is skipped whole (both arms,
    matching the visitor's early return), everything else recurses.
    """

    eager_targets: list[str] = []

    def walk(stmts: list[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # deferred
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                continue  # typing-only
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("torchlens."):
                        eager_targets.append(alias.name[len("torchlens.") :])
                continue
            if isinstance(node, ast.ImportFrom):
                target = _resolve_relative(dotted, path, node)
                if target is None:
                    continue
                for alias in node.names:
                    candidate = f"{target}.{alias.name}" if target else alias.name
                    if _target_exists(candidate):
                        eager_targets.append(candidate)
                    elif target and _target_exists(target):
                        eager_targets.append(target)
                continue
            for field in _STMT_LIST_FIELDS:
                children = getattr(node, field, None)
                if children:
                    walk(children)
            for handler in getattr(node, "handlers", None) or ():
                walk(handler.body)
            for case in getattr(node, "cases", None) or ():
                walk(case.body)

    walk(tree.body)
    return eager_targets


def _compute_inversions() -> dict[str, list[tuple[str, str, str, str]]]:
    """Return file -> [(source_layer, target, target_layer, kind)] inversions."""

    sources = _module_sources()
    layers: dict[str, str | None] = {
        dotted: _declared_layer(dotted, sources[dotted]) for dotted, _ in _iter_modules()
    }

    findings: dict[str, list[tuple[str, str, str, str]]] = {}
    for dotted, path in _iter_modules():
        source_layer = layers.get(dotted)
        if source_layer is None or source_layer not in LAYER_ORDER:
            continue  # FACADE-role and unmapped modules are outside the denominator
        for target in _eager_import_targets(ast.parse(sources[dotted]), dotted, path):
            probe = target
            while probe and probe not in layers and "." in probe:
                probe = probe.rsplit(".", 1)[0]
            target_layer = layers.get(probe)
            if target_layer is None or target_layer not in LAYER_ORDER:
                continue
            if LAYER_ORDER[target_layer] > LAYER_ORDER[source_layer]:
                findings.setdefault(dotted, []).append(
                    (source_layer, target, target_layer, "eager")
                )
    return findings


def _baseline_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in BASELINE_PATH.read_text().splitlines():
        if not line or line.startswith("#") or line.startswith("module\t"):
            continue
        module, count = line.split("\t")
        counts[module] = int(count)
    return counts


def test_eager_upward_inversions_never_increase() -> None:
    findings = _compute_inversions()
    live = {module: len(rows) for module, rows in findings.items()}
    total = sum(live.values())
    print(f"\nLAYER LINT: {total} eager upward inversions across {len(live)} modules")

    if os.environ.get("TORCHLENS_REGEN_LAYER_BASELINE") == "1":
        from _oracle_env import guard_wrap_state_for_golden_update

        # AST-only generation (no capture), but the guard is the structural
        # census rule for every in-process regen flag (SF-53).
        guard_wrap_state_for_golden_update("TORCHLENS_REGEN_LAYER_BASELINE")
        body = "\n".join(f"{module}\t{count}" for module, count in sorted(live.items()))
        BASELINE_PATH.write_text(
            "# test_arch_spine_layer_baseline.tsv -- FROZEN eager-upward inversion\n"
            "# baseline per module, recomputed under the memo tiering after the C01\n"
            "# moves (item 1). Counts only shrink; regenerate deliberately with\n"
            "# TORCHLENS_REGEN_LAYER_BASELINE=1 and review the diff.\n"
            "module\tcount\n" + body + "\n"
        )
        pytest.skip("baseline regenerated; review the diff and rerun without the flag")

    baseline = _baseline_counts()
    regressions = {
        module: (baseline.get(module, 0), count)
        for module, count in live.items()
        if count > baseline.get(module, 0)
    }
    assert not regressions, (
        "eager upward imports INCREASED (module: baseline -> live): "
        f"{regressions} -- move the vocabulary down, defer the import inside a "
        "function (Rule F requires a declared FACADE-role module), or fix the "
        "layer map row if the module is genuinely mis-tiered"
    )
    burned_down = {
        module: (count, live.get(module, 0))
        for module, count in baseline.items()
        if live.get(module, 0) < count
    }
    if burned_down:
        print(f"burn-down candidates (baseline -> live): {burned_down}")


def test_layer_map_covers_every_package() -> None:
    """Every package directory resolves to a declared layer or role."""

    unmapped = sorted(
        dotted
        for dotted, path in _iter_modules()
        if path.name == "__init__.py"
        and dotted
        and _declared_layer(dotted, _module_sources()[dotted]) is None
    )
    assert unmapped == [], f"packages with no layer row: {unmapped}"


def test_declared_attributes_agree_with_the_table() -> None:
    """Where both a module attribute and a table row exist, they agree."""

    conflicts = []
    for dotted, path in _iter_modules():
        match = _TL_LAYER_PATTERN.search(_module_sources()[dotted])
        if not match:
            continue
        if match.group(1) == "FACADE" and path.name == "__init__.py":
            # A package __init__ may declare the FACADE role while the table
            # row carries the package's dominant stratum for its submodules.
            continue
        table_layer = PACKAGE_LAYERS.get(dotted)
        if table_layer is not None and table_layer != match.group(1):
            conflicts.append((dotted, match.group(1), table_layer))
    assert conflicts == [], f"__tl_layer__ vs table conflicts (module, attr, table): {conflicts}"


_TORCH_PRIVATE_PATTERN = re.compile(r"\btorch\._|from torch import _|_torch_compat")


def test_torch_privates_license_is_declared_and_shrink_only() -> None:
    """Item 12: packages touching torch privates carry a license row."""

    touching: set[str] = set()
    for dotted, source in _module_sources().items():
        if dotted == "_architecture":
            continue  # the license declaration module names the chokepoint
        if _TORCH_PRIVATE_PATTERN.search(source):
            touching.add(dotted.split(".")[0] if dotted else "")
    touching.discard("")
    print(f"\nTORCH-PRIVATES LICENSE: {len(touching)} packages touch torch privates")
    undeclared = sorted(touching - TORCH_PRIVATE_LICENSED_PACKAGES)
    assert undeclared == [], (
        f"packages touching torch._/_torch_compat without a license row: {undeclared} "
        "-- route the probe through torchlens/utils/_torch_compat.py and add the "
        "package to TORCH_PRIVATE_LICENSED_PACKAGES only with a reviewed reason"
    )
    stale = sorted(TORCH_PRIVATE_LICENSED_PACKAGES - touching)
    assert stale == [], (
        f"license rows whose packages no longer touch torch privates: {stale} "
        "-- the count is shrink-only; delete the stale rows in this change"
    )
