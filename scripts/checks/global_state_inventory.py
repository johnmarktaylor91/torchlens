"""Global-state inventory: every mutable module global in the package is classified.

Moved from ``tests/test_global_state_inventory.py`` (which keeps the exception-restoration
and concurrency tests that capture models). Three checks read the package source only:

- every mutable module global in ``torchlens/`` has exactly one lifecycle class, and every
  class row still names a global that exists (shrink-only, both directions);
- the hot capture/wrapper subset census (the b2p2-F2 sentinel), kept as its own line so a
  failure in that territory names it;
- the weak/strong storage split of every inventory member equals the ``_WEAKLY_HELD`` ledger.

The rows live in ``global_state_rows.py`` beside this file: a new global in ``torchlens/``
lands there as a row, with a rationale comment when its class is not obvious from the name.
One whole-package parse (``_mutable_module_state_cached``) serves all three checks.
"""

from __future__ import annotations

import ast
import functools
import sys
from pathlib import Path

from _harness import REPO_ROOT, run_checks, use_test_helpers
from global_state_rows import _LIFECYCLE_CLASSES, _WEAK_SUBJECT_TABLES, _WEAKLY_HELD

use_test_helpers()

from _source_corpus import package_ast  # noqa: E402

MUTATING_METHODS = frozenset(
    {
        "append",
        "appendleft",
        "add",
        "clear",
        "discard",
        "extend",
        "insert",
        "move_to_end",
        "pop",
        "popitem",
        "popleft",
        "remove",
        "setdefault",
        "sort",
        "update",
        "__setitem__",
    }
)
"""Method names whose call mutates a container in place."""


MUTABLE_FACTORIES = frozenset(
    {
        "ChainMap",
        "Counter",
        "OrderedDict",
        "WeakKeyDictionary",
        "WeakSet",
        "WeakValueDictionary",
        "defaultdict",
        "deque",
        "dict",
        "list",
        "set",
    }
)
"""Callables whose result is a mutable container."""


#: Each module tree's nodes in ``ast.walk`` order, keyed by the shared tree's id. The
#: census walks every module four times (globals, in-place mutation, alias targets,
#: cross-module rebinds); flattening each tree once keeps the run inside the check
#: runner's time budget. The trees are the shared ``_source_corpus`` objects, alive for
#: the whole process, so an id key cannot be reused by another tree.
_WALKED: dict[int, tuple[ast.AST, ...]] = {}


def _walk(tree: ast.Module) -> tuple[ast.AST, ...]:
    """Return ``tuple(ast.walk(tree))`` for a shared module tree, computed once."""

    nodes = _WALKED.get(id(tree))
    if nodes is None:
        nodes = tuple(ast.walk(tree))
        _WALKED[id(tree)] = nodes
    return nodes


def _package_python_paths(repo: Path) -> list[Path]:
    """Return every Python source file in the package.

    The inventory used to scan only ``_state.py`` plus ``capture/``,
    ``validation/`` and ``backends/torch/``, which governed a MINORITY of the
    surface it claimed: the process-global per-capture tap observers in the
    mlx/paddle backends, the mutable facet toggle, the RF rule registry, the
    warn-once sentinels in ``_io``/``visualization``/``distributed`` and the
    whole ``_torch_compat`` capability block were all out of scope, so a new
    unclassified global could land in them without failing anything.

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    list[Path]
        Sorted package Python source paths.
    """

    return sorted((repo / "torchlens").rglob("*.py"))


def _module_level_mutable_bindings(tree: ast.Module) -> dict[str, str]:
    """Return module-level names bound to a mutable container, with their source.

    Parameters
    ----------
    tree:
        Parsed module.

    Returns
    -------
    dict[str, str]
        Name -> unparsed initializer for each module-level mutable binding.
    """

    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        mutable = isinstance(
            value, (ast.Dict, ast.Set, ast.List, ast.DictComp, ast.SetComp, ast.ListComp)
        )
        if isinstance(value, ast.Call):
            factory = value.func
            factory_name = (
                factory.id
                if isinstance(factory, ast.Name)
                else factory.attr
                if isinstance(factory, ast.Attribute)
                else None
            )
            mutable = mutable or factory_name in MUTABLE_FACTORIES
        if not mutable:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = ast.unparse(value)
    return bindings


def _names_mutated_in_place(trees: dict[str, ast.Module]) -> set[str]:
    """Return every name mutated in place anywhere in the package.

    Both spellings count, because module state is routinely mutated through an
    imported module alias: a bare ``_CACHE[key] = value`` in the owning module
    AND an ``_state._dir_cache[key] = value`` from another one. The attribute
    spelling is matched by ATTRIBUTE NAME, which can over-include a same-named
    attribute on an unrelated object; over-inclusion only ever asks for one more
    classification row, whereas under-inclusion is the blind spot this closes.

    Parameters
    ----------
    trees:
        Relative path -> parsed module for the whole package.

    Returns
    -------
    set[str]
        Names observed under an in-place mutation.
    """

    mutated: set[str] = set()
    for tree in trees.values():
        for node in _walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
                targets = (
                    node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                )
                for target in targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    base = target.value
                    if isinstance(base, ast.Name):
                        mutated.add(base.id)
                    elif isinstance(base, ast.Attribute):
                        mutated.add(base.attr)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in MUTATING_METHODS
            ):
                base = node.func.value
                if isinstance(base, ast.Name):
                    mutated.add(base.id)
                elif isinstance(base, ast.Attribute):
                    mutated.add(base.attr)
    return mutated


def _module_alias_targets(
    relative: str, tree: ast.Module, module_files: frozenset[str]
) -> dict[str, str]:
    """Map local names bound to imported TORCHLENS modules onto their files.

    Resolves both absolute (``from torchlens import _state``) and relative
    (``from ... import _state``, ``from ..utils import rng as rng_mod``)
    module imports, so attribute rebinds through the alias can be attributed
    to the OWNING module.

    Parameters
    ----------
    relative:
        Module path relative to the repository root.
    tree:
        Parsed module.
    module_files:
        Every package Python file, as repo-relative POSIX paths.

    Returns
    -------
    dict[str, str]
        Local alias name -> owning module's repo-relative path.
    """

    package_parts = relative[: -len(".py")].split("/")[:-1]
    if relative.endswith("/__init__.py"):
        package_parts = relative[: -len("/__init__.py")].split("/")
    aliases: dict[str, str] = {}
    for node in _walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is None or node.module.split(".")[0] != "torchlens":
                    continue
                base = node.module.split(".")
            else:
                keep = len(package_parts) - (node.level - 1)
                if keep < 0:
                    continue
                base = package_parts[:keep]
                if node.module:
                    base = [*base, *node.module.split(".")]
            for alias in node.names:
                module_candidate = "/".join([*base, alias.name]) + ".py"
                package_candidate = "/".join([*base, alias.name, "__init__.py"])
                if module_candidate in module_files:
                    aliases[alias.asname or alias.name] = module_candidate
                elif package_candidate in module_files:
                    aliases[alias.asname or alias.name] = package_candidate
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "torchlens":
                    continue
                module_candidate = alias.name.replace(".", "/") + ".py"
                package_candidate = alias.name.replace(".", "/") + "/__init__.py"
                local = alias.asname or alias.name.split(".")[0]
                if alias.asname is None and "." in alias.name:
                    continue  # ``import torchlens.x`` binds only ``torchlens``
                if module_candidate in module_files:
                    aliases[local] = module_candidate
                elif package_candidate in module_files:
                    aliases[local] = package_candidate
    return aliases


def _cross_module_attribute_rebinds(
    trees: dict[str, ast.Module], module_files: frozenset[str]
) -> set[tuple[str, str]]:
    """Return module globals rebound THROUGH an imported-module attribute.

    ``_declared_globals`` walks ``ast.Global`` only, so a control slot that is
    declared in one module and assigned exclusively from OTHERS
    (``_state._nonowner_belt_armed = True`` in ``_completeness_patches.py``)
    could never be classified: four live per-capture guards evaded the
    purported whole-package inventory this way (hunt-b2-sol R54). Attribute
    rebinds through a resolved torchlens module alias are attributed to the
    OWNING module.

    Parameters
    ----------
    trees:
        Relative path -> parsed module for the whole package.
    module_files:
        Every package Python file, as repo-relative POSIX paths.

    Returns
    -------
    set[tuple[str, str]]
        ``(owning module relative path, attribute name)`` rebind sites.
    """

    rebinds: set[tuple[str, str]] = set()
    for relative, tree in trees.items():
        aliases = _module_alias_targets(relative, tree, module_files)
        if not aliases:
            continue
        for node in _walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    owner = aliases.get(target.value.id)
                    if owner is not None:
                        rebinds.add((owner, target.attr))
    return rebinds


def _mutable_module_state(repo: Path) -> dict[tuple[str, str], str]:
    """Return every mutable module global in the package, with its initializer.

    Three detectors, because each alone has a structural blind spot:

    * ``global`` declarations catch REBINDING (``_flag = True``) but can never
      see a container mutated in place -- ``_CACHE[key] = value`` needs no
      ``global`` statement at all, so the whole cache/registry class was
      invisible to the previous gate.
    * Module-level mutable bindings catch the container class, qualified by
      package-wide evidence that something actually mutates them, so frozen
      lookup tables are not dragged in.
    * Cross-module attribute rebinds (``_state.flag = value`` from another
      module) need no ``global`` statement in ANY module, so scalar control
      slots assigned only through the module object were invisible to both
      detectors above (hunt-b2-sol R54).

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    dict[tuple[str, str], str]
        ``(relative path, name)`` -> unparsed initializer (empty for names known
        only from a ``global`` declaration).
    """

    return _mutable_module_state_cached(repo)


@functools.lru_cache(maxsize=1)
def _mutable_module_state_cached(repo: Path) -> dict[tuple[str, str], str]:
    """One whole-package scan per session, shared by every census consumer.

    The scan is parse-dominated; the whole-package census, the weak-ledger check
    and the hot-subset census below all read this one result.
    """

    trees = {
        path.relative_to(repo).as_posix(): package_ast(path) for path in _package_python_paths(repo)
    }
    mutated_names = _names_mutated_in_place(trees)
    state: dict[tuple[str, str], str] = {}
    for relative, tree in trees.items():
        bindings = _module_level_mutable_bindings(tree)
        for node in _walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    state.setdefault((relative, name), bindings.get(name, ""))
        for name, initializer in bindings.items():
            if name in mutated_names:
                state[(relative, name)] = initializer
    module_files = frozenset(trees)
    for owner, attribute in _cross_module_attribute_rebinds(trees, module_files):
        state.setdefault((owner, attribute), "")
    return state


def global_state_inventory_is_classified_and_shrink_only() -> None:
    """Every mutable module global in the PACKAGE has exactly one lifecycle class.

    Scope and detection are both wider than they were. The inventory covered
    ``_state.py`` plus three subpackages, and classified only ``global``
    declarations -- so the majority of the surface it claimed to govern was
    unreachable by it, and the entire mutated-in-place class (``_CACHE[key] =
    value`` needs no ``global`` statement) was structurally invisible even
    in-lane. See ``_package_python_paths`` and ``_mutable_module_state``.
    """

    repo = REPO_ROOT
    classified = set().union(*_LIFECYCLE_CLASSES)

    assert sum(len(category) for category in _LIFECYCLE_CLASSES) == len(classified), (
        "global lifecycle classes overlap"
    )
    observed = _mutable_module_state(repo)
    missing = sorted(set(observed) - classified)
    stale = sorted(classified - set(observed))
    assert not missing, (
        "unclassified mutable module state (assign it a lifecycle class in "
        f"scripts/checks/global_state_rows.py): {missing}"
    )
    assert not stale, f"inventory rows no longer present in the package: {stale}"


#: Historically hot state-bearing surface for the SMOKE-tier census sentinel:
#: the capture/wrapper territory where the b2p2-F2 incident landed 24
#: unclassified globals across 3 commits with no commit-gate red, plus the
#: root-level state modules. Prefix-matched against package-relative paths.
_HOT_STATE_SUBSET_PREFIXES = (
    "torchlens/_state.py",
    "torchlens/_capture_state_helpers.py",
    "torchlens/_save_budget.py",
    "torchlens/backends/torch/",
    "torchlens/_capture/",
    "torchlens/recording/",
    "torchlens/_utils/rng/",
    "torchlens/_utils/_torch_compat.py",
)


def hot_state_subset_census() -> None:
    """Commit-gate sentinel over the hot capture/wrapper state surface.

    The b2p2-F2 incident landed 24 unclassified globals in the capture and
    wrapper territory across three commits with no commit-gate red. This census
    over that territory reports on its own line, so a failure there names the
    incident class; the whole-package census above is the exhaustive authority.
    """

    repo = REPO_ROOT
    classified = set().union(*_LIFECYCLE_CLASSES)
    subset_paths = [
        path
        for path in _package_python_paths(repo)
        if path.relative_to(repo).as_posix().startswith(_HOT_STATE_SUBSET_PREFIXES)
    ]
    trees = {path.relative_to(repo).as_posix(): package_ast(path) for path in subset_paths}
    mutated_names = _names_mutated_in_place(trees)
    observed: set[tuple[str, str]] = set()
    for relative, tree in trees.items():
        bindings = _module_level_mutable_bindings(tree)
        for node in _walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    observed.add((relative, name))
        for name in bindings:
            if name in mutated_names:
                observed.add((relative, name))
    for owner, attribute in _cross_module_attribute_rebinds(trees, frozenset(trees)):
        observed.add((owner, attribute))
    missing = sorted(observed - classified)
    assert not missing, (
        "unclassified mutable module state landed in the hot capture/wrapper "
        "surface (assign each a lifecycle class in scripts/checks/global_state_rows.py; the "
        f"whole-package census is the exhaustive authority): {missing}"
    )


def weakly_held_state_is_exactly_the_declared_ledger() -> None:
    """The weak/strong split of every inventory member is frozen and exact.

    This is what keeps the inventory from being a rubber stamp. A lifecycle row
    says what a global is FOR; this says how it HOLDS its subjects, which is the
    difference between a side table that dies with its trace and one that pins
    every captured graph in the process. Both directions are checked, so neither
    weakening nor strengthening a container can pass unreviewed.
    """

    repo = REPO_ROOT
    observed = _mutable_module_state(repo)
    weakly_held = {
        entry for entry, initializer in observed.items() if "weakref.Weak" in initializer
    }

    became_strong = sorted(_WEAKLY_HELD - weakly_held)
    became_weak = sorted(weakly_held - _WEAKLY_HELD)
    assert not became_strong, (
        "declared-weak module state is no longer bound to a weakref container "
        f"(it now pins its subjects): {became_strong}"
    )
    assert not became_weak, (
        "module state became weak without updating the ledger (welcome, but the "
        f"row has to move): {became_weak}"
    )
    assert set().union(*_LIFECYCLE_CLASSES) >= _WEAKLY_HELD, (
        "weak-ledger rows missing from every lifecycle class"
    )
    assert _WEAK_SUBJECT_TABLES <= _WEAKLY_HELD, (
        "a member of the weak-subject-table CLASS is not in the weak ledger"
    )


CHECKS = (
    global_state_inventory_is_classified_and_shrink_only,
    hot_state_subset_census,
    weakly_held_state_is_exactly_the_declared_ledger,
)


def main() -> int:
    """Run every global_state_inventory check and return the exit status."""

    return run_checks("global_state_inventory", CHECKS)


if __name__ == "__main__":
    sys.exit(main())
