"""Reachable-surface walk and the 4-way classification (oracles D2/D3).

The enumeration root is what a user can actually reach: every non-underscore
attribute of the ``torchlens`` module (module layer) and every non-underscore
member of every class object reachable there (class layer). ``__all__`` is a
CLAIM gated against this walk, never the root itself -- the module-level
``summary`` report door whose bugs commissioned the oracles panel was outside
``__all__``, so an ``__all__``-rooted harness would never have tested it.

Classification is 4-way and closed:

- ``declared``   -- the name is in ``torchlens.__all__``.
- ``submodule``  -- the value is a module object (its own ``__all__`` is
  gated by tests/test_api_surface.py::test_submodules_have_all).
- ``deprecated`` -- the name is registered in the deprecation-door registry
  (``data/deprecated_doors.tsv``; empty today -- the package is pinned
  deprecation-free by tests/test_deprecation_inventory.py).
- ``undeclared`` -- everything else. An undeclared name is RED unless a
  dated KNOWN-GAP row licenses it (D25).

The CLASS layer classifies too (5-way, closed): every public member of every
reachable class must carry a source-witnessed LICENSE -- declared in the class
body, statically assigned in the class's defining module, installed by a
package-defined descriptor/callable, or inherited from a non-package base.
Anything else (runtime injections above all) is ``unlicensed`` and RED. The
licenses are LOCAL facts about committed source, so two sibling branches that
each consciously add class surface compose without either knowing the other
exists -- a frozen member inventory cannot do that (the T03 train bounce), and
it also breaks on Python/torch version bumps that move inherited members.

Both walkers take the namespace as an argument so the plants can hand them a
doctored namespace without touching the live package.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap
import types
from dataclasses import dataclass
from functools import cached_property

DECLARED = "declared"
SUBMODULE = "submodule"
DEPRECATED = "deprecated"
UNDECLARED = "undeclared"

#: The closed classification vocabulary. A fifth class is a design change,
#: not a data change.
CLASSIFICATIONS = (DECLARED, SUBMODULE, DEPRECATED, UNDECLARED)

#: Counted-set identity for the module-layer denominator (D8: no count
#: publishes without its counted-set definition beside it).
MODULE_SURFACE_COUNTED_SET = (
    "non-underscore attributes of the torchlens module object (dir()) UNION "
    "the package's public top-level filesystem submodules (pkgutil, "
    "unimported) -- import-state-independent; submodule CONTENTS are a "
    "declared deferred root"
)

#: Counted-set identity for the class-layer denominator.
CLASS_SURFACE_COUNTED_SET = (
    "non-underscore members (dir()) of every class object reachable as a "
    "module-level torchlens attribute, keyed by reachable name, with every "
    "public top-level filesystem submodule imported first (import-state "
    "normalization: opt-in installers like kernel_telemetry always counted); "
    "each member carries a source-witnessed license from the closed 5-way "
    "vocabulary CLASS_MEMBER_LICENSES"
)

CLASS_BODY = "class_body"
MODULE_ASSIGNED = "module_assigned"
INSTALLED_BY_PACKAGE = "installed_by_package"
INHERITED_EXTERNAL = "inherited_external"
UNLICENSED = "unlicensed"

#: The closed class-member license vocabulary (D3 closure by construction:
#: the walk is the root; the CLASSIFICATION RULES are the committed claim,
#: never a frozen member inventory). A sixth license is a design change.
CLASS_MEMBER_LICENSES = (
    CLASS_BODY,
    MODULE_ASSIGNED,
    INSTALLED_BY_PACKAGE,
    INHERITED_EXTERNAL,
    UNLICENSED,
)


@dataclass(frozen=True)
class ModuleSurfaceEntry:
    """One module-layer reachable name with its classification.

    Parameters
    ----------
    name:
        The reachable attribute name.
    classification:
        One of ``CLASSIFICATIONS``.
    """

    name: str
    classification: str


def _filesystem_submodules(namespace: types.ModuleType) -> frozenset[str]:
    """Enumerate the package's public top-level submodules WITHOUT importing.

    ``dir(package)`` only shows a submodule once something has imported it,
    so a raw ``dir()`` walk is an import-state-dependent denominator -- the
    walk would count differently depending on which tests ran first. The
    filesystem submodule set is deterministic by construction and every
    member is reachable (``import torchlens.<name>`` works), so the walk
    unions it in; classification never needs to import them.

    Parameters
    ----------
    namespace:
        The package (plants may pass a plain module; no ``__path__`` means
        an empty set).

    Returns
    -------
    frozenset[str]
        Public top-level submodule names.
    """

    search_path = getattr(namespace, "__path__", None)
    if search_path is None:
        return frozenset()
    return frozenset(
        info.name for info in pkgutil.iter_modules(search_path) if not info.name.startswith("_")
    )


def walk_module_surface(
    namespace: types.ModuleType,
    deprecated_doors: frozenset[str],
) -> tuple[ModuleSurfaceEntry, ...]:
    """Walk the module-layer reachable surface and classify every name.

    Parameters
    ----------
    namespace:
        The module object to walk (the live ``torchlens`` in gates; a
        doctored namespace in plants).
    deprecated_doors:
        Registered deprecated door names (the D21 registry).

    Returns
    -------
    tuple[ModuleSurfaceEntry, ...]
        One entry per non-underscore reachable name, sorted by name.
    """

    declared = frozenset(getattr(namespace, "__all__", ()))
    submodules = _filesystem_submodules(namespace)
    dir_names = frozenset(n for n in dir(namespace) if not n.startswith("_"))
    entries = []
    for name in sorted(dir_names | submodules):
        if name in declared:
            classification = DECLARED
        elif name in submodules or isinstance(getattr(namespace, name), types.ModuleType):
            classification = SUBMODULE
        elif name in deprecated_doors:
            classification = DEPRECATED
        else:
            classification = UNDECLARED
        entries.append(ModuleSurfaceEntry(name=name, classification=classification))
    return tuple(entries)


def _normalize_import_state(namespace: types.ModuleType) -> None:
    """Import every public top-level submodule so ``dir(cls)`` is stable.

    Opt-in submodules may install documented members on public classes at
    import time (``torchlens.kernel_telemetry`` installs ``gpu_kernels`` on
    ``Op``/``AtenOp``), so a raw ``dir(cls)`` walk counts differently
    depending on which tests ran first -- green alone, red in a session.
    Importing the deterministic filesystem submodule set pins the walk to
    ONE canonical maximal state; installers are idempotent by contract
    (re-import is a module-cache hit).

    Parameters
    ----------
    namespace:
        The package to normalize (plants pass plain modules; no
        ``__path__`` means nothing to import).
    """

    package_name = getattr(namespace, "__name__", "")
    if getattr(namespace, "__path__", None) is None or not package_name:
        return
    for submodule in sorted(_filesystem_submodules(namespace)):
        importlib.import_module(f"{package_name}.{submodule}")


def walk_class_surface(namespace: types.ModuleType) -> tuple[tuple[str, str], ...]:
    """Walk the class layer: public members of every reachable class.

    Classes declare no ``__all__``, so the class layer gets closure BY
    CONSTRUCTION (D3): the walk is the root, and the committed baseline is
    the claim. A new public member on ``Trace``/``Layer`` is a new public
    door and must land in the baseline consciously. The walk normalizes the
    import state first so the denominator is test-order-independent.

    Parameters
    ----------
    namespace:
        The module object whose module-level classes are walked.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Sorted ``(reachable_class_name, member_name)`` rows.
    """

    _normalize_import_state(namespace)
    rows: list[tuple[str, str]] = []
    for name in sorted(n for n in dir(namespace) if not n.startswith("_")):
        value = getattr(namespace, name)
        if not inspect.isclass(value):
            continue
        for member in dir(value):
            if not member.startswith("_"):
                rows.append((name, member))
    return tuple(sorted(rows))


_CLASS_BODY_CACHE: dict[type, frozenset[str]] = {}
_MODULE_ASSIGNED_CACHE: dict[str, dict[str, frozenset[str]]] = {}


def _class_body_names(cls: type) -> frozenset[str]:
    """Return the names declared in ``cls``'s own class-body source.

    The witness is the committed source itself (AST: defs, assignments, and
    annotated assignments in the class body), so it is a LOCAL fact -- two
    branches adding members to the same class each carry their own witness
    and compose under merge. Unparseable classes (dynamically created, no
    retrievable source) yield the empty set: fail closed, never guess.

    Parameters
    ----------
    cls:
        The class whose body to read.

    Returns
    -------
    frozenset[str]
        Names bound by class-body statements.
    """

    cached = _CLASS_BODY_CACHE.get(cls)
    if cached is not None:
        return cached
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
        body = tree.body[0]
        if not isinstance(body, ast.ClassDef):
            raise TypeError(f"source of {cls!r} did not parse to a ClassDef")
    except (OSError, TypeError, SyntaxError):
        _CLASS_BODY_CACHE[cls] = frozenset()
        return _CLASS_BODY_CACHE[cls]
    names: set[str] = set()
    for node in body.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.update(e.id for e in target.elts if isinstance(e, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    _CLASS_BODY_CACHE[cls] = frozenset(names)
    return _CLASS_BODY_CACHE[cls]


def _module_assigned_names(cls: type) -> frozenset[str]:
    """Return members statically assigned onto ``cls`` in its home module.

    ``Trace.FIELD_FORK_POLICY = ...`` at the top level of ``trace.py`` is as
    conscious as a class-body declaration -- it is committed, reviewed source
    in the class's own defining module. Only TOP-LEVEL attribute assignments
    naming the class count (a setattr inside a function is a runtime act and
    must earn ``installed_by_package`` through its value instead).

    Parameters
    ----------
    cls:
        The class whose defining module to read.

    Returns
    -------
    frozenset[str]
        Attribute names assigned as ``<ClassName>.<attr> = ...`` at module
        top level.
    """

    module_name = getattr(cls, "__module__", "") or ""
    per_class = _MODULE_ASSIGNED_CACHE.get(module_name)
    if per_class is None:
        per_class = {}
        module = importlib.import_module(module_name) if module_name else None
        try:
            tree = ast.parse(inspect.getsource(module)) if module else None
        except (OSError, TypeError, SyntaxError):
            tree = None
        if tree is not None:
            collected: dict[str, set[str]] = {}
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        collected.setdefault(target.value.id, set()).add(target.attr)
            per_class = {name: frozenset(attrs) for name, attrs in collected.items()}
        _MODULE_ASSIGNED_CACHE[module_name] = per_class
    return per_class.get(cls.__name__, frozenset())


def _attributed_module(value: object) -> str:
    """Return the module a class-dict value is attributable to.

    Properties attribute to their getter, class/staticmethods to the wrapped
    function, cached_properties to their function; anything else to its own
    ``__module__`` when it has one (functions, torchlens descriptor
    instances via their class) and to its TYPE's module otherwise (plain
    data such as a frozenset attributes to ``builtins`` and can never
    launder a runtime injection).

    Parameters
    ----------
    value:
        The raw ``vars(cls)`` entry (descriptors NOT invoked).

    Returns
    -------
    str
        The attributed module name; ``""`` when nothing is attributable.
    """

    target: object = value
    if isinstance(value, property):
        target = value.fget or value.fset or value.fdel
    elif isinstance(value, (classmethod, staticmethod)):
        target = value.__func__
    elif isinstance(value, cached_property):
        target = value.func
    module = getattr(target, "__module__", None)
    if module is None:
        module = type(target).__module__
    return module or ""


def _in_package(module_name: str, package: str) -> bool:
    """Return whether ``module_name`` lives under ``package``."""

    return module_name == package or module_name.startswith(package + ".")


def classify_class_member(cls: type, member: str, package: str) -> str:
    """Classify one public class member under the closed license vocabulary.

    The provider is the FIRST class on the MRO whose ``__dict__`` carries the
    member -- the class attribute lookup actually serves it, so a runtime
    injection shadowing an inherited name is judged as the injection, never
    laundered through the base it shadows.

    Parameters
    ----------
    cls:
        The reachable class object.
    member:
        The non-underscore member name from the walk.
    package:
        The package root whose committed source licenses members (the live
        gates pass ``"torchlens"``; plants may license a doctored package).

    Returns
    -------
    str
        One of ``CLASS_MEMBER_LICENSES``.
    """

    for base in cls.__mro__:
        if member not in vars(base):
            continue
        if not _in_package(getattr(base, "__module__", "") or "", package):
            return INHERITED_EXTERNAL
        if member in _class_body_names(base):
            return CLASS_BODY
        if member in _module_assigned_names(base):
            return MODULE_ASSIGNED
        if _in_package(_attributed_module(vars(base)[member]), package):
            return INSTALLED_BY_PACKAGE
        return UNLICENSED
    return UNLICENSED


def walk_class_surface_classified(
    namespace: types.ModuleType,
) -> tuple[tuple[str, str, str], ...]:
    """Walk the class layer and license every member (D3, composition-safe).

    Same denominator as :func:`walk_class_surface`; each row additionally
    carries its license so the gate asserts the CLASSIFICATION RULES -- a
    frozen member inventory cannot compose across sibling branches that each
    consciously add surface (the T03 train bounce), and it breaks on version
    bumps that move inherited members.

    Parameters
    ----------
    namespace:
        The module object whose module-level classes are walked; its
        ``__name__`` is the licensing package root.

    Returns
    -------
    tuple[tuple[str, str, str], ...]
        Sorted ``(reachable_class_name, member_name, license)`` rows.
    """

    package = getattr(namespace, "__name__", "") or ""
    return tuple(
        (name, member, classify_class_member(getattr(namespace, name), member, package))
        for name, member in walk_class_surface(namespace)
    )


def surface_counts(entries: tuple[ModuleSurfaceEntry, ...]) -> dict[str, int]:
    """Return per-classification counts plus the total.

    Parameters
    ----------
    entries:
        Walk output from :func:`walk_module_surface`.

    Returns
    -------
    dict[str, int]
        Counts keyed by classification, plus ``"total"``.
    """

    counts = dict.fromkeys(CLASSIFICATIONS, 0)
    for entry in entries:
        counts[entry.classification] += 1
    counts["total"] = len(entries)
    return counts
