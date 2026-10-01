"""Shared five-step lazy-facade resolution for TorchLens namespaces.

THE one PEP-562 ``__getattr__`` mechanism (architecture memo section 5.4,
five-step testable spec, D15): the root package, every appliance namespace
(``torchlens.neuro``, ``torchlens.notebook``), and the integration namespaces
(``torchlens.bridge``, ``torchlens.callbacks``) resolve attributes through the
same ordered steps, so ``hasattr``/``getattr``-with-default can never explode,
IDE/notebook dunder probes fail fast, and dependency teaching happens per-name
instead of as an all-or-nothing namespace gate.

The five steps, in order:

1. Underscore-prefixed name -> plain ``AttributeError`` immediately, no
   dependency check (IPython canaries, pickle protocol probes, dunders).
2. Redirect table -> typed teaching ``AttributeError`` naming the canonical
   spelling, NO dependency check (a redirect needs no foreign package).
3. Refusal table -> typed teaching ``AttributeError``, no dependency check.
4. Real active name -> per-name dependency gate: if the name declares a
   foreign dependency that is not installed, a typed ``AttributeError``
   subclass names the exact package and install command; otherwise the
   target resolves and is cached in the owning module's globals.
5. Otherwise -> plain ``AttributeError``.

Every typed error subclasses ``AttributeError`` so ``hasattr`` answers
``False`` instead of propagating; the dependency-gate error carries the
ImportError SEMANTICS (exact package + install command, in the message and
on ``fields``) without ``ImportError`` lineage, which CPython's
instance-layout conflict forbids alongside ``AttributeError``. Resolution
never imports foreign code to answer a NEGATIVE:
dependency availability is probed with ``importlib.util.find_spec`` (a
metadata/path search that executes nothing), which also keeps the facade
pickle-safe -- unpickling a reference into a facade namespace imports nothing
foreign to discover the name does not resolve.

Spellings here are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Collection, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from .._errors import FacadeTeachingError, MissingDependencyError

__all__ = [
    "DependencyGate",
    "facade_dir",
    "resolve_facade_attr",
]


@dataclass(frozen=True)
class DependencyGate:
    """Per-name foreign-dependency declaration for facade step 4.

    Parameters
    ----------
    module:
        Top-level importable module name of the foreign dependency
        (``"rsatoolbox"``), probed with ``importlib.util.find_spec`` so the
        gate itself never executes foreign code.
    install:
        Exact install command taught when the dependency is absent
        (``'pip install "torchlens[neuro]"'``).
    """

    module: str
    install: str


def _plain_missing(owner: str, name: str) -> AttributeError:
    """Return the plain step-1/step-5 AttributeError."""

    return AttributeError(f"module {owner!r} has no attribute {name!r}")


def resolve_facade_attr(
    *,
    owner: str,
    name: str,
    module_globals: MutableMapping[str, Any],
    lazy_attrs: Mapping[str, tuple[str, str | None]] | None = None,
    submodules: Collection[str] | None = None,
    redirects: Mapping[str, str] | None = None,
    refusals: Mapping[str, str] | None = None,
    dependencies: Mapping[str, DependencyGate] | None = None,
) -> Any:
    """Resolve one facade attribute through the five-step order.

    Parameters
    ----------
    owner:
        Fully qualified name of the owning module (``"torchlens.neuro"``).
    name:
        Attribute requested from the owning module.
    module_globals:
        The owning module's ``globals()``; successful resolutions are cached
        there so subsequent access bypasses ``__getattr__`` entirely.
    lazy_attrs:
        Real active names: ``name -> (module_path, attr_name_or_None)``.
        ``None`` as the attr name resolves to the module object itself.
    submodules:
        Real active child-module names resolved as ``owner.name`` (the
        ``torchlens.bridge`` idiom). A name present in both tables resolves
        through ``lazy_attrs`` first.
    redirects:
        Step-2 teaching rows: ``name -> canonical spelling / guidance``.
    refusals:
        Step-3 teaching rows: ``name -> refusal message``.
    dependencies:
        Step-4 per-name gates: ``name -> DependencyGate``. Only consulted for
        real active names.

    Returns
    -------
    Any
        The resolved attribute.

    Raises
    ------
    AttributeError
        Steps 1 and 5 (plain), steps 2 and 3 (typed teaching subclasses),
        and step 4 (typed, with the missing package and install command on
        the message and ``fields``) per the module contract.
    """

    # Step 1: underscore short-circuit -- no table lookup, no dependency
    # check. Dunder probes from IPython, copyreg, and inspect must fail fast.
    if name.startswith("_"):
        raise _plain_missing(owner, name)

    # Step 2: redirect table -> typed teaching AttributeError, no dep check.
    if redirects and name in redirects:
        raise FacadeTeachingError(
            f"{owner}.{name} is not an active name",
            code="facade_redirect",
            remedy=redirects[name],
            owner=owner,
            attribute=name,
        )

    # Step 3: refusal table -> typed teaching AttributeError, no dep check.
    if refusals and name in refusals:
        raise FacadeTeachingError(
            f"{owner}.{name} is deliberately not provided",
            code="facade_refusal",
            remedy=refusals[name],
            owner=owner,
            attribute=name,
        )

    # Step 4: real active name -> per-name dependency gate, then resolve.
    target: tuple[str, str | None] | None = None
    if lazy_attrs and name in lazy_attrs:
        target = lazy_attrs[name]
    elif submodules and name in submodules:
        target = (f"{owner}.{name}", None)
    if target is not None:
        gate = dependencies.get(name) if dependencies else None
        if gate is not None and importlib.util.find_spec(gate.module) is None:
            raise MissingDependencyError(
                f"{owner}.{name} requires the optional dependency "
                f"{gate.module!r}, which is not installed",
                code="facade_dependency_missing",
                remedy=f"install it with `{gate.install}`",
                owner=owner,
                attribute=name,
                dependency=gate.module,
                install=gate.install,
            )
        module_path, attr_name = target
        try:
            module_obj = importlib.import_module(module_path)
        except ImportError as exc:
            # A real name whose resolution import fails must still answer
            # hasattr with False, not explode it; the chained cause keeps the
            # true import failure diagnosable.
            raise MissingDependencyError(
                f"{owner}.{name} could not be resolved: importing {module_path!r} failed ({exc})",
                code="facade_dependency_missing",
                remedy=(
                    f"fix the failing import of {module_path!r} "
                    "(see the chained exception for the root cause)"
                ),
                owner=owner,
                attribute=name,
                dependency=module_path,
            ) from exc
        value = module_obj if attr_name is None else getattr(module_obj, attr_name)
        module_globals[name] = value
        return value

    # Step 5: everything else -> plain AttributeError.
    raise _plain_missing(owner, name)


def facade_dir(
    module_globals: Mapping[str, Any],
    *name_sets: Collection[str],
) -> list[str]:
    """Return the real public names of a facade namespace and nothing else.

    Redirect and refusal rows are teaching surfaces, not attributes, so they
    are deliberately absent; underscore-prefixed accidental globals
    (``importlib``, aliased typing helpers) and non-dunder private names are
    filtered so ``dir()`` stops advertising implementation imports. Module
    dunders present in the globals (``__name__``, ``__doc__``, ...) are kept
    for introspection tools.

    Parameters
    ----------
    module_globals:
        The owning module's ``globals()``.
    *name_sets:
        Additional real-name collections (lazy tables, submodule sets).

    Returns
    -------
    list[str]
        Sorted public names.
    """

    names: set[str] = set()
    for name in module_globals:
        if name.startswith("__") and name.endswith("__") or not name.startswith("_"):
            names.add(name)
    for name_set in name_sets:
        names.update(name for name in name_set if not name.startswith("_"))
    return sorted(names)
