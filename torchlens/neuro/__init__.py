"""Extras-gated neuroscience namespace with no public objects yet.

Import time stays inert: this module must NOT import ``rsatoolbox`` (the
``neuro`` extra's foreign third-party dependency) as a side effect of
``import torchlens.neuro``. A bare package import can happen incidentally --
e.g. a portable ``.tlspec`` bundle's metadata unpickler resolving a pickled
global whose module path names ``torchlens.neuro`` -- and an eager
import-time dependency check would run that foreign code with no trust
opt-in.

Attribute access resolves through the shared five-step facade order (neuro
memo D15; ``torchlens.utils.facade``). The historical ALL-OR-NOTHING
conjunction gate (rsatoolbox AND brainscore_core, checked by IMPORTING them
on every attribute access) is gone: it locked the whole namespace on Python
3.10 even with rsatoolbox present, made ``hasattr`` raise ``ImportError``
instead of answering, executed foreign code to answer dunder probes, and
demanded a package nothing here imports. Dependency checks are now PER-NAME
(each future adapter declares its own ``DependencyGate``), probed with
``importlib.util.find_spec`` so answering never executes foreign code, and
every typed refusal subclasses ``AttributeError`` so ``hasattr``/IPython
canary probes degrade instead of erroring. The facade is pickle-safe:
resolving this module imports nothing foreign.

The neuro extra installs ``rsatoolbox`` (``pip install "torchlens[neuro]"``);
adapters landing in this namespace gate on it per-name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from typing import Any

__all__: list[str] = []

#: Real active names: none yet. Adapters land here with per-name
#: ``DependencyGate`` rows in ``_DEPENDENCIES``.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {}

#: Redirect/refusal teaching tables (five-step steps 2-3). Content lands with
#: the neuro teaching-surface lane; the resolution order is declared now.
_REDIRECTS: dict[str, str] = {}
_REFUSALS: dict[str, str] = {}

#: Per-name dependency gates (five-step step 4), e.g.
#: ``{"datasets": DependencyGate("rsatoolbox", 'pip install "torchlens[neuro]"')}``.
_DEPENDENCIES: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Resolve attributes through the shared five-step facade order.

    Parameters
    ----------
    name:
        Requested module attribute name.

    Returns
    -------
    Any
        The resolved attribute (none exist yet; every access teaches).

    Raises
    ------
    AttributeError
        Per the five-step contract: plain for underscore and unknown names,
        typed teaching subclasses for redirect/refusal rows, and a typed
        ``ImportError``-and-``AttributeError`` for gated names whose
        dependency is absent.
    """

    from ..utils.facade import resolve_facade_attr

    return resolve_facade_attr(
        owner=__name__,
        name=name,
        module_globals=globals(),
        lazy_attrs=_LAZY_ATTRS,
        redirects=_REDIRECTS,
        refusals=_REFUSALS,
        dependencies=_DEPENDENCIES,
    )


def __dir__() -> list[str]:
    """Return the real public names of the namespace and nothing else.

    Returns
    -------
    list[str]
        Sorted public names; accidental implementation imports are not
        advertised.
    """

    from ..utils.facade import facade_dir

    return facade_dir(globals(), _LAZY_ATTRS)


# ``from __future__ import annotations`` binds ``annotations`` as a reachable
# module attribute; nothing reads the binding (the future feature is a
# compile-time flag), so unbind it -- the root facade's own idiom.
del annotations
