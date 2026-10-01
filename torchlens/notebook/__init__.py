"""Extras-gated notebook namespace with no public objects yet.

Import time stays inert: this module must NOT import ``IPython`` /
``jupyter_client`` (the ``notebook`` extra's foreign third-party
dependencies) as a side effect of ``import torchlens.notebook``. A bare
package import can happen incidentally -- e.g. a portable ``.tlspec``
bundle's metadata unpickler resolving a pickled global whose module path
names ``torchlens.notebook`` -- and an eager import-time dependency check
would run that foreign code with no trust opt-in.

Attribute access resolves through the shared five-step facade order
(``torchlens.utils.facade``): the historical all-or-nothing conjunction gate
(which IMPORTED the foreign dependencies on every attribute access and made
``hasattr`` raise ``ImportError`` instead of answering) is replaced by
per-name ``DependencyGate`` rows probed with ``importlib.util.find_spec``,
so answering an attribute probe never executes foreign code and every typed
refusal subclasses ``AttributeError``. The facade is pickle-safe.

The notebook extra installs the dependencies
(``pip install "torchlens[notebook]"``); adapters landing here gate on them
per-name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from typing import Any

__all__: list[str] = ["cards", "cardtree", "frontier"]

#: Real active names. Card generation is stdlib + torch only (treescope
#: memo decision 3): NO row here carries a ``DependencyGate`` -- the
#: notebook extra gates integration helpers, never whether ``_repr_html_``
#: can return safe HTML.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {}

#: Real active child modules (the ``torchlens.bridge`` idiom): the CardTree
#: IR (B1), the four cards (B2), and the diagnostic frontier query (B6).
_SUBMODULES: set[str] = {"cards", "cardtree", "frontier"}

#: Redirect/refusal teaching tables (five-step steps 2-3).
_REDIRECTS: dict[str, str] = {}
_REFUSALS: dict[str, str] = {}

#: Per-name dependency gates (five-step step 4), e.g.
#: ``{"widget": DependencyGate("IPython", 'pip install "torchlens[notebook]"')}``.
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
        submodules=_SUBMODULES,
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

    return facade_dir(globals(), _LAZY_ATTRS, _SUBMODULES)


# ``from __future__ import annotations`` binds ``annotations`` as a reachable
# module attribute; nothing reads the binding (the future feature is a
# compile-time flag), so unbind it -- the root facade's own idiom.
del annotations
