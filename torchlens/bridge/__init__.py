"""External-tool bridge namespace.

Every bridge module is import-inert without its foreign peer (the L8 rule):
resolving ``tl.bridge.captum`` imports only TorchLens code, and the peer
import is deferred into the functions that need it. Attribute access resolves
through the shared five-step facade order (``torchlens.utils.facade``), so
``hasattr`` probes and dunder lookups can never explode and a bridge module
whose own import fails surfaces as a typed error that is BOTH an
``ImportError`` and an ``AttributeError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from types import ModuleType

_BRIDGE_MODULES = {
    "brain_score",
    "captum",
    "depyf",
    "dialz",
    "gradcam",
    "hf",
    "huggingface",
    "inseq",
    "lit",
    "mcp",
    "nnsight",
    "profiler",
    "repeng",
    "rsatoolbox",
    "sae",
    "sae_lens",
    "shap",
    "steering_vectors",
}


def __getattr__(name: str) -> ModuleType:
    """Import bridge modules lazily through the five-step facade order.

    Parameters
    ----------
    name:
        Bridge module name.

    Returns
    -------
    ModuleType
        Imported bridge module.

    Raises
    ------
    AttributeError
        Per the five-step contract; unknown names raise plain
        ``AttributeError``.
    """

    from ..utils.facade import resolve_facade_attr

    resolved: ModuleType = resolve_facade_attr(
        owner=__name__,
        name=name,
        module_globals=globals(),
        submodules=_BRIDGE_MODULES,
    )
    return resolved


def __dir__() -> list[str]:
    """Return visible bridge namespace members.

    Returns
    -------
    list[str]
        Sorted bridge module names plus real public globals.
    """

    from ..utils.facade import facade_dir

    return facade_dir(globals(), _BRIDGE_MODULES)


__all__ = [
    "brain_score",
    "captum",
    "depyf",
    "dialz",
    "gradcam",
    "hf",
    "huggingface",
    "inseq",
    "lit",
    "mcp",
    "nnsight",
    "profiler",
    "repeng",
    "rsatoolbox",
    "sae",
    "sae_lens",
    "shap",
    "steering_vectors",
]

# ``from __future__ import annotations`` binds ``annotations`` as a reachable
# module attribute; nothing reads the binding (the future feature is a
# compile-time flag), so unbind it -- the root facade's own idiom.
del annotations
