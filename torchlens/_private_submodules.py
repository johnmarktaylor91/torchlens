"""Typed refusals for submodule paths that 3.0 made private.

The public import paths of a namespace are the names it lists in ``__all__``
plus the taught module paths (``torchlens.backends.torch``,
``torchlens.backends.registry``, ``torchlens.intervention.errors``,
``torchlens.intervention.hooks``, ``torchlens.intervention.stochastic`` and the
per-framework backend namespaces ``torchlens.backends.jax``, ``mlx``, ``paddle``,
``tf`` and ``tinygrad``).
Every other submodule carries a leading underscore. Reading one of the old
spellings as an attribute of its parent (``tl.intervention.types``) raises the
typed ``facade_redirect`` error from the parent's ``__getattr__``.
"""

from __future__ import annotations

from ._errors import FacadeTeachingError

#: Parent namespace -> submodule names it no longer exposes.
PRIVATE_SUBMODULES: dict[str, frozenset[str]] = {
    "torchlens.agent": frozenset({"cli"}),
    "torchlens.attribution": frozenset({"graph", "onebackward"}),
    "torchlens.backends": frozenset({"default_specs"}),
    "torchlens.bridge": frozenset({"hf_attention"}),
    "torchlens.errors": frozenset({"episode", "runnable"}),
    "torchlens.intervention": frozenset(
        {
            "audit",
            "binding",
            "bundle",
            "edge_semantics",
            "edge_substitution",
            "handles",
            "helpers",
            "injection",
            "masked_edit",
            "model_door",
            "param_substitution",
            "predicates",
            "regions",
            "replay",
            "rerun",
            "runtime",
            "save",
            "site_keys",
            "spec",
            "spec_compat",
            "sweep",
            "types",
        },
    ),
    "torchlens.recording": frozenset(
        {"cleanup", "exceptions", "options", "storage_disk", "storage_ram", "types"},
    ),
    "torchlens.validation": frozenset(
        {
            "backward",
            "consolidated",
            "core",
            "diagnostics",
            "exemptions",
            "invariants",
            "status",
        },
    ),
    "torchlens.viz": frozenset({"batch_summary", "node_plots"}),
}


def _remedy(owner: str, name: str) -> str:
    """Return the remedy text for one private submodule spelling."""

    return (
        f"the module {owner}.{name} is private; import its public names from {owner} "
        f"(dir({owner}) lists them)"
    )


def refuse_private_submodule(owner: str, name: str) -> None:
    """Raise the typed redirect when ``name`` is a submodule ``owner`` made private.

    Parameters
    ----------
    owner:
        Fully qualified parent namespace (``"torchlens.intervention"``).
    name:
        Attribute the parent's ``__getattr__`` was asked for.

    Raises
    ------
    FacadeTeachingError
        When ``owner.name`` is a submodule path made private in 3.0.
    """

    if name in PRIVATE_SUBMODULES.get(owner, ()):
        raise FacadeTeachingError(
            f"{owner}.{name} is not a public module",
            code="facade_redirect",
            remedy=_remedy(owner, name),
            owner=owner,
            attribute=name,
        )


__all__ = ["PRIVATE_SUBMODULES", "refuse_private_submodule"]
