"""The transforms domain door over the private registry kernel (memo B1).

One registry in the S4 predicate-registry shape: shipped populated and live,
refusals name the registry, builtin names are a CLOSED SET and
non-replaceable, custom registrations are versioned and cannot shadow them.
Registration is what makes a transform countable (and rehydratable from an
artifact record without importing user code).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from .._registry import TORCHLENS_PROVIDER, ProviderInfo, Registry, create_registry
from ._errors import TransformContractError
from ._spec import TransformDefinition

__tl_layer__ = "L4"

__all__ = ["lookup_transform", "register_transform", "registered_transform_names"]

#: The one transforms registry (kernel-enrolled at import).
_TRANSFORMS_REGISTRY: Registry[TransformDefinition] = create_registry(
    "transforms", kind_label="activation transform"
)

#: Closed builtin name set; populated by ``_kernels`` through the SAME door
#: custom registrations use, then frozen by ``_seal_builtins``.
_BUILTIN_NAMES: set[str] = set()
_BUILTINS_SEALED: bool = False


def _register_builtin(definition: TransformDefinition) -> None:
    """Register one TorchLens builtin through the public door path.

    Parameters
    ----------
    definition:
        The builtin's :class:`TransformDefinition`.
    """

    if _BUILTINS_SEALED:
        raise TransformContractError(
            f"Builtin transform registration for {definition.name!r} arrived "
            "after the builtin set was sealed; the builtin name set is closed.",
            code="transform_builtin_shadowed",
            remedy="register user transforms via register_transform() instead",
            name=definition.name,
        )
    _TRANSFORMS_REGISTRY.register(
        definition.name,
        definition,
        capabilities={
            "kind": "builtin",
            "version": definition.version,
            "context_capable": definition.context_capable,
            "stream_safe": definition.stream_safe,
            "zero_param_preset": definition.zero_param_preset,
            "stochastic": definition.stochastic,
        },
        provider=TORCHLENS_PROVIDER,
    )
    _BUILTIN_NAMES.add(definition.name)


def _seal_builtins() -> None:
    """Freeze the builtin name set after ``_kernels`` finishes registering."""

    global _BUILTINS_SEALED
    _BUILTINS_SEALED = True


def register_transform(
    definition: TransformDefinition,
    *,
    provider: ProviderInfo | None = None,
) -> None:
    """Register a user transform definition (versioned; cannot shadow builtins).

    Parameters
    ----------
    definition:
        The transform's :class:`~torchlens.transforms.TransformDefinition`.
    provider:
        Optional provider identity for out-of-tree distributions; defaults to
        an anonymous user provider row.

    Raises
    ------
    TransformContractError
        ``transform_builtin_shadowed`` when the name collides with a builtin
        (builtin names are closed and non-replaceable).
    """

    if not isinstance(definition, TransformDefinition):
        raise TransformContractError(
            f"register_transform() needs a TransformDefinition; got {type(definition).__name__}.",
            code="transform_coercion_invalid",
            remedy="build a TransformDefinition and pass it",
            value_type=type(definition).__name__,
        )
    if definition.name in _BUILTIN_NAMES:
        raise TransformContractError(
            f"Transform name {definition.name!r} is a TorchLens builtin; "
            "builtin names are a closed set and cannot be shadowed or "
            "replaced.",
            code="transform_builtin_shadowed",
            remedy="pick a distinct name for the custom transform",
            name=definition.name,
        )
    _TRANSFORMS_REGISTRY.register(
        definition.name,
        definition,
        capabilities={
            "kind": "custom",
            "version": definition.version,
            "context_capable": definition.context_capable,
            "stream_safe": definition.stream_safe,
            "zero_param_preset": definition.zero_param_preset,
            "stochastic": definition.stochastic,
        },
        provider=provider if provider is not None else ProviderInfo(provider_id="user"),
    )


def lookup_transform(name: str) -> TransformDefinition:
    """Resolve a registered transform name or refuse naming the registry.

    Parameters
    ----------
    name:
        Registered transform name.

    Returns
    -------
    TransformDefinition
        The registered definition.

    Raises
    ------
    TransformContractError
        ``transform_name_unknown`` when the name is not registered.
    """

    snapshot = _TRANSFORMS_REGISTRY.snapshot()
    if name not in snapshot.values:
        raise TransformContractError(
            f"Transform name {name!r} is not registered in the 'transforms' "
            f"registry (registered: {sorted(snapshot.values)}). Strings "
            "never resolve as import paths or a parameter mini-language.",
            code="transform_name_unknown",
            remedy=(
                "use a builtin name, register the transform with "
                "torchlens.transforms.register_transform(), or pass the "
                "callable itself"
            ),
            name=name,
            registered=sorted(snapshot.values),
        )
    return snapshot.values[name]


def registered_transform_names() -> tuple[str, ...]:
    """Return the sorted names currently registered in the transforms registry.

    Returns
    -------
    tuple[str, ...]
        Sorted registered names (builtins plus customs).
    """

    return tuple(sorted(_TRANSFORMS_REGISTRY.snapshot().values))
