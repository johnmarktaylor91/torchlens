"""The export-target registry: ONE L7 door for every exporter (C01 item 17).

Four panels were queued to add flat functions to a single-module package;
this door (architecture memo 6.3 seam 3) is where they register instead.
Members carry PER-MEMBER tier rows -- ``tl.export.netron``'s writer is
bridge-tier, ``tl.export.csv`` present-tier -- one namespace, one door, the
capability row is authoritative (namespace shape is necessary but
insufficient). Builtins register through this same public function at
package import; out-of-tree exporters use it identically.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .._registry import (
    TORCHLENS_PROVIDER,
    ProviderInfo,
    RegistrationInfo,
    RegistryError,
    create_registry,
)

__tl_layer__ = "L7"

#: Closed per-member tier vocabulary: present-tier emitters hold no foreign
#: shape; bridge-tier writers are shaped by a foreign peer (Netron, Model
#: Explorer, TensorBoard, W&B, MLflow, Aim).
EXPORT_TARGET_TIERS = frozenset({"present", "bridge"})

_EXPORT_REGISTRY = create_registry(
    "export_targets",
    kind_label="export target",
    missing_provider_behavior="typed_refusal",
)


def register_export_target(
    name: str,
    fn: Callable[..., Any],
    *,
    tier: str,
    capabilities: Mapping[str, Any] | None = None,
    provider: ProviderInfo | None = None,
    replace: bool = False,
    conformance_ref: str | None = None,
) -> RegistrationInfo:
    """Register one export target through the public door.

    Parameters
    ----------
    name:
        Stable target name (the ``tl.export.<name>`` spelling for builtins).
    fn:
        The exporter callable (``fn(log, ...)``).
    tier:
        Per-member tier row: ``"present"`` (native emitter) or ``"bridge"``
        (foreign-peer-shaped writer).
    capabilities:
        Extra capability rows; the door always records the tier.
    provider:
        Stable provider identity; defaults to the TorchLens builtin row.
    replace:
        Explicit replacement opt-in (collision refusal otherwise).
    conformance_ref:
        Optional pointer to the registration's conformance row.

    Returns
    -------
    RegistrationInfo
        The immutable registration metadata.
    """

    if not callable(fn):
        raise RegistryError(
            f"Export target {name!r} must be callable; got {type(fn).__name__!r}.",
            code="export_target_not_callable",
            entry_id=str(name),
            remedy="Pass the exporter function itself, not its result or its name.",
        )
    if tier not in EXPORT_TARGET_TIERS:
        raise RegistryError(
            f"Export target {name!r} declares unknown tier {tier!r}; the closed "
            f"vocabulary is {sorted(EXPORT_TARGET_TIERS)}.",
            code="export_target_tier_invalid",
            entry_id=str(name),
            tier=str(tier),
            remedy="Declare tier='present' for native emitters or tier='bridge' for foreign-peer writers.",
        )
    rows: dict[str, Any] = {"tier": tier}
    if capabilities:
        rows.update(capabilities)
    return _EXPORT_REGISTRY.register(
        name,
        fn,
        capabilities=rows,
        provider=provider or TORCHLENS_PROVIDER,
        replace=replace,
        conformance_ref=conformance_ref,
    )


def unregister_export_target(name: str) -> None:
    """Remove one export-target registration."""

    _EXPORT_REGISTRY.unregister(name)


def export_targets() -> tuple[str, ...]:
    """Return the registered export-target names."""

    return _EXPORT_REGISTRY.list_ids()


def export_target_info(name: str) -> RegistrationInfo:
    """Return one export target's registration metadata (tier row included)."""

    return _EXPORT_REGISTRY.info(name)


def resolve_export_target(name: str) -> Callable[..., Any]:
    """Return one registered exporter callable; unknown names refuse teaching."""

    return _EXPORT_REGISTRY.get(name)
