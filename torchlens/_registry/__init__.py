"""Private registry kernel package (architecture memo 6.1; never public API).

One private kernel; many typed public domain doors; no public universal
registry. Domain doors import from here; users import the doors.
"""

from __future__ import annotations

from .kernel import (
    TORCHLENS_PROVIDER,
    MissingProviderBehavior,
    ProviderInfo,
    RegistrationInfo,
    Registry,
    RegistryError,
    RegistrySnapshot,
    create_registry,
    kernel_registries,
    kernel_universe_rows,
)

__tl_layer__ = "L1"

__all__ = [
    "TORCHLENS_PROVIDER",
    "MissingProviderBehavior",
    "ProviderInfo",
    "RegistrationInfo",
    "Registry",
    "RegistryError",
    "RegistrySnapshot",
    "create_registry",
    "kernel_registries",
    "kernel_universe_rows",
]
