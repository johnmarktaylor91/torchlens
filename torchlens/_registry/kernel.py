"""The ONE private registry kernel behind every typed public domain door.

Registry law (architecture memo 6.1): one PRIVATE kernel, many typed public
domain doors, never a public universal registry. Every supported open class
gets lifecycle completeness (register / unregister / list / info / snapshot),
a monotone epoch, capability rows REQUIRED at registration, collision refusal
by default with explicit replacement, setup-time mutation under a lock with
immutable per-operation snapshots, epoch + provider version in cache keys,
provider distribution/version/stable-ID metadata, declared missing-provider
behavior, and a universe row (registration is what makes you countable).

This module is deliberately PRIVATE: users never import it. Domain doors
(export targets, renderers, sidecar families, transform providers, ...) wrap
one :class:`Registry` each with their own typed unit, coercion door, and
public spelling; builtins register through the door's own public function so
the in-tree path and the out-of-tree path are the same path.

Discriminator (memo 6.1): a name recorded in provenance, cache, artifact, or
refusal uses a REGISTRY; a local unnamed callable uses a CONTRACT.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar

from ..errors._base import ConfigurationError

__tl_layer__ = "L1"

T = TypeVar("T")

#: Capability row values must be portable scalars (or tuples of them): they
#: are recorded in provenance, cache keys, universe rows, and refusal text,
#: so an unrenderable value would poison every one of those surfaces.
_CAPABILITY_SCALARS = (bool, int, float, str)

#: Declared behavior when an artifact or request names a provider that is not
#: registered in this process: doors either refuse typed or degrade to a
#: declared analysis-only view. Never an import (memo 6.4: artifact load
#: never imports a provider).
MissingProviderBehavior = Literal["typed_refusal", "analysis_only"]


class RegistryError(ConfigurationError):
    """Raised for registry-kernel lifecycle refusals (stable ``fields['code']``)."""


@dataclass(frozen=True)
class ProviderInfo:
    """Stable identity of the party that registered an entry.

    Parameters
    ----------
    provider_id:
        Stable provider ID recorded in provenance and cache keys. Builtins
        registered by TorchLens itself use ``"torchlens"``.
    distribution:
        Optional installed-distribution name (e.g. the PyPI project) for
        out-of-tree providers.
    version:
        Optional provider version string; participates in cache keys so a
        provider upgrade invalidates caches keyed on the registry.
    """

    provider_id: str
    distribution: str | None = None
    version: str | None = None


#: The provider row TorchLens builtins register under (through the public
#: door of their domain, never through a kernel side channel).
TORCHLENS_PROVIDER = ProviderInfo(provider_id="torchlens", distribution="torchlens")


@dataclass(frozen=True)
class RegistrationInfo:
    """Immutable per-entry metadata served by ``info()`` and snapshots."""

    registry: str
    entry_id: str
    provider: ProviderInfo
    capabilities: Mapping[str, Any]
    value_type: str
    registered_epoch: int
    replaced_prior: bool
    conformance_ref: str | None = None


@dataclass(frozen=True)
class RegistrySnapshot(Generic[T]):
    """Immutable point-in-time view of one registry.

    Consumers that iterate (dispatch tables, renderers, exporters) take ONE
    snapshot per operation so a concurrent registration can never produce a
    half-old, half-new view.
    """

    name: str
    epoch: int
    entries: Mapping[str, RegistrationInfo]
    values: Mapping[str, T]

    def cache_key(self) -> str:
        """Return the epoch- and provider-version-keyed cache key."""

        digest_source = "|".join(
            f"{entry_id}:{info.provider.provider_id}:{info.provider.version or ''}"
            for entry_id, info in sorted(self.entries.items())
        )
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
        return f"{self.name}:e{self.epoch}:{digest}"


def _validate_capabilities(registry_name: str, entry_id: str, capabilities: Any) -> dict[str, Any]:
    """Validate and freeze one capability-row mapping.

    Capability rows are REQUIRED at registration: a seam that registers
    without capability declarations manufactures lies of absence (memo 6.1)
    -- an empty row would read as "this provider can do nothing" to one
    consumer and "capabilities unknown, assume everything" to another.
    """

    if not isinstance(capabilities, Mapping) or not capabilities:
        raise RegistryError(
            f"Registration {entry_id!r} in registry {registry_name!r} declares no "
            "capability rows. Capability rows are REQUIRED at registration: "
            "consumers gate on declared capabilities, and an absent row "
            "manufactures a lie of absence.",
            code="registry_capabilities_missing",
            registry=registry_name,
            entry_id=entry_id,
            remedy=(
                "Pass capabilities={...} declaring what the provider can and "
                "cannot do (boolean/str/int/float values)."
            ),
        )
    frozen: dict[str, Any] = {}
    for key, value in capabilities.items():
        if not isinstance(key, str) or not key:
            raise RegistryError(
                f"Registration {entry_id!r} in registry {registry_name!r} has a "
                f"non-string capability key {key!r}.",
                code="registry_capability_key_invalid",
                registry=registry_name,
                entry_id=entry_id,
                remedy="Use non-empty string capability names.",
            )
        if isinstance(value, tuple):
            if not all(isinstance(item, str) for item in value):
                raise RegistryError(
                    f"Capability {key!r} of registration {entry_id!r} in registry "
                    f"{registry_name!r} is a tuple with non-string members.",
                    code="registry_capability_value_invalid",
                    registry=registry_name,
                    entry_id=entry_id,
                    capability=key,
                    remedy="Capability tuples must contain only strings.",
                )
        elif not isinstance(value, _CAPABILITY_SCALARS):
            raise RegistryError(
                f"Capability {key!r} of registration {entry_id!r} in registry "
                f"{registry_name!r} has unportable value type "
                f"{type(value).__name__!r}. Capability rows ride provenance, "
                "cache keys, and refusal text, so values must be portable "
                "scalars.",
                code="registry_capability_value_invalid",
                registry=registry_name,
                entry_id=entry_id,
                capability=key,
                remedy="Use bool, int, float, str, or tuple-of-str capability values.",
            )
        frozen[key] = value
    return frozen


@dataclass
class Registry(Generic[T]):
    """One domain's registrations, owned by exactly one typed public door.

    Never exposed to users directly: doors wrap it. All mutation happens
    under the per-registry lock; every read that spans more than one entry
    goes through an immutable snapshot.
    """

    name: str
    kind_label: str
    missing_provider_behavior: MissingProviderBehavior = "typed_refusal"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _epoch: int = 0
    _values: dict[str, T] = field(default_factory=dict, repr=False)
    _entries: dict[str, RegistrationInfo] = field(default_factory=dict, repr=False)

    def register(
        self,
        entry_id: str,
        value: T,
        *,
        capabilities: Mapping[str, Any],
        provider: ProviderInfo,
        replace: bool = False,
        conformance_ref: str | None = None,
    ) -> RegistrationInfo:
        """Register one entry; collision refusal by default.

        Parameters
        ----------
        entry_id:
            Stable entry ID (non-empty string). This is the name recorded in
            provenance, cache keys, and artifacts.
        value:
            The registered unit (door-typed; the kernel stores it opaquely).
        capabilities:
            REQUIRED non-empty capability rows (portable scalar values).
        provider:
            Stable provider identity; builtins pass ``TORCHLENS_PROVIDER``.
        replace:
            Explicit replacement opt-in. Default refuses collisions.
        conformance_ref:
            Optional pointer to the registration's real-model conformance row
            (test id or obligation id).

        Returns
        -------
        RegistrationInfo
            The immutable registration metadata.
        """

        if not isinstance(entry_id, str) or not entry_id:
            raise RegistryError(
                f"Registry {self.name!r} received an invalid entry id {entry_id!r}; "
                "entry ids are stable non-empty strings recorded in provenance "
                "and cache keys.",
                code="registry_entry_id_invalid",
                registry=self.name,
                remedy="Pass a non-empty string entry id.",
            )
        if not isinstance(provider, ProviderInfo) or not provider.provider_id:
            raise RegistryError(
                f"Registration {entry_id!r} in registry {self.name!r} has no stable "
                "provider identity.",
                code="registry_provider_invalid",
                registry=self.name,
                entry_id=entry_id,
                remedy=(
                    "Pass provider=ProviderInfo(provider_id=..., distribution=..., "
                    "version=...); TorchLens builtins use TORCHLENS_PROVIDER."
                ),
            )
        frozen_capabilities = _validate_capabilities(self.name, entry_id, capabilities)
        with self._lock:
            prior = self._entries.get(entry_id)
            if prior is not None and not replace:
                raise RegistryError(
                    f"Registry {self.name!r} already has an entry {entry_id!r} "
                    f"(provider {prior.provider.provider_id!r}). Collisions refuse "
                    "by default so two providers can never silently shadow each "
                    "other.",
                    code="registry_entry_duplicate",
                    registry=self.name,
                    entry_id=entry_id,
                    existing_provider=prior.provider.provider_id,
                    remedy=(
                        "Pass replace=True to explicitly replace the existing "
                        "entry, or register under a different id."
                    ),
                )
            self._epoch += 1
            info = RegistrationInfo(
                registry=self.name,
                entry_id=entry_id,
                provider=provider,
                capabilities=MappingProxyType(frozen_capabilities),
                value_type=type(value).__name__,
                registered_epoch=self._epoch,
                replaced_prior=prior is not None,
                conformance_ref=conformance_ref,
            )
            self._values[entry_id] = value
            self._entries[entry_id] = info
            return info

    def unregister(self, entry_id: str) -> None:
        """Remove one entry; unknown ids refuse typed."""

        with self._lock:
            if entry_id not in self._entries:
                raise self._unknown_entry_error(entry_id, code="registry_entry_unknown")
            self._epoch += 1
            del self._entries[entry_id]
            del self._values[entry_id]

    def get(self, entry_id: str) -> T:
        """Return one registered unit; unknown ids refuse teaching-typed."""

        with self._lock:
            if entry_id not in self._values:
                raise self._unknown_entry_error(entry_id, code="registry_entry_unknown")
            return self._values[entry_id]

    def info(self, entry_id: str) -> RegistrationInfo:
        """Return one entry's immutable registration metadata."""

        with self._lock:
            if entry_id not in self._entries:
                raise self._unknown_entry_error(entry_id, code="registry_entry_unknown")
            return self._entries[entry_id]

    def list_ids(self) -> tuple[str, ...]:
        """Return the sorted registered entry ids."""

        with self._lock:
            return tuple(sorted(self._entries))

    def snapshot(self) -> RegistrySnapshot[T]:
        """Return an immutable point-in-time view (entries + values + epoch)."""

        with self._lock:
            return RegistrySnapshot(
                name=self.name,
                epoch=self._epoch,
                entries=MappingProxyType(dict(self._entries)),
                values=MappingProxyType(dict(self._values)),
            )

    @property
    def epoch(self) -> int:
        """Return the monotone mutation epoch (0 = never mutated)."""

        with self._lock:
            return self._epoch

    def cache_key(self) -> str:
        """Return the epoch- and provider-version-keyed cache key."""

        return self.snapshot().cache_key()

    def _unknown_entry_error(self, entry_id: str, *, code: str) -> RegistryError:
        """Build the teaching refusal for an unknown entry id.

        ``code`` is passed AT each raise site (always
        ``"registry_entry_unknown"``) so the S-17 census sees the code where
        the raise happens, not buried in this factory.
        """

        known = ", ".join(repr(name) for name in sorted(self._entries)) or "none registered"
        return RegistryError(
            f"Registry {self.name!r} has no {self.kind_label} {entry_id!r}. Registered: {known}.",
            code=code,
            registry=self.name,
            entry_id=entry_id,
            registered=tuple(sorted(self._entries)),
            remedy=(
                f"Register the {self.kind_label} through its public door first, "
                "or use one of the registered ids."
            ),
        )


_KERNEL_LOCK = threading.Lock()
_REGISTRIES: dict[str, Registry[Any]] = {}


def create_registry(
    name: str,
    *,
    kind_label: str,
    missing_provider_behavior: MissingProviderBehavior = "typed_refusal",
) -> Registry[Any]:
    """Create (and kernel-enroll) one domain registry.

    Called exactly once per domain door, at the door module's import. The
    kernel enrollment is what makes the domain COUNTABLE: universe rows and
    the SURFACE table read :func:`kernel_universe_rows`.

    Parameters
    ----------
    name:
        Kernel-unique registry name (the domain, e.g. ``"export_targets"``).
    kind_label:
        Human noun for entries (``"export target"``), used in teaching
        refusals.
    missing_provider_behavior:
        Declared door behavior when an artifact or request names an
        unregistered provider: ``"typed_refusal"`` or ``"analysis_only"``.

    Returns
    -------
    Registry
        The new registry, enrolled in the kernel inventory.
    """

    if not isinstance(name, str) or not name:
        raise RegistryError(
            f"Registry name {name!r} is invalid; kernel registries need stable "
            "non-empty string names.",
            code="registry_name_invalid",
            remedy="Pass a non-empty string registry name.",
        )
    with _KERNEL_LOCK:
        if name in _REGISTRIES:
            raise RegistryError(
                f"A registry named {name!r} already exists in the kernel. Each "
                "domain door creates its registry exactly once at import.",
                code="registry_domain_duplicate",
                registry=name,
                remedy=(
                    "Import the owning door module instead of re-creating its "
                    "registry; tests needing isolation should register/unregister "
                    "entries, not domains."
                ),
            )
        registry: Registry[Any] = Registry(
            name=name,
            kind_label=kind_label,
            missing_provider_behavior=missing_provider_behavior,
        )
        _REGISTRIES[name] = registry
        return registry


def kernel_registries() -> tuple[str, ...]:
    """Return the sorted names of every kernel-enrolled registry."""

    with _KERNEL_LOCK:
        return tuple(sorted(_REGISTRIES))


def kernel_universe_rows() -> dict[str, int]:
    """Return registry name -> entry count (the compo universe feed)."""

    with _KERNEL_LOCK:
        registries = list(_REGISTRIES.values())
    return {registry.name: len(registry.list_ids()) for registry in registries}
