"""Registry-kernel lifecycle, epoch, snapshot, and refusal gates (C01 item 13).

The kernel is PRIVATE plumbing (architecture memo 6.1): these tests exercise
the lifecycle law every typed public door inherits -- register / unregister /
list / info / snapshot, monotone epoch, collision refusal by default,
capability rows REQUIRED at registration, snapshot immutability, and
epoch+provider-version cache keys.
"""

from __future__ import annotations

import threading

import pytest

from torchlens._registry import (
    TORCHLENS_PROVIDER,
    ProviderInfo,
    Registry,
    RegistryError,
    create_registry,
    kernel_registries,
    kernel_universe_rows,
)


def _fresh_registry(name: str = "test-domain") -> Registry[object]:
    """Return an unenrolled registry (kernel inventory stays clean)."""

    return Registry(name=name, kind_label="test unit")


CAPS = {"supports_x": True, "grain": "op"}


class TestLifecycle:
    def test_replace_requires_explicit_opt_in(self) -> None:
        registry = _fresh_registry()
        registry.register("a", 1, capabilities=CAPS, provider=TORCHLENS_PROVIDER)
        with pytest.raises(RegistryError) as excinfo:
            registry.register("a", 2, capabilities=CAPS, provider=TORCHLENS_PROVIDER)
        assert excinfo.value.fields["code"] == "registry_entry_duplicate"
        info = registry.register(
            "a", 2, capabilities=CAPS, provider=TORCHLENS_PROVIDER, replace=True
        )
        assert info.replaced_prior is True
        assert registry.get("a") == 2

    def test_unknown_entry_refusals_teach_registered_names(self) -> None:
        registry = _fresh_registry()
        registry.register("known", 1, capabilities=CAPS, provider=TORCHLENS_PROVIDER)
        for operation in (registry.get, registry.info, registry.unregister):
            with pytest.raises(RegistryError) as excinfo:
                operation("missing")
            assert excinfo.value.fields["code"] == "registry_entry_unknown"
            assert "'known'" in str(excinfo.value)
            assert excinfo.value.fields["registered"] == ("known",)


class TestCapabilityRows:
    def test_registration_without_capabilities_is_red(self) -> None:
        registry = _fresh_registry()
        for bad in ({}, None):
            with pytest.raises(RegistryError) as excinfo:
                registry.register("u", 1, capabilities=bad, provider=TORCHLENS_PROVIDER)
            assert excinfo.value.fields["code"] == "registry_capabilities_missing"

    def test_unportable_capability_values_refuse(self) -> None:
        registry = _fresh_registry()
        with pytest.raises(RegistryError) as excinfo:
            registry.register(
                "u", 1, capabilities={"cb": lambda: None}, provider=TORCHLENS_PROVIDER
            )
        assert excinfo.value.fields["code"] == "registry_capability_value_invalid"
        with pytest.raises(RegistryError) as excinfo:
            registry.register("u", 1, capabilities={"t": (1, 2)}, provider=TORCHLENS_PROVIDER)
        assert excinfo.value.fields["code"] == "registry_capability_value_invalid"

    def test_capability_keys_must_be_nonempty_strings(self) -> None:
        registry = _fresh_registry()
        with pytest.raises(RegistryError) as excinfo:
            registry.register("u", 1, capabilities={"": True}, provider=TORCHLENS_PROVIDER)
        assert excinfo.value.fields["code"] == "registry_capability_key_invalid"


class TestProviderIdentity:
    def test_provider_required_with_stable_id(self) -> None:
        registry = _fresh_registry()
        with pytest.raises(RegistryError) as excinfo:
            registry.register("u", 1, capabilities=CAPS, provider=ProviderInfo(provider_id=""))
        assert excinfo.value.fields["code"] == "registry_provider_invalid"

    def test_entry_id_must_be_nonempty_string(self) -> None:
        registry = _fresh_registry()
        with pytest.raises(RegistryError) as excinfo:
            registry.register("", 1, capabilities=CAPS, provider=TORCHLENS_PROVIDER)
        assert excinfo.value.fields["code"] == "registry_entry_id_invalid"


class TestSnapshotsAndCacheKeys:
    def test_cache_key_moves_with_epoch_and_provider_version(self) -> None:
        registry = _fresh_registry()
        registry.register(
            "a",
            1,
            capabilities=CAPS,
            provider=ProviderInfo(provider_id="p", version="1.0"),
        )
        key_v1 = registry.cache_key()
        registry.register(
            "a",
            1,
            capabilities=CAPS,
            provider=ProviderInfo(provider_id="p", version="2.0"),
            replace=True,
        )
        key_v2 = registry.cache_key()
        assert key_v1 != key_v2
        assert key_v2.startswith("test-domain:e")

    def test_concurrent_registrations_stay_consistent(self) -> None:
        registry = _fresh_registry()
        errors: list[Exception] = []

        def _register(start: int) -> None:
            try:
                for index in range(start, start + 25):
                    registry.register(
                        f"u{index}", index, capabilities=CAPS, provider=TORCHLENS_PROVIDER
                    )
            except RegistryError as exc:  # pragma: no cover - failure diagnostics
                errors.append(exc)

        threads = [threading.Thread(target=_register, args=(base,)) for base in (0, 25, 50, 75)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert len(registry.list_ids()) == 100
        assert registry.epoch == 100


class TestKernelInventory:
    def test_create_registry_enrolls_and_counts(self) -> None:
        name = "test-kernel-inventory-domain"
        assert name not in kernel_registries()
        registry = create_registry(name, kind_label="planted unit")
        try:
            assert name in kernel_registries()
            assert kernel_universe_rows()[name] == 0
            registry.register("one", 1, capabilities=CAPS, provider=TORCHLENS_PROVIDER)
            assert kernel_universe_rows()[name] == 1
        finally:
            registry.unregister("one")

    def test_duplicate_domain_creation_refuses(self) -> None:
        name = "test-kernel-duplicate-domain"
        create_registry(name, kind_label="planted unit")
        with pytest.raises(RegistryError) as excinfo:
            create_registry(name, kind_label="planted unit")
        assert excinfo.value.fields["code"] == "registry_domain_duplicate"

    def test_invalid_domain_name_refuses(self) -> None:
        with pytest.raises(RegistryError) as excinfo:
            create_registry("", kind_label="planted unit")
        assert excinfo.value.fields["code"] == "registry_name_invalid"
