"""The facet-snapshot slot: the engine's registered route to the facets appliance."""

from __future__ import annotations

import pytest

from torchlens._registry import _facet_snapshot
from torchlens._registry.kernel import RegistryError


def test_facets_module_registers_the_direct_snapshot() -> None:
    """Importing the facets appliance installs its ``snapshot`` as the provider."""

    from torchlens._semantic import facets

    assert _facet_snapshot._PROVIDER is facets.snapshot
    result = _facet_snapshot.facet_snapshot(None)
    assert isinstance(result, facets.FacetRegistrySnapshot)


def test_capture_door_provider_builds_a_real_snapshot() -> None:
    """The public door's lazy provider yields the same snapshot type."""

    from torchlens import _user_funcs
    from torchlens._semantic import facets

    result = _user_funcs._facet_snapshot_provider(None)
    assert isinstance(result, facets.FacetRegistrySnapshot)


def test_no_provider_and_no_recipes_records_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Readers fall back to the live registry when nothing was registered."""

    monkeypatch.setattr(_facet_snapshot, "_PROVIDER", None)
    assert _facet_snapshot.facet_snapshot(None) is None
    assert _facet_snapshot.facet_snapshot([]) is None


def test_no_provider_with_recipes_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-trace recipes are never dropped silently."""

    monkeypatch.setattr(_facet_snapshot, "_PROVIDER", None)
    with pytest.raises(RegistryError) as info:
        _facet_snapshot.facet_snapshot([lambda record: {}])
    assert info.value.fields["code"] == "facet_snapshot_provider_missing"
