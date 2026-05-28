"""Tests for the semantic facets registry and view surface."""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from torchlens.semantic import FacetRecipe, FacetView
from torchlens.semantic import facets as facets_mod


class _Record:
    """Simple record-like object for facet tests."""

    class_name = "UnitFacetRecord"
    class_qualname = "tests.UnitFacetRecord"


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    """Restore the global facet registry after each test."""

    original = list(facets_mod._REGISTRY)
    yield
    facets_mod._REGISTRY[:] = original


def test_facet_view_dict_and_attribute_access_uniformity() -> None:
    """Attribute and item access return the same cached value."""

    @tl.facets.register(class_name="UnitFacetRecord")
    def unit_recipe(record: Any) -> dict[str, Any]:
        """Return a test facet."""

        return {"x": 10}

    view = FacetView(_Record())
    assert view.x == 10
    assert view["x"] == 10


def test_lazy_compute_keys_has_and_cache() -> None:
    """Recipe functions run only when a value is accessed."""

    calls = 0

    @tl.facets.register(class_name="UnitFacetRecord")
    def lazy_recipe(record: Any) -> dict[str, Any]:
        """Return a lazily computed facet."""

        nonlocal calls
        calls += 1
        return {"x": 10}

    view = FacetView(_Record())
    assert view.keys() == ["x"]
    assert view.has("x")
    assert list(view) == ["x"]
    assert len(view) == 1
    assert calls == 0
    assert view.x == 10
    assert view.x == 10
    assert calls == 1


def test_invalidate_clears_cached_values() -> None:
    """Invalidation causes the next access to re-run the recipe."""

    calls = 0

    @tl.facets.register(class_name="UnitFacetRecord")
    def invalidated_recipe(record: Any) -> dict[str, Any]:
        """Return a monotonically increasing test facet."""

        nonlocal calls
        calls += 1
        return {"x": calls}

    view = FacetView(_Record())
    assert view.x == 1
    view.invalidate()
    assert view.x == 2


def test_empty_facet_view() -> None:
    """Unmatched records have an empty but usable view."""

    view = FacetView(object())
    assert view.keys() == []
    assert not view.has("x")
    assert list(view) == []
    assert len(view) == 0
    assert view.recipe_source is None


def test_multi_recipe_merge_warns_and_last_wins() -> None:
    """Multiple matching recipes merge with last-registered conflict precedence."""

    @tl.facets.register(class_name="UnitFacetRecord")
    def first_recipe(record: Any) -> dict[str, Any]:
        """Return the first test facets."""

        return {"x": 1, "y": 2}

    @tl.facets.register(class_name="UnitFacetRecord")
    def second_recipe(record: Any) -> dict[str, Any]:
        """Return overriding test facets."""

        return {"x": 3, "z": 4}

    view = FacetView(_Record())
    assert view.recipe_source == ("first_recipe", "second_recipe")
    with pytest.warns(UserWarning, match="overrides facet 'x'"):
        assert view.x == 3
    assert view.y == 2
    assert view.z == 4


def test_list_info_glob_and_recipe_record() -> None:
    """Discoverability functions expose registered recipe metadata."""

    @tl.facets.register(class_name=("UnitFacetRecord", "UnitAttention"))
    def discovery_recipe(record: Any) -> dict[str, Any]:
        """Return discoverable facets."""

        return {"q": 1}

    recipes = tl.facets.list(class_name="Unit*")
    assert len(recipes) == 1
    assert isinstance(recipes[0], FacetRecipe)
    assert recipes[0].recipe_name == "discovery_recipe"
    assert tl.facets.info("UnitFacetRecord") == {
        "recipes": ["discovery_recipe"],
        "facets": ["q"],
    }


def test_register_preserves_function_object_and_docstring() -> None:
    """The register decorator returns the original function unchanged."""

    def original_recipe(record: Any) -> dict[str, Any]:
        """Original docstring."""

        return {"x": 1}

    registered = tl.facets.register(class_name="UnitFacetRecord")(original_recipe)
    assert registered is original_recipe
    assert registered.__doc__ == "Original docstring."
