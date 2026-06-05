"""Tests for the semantic facets registry and view surface."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

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
    """Same-tier facet collisions warn and choose a deterministic winner."""

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
    with pytest.warns(UserWarning, match="ambiguous same-tier recipes"):
        assert view.x == 3
    assert view.y == 2
    assert view.z == 4


def test_trace_snapshot_is_immune_to_later_registry_mutation() -> None:
    """A trace uses the recipe snapshot captured during its forward pass."""

    class Tiny(nn.Module):
        """Tiny module with one named child."""

        def __init__(self) -> None:
            """Initialize the child linear layer."""

            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the child layer."""

            return self.linear(x)

    @tl.facets.register(class_name="Linear")
    def first_linear(record: Any) -> dict[str, Any]:
        """Return the first snapshot value."""

        return {"snapshot_value": 1}

    log = tl.trace(Tiny(), torch.randn(1, 2), layers_to_save="all")

    @tl.facets.register(class_name="Linear")
    def second_linear(record: Any) -> dict[str, Any]:
        """Return a later value that must not affect old traces."""

        return {"snapshot_value": 2}

    assert log.modules["linear"].facets.snapshot_value == 1


def test_trace_recipes_and_using_are_capture_time_additive() -> None:
    """Per-trace and context recipes affect snapshots only at capture time."""

    class Tiny(nn.Module):
        """Tiny single-module model."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return an incremented tensor."""

            return x + 1

    def local_recipe(record: Any) -> dict[str, Any]:
        """Return a local facet for any record."""

        return {"local_value": "present"}

    model = Tiny()
    x = torch.randn(1, 2)
    per_trace = tl.trace(model, x, layers_to_save="all", recipes=[local_recipe])
    assert per_trace.modules["self"].facets.local_value == "present"

    with tl.facets.using(local_recipe):
        contextual = tl.trace(model, x, layers_to_save="all")
    assert contextual.modules["self"].facets.local_value == "present"

    outside = tl.trace(model, x, layers_to_save="all")
    assert not outside.modules["self"].facets.has("local_value")


def test_structural_output_facets_expose_names_and_method_collisions() -> None:
    """Existing container output names are available through item access."""

    class DictBlock(nn.Module):
        """Return dict keys that include a FacetView method name."""

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            """Return named tensors."""

            return {"keys": x + 1, "hidden": x + 2}

    class Model(nn.Module):
        """Wrapper with a named child module."""

        def __init__(self) -> None:
            """Initialize the child module."""

            super().__init__()
            self.block = DictBlock()

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            """Return the child output."""

            return self.block(x)

    x = torch.randn(2, 3)
    log = tl.trace(Model(), x, layers_to_save="all")
    facets = log.modules["block"].facets

    assert torch.equal(facets["keys"], x + 1)
    assert torch.equal(facets["hidden"], x + 2)
    assert callable(facets.keys)


def test_structural_output_facets_expose_namedtuple_and_dataclass_names() -> None:
    """NamedTuple and dataclass output names surface as structural facets."""

    Pair = namedtuple("Pair", ["values", "indices"])

    @dataclass
    class DataOut:
        """Dataclass output container."""

        first: torch.Tensor
        second: torch.Tensor

    class StructBlock(nn.Module):
        """Return nested named containers."""

        def forward(self, x: torch.Tensor) -> tuple[Pair, DataOut]:
            """Return tuple, namedtuple, and dataclass outputs."""

            return Pair(x + 1, x + 2), DataOut(x + 3, x + 4)

    class Model(nn.Module):
        """Wrapper exposing the structured output as a child module."""

        def __init__(self) -> None:
            """Initialize the structured child."""

            super().__init__()
            self.block = StructBlock()

        def forward(self, x: torch.Tensor) -> tuple[Pair, DataOut]:
            """Return the child output."""

            return self.block(x)

    log = tl.trace(Model(), torch.randn(2, 3), layers_to_save="all")
    facets = log.modules["block"].facets

    assert "out0.values" in facets.keys()
    assert "out0.indices" in facets.keys()
    assert "out1.first" in facets.keys()
    assert "out1.second" in facets.keys()


def test_structseq_output_facets_expose_torch_return_type_names() -> None:
    """Torch return_types names surface as structural facets when preserved."""

    class MaxBlock(nn.Module):
        """Return a torch structseq output."""

        def forward(self, x: torch.Tensor) -> Any:
            """Return max values and indices."""

            return torch.max(x, dim=1)

    class Model(nn.Module):
        """Wrapper exposing the structseq return as a child module."""

        def __init__(self) -> None:
            """Initialize the child module."""

            super().__init__()
            self.block = MaxBlock()

        def forward(self, x: torch.Tensor) -> Any:
            """Return the child output."""

            return self.block(x)

    x = torch.randn(2, 3)
    log = tl.trace(Model(), x, layers_to_save="all")
    facets = log.modules["block"].facets
    expected = torch.max(x, dim=1)

    assert torch.equal(facets["values"], expected.values)
    assert torch.equal(facets["indices"], expected.indices)


def test_lstm_multi_output_facets_preserve_single_call_roles() -> None:
    """LSTM outputs expose role names without becoming recurrent passes."""

    class LSTMModel(nn.Module):
        """Wrapper exposing an LSTM child."""

        def __init__(self) -> None:
            """Initialize the LSTM child."""

            super().__init__()
            self.lstm = nn.LSTM(3, 5)

        def forward(self, x: torch.Tensor) -> Any:
            """Return the LSTM output tuple."""

            return self.lstm(x)

    x = torch.randn(4, 2, 3)
    log = tl.trace(LSTMModel(), x, layers_to_save="all")
    lstm = log.modules["lstm"]

    assert lstm.num_calls == 1
    assert [log.ops[label].multi_output_name for label in lstm.calls[0].output_ops] == [
        "output",
        "h_n",
        "c_n",
    ]
    assert torch.equal(lstm.facets["output"], lstm.outs[0])
    assert torch.equal(lstm.facets["h_n"], lstm.outs[1])
    assert torch.equal(lstm.facets["c_n"], lstm.outs[2])


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
