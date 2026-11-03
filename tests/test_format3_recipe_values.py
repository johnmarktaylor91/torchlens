"""Recipe values retain their closed container types while payloads materialize."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
import torch

from torchlens._io.format3.decode import CallablePlaceholder
from torchlens._io.format3.payloads import PayloadStore
from torchlens._io.format3.recipe import _bind, _RecipeEncoder


def _round_trip(value, root: Path):
    """Write actual recipe chunks, bind their values, and check the resaved wire."""

    store = PayloadStore(root)
    wire = _RecipeEncoder(store).encode(value, "recipe.value")
    store.close()
    loaded = _bind(wire, root, "recipe.value")
    again_store = PayloadStore(root / "again")
    assert _RecipeEncoder(again_store).encode(loaded, "recipe.value") == wire
    again_store.close()
    return loaded


def test_recipe_keeps_size_type(tmp_path: Path) -> None:
    """A dimension value remains a Size instead of silently becoming an ordinary tuple."""

    original = torch.Size([2, 3])
    loaded = _round_trip(original, tmp_path)
    assert type(loaded) is torch.Size
    assert loaded == original


def test_recipe_keeps_structseq_with_materialized_tensors(tmp_path: Path) -> None:
    """A torch result keeps its named fields and the exact values of both chunk payloads."""

    original = torch.max(torch.tensor([[1.0, 4.0], [3.0, 2.0]]), dim=0)
    loaded = _round_trip(original, tmp_path)
    assert type(loaded) is type(original)
    assert torch.equal(loaded.values, original.values)
    assert torch.equal(loaded.indices, original.indices)


def test_recipe_keeps_builtin_defaultdict_factory(tmp_path: Path) -> None:
    """The builtin factory and the nested tensor survive without creating a missing key."""

    original = defaultdict(list, {"seen": torch.tensor([2.0, 3.0])})
    loaded = _round_trip(original, tmp_path)
    assert isinstance(loaded, defaultdict)
    assert loaded.default_factory is list
    assert list(loaded) == ["seen"]
    assert torch.equal(loaded["seen"], original["seen"])
    assert loaded["new"] == []


@pytest.mark.parametrize("position", [0, "0.args[1]"])
def test_recipe_placeholder_resaves_without_becoming_executable(tmp_path: Path, position) -> None:
    """The same data-only carrier and exact wire fields pass through the recipe binder."""

    wire = {
        "$tl": "callable_placeholder",
        "position": position,
        "qualname": "opaque.helper",
        "repr_length": 300,
    }
    loaded = _bind(wire, tmp_path, "recipe.value")
    assert type(loaded) is CallablePlaceholder
    assert not callable(loaded)
    assert _RecipeEncoder(PayloadStore(tmp_path)).encode(loaded, "recipe.value") == wire
