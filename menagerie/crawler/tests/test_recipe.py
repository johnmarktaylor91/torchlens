"""Tests for closed recipes and typed adapters."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from menagerie.crawler.recipe import (
    RecipeError,
    load_declarative_recipe,
    load_typed_adapter,
    reject_opaque_recipe,
)


def test_opaque_eval_and_exec_recipes_are_rejected() -> None:
    """Legacy executable strings never enter the Slice B recipe path."""

    with pytest.raises(RecipeError):
        reject_opaque_recipe({"type": "expression", "code": "eval('bad')"})
    with pytest.raises(RecipeError):
        reject_opaque_recipe({"exec": "model = bad()"})


def test_declarative_r1_loads_direct_constructor() -> None:
    """A closed R1 recipe imports a direct symbol and applies JSON kwargs."""

    loaded = load_declarative_recipe(
        {
            "distribution": "torch",
            "version": torch.__version__,
            "module": "torch.nn",
            "symbol": "Linear",
            "kwargs": {"in_features": 4, "out_features": 2},
            "pretrained_disable_fields": [],
        }
    )
    model = loaded.build_model()

    assert isinstance(model, torch.nn.Linear)
    assert model.in_features == 4
    assert model.out_features == 2


def test_typed_adapter_loads_required_functions(tmp_path: Path) -> None:
    """A statically safe typed module supplies both required typed entry points."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import torch

def build_model() -> object:
    return torch.nn.Linear(4, 2)

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
""",
        encoding="utf-8",
    )

    loaded = load_typed_adapter(adapter)
    args, kwargs = loaded.make_dummy_call(3, "cpu") if loaded.make_dummy_call else ((), {})

    assert isinstance(loaded.build_model(), torch.nn.Linear)
    assert tuple(getattr(args[0], "shape")) == (1, 4)
    assert kwargs == {}
