"""Tests for closed recipes and typed adapters."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from menagerie.crawler.recipe import (
    DeclarativeRecipe,
    RecipeError,
    load_declarative_recipe,
    load_typed_adapter,
    reject_opaque_recipe,
    validate_pretrained_disable_fields,
    validate_pretrained_disposition,
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


@pytest.mark.smoke
def test_asserted_pretrained_fields_absent_loads_a_config_only_constructor() -> None:
    """The checked absence assertion is the R1 path for nothing-to-disable shapes.

    ``torch.nn.Linear`` stands in for every ``XForCausalLM(config)``, GNN layer,
    and SNN module whose constructor exposes no pretrained-capable keyword: the
    assertion is verified against the real signature and the build proceeds.
    """

    loaded = load_declarative_recipe(
        {
            "distribution": "torch",
            "version": torch.__version__,
            "module": "torch.nn",
            "symbol": "Linear",
            "kwargs": {"in_features": 4, "out_features": 2},
            "pretrained_disable_fields": [],
            "pretrained_fields_absent": True,
        }
    )
    assert isinstance(loaded.build_model(), torch.nn.Linear)


@pytest.mark.smoke
def test_false_absence_assertion_is_refused_by_the_real_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserting absence over a constructor that exposes ``pretrained`` refuses.

    The assertion is a checked fact, not a recorded one: the load-time signature
    scan proves it false and names the remedy, so the assertion can never paper
    over a real pretrained flag.
    """

    class FakePretrainedNet(torch.nn.Module):
        def __init__(self, pretrained: bool = False) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    monkeypatch.setattr(torch.nn, "FakePretrainedNet", FakePretrainedNet, raising=False)
    with pytest.raises(RecipeError, match="contradicted by the pinned constructor signature"):
        load_declarative_recipe(
            {
                "distribution": "torch",
                "version": torch.__version__,
                "module": "torch.nn",
                "symbol": "FakePretrainedNet",
                "kwargs": {},
                "pretrained_disable_fields": [],
                "pretrained_fields_absent": True,
            }
        )


@pytest.mark.smoke
def test_absence_assertion_contradicting_disable_fields_is_refused() -> None:
    """Declaring both an absence and a disable list is a contradiction."""

    with pytest.raises(RecipeError, match="contradicts"):
        DeclarativeRecipe.from_mapping(
            {
                "distribution": "torch",
                "version": torch.__version__,
                "module": "torch.nn",
                "symbol": "Linear",
                "kwargs": {"weights": None},
                "pretrained_disable_fields": ["weights"],
                "pretrained_fields_absent": True,
            }
        )


@pytest.mark.smoke
def test_pretrained_capable_kwargs_key_cannot_ride_through_unlisted() -> None:
    """A known pretrained-asset key never rides through the declaration unlisted.

    Two distinct refusals close the historical dodge where listing a harmless
    field satisfied the non-empty rule while ``weights`` carried an enabling
    value unlisted: an ENABLING value on a known key is refused outright as the
    safety hazard it is, and even a DISABLING value on a known key is refused
    until the key is declared in ``pretrained_disable_fields``, so the disable
    list stays a complete account of the constructor's known pretrained surface.
    """

    with pytest.raises(RecipeError, match="leave known pretrained keywords enabled"):
        validate_pretrained_disposition(
            {"weights": "IMAGENET1K_V1", "progress": False}, ["progress"], fields_absent=False
        )
    with pytest.raises(RecipeError, match="pretrained-capable keys"):
        validate_pretrained_disposition(
            {"weights": None, "progress": False}, ["progress"], fields_absent=False
        )
    # The complete disabling declaration for the same key remains satisfiable.
    validate_pretrained_disposition(
        {"weights": None, "progress": False}, ["weights"], fields_absent=False
    )


@pytest.mark.smoke
def test_enabled_pretrained_flag_still_refuses_even_when_listed() -> None:
    """The original tripwire fires unchanged: a listed field must disable."""

    with pytest.raises(RecipeError, match="does not carry a disabling value"):
        validate_pretrained_disable_fields({"pretrained": True}, ["pretrained"])


@pytest.mark.smoke
def test_absence_assertion_is_omitted_from_default_recipe_payloads() -> None:
    """Historical recipes keep their exact recipe-revision hash preimage."""

    recipe = DeclarativeRecipe.from_mapping(
        {
            "distribution": "torch",
            "version": torch.__version__,
            "module": "torch.nn",
            "symbol": "Linear",
            "kwargs": {"in_features": 4, "out_features": 2},
            "pretrained_disable_fields": [],
        }
    )
    assert "pretrained_fields_absent" not in recipe.to_dict()
    asserted = DeclarativeRecipe.from_mapping(
        {**recipe.to_dict(), "pretrained_fields_absent": True}
    )
    assert asserted.to_dict()["pretrained_fields_absent"] is True


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
