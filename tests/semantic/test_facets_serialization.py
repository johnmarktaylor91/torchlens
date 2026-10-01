"""Tests for facets and portable serialization."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import facets as facets_mod


class _LinearModel(nn.Module):
    """Small model used for serialization tests."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.linear(x)


def test_facets_rerive_after_tlspec_load_when_recipe_registered(tmp_path: Any) -> None:
    """Facet views are not serialized and are rebuilt against the current registry."""

    original = list(facets_mod._REGISTRY)
    try:

        @tl.facets.register(class_name="Linear")
        def linear_recipe(record: Any) -> dict[str, Any]:
            """Return a user facet for Linear modules."""

            return {"linear_out_shape": tuple(record.out.shape)}

        path = tmp_path / "linear.tlspec"
        log = tl.trace(
            _LinearModel(),
            torch.randn(1, 3),
            capture=tl.options.CaptureOptions(layers_to_save="all"),
        )
        log.save(path)

        facets_mod._REGISTRY[:] = original
        loaded_without_recipe = tl.load(path)
        assert not loaded_without_recipe.modules["linear"].facets.has("linear_out_shape")

        tl.facets.register(class_name="Linear")(linear_recipe)
        loaded_with_recipe = tl.load(path)
        assert loaded_with_recipe.modules["linear"].facets.linear_out_shape == (1, 3)
    finally:
        facets_mod._REGISTRY[:] = original


def test_facets_read_before_save_does_not_poison_tlspec(tmp_path: Any) -> None:
    """Reading ``.facets`` pre-save must leave the trace saveable and loadable.

    Fail-before: the ModuleCall/Module ``.facets`` getters cached a FacetView
    in ``__dict__["_facets_cache"]`` with no PORTABLE_STATE_SPEC row, so this
    exact sequence refused with ``ModuleCall._facets_cache is missing from
    PORTABLE_STATE_SPEC`` -- a read-only documented accessor poisoned the
    artifact path. The suite's other tests only read facets on LOADED traces,
    which is why the gap shipped.
    """

    log = tl.trace(
        _LinearModel(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    module_call = next(iter(log.module_calls))
    assert module_call.facets is module_call.facets  # cached, not rebuilt per read
    assert log.modules["linear"].facets is not None

    path = tmp_path / "facets_touched.tlspec"
    tl.save(log, path)

    loaded = tl.load(path)
    reloaded_call = next(iter(loaded.module_calls))
    assert type(reloaded_call.facets).__name__ == "FacetView"
    assert type(loaded.modules["linear"].facets).__name__ == "FacetView"


def test_facets_read_before_pickle_round_trips(tmp_path: Any) -> None:
    """A pre-pickle ``.facets`` read is dropped from state and rebuilt on access."""

    import pickle

    log = tl.trace(
        _LinearModel(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    module_call = next(iter(log.module_calls))
    _ = module_call.facets
    _ = log.modules["linear"].facets
    assert "_facets_cache" not in module_call.__getstate__()

    round_tripped = pickle.loads(pickle.dumps(log))
    reloaded_call = next(iter(round_tripped.module_calls))
    assert type(reloaded_call.facets).__name__ == "FacetView"
