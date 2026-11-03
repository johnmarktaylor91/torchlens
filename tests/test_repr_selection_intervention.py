"""Display contracts for selections, intervention objects, options and reports."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._selection.resolver import _SiteTable
from torchlens._utils._display_errors import candidate_keys, lookup_message
from torchlens.intervention import _edit_repr
from torchlens.intervention.types import CapturedArgTemplate, FrozenInterventionSpec, LiteralTensor
from torchlens.options import (
    CaptureOptions,
    EpisodeOptions,
    InterventionOptions,
    NarrationOptions,
    ReplayOptions,
    StorageOptions,
    VisualizationOptions,
)


@pytest.mark.parametrize(
    "kind",
    [
        CaptureOptions,
        EpisodeOptions,
        InterventionOptions,
        NarrationOptions,
        ReplayOptions,
        StorageOptions,
        VisualizationOptions,
    ],
)
def test_default_options_omit_all_fields(kind: type) -> None:
    """Default reprs stay pasteable; the card still lists every public field."""

    options = kind()
    assert repr(options) == f"{kind.__name__}()"
    for item in fields(options):
        if item.repr and not item.name.startswith("_"):
            assert item.name in str(options)


def test_options_display_only_changes_and_bounds_long_values() -> None:
    """Explicit defaults do not create noise, and a long user name cannot dump."""

    assert repr(CaptureOptions(save_grads=True, verbose=False)) == "CaptureOptions(save_grads=True)"
    assert len(repr(CaptureOptions(name="x" * 1000))) <= 100


def test_selectors_rebuild_nested_boolean_trees() -> None:
    """Parentheses preserve nested conjunction, disjunction and complement."""

    selector = ~(tl.func("relu") | tl.func("sigmoid")) & (tl.in_module("block") | tl.func("tanh"))
    assert eval(repr(selector), {"tl": tl}).to_target_spec() == selector.to_target_spec()


def test_recipe_spelling_rebuilds_exactly_or_discloses_rule_count() -> None:
    """Only a reconstructible single clause uses the public tl.when spelling."""

    recipe = tl.when(tl.func("relu"), tl.zero_ablate())
    assert repr(recipe) == "tl.when(tl.func('relu'), tl.zero_ablate())"
    assert eval(repr(recipe), {"tl": tl}) == recipe
    assert str(recipe).splitlines()[0] == repr(recipe)
    opaque = tl.when(tl.func("relu"), lambda x: x)
    assert repr(opaque) == "Recipe: 1 rule"
    merged = recipe & tl.when(tl.func("tanh"), tl.scale(2.0))
    assert repr(merged) == "Recipe: 2 rules"
    assert str(merged).startswith("Recipe: 2 rules\n  rule 1")


def test_tensor_printouts_never_read_elements(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edits, frozen specs and captured arguments use tensor metadata only."""

    tensor = torch.ones(4)
    objects = [
        tl.project_out(tensor),
        FrozenInterventionSpec(value=tensor),
        CapturedArgTemplate(args=(LiteralTensor(tensor),)),
    ]

    def forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("display attempted a tensor read")

    monkeypatch.setattr(torch.Tensor, "__repr__", forbidden)
    for obj in objects:
        assert len(repr(obj)) <= 100
        assert "display unavailable" not in str(obj)
        assert "<tensor float32 [4] on cpu>" in str(obj)


def test_edit_signatures_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated rendering inspects a constructor signature only once."""

    _edit_repr._signature.cache_clear()
    original = _edit_repr.inspect.signature
    calls = []

    def signature(target: object):
        calls.append(target)
        return original(target)

    monkeypatch.setattr(_edit_repr.inspect, "signature", signature)
    edit = tl.clip_norm(1.0)
    assert repr(edit) == repr(edit) == "tl.clip_norm(1.0)"
    assert len(calls) == 1


def test_selection_and_site_table_bound_before_rendering() -> None:
    """Concrete selections use compact counts and only a bounded site prefix."""

    trace = tl.trace(nn.Sequential(*[nn.ReLU() for _ in range(8)]), torch.ones(2, 4))
    selection = trace[tl.func("relu")]
    assert repr(selection) == "Selection ACT: 8 sites, 64 values, from tl.func('relu')"
    assert "... and 3 more" in str(selection)
    assert repr(selection[0]).startswith("Site ACT relu_1:")
    table = _SiteTable(tuple(trace._layer_list))
    assert repr(table) == f"_SiteTable: {len(table)} sites"
    assert "... and" in str(table)


def test_candidate_rows_use_five_stored_display_keys() -> None:
    """The problem and remedy remain readable while suggestions have a hard bound."""

    class Record:
        short_layer_label = "relu_1"
        num_passes = 1

    keys = candidate_keys(["relu_1_22"], {"relu_1_22": Record()})
    assert keys == ["relu_1"]
    message = lookup_message(
        "Layer 'missing' not found.", [*keys, *map(str, range(8))], "choose a key."
    )
    assert message.splitlines() == [
        "Layer 'missing' not found.",
        "  candidates 'relu_1', '0', '1', '2', '3'; and 4 more",
        "  remedy    choose a key.",
    ]
    error = tl.errors.InvalidArgumentError(message, code="example", remedy="choose a key.")
    assert str(error) == message
    assert error.fields["code"] == "example"
    assert error.fields["remedy"] == "choose a key."
    empty = lookup_message("No matching layers.", [], "choose a key.")
    error = tl.errors.InvalidArgumentError(empty, code="example", remedy="choose a key.")
    assert str(error) == empty


@pytest.mark.parametrize("candidates", [[], ["relu_1"]])
def test_candidate_rows_preserve_the_structured_remedy(candidates: list[str]) -> None:
    """Changing refusal layout must retain the programmatic remedy contract."""

    message = lookup_message("Layer not found.", candidates, "choose a key.")
    error = tl.errors.TorchLensError(message)
    assert error.fields["remedy"] == "choose a key"
    explicit = tl.errors.TorchLensError(message, remedy="use another trace")
    assert explicit.fields["remedy"] == "use another trace"
    legacy = tl.errors.TorchLensError("Layer not found. Remedy: choose a key.")
    assert legacy.fields["remedy"] == error.fields["remedy"]
