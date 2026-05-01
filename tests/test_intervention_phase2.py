"""Smoke tests for intervention Phase 2 selector resolution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import warnings

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import (
    MultiMatchWarning,
    SiteAmbiguityError,
    SiteResolutionError,
)
from torchlens.intervention.resolver import SiteTable


class _Phase2Block(torch.nn.Module):
    """Nested block that creates module-contained and module-output sites."""

    def __init__(self) -> None:
        """Initialise the block."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer followed by ReLU.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Block output.
        """

        y = self.linear(x)
        return torch.relu(y)


class _Phase2Model(torch.nn.Module):
    """Small model with repeated function names and nested modules."""

    def __init__(self) -> None:
        """Initialise the model."""

        super().__init__()
        self.block = _Phase2Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with two ReLU operations.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        block_out = self.block(x)
        second_relu = torch.relu(block_out)
        return block_out + second_relu


@pytest.fixture()
def phase2_log() -> tl.ModelLog:
    """Return a model log with repeated functions and nested modules.

    Returns
    -------
    tl.ModelLog
        Completed model log.
    """

    torch.manual_seed(0)
    model = _Phase2Model()
    return tl.log_forward_pass(model, torch.randn(2, 3), vis_opt="none")


@pytest.mark.smoke
def test_selector_factories_are_immutable_hashable_and_repr_friendly() -> None:
    """Selector constructors return stable immutable value objects."""

    selector = tl.label("relu_1_2")

    assert selector == tl.label("relu_1_2")
    assert hash(selector) == hash(tl.label("relu_1_2"))
    assert repr(selector) == "tl.label('relu_1_2')"
    assert repr(tl.func("relu")) == "tl.func('relu')"
    assert repr(tl.module("block")) == "tl.module('block')"
    assert repr(tl.contains("relu")) == "tl.contains('relu')"
    assert repr(tl.in_module("block")) == "tl.in_module('block')"


@pytest.mark.smoke
def test_resolve_sites_supports_all_phase2_match_types(phase2_log: tl.ModelLog) -> None:
    """Selectors resolve labels, functions, modules, containment, and predicates."""

    relu_sites = phase2_log.resolve_sites(tl.func("relu"), max_fanout=4)
    first_relu_label = relu_sites.labels()[0]

    assert isinstance(relu_sites, SiteTable)
    assert relu_sites.labels() == ("relu_1_2", "relu_2_3")
    assert phase2_log.resolve_sites(tl.label(first_relu_label)).labels() == (first_relu_label,)
    assert phase2_log.resolve_sites(tl.label(f"{first_relu_label}:1")).labels() == (
        first_relu_label,
    )
    assert phase2_log.resolve_sites(tl.contains("linear")).labels() == ("linear_1_1",)
    assert phase2_log.resolve_sites(tl.module("block.linear")).labels() == ("linear_1_1",)
    assert phase2_log.resolve_sites(tl.module("block")).labels() == ("relu_1_2",)
    assert phase2_log.resolve_sites(tl.in_module("block"), max_fanout=4).labels() == (
        "linear_1_1",
        "relu_1_2",
    )
    assert phase2_log.resolve_sites(
        tl.where(lambda site: site.func_name == "__add__", name_hint="adds")
    ).labels() == ("add_1_4",)


@pytest.mark.smoke
def test_site_table_methods_and_getitem_singleton(phase2_log: tl.ModelLog) -> None:
    """SiteTable exposes sequence, filtering, first, labels, and DataFrame helpers."""

    table = phase2_log.find_sites(tl.func("relu"), max_fanout=4)

    assert len(table) == 2
    assert [site.layer_label for site in table] == ["relu_1_2", "relu_2_3"]
    assert table[0].layer_label == "relu_1_2"
    assert isinstance(table[:1], SiteTable)
    assert (
        table.where(lambda site: site.layer_label == "relu_2_3").first().layer_label == "relu_2_3"
    )
    assert table.labels() == ("relu_1_2", "relu_2_3")
    assert list(table.to_dataframe()["layer_label"]) == ["relu_1_2", "relu_2_3"]
    assert phase2_log[tl.label("linear_1_1")].layer_label == "linear_1_1"
    assert phase2_log["linear"].layer_label == "linear_1_1"


@pytest.mark.smoke
def test_resolution_errors_strict_mode_and_warnings(phase2_log: tl.ModelLog) -> None:
    """Resolver fails closed for strict strings, empty matches, and excess fanout."""

    with pytest.raises(SiteResolutionError, match="Bare strings"):
        tl.resolve_sites(phase2_log, "relu", strict=True)

    with pytest.raises(SiteResolutionError, match="matched 0 sites"):
        phase2_log.resolve_sites(tl.func("does_not_exist"))

    with pytest.raises(SiteResolutionError, match="max_fanout=None"):
        tl.resolve_sites(phase2_log, tl.func("relu"), max_fanout=None)  # type: ignore[arg-type]

    with pytest.raises(SiteAmbiguityError, match="exceeding max_fanout=1"):
        phase2_log.resolve_sites(tl.func("relu"), max_fanout=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        phase2_log.resolve_sites(tl.func("relu"), max_fanout=4)

    assert any(issubclass(item.category, MultiMatchWarning) for item in caught)


@pytest.mark.smoke
def test_selector_composition_intersection_and_union(phase2_log: tl.ModelLog) -> None:
    """Only ``&`` and ``|`` composition are supported in Phase 2."""

    intersection = tl.func("relu") & tl.in_module("block")
    union = tl.module("block.linear") | tl.module("block")

    assert phase2_log.resolve_sites(intersection, max_fanout=4).labels() == ("relu_1_2",)
    assert phase2_log.resolve_sites(union, max_fanout=4).labels() == (
        "linear_1_1",
        "relu_1_2",
    )


@pytest.mark.smoke
def test_target_specs_and_empty_site_table_behave_clearly(phase2_log: tl.ModelLog) -> None:
    """Selectors convert to target specs and empty tables reject ``first``."""

    target_spec = tl.func("relu").to_target_spec()
    target_spec.selector_value = ["relu"]
    target_spec.metadata["nested"] = {"labels": ["relu_1_2"]}
    frozen_spec = target_spec.freeze()
    empty_table = SiteTable(())

    assert frozen_spec.selector_value == ("relu",)
    assert frozen_spec.metadata == (("nested", (("labels", ("relu_1_2",)),)),)
    with pytest.raises(FrozenInstanceError):
        frozen_spec.selector_kind = "label"  # type: ignore[misc]

    portable_target_spec = tl.func("relu").to_target_spec()
    assert phase2_log.resolve_sites(portable_target_spec, max_fanout=4).labels() == (
        "relu_1_2",
        "relu_2_3",
    )
    assert phase2_log.resolve_sites(portable_target_spec.freeze(), max_fanout=4).labels() == (
        "relu_1_2",
        "relu_2_3",
    )
    assert empty_table.labels() == ()
    assert empty_table.to_dataframe().empty
    with pytest.raises(SiteResolutionError, match="empty"):
        empty_table.first()
