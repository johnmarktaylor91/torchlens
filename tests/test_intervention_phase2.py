"""Smoke tests for intervention Phase 2 selector resolution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import (
    AmbiguousSiteError,
    SiteResolutionError,
)
from torchlens.intervention.selectors import output_at


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
def phase2_log() -> tl.Trace:
    """Return a model log with repeated functions and nested modules.

    Returns
    -------
    tl.Trace
        Completed model log.
    """

    torch.manual_seed(0)
    model = _Phase2Model()
    return tl.trace(model, torch.randn(2, 3))


def test_selector_factories_are_immutable_hashable_and_repr_friendly() -> None:
    """Selector constructors return stable immutable value objects."""

    selector = tl.label("relu_1_2")

    assert selector == tl.label("relu_1_2")
    assert hash(selector) == hash(tl.label("relu_1_2"))
    assert repr(selector) == "tl.label('relu_1_2')"
    assert repr(tl.func("relu")) == "tl.func('relu')"
    assert repr(tl.module("block")) == "tl.module('block')"
    assert repr(tl.label_contains("relu")) == "tl.label_contains('relu')"
    assert repr(tl.in_module("block")) == "tl.in_module('block')"
    assert repr(output_at("logits")) == "tl.intervention.output_at(('logits',))"
    assert repr(tl.without_op()) == "tl.without_op()"


def test_trace_getitem_supports_all_phase2_match_types(phase2_log: tl.Trace) -> None:
    """``trace[selector]`` resolves labels, functions, modules, containment, and predicates."""

    relu_sites = phase2_log[tl.func("relu")]
    first_relu_label = phase2_log[relu_sites.labels[0]].layer_label

    assert isinstance(relu_sites, tl.Selection)
    assert relu_sites.labels == ("relu_1_2:1", "relu_2_3:1")
    assert phase2_log[tl.label(first_relu_label)].layer_label == first_relu_label
    assert phase2_log[tl.label(f"{first_relu_label}:1")].layer_label == first_relu_label
    assert phase2_log[tl.label_contains("linear")].labels == ("linear_1_1:1",)
    assert phase2_log[tl.module("block.linear")].labels == ("linear_1_1:1",)
    assert phase2_log[tl.module("block")].labels == ("relu_1_2:1",)
    block_sites = phase2_log[tl.in_module("block")]
    assert block_sites.labels == ("linear_1_1:1", "relu_1_2:1")
    assert phase2_log[
        tl.where(lambda site: site.func_name == "__add__", name_hint="adds")
    ].labels == ("add_1_4:1",)


def test_selection_sequence_rows_and_getitem_singleton(phase2_log: tl.Trace) -> None:
    """A Selection exposes sequence, slicing, labels, and DataFrame helpers."""

    table = phase2_log[tl.func("relu")]

    assert len(table) == 2
    assert [phase2_log[label].layer_label for label in table.labels] == ["relu_1_2", "relu_2_3"]
    assert table[0].key == ("relu_1_2", 1)
    assert isinstance(table[:1], tl.Selection)
    assert [label for label in table.labels if phase2_log[label].layer_label == "relu_2_3"] == [
        "relu_2_3:1"
    ]
    assert table.labels == ("relu_1_2:1", "relu_2_3:1")
    assert list(table.to_pandas()["label"]) == ["relu_1_2:1", "relu_2_3:1"]
    assert phase2_log[tl.label("linear_1_1")].layer_label == "linear_1_1"
    assert phase2_log["linear"].layer_label == "linear_1_1"


def test_resolution_empty_selection_and_one_thing_refusals(phase2_log: tl.Trace) -> None:
    """A set selector may be empty and uncapped; a one-thing spelling must match one site."""

    assert phase2_log[tl.func("does_not_exist")].is_empty
    assert len(phase2_log[tl.func("does_not_exist")]) == 0
    assert len(phase2_log[tl.func("relu")]) == 2

    with pytest.raises(SiteResolutionError, match="matched 0 sites"):
        phase2_log[tl.label("does_not_exist")]

    with pytest.raises(AmbiguousSiteError, match="names one site"):
        phase2_log[tl.func("relu").to_target_spec()]


def test_selector_composition_intersection_and_union(phase2_log: tl.Trace) -> None:
    """Only ``&`` and ``|`` composition are supported in Phase 2."""

    intersection = tl.func("relu") & tl.in_module("block")
    union = tl.module("block.linear") | tl.module("block")

    assert phase2_log[intersection].labels == ("relu_1_2:1",)
    assert phase2_log[union].labels == ("linear_1_1:1", "relu_1_2:1")


def test_target_specs_freeze_and_resolve_clearly(phase2_log: tl.Trace) -> None:
    """Selectors convert to target specs that freeze immutably and resolve one site."""

    target_spec = tl.func("relu").to_target_spec()
    target_spec.selector_value = ["relu"]
    target_spec.metadata["nested"] = {"labels": ["relu_1_2"]}
    frozen_spec = target_spec.freeze()

    assert frozen_spec.selector_value == ("relu",)
    assert frozen_spec.metadata == (("nested", (("labels", ("relu_1_2",)),)),)
    with pytest.raises(FrozenInstanceError):
        frozen_spec.selector_kind = "label"  # type: ignore[misc]

    portable_target_spec = tl.func("relu").to_target_spec()
    with pytest.raises(AmbiguousSiteError, match="names one site"):
        phase2_log[portable_target_spec]
    with pytest.raises(AmbiguousSiteError, match="names one site"):
        phase2_log[portable_target_spec.freeze()]
    one_spec = tl.module("block.linear").to_target_spec()
    assert phase2_log[one_spec].layer_label == "linear_1_1"
    assert phase2_log[one_spec.freeze()].layer_label == "linear_1_1"
