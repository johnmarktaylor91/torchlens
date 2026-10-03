"""Tests for memoized ``TargetSpec.freeze`` and dedup-scan equivalence."""

from __future__ import annotations

import pickle

import torch

import torchlens as tl
from torchlens.intervention.types import (
    FrozenTargetSpec,
    InterventionSpec,
    TargetSpec,
    TensorSliceSpec,
)


def test_freeze_cache_hit_is_byte_identical() -> None:
    """Repeated freezes of a stable spec reuse one identical frozen view."""

    target = TargetSpec("label", "relu_1_1")
    first = target.freeze()
    second = target.freeze()
    assert second is first
    assert first == TargetSpec("label", "relu_1_1").freeze()


def test_freeze_cache_invalidates_on_field_reassignment() -> None:
    """Reassigned fields always produce a freshly computed frozen view."""

    target = TargetSpec("label", "relu_1_1")
    baseline = target.freeze()

    target.selector_value = "relu_2_2"
    assert target.freeze() == TargetSpec("label", "relu_2_2").freeze()

    target.strict = True
    assert target.freeze() == TargetSpec("label", "relu_2_2", strict=True).freeze()

    slice_spec = TensorSliceSpec(positions=(0, 1))
    target.slice_spec = slice_spec
    assert (
        target.freeze()
        == TargetSpec("label", "relu_2_2", strict=True, slice_spec=slice_spec).freeze()
    )
    assert baseline == TargetSpec("label", "relu_1_1").freeze()


def test_freeze_reflects_in_place_metadata_mutation() -> None:
    """Metadata added after a cached freeze appears in the next freeze."""

    target = TargetSpec("label", "relu_1_1")
    assert target.freeze().metadata == ()

    target.metadata["direction"] = "forward"
    mutated = target.freeze()
    assert mutated.metadata == (("direction", "forward"),)
    assert mutated == TargetSpec("label", "relu_1_1", metadata={"direction": "forward"}).freeze()

    target.metadata.clear()
    assert target.freeze() == TargetSpec("label", "relu_1_1").freeze()


def test_freeze_snapshots_mutable_selector_values_per_call() -> None:
    """Container selector values are never cached and track in-place edits."""

    labels = ["relu_1_1"]
    target = TargetSpec("labels", labels)
    assert target.freeze().selector_value == ("relu_1_1",)

    labels.append("relu_2_2")
    assert target.freeze().selector_value == ("relu_1_1", "relu_2_2")

    nested = TargetSpec("labels", ("outer", ["inner"]))
    first = nested.freeze()
    nested.selector_value[1].append("added")
    second = nested.freeze()
    assert first.selector_value == ("outer", ("inner",))
    assert second.selector_value == ("outer", ("inner", "added"))


def test_freeze_cache_is_not_pickled() -> None:
    """Pickled specs carry no cache slot and freeze correctly after load."""

    target = TargetSpec("label", "relu_1_1")
    frozen = target.freeze()
    assert "_tl_frozen_cache" in target.__dict__

    restored = pickle.loads(pickle.dumps(target))
    assert "_tl_frozen_cache" not in restored.__dict__
    assert restored == target
    assert restored.freeze() == frozen


def test_dedup_scan_semantics_unchanged() -> None:
    """The call-site dedup pattern keeps first-seen order and drops repeats."""

    spec = InterventionSpec()
    added: list[FrozenTargetSpec] = []
    for label in ["relu_1_1", "relu_2_2", "relu_1_1", "relu_3_3", "relu_2_2"]:
        target = TargetSpec("label", label)
        if not any(existing.freeze() == target.freeze() for existing in spec.targets):
            spec.targets.append(target)
            added.append(target.freeze())

    assert [target.selector_value for target in spec.targets] == [
        "relu_1_1",
        "relu_2_2",
        "relu_3_3",
    ]
    assert tuple(added) == spec.freeze().targets


def test_predicate_intervention_targets_unchanged() -> None:
    """Dense predicate interventions produce one ordered target per site."""

    class _Chain(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(6):
                x = torch.relu(x + 1.0)
            return x

    trace = tl.trace(
        _Chain(),
        torch.randn(2, 3),
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    frozen = trace.intervention_spec
    labels = [target.selector_value for target in frozen.targets]
    assert labels == sorted(set(labels), key=labels.index)
    assert len(labels) == 6
    assert all(target.selector_kind == "label" for target in frozen.targets)
