"""Tests for memoized ``TargetSpec.freeze`` and dedup-scan equivalence."""

from __future__ import annotations

import pickle

from torchlens.intervention._types import (
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

    target.metadata["signal"] = "forward"
    mutated = target.freeze()
    assert mutated.metadata == (("signal", "forward"),)
    assert mutated == TargetSpec("label", "relu_1_1", metadata={"signal": "forward"}).freeze()

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
