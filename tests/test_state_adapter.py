"""Tests for class-agnostic TorchLens state adapter helpers."""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from torchlens._data_classes._state_adapter import state_items
from torchlens._io import ArtifactError, FieldPolicy
from torchlens._io.runnable import assert_sparse_core_has_no_tensor_payload
from torchlens._io.scrub import _scrub_value, _ScrubOptions
from torchlens.errors.runnable import SparseCorePayloadError


class _SlottedState:
    """Small slotted object for adapter enumeration checks."""

    __slots__ = ("alpha", "beta", "unset")

    def __init__(self) -> None:
        """Populate two of three declared slots."""

        self.alpha = 1
        self.beta = 2


class _MixedState:
    """Small object with independently populated dict and slotted fields."""

    __slots__ = ("slot_value", "__dict__")

    def __init__(self, slot_value: object, dict_value: object) -> None:
        """Populate one slot and one dict-backed field.

        Parameters
        ----------
        slot_value:
            Value stored in the slotted field.
        dict_value:
            Value stored in the dict-backed field.
        """

        self.slot_value = slot_value
        self.dict_value = dict_value


class _IncompletePortableState:
    """Portable-state object with an intentionally missing field policy."""

    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {"covered": FieldPolicy.KEEP}

    def __init__(self) -> None:
        """Populate a missing field to exercise the scrub tripwire."""

        self.covered = "ok"
        self.missing = "tripwire"


def test_state_items_enumerates_every_set_slot_field() -> None:
    """The adapter enumerates set slots and skips unset slots."""

    obj = _SlottedState()

    assert list(state_items(obj)) == [("alpha", 1), ("beta", 2)]


def test_state_items_enumerates_dict_and_slot_fields_for_mixed_object() -> None:
    """Mixed-shape objects expose every dict and slotted state field once."""

    obj = _MixedState(slot_value="slot", dict_value="dict")

    assert list(state_items(obj)) == [("dict_value", "dict"), ("slot_value", "slot")]


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("slot_value", "dict_value", "expected_field"),
    ((torch.ones(1), None, "slot_value"),),
)
def test_sparse_core_backstop_inspects_mixed_dict_and_slot_fields(
    slot_value: object,
    dict_value: object,
    expected_field: str,
) -> None:
    """Sparse-core traversal catches tensor payloads in either mixed field store."""

    with pytest.raises(SparseCorePayloadError, match=expected_field) as caught:
        assert_sparse_core_has_no_tensor_payload(_MixedState(slot_value, dict_value))
    assert caught.value.fields["code"] == "sparse_core_payload_invalid"
    # The typed tripwire stays catchable as AssertionError for historical callers.
    assert isinstance(caught.value, AssertionError)


def test_scrub_completeness_tripwire_uses_adapter_enumeration() -> None:
    """Scrub still raises when adapter-enumerated state lacks a field policy."""

    options = _ScrubOptions(
        include_outs=True,
        include_grads=True,
        include_saved_args=True,
        include_rng_states=True,
    )

    with pytest.raises(ArtifactError, match="missing from _PORTABLE_STATE_SPEC"):
        _scrub_value(
            _IncompletePortableState(),
            options,
            memo={},
            blob_specs=[],
            blob_counter=[0],
        )
