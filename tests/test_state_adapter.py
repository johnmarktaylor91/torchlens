"""Tests for class-agnostic TorchLens state adapter helpers."""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from torchlens._io import FieldPolicy, TorchLensIOError
from torchlens._io.runnable import assert_sparse_core_has_no_tensor_payload
from torchlens._io.scrub import _ScrubOptions, _scrub_value
from torchlens.data_classes._state_adapter import state_items, state_new, state_restore


class _DictBackedState:
    """Small dict-backed object for adapter round-trip checks."""

    def __init__(self) -> None:
        """Populate deterministic live state."""

        self.alpha = 1
        self.beta = {"nested": [2, 3]}


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

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {"covered": FieldPolicy.KEEP}

    def __init__(self) -> None:
        """Populate a missing field to exercise the scrub tripwire."""

        self.covered = "ok"
        self.missing = "tripwire"


def test_state_items_enumerates_every_set_dict_field() -> None:
    """The adapter enumerates all live fields on dict-backed objects."""

    obj = _DictBackedState()

    assert list(state_items(obj)) == list(vars(obj).items())


def test_state_items_enumerates_every_set_slot_field() -> None:
    """The adapter enumerates set slots and skips unset slots."""

    obj = _SlottedState()

    assert list(state_items(obj)) == [("alpha", 1), ("beta", 2)]


def test_state_items_enumerates_dict_and_slot_fields_for_mixed_object() -> None:
    """Mixed-shape objects expose every dict and slotted state field once."""

    obj = _MixedState(slot_value="slot", dict_value="dict")

    assert list(state_items(obj)) == [("dict_value", "dict"), ("slot_value", "slot")]


@pytest.mark.parametrize(
    ("slot_value", "dict_value", "expected_field"),
    (
        (torch.ones(1), None, "slot_value"),
        (None, torch.ones(1), "dict_value"),
    ),
)
def test_sparse_core_backstop_inspects_mixed_dict_and_slot_fields(
    slot_value: object,
    dict_value: object,
    expected_field: str,
) -> None:
    """Sparse-core traversal catches tensor payloads in either mixed field store."""

    with pytest.raises(AssertionError, match=expected_field):
        assert_sparse_core_has_no_tensor_payload(_MixedState(slot_value, dict_value))


def test_state_new_restore_round_trips_dict_backed_state() -> None:
    """Uninitialized objects can be restored from adapter-enumerated state."""

    obj = _DictBackedState()
    restored = state_restore(state_new(type(obj)), dict(state_items(obj)))

    assert type(restored) is type(obj)
    assert vars(restored) == vars(obj)
    assert restored is not obj


def test_state_new_restore_round_trips_slotted_state() -> None:
    """Uninitialized slotted objects can be restored slot by slot."""

    obj = _SlottedState()
    restored = state_restore(state_new(type(obj)), dict(state_items(obj)))

    assert type(restored) is type(obj)
    assert list(state_items(restored)) == list(state_items(obj))
    assert restored is not obj


def test_scrub_completeness_tripwire_uses_adapter_enumeration() -> None:
    """Scrub still raises when adapter-enumerated state lacks a field policy."""

    options = _ScrubOptions(
        include_outs=True,
        include_grads=True,
        include_saved_args=True,
        include_rng_states=True,
    )

    with pytest.raises(TorchLensIOError, match="missing from PORTABLE_STATE_SPEC"):
        _scrub_value(
            _IncompletePortableState(),
            options,
            memo={},
            blob_specs=[],
            blob_counter=[0],
        )
