"""Tests for class-agnostic TorchLens state adapter helpers."""

from __future__ import annotations

from typing import ClassVar

import pytest

from torchlens._io import FieldPolicy, TorchLensIOError
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
