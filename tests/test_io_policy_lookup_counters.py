"""Deterministic call counters for import-free hot paths (persistence policy lookups, Trace misses).

``effective_policy`` runs once per persisted field on every save and load
(about 650 calls per op), and the nested ``BlobRef`` resave guard visits every
portable object. Neither may pay import machinery per call, in either position
of the pre-release switch.

Every count is hermetic (``tests/_hermetic_counters.py``): the switch is pinned
by ``monkeypatch``, pending garbage is collected before the window and the
cyclic collector is off inside it, and only import statements run by package
frames on this thread count, so the result does not depend on which tests ran
earlier in the same worker process.
"""

from __future__ import annotations

from typing import Any

import pytest
from _hermetic_counters import count_package_imports

from torchlens._io import FieldPolicy, prerelease
from torchlens._io.bundle import _contains_nested_blob_refs


class _PortableRecord:
    """Minimal portable object with one KEEP field and one DROP field."""

    _PORTABLE_STATE_SPEC = {"kept": FieldPolicy.KEEP, "dropped": FieldPolicy.DROP}

    def __init__(self, kept: Any, dropped: Any = None) -> None:
        self.kept = kept
        self.dropped = dropped


def test_effective_policy_switch_off_imports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the switch off, N lookups perform zero imports and zero registry lookups."""

    monkeypatch.setattr(prerelease, "_ACTIVE", False)
    calls = {"override": 0}

    def _counting_override(owner_type: type, field_name: str) -> Any:
        calls["override"] += 1
        return None

    monkeypatch.setattr(prerelease, "persisted_policy_override", _counting_override)
    with count_package_imports() as counter:
        for _ in range(1000):
            assert prerelease.effective_policy(_PortableRecord, "dropped", FieldPolicy.DROP) is (
                FieldPolicy.DROP
            )
            assert prerelease.effective_policy(_PortableRecord, "kept", FieldPolicy.KEEP) is (
                FieldPolicy.KEEP
            )
    assert counter.sites == []
    assert calls["override"] == 0


def test_effective_policy_switch_on_still_consults_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reorder keeps the switched-on behaviour: DROP fields consult the registry."""

    seen: list[tuple[type, str]] = []

    def _recording_override(owner_type: type, field_name: str) -> Any:
        seen.append((owner_type, field_name))
        return FieldPolicy.KEEP

    monkeypatch.setattr(prerelease, "persisted_policy_override", _recording_override)
    monkeypatch.setattr(prerelease, "_ACTIVE", True)
    assert prerelease.effective_policy(_PortableRecord, "dropped", FieldPolicy.DROP) is (
        FieldPolicy.KEEP
    )
    assert prerelease.effective_policy(_PortableRecord, "kept", FieldPolicy.KEEP) is (
        FieldPolicy.KEEP
    )
    assert seen == [(_PortableRecord, "dropped")]


def _chain_of_records(length: int) -> Any:
    chain: Any = _PortableRecord(kept=[1, "x", None])
    for _ in range(length - 1):
        chain = _PortableRecord(kept=[chain, {"k": (1, 2)}])
    return chain


@pytest.mark.parametrize("switch_on", [False, True], ids=["switch_off", "switch_on"])
def test_nested_blob_ref_walk_imports_nothing_per_object(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    """Walking 150 nested portable objects performs zero imports, switch off or on."""

    monkeypatch.setattr(prerelease, "_ACTIVE", switch_on)
    # Warm the per-type kind memo outside the window; the claim is per object.
    assert _contains_nested_blob_refs(_chain_of_records(2), set(), set()) is False
    chain = _chain_of_records(150)
    with count_package_imports() as counter:
        assert _contains_nested_blob_refs(chain, set(), set()) is False
    assert counter.sites == []


def test_missing_trace_attribute_path_imports_nothing() -> None:
    """A defaulted ``getattr`` miss on a Trace (tens of thousands per validate) imports nothing."""

    from torchlens._data_classes._trace_missing_attrs import _raise_missing_trace_attribute

    class _TraceLike:
        pass

    probe = _TraceLike()
    with pytest.raises(AttributeError):
        _raise_missing_trace_attribute(probe, "not_a_field")  # type: ignore[arg-type]
    with count_package_imports() as counter:
        for _ in range(1000):
            with pytest.raises(AttributeError):
                _raise_missing_trace_attribute(probe, "not_a_field")  # type: ignore[arg-type]
    assert counter.sites == []
