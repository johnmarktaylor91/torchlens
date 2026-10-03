"""grind-r3 T-CACHES: RF schema-operand-slot cache bounds and fail-closed lookup.

Round-2 F39-4a: the unbounded-negative-entry leak was fixed by NOT caching
``None``, which converted every unrecognized func name (loaded traces, custom
ops, non-aten labels) into an unbounded RE-computation -- torch does not
memoize a failed ``torch.ops.aten`` lookup, so each miss re-entered the C++
registry per (edge, arg) on every solve. Misses now land in a bounded FIFO
companion. Sibling: ``torch.ops.aten`` is a live namespace instance whose
plain Python attributes resolve before operator lookup, so a name like
``name`` used to sail past the ``None`` guard and die with a raw
``AttributeError`` out of a public engine documented as failing closed.
"""

from __future__ import annotations

import pytest

from torchlens.receptive_field import _engine


def _snapshot():
    return dict(_engine._SCHEMA_OPERAND_SLOTS_CACHE), dict(_engine._SCHEMA_OPERAND_MISS_NAMES)


def _restore(snapshot) -> None:
    _engine._SCHEMA_OPERAND_SLOTS_CACHE.clear()
    _engine._SCHEMA_OPERAND_SLOTS_CACHE.update(snapshot[0])
    _engine._SCHEMA_OPERAND_MISS_NAMES.clear()
    _engine._SCHEMA_OPERAND_MISS_NAMES.update(snapshot[1])


def test_unrecognized_names_are_negative_cached_not_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated miss consults the bounded miss set, not the C++ registry."""

    snapshot = _snapshot()
    calls: list[str] = []
    real_compute = _engine._compute_schema_operand_slots

    def counting_compute(canonical: str):
        calls.append(canonical)
        return real_compute(canonical)

    monkeypatch.setattr(_engine, "_compute_schema_operand_slots", counting_compute)
    try:
        _engine._SCHEMA_OPERAND_SLOTS_CACHE.clear()
        _engine._SCHEMA_OPERAND_MISS_NAMES.clear()
        for _ in range(5):
            assert (
                _engine._schema_edge_is_metadata_only("definitely_not_an_op", "positional", (1,))
                is False
            )
        assert calls == ["definitely_not_an_op"], (
            "a known-miss name must be computed once, then served from the miss cache"
        )
    finally:
        _restore(snapshot)


def test_miss_cache_is_bounded_fifo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct unrecognized names beyond the cap evict FIFO, never grow."""

    snapshot = _snapshot()
    monkeypatch.setattr(_engine, "_SCHEMA_OPERAND_MISS_NAMES_MAX_ENTRIES", 16)
    try:
        _engine._SCHEMA_OPERAND_MISS_NAMES.clear()
        for index in range(48):
            _engine._schema_edge_is_metadata_only(f"tl_missing_{index}", "positional", (1,))
        assert len(_engine._SCHEMA_OPERAND_MISS_NAMES) <= 16
        assert "tl_missing_47" in _engine._SCHEMA_OPERAND_MISS_NAMES
        assert "tl_missing_0" not in _engine._SCHEMA_OPERAND_MISS_NAMES
    finally:
        _restore(snapshot)


@pytest.mark.smoke
def test_namespace_attribute_names_fail_closed_not_raw() -> None:
    """Names resolving to non-packet namespace attributes return, never raise.

    ``.strip("_")`` maps ``name`` / ``_name`` / ``__name__`` spellings onto
    ``getattr(torch.ops.aten, "name")`` -- a plain string, whose
    ``.overloads()`` access used to raise a raw AttributeError.
    """

    snapshot = _snapshot()
    try:
        for spelling in ("name", "_name_", "__module__"):
            assert _engine._schema_edge_is_metadata_only(spelling, "positional", (0,)) is False
        assert _engine._compute_schema_operand_slots("name") is None
    finally:
        _restore(snapshot)


@pytest.mark.smoke
def test_positive_cache_still_serves_real_operators() -> None:
    """Real aten packets keep resolving and memoizing (no false negatives)."""

    snapshot = _snapshot()
    try:
        _engine._SCHEMA_OPERAND_SLOTS_CACHE.clear()
        _engine._SCHEMA_OPERAND_MISS_NAMES.clear()
        slots = _engine._compute_schema_operand_slots("conv2d")
        assert slots is not None
        assert slots.operand_positions
    finally:
        _restore(snapshot)
