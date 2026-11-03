"""Index sets of the per-unit receptive-field walks.

An :class:`_IndexSet` is a sorted union of arithmetic progressions over one
tensor axis, carrying an ``exact`` flag; ``None`` in an :data:`_AxisSets` tuple
means an unconstrained axis. The module imports nothing else from the package
so that the engines, the set walks and their helpers can all share it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import TypeAlias

_PROGRESSION_BUDGET = 16


@dataclass(frozen=True, order=True)
class _Progression:
    """Finite arithmetic progression of integer indices."""

    start: int
    step: int
    count: int

    def __post_init__(self) -> None:
        """Validate canonical progression parameters."""
        if self.step <= 0 or self.count <= 0:
            raise ValueError("progression step and count must be positive")

    @property
    def stop(self) -> int:
        """Return the inclusive final index."""
        return self.start + self.step * (self.count - 1)

    def values(self) -> range:
        """Return a lazy range over all represented indices."""
        return range(self.start, self.stop + 1, self.step)


def _canonical_progression(start: int, step: int, count: int) -> _Progression:
    """Return the run :meth:`_IndexSet.from_values` derives for one progression's values.

    A single value is the run ``(value, 1, 1)``; two or more equally spaced values
    are one run with their spacing.
    """
    return _Progression(start, step if count > 1 else 1, count)


def _progression_sumset(first: _Progression, second: _Progression) -> _Progression | None:
    """Return ``{a + b : a in first, b in second}`` when it is one progression, else ``None``.

    With steps ``g <= G`` and counts ``c`` (for ``g``) and ``C``: when ``G`` is a
    multiple ``q * g`` with ``q <= c``, the sums ``g * (i + q * j)`` for
    ``0 <= i < c`` and ``0 <= j < C`` cover every multiple of ``g`` from ``0`` to
    ``g * (c - 1 + q * (C - 1))`` with no gap. Any other combination returns
    ``None`` and the caller enumerates instead, so the result is always the set
    the enumeration would produce.
    """
    start = first.start + second.start
    if first.count == 1 or second.count == 1:
        other = second if first.count == 1 else first
        return _canonical_progression(start, other.step, other.count)
    small, large = sorted((first, second), key=lambda item: (item.step, item.count))
    if small.step <= 0 or large.step % small.step:
        return None
    ratio = large.step // small.step
    if ratio > small.count:
        return None
    return _Progression(start, small.step, small.count + ratio * (large.count - 1))


def _clipped_progression(item: _Progression, extent: int, *, exact: bool) -> _IndexSet:
    """Intersect one progression with ``[0, extent)`` in closed form.

    Equals ``_IndexSet._from_sorted_unique`` over the surviving values: the
    survivors of an ascending progression are one contiguous run of its terms.
    """
    first = max(0, -(item.start // item.step)) if item.start < 0 else 0
    last = min(item.count - 1, (extent - 1 - item.start) // item.step)
    if last < first:
        return _IndexSet.empty(exact=exact)
    return _IndexSet(
        (_canonical_progression(item.start + item.step * first, item.step, last - first + 1),),
        exact,
    )


@dataclass(frozen=True)
class _IndexSet:
    """Bounded union of arithmetic progressions with honest collapse metadata.

    CANONICAL FORM INVARIANT. Every instance is produced either by
    :meth:`empty`, :meth:`singleton`, :meth:`interval`, :meth:`from_values`, or
    by re-wrapping another instance's ``progressions`` with a weaker ``exact``
    flag. All of those yield exactly the greedy run decomposition that
    :meth:`from_values` derives from the sorted, de-duplicated value list, so
    ``progressions`` is always ascending, disjoint, maximal, and no longer than
    ``_PROGRESSION_BUDGET``. The fast paths below rely on that invariant: they
    are shortcuts for results the general algorithm would reproduce, never a
    different answer.
    """

    progressions: tuple[_Progression, ...]
    exact: bool = True

    @classmethod
    def empty(cls, *, exact: bool = True) -> _IndexSet:
        """Construct an empty index set with the requested exactness."""
        return cls((), exact)

    @classmethod
    def singleton(cls, value: int) -> _IndexSet:
        """Construct a singleton index set."""
        return cls((_Progression(value, 1, 1),))

    @classmethod
    def interval(cls, start: int, stop: int, *, exact: bool = True) -> _IndexSet:
        """Construct an inclusive integer interval."""
        if stop < start:
            return cls.empty(exact=exact)
        return cls((_Progression(start, 1, stop - start + 1),), exact)

    @classmethod
    def from_values(cls, values: Iterable[int], *, exact: bool = True) -> _IndexSet:
        """Compress values into at most sixteen runs, else an inexact dense hull."""
        return cls._from_sorted_unique(sorted(set(values)), exact=exact)

    @classmethod
    def _from_sorted_unique(cls, ordered: Sequence[int], *, exact: bool = True) -> _IndexSet:
        """Compress an already sorted, de-duplicated value list into runs.

        This is the body of :meth:`from_values`; callers that already hold
        ascending distinct values skip the redundant ``sorted(set(...))``.
        """
        if not ordered:
            return cls.empty(exact=exact)
        runs: list[_Progression] = []
        run_start = ordered[0]
        run_step = 1
        run_count = 1
        previous = run_start
        for value in islice(ordered, 1, None):
            difference = value - previous
            previous = value
            if run_count == 1:
                run_step = difference
                run_count = 2
            elif difference == run_step:
                run_count += 1
            else:
                runs.append(_Progression(run_start, run_step, run_count))
                run_start = value
                run_step = 1
                run_count = 1
        runs.append(_Progression(run_start, run_step, run_count))
        if len(runs) > _PROGRESSION_BUDGET:
            return cls.interval(ordered[0], ordered[-1], exact=False)
        return cls(tuple(runs), exact)

    @property
    def is_empty(self) -> bool:
        """Return whether no indices are represented."""
        return not self.progressions

    @property
    def minimum(self) -> int | None:
        """Return the smallest index, or ``None`` for an empty set."""
        return None if self.is_empty else min(item.start for item in self.progressions)

    @property
    def maximum(self) -> int | None:
        """Return the largest index, or ``None`` for an empty set."""
        return None if self.is_empty else max(item.stop for item in self.progressions)

    def values(self) -> tuple[int, ...]:
        """Materialize represented indices in sorted order."""
        progressions = self.progressions
        if len(progressions) == 1:
            item = progressions[0]
            return tuple(range(item.start, item.stop + 1, item.step))
        # Canonical progressions are ascending and disjoint, so chaining them
        # already yields the sorted distinct order the contract promises.
        return tuple(value for item in progressions for value in item.values())

    def clipped(self, extent: int) -> _IndexSet:
        """Intersect this set with ``[0, extent)`` without changing exactness."""
        progressions = self.progressions
        if not progressions:
            return self
        if progressions[0].start >= 0 and progressions[-1].stop < extent:
            # Nothing is dropped, so recompressing would rebuild this exact
            # canonical decomposition with this exact ``exact`` flag.
            return self
        if len(progressions) == 1:
            return _clipped_progression(progressions[0], extent, exact=self.exact)
        return _IndexSet._from_sorted_unique(
            [value for value in self.values() if 0 <= value < extent], exact=self.exact
        )

    @classmethod
    def union(cls, sets: Iterable[_IndexSet]) -> _IndexSet:
        """Union progression sets and enforce the global progression budget."""
        # Path enumeration hands the same set in many times. Union is idempotent
        # and ``all(...)`` over exactness ignores repeats, so collapsing equal
        # operands leaves both the value set and the exactness flag unchanged.
        materialized = tuple(dict.fromkeys(sets))
        if len(materialized) == 1:
            # Recompressing one canonical set with its own exactness is identity.
            return materialized[0]
        return cls.from_values(
            (value for index_set in materialized for value in index_set.values()),
            exact=all(index_set.exact for index_set in materialized),
        )


_AxisSets: TypeAlias = tuple[_IndexSet | None, ...]


__all__: list[str] = []
