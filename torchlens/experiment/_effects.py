"""The effect table read view: ``EffectsView`` and ``bundle_effects``.

A ``site_sweep`` operation (flat ``sweep_sites`` or axis-shaped
``sweep_grid``) persists one ``MemberEffectTable`` on the Bundle; this
module serves it as a read view with ranking, sub-views and the explicit
Selection conversion. ``EffectsView.to_grid`` rebuilds a ``SweepGrid`` when
the operation recorded axes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..bundle._lineage import MemberEffectRow, MemberEffectTable
from ..errors._episode import BundleExperimentError

if TYPE_CHECKING:
    from ..bundle import Bundle

__all__ = ["EffectsView", "bundle_effects"]


@dataclass(frozen=True)
class EffectsView:
    """Read view over one persisted MemberEffectTable (item 8).

    ``most_changed()`` here is the MEMBER axis ("which candidate mattered"),
    unambiguous because rows are per candidate; ``Bundle.rank_changes`` keeps
    its SITE axis. ``selection()`` is the EXPLICIT tier-c conversion: the
    union of the top candidates' WHERE regions, session-only (a loaded
    artifact carries no Selection objects and refuses typed).
    """

    table: MemberEffectTable
    _selections: Mapping[str, Any] | None = None
    #: The recorded ``site_sweep`` operation's params (a grid's axes ride
    #: here) and the bundle the view was read from; both None on a sub-view
    #: built without them.
    _operation_params: Mapping[str, Any] | None = None
    _bundle: Any = None

    @property
    def rows(self) -> tuple[MemberEffectRow, ...]:
        return self.table.rows

    def to_grid(self) -> Any:
        """Rebuild the ``SweepGrid`` of a ``sweep_grid`` operation from the table.

        Reads the persisted effect table and the operation's recorded
        ``axes`` (no live members needed, so it works on a loaded artifact).

        Returns
        -------
        SweepGrid
            The labeled result over this view's rows.

        Raises
        ------
        BundleExperimentError
            ``grid_axes_unavailable`` when the operation recorded no axes;
            ``grid_result_shape_mismatch`` when rows and axes disagree.
        """

        from ._grid import grid_from_effects

        return grid_from_effects(self)

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for row in self.table.rows:
            if row.candidate_id == "__baseline__":
                continue
            counts[row.status] = counts.get(row.status, 0) + 1
        summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        return (
            f"EffectsView(operation={self.table.operation_id!r}, {summary or 'empty'}, "
            f"retain={self.table.retain_policy}, lane={self.table.lane})"
        )

    @property
    def baseline_value(self) -> float | None:
        """The un-edited baseline's metric value (None when not measured)."""

        for row in self.table.rows:
            if row.candidate_id == "__baseline__":
                return row.value
        return None

    def rank_changes(self, k: int | None = None) -> list[tuple[str, float]]:
        """Rank candidates by |value - baseline| descending (member axis)."""

        base = self.baseline_value or 0.0
        scored = [
            (row.candidate_id, abs((row.value or 0.0) - base))
            for row in self.table.rows
            if row.candidate_id != "__baseline__" and row.value is not None
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored if k is None else scored[:k]

    def top(self, k: int) -> EffectsView:
        """Sub-view holding the k largest-effect measured candidates."""

        keep = {name for name, _ in self.rank_changes(k=k)}
        rows = tuple(
            row
            for row in self.table.rows
            if row.candidate_id == "__baseline__" or row.candidate_id in keep
        )
        sub_table = MemberEffectTable(
            operation_id=self.table.operation_id,
            rows=rows,
            baseline_member=self.table.baseline_member,
            metric_repr=self.table.metric_repr,
            edit_repr=self.table.edit_repr,
            lane=self.table.lane,
            retain_policy=self.table.retain_policy,
        )
        return EffectsView(
            table=sub_table,
            _selections=self._selections,
            _operation_params=self._operation_params,
            _bundle=self._bundle,
        )

    def to_selection(self) -> Any:
        """EXPLICIT tier-c conversion: the union of this view's WHERE regions.

        Session-only: Selection objects are never persisted, so a view over
        a loaded artifact refuses typed. The decision rule (this view's
        candidate subset) is the caller's explicit act — TorchLens supplies
        no interestingness threshold.
        """

        if not self._selections:
            raise BundleExperimentError(
                "effects.selection() needs the sweep's session WHERE objects; "
                "a loaded artifact carries only the numeric table (Selections "
                "are session-only by design). Re-run the sweep, or rebuild the "
                "region from the candidate ids.",
                code="effects_selection_unavailable",
            )
        composed = None
        for row in self.table.rows:
            if row.candidate_id == "__baseline__":
                continue
            where = self._selections.get(row.candidate_id)
            if where is None:
                continue
            lifted = where
            converter = getattr(lifted, "__selection__", None)
            if callable(converter):
                lifted = converter()
            composed = lifted if composed is None else composed | lifted
        if composed is None:
            raise BundleExperimentError(
                "effects.selection() found no convertible WHERE region among "
                "this view's candidates (facet selectors and labels convert; "
                "released candidates keep their rows but not their regions).",
                code="effects_selection_unavailable",
            )
        return composed


def bundle_effects(bundle: Bundle, operation_id: str | None = None) -> EffectsView:
    """Serve the stored effect table (latest by default) as an EffectsView."""

    tables = bundle._effect_tables
    if not tables:
        raise BundleExperimentError(
            "this bundle carries no effect tables; run site_sweep (or an "
            "engine customer) to produce one.",
            code="effects_table_missing",
        )
    if operation_id is None:
        operation_id = next(reversed(tables))
    if operation_id not in tables:
        raise BundleExperimentError(
            f"no effect table for operation {operation_id!r}; stored tables: {list(tables)}",
            code="effects_table_missing",
            operation_id=operation_id,
        )
    selections = getattr(bundle, "_session_effect_selections", None)
    params = next(
        (dict(op.params) for op in bundle.history if op.operation_id == operation_id), None
    )
    return EffectsView(
        table=tables[operation_id],
        _selections=selections,
        _operation_params=params,
        _bundle=bundle,
    )
