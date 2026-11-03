"""Facet-coverage reporting for the recipe-maintenance pipeline.

``facet_coverage`` inspects a completed trace and reports, per module record,
which facet recipes matched, which declared facets are available, and which
are absent with their typed reasons. The report is the machine-readable input
to the facet-maintenance pipeline (``tools/facet_maintenance/``): it surfaces
architecture families whose modules carry NO semantic classification so a
reviewing human or agent can decide whether a recipe is warranted. The
pipeline produces PROPOSALS FOR REVIEW only -- nothing here writes or
registers recipes. Every spelling is DOCUMENTED-UNSTABLE pending
naming-session ratification.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

__all__ = ["FacetCoverageReport", "ModuleCoverageRow", "facet_coverage"]


@dataclass(frozen=True)
class ModuleCoverageRow:
    """Facet coverage for one module record.

    Parameters
    ----------
    address:
        Module address in the trace.
    class_name:
        Module class name.
    recipes:
        Names of the facet recipes that matched this module (empty when the
        module has only structural facets).
    available:
        Recipe-produced facet names readable in this capture.
    missing:
        ``(facet, status, detail)`` for declared-but-unavailable facets;
        ``status`` is the typed absence status (``needs_capture``,
        ``structurally_absent``, ``declared_not_produced``).
    note:
        Disclosed reason the facet view itself was unavailable (e.g. a
        multi-call module refuses the single-call ``facets`` accessor typed);
        empty when the view resolved.
    """

    address: str
    class_name: str
    recipes: tuple[str, ...]
    available: tuple[str, ...]
    missing: tuple[tuple[str, str, str], ...]
    note: str = ""


@dataclass(frozen=True)
class FacetCoverageReport:
    """Whole-trace facet-coverage report.

    Parameters
    ----------
    rows:
        One row per module record, in trace order.
    """

    rows: tuple[ModuleCoverageRow, ...]

    @property
    def classified(self) -> tuple[ModuleCoverageRow, ...]:
        """Return rows where at least one semantic recipe matched."""

        return tuple(row for row in self.rows if row.recipes)

    @property
    def unclassified(self) -> tuple[ModuleCoverageRow, ...]:
        """Return rows with no semantic recipe (structural facets only).

        These are the pipeline's candidates: a reviewing human or agent judges
        whether each class deserves a recipe. The report deliberately applies
        NO "looks semantic" heuristic -- a wrong candidate label would bias
        review toward fabricating facets.
        """

        return tuple(row for row in self.rows if not row.recipes and not row.note)

    @property
    def unresolved(self) -> tuple[ModuleCoverageRow, ...]:
        """Return rows whose facet view itself refused (disclosed, not judged)."""

        return tuple(row for row in self.rows if row.note)

    def recipe_counts(self) -> dict[str, int]:
        """Return how many modules each recipe classified."""

        counts: Counter[str] = Counter()
        for row in self.rows:
            counts.update(row.recipes)
        return dict(counts)

    def missing_counts(self) -> dict[tuple[str, str], int]:
        """Return ``(facet, status) -> count`` over all declared absences."""

        counts: Counter[tuple[str, str]] = Counter()
        for row in self.rows:
            counts.update((facet, status) for facet, status, _detail in row.missing)
        return dict(counts)

    def to_markdown(self) -> str:
        """Render the report as a review-ready markdown fragment.

        Returns
        -------
        str
            Markdown with a summary, the classified-module table, declared
            absences, and the unclassified class inventory.
        """

        lines = [
            f"Modules: {len(self.rows)} | classified: {len(self.classified)} | "
            f"unclassified (structural only): {len(self.unclassified)} | "
            f"unresolved views: {len(self.unresolved)}",
            "",
            "| address | class | recipes | available facets | missing (status) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in self.classified:
            missing = ", ".join(f"{facet} ({status})" for facet, status, _ in row.missing)
            lines.append(
                f"| {row.address} | {row.class_name} | {', '.join(row.recipes)} | "
                f"{', '.join(row.available)} | {missing} |"
            )
        lines.append("")
        lines.append("Unclassified module classes (candidates for recipe review):")
        class_counts: Counter[str] = Counter(row.class_name for row in self.unclassified)
        for class_name, count in sorted(class_counts.items()):
            addresses = [row.address for row in self.unclassified if row.class_name == class_name]
            shown = ", ".join(addresses[:3]) + (" ..." if len(addresses) > 3 else "")
            lines.append(f"- {class_name} x{count} (e.g. {shown})")
        if not self.unclassified:
            lines.append("- (none)")
        if self.unresolved:
            lines.append("")
            lines.append("Unresolved facet views (disclosed, review per-call):")
            for row in self.unresolved:
                lines.append(f"- {row.address} ({row.class_name}): {row.note}")
        return "\n".join(lines)


def facet_coverage(trace: Any) -> FacetCoverageReport:
    """Report per-module facet-recipe coverage for a completed trace.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.

    Returns
    -------
    FacetCoverageReport
        Structured coverage rows in trace module order.
    """

    from ..errors import InvalidArgumentError

    rows: list[ModuleCoverageRow] = []
    for module in trace.modules:
        try:
            view = module.facets
            menu = view.declared()
        except InvalidArgumentError as exc:
            # A reused (multi-call) module refuses the single-call facets
            # accessor typed; the audit discloses the row instead of dying.
            rows.append(
                ModuleCoverageRow(
                    address=str(getattr(module, "address", "")),
                    class_name=str(getattr(module, "class_name", "")),
                    recipes=(),
                    available=(),
                    missing=(),
                    note=str(exc),
                )
            )
            continue
        recipe_source = view.function_source
        if recipe_source is None:
            recipes: tuple[str, ...] = ()
        elif isinstance(recipe_source, str):
            recipes = (recipe_source,)
        else:
            recipes = tuple(recipe_source)
        available = tuple(
            str(name)
            for name, item in menu.items()
            if item.status == "available_now" and item.recipe is not None
        )
        missing = tuple(
            (str(name), str(item.status), _absence_detail(view, name))
            for name, item in menu.items()
            if item.status != "available_now"
        )
        rows.append(
            ModuleCoverageRow(
                address=str(getattr(module, "address", "")),
                class_name=str(getattr(module, "class_name", "")),
                recipes=recipes,
                available=available,
                missing=missing,
            )
        )
    return FacetCoverageReport(rows=tuple(rows))


def _absence_detail(view: Any, name: Any) -> str:
    """Return the typed absence detail for a declared-but-missing facet."""

    reason = view._missing.get(name)
    detail = getattr(reason, "detail", None)
    return str(detail) if detail else ""
