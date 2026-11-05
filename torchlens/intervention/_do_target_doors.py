"""Region and slice target doors of ``do``, reached from ``_trace_edits.apply_do_mutation``.

A ``RegionTarget`` applies as a unit through ``regions.apply_region_do``; a ``TraceSlice``
lifts to its whole-site selection before the selection doors see it. Both modules are
looked up through ``sys.modules`` so ``do()`` imports neither.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ._edit_guards import warn_if_root_mutation

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__: list[str] = []


def apply_region_target_do(  # noqa: PLR0913 - apply_do_mutation's own arguments
    trace: Trace,
    site: Any,
    value_or_hook: Any,
    *,
    engine: str,
    strict: bool,
) -> tuple[str, tuple[str, ...]] | None:
    """Apply ``do`` to a ``RegionTarget``; return ``None`` for any other site."""
    # A RegionTarget (F01) is the region-as-a-unit door: the replay lowering
    # substitutes the region's derived exit values without replaying the
    # interior. Distinct from the TraceSlice lift below, which keeps its shipped
    # member-site family semantics. sys.modules gate keeps do() free of the regions import.
    regions_module = sys.modules.get("torchlens.intervention.regions")
    if regions_module is not None and isinstance(site, regions_module.RegionTarget):
        warn_if_root_mutation(trace)
        return (
            regions_module.apply_region_do(
                trace, site, value_or_hook, engine=engine, strict=strict
            ),
            (),
        )
    return None


def lift_trace_slice(site: Any) -> Any:
    """Return a ``TraceSlice``'s whole-site selection; any other site unchanged."""
    # A TraceSlice targets its member family: lift it to the whole-site
    # QUERY (re-resolved on THIS trace by site name, so a slice built on the
    # source log addresses the same sites on a fork). sys.modules gate keeps do() slice-free.
    slice_module = sys.modules.get("torchlens._trace_slice")
    if slice_module is not None and isinstance(site, slice_module.TraceSlice):
        site = site.__selection__()
    return site
