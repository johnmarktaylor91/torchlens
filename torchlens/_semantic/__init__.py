"""Semantic facet views for TorchLens records."""

from .._registry._facet_snapshot import register_facet_snapshot_provider
from . import facets, recipes as recipes
from .coverage import (
    FacetCoverageReport as FacetCoverageReport,
    ModuleCoverageRow as ModuleCoverageRow,
)
from .facets import (
    FacetCapabilityFlags as FacetCapabilityFlags,
    FacetMenuItem as FacetMenuItem,
    TransformPrimitive as TransformPrimitive,
)
from .logit_lens import (
    LogitLensEntry as LogitLensEntry,
    logit_lens,
    logit_lens_predictions,
)

__all__ = [
    "logit_lens",
    "logit_lens_predictions",
]

# The engine builds every Trace's facet snapshot through the basis slot and never imports
# this appliance; loading the package installs the direct function over the capture
# door's lazy provider.
register_facet_snapshot_provider(facets.snapshot)
