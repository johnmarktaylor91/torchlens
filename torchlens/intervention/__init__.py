"""Import surface for TorchLens intervention selectors, hooks, reruns, and bundles.

One of this package's public names is resolved LAZILY through the module
``__getattr__`` below rather than imported eagerly, because its module imports
BACK into modules whose own import pulls in ``torchlens.intervention``:

* ``Bundle`` lives in :mod:`torchlens.bundle`, which imports
  ``torchlens.intervention._metrics`` / ``._super`` / ``._topology``; importing any
  of those first executes THIS ``__init__``, so an eager ``from .bundle import
  Bundle`` here made ``import torchlens.bundle`` -- a PUBLIC module -- fail with a
  partially-initialized-module ``ImportError``.

Deferring exactly this name keeps every module in the package standalone
importable (gated by ``tests/test_module_import_isolation.py``) while leaving the
public surface, ``__all__``, and static types unchanged.
"""

from typing import Any

from ._affine_source import set_affine_coordinates
from ._directions import Direction, fit_direction
from ._gradient_contract import keep_grad
from ._metrics import (
    METRIC_REGISTRY as METRIC_REGISTRY,
    pearson_correlation_distance as pearson_correlation_distance,
    resolve_metric as resolve_metric,
)
from ._subspace_edits import subspace_ablate, subspace_interchange
from ._super.super_op import (
    SuperLayer as SuperLayer,
    SuperLayerAccessor as SuperLayerAccessor,
    SuperOpAccessor as SuperOpAccessor,
    TraceAccessor as TraceAccessor,
)
from ._topology.topology import (
    Supergraph as Supergraph,
    SupergraphNode as SupergraphNode,
    TopologyDiff as TopologyDiff,
    build_supergraph as build_supergraph,
)

# F01 surgery live lanes: the capture-free bound executor, the bind-side
# steer wrapper (module named ``steering`` -- the ``steer`` attribute is the
# historical helper from .helpers), and the region noun (derived
# admissibility on the existing verbs).
from .binding import BindReport, BoundModel
from .compose import compose
from .errors import (
    BundleComparisonError,
    SpecPortabilityError,
)
from .helpers import (
    _RESAMPLE_ABLATE_REDIRECT,
    scramble_elements,
)
from .hooks import (
    HookContext,
    normalize_hook as normalize_hook,
)
from .population import Reference, population
from .regions import RegionBoundary, RegionEdge, RegionInstance, RegionTarget
from .save import (
    SpecCompat,
    TargetManifestDiff,
    check_spec_compat,
)
from .selectors import (
    label_contains,
    label_matches,
    model_input,
    model_output,
)
from .sites import SiteSpec as SiteSpec

# The PUBLIC immutable multi-clause spec (C03 substrate; surgery memo 3.1)
# claims the package-level ``InterventionSpec`` name. The historical mutable
# sticky-hook recipe keeps its identity at ``.types.InterventionSpec`` --
# internal code and persisted pickles are untouched.
from .spec import (
    Rule,
)
from .stochastic import (
    SamplingPlan,
    permute_batch,
    resample_rows_from,
    sample_from,
    set_direction_mean,
)
from .types import (
    ArgComponent as ArgComponent,
    FireRecord,
    FrozenInterventionSpec as FrozenInterventionSpec,
    Relationship,
)

__all__ = [
    "BindReport",
    "BoundModel",
    "BundleComparisonError",
    "FireRecord",
    "HookContext",
    "Rule",
    "RegionBoundary",
    "RegionEdge",
    "RegionInstance",
    "RegionTarget",
    "Relationship",
    "SpecCompat",
    "SpecPortabilityError",
    "TargetManifestDiff",
    "check_spec_compat",
    "compose",
    "label_contains",
    "Direction",
    "fit_direction",
    "model_input",
    "label_matches",
    "permute_batch",
    "Reference",
    "population",
    "SamplingPlan",
    "model_output",
    "resample_rows_from",
    "sample_from",
    "set_direction_mean",
    "keep_grad",
    "set_affine_coordinates",
    "subspace_ablate",
    "subspace_interchange",
    "scramble_elements",
]

# Names whose defining module imports back into a module that imports this
# package; see the module docstring. Resolved on first attribute access, after
# this ``__init__`` has finished executing, so no partially-initialized module is
# ever observed.
_LAZY_NAMES: dict[str, str] = {}

#: Removed spellings: each raises the typed ``facade_redirect`` error naming
#: its replacement (clean break, no alias).
_REDIRECTS: dict[str, str] = {
    "resample_ablate": _RESAMPLE_ABLATE_REDIRECT,
    "intervening": "use torchlens.without_op -- intervening was renamed",
    "replay_from": "use Trace.do -- replay_from was removed",
}


def __getattr__(name: str) -> Any:
    """Resolve the cycle-deferred public names on first access.

    Parameters
    ----------
    name:
        Attribute name being looked up on ``torchlens.intervention``.

    Returns
    -------
    Any
        The resolved attribute.

    Raises
    ------
    AttributeError
        For any name this package does not define.
    """

    module_name = _LAZY_NAMES.get(name)
    if module_name is None:
        from .._utils.facade import resolve_facade_attr

        return resolve_facade_attr(
            owner=__name__, name=name, module_globals=globals(), redirects=_REDIRECTS
        )
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # memoize; subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    """Return the public surface including the lazily-resolved names."""

    return sorted(set(__all__) | set(globals()))
