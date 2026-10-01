"""The pure collection builder: ``to_model_explorer_dict`` (memo H-1, D10).

Deterministic ``trace -> Model Explorer graph-collection dict`` with no
server, no browser, no GPU; the file/embed/serve writers all wrap this one
core. Unrolled exact execution is ``graphs[0]`` and the default; a rolled
graph is appended only when rolling changes node/edge identity; EPISODE
captures emit ONE collection: the full exact episode graph first, then
zero-padded per-step graphs. The top-level payload carries ONLY the vendor
``GraphCollection`` keys (``label``/``graphs``/``graphSorting``) -- extra
top-level keys are MEASURED to fail Model Explorer's strict parse, so
visible facts ride the ``""`` group row and machine provenance rides the
manifest sidecar (memo D7).
"""

from __future__ import annotations

import json
from typing import Any

from ..._capture_honesty import capture_honesty_facts
from ..._errors import InvalidArgumentError
from ...errors._base import TorchLensError
from .._common import _iter_layers
from ._attrs import AttrContext
from ._build import GraphSpec, build_graph, build_rolled_graph, rolling_changes_identity
from ._options import ModelExplorerOptions

__tl_layer__ = "L8"

#: Closed privacy-profile vocabulary (memo D14): ONE mechanism for tokens,
#: source paths, code lines, and value-derived attrs.
PRIVACY_PROFILES = frozenset({"local", "public"})

EXECUTION_GRAPH_ID = "00-execution"
ROLLED_GRAPH_ID = "01-rolled"


def to_model_explorer_dict(log: Any, **options: Any) -> dict[str, Any]:
    """Build the Model Explorer graph-collection payload for one capture.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    **options:
        Flat keyword spellings of :class:`ModelExplorerOptions` (an unknown
        keyword raises the standard ``TypeError``):

        ``label``
            Collection label; defaults to the trace label or model class name.
        ``privacy_profile``
            ``"local"`` (default) or ``"public"``; public drops tokens, source
            paths, code lines, and value-derived attrs BY CONSTRUCTION.
        ``strict_namespace``
            Refuse typed on malformed module stacks instead of degrading with
            disclosure.
        ``include_source``
            Opt into per-node ``source`` (basename:line) attrs (local profile
            only).
        ``include_rolled``
            ``None`` (default) appends the rolled projection exactly when
            rolling changes node/edge identity; ``False`` suppresses it.
        ``per_step``
            Episode captures only: ``None``/``True`` emit per-step graphs
            after the full episode graph; ``False`` emits the full graph only.
        ``boundary_proxies``
            Episode captures only: deterministic boundary-proxy nodes for
            cross-step edges (default) versus drop-plus-disclose.
        ``step_budget_bytes``
            Serialized-payload budget for the episode collection (memo D13).
        ``max_step_graphs``
            Step-graph count guard, whichever binds first (memo D13).

    Returns
    -------
    dict[str, Any]
        A strict vendor ``GraphCollection`` payload.
    """

    opts = ModelExplorerOptions(**options)
    if opts.privacy_profile not in PRIVACY_PROFILES:
        raise InvalidArgumentError(
            f"Unknown privacy_profile {opts.privacy_profile!r}; the closed vocabulary is "
            f"{sorted(PRIVACY_PROFILES)}",
            code="model_explorer_privacy_profile_invalid",
            remedy="pass privacy_profile='local' or privacy_profile='public'",
            argument="privacy_profile",
        )
    collection_label = str(
        opts.label
        or getattr(log, "trace_label", None)
        or getattr(log, "model_class_name", None)
        or "model"
    )
    attr_context = _attr_context(log, opts.privacy_profile, opts.include_source)
    entries = _iter_layers(log)
    episode_ledger = _episode_ledger(log)
    if episode_ledger is not None:
        from ._episode import EpisodeExportContext, build_episode_graphs

        graphs = build_episode_graphs(
            log,
            entries,
            episode_ledger,
            EpisodeExportContext(
                attr_context=attr_context,
                root_facts=_base_root_facts(log, opts.privacy_profile),
                options=opts,
            ),
        )
        return {"label": collection_label, "graphs": graphs, "graphSorting": "name_asc"}
    graphs = [
        build_graph(
            log,
            entries,
            GraphSpec(
                graph_id=EXECUTION_GRAPH_ID,
                attr_context=attr_context,
                strict_namespace=opts.strict_namespace,
                root_facts={**_base_root_facts(log, opts.privacy_profile), "view": "execution"},
            ),
        ).graph
    ]
    if opts.include_rolled is not False and rolling_changes_identity(entries):
        graphs.append(
            build_rolled_graph(
                log,
                entries,
                graph_id=ROLLED_GRAPH_ID,
                attr_context=attr_context,
                root_facts={
                    **_base_root_facts(log, opts.privacy_profile),
                    "view": "rolled (disclosed DAG projection)",
                },
            ).graph
        )
    return {"label": collection_label, "graphs": graphs, "graphSorting": "name_asc"}


def _attr_context(log: Any, privacy_profile: str, include_source: bool) -> AttrContext:
    """Resolve the trace-level attr facts (nonfinite record, privacy)."""

    nonfinite_labels: frozenset[str] = frozenset()
    basis: str | None = None
    try:
        nonfinite_labels = frozenset(str(item) for item in (log.nonfinite_ops or ()))
        coverage = log.nonfinite_coverage
        basis = str(getattr(coverage, "basis", "") or "") or None
    except (AttributeError, TorchLensError):
        # A capture whose nonfinite record refuses (legacy artifacts, typed)
        # or lacks the surface entirely simply gets no nonfinite attr rows --
        # absence, never a false all-clear.
        nonfinite_labels = frozenset()
        basis = None
    return AttrContext(
        nonfinite_labels=nonfinite_labels,
        nonfinite_basis=basis,
        public=privacy_profile == "public",
        include_source=include_source,
    )


def _episode_ledger(log: Any) -> dict[str, Any] | None:
    """Return the episode status ledger when this capture is an episode."""

    annotations = getattr(log, "annotations", None) or {}
    ledger = annotations.get("episode") if isinstance(annotations, dict) else None
    return ledger if isinstance(ledger, dict) else None


def _base_root_facts(log: Any, privacy_profile: str) -> dict[str, str]:
    """Build the shared ``""`` provenance facts (memo D7 disclosure block)."""

    import torchlens

    facts: dict[str, str] = {
        "produced_by": f"torchlens {torchlens.__version__}",
        "backend": str(getattr(log, "backend", "") or "unknown"),
        "privacy_profile": privacy_profile,
    }
    for key, value in capture_honesty_facts(log).items():
        if key == "schema":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            facts[key] = str(value)
        else:
            facts[key] = json.dumps(value, sort_keys=True, default=str)[:500]
    return facts
