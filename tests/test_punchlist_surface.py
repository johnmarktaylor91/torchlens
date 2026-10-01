"""Surface punch-list pins: List-C removals and the zero-evidence demotions.

The 2026-08 completeness audit (E1 + E1-r2, adopted by RULINGS Batch-8 as the
funded punch list) classified the advertised public surface into REMOVE rows
(dead renames, duplicate exports -- deleted outright) and DEMOTE rows
(zero-external-evidence names dropped from ``__all__`` while staying
importable, so nothing breaks and the advertised surface stops overselling).

This module pins both outcomes so they cannot silently regress:

- the REMOVED dead rename is gone entirely (not importable, not advertised);
- every DEMOTED name is absent from its module's ``__all__`` AND still
  importable from the same module (the demotion contract: unadvertised,
  never broken);
- no statically-declared ``__all__`` in the package carries a duplicate
  entry (generalizes the audit's M4 finding -- ``"label"`` appeared twice in
  ``intervention.__all__``);
- the retirement tail stays retired: no ``__all__`` in the package
  re-advertises a Batch-8-deleted alias spelling, and the canonical
  replacement verbs stay advertised.

Demotions re-verified against the live tree at lane F38 time (2026-08-29):
two audit rows had GAINED evidence since the 2026-08-20 sweep and were
deliberately NOT demoted -- ``stats.StreamingStat`` (documented protocol,
docs/monitor_training.md + docs/reference/observability_substrate.md) and
``semantic.AttentionHeadView`` (real semantic-attention test coverage).
"""

from __future__ import annotations

import ast
import functools
import importlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: DEMOTED names per public module: dropped from ``__all__``, still
#: importable. Every row re-verified zero-external-evidence (0 docs, 0
#: non-inventory tests) at demotion time; internal callers import the names
#: directly and are unaffected.
DEMOTED: dict[str, tuple[str, ...]] = {
    "torchlens.merged": ("CollectiveJoin", "JoinRecord", "MergeDerivation", "PerRankRef"),
    "torchlens.distributed": (
        "CollectiveRecognizer",
        "LineageEntry",
        "LineageVector",
        "MembershipLineageVerdict",
    ),
    "torchlens.options": (
        "merge_capture_options",
        "merge_intervention_options",
        "merge_replay_options",
        "merge_save_options",
        "merge_streaming_options",
        "merge_visualization_options",
        "visualization_to_render_kwargs",
    ),
    "torchlens.debug": (
        "PrecisionRow",
        "AuditFinding",
        "CompileCounts",
        "GraphBreak",
        "GraphBreakReport",
    ),
    "torchlens.runnable": (
        "GapSpec",
        "ActivationPayloadMember",
        "RUNNABLE_CALLABLE_REF_SCHEMA_VERSION",
        "control_dependency_site_label",
        "StateByteDigest",
        "SlotByteDigest",
        "refuse_collective_boundary_trace",
    ),
    "torchlens.ir": (
        "register_live_event",
        "is_deferred_value",
        "ContainerSnapshot",
        "EdgeUseKind",
        "ModuleSite",
        "OutputVersionEvent",
        "WalkResult",
        "coerce_deferred_value",
        "OpEventKind",
    ),
    "torchlens.intervention": (
        "METRIC_REGISTRY",
        "SiteSpec",
        "TopologyDiff",
        "pearson_correlation_distance",
        "ArgComponent",
        "FrozenInterventionSpec",
        "SuperLayer",
        "SuperLayerAccessor",
        "SuperOpAccessor",
        "Supergraph",
        "SupergraphNode",
        "TraceAccessor",
        "build_supergraph",
        "normalize_hook",
        "resolve_metric",
    ),
    "torchlens.semantic": (
        "FacetCoverageReport",
        "LogitLensEntry",
        "ModuleCoverageRow",
        "FacetCapabilityFlags",
        "FacetMenuItem",
        "TransformPrimitive",
    ),
    "torchlens.semantic.facets": (
        "FacetKey",
        "RecordScope",
        "mark_current_registry_as_builtins",
        "FacetCapabilityFlags",
        "FacetMenuItem",
        "TransformPrimitive",
    ),
    "torchlens.capture.outcome": (
        "count_committed_ops",
        "current_capture_phase",
        "set_capture_phase",
        "stamp_recording_outcome",
    ),
    "torchlens.experimental": ("AutoCaptureSession",),
    "torchlens.backends": ("BackendName", "TORCH_BACKEND_NAME", "TINYGRAD_BACKEND_NAME"),
    "torchlens.data_classes": ("FuncExecutionContext", "VisualizationOverrides"),
    "torchlens.trace_slice": ("build_slice_between", "build_slice_from_selection"),
    "torchlens.errors": ("EpisodeErrorCode",),
    "torchlens.validation": ("ValidationReplayState",),
}

#: Audit rows deliberately KEPT advertised: evidence gained since the sweep.
KEPT_WITH_EVIDENCE: dict[str, tuple[str, ...]] = {
    "torchlens.stats": ("StreamingStat",),
    "torchlens.semantic": ("AttentionHeadView",),
}

#: Batch-8-retired alias spellings that must never be re-advertised by ANY
#: ``__all__`` in the package (the deleted-spelling lint pins the taught
#: spellings; this pins the advertisement channel specifically).
RETIRED_ADVERTISEMENTS = frozenset(
    {
        "peek",
        "batched_extract",
        "intervening",
        "rerun",
        "replay",
        "replay_from",
        "record_span",
        "validate_saved_outs",
        "validate_trace_saved_outs",
        "get_model_metadata",
    }
)


@functools.lru_cache(maxsize=1)
def _iter_static_all_lists() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return ``(relative_path, names)`` for every static ``__all__`` literal.

    Returns
    -------
    tuple[tuple[str, tuple[str, ...]], ...]
        One row per statically-declared string-list ``__all__`` in shipped
        package code. Dynamic construction (e.g. the errors lazy splice) is
        out of AST reach and is covered by the module-level duplicate check
        at import time for the modules this file imports.

    The walk is cached across the tests that share it (parsing ~900 files
    dominates their runtime), and files whose source never mentions
    ``__all__`` are skipped before parsing -- a strict superset of every
    file that could declare one.
    """

    rows: list[tuple[str, tuple[str, ...]]] = []
    for path in sorted((_REPO_ROOT / "torchlens").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "__all__" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover - shipped code parses
            continue
        for node in tree.body:
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                continue
            value = node.value if isinstance(node, ast.Assign) else node.value
            if not isinstance(value, ast.List):
                continue
            names = tuple(
                elt.value
                for elt in value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            )
            rows.append((path.relative_to(_REPO_ROOT).as_posix(), names))
    return tuple(rows)


@pytest.mark.smoke
def test_removed_dead_rename_is_gone() -> None:
    """The zero-reference rename re-export is deleted, not just unadvertised."""

    validation = importlib.import_module("torchlens.validation")
    assert "validate_trace_saved_outs" not in validation.__all__
    assert not hasattr(validation, "validate_trace_saved_outs")


@pytest.mark.smoke
def test_demoted_names_unadvertised_but_importable() -> None:
    """Every demoted name left ``__all__`` and stayed importable."""

    problems: list[str] = []
    for module_name, names in DEMOTED.items():
        module = importlib.import_module(module_name)
        advertised = set(module.__all__)
        for name in names:
            if name in advertised:
                problems.append(f"{module_name}.{name}: still advertised in __all__")
            if not hasattr(module, name):
                problems.append(f"{module_name}.{name}: no longer importable (demote broke it)")
    assert not problems, "demotion contract violated:\n    " + "\n    ".join(problems)


@pytest.mark.smoke
def test_kept_rows_stay_advertised() -> None:
    """Audit rows kept for cause stay advertised until re-adjudicated."""

    for module_name, names in KEPT_WITH_EVIDENCE.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert name in module.__all__, (
                f"{module_name}.{name} was kept advertised at F38 time because it has "
                "real external evidence; removing it needs a fresh evidence sweep"
            )


@pytest.mark.smoke
def test_no_all_list_carries_duplicates() -> None:
    """No statically-declared ``__all__`` in the package has a duplicate row."""

    offenders = []
    for rel, names in _iter_static_all_lists():
        seen: set[str] = set()
        for name in names:
            if name in seen:
                offenders.append(f"{rel}: duplicate __all__ entry {name!r}")
            seen.add(name)
    assert not offenders, "\n    ".join(offenders)


@pytest.mark.smoke
def test_retired_spellings_stay_unadvertised() -> None:
    """No ``__all__`` re-advertises a Batch-8-retired alias spelling."""

    offenders = []
    for rel, names in _iter_static_all_lists():
        hits = RETIRED_ADVERTISEMENTS.intersection(names)
        if hits:
            offenders.append(f"{rel}: {sorted(hits)}")
    assert not offenders, (
        "retired alias spellings crept back into an __all__ advertisement "
        "(Batch-8 full deletion; docs/reference/deprecations.md names the "
        "canonical replacements):\n    " + "\n    ".join(offenders)
    )


@pytest.mark.smoke
def test_canonical_replacement_verbs_stay_advertised() -> None:
    """The retirement tail's canonical verbs remain the advertised surface."""

    intervention = importlib.import_module("torchlens.intervention")
    for canonical in ("push", "push_from", "run", "do"):
        assert canonical in intervention.__all__, canonical
    import torchlens

    for canonical in ("push", "push_from", "run", "pluck", "extract_dataset", "span"):
        assert canonical in torchlens.__all__, canonical
