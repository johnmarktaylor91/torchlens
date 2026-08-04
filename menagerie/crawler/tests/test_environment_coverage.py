"""Unservable-routing coverage tests.

The load-bearing case is the real m538 proposal from the rung-4 archive: a DGL
``TAGConv`` that authored correctly, routed to ``core`` because its roster zoo
carried no graph marker, and terminalized ``failed:runner`` /
``protocol-violation`` on "routed environment does not install distribution
'dgl'". Its ``library_recipe`` is reproduced verbatim below, and the assessment
must turn that into an honest deferral naming the ``graph`` intent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.environment_coverage import (
    COVERAGE_DEFERRAL_DISPOSITION,
    CoverageBasis,
    CoverageDeferralError,
    CoverageVerdict,
    append_coverage_deferral_row,
    assess_environment_coverage,
    build_coverage_deferral_row,
    coverage_deferral_path,
    declared_library,
    dependency_spec_name,
    find_covering_intents,
    load_coverage_deferral_rows,
)
from menagerie.crawler.envs import load_environment_registry
from menagerie.crawler.recipe import RecipeError, bind_library_artifact_digest

pytestmark = pytest.mark.smoke

#: The exact ``library_recipe`` from the archived m538 author result whose
#: routed ``core`` environment installs no DGL.
M538_LIBRARY_RECIPE: dict[str, Any] = {
    "distribution": "dgl",
    "kwargs": {
        "activation": None,
        "bias": True,
        "in_feats": 10,
        "k": 2,
        "out_feats": 2,
    },
    "module": "dgl.nn.pytorch.conv",
    "pretrained_disable_fields": [],
    "pretrained_fields_absent": True,
    "symbol": "TAGConv",
    "version": "1.1.3",
}


def _m538_implementation() -> dict[str, Any]:
    """Return a mutable implementation block carrying the real m538 recipe."""

    return {
        "recipe_type": "declarative-library",
        "library_recipe": dict(M538_LIBRARY_RECIPE),
    }


#: A trimmed but shape-faithful ``core`` inventory: conda spells the torch
#: distribution ``pytorch``, and nothing here provides ``dgl``.
CORE_PACKAGES: tuple[dict[str, str], ...] = (
    {"name": "python", "version": "3.12.13", "sha256": "sha256:" + "1" * 64},
    {"name": "pytorch", "version": "2.13.0", "sha256": "sha256:" + "2" * 64},
    {"name": "torchvision", "version": "0.28.0", "sha256": "sha256:" + "3" * 64},
    {"name": "timm", "version": "1.0.28", "sha256": "sha256:" + "4" * 64},
)

#: The same shape for ``graph``, which does carry DGL.
GRAPH_PACKAGES: tuple[dict[str, str], ...] = (
    {"name": "python", "version": "3.12.13", "sha256": "sha256:" + "5" * 64},
    {"name": "pytorch", "version": "2.3.1", "sha256": "sha256:" + "6" * 64},
    {"name": "dgl", "version": "2.3.0", "sha256": "sha256:" + "7" * 64},
    {"name": "pyg", "version": "2.6.1", "sha256": "sha256:" + "8" * 64},
)


def test_the_refusal_this_explains_still_refuses() -> None:
    """The digest binding is untouched: an absent library still fails closed.

    This deferral is a disposition, not a relaxation. If this ever stops raising,
    a model whose library is not installed could reach a run with a fabricated or
    absent artifact identity, which is exactly the failure the refusal exists to
    prevent.
    """

    with pytest.raises(RecipeError, match="does not install distribution 'dgl'"):
        bind_library_artifact_digest(_m538_implementation(), list(CORE_PACKAGES))


def test_the_real_m538_routing_is_unservable_and_names_graph() -> None:
    """The archived DGL model becomes an actionable deferral, not a failure."""

    registry = load_environment_registry()
    assessment = assess_environment_coverage(
        _m538_implementation(),
        routed_intent="core",
        routed_packages=list(CORE_PACKAGES),
        registry=registry,
    )
    assert assessment.verdict is CoverageVerdict.UNSERVABLE
    assert assessment.deferred
    assert assessment.library is not None
    assert assessment.library.distribution == "dgl"
    assert assessment.preferred_intent == "graph"
    assert "core" in assessment.explanation
    assert "dgl" in assessment.explanation


def test_a_served_routing_is_admitted_unchanged() -> None:
    """The same model routed to the intent that carries DGL is not deferred."""

    assessment = assess_environment_coverage(
        _m538_implementation(),
        routed_intent="graph",
        routed_packages=list(GRAPH_PACKAGES),
        registry=load_environment_registry(),
    )
    assert assessment.verdict is CoverageVerdict.SERVED
    assert not assessment.deferred


def test_the_renamed_conda_distribution_still_reads_as_served() -> None:
    """``pytorch`` provides ``torch``; the namespace bridge is honoured here too."""

    implementation = {
        "recipe_type": "declarative-library",
        "library_recipe": {"distribution": "torch", "version": "2.13.0"},
    }
    assessment = assess_environment_coverage(
        implementation,
        routed_intent="core",
        routed_packages=list(CORE_PACKAGES),
        registry=None,
    )
    assert assessment.verdict is CoverageVerdict.SERVED


@pytest.mark.parametrize(
    "implementation",
    [
        {"recipe_type": "typed-adapter", "library_recipe": None},
        {"recipe_type": "declarative-library", "library_recipe": None},
        {"recipe_type": "declarative-library", "library_recipe": {"distribution": "dgl"}},
        {"recipe_type": "declarative-library", "library_recipe": {"distribution": "  "}},
        {},
    ],
)
def test_a_non_declarative_recipe_is_never_assessed(implementation: dict[str, Any]) -> None:
    """Only the recipes the digest binding actually resolves are in scope."""

    assessment = assess_environment_coverage(
        implementation,
        routed_intent="core",
        routed_packages=list(CORE_PACKAGES),
        registry=None,
    )
    assert assessment.verdict is CoverageVerdict.NOT_ASSESSABLE
    assert not assessment.deferred
    assert declared_library(implementation) is None


def test_an_unlocked_intent_is_unknown_not_absent() -> None:
    """No inventory means no claim; the model is admitted, never withheld."""

    assessment = assess_environment_coverage(
        _m538_implementation(),
        routed_intent="core",
        routed_packages=[],
        registry=load_environment_registry(),
    )
    assert assessment.verdict is CoverageVerdict.NOT_ASSESSABLE
    assert not assessment.deferred


def test_the_shipped_registry_covers_dgl_and_torch_geometric() -> None:
    """Both graph libraries resolve to the graph intent from the real registry."""

    registry = load_environment_registry()
    dgl = find_covering_intents("dgl", registry, exclude="core")
    assert [item.intent for item in dgl] == ["graph"]
    # `dgl` is spelled identically in both namespaces, so the declared spec finds it.
    assert dgl[0].basis is CoverageBasis.DECLARED_DEPENDENCY
    pyg = find_covering_intents("torch_geometric", registry, exclude="core")
    assert [item.intent for item in pyg] == ["graph"]
    # `pyg` is the conda spelling of the `torch_geometric` distribution and is not
    # in the machine-owned provision registry, so the routing table is what proves
    # it -- and the basis says so rather than overstating the evidence.
    assert pyg[0].basis is CoverageBasis.ROUTING_TABLE


def test_a_locked_sibling_inventory_outranks_a_declared_dependency() -> None:
    """Coverage proved by a solved lock is reported as the stronger basis."""

    class _Lock:
        export_bytes = None

    class _Intent:
        def __init__(self, dependencies: tuple[str, ...]) -> None:
            self.dependencies = dependencies
            self.lock = _Lock()

    class _Registry:
        intents = {"graph": _Intent(("dgl",)), "audio": _Intent(("torchaudio",))}

    covering = find_covering_intents("dgl", _Registry(), exclude="core")
    assert [(item.intent, item.basis) for item in covering] == [
        ("graph", CoverageBasis.DECLARED_DEPENDENCY)
    ]


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("pytorch>=2.3", "pytorch"),
        ("python>=3.11,<3.13", "python"),
        ("conda-forge::segmentation-models-pytorch", "segmentation-models-pytorch"),
        ("pyg", "pyg"),
        ("  dgl  ", "dgl"),
        ("", ""),
    ],
)
def test_dependency_spec_names_are_parsed_off_their_bounds(spec: str, expected: str) -> None:
    """Coverage is a question about the NAME, not the version bound."""

    assert dependency_spec_name(spec) == expected


def test_an_uncovered_distribution_still_names_a_remedy() -> None:
    """A library no intent declares is filed with the action that would fix it."""

    implementation = {
        "recipe_type": "declarative-library",
        "library_recipe": {"distribution": "not-in-any-intent", "version": "1.0.0"},
    }
    assessment = assess_environment_coverage(
        implementation,
        routed_intent="core",
        routed_packages=list(CORE_PACKAGES),
        registry=load_environment_registry(),
    )
    assert assessment.deferred
    assert assessment.covering == ()
    row = build_coverage_deferral_row(
        stable_id="m9999",
        work_id="work-m9999",
        name="Unhomed",
        campaign_id="c1-mech",
        run_id="run-1",
        machine_id="mymini",
        created_at="2026-08-04T02:52:37.527339Z",
        assessment=assessment,
    )
    assert "declare distribution 'not-in-any-intent'" in row["recheck_hint"]


def _m538_row(*, run_id: str = "run-1", created_at: str = "2026-08-04T11:30:09.516753Z") -> Any:
    """Return a complete deferral row for the real m538 assessment."""

    return build_coverage_deferral_row(
        stable_id="m538",
        work_id="work-m538",
        name="TAGConv",
        campaign_id="c1-mech",
        run_id=run_id,
        machine_id="mymini",
        created_at=created_at,
        assessment=assess_environment_coverage(
            _m538_implementation(),
            routed_intent="core",
            routed_packages=list(CORE_PACKAGES),
            registry=load_environment_registry(),
        ),
    )


def test_deferral_row_names_the_distribution_and_the_covering_intent() -> None:
    """The record is actionable without the code that produced it."""

    row = _m538_row()
    assert row["disposition"] == COVERAGE_DEFERRAL_DISPOSITION
    assert row["coverage"]["library"]["distribution"] == "dgl"
    assert row["coverage"]["routed_intent"] == "core"
    assert [entry["intent"] for entry in row["coverage"]["covering_intents"]] == ["graph"]
    assert "'graph'" in row["recheck_hint"]
    assert row["row_sha256"].startswith("sha256:")


def test_only_an_unservable_assessment_may_be_recorded() -> None:
    """A served routing cannot be written to the deferral ledger."""

    served = assess_environment_coverage(
        _m538_implementation(),
        routed_intent="graph",
        routed_packages=list(GRAPH_PACKAGES),
        registry=None,
    )
    with pytest.raises(CoverageDeferralError):
        build_coverage_deferral_row(
            stable_id="m538",
            work_id="work-m538",
            name="TAGConv",
            campaign_id="c1-mech",
            run_id="run-1",
            machine_id="mymini",
            created_at="2026-08-04T11:30:09.516753Z",
            assessment=served,
        )


def test_deferral_ledger_appends_idempotently(tmp_path: Path) -> None:
    """Re-deriving the same deferral on a resume is a no-op, not a conflict."""

    path = coverage_deferral_path(tmp_path)
    row = _m538_row()
    assert append_coverage_deferral_row(path, row) == row
    assert append_coverage_deferral_row(path, row) == row
    loaded = load_coverage_deferral_rows([path])
    assert len(loaded) == 1
    assert loaded[0]["stable_id"] == "m538"
    assert loaded[0]["disposition"] == COVERAGE_DEFERRAL_DISPOSITION


def test_a_resume_keeps_the_first_record(tmp_path: Path) -> None:
    """``created_at`` and ``run_id`` differ on a resume; that is not a conflict."""

    path = coverage_deferral_path(tmp_path)
    first = append_coverage_deferral_row(path, _m538_row())
    resumed = _m538_row(run_id="run-2", created_at="2026-08-04T12:30:09.516753Z")
    assert append_coverage_deferral_row(path, resumed) == first
    assert len(load_coverage_deferral_rows([path])) == 1


def test_deferral_ledger_refuses_a_tampered_row(tmp_path: Path) -> None:
    """A row whose covering intent was edited after the fact does not load."""

    path = coverage_deferral_path(tmp_path)
    append_coverage_deferral_row(path, _m538_row())
    tampered = path.read_text(encoding="utf-8").replace('"graph"', '"core"')
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(CoverageDeferralError):
        load_coverage_deferral_rows([path])
