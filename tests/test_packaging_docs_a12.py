"""Packaging + crash-class-docs gates (workstream A12).

Four defect families this lane closed, each pinned here so it cannot silently
reopen:

* extras composition -- ``[all]``/``[all-stretch]`` resolve by exclusion
  (lit-nlp named out), no empty extras, the tf/tensorflow rollup alias, and
  the transformers band-coherence rule between the ``hf`` and ``test`` extras;
* README-on-PyPI -- every image reference absolute (relative paths rendered
  22/22 broken on PyPI) and the post-trace ``release_model`` remedy mentioned;
* SECURITY.md -- exists, documents the transformers 4.x advisory set, and
  stays in lockstep with every ``--ignore-vuln`` waiver in the workflows (a
  new waiver cannot land undocumented);
* removed-spelling crash classes in executable docs -- ``tl.sites``,
  ``train_mode=``, ``vis_mode=`` on ``trace()``, the deleted
  ``notebooks/total_audit/_shared.py`` helper import, and the other spellings
  that killed 22 example notebooks at their first capture cell.

Also validates the two seeded migration ledgers
(``docs/migration/_claims.json`` + ``_migration_ledger.json``) that lanes
D01/F30 consume.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py3.10 fallback used by the repo's own tooling
    tomllib = None

REPO_ROOT = Path(__file__).parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
README = REPO_ROOT / "README.md"
SECURITY = REPO_ROOT / "SECURITY.md"
CLAIMS = REPO_ROOT / "docs" / "migration" / "_claims.json"
LEDGER = REPO_ROOT / "docs" / "migration" / "_migration_ledger.json"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _load_pyproject() -> dict:
    if tomllib is not None:
        with PYPROJECT.open("rb") as fh:
            return tomllib.load(fh)
    import tomli  # pragma: no cover

    with PYPROJECT.open("rb") as fh:  # pragma: no cover
        return tomli.load(fh)


def _extras() -> dict[str, list[str]]:
    return _load_pyproject()["project"]["optional-dependencies"]


@pytest.mark.smoke
def test_extras_composition_repairs_hold() -> None:
    """The A12 extras repairs cannot silently regress."""

    extras = _extras()

    # fix-by-exclusion: both meta-extras exist, alias each other, and lit-nlp
    # (the shap<0.46 conflict) stays OUT of the rollup.
    assert extras["all"] == ["torchlens[all-stretch]"]
    assert not any("lit-nlp" in req for req in extras["all-stretch"]), (
        "lit-nlp re-entered all-stretch: lit-nlp 1.3 requires shap<0.46 against "
        "the shap~=0.46 pin, which made [all] known-unsatisfiable through 2.34.1"
    )

    # the lit extra stays bounded to the 1.3 line (LIT-panel memo item 1).
    assert extras["lit"] == ["lit-nlp>=1.3,<1.4"]

    # tensorflow aliases tf through the rollup idiom (they were byte-identical
    # literal twins before, free to drift).
    assert extras["tensorflow"] == ["torchlens[tf]"]

    # the tl.export.tensorboard exporter has a declared install path.
    assert "tensorboard" in extras
    assert any(req.startswith("tensorboard") for req in extras["tensorboard"])

    # the empty-extra class is closed: the no-op [profiler] marker is gone and
    # nothing else may ship an extra that installs nothing.
    assert "profiler" not in extras
    empty = sorted(name for name, reqs in extras.items() if not reqs)
    assert not empty, f"empty extras install nothing and market nothing: {empty}"

    # neuro memo D17 (landed by D05, 2026-09-02): [neuro] is rsatoolbox-only and the
    # Brain-Score seam is its own [brainscore] extra (brainscore_vision 2.3 pins
    # scikit-learn<1.6, under which rsatoolbox 0.3 cannot import); brainscore_core
    # is imported nowhere and arrives transitively.
    assert not any(req.startswith("brainscore") for req in extras["neuro"])
    assert any(req.startswith("brainscore_vision") for req in extras["brainscore"])
    assert not any(req.startswith("brainscore_core") for req in extras["brainscore"])


@pytest.mark.smoke
def test_transformers_band_coherence() -> None:
    """The hf extra and test extra carry the SAME transformers band.

    Band-coherence rule (pyproject r7 R83-2): widen an adapter band and the
    test extra in the same change, never one side alone. The A12 in-repo widen
    (megaplan FORK-2) moved both to >=4.45,<6; this pins the EQUALITY so a
    future widen cannot fork them again, and checks the band actually admits
    both the validated 4.x floor and the 5.x line CI pins.
    """

    from packaging.requirements import Requirement

    extras = _extras()
    bands = {}
    for extra in ("hf", "test"):
        reqs = [Requirement(r) for r in extras[extra]]
        (transformers_req,) = [r for r in reqs if r.name == "transformers"]
        bands[extra] = transformers_req.specifier
    assert str(bands["hf"]) == str(bands["test"]), (
        f"transformers band fork: hf={bands['hf']} test={bands['test']} -- "
        "widen both in the same change, never one side alone"
    )
    assert bands["hf"].contains("4.45.0"), "the validated 4.x floor left the band"
    assert bands["hf"].contains("5.14.1"), "the CI-pinned 5.x line left the band"
    assert not bands["hf"].contains("6.0.0"), "the band must stay bounded below 6"


@pytest.mark.smoke
def test_requires_python_floor() -> None:
    """requires-python stays at the real (slots=True) 3.10 floor."""

    assert _load_pyproject()["project"]["requires-python"] == ">=3.10"


@pytest.mark.smoke
def test_readme_renders_on_pypi() -> None:
    """Every README image reference is absolute (PyPI cannot resolve relative
    paths -- all 22 rendered broken there) and the post-trace pickling remedy
    is taught."""

    text = README.read_text(encoding="utf-8")
    for src in re.findall(r'<img\s+[^>]*src="([^"]+)"', text):
        assert src.startswith(("https://", "http://")), f"relative README image: {src}"
    for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if target.split("#")[0].lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".gif")):
            assert target.startswith(("https://", "http://")), (
                f"relative README markdown image: {target}"
            )
    assert "release_model" in text, (
        "README must mention tl.release_model (post-trace PicklingError remedy; "
        "listA row 29 doc-now portion)"
    )


@pytest.mark.smoke
def test_security_md_documents_the_waiver_surface() -> None:
    """SECURITY.md exists, names the transformers 4.x advisory set, and stays
    in lockstep with every --ignore-vuln waiver in the workflows."""

    assert SECURITY.exists(), "SECURITY.md deleted"
    text = SECURITY.read_text(encoding="utf-8")
    for pysec in ("PYSEC-2025-217", "PYSEC-2026-2288", "PYSEC-2026-2289", "PYSEC-2026-2290"):
        assert pysec in text, f"SECURITY.md lost the transformers advisory row {pysec}"

    waived = set()
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        # Advisory-shaped IDs only: workflow comments legitimately say things
        # like "the --ignore-vuln flags" in prose.
        waived.update(
            re.findall(
                r"--ignore-vuln\s+((?:PYSEC|GHSA|CVE)-[A-Za-z0-9-]+)",
                wf.read_text(encoding="utf-8"),
            )
        )
    undocumented = sorted(vuln for vuln in waived if vuln not in text)
    assert not undocumented, (
        f"pip-audit waivers not documented in SECURITY.md: {undocumented} -- a "
        "waiver carries its exposure analysis or it does not land"
    )


@pytest.mark.smoke
def test_claims_registry_seed_is_well_formed() -> None:
    """docs/migration/_claims.json: D6 schema, verbatim-template registry."""

    doc = json.loads(CLAIMS.read_text(encoding="utf-8"))
    assert doc["schema"] == "torchlens.migration_claims.v1"
    assert doc["wording_states"] == ["full", "narrowed_scope", "limitation"]
    assert len(doc["claims"]) == 5
    assert len(doc["cross_cutting_rules"]) == 4
    assert len(doc["tl_side_gates"]) == 3

    row_keys = {
        "id",
        "competitor",
        "neutral_fact",
        "allowed_public_wording",
        "narrowed_scope_wording",
        "limitation_wording",
        "evidence_refs",
        "tl_side_gates",
        "last_verified",
        "owner",
    }
    gate_ids = set(doc["tl_side_gates"])
    seen = set()
    for row in doc["claims"] + doc["lints"]:
        missing = row_keys - set(row)
        assert not missing, f"claim/lint row {row.get('id')} missing keys: {sorted(missing)}"
        assert row["id"] not in seen, f"duplicate row id {row['id']}"
        seen.add(row["id"])
    # claim-row gate references resolve to registered gates (lint rows may
    # reference lane-merge gates outside the registered three).
    for row in doc["claims"]:
        for gate in row["tl_side_gates"]:
            assert gate in gate_ids, f"{row['id']} references unregistered gate {gate}"
    for gate, spec in doc["tl_side_gates"].items():
        assert spec["status"] in {"open", "closed"}, (gate, spec["status"])


@pytest.mark.smoke
def test_migration_ledger_seed_is_well_formed() -> None:
    """docs/migration/_migration_ledger.json: the D01/F30 work queue parses."""

    doc = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert doc["schema"] == "torchlens.migration_ledger.v1"
    kinds = {"undersell-falsehood", "missing-row", "structural", "harness", "current"}
    statuses = {"open", "fixed", "current"}
    assert doc["rows"], "empty ledger"
    for row in doc["rows"]:
        assert set(row) >= {"id", "doc", "kind", "receipt", "status", "verified_at", "owner_next"}
        assert row["kind"] in kinds, (row["id"], row["kind"])
        assert row["status"] in statuses, (row["id"], row["status"])


# ---------------------------------------------------------------------------
# Removed-spelling crash classes in executable docs.
# ---------------------------------------------------------------------------

#: Spellings that KILLED example notebooks at their first capture cell
#: (walkthrough A-VI item 31 + this lane's execution sweep). Scanned over CODE
#: CELLS and example scripts only, so prose that documents a removal (e.g.
#: "tl.sites was removed; use find_sites") stays legal, and
#: docs/migration/v2.0_api_changes.md's deliberate old-spelling column is out
#: of scope by construction.
_REMOVED_SPELLING_PATTERNS: dict[str, re.Pattern[str]] = {
    # tl.sites was removed; trace.find_sites is the spelling.
    "tl.sites(...) call": re.compile(r"\btl\.sites\("),
    # trace() lost its flat kwargs; vis_mode never existed there.
    "vis_mode= inside a trace() call": re.compile(r"\btrace\([^)]*\bvis_mode=", re.DOTALL),
    # the real capture spelling is CaptureOptions(backward_ready=True).
    "train_mode= kwarg": re.compile(r"\btrain_mode="),
    # canonical spellings live under tl.options / tl.errors.
    "tl.StreamingOptions": re.compile(r"\btl\.StreamingOptions\b"),
    "tl.TrainingModeConfigError": re.compile(r"\btl\.TrainingModeConfigError\b"),
    # typo'd kwarg the training tutorial taught (real: detach_saved_activations).
    "detach_saved_tensorss=": re.compile(r"detach_saved_tensorss"),
    # helper deleted in 9a324000; six 5min notebooks + one recipe died on it.
    "deleted total_audit/_shared.py import": re.compile(r"total_audit"),
    # do() lost flat confirm_mutation; InterventionOptions carries it now.
    "flat do(confirm_mutation=)": re.compile(r"\.do\([^)]*\bconfirm_mutation=", re.DOTALL),
}


def _executable_doc_sources() -> list[tuple[str, str]]:
    """(label, source) for every notebook code cell and example script."""

    sources: list[tuple[str, str]] = []
    for nb_path in sorted((REPO_ROOT / "examples").rglob("*.ipynb")) + sorted(
        (REPO_ROOT / "notebooks").rglob("*.ipynb")
    ):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code":
                rel = nb_path.relative_to(REPO_ROOT)
                sources.append((f"{rel}::cell{i}", "".join(cell.get("source", []))))
    for py_path in sorted((REPO_ROOT / "examples").rglob("*.py")):
        rel = py_path.relative_to(REPO_ROOT)
        sources.append((str(rel), py_path.read_text(encoding="utf-8")))
    return sources


def test_executable_docs_teach_no_removed_spellings() -> None:
    """No notebook code cell or example script uses a removed spelling.

    This is the tripwire for the crash class where 22 example-notebook call
    sites passed vis_mode= (and friends) to trace() and every affected
    notebook died at its FIRST capture cell.
    """

    offenders: list[str] = []
    for label, src in _executable_doc_sources():
        for name, pattern in _REMOVED_SPELLING_PATTERNS.items():
            if pattern.search(src):
                offenders.append(f"{label}: {name}")
    assert not offenders, "removed spellings back in executable docs:\n" + "\n".join(offenders)
