"""Behavior of the name lockstep tooling (``tools/name_lockstep``).

The inventory, diff, lockstep check, codemod and gate input run on small synthetic
packages and repositories in ``tmp_path``, so every test is fast and torch-free; one
slow test walks the real package.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from name_lockstep import check, codemod, docsections, gate_input  # noqa: E402
from name_lockstep.cli import main  # noqa: E402
from name_lockstep.diff import diff_inventories, proposals_to_map, renamed_id  # noqa: E402
from name_lockstep.inventory import build_inventory, load_inventory  # noqa: E402

FAKE_INIT = '''
"""Fake package."""
from fakelens_pkg.core import Probe, run_probe

__all__ = ["Probe", "run_probe"]
_LAZY_ATTRS = {}
'''
FAKE_CORE = '''
"""Core."""
import dataclasses


@dataclasses.dataclass
class Probe:
    """A probe."""

    depth: int = 0

    @property
    def label(self) -> str:
        """The label."""
        return "p"

    def fire(self, strength: float = 1.0) -> float:
        """Fire the probe."""
        return strength


def run_probe(probe: Probe, *, budget: int = 3) -> int:
    """Run a probe."""
    return budget
'''
FAKE_NS = '''
"""Namespace."""
from fakelens_pkg.core import run_probe as _run

__all__ = ["helper"]


def helper(value: int) -> int:
    """Help."""
    return _run(value)
'''


def _write(root: Path, files: dict[str, str]) -> None:
    """Write dedented files under ``root``."""

    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


@pytest.fixture()
def fake_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Inventory a synthetic package with a root class, a function and a namespace."""

    _write(
        tmp_path,
        {
            "fakelens_pkg/__init__.py": FAKE_INIT,
            "fakelens_pkg/core.py": FAKE_CORE,
            "fakelens_pkg/ns.py": FAKE_NS,
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith("fakelens_pkg")]:
        monkeypatch.delitem(sys.modules, name)
    return {row["id"]: row for row in build_inventory("fakelens_pkg")["rows"]}


def test_inventory_rows_use_the_stable_id_scheme(fake_inventory: dict) -> None:
    """Root, export, member and param rows carry the naming-sweep ids."""

    ids = set(fake_inventory)
    assert {"root:tl.Probe", "root:tl.run_probe", "root:tl.core", "root:tl.ns"} <= ids
    assert "export:tl.ns.helper" in ids
    assert {
        "member:fakelens_pkg.core:Probe.depth",
        "member:fakelens_pkg.core:Probe.label",
        "member:fakelens_pkg.core:Probe.fire",
    } <= ids
    assert fake_inventory["member:fakelens_pkg.core:Probe.label"]["subkind"] == "property"
    assert fake_inventory["param:budget"]["uses"] == ["tl.core.run_probe", "tl.run_probe"]
    assert "Probe.fire" in fake_inventory["param:strength"]["uses"]
    assert "param:self" not in ids


def _row(row_id: str, kind: str, name: str, path: str, fingerprint: str = "", **extra) -> dict:
    """Build a minimal inventory row."""

    return {
        "id": row_id,
        "kind": kind,
        "name": name,
        "path": path,
        "fingerprint": fingerprint,
        **extra,
    }


BASE = {
    "root:tl.trace": _row("root:tl.trace", "root", "trace", "tl.trace", "f-trace"),
    "export:tl.io.old_save": _row(
        "export:tl.io.old_save", "export", "old_save", "tl.io.old_save", "f-save"
    ),
    "member:m:Trace.layer_list": _row(
        "member:m:Trace.layer_list",
        "member",
        "layer_list",
        "Trace.layer_list",
        "f-list",
        owner_names=["Trace"],
    ),
    "export:tl.io.gone": _row("export:tl.io.gone", "export", "gone", "tl.io.gone", "f-dup"),
    "export:tl.io.gone2": _row("export:tl.io.gone2", "export", "gone2", "tl.io.gone2", "f-dup"),
}
HEAD = {
    "root:tl.trace": BASE["root:tl.trace"],
    "export:tl.io.dump": _row("export:tl.io.dump", "export", "dump", "tl.io.dump", "f-save"),
    "member:m:Trace.ops": _row(
        "member:m:Trace.ops", "member", "ops", "Trace.ops", "f-other", owner_names=["Trace"]
    ),
    "export:tl.io.fresh": _row("export:tl.io.fresh", "export", "fresh", "tl.io.fresh", "f-dup"),
}


def test_diff_lists_new_removed_and_renamed() -> None:
    """Fingerprint pairs are unique on both sides; declared renames win."""

    plain = diff_inventories(BASE, HEAD)
    assert [(r.old_id, r.new_id, r.basis) for r in plain.renamed] == [
        ("export:tl.io.old_save", "export:tl.io.dump", "fingerprint")
    ]
    # Two removed rows share the fresh row's fingerprint, so neither pairs.
    assert {row["id"] for row in plain.new} == {"member:m:Trace.ops", "export:tl.io.fresh"}
    assert {row["id"] for row in plain.removed} == {
        "member:m:Trace.layer_list",
        "export:tl.io.gone",
        "export:tl.io.gone2",
    }
    declared = diff_inventories(BASE, HEAD, {"member:m:Trace.layer_list": "ops"})
    assert ("member:m:Trace.layer_list", "member:m:Trace.ops", "declared") in [
        (r.old_id, r.new_id, r.basis) for r in declared.renamed
    ]
    assert declared.to_json()["counts"] == {"new": 1, "removed": 2, "renamed": 2}


def test_renamed_id_and_proposal_map() -> None:
    """Respelled ids keep their owner; params regroup under the new name."""

    assert renamed_id(BASE["member:m:Trace.layer_list"], "ops") == "member:m:Trace.ops"
    assert renamed_id({"id": "param:dim", "kind": "param", "name": "dim"}, "axis") == "param:axis"
    rows = {
        "a": {"id": "a", "name": "x", "proposed": "y"},
        "b": {"id": "b", "name": "z", "proposed": "z"},
    }
    assert proposals_to_map(rows) == {"a": "y"}


def _lockstep_repo(root: Path, glossary: str, example: str) -> check.Corpus:
    """Write a minimal repository with the four lockstep places."""

    _write(
        root,
        {
            "docs/reference/glossary.md": glossary,
            "docs/agent-reference/surface.md": "Use `tl.trace(model, x)` and `tl.io.dump`.\n",
            "notebooks/audit/a.py": "trace = tl.trace(model, x)\n",
            "examples/demo.py": example,
            "tests/test_demo.py": "def test_x():\n    trace.ops\n    tl.io.dump(trace, 'p')\n",
        },
    )
    return check.Corpus.read(root, [Path("docs/reference/glossary.md")], snippets=True)


def test_check_reports_missing_glossary_and_stale_spellings(tmp_path: Path) -> None:
    """New names need a glossary entry; gone spellings must leave every place."""

    corpus = _lockstep_repo(
        tmp_path,
        glossary="**Trace**\n: The record. `tl.io.old_save` writes it.\n",
        example="trace.layer_list\ntl.io.gone(trace)\n",
    )
    surface_diff = diff_inventories(BASE, HEAD, {"member:m:Trace.layer_list": "ops"})
    findings = check.check_lockstep(surface_diff, corpus, HEAD)
    rendered = {(f.row_id, f.place, f.problem, f.location) for f in findings}
    assert ("export:tl.io.dump", "glossary", "missing", "glossary") in rendered
    assert ("export:tl.io.dump", "glossary", "stale", "docs/reference/glossary.md:2") in rendered
    assert ("member:m:Trace.ops", "examples", "stale", "examples/demo.py:1") in rendered
    assert ("export:tl.io.gone", "examples", "stale", "examples/demo.py:2") in rendered
    assert ("export:tl.io.fresh", "agent-reference", "missing", "agent-reference") in rendered
    assert all(f.render() for f in findings)


def test_check_passes_a_complete_change(tmp_path: Path) -> None:
    """A rename with every place updated yields no findings."""

    corpus = _lockstep_repo(
        tmp_path,
        glossary="`tl.io.dump` writes a trace. `Trace.ops` lists operations.\n",
        example="trace.ops\n",
    )
    head = {key: HEAD[key] for key in ("root:tl.trace", "export:tl.io.dump", "member:m:Trace.ops")}
    base = {key: BASE[key] for key in ("root:tl.trace", "export:tl.io.old_save")}
    base["member:m:Trace.layer_list"] = BASE["member:m:Trace.layer_list"]
    surface_diff = diff_inventories(base, head, {"member:m:Trace.layer_list": "ops"})
    assert check.check_lockstep(surface_diff, corpus, head) == []


def test_contextual_residue_skips_a_leaf_another_row_keeps(tmp_path: Path) -> None:
    """``.name`` is not stale while another head member still spells it."""

    corpus = _lockstep_repo(tmp_path, glossary="`Layer.ops`\n", example="layer.ops\n")
    base = {"member:m:Trace.ops": HEAD["member:m:Trace.ops"]}
    head = {
        "member:m:Layer.ops": _row(
            "member:m:Layer.ops", "member", "ops", "Layer.ops", "f-l", owner_names=["Layer"]
        )
    }
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert [f.place for f in findings if f.problem == "stale"] == []


def test_gate_items_carry_snippet_and_reference(tmp_path: Path) -> None:
    """Each new or renamed name gets a call site and the glossary meaning."""

    corpus = _lockstep_repo(
        tmp_path, glossary="`tl.io.dump` writes a trace to disk.\n", example="tl.io.dump(t, 'p')\n"
    )
    items = gate_input.gate_items(diff_inventories(BASE, HEAD), HEAD, corpus)
    dump = next(item for item in items if item["id"] == "export:tl.io.dump")
    assert dump["change"] == "renamed" and dump["old_id"] == "export:tl.io.old_save"
    assert dump["reference_source"] == "glossary:docs/reference/glossary.md:1"
    assert "tl.io.dump" in dump["snippet"]
    paths = {s["path"] for s in dump["snippets"]}
    assert paths == {"examples/demo.py", "tests/test_demo.py", "docs/agent-reference/surface.md"}


DOCUMENTED_CORE = '''
"""Core with numpydoc-documented fields and keywords."""
import dataclasses


@dataclasses.dataclass
class Base:
    """A base.

    Attributes
    ----------
    origin:
        Where the base came from.
    """

    origin: str = ""


@dataclasses.dataclass
class Curve(Base):
    """A curve and its area.

    Attributes
    ----------
    area : float
        Area under the curve, by the
        trapezoid rule.

        A second paragraph that is not part of the meaning.
    steps, points
        One record per evaluated fraction.
    bare: int
    """

    area: float = 0.0
    steps: tuple = ()
    points: tuple = ()
    bare: int = 0

    @property
    def width(self) -> float:
        return 1.0


def make_curve(*, area: float = 0.0, budget: int = 1) -> Curve:
    """Make a curve.

    Parameters
    ----------
    area : float
        Area to start from.
    budget : int
        How many evaluations to spend.

    Returns
    -------
    Curve
        The curve.
    """

    return Curve(area=area)


def plain(*, budget: int = 1) -> int:
    """Take a budget, documented elsewhere."""

    return budget
'''


@pytest.fixture()
def documented_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Inventory a synthetic package whose fields and keywords are documented in sections."""

    _write(
        tmp_path,
        {
            "doclens_pkg/__init__.py": '''
            """Fake package."""
            from doclens_pkg.core import Curve, make_curve, plain

            __all__ = ["Curve", "make_curve", "plain"]
            ''',
            "doclens_pkg/core.py": DOCUMENTED_CORE,
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith("doclens_pkg")]:
        monkeypatch.delitem(sys.modules, name)
    return {row["id"]: row for row in build_inventory("doclens_pkg")["rows"]}


def test_numpydoc_sections_give_the_first_paragraph() -> None:
    """Entries with and without types, shared entries and empty entries parse."""

    doc = textwrap.dedent(DOCUMENTED_CORE.split("class Curve(Base):\n    ")[1].split('"""')[1])
    entries = docsections._entries(docsections._sections(doc)["Attributes"])
    assert entries["area"] == "Area under the curve, by the trapezoid rule."
    assert entries["steps"] == entries["points"] == "One record per evaluated fraction."
    assert "bare" not in entries
    assert docsections._clip("word " * 100).endswith(" ...")
    assert len(docsections._clip("word " * 100)) <= docsections.MAX_MEANING + 4


def test_fields_and_keywords_get_their_documented_meaning(documented_inventory: dict) -> None:
    """A field reads its owner's Attributes; a keyword reads the Parameters of its callables."""

    rows = documented_inventory
    area = rows["member:doclens_pkg.core:Curve.area"]
    assert area["doc_first_line"] == "Area under the curve, by the trapezoid rule."
    assert area["doc_source"] == "Curve Attributes"
    origin = rows["member:doclens_pkg.core:Base.origin"]
    assert (origin["doc_first_line"], origin["doc_source"]) == (
        "Where the base came from.",
        "Base Attributes",
    )
    assert rows["member:doclens_pkg.core:Curve.bare"]["doc_first_line"] == ""
    assert rows["member:doclens_pkg.core:Curve.width"]["doc_first_line"] == ""
    budget = rows["param:budget"]
    assert budget["doc_first_line"] == "How many evaluations to spend."
    assert [d["use"] for d in budget["param_docs"]] == ["tl.core.make_curve", "tl.make_curve"]
    area_kw = rows["param:area"]
    assert "Area to start from." in area_kw["doc_first_line"]
    assert "Area under the curve" in area_kw["doc_first_line"]
    assert area_kw["doc_first_line"].count(" | ") == 1
    assert area_kw["doc_source"] == "tl.Curve Attributes"


def test_section_meanings_stay_out_of_the_fingerprint(documented_inventory: dict) -> None:
    """Two fields of one class still share a fingerprint, as before meanings were read."""

    rows = documented_inventory
    area = rows["member:doclens_pkg.core:Curve.area"]["fingerprint"]
    assert rows["member:doclens_pkg.core:Curve.steps"]["fingerprint"] == area


def test_gate_items_use_documented_meaning_over_a_leaf_index_row(
    tmp_path: Path, documented_inventory: dict
) -> None:
    """A bare-word index row loses to the field's documented meaning; snippets stay in context."""

    corpus = _lockstep_repo(
        tmp_path,
        glossary=(
            "| Unrelated rows | `area`, `steps` | unstable |\n`tl.make_curve` makes a curve.\n"
        ),
        example="other.area = 3\n" + "\n" * 20 + "c = tl.make_curve()\nc.area\n",
    )
    head = documented_inventory
    base = {
        k: v
        for k, v in head.items()
        if not k.startswith(("member:doclens_pkg.core:Curve.", "param:area"))
    }
    items = {i["id"]: i for i in gate_input.gate_items(diff_inventories(base, head), head, corpus)}
    area = items["member:doclens_pkg.core:Curve.area"]
    assert area["reference"] == "Area under the curve, by the trapezoid rule."
    assert area["reference_source"] == "doc:Curve Attributes"
    assert area["doc_first_line"] == area["reference"]
    assert area["doc_source"] == "Curve Attributes"
    bare = items["member:doclens_pkg.core:Curve.bare"]
    assert bare["reference_source"] == "none" and bare["doc_first_line"] is None
    assert items["param:area"]["reference_source"] == "doc:tl.Curve Attributes"
    assert gate_input._context_terms(head["param:area"]) == ["Curve", "make_curve"]
    made_by = gate_input.producers(head)
    assert made_by["Curve"] == {"make_curve"}
    area_row = head["member:doclens_pkg.core:Curve.area"]
    assert gate_input._context_terms(area_row, made_by) == ["Curve", "make_curve"]
    assert area["snippet"].splitlines()[-1] == "c.area"


def test_reference_takes_definition_text_and_skips_spelling_index_rows(tmp_path: Path) -> None:
    """A term entry gives its definition; an index row loses to the docstring; a row id wins."""

    glossary = (
        "| Dump surface | `dump`, `fresh` | unstable |\n"
        "\n"
        "**tl.io.dump**\n"
        ": Writes a trace to disk,\n"
        "  one file per trace.\n"
        "\n"
        "| `fresh` | was `stale` | makes a fresh thing |\n"
    )
    corpus = _lockstep_repo(tmp_path, glossary=glossary, example="\n")
    dump = _row("export:tl.io.dump", "export", "dump", "tl.io.dump", doc_first_line="Dump it.")
    assert gate_input._reference(dump, corpus) == (
        "Writes a trace to disk, one file per trace.",
        "glossary:docs/reference/glossary.md:3",
    )
    fresh = _row("export:tl.io.fresh", "export", "fresh", "tl.io.fresh", doc_first_line="Fresh.")
    text, source = gate_input._reference(fresh, corpus)
    assert (text, source) == ("| `fresh` | was `stale` | makes a fresh thing |", source)
    assert source.endswith(":7")
    only_index = _lockstep_repo(
        tmp_path / "b", glossary="| S | `fresh`, `x` | unstable |\n", example="\n"
    )
    assert gate_input._reference(fresh, only_index) == ("Fresh.", "docstring")
    no_doc = dict(fresh, doc_first_line="")
    assert gate_input._reference(no_doc, only_index)[1] == "glossary:docs/reference/glossary.md:1"
    by_id = _lockstep_repo(
        tmp_path / "c",
        glossary="**fresh**\n: Another fresh.\n\n- `tl.io.fresh` (export; id `export:tl.io.fresh`; guess pass): The fresh one.\n",
        example="\n",
    )
    assert gate_input._reference(fresh, by_id) == (
        "The fresh one.",
        "glossary:docs/reference/glossary.md:4",
    )


def test_snippet_prefers_a_hit_near_the_owner() -> None:
    """A contextual hit beside the owner's name outranks an earlier unrelated one."""

    lines = ["spec.area"] + [""] * 30 + ["c = Curve()", "c.area"]
    hits = [(1, 5, 9, "contextual"), (33, 2, 6, "contextual")]
    assert gate_input._best_hit(hits, lines, ["Curve"]) == (1, 33)
    assert gate_input._best_hit(hits, lines, []) == (2, 1)
    assert gate_input._best_hit([(1, 0, 4, "qualified")] + hits, lines, ["Curve"]) == (0, 1)


CODEMOD_PKG = '''
"""Pkg."""

__all__ = ["run_probe", "Probe"]


class Probe:
    """A probe."""

    layer_list: list

    def summary(self) -> int:
        """Summarise."""
        return len(self.layer_list)


def run_probe(probe, budget=3):
    """Run."""
    return budget + 1


def other(run_probe):
    """Shadows the name locally."""
    return run_probe
'''


def _codemod_repo(root: Path) -> list[dict]:
    """Write a repository with a package, tests, docs and a changelog."""

    _write(
        root,
        {
            "torchlens/__init__.py": CODEMOD_PKG,
            "tests/test_a.py": (
                "import torchlens as tl\nfrom torchlens import run_probe\n\n"
                "def test_a(p):\n    tl.run_probe(p, budget=2)\n    run_probe(p)\n"
                "    p.layer_list\n    Probe.layer_list\n"
            ),
            "docs/guide.md": "Call `tl.run_probe(p, budget=1)`; read `Probe.layer_list`.\n",
            "CHANGELOG.md": "Added tl.run_probe.\n",
        },
    )
    return [
        {
            "id": "root:tl.run_probe",
            "kind": "root",
            "name": "run_probe",
            "path": "tl.run_probe",
            "module": "torchlens",
            "proposed": "probe_run",
        },
        {
            "id": "member:torchlens:Probe.layer_list",
            "kind": "member",
            "name": "layer_list",
            "path": "Probe.layer_list",
            "owner": "torchlens:Probe",
            "owner_names": ["Probe"],
            "module": "torchlens",
            "proposed": "ops",
        },
        {
            "id": "param:budget",
            "kind": "param",
            "name": "budget",
            "path": "budget=",
            "uses": ["tl.run_probe"],
            "proposed": "limit",
        },
        {
            "id": "root:tl.Probe",
            "kind": "root",
            "name": "Probe",
            "path": "tl.Probe",
            "proposed": None,
        },
    ]


def test_codemod_dry_run_writes_nothing_and_classifies(tmp_path: Path) -> None:
    """Planning classifies edits and leaves every file untouched."""

    rows = _codemod_repo(tmp_path)
    before = {p: p.read_text() for p in tmp_path.rglob("*") if p.is_file()}
    edits = codemod.plan(rows, tmp_path, roots=("torchlens", "tests", "docs", "CHANGELOG.md"))
    assert {p: p.read_text() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert not any(edit.path == "CHANGELOG.md" for edit in edits)
    by_site = {(e.path, e.line, e.old): e.certainty for e in edits}
    assert by_site[("torchlens/__init__.py", 3, "run_probe")] == codemod.DEFINITE
    assert by_site[("tests/test_a.py", 2, "run_probe")] == codemod.DEFINITE
    assert by_site[("tests/test_a.py", 7, "layer_list")] == codemod.AMBIGUOUS
    assert by_site[("tests/test_a.py", 8, "layer_list")] == codemod.DEFINITE
    assert by_site[("tests/test_a.py", 5, "budget")] == codemod.AMBIGUOUS
    assert by_site[("torchlens/__init__.py", 16, "budget")] == codemod.DEFINITE
    # The local parameter that shadows the root name in ``other`` is left alone.
    assert ("torchlens/__init__.py", 21, "run_probe") not in by_site
    assert ("torchlens/__init__.py", 23, "run_probe") not in by_site


def test_codemod_apply_writes_definite_edits_only(tmp_path: Path) -> None:
    """Apply respells definite sites and keeps ambiguous ones unless asked."""

    rows = _codemod_repo(tmp_path)
    edits = codemod.plan(rows, tmp_path, roots=("torchlens", "tests", "docs"))
    written = codemod.apply(edits, tmp_path)
    assert written == sum(edit.certainty == codemod.DEFINITE for edit in edits)
    package = (tmp_path / "torchlens/__init__.py").read_text()
    assert '__all__ = ["probe_run", "Probe"]' in package
    assert "def probe_run(probe, limit=3):" in package and "return limit + 1" in package
    assert "def other(run_probe):" in package
    assert "len(self.ops)" in package and "    ops: list" in package
    test_text = (tmp_path / "tests/test_a.py").read_text()
    assert "from torchlens import probe_run" in test_text and "    probe_run(p)" in test_text
    assert "tl.probe_run(p, budget=2)" in test_text and "p.layer_list" in test_text
    assert "Probe.ops" in test_text
    assert "`tl.probe_run(p, budget=1)`" in (tmp_path / "docs/guide.md").read_text()


def test_codemod_refuses_bad_proposals_and_stale_plans(tmp_path: Path) -> None:
    """Non-identifier proposals and files edited after planning are refused."""

    rows = _codemod_repo(tmp_path)
    with pytest.raises(ValueError, match="not an identifier"):
        codemod.proposals([{**rows[0], "proposed": "probe-run"}])
    edits = codemod.plan(rows[:1], tmp_path, roots=("docs",))
    (tmp_path / "docs/guide.md").write_text("rewritten\n")
    with pytest.raises(ValueError, match="changed since planning"):
        codemod.apply(edits, tmp_path)


def test_cli_diff_check_and_codemod(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command line runs each subcommand and exits 1 on lockstep findings."""

    _lockstep_repo(tmp_path, glossary="nothing\n", example="\n")
    base_file, head_file = tmp_path / "base.json", tmp_path / "head.json"
    base_file.write_text(json.dumps({"rows": list(BASE.values())}))
    head_file.write_text(json.dumps({"rows": list(HEAD.values())}))
    common = ["--repo", str(tmp_path)]
    assert main([*common, "diff", "--base", str(base_file), "--head", str(head_file)]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["renamed"] == 1
    assert main([*common, "check", "--base", str(base_file), "--head", str(head_file)]) == 1
    assert "lockstep:" in capsys.readouterr().out
    out = tmp_path / "gate.json"
    args = [*common, "gate-input", "--base", str(base_file), "--head", str(head_file)]
    assert main([*args, "--out", str(out)]) == 0
    assert len(json.loads(out.read_text())) == 3
    rows = _codemod_repo(tmp_path)
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps(rows))
    assert main([*common, "codemod", "--proposals", str(proposals)]) == 0
    assert "dry run: nothing written" in capsys.readouterr().out
    assert load_inventory(proposals)["param:budget"]["proposed"] == "limit"


def test_load_inventory_rejects_rows_without_ids(tmp_path: Path) -> None:
    """Rows need id, kind and name; a foreign inventory schema is refused."""

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"name": "x"}]))
    with pytest.raises(ValueError, match="lacks"):
        load_inventory(bad)
    bad.write_text(json.dumps({"schema": "name-lockstep-inventory/0", "rows": []}))
    with pytest.raises(ValueError, match="schema"):
        load_inventory(bad)


@pytest.mark.slow
def test_real_package_inventory_has_the_core_names() -> None:
    """The walk over the real package reaches the core nouns and their members."""

    rows = {row["id"]: row for row in build_inventory()["rows"]}
    assert {"root:tl.trace", "root:tl.Trace", "param:layers_to_save"} <= set(rows)
    trace_members = [
        r for r in rows.values() if r["kind"] == "member" and "Trace" in r["owner_names"]
    ]
    assert len(trace_members) > 100
