"""Regression tests for the rename codemod (``tools/name_lockstep/codemod.py``).

Each test reproduces one defect the phase-4 dry run found, on a tiny synthetic
repository in ``tmp_path``: relative imports, assignments taken for keywords,
agreeing overlaps, atomic apply, non-identifier proposals, field-table string keys,
the row id in the dry-run output and per-file parse reuse.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from name_lockstep import codemod  # noqa: E402
from name_lockstep.cli import main  # noqa: E402


def _repo(root: Path, files: dict[str, str]) -> Path:
    """Write ``files`` under ``root`` and return it."""

    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _sites(edits: list[codemod.Edit]) -> set[tuple[str, int, int, str]]:
    """Return ``(path, line, 1-based column, certainty)`` per edit."""

    return {(e.path, e.line, e.start + 1, e.certainty) for e in edits}


def test_relative_import_and_reexport_all_are_renamed(tmp_path: Path) -> None:
    """F1: a relative from-import and the re-exporting ``__all__`` follow the definition."""

    root = _repo(
        tmp_path,
        {
            "torchlens/__init__.py": "",
            "torchlens/errors/__init__.py": (
                "from .episode import OldError\n\n__all__ = ['OldError']\n"
            ),
            "torchlens/errors/episode.py": "class OldError(Exception):\n    pass\n",
            "torchlens/errors/deep/__init__.py": "from ..episode import OldError\n",
        },
    )
    row = {
        "id": "export:tl.errors.OldError",
        "kind": "export",
        "name": "OldError",
        "proposed": "NewError",
        "path": "tl.errors.OldError",
        "module": "torchlens.errors.episode",
    }
    edits = codemod.plan([row], root, roots=("torchlens",))
    assert _sites(edits) == {
        ("torchlens/errors/episode.py", 1, 7, codemod.DEFINITE),
        ("torchlens/errors/__init__.py", 1, 22, codemod.DEFINITE),
        ("torchlens/errors/__init__.py", 3, 13, codemod.DEFINITE),
        ("torchlens/errors/deep/__init__.py", 1, 23, codemod.DEFINITE),
    }
    codemod.apply(edits, root)
    assert (root / "torchlens/errors/__init__.py").read_text() == (
        "from .episode import NewError\n\n__all__ = ['NewError']\n"
    )


def test_relative_import_outside_the_package_stays_ambiguous(tmp_path: Path) -> None:
    """F1: a relative import that resolves outside ``torchlens`` is not definite."""

    root = _repo(
        tmp_path,
        {
            "torchlens/__init__.py": "class OldError(Exception):\n    pass\n",
            "tests/__init__.py": "",
            "tests/helpers.py": "from .fixtures import OldError\n",
        },
    )
    row = {
        "id": "root:tl.OldError",
        "kind": "root",
        "name": "OldError",
        "proposed": "NewError",
        "path": "tl.OldError",
        "module": "torchlens",
    }
    edits = codemod.plan([row], root, roots=("torchlens", "tests"))
    assert not [e for e in edits if e.path == "tests/helpers.py"]


def test_keyword_rows_skip_assignment_statements(tmp_path: Path) -> None:
    """F2: ``x = x + 1`` is not a keyword; call keywords and signature defaults are."""

    root = _repo(
        tmp_path,
        {
            "torchlens/__init__.py": "",
            "tests/test_m.py": (
                "def forward(x):\n"
                "    x = x + 1\n"
                "    return f(x=x)\n"
                "\n"
                "def helper(y, x=2):\n"
                "    total = dict(x =3)\n"
                "    return total\n"
            ),
            "docs/page.md": "Call `f(x=1)`.\n\n```python\nx = 3\nf(\n    x=x,\n)\n```\n",
        },
    )
    row = {"id": "param:x", "kind": "param", "name": "x", "proposed": "inputs", "uses": []}
    edits = codemod.plan([row], root, roots=("tests", "docs"))
    assert _sites(edits) == {
        ("tests/test_m.py", 3, 14, codemod.AMBIGUOUS),
        ("tests/test_m.py", 5, 15, codemod.AMBIGUOUS),
        ("tests/test_m.py", 6, 18, codemod.AMBIGUOUS),
        ("docs/page.md", 1, 9, codemod.AMBIGUOUS),
        ("docs/page.md", 6, 5, codemod.AMBIGUOUS),
    }


def test_keyword_rows_still_rename_inside_listed_callables(tmp_path: Path) -> None:
    """F2: inside a callable named in ``uses``, the parameter and its body uses are definite."""

    root = _repo(
        tmp_path,
        {"torchlens/__init__.py": "def run(x):\n    x = x + 1\n    return x\n"},
    )
    row = {"id": "param:x", "kind": "param", "name": "x", "proposed": "inputs", "uses": ["tl.run"]}
    edits = codemod.plan([row], root, roots=("torchlens",))
    assert {(e.line, e.certainty) for e in edits} == {
        (1, codemod.DEFINITE),
        (2, codemod.DEFINITE),
        (3, codemod.DEFINITE),
    }
    assert len(edits) == 4


def _overlap_repo(root: Path) -> list[dict]:
    """Write the two-rows-one-leaf repository and return its rows."""

    _repo(
        root,
        {
            "torchlens/__init__.py": "",
            "torchlens/layer.py": "class Layer:\n    def __init__(self):\n        self.parents = []\n",
            "torchlens/op.py": "class Op:\n    def __init__(self):\n        self.parents = []\n",
            "tests/a_first.py": "import torchlens as tl\n\ndef f(tr):\n    return tl.old_fn(tr)\n",
            "tests/b_second.py": "def g(node):\n    return node.parents\n",
        },
    )
    member = {"kind": "member", "name": "parents", "proposed": "parent_labels"}
    return [
        {
            **member,
            "id": "member:torchlens.layer:Layer.parents",
            "path": "Layer.parents",
            "module": "torchlens.layer",
            "owner": "torchlens.layer:Layer",
        },
        {
            **member,
            "id": "member:torchlens.op:Op.parents",
            "path": "Op.parents",
            "module": "torchlens.op",
            "owner": "torchlens.op:Op",
        },
        {
            "id": "root:tl.old_fn",
            "kind": "root",
            "name": "old_fn",
            "proposed": "new_fn",
            "path": "tl.old_fn",
            "module": "torchlens.funcs",
        },
    ]


def test_agreeing_overlap_applies_once(tmp_path: Path) -> None:
    """F3: two rows that respell one position the same way write it once."""

    rows = _overlap_repo(tmp_path)
    edits = codemod.plan(rows, tmp_path, roots=("tests",))
    shared = [e for e in edits if e.path == "tests/b_second.py"]
    assert len(shared) == 2 and len({e.row_id for e in shared}) == 2
    written = codemod.apply(edits, tmp_path, include_ambiguous=True)
    assert written == 2
    assert (tmp_path / "tests/b_second.py").read_text().endswith("node.parent_labels\n")
    assert (tmp_path / "tests/a_first.py").read_text().endswith("tl.new_fn(tr)\n")


def test_disagreeing_overlap_refuses_and_leaves_the_tree_untouched(tmp_path: Path) -> None:
    """F3 and F4: rows that disagree refuse the apply, and no file is written."""

    rows = _overlap_repo(tmp_path)
    rows[1] = {**rows[1], "proposed": "inputs"}
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    edits = codemod.plan(rows, tmp_path, roots=("tests",))
    with pytest.raises(ValueError, match="different spellings") as refused:
        codemod.apply(edits, tmp_path, include_ambiguous=True)
    assert "member:torchlens.op:Op.parents" in str(refused.value)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_stale_file_refusal_writes_nothing(tmp_path: Path) -> None:
    """F4: a file changed after planning refuses the run before any file is replaced."""

    rows = _overlap_repo(tmp_path)
    edits = codemod.plan(rows, tmp_path, roots=("tests",))
    (tmp_path / "tests/b_second.py").write_text("def g(node):\n    return node.other\n")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="changed since planning"):
        codemod.apply(edits, tmp_path, include_ambiguous=True)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_apply_keeps_crlf_line_endings_and_file_mode(tmp_path: Path) -> None:
    """F4: the temp-file replace keeps the original bytes apart from the edit."""

    root = _repo(tmp_path, {"torchlens/__init__.py": ""})
    script = root / "tests" / "run.py"
    script.parent.mkdir()
    script.write_bytes(b"import torchlens as tl\r\ntl.old_fn()\r\n")
    script.chmod(0o755)
    rows = _overlap_repo(tmp_path / "unused")[2:]
    edits = codemod.plan(rows, root, roots=("tests",))
    assert codemod.apply(edits, root) == 1
    assert script.read_bytes() == b"import torchlens as tl\r\ntl.new_fn()\r\n"
    assert script.stat().st_mode & 0o777 == 0o755
    assert not [p for p in script.parent.iterdir() if p.name != "run.py"]


def test_non_identifier_proposal_is_skipped_per_row(tmp_path: Path) -> None:
    """F5: a dotted proposal is reported and skipped; the other rows still plan."""

    rows = _overlap_repo(tmp_path)
    moved = {**rows[0], "proposed": "Layer.graph.parents"}
    chosen = codemod.proposals([moved, rows[2]])
    assert [row["id"] for row in chosen] == ["root:tl.old_fn"]
    assert codemod.skipped_proposals([moved, rows[2]]) == [
        {
            "id": moved["id"],
            "proposed": "Layer.graph.parents",
            "reason": "not an identifier (a move or restructure; out of the codemod's scope)",
        }
    ]
    edits = codemod.plan([moved, rows[2]], tmp_path, roots=("tests",))
    assert {e.row_id for e in edits} == {"root:tl.old_fn"}


def test_field_table_string_keys_follow_their_field(tmp_path: Path) -> None:
    """F6: the field's entry in its declared field table and its class tables is renamed."""

    root = _repo(
        tmp_path,
        {
            "torchlens/__init__.py": "",
            "torchlens/constants.py": "TRACE_FIELDS = ('backward_durations', 'label')\n",
            "torchlens/trace.py": (
                "_DEFAULTS = {'backward_durations': [], 'label': ''}\n"
                "\n"
                "class Trace:\n"
                "    SPEC = {'backward_durations': 'keep'}\n"
                "\n"
                "    def total(self):\n"
                "        return sum(self.backward_durations)\n"
                "\n"
                "    def other(self):\n"
                "        return {'backward_durations': 1}\n"
            ),
            "torchlens/other.py": "NAMES = ('backward_durations',)\n",
        },
    )
    row = {
        "id": "member:torchlens.trace:Trace.backward_durations",
        "kind": "member",
        "name": "backward_durations",
        "proposed": "backward_times",
        "path": "Trace.backward_durations",
        "module": "torchlens.trace",
        "owner": "torchlens.trace:Trace",
        "field_tables": ["torchlens.constants:TRACE_FIELDS"],
    }
    edits = codemod.plan([row], root, roots=("torchlens",))
    assert _sites(edits) == {
        ("torchlens/constants.py", 1, 18, codemod.DEFINITE),
        ("torchlens/trace.py", 1, 15, codemod.AMBIGUOUS),
        ("torchlens/trace.py", 4, 14, codemod.DEFINITE),
        ("torchlens/trace.py", 7, 25, codemod.DEFINITE),
    }
    codemod.apply(edits, root)
    assert (root / "torchlens/constants.py").read_text() == (
        "TRACE_FIELDS = ('backward_times', 'label')\n"
    )
    assert "NAMES = ('backward_durations',)" in (root / "torchlens/other.py").read_text()


def test_class_table_lookups_follow_the_renamed_key(tmp_path: Path) -> None:
    """F6: string lookups on a class table whose key is renamed follow it; other tables' stay."""

    root = _repo(
        tmp_path,
        {
            "torchlens/__init__.py": "",
            "torchlens/call.py": (
                "OTHER = {'label': 0}\n"
                "\n"
                "class Call:\n"
                "    SPEC = {'label': 'keep'}\n"
                "    ALIASES = {'call_label': SPEC['label']}\n"
                "\n"
                "    def policy(self):\n"
                "        return self.SPEC.get('label'), OTHER['label']\n"
                "\n"
                "\n"
                "KEEP = Call.SPEC['label']\n"
            ),
        },
    )
    row = {
        "id": "member:torchlens.call:Call.label",
        "kind": "member",
        "name": "label",
        "proposed": "call_name",
        "path": "Call.label",
        "module": "torchlens.call",
        "owner": "torchlens.call:Call",
    }
    edits = codemod.plan([row], root, roots=("torchlens",))
    definite = {(line, col) for _, line, col, how in _sites(edits) if how == codemod.DEFINITE}
    assert definite == {(4, 14), (5, 36), (8, 31), (11, 19)}
    codemod.apply(edits, root)
    text = (root / "torchlens/call.py").read_text()
    assert "ALIASES = {'call_label': SPEC['call_name']}" in text
    assert "self.SPEC.get('call_name'), OTHER['label']" in text
    assert "OTHER = {'label': 0}" in text
    assert "KEEP = Call.SPEC['call_name']" in text


def test_dry_run_output_carries_the_row_id_and_skips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """F8: every dry-run edit line names its row; skipped rows are listed."""

    rows = _overlap_repo(tmp_path)
    rows[0] = {**rows[0], "proposed": "Layer.graph.parents"}
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps(rows))
    assert main(["--repo", str(tmp_path), "codemod", "--proposals", str(proposals)]) == 0
    out = capsys.readouterr().out
    assert "definite  tests/a_first.py:4:15 old_fn -> new_fn [root:tl.old_fn]" in out
    assert "ambiguous tests/b_second.py:2:17 parents -> parent_labels" in out
    assert "[member:torchlens.op:Op.parents]" in out
    assert "skipped   member:torchlens.layer:Layer.parents: 'Layer.graph.parents'" in out
    assert "skipped rows: 1" in out


def test_plan_parses_each_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F9: planning many rows reads and parses each Python file once."""

    rows = _overlap_repo(tmp_path)
    calls: list[str] = []
    real_parse = ast.parse

    def counting_parse(source, *args, **kwargs):
        calls.append(source)
        return real_parse(source, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", counting_parse)
    extra = [{**rows[2], "id": f"root:tl.old_fn{i}", "proposed": f"new_fn{i}"} for i in range(5)]
    codemod.plan([*rows, *extra], tmp_path, roots=("torchlens", "tests"))
    python_files = list(tmp_path.rglob("*.py"))
    assert len(calls) <= len(python_files)
