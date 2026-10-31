"""Precision of the lockstep check on spellings that only look stale.

Each cause the check resolves is tested both ways: the false-positive shape reports
nothing, and a real stale spelling of the same shape (the gone owner's member, used as
that owner) still reports. Synthetic inventories and repositories in ``tmp_path``; no
torch.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from name_lockstep import attribution, check, gate_input  # noqa: E402
from name_lockstep.diff import diff_inventories  # noqa: E402


def _row(row_id: str, kind: str, name: str, path: str, fingerprint: str, **extra) -> dict:
    """Build a minimal inventory row."""

    return {
        "id": row_id,
        "kind": kind,
        "name": name,
        "path": path,
        "fingerprint": fingerprint,
        **extra,
    }


def _member(module: str, owner: str, leaf: str, fingerprint: str) -> dict:
    """Build a member row defined in ``module``."""

    return _row(
        f"member:{module}:{owner}.{leaf}",
        "member",
        leaf,
        f"{owner}.{leaf}",
        fingerprint,
        owner_names=[owner],
        module=module,
    )


def _inventory(*rows: dict) -> dict[str, dict]:
    """Key rows on their ids."""

    return {row["id"]: row for row in rows}


def _corpus(root: Path, glossary: str, example: str) -> check.Corpus:
    """Write a repository whose only lockstep text is a glossary and one example."""

    for relative, text in {
        "docs/reference/glossary.md": glossary,
        "docs/agent-reference/surface.md": "Use `tl.trace(model, x)`.\n",
        "examples/demo.py": example,
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return check.Corpus.read(root, [Path("docs/reference/glossary.md")], snippets=True)


def _stale_locations(findings: list[check.Finding]) -> set[str]:
    """Return the ``path:line`` of every stale finding."""

    return {finding.location for finding in findings if finding.problem == "stale"}


TRACE = _row("root:tl.trace", "root", "trace", "tl.trace", "f-trace")
OP_TENSOR_PUBLIC = _member("tl.data_classes.op", "Op", "tensor", "f-1")
OP_TENSOR_PRIVATE = _member("tl._data_classes.op", "Op", "tensor", "f-1")


# --- cause 1: the defining module moved, the public spelling did not ------------------


def test_a_member_whose_module_moved_is_one_name_not_a_removal_and_an_addition(
    tmp_path: Path,
) -> None:
    """``Op.tensor`` defined in a private module is still ``Op.tensor``: no residue."""

    old, new = OP_TENSOR_PUBLIC, OP_TENSOR_PRIVATE
    base, head = _inventory(TRACE, old), _inventory(TRACE, new)
    surface_diff = diff_inventories(base, head)
    assert surface_diff.removed == () and surface_diff.new == ()
    assert [(m.old_id, m.new_id, m.basis) for m in surface_diff.moved] == [
        (old["id"], new["id"], "moved")
    ]
    assert surface_diff.to_json()["counts"] == {"new": 0, "removed": 0, "renamed": 0, "moved": 1}
    corpus = _corpus(tmp_path, glossary="`Op.tensor` is the saved output.\n", example="op.tensor\n")
    assert check.check_lockstep(surface_diff, corpus, head) == []
    # A moved name is not a new name: it is not sent to the legibility guess.
    assert gate_input.gate_items(surface_diff, head, corpus) == []


def test_a_moved_member_keeps_the_glossary_entry_obligation(tmp_path: Path) -> None:
    """The glossary must still have an entry for a name whose module moved."""

    base, head = _inventory(OP_TENSOR_PUBLIC), _inventory(OP_TENSOR_PRIVATE)
    corpus = _corpus(tmp_path, glossary="Nothing about it.\n", example="")
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert [(f.row_id, f.change, f.place, f.problem) for f in findings] == [
        (OP_TENSOR_PRIVATE["id"], "moved", "glossary", "missing")
    ]


def test_a_member_that_really_left_still_reports_even_beside_a_moved_sibling(
    tmp_path: Path,
) -> None:
    """Only the exact public spelling pairs; ``Op.grad`` with no head row is removed."""

    grad = _member("tl.data_classes.op", "Op", "grad", "f-2")
    base, head = _inventory(OP_TENSOR_PUBLIC, grad), _inventory(OP_TENSOR_PRIVATE)
    surface_diff = diff_inventories(base, head)
    assert [row["id"] for row in surface_diff.removed] == [grad["id"]]
    corpus = _corpus(tmp_path, glossary="`Op.tensor`\n", example="op.grad\nOp.grad\n")
    findings = check.check_lockstep(surface_diff, corpus, head)
    assert _stale_locations(findings) == {"examples/demo.py:1", "examples/demo.py:2"}


def test_two_owners_of_one_spelling_across_modules_do_not_pair_by_spelling() -> None:
    """An ambiguous same-spelling pair is left to the fingerprint; the other stays removed."""

    one = _member("pkg.one", "Registry", "info", "f-1")
    two = _member("pkg.two", "Registry", "info", "f-3")
    private = _member("pkg._one", "Registry", "info", "f-1")
    surface_diff = diff_inventories(_inventory(one, two), _inventory(private))
    assert [(m.old_id, m.new_id, m.basis) for m in surface_diff.moved] == [
        (one["id"], private["id"], "fingerprint")
    ]
    assert [row["id"] for row in surface_diff.removed] == [two["id"]]
    assert surface_diff.new == () and surface_diff.renamed == ()
    # With no fingerprint to decide, nothing pairs: both stay removed, the head row new.
    blind = diff_inventories(_inventory(one, two), _inventory({**private, "fingerprint": "f-9"}))
    assert blind.moved == () and len(blind.removed) == 2 and len(blind.new) == 1


# --- cause 2: the leaf is a live head name at that position ---------------------------


RECORD_ROOT = _row("root:tl.record", "root", "record", "tl.record", "f-rec")
REGISTER_EXPORT = _row(
    "export:tl.facets.register", "export", "register", "tl.facets.register", "f-reg"
)


def test_a_contextual_hit_owned_by_a_live_root_or_export_is_not_residue(tmp_path: Path) -> None:
    """``tl.record(...)`` is the live root, not the gone ``Resolution.record`` field."""

    base = _inventory(
        RECORD_ROOT,
        REGISTER_EXPORT,
        _member("tl.preprocessing", "Resolution", "record", "f-r"),
        _member("tl.autoroute", "Registry", "register", "f-g"),
    )
    head = _inventory(RECORD_ROOT, REGISTER_EXPORT)
    corpus = _corpus(
        tmp_path,
        glossary="`tl.record` records a pass. `tl.facets.register` adds a facet.\n",
        example="rec = tl.record(model, x)\nfacets.register(my_facet)\ntorchlens.facets.register(f)\n",
    )
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert _stale_locations(findings) == set()


def test_the_gone_member_used_as_its_owner_still_reports(tmp_path: Path) -> None:
    """``resolution.record`` and ``Resolution.record`` are the gone field."""

    base = {
        "root:tl.record": RECORD_ROOT,
        "r": _member("tl.preprocessing", "Resolution", "record", "f-r"),
    }
    head = {"root:tl.record": RECORD_ROOT}
    corpus = _corpus(
        tmp_path,
        glossary="`Resolution.record` holds provenance.\n",
        example="resolution.record\nrec = tl.record(model, x)\n",
    )
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert _stale_locations(findings) == {"docs/reference/glossary.md:1", "examples/demo.py:1"}


# --- cause 3: an attribute of an imported third-party object --------------------------


def test_an_attribute_of_an_imported_foreign_module_is_not_residue(tmp_path: Path) -> None:
    """``torch.tensor(`` after ``import torch`` is PyTorch, not the gone ``Op.tensor``."""

    base, head = _inventory(TRACE, OP_TENSOR_PUBLIC), _inventory(TRACE)
    corpus = _corpus(
        tmp_path,
        glossary="```python\nimport torch as _torch\nx = _torch.tensor([1.0])\n```\n",
        example="import torch\nfrom PIL import Image\nx = torch.tensor([1])\nImage.tensor\n",
    )
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert _stale_locations(findings) == set()


def test_the_same_leaf_on_a_variable_or_without_the_import_still_reports(tmp_path: Path) -> None:
    """``op.tensor`` is unresolved; ``torch.tensor`` with no import stays a finding too."""

    base, head = _inventory(TRACE, OP_TENSOR_PUBLIC), _inventory(TRACE)
    corpus = _corpus(
        tmp_path,
        glossary="torch.tensor(x) is not imported here.\n",
        example="import torch\nimport torchlens as tl\nop.tensor\ntl.tensor\n",
    )
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert _stale_locations(findings) == {
        "docs/reference/glossary.md:1",
        "examples/demo.py:3",
        "examples/demo.py:4",
    }


def test_imports_are_read_from_python_markdown_and_notebook_json() -> None:
    """Every file shape binds names the same way."""

    assert attribution.imports(
        "import torch\nimport numpy as np\nfrom PIL import Image, ImageDraw as D\n"
    ) == {
        "torch": "torch",
        "np": "numpy",
        "Image": "PIL",
        "D": "PIL",
    }
    notebook = '    "import torch\\n",\n    "from torchlens import trace\\n",\n'
    assert attribution.imports(notebook) == {"torch": "torch", "trace": "torchlens"}
    assert attribution.third_party("y = torch.nn.functional.log(x)", 24, {"torch": "torch"})
    assert not attribution.third_party("tl.record(x)", 3, {"tl": "torchlens"})
    assert not attribution.third_party("op.tensor", 3, {"torch": "torch"})


# --- cause 4: part of a URL -----------------------------------------------------------


def test_a_leaf_inside_a_url_is_not_residue_but_a_call_on_an_object_is(tmp_path: Path) -> None:
    """``rome.baulab.info`` is a host name; ``registry.info()`` is the gone method."""

    base = _inventory(TRACE, _member("tl.autoroute", "Registry", "info", "f-1"))
    head = _inventory(TRACE)
    corpus = _corpus(
        tmp_path,
        glossary="See https://rome.baulab.info/data for the dataset.\n",
        example="# https://rome.baulab.info/\nregistry.info()\n",
    )
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert _stale_locations(findings) == {"examples/demo.py:2"}


# --- cause 5: a keyword spelling that is an assignment target --------------------------


LOG_PARAM = _row("param:log", "param", "log", "log=", "f-log")


def test_an_assignment_target_is_not_a_gone_keyword(tmp_path: Path) -> None:
    """``log = ...`` opens a statement; nothing is passed."""

    base, head = _inventory(TRACE, LOG_PARAM), _inventory(TRACE)
    notebook = (
        '{"cells": [{"source": [\n    "import torch\\n",\n    "log = tl.trace(model)\\n"\n]}]}\n'
    )
    (tmp_path / "notebooks/audit").mkdir(parents=True)
    (tmp_path / "notebooks/audit/a.ipynb").write_text(notebook, encoding="utf-8")
    corpus = _corpus(
        tmp_path,
        glossary="Nothing.\n",
        example="log = tl.trace(model)\nif ok:\n    log = other\n",
    )
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert _stale_locations(findings) == set()


def test_a_keyword_argument_still_reports_on_one_line_split_or_in_prose(tmp_path: Path) -> None:
    """``log=True`` inside a call, on a continuation line, in a notebook or in backticks."""

    base, head = _inventory(TRACE, LOG_PARAM), _inventory(TRACE)
    notebook = (
        '{"cells": [{"source": [\n    "tl.trace(\\n",\n    "    log=True,\\n",\n    ")\\n"\n]}]}\n'
    )
    (tmp_path / "notebooks/audit").mkdir(parents=True)
    (tmp_path / "notebooks/audit/a.ipynb").write_text(notebook, encoding="utf-8")
    corpus = _corpus(
        tmp_path,
        glossary="Pass `log=True` to keep the log.\n",
        example="tl.trace(model, log=True)\ntl.trace(\n    model,\n    log=True,\n)\n",
    )
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert _stale_locations(findings) == {
        "docs/reference/glossary.md:1",
        "examples/demo.py:1",
        "examples/demo.py:4",
        "notebooks/audit/a.ipynb:3",
    }


# --- what stays: qualified spellings and unresolved shapes ---------------------------


@pytest.mark.parametrize(
    "example",
    ["Op.tensor\n", "import torch\nOp.tensor\n", "ops = [o.tensor for o in trace.ops]\n"],
)
def test_qualified_and_unresolved_shapes_keep_reporting(tmp_path: Path, example: str) -> None:
    """Attribution never removes a qualified hit or a hit it cannot explain."""

    base, head = _inventory(TRACE, OP_TENSOR_PUBLIC), _inventory(TRACE)
    corpus = _corpus(tmp_path, glossary="`Op.tensor`\n", example=example)
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    assert "docs/reference/glossary.md:1" in _stale_locations(findings)
    assert any(location.startswith("examples/demo.py:") for location in _stale_locations(findings))


# --- a renamed keyword reports its real old spelling -----------------------------------


def test_a_renamed_keyword_reports_the_old_spelling_with_its_equals_sign(tmp_path: Path) -> None:
    """``corrupt=`` renamed to ``corrupted_graph=`` is reported as ``corrupt=``, not ``ccorrupt``."""

    old = _row("param:corrupt", "param", "corrupt", "corrupt=", "f-c")
    new = _row("param:corrupted_graph", "param", "corrupted_graph", "corrupted_graph=", "f-c")
    base, head = _inventory(TRACE, old), _inventory(TRACE, new)
    corpus = _corpus(
        tmp_path,
        glossary="`corrupted_graph=` names the corrupted run.\n",
        example="tl.trace(model, corrupt=run)\n",
    )
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    stale = [f for f in findings if f.problem == "stale"]
    assert [(f.spelling, f.location) for f in stale] == [("corrupt=", "examples/demo.py:1")]
    assert check._old_path({"kind": "member", "path": "Trace.ops"}, "ops", "layer_list") == (
        "Trace.layer_list"
    )


# --- a moved member whose owner set shrank ---------------------------------------------


def test_a_moved_member_pairs_on_its_spelling_and_keeps_retired_owners_as_residue(
    tmp_path: Path,
) -> None:
    """``Accessor.head`` owned by ``Accessor`` alone: live; ``LayerAccessor.head``: stale."""

    old = _row(
        "member:tl.data_classes._accessor_base:Accessor.head",
        "member",
        "head",
        "Accessor.head",
        "f-h",
        owner_names=["LayerAccessor", "ModuleAccessor"],
    )
    new = _row(
        "member:tl._data_classes._accessor_base:Accessor.head",
        "member",
        "head",
        "Accessor.head",
        "f-h",
        owner_names=["Accessor"],
    )
    surface_diff = diff_inventories(_inventory(TRACE, old), _inventory(TRACE, new))
    assert [(m.basis, m.retired_owners) for m in surface_diff.moved] == [
        ("moved", ("LayerAccessor", "ModuleAccessor"))
    ]
    corpus = _corpus(
        tmp_path,
        glossary="`Accessor.head` returns the first items.\n",
        example="trace.layers.head(3)\nLayerAccessor.head\naccessor.head\n",
    )
    findings = check.check_lockstep(surface_diff, corpus, _inventory(TRACE, new))
    assert [(f.change, f.location) for f in findings if f.problem == "stale"] == [
        ("moved", "examples/demo.py:2")
    ]
