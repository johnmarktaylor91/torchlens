"""The name lockstep inventory sees every public spelling kind (``tools/name_lockstep``).

The gate once passed a branch that added an option field, four refusal codes, a result
record key and two protocol dunders, because codes, record keys and dunders were not
inventoried and a bare backticked leaf anywhere in the glossary counted as documenting a
member. A synthetic torch-free package reproduces each kind; the checks here pin that
each is a row, that its documentation is found only where it means this row, and that the
guess-test input skips what has no meaning to guess.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from name_lockstep import check, gate_input, spelling, static_surface  # noqa: E402
from name_lockstep.diff import diff_inventories  # noqa: E402
from name_lockstep.inventory import PROTOCOL_DUNDERS, build_inventory  # noqa: E402

PKG = "kindlens_pkg"
FILES = {
    "__init__.py": '''
        """Fake package."""
        from kindlens_pkg.core import Options, Trace, run

        __all__ = ["Options", "Trace", "run"]
        ''',
    "core.py": '''
        """Core objects."""
        import dataclasses


        @dataclasses.dataclass
        class Options:
            """Options."""

            strict: bool = False
            differentiable: bool = False


        class Trace:
            """The record."""

            def __init__(self):
                self.last_run = {"engine": "eager", "unfired_plan_ids": []}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.last_run["duration_s"] = 0.0
                return False

            def __repr__(self):
                return "Trace()"

            def _private(self):
                return None

            def finish(self):
                """Finish."""
                self.last_run.update({"finished": True})
                self.last_run.setdefault("seed", None)
                last_run = {"local_only": 1}
                return last_run


        def _refuse(problem, *, code: str = "input_invalid", **context):
            raise ValueError(problem, code, context)


        def run(model, *, differentiable=False):
            """Run."""
            if model is None:
                raise ValueError(code="model_missing")
            code = "recipe_audit_only"
            if differentiable:
                code = "predicate_type_invalid"
            return Trace()
        ''',
}


def _write_package(root: Path, files: dict[str, str] = FILES) -> Path:
    """Write the fake package under ``root`` and return its directory."""

    package = root / PKG
    package.mkdir(parents=True, exist_ok=True)
    for relative, text in files.items():
        (package / relative).write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return package


@pytest.fixture()
def inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Inventory the fake package, keyed on id."""

    _write_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith(PKG)]:
        monkeypatch.delitem(sys.modules, name)
    return {row["id"]: row for row in build_inventory(PKG)["rows"]}


def test_protocol_dunders_are_members_and_other_dunders_are_not(inventory: dict) -> None:
    """``__enter__``/``__exit__`` are rows; ``__repr__``, ``__init__`` and ``_private`` are not."""

    assert f"member:{PKG}.core:Trace.__enter__" in inventory
    assert f"member:{PKG}.core:Trace.__exit__" in inventory
    assert f"member:{PKG}.core:Trace.__repr__" not in inventory
    assert f"member:{PKG}.core:Trace.__init__" not in inventory
    assert f"member:{PKG}.core:Trace._private" not in inventory
    assert {"__enter__", "__exit__", "__iter__", "__len__", "__getitem__", "__call__"} <= set(
        PROTOCOL_DUNDERS
    )
    # A dunder's parameters are not public keywords.
    assert "param:exc" not in inventory


def test_option_fields_are_member_rows(inventory: dict) -> None:
    """Dataclass fields of a public options class are rows."""

    assert inventory[f"member:{PKG}.core:Options.differentiable"]["subkind"] == "field"


def test_refusal_codes_are_rows_from_every_declaration_form(inventory: dict) -> None:
    """Call keyword, plain and annotated assignment, and parameter default each declare a code."""

    codes = {row["name"] for row in inventory.values() if row["kind"] == "code"}
    assert codes == {
        "model_missing",
        "recipe_audit_only",
        "predicate_type_invalid",
        "input_invalid",
    }
    row = inventory["code:recipe_audit_only"]
    assert row["path"] == "recipe_audit_only" and row["modules"] == [f"{PKG}.core"]


def test_record_keys_are_rows_bounded_to_the_record_attribute(inventory: dict) -> None:
    """Dict display, subscript store, ``update`` and ``setdefault`` keys are rows; other dicts are not."""

    keys = {row["name"] for row in inventory.values() if row["kind"] == "receipt"}
    assert keys == {"engine", "unfired_plan_ids", "duration_s", "finished", "seed", "local_only"}
    row = inventory["receipt:last_run.unfired_plan_ids"]
    assert row["owner"] == "last_run" and row["path"] == "last_run.unfired_plan_ids"


def test_static_surface_skips_unparseable_files(tmp_path: Path) -> None:
    """A file with a syntax error is skipped, not fatal."""

    package = _write_package(tmp_path)
    (package / "broken.py").write_text("def (:\n", encoding="utf-8")
    assert "code:model_missing" in static_surface.code_rows(package, PKG)


def _corpus(root: Path, glossary: str) -> check.Corpus:
    """Write a minimal lockstep repository whose glossary is ``glossary``."""

    files = {
        "docs/reference/glossary.md": glossary,
        "docs/internal/agent-reference/surface.md": "Use `tl.run(model)`.\n",
        "notebooks/audit/a.py": "trace = tl.run(model)\n",
        "examples/demo.py": "trace = tl.run(model)\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return check.Corpus.read(root, [Path("docs/reference/glossary.md")], snippets=True)


def _missing_in_glossary(inventory: dict, corpus: check.Corpus) -> set[str]:
    """Return the ids of the new rows (everything, against an empty base) lacking a glossary entry."""

    findings = check.check_lockstep(diff_inventories({}, inventory), corpus, inventory)
    return {f.row_id for f in findings if f.place == "glossary" and f.problem == "missing"}


def test_member_leaf_counts_only_beside_its_owner(inventory: dict, tmp_path: Path) -> None:
    """A bare `differentiable` under another entry does not document ``Options.differentiable``."""

    corpus = _corpus(
        tmp_path,
        """
        ## Keywords

        | `differentiable=` | keyword of `run` |

        ## Codes

        `model_missing`, `recipe_audit_only`, `predicate_type_invalid`, `input_invalid`
        """,
    )
    missing = _missing_in_glossary(inventory, corpus)
    assert f"member:{PKG}.core:Options.differentiable" in missing
    assert not any(row_id.startswith("code:") for row_id in missing)

    corpus = _corpus(
        tmp_path,
        """
        ## Options

        **differentiable** (`Options`)
        : Keep the graph.
        """,
    )
    assert f"member:{PKG}.core:Options.differentiable" not in _missing_in_glossary(
        inventory, corpus
    )

    corpus = _corpus(tmp_path, "## Options\n\n**differentiable**\n: Keep the graph.\n")
    assert f"member:{PKG}.core:Options.differentiable" not in _missing_in_glossary(
        inventory, corpus
    )


def test_dunders_and_record_keys_need_an_owned_spelling(inventory: dict, tmp_path: Path) -> None:
    """``Trace.__enter__`` and ``last_run["engine"]`` count; a stray ``engine`` does not."""

    corpus = _corpus(
        tmp_path,
        """
        The `engine` option picks a backend. A trace is a context manager (`__enter__`).
        """,
    )
    missing = _missing_in_glossary(inventory, corpus)
    assert "receipt:last_run.engine" in missing
    assert f"member:{PKG}.core:Trace.__enter__" in missing

    corpus = _corpus(
        tmp_path,
        """
        **Trace**
        : `Trace.__enter__` and `Trace.__exit__` make it a context manager.
        `trace.last_run["engine"]` names the backend; `last_run` also carries `unfired_plan_ids`.
        """,
    )
    missing = _missing_in_glossary(inventory, corpus)
    assert "receipt:last_run.engine" not in missing
    assert "receipt:last_run.unfired_plan_ids" not in missing
    assert f"member:{PKG}.core:Trace.__enter__" not in missing
    assert f"member:{PKG}.core:Trace.__exit__" not in missing


def test_removed_code_spelled_in_docs_is_stale(inventory: dict, tmp_path: Path) -> None:
    """A code that left the package but is still quoted in the glossary is residue."""

    corpus = _corpus(tmp_path, 'Refused with `code="gone_code"`.\n')
    base = {
        **inventory,
        "code:gone_code": {
            **inventory["code:model_missing"],
            "id": "code:gone_code",
            "name": "gone_code",
            "path": "gone_code",
        },
    }
    findings = check.check_lockstep(diff_inventories(base, inventory), corpus, inventory)
    assert ("code:gone_code", "stale") in {(f.row_id, f.problem) for f in findings}


def test_code_spellings_are_quoted_or_backticked_only() -> None:
    """A refusal code is qualified when quoted; its bare leaf counts only when emphasised."""

    patterns = spelling.spellings({"kind": "code", "name": "model_missing"})
    strengths = (spelling.QUALIFIED, spelling.CONTEXTUAL, spelling.LEAF)
    assert spelling.find('code="model_missing"', patterns, strengths, prose=True)
    assert spelling.find("the `model_missing` code", patterns, strengths, prose=True)
    assert not spelling.find("the model_missing code", patterns, strengths, prose=True)


def test_gate_input_skips_dunders_and_documents_codes_from_the_contract(
    inventory: dict, tmp_path: Path
) -> None:
    """Protocol dunders are not guess items; a code item's meaning comes from the error contract."""

    corpus = _corpus(tmp_path, "nothing\n")
    contract = tmp_path / static_surface.CONTRACT_DOC
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(
        "| code | meaning | remedy |\n|---|---|---|\n"
        "| `model_missing` | No model was given. | Pass a model. |\n",
        encoding="utf-8",
    )
    items = gate_input.gate_items(diff_inventories({}, inventory), inventory, corpus)
    by_id = {item["id"]: item for item in items}
    assert f"member:{PKG}.core:Trace.__enter__" not in by_id
    assert f"member:{PKG}.core:Options.differentiable" in by_id
    assert by_id["code:model_missing"]["reference"] == "No model was given."
    assert by_id["code:model_missing"]["kind"].startswith("error code, refusal code")
    assert by_id["receipt:last_run.engine"]["name"] == "last_run.engine"
    json.dumps(items)  # every item is serialisable
