"""Regression tests for the lockstep inventory's coverage (``tools/name_lockstep/inventory.py``).

The phase-4 dry run found names the inventory missed: stored fields declared only in a
``FIELD_POLICY`` table, classes filed under a re-export shim, module constants of a
namespace without ``__all__``, nested namespaces, a submodule shadowed by a root name,
and variadic or underscore keywords. A synthetic package in ``tmp_path`` reproduces
each; one slow test checks the real package.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from name_lockstep import check, codemod, gate_input  # noqa: E402
from name_lockstep.diff import diff_inventories  # noqa: E402
from name_lockstep.inventory import build_inventory  # noqa: E402

PKG = "fieldlens_pkg"
FILES = {
    "__init__.py": '''
        """Fake package."""
        from fieldlens_pkg._state_shim import State
        from fieldlens_pkg.bundle import Bundle
        from fieldlens_pkg.caps import Caps, FastCaps
        from fieldlens_pkg.record import Accessor, Record

        __all__ = ["Accessor", "Bundle", "Caps", "FastCaps", "Record", "State", "bundle"]


        def bundle(*items):
            """Bundle items (shadows the submodule of the same name)."""
            return Bundle()
        ''',
    "constants.py": '''
        """Constants."""
        import os

        RECORD_FIELD_ORDER = ["depth", "label"]
        RAW_SUFFIX = "_raw"


        def helper():
            """Help."""
            return os.sep
        ''',
    "caps.py": '''
        """Capability dataclasses."""
        import dataclasses


        @dataclasses.dataclass
        class Caps:
            """Capabilities."""

            replay: bool = False
            cache: dict = dataclasses.field(default_factory=dict, init=False)


        @dataclasses.dataclass
        class FastCaps(Caps):
            """Fast capabilities."""

            fast: bool = True
        ''',
    "_impl.py": '''
        """Where Accessor is written."""


        class Accessor:
            """An accessor."""

            def total(self) -> int:
                """Count."""
                return 0


        Accessor.__module__ = "fieldlens_pkg.record"
        ''',
    "record.py": '''
        """Records."""
        from fieldlens_pkg._impl import Accessor
        from fieldlens_pkg.constants import RECORD_FIELD_ORDER


        class _Policy:
            def __init__(self, user_facing: bool) -> None:
                self.user_facing = user_facing


        class Record:
            """A record."""

            FIELD_POLICY = {
                **{name: _Policy(True) for name in RECORD_FIELD_ORDER},
                "runtime_cache": _Policy(False),
            }

            def __init__(self, depth=0, *items, _trace=None, **extra_kwargs):
                self.depth = depth
                self.label = ""
        ''',
    "_vocab.py": '''
        """Where State is written."""
        import enum


        class State(enum.Enum):
            """A state."""

            READY = 1
            DONE = 2


        State.__module__ = "fieldlens_pkg._state_shim"
        ''',
    "_state_shim.py": '''
        """Re-export shim."""
        from ._vocab import State

        __all__ = ["State"]
        ''',
    "bundle.py": '''
        """Bundle submodule."""


        class Bundle:
            """A bundle."""
        ''',
    "experimental/__init__.py": '''
        """Experimental."""
        from . import inner

        __all__ = ["inner"]
        ''',
    "experimental/inner/__init__.py": '''
        """Inner namespace."""

        __all__ = ["draw"]


        def draw():
            """Draw."""
            return None
        ''',
}


@pytest.fixture()
def field_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Inventory the synthetic package."""

    for relative, text in FILES.items():
        path = tmp_path / PKG / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith(PKG)]:
        monkeypatch.delitem(sys.modules, name)
    return {row["id"]: row for row in build_inventory(PKG)["rows"]}


def test_policy_table_fields_are_members_with_their_field_tables(field_inventory: dict) -> None:
    """F7 and F6: ``FIELD_POLICY`` fields are members and name the table listing them."""

    for name in ("depth", "label"):
        row = field_inventory[f"member:{PKG}.record:Record.{name}"]
        assert row["subkind"] == "field"
        assert row["field_tables"] == [f"{PKG}.constants:RECORD_FIELD_ORDER"]
    assert f"member:{PKG}.record:Record.runtime_cache" not in field_inventory


def test_classes_are_filed_under_their_defining_module(field_inventory: dict) -> None:
    """F7: a class whose ``__module__`` names a shim is filed where its source is."""

    assert {f"member:{PKG}._vocab:State.READY", f"member:{PKG}._vocab:State.DONE"} <= set(
        field_inventory
    )
    assert field_inventory["root:tl.State"]["module"] == f"{PKG}._vocab"
    assert f"member:{PKG}._impl:Accessor.total" in field_inventory
    assert field_inventory["root:tl.Accessor"]["module"] == f"{PKG}._impl"
    assert not [row_id for row_id in field_inventory if "_state_shim:" in row_id]


def test_constants_nested_namespaces_and_shadowed_submodules_are_exports(
    field_inventory: dict,
) -> None:
    """F7: constants, nested namespace exports and a shadowed submodule's exports appear."""

    assert {
        "export:tl.constants.RECORD_FIELD_ORDER",
        "export:tl.constants.RAW_SUFFIX",
        "export:tl.constants.helper",
        "export:tl.experimental.inner",
        "export:tl.experimental.inner.draw",
        "export:tl.bundle.Bundle",
    } <= set(field_inventory)
    assert "export:tl.constants.os" not in field_inventory
    assert field_inventory["root:tl.bundle"]["subkind"] == "function"


def test_variadic_and_underscore_keywords_are_rows(field_inventory: dict) -> None:
    """F7: ``*items``, ``**extra_kwargs`` and ``_trace=`` are keyword rows."""

    for name in ("items", "extra_kwargs", "_trace", "depth"):
        assert "tl.Record" in field_inventory[f"param:{name}"]["uses"]
    assert "param:self" not in field_inventory


def test_dataclass_fields_name_the_classes_taking_them_as_keywords(field_inventory: dict) -> None:
    """A dataclass field row lists every reached class whose constructor takes it."""

    assert field_inventory[f"member:{PKG}.caps:Caps.replay"]["init_classes"] == [
        "Caps",
        "FastCaps",
    ]
    assert field_inventory[f"member:{PKG}.caps:FastCaps.fast"]["init_classes"] == ["FastCaps"]
    assert "init_classes" not in field_inventory.get(f"member:{PKG}.caps:Caps.cache", {})
    assert "init_classes" not in field_inventory[f"member:{PKG}.record:Record.depth"]


def test_inventory_field_tables_drive_the_codemod(field_inventory: dict, tmp_path: Path) -> None:
    """F6: renaming a policy field respells its entry in the declared field-order table."""

    row = {**field_inventory[f"member:{PKG}.record:Record.depth"], "proposed": "level"}
    edits = codemod.plan([row], tmp_path, roots=(PKG,))
    table = [e for e in edits if e.path == f"{PKG}/constants.py"]
    assert [(e.line, e.certainty) for e in table] == [(4, codemod.DEFINITE)]


def test_underscore_keywords_need_no_glossary_entry_or_guess_item(tmp_path: Path) -> None:
    """A new ``_name=`` keyword is seen by the diff but needs no lockstep place."""

    glossary = tmp_path / "docs/reference/glossary.md"
    glossary.parent.mkdir(parents=True)
    glossary.write_text("nothing\n", encoding="utf-8")
    head = {
        "param:_trace": {
            "id": "param:_trace",
            "kind": "param",
            "name": "_trace",
            "path": "_trace=",
            "fingerprint": "f-a",
            "uses": ["tl.Record"],
        },
        "param:budget": {
            "id": "param:budget",
            "kind": "param",
            "name": "budget",
            "path": "budget=",
            "fingerprint": "f-b",
            "uses": ["tl.Record"],
        },
    }
    surface_diff = diff_inventories({}, head, {})
    assert {row["id"] for row in surface_diff.new} == {"param:_trace", "param:budget"}
    corpus = check.Corpus.read(tmp_path, [glossary], snippets=True)
    findings = check.check_lockstep(surface_diff, corpus, head)
    assert {finding.row_id for finding in findings} == {"param:budget"}
    items = gate_input.gate_items(surface_diff, head, corpus)
    assert [item["id"] for item in items] == ["param:budget"]


@pytest.mark.slow
def test_real_package_inventory_covers_field_tables_and_shims() -> None:
    """The walk over the real package reaches the names the dry run found missing."""

    rows = {row["id"]: row for row in build_inventory()["rows"]}
    durations = rows["member:torchlens.data_classes.trace:Trace.backward_durations"]
    assert "torchlens.constants:MODEL_LOG_FIELD_ORDER" in durations["field_tables"]
    assert {
        "member:torchlens._vocab.trace_state:TraceState.APPENDED",
        "export:tl.constants.MODEL_LOG_FIELD_ORDER",
        "export:tl.bundle.Bundle",
        "export:tl.experimental.dagua.NodeSpec",
        "param:_depth",
    } <= set(rows)
    assert rows["export:tl.io.TraceState"]["module"] == "torchlens._vocab.trace_state"
