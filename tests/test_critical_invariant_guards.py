"""Direct guards for Critical Invariants that had no dedicated test.

Invariant 1: ``torchlens/_state.py`` imports no torchlens module except the one
sanctioned cycle-safe leaf, ``from .errors._base import CaptureError`` (plus
``TYPE_CHECKING``-only imports, which never run).

Invariant 16: smart-collapse metadata is computed at access time and must stay
out of every ``*_FIELD_ORDER`` schema until the policy is stabilized.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import torchlens.constants as tl_constants

pytestmark = pytest.mark.smoke

STATE_PATH = Path(__file__).resolve().parents[1] / "torchlens" / "_state.py"

#: The only torchlens import ``_state.py`` may execute at runtime.
SANCTIONED_STATE_IMPORTS = {(1, "errors._base", "CaptureError")}

#: Access-time smart-collapse metadata that must never become a portable field.
COLLAPSE_METADATA_NAMES = ("collapse_score", "module_collapse_order", "collapse_order")


def _is_type_checking_guard(node: ast.If) -> bool:
    """Return whether an ``if`` statement is an ``if TYPE_CHECKING:`` guard.

    Parameters
    ----------
    node:
        The ``if`` statement to inspect.

    Returns
    -------
    bool
        True for ``if TYPE_CHECKING:`` and ``if typing.TYPE_CHECKING:``.
    """
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _runtime_torchlens_imports(tree: ast.Module) -> set[tuple[int, str, str]]:
    """Collect every torchlens import a module executes outside ``TYPE_CHECKING``.

    Parameters
    ----------
    tree:
        Parsed module.

    Returns
    -------
    set[tuple[int, str, str]]
        ``(relative level, module, imported name)`` triples; absolute
        ``torchlens`` imports use level 0.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            guarded.update(id(child) for stmt in node.body for child in ast.walk(stmt))
    found: set[tuple[int, str, str]] = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0 or module == "torchlens" or module.startswith("torchlens."):
                found.update((node.level, module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torchlens" or alias.name.startswith("torchlens."):
                    found.add((0, alias.name, alias.name))
    return found


def test_state_module_imports_only_the_sanctioned_leaf() -> None:
    """Invariant 1: ``_state.py`` executes no torchlens import but ``CaptureError``."""
    tree = ast.parse(STATE_PATH.read_text(encoding="utf-8"))
    imports = _runtime_torchlens_imports(tree)
    assert imports == SANCTIONED_STATE_IMPORTS, (
        "torchlens/_state.py must import no torchlens module except "
        "`from .errors._base import CaptureError`; found runtime imports "
        f"{sorted(imports)}"
    )


def test_state_import_scan_sees_a_planted_violation() -> None:
    """The invariant-1 scanner flags a runtime import and ignores a guarded one."""
    planted = ast.parse(
        "from typing import TYPE_CHECKING\n"
        "from .errors._base import CaptureError\n"
        "if TYPE_CHECKING:\n"
        "    from .data_classes.trace import Trace\n"
        "def late():\n"
        "    from .capture import trace\n"
    )
    assert _runtime_torchlens_imports(planted) == {
        (1, "errors._base", "CaptureError"),
        (1, "capture", "trace"),
    }


def test_collapse_metadata_stays_out_of_every_field_order() -> None:
    """Invariant 16: collapse metadata is never declared in a ``*_FIELD_ORDER``."""
    field_orders = {
        name: value
        for name, value in vars(tl_constants).items()
        if name.endswith("FIELD_ORDER") and isinstance(value, (list, tuple))
    }
    assert "MODULE_LOG_FIELD_ORDER" in field_orders
    assert "MODEL_LOG_FIELD_ORDER" in field_orders
    leaks = sorted(
        (name, field)
        for name, fields in field_orders.items()
        for field in fields
        if field in COLLAPSE_METADATA_NAMES
    )
    assert not leaks, (
        "smart-collapse metadata is computed at access time and must stay out of "
        f"*_FIELD_ORDER schemas (Critical Invariant 16); found {leaks}"
    )
