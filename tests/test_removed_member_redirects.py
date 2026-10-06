"""Removed public methods and attributes raise a typed redirect naming the replacement.

The 2026-10-01 shim removal deleted the alias members outright (remove-and-
rename, no alias), and lane F12 deleted ``VisualizationTheme.legend_items``.
A lookup of any of them must raise ``FacadeTeachingError`` (code
``facade_redirect``, ``AttributeError`` lineage) whose message names the
replacement, never a bare ``AttributeError``. Every other missing name keeps
failing plainly, and ``getattr`` with a default still degrades.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import FacadeTeachingError
from torchlens.options import VisualizationOptions
from torchlens.visualization.themes import THEME_PRESETS


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


@pytest.fixture(scope="module")
def owners():
    """One live instance of every class that lost public members."""

    first = tl.trace(_Toy(), torch.randn(1, 4))
    second = tl.trace(_Toy(), torch.randn(1, 4))
    try:
        yield {
            "Trace": first,
            "Bundle": tl.bundle({"a": first, "b": second}),
            "VisualizationOptions": VisualizationOptions(),
            "VisualizationTheme": THEME_PRESETS["torchlens"],
        }
    finally:
        first.cleanup()
        second.cleanup()


#: (owner class, removed member, live replacement member)
REMOVED_MEMBERS: list[tuple[str, str, str]] = [
    ("Trace", "replay", "push"),
    ("Trace", "replay_from", "push_from"),
    ("Trace", "rerun", "run"),
    ("Trace", "validate_saved_outs", "validate_forward_pass"),
    ("Trace", "conditional_then_entry_edges", "conditional_arm_entry_edges"),
    ("Trace", "conditional_elif_entry_edges", "conditional_arm_entry_edges"),
    ("Trace", "conditional_else_entry_edges", "conditional_arm_entry_edges"),
    ("Bundle", "replay", "push"),
    ("Bundle", "rerun", "run"),
    ("VisualizationOptions", "mode", "view"),
    ("VisualizationOptions", "max_module_depth", "depth"),
    ("VisualizationOptions", "layout_engine", "layout"),
    ("VisualizationOptions", "node_mode", "node_style"),
    ("VisualizationTheme", "legend_items", "semantic_palette"),
]


@pytest.mark.parametrize(("owner", "removed", "replacement"), REMOVED_MEMBERS)
def test_removed_member_raises_typed_redirect(
    owners: dict[str, Any], owner: str, removed: str, replacement: str
) -> None:
    """The removed spelling refuses typed and names a replacement that exists."""

    instance = owners[owner]
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(instance, removed)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert removed in str(excinfo.value)
    assert replacement in str(excinfo.value)
    assert getattr(instance, removed, "absent") == "absent"
    getattr(instance, replacement)  # the named replacement is live


#: (public module, removed module-level name, live replacement on that module)
REMOVED_MODULE_NAMES: list[tuple[str, str, str]] = [
    ("torchlens.validation", "validate_saved_outs", "validate"),
    ("torchlens.validation", "validate_trace_saved_outs", "validate"),
    ("torchlens.io", "get_model_metadata", "log_model_metadata"),
    ("torchlens.observers", "record_span", "span"),
    ("torchlens.intervention", "intervening", "without_op"),
    ("torchlens.intervention", "replay_from", "push_from"),
]


@pytest.mark.parametrize(("module_name", "removed", "replacement"), REMOVED_MODULE_NAMES)
def test_removed_module_name_raises_typed_redirect(
    module_name: str, removed: str, replacement: str
) -> None:
    """A public subpackage's removed name refuses typed and names a live replacement."""

    import importlib

    module = importlib.import_module(module_name)
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(module, removed)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert removed in str(excinfo.value)
    assert f"{module_name}.{replacement}" in str(excinfo.value)
    assert getattr(module, removed, "absent") == "absent"
    assert removed not in getattr(module, "__all__", ())
    getattr(module, replacement)  # the named replacement is live


@pytest.mark.parametrize("owner", ["Trace", "Bundle", "VisualizationOptions", "VisualizationTheme"])
def test_unknown_member_still_fails_plain(owners: dict[str, Any], owner: str) -> None:
    """Only removed public names redirect; any other miss stays a plain AttributeError."""

    with pytest.raises(AttributeError) as excinfo:
        getattr(owners[owner], "no_such_member_anywhere")
    assert not isinstance(excinfo.value, FacadeTeachingError)


_REPO = Path(__file__).resolve().parents[1]

#: Redirect hooks added for removed names; each must be invisible to type
#: checkers, or mypy would accept every attribute (removed or typo) as Any.
_GUARDED_HOOKS: list[tuple[str, str | None]] = [
    ("torchlens/intervention/helpers.py", None),
    ("torchlens/validation/__init__.py", None),
    ("torchlens/io/__init__.py", None),
    ("torchlens/observers.py", None),
    ("torchlens/options.py", "VisualizationOptions"),
    ("torchlens/visualization/themes.py", "VisualizationTheme"),
]


def _is_not_type_checking(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "TYPE_CHECKING"
    )


@pytest.mark.parametrize(("path", "class_name"), _GUARDED_HOOKS)
def test_redirect_hook_is_hidden_from_type_checkers(path: str, class_name: str | None) -> None:
    """The ``__getattr__`` hook sits under ``if not TYPE_CHECKING:`` and nowhere else."""

    tree = ast.parse((_REPO / path).read_text(encoding="utf-8"))
    scope: list[ast.stmt] = tree.body
    if class_name is not None:
        (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
        scope = cls.body
    bare = [n for n in scope if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"]
    guarded = [
        inner
        for n in scope
        if _is_not_type_checking(n)
        for inner in n.body
        if isinstance(inner, ast.FunctionDef) and inner.name == "__getattr__"
    ]
    assert bare == [] and len(guarded) == 1


def test_from_import_of_a_removed_name_is_a_plain_import_error() -> None:
    """Pins the documented CPython limit: ``from`` imports drop the typed redirect.

    ``from pkg import name`` replaces the module ``__getattr__``'s
    AttributeError with a generic ImportError, so only attribute access
    carries the facade_redirect message (MIGRATIONS.md, error contract). If a
    future CPython keeps the original error, this test flags the docs as stale.
    """

    namespace: dict[str, Any] = {}
    with pytest.raises(ImportError) as excinfo:
        exec("from torchlens import resample_ablate", namespace)
    assert not isinstance(excinfo.value, FacadeTeachingError)
    assert "cannot import name" in str(excinfo.value)


def test_root_resample_ablate_row_matches_the_shared_text() -> None:
    """The root package's literal copy equals the intervention packages' shared text."""

    from torchlens import _REDIRECTS as root_redirects
    from torchlens.intervention import _REDIRECTS as package_redirects
    from torchlens.intervention.helpers import _REDIRECTS as helper_redirects

    shared = helper_redirects["resample_ablate"]
    assert root_redirects["resample_ablate"] == shared
    assert package_redirects["resample_ablate"] == shared
