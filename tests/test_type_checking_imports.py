"""Every package import under ``if TYPE_CHECKING:`` names a module that exists.

Annotation-only imports never execute, and mypy runs with
``ignore_missing_imports``, so a relative import with one dot too few (a
``...data_classes`` that should read ``...._data_classes``) passes every
runtime test and every type check while leaving the annotation unresolvable
for readers, IDEs and ``typing.get_type_hints``. This scan resolves each such
import against the source tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "torchlens"


def _module_exists(parts: list[str]) -> bool:
    """Return whether dotted ``parts`` name a module or package in the tree."""

    path = _REPO_ROOT.joinpath(*parts)
    return (
        path.with_suffix(".py").is_file()
        or path.with_suffix(".pyi").is_file()
        or (path / "__init__.py").is_file()
    )


def _package_parts(source: Path) -> list[str]:
    """Return the dotted package that relative imports in ``source`` start from."""

    parts = list(source.relative_to(_REPO_ROOT).with_suffix("").parts)
    return parts[:-1]


def _is_type_checking_guard(node: ast.If) -> bool:
    """Return whether ``node`` is ``if TYPE_CHECKING:`` or ``if typing.TYPE_CHECKING:``."""

    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _target_parts(source: Path, node: ast.ImportFrom) -> list[str] | None:
    """Return the dotted target of a package import, or ``None`` for third-party ones."""

    module = node.module.split(".") if node.module else []
    if node.level:
        base = _package_parts(source)
        if node.level > 1:
            base = base[: len(base) - (node.level - 1)]
        return base + module
    if module and module[0] == "torchlens":
        return module
    return None


def _unresolved_in(source: Path) -> list[str]:
    """Return ``path:line import`` rows for unresolvable guarded package imports."""

    tree = ast.parse(source.read_text(encoding="utf-8"))
    rows: list[str] = []
    for guard in ast.walk(tree):
        if not (isinstance(guard, ast.If) and _is_type_checking_guard(guard)):
            continue
        for node in ast.walk(guard):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _target_parts(source, node)
            if target is None or _module_exists(target):
                continue
            if all(_module_exists([*target, alias.name]) for alias in node.names):
                continue
            spelled = "." * node.level + (node.module or "")
            rows.append(f"{source.relative_to(_REPO_ROOT)}:{node.lineno} {spelled}")
    return rows


def test_type_checking_package_imports_resolve() -> None:
    """No annotation-only import points at a package module that does not exist."""

    unresolved = [
        row for source in sorted(_PACKAGE_ROOT.rglob("*.py")) for row in _unresolved_in(source)
    ]
    assert not unresolved, "unresolvable TYPE_CHECKING imports:\n  " + "\n  ".join(unresolved)
