"""AST guardrail for training-mode detach chokepoints."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_PATHS = (
    REPO_ROOT / "torchlens" / "capture",
    REPO_ROOT / "torchlens" / "decoration",
    REPO_ROOT / "torchlens" / "postprocess",
    REPO_ROOT / "torchlens" / "fastlog",
    REPO_ROOT / "torchlens" / "data_classes" / "layer_pass_log.py",
)
ALLOWLIST = {
    "LayerPassLog._tensor_contents_str_helper",
    "LayerPassLog.log_tensor_grad",
    "log_tensor_grad",
}
MAX_NOQA_DETACH_EXEMPTIONS = 5


def _python_files(path: Path) -> list[Path]:
    """Return Python files under ``path`` in deterministic order.

    Parameters
    ----------
    path:
        File or directory to scan.

    Returns
    -------
    list[Path]
        Python files covered by the guardrail.
    """

    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.rglob("*.py") if candidate.is_file())


def _qualified_name(stack: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a function qualified name from the current class stack.

    Parameters
    ----------
    stack:
        Active class/function name stack.
    node:
        Function node being inspected.

    Returns
    -------
    str
        Dot-qualified function name.
    """

    return ".".join([*stack, node.name])


def _has_noqa_detach(source_lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function body contains a ``# noqa: detach`` exemption.

    Parameters
    ----------
    source_lines:
        Source file split into individual lines.
    node:
        Function node being inspected.

    Returns
    -------
    bool
        Whether the function has an explicit detach exemption comment.
    """

    end_lineno = node.end_lineno or node.lineno
    body_lines = source_lines[node.lineno - 1 : end_lineno]
    return any("# noqa: detach" in line for line in body_lines)


class _DetachCallVisitor(ast.NodeVisitor):
    """Find bare ``.detach(...)`` calls in non-exempt functions."""

    def __init__(self, source_lines: list[str]) -> None:
        """Initialize the visitor.

        Parameters
        ----------
        source_lines:
            Source file split into individual lines.
        """

        self.source_lines = source_lines
        self.name_stack: list[str] = []
        self.violations: list[tuple[int, str]] = []
        self.noqa_exemptions = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition while tracking qualified names."""

        self.name_stack.append(node.name)
        self.generic_visit(node)
        self.name_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition."""

        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition."""

        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Inspect a function body unless it is explicitly exempt."""

        qualified_name = _qualified_name(self.name_stack, node)
        if qualified_name in ALLOWLIST or _has_noqa_detach(self.source_lines, node):
            if _has_noqa_detach(self.source_lines, node):
                self.noqa_exemptions += 1
            return

        self.name_stack.append(node.name)
        for child in node.body:
            self.visit(child)
        self.name_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Record calls to attributes named ``detach``."""

        if isinstance(node.func, ast.Attribute) and node.func.attr == "detach":
            self.violations.append((node.lineno, "bare .detach() call"))
        self.generic_visit(node)


def test_no_bare_detach_outside_training_guardrail_allowlist() -> None:
    """Scanned training hot paths route detach behavior through chokepoints."""

    failures: list[str] = []
    noqa_exemptions = 0
    for scanned_path in SCANNED_PATHS:
        for path in _python_files(scanned_path):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            visitor = _DetachCallVisitor(source.splitlines())
            visitor.visit(tree)
            noqa_exemptions += visitor.noqa_exemptions
            rel_path = path.relative_to(REPO_ROOT)
            failures.extend(
                f"{rel_path}:{line_no}: {message}" for line_no, message in visitor.violations
            )

    assert noqa_exemptions <= MAX_NOQA_DETACH_EXEMPTIONS
    assert failures == []
