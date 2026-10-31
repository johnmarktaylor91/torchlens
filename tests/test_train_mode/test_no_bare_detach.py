"""AST guardrail for training-mode detach chokepoints."""

from __future__ import annotations

import ast
from pathlib import Path

from _source_corpus import module_ast, module_source

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_PATHS = (
    REPO_ROOT / "torchlens" / "capture",
    REPO_ROOT / "torchlens" / "backends" / "torch",
    REPO_ROOT / "torchlens" / "_postprocess",
    REPO_ROOT / "torchlens" / "fastlog",
    REPO_ROOT / "torchlens" / "_data_classes" / "op.py",
)
ALLOWLIST = {
    "Op._tensor_contents_str_helper",
    "Op._log_tensor_grad",
    "_log_tensor_grad",
}
MAX_DETACH_OK_EXEMPTIONS = 5


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

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist. A missing entry previously fell through to
        the ``rglob`` branch and yielded ``[]`` silently, leaving that leg of
        the guardrail vacuously green (the ``op_log.py`` -> ``op.py`` rename
        disarmed the op.py leg for months without a failure).
    """

    if not path.exists():
        raise FileNotFoundError(f"SCANNED_PATHS entry does not exist: {path}")
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


def _has_detach_ok(source_lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function body carries an explicit ``# detach-ok:`` exemption.

    The marker used to be spelled ``# noqa: detach``, which was a mistake: ruff
    parses ``# noqa:`` and rejected ``detach`` as a rule code, so it warned on
    every run and suppressed nothing, while THIS scanner silently treated the
    same comment as a valid exemption. One comment, two tools, two verdicts
    (grind r3, SF-24). ``# detach-ok:`` is unambiguous -- ruff never reads it,
    and it is the only spelling this scanner accepts.

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
    return any("# detach-ok:" in line for line in body_lines)


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
        self.detach_ok_exemptions = 0

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
        if qualified_name in ALLOWLIST or _has_detach_ok(self.source_lines, node):
            if _has_detach_ok(self.source_lines, node):
                self.detach_ok_exemptions += 1
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
    detach_ok_exemptions = 0
    for scanned_path in SCANNED_PATHS:
        for path in _python_files(scanned_path):
            source = module_source(path)
            tree = module_ast(path)
            visitor = _DetachCallVisitor(source.splitlines())
            visitor.visit(tree)
            detach_ok_exemptions += visitor.detach_ok_exemptions
            rel_path = path.relative_to(REPO_ROOT)
            failures.extend(
                f"{rel_path}:{line_no}: {message}" for line_no, message in visitor.violations
            )

    assert detach_ok_exemptions <= MAX_DETACH_OK_EXEMPTIONS
    assert failures == []
