"""Shared result type and file helpers for the policy checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

OK = "ok"
FAIL = "fail"
SKIP = "skip"

BASELINES = Path("scripts/checks/baselines.json")


@dataclass(frozen=True)
class Outcome:
    """One check's verdict and the one line printed for it."""

    status: str
    message: str
    details: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def load_baselines(repo: Path) -> dict[str, dict[str, object]]:
    """Read ``scripts/checks/baselines.json`` under ``repo``."""

    data = json.loads((repo / BASELINES).read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def ratchet(name: str, measured: dict[str, int], baseline: dict[str, object]) -> Outcome:
    """Fail when any measured number exceeds its recorded baseline.

    A number below its baseline passes and is reported, so the baseline can be
    lowered in the same change. A measured key with no baseline fails: new
    counters start with a recorded number.
    """

    over, under = [], []
    for key, value in sorted(measured.items()):
        limit = baseline.get(key)
        if not isinstance(limit, int):
            over.append(f"{key}={value} has no baseline")
        elif value > limit:
            over.append(f"{key}={value} > baseline {limit}")
        elif value < limit:
            under.append(f"{key}={value} < baseline {limit} (lower the baseline)")
    shown = ", ".join(f"{k}={v}" for k, v in sorted(measured.items()))
    if over:
        return Outcome(FAIL, f"{name}: over baseline: {'; '.join(over)}", tuple(under))
    return Outcome(OK, f"{name}: {shown}", tuple(under))


def example_section_lines(text: str) -> frozenset[int]:
    """Return the 1-based line numbers of every docstring ``Examples`` section in ``text``.

    The docstring standard asks every public function and class for a runnable
    ``Examples`` section, so the size caps (file-cap, file-size-ratchet) leave exactly those
    lines out of their counts: the blank lines directly before the ``Examples`` header, the
    header, its underline and the section body through its last non-blank line, up to the
    next numpydoc section header or the end of the docstring. Every other line, docstring
    prose included, counts as before. Unparsable text has no sections.
    """

    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return frozenset()
    lines = text.splitlines()
    excluded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        excluded.update(_examples_span(lines, value.lineno, value.end_lineno or value.lineno))
    return frozenset(excluded)


def _is_section_header(lines: list[str], index: int, last: int) -> bool:
    """Whether 0-based ``index`` holds a numpydoc section title underlined on the next line."""

    title = lines[index].strip()
    if not title or title.startswith(('"""', "'''")) or index + 1 > last:
        return False
    underline = lines[index + 1].strip()
    return bool(underline) and set(underline) == {"-"}


def _examples_span(lines: list[str], first: int, last: int) -> set[int]:
    """Return the excluded 1-based line numbers of the ``Examples`` section in one docstring."""

    span: set[int] = set()
    start = None
    for index in range(first - 1, last):
        if start is None:
            if _is_section_header(lines, index, last - 1) and lines[index].strip() == "Examples":
                start = index
            continue
        if _is_section_header(lines, index, last - 1):
            break
    else:
        index = last
    if start is None:
        return span
    end = index  # exclusive; the next header or the docstring's closing line
    if end == last:
        # The section runs to the docstring's end: the closing-quote line and any blank
        # line before it hold neither code nor prose, so they leave with the section.
        end = last
    else:
        while end > start and not lines[end - 1].strip():
            end -= 1
    before = start
    while before > first - 1 and not lines[before - 1].strip():
        before -= 1
    span.update(range(before + 1, end + 1))
    return span
