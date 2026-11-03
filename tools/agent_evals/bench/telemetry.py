"""Transcript parsing, code metrics and the one-record-per-run schema.

The runner reads the agent's event stream as JSON lines. The parser understands the common
headless event shape: ``assistant`` events carry a ``message`` with ``content`` blocks
(``tool_use`` blocks name the tool and its input) and ``usage``; a final ``result`` event may
carry the run's total ``usage``. Usage fields are ``input_tokens`` (uncached),
``cache_read_input_tokens``, ``cache_creation_input_tokens`` and ``output_tokens``. Lines that do
not parse are counted, never fatal.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
#: Every key a run record must carry. ``validate_record`` checks them.
RECORD_KEYS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "task",
    "arm",
    "tier",
    "repeat",
    "started_at",
    "elapsed_s",
    "end_reason",
    "passed",
    "tokens",
    "tool_calls",
    "executions",
    "load_bearing_lines",
    "checkpoints",
    "guard",
    "first_library_call",
    "environment",
    "artifacts",
)
END_REASONS = frozenset(
    {
        "finished",
        "wall_cap",
        "tool_cap",
        "token_cap",
        "agent_error",
        "harness_error",
        "not_applicable",
    }
)
TOKEN_KEYS = ("input_uncached", "input_cache_read", "input_cache_write", "output")
_USAGE_MAP = {
    "input_tokens": "input_uncached",
    "cache_read_input_tokens": "input_cache_read",
    "cache_creation_input_tokens": "input_cache_write",
    "output_tokens": "output",
}
#: A shell tool call counts as an execution when its command runs an interpreter.
EXEC_PATTERN = re.compile(r"(^|[\s;&|(])(python3?|pytest|ipython)(\s|$)")


@dataclass
class StreamStats:
    """Running totals over an event stream, updated one line at a time."""

    shell_tools: frozenset[str] = frozenset({"Bash", "shell", "bash"})
    non_substantive: frozenset[str] = frozenset()
    tool_counts: Counter[str] = field(default_factory=Counter)
    executions: int = 0
    tokens: dict[str, int] = field(default_factory=lambda: dict.fromkeys(TOKEN_KEYS, 0))
    final_usage: dict[str, int] | None = None
    bad_lines: int = 0
    code_snippets: list[str] = field(default_factory=list)
    _seen_messages: set[str] = field(default_factory=set)

    @property
    def substantive_calls(self) -> int:
        """Tool calls excluding the configured bookkeeping tools."""

        return sum(n for name, n in self.tool_counts.items() if name not in self.non_substantive)

    def total_tokens(self) -> dict[str, int]:
        """Token totals: the stream's final usage when it reported one, else the running sum."""

        return dict(self.final_usage) if self.final_usage is not None else dict(self.tokens)

    def feed(self, line: str) -> None:
        """Fold one stream line into the totals."""

        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except ValueError:
            self.bad_lines += 1
            return
        if not isinstance(event, Mapping):
            self.bad_lines += 1
            return
        if event.get("type") == "result" and isinstance(event.get("usage"), Mapping):
            self.final_usage = _usage(event["usage"])
        message = event.get("message")
        if event.get("type") != "assistant" or not isinstance(message, Mapping):
            return
        message_id = str(message.get("id", ""))
        if isinstance(message.get("usage"), Mapping) and message_id not in self._seen_messages:
            # A message split over several events repeats its usage; count it once.
            if message_id:
                self._seen_messages.add(message_id)
            for key, value in _usage(message["usage"]).items():
                self.tokens[key] += value
        for block in message.get("content") or []:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                self._tool(str(block.get("name", "?")), block.get("input") or {})

    def _tool(self, name: str, tool_input: Any) -> None:
        self.tool_counts[name] += 1
        strings = list(_strings(tool_input))
        if name in self.shell_tools and any(EXEC_PATTERN.search(s) for s in strings):
            self.executions += 1
        self.code_snippets.extend(strings)


def _usage(raw: Mapping[str, Any]) -> dict[str, int]:
    out = dict.fromkeys(TOKEN_KEYS, 0)
    for src, dst in _USAGE_MAP.items():
        value = raw.get(src)
        if isinstance(value, int):
            out[dst] = value
    return out


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def agent_python_files(work: Path, exclude: Iterable[str] = ()) -> list[Path]:
    """Return the Python files in the work folder, excluding provided inputs and test files."""

    skip = set(exclude)
    return sorted(
        path
        for path in work.rglob("*.py")
        if path.relative_to(work).as_posix() not in skip
        and not path.name.startswith("test_")
        and not any(
            part.startswith(".") or part == "__pycache__" for part in path.relative_to(work).parts
        )
    )


def load_bearing_lines(source: str) -> int:
    """Count source lines that carry code: not blank, not comment-only, not inside a docstring.

    Self-tests (files named ``test_*.py``) are excluded by the caller. An unparsable file
    counts its non-blank, non-comment lines.
    """

    lines = source.splitlines()
    docstring_lines: set[int] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list) or not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstring_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return sum(
        1
        for number, text in enumerate(lines, start=1)
        if text.strip() and not text.strip().startswith("#") and number not in docstring_lines
    )


def first_library_call(snippets: Sequence[str], library: str) -> str | None:
    """Return the first call into ``library`` (dotted, as resolved through imports), or ``None``.

    ``snippets`` are code strings in the order the agent wrote them (tool inputs, then final
    files). Within a snippet, calls are taken in source order. Snippets that are not Python are
    skipped; a shell heredoc or ``python -c`` body is tried as Python after stripping the
    shell lines around it.
    """

    for snippet in snippets:
        for candidate in _python_candidates(snippet):
            call = _first_call(candidate, library)
            if call:
                return call
    return None


def _python_candidates(snippet: str) -> list[str]:
    if "import" not in snippet:
        return []
    candidates = [snippet]
    heredoc = re.search(r"<<\s*'?(\w+)'?\n(.*?)\n\1", snippet, re.S)
    if heredoc:
        candidates.append(heredoc.group(2))
    inline = re.search(r"python3?\s+-c\s+(['\"])(.*?)\1", snippet, re.S)
    if inline:
        candidates.append(inline.group(2))
    return candidates


def _first_call(source: str, library: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == library:
                    aliases[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else library
                    )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == library
        ):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    if not aliases:
        return None
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for call in calls:
        parts: list[str] = []
        func: ast.AST = call.func
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name) and func.id in aliases:
            return ".".join([aliases[func.id], *reversed(parts)])
    return None


def validate_record(record: Mapping[str, Any]) -> list[str]:
    """Return the schema problems in one run record (empty when valid)."""

    problems = [f"missing {key}" for key in RECORD_KEYS if key not in record]
    if problems:
        return problems
    if record["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version {record['schema_version']} != {SCHEMA_VERSION}")
    if record["end_reason"] not in END_REASONS:
        problems.append(f"unknown end_reason {record['end_reason']!r}")
    if not isinstance(record["passed"], bool):
        problems.append("passed is not a bool")
    tokens = record["tokens"]
    if not isinstance(tokens, Mapping) or any(
        not isinstance(tokens.get(key), int) for key in TOKEN_KEYS
    ):
        problems.append("tokens must hold integer " + ", ".join(TOKEN_KEYS))
    calls = record["tool_calls"]
    if not isinstance(calls, Mapping) or not {"total", "substantive", "by_name"} <= set(calls):
        problems.append("tool_calls must hold total, substantive and by_name")
    if not isinstance(record["checkpoints"], Mapping):
        problems.append("checkpoints is not a mapping")
    else:
        for name, verdict in record["checkpoints"].items():
            if not isinstance(verdict, Mapping) or not isinstance(verdict.get("passed"), bool):
                problems.append(f"checkpoint {name} has no boolean passed")
    return problems
