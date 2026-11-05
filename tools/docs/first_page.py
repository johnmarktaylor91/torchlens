"""First-page allowlist: the README's first screen and the landing page name only core calls.

The allowed names live in ``first_page_allowlist.json`` beside this file, so the
spellings can change without touching code. A page's first screen is everything
above its first level-2 heading. Code is read from fenced blocks and inline code
spans; a name counts when it is a ``tl.``/``torchlens.`` attribute, a method or
attribute of a trace variable (``trace = tl.trace(...)`` or one of the configured
names), or an attribute read straight off an indexed trace (``trace["x"].out``).

Run ``python tools/docs/first_page.py`` to list violations on every configured page.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path(__file__).with_name("first_page_allowlist.json")
SCOPES = ("first_screen", "whole")

_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
_INLINE = re.compile(r"`([^`\n]+)`")
_H2 = re.compile(r"^## ", re.MULTILINE)
_TL_NAME = re.compile(r"(?<![\w.])(?:tl|torchlens)\.([A-Za-z_]\w*)")
_TRACE_ASSIGN = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*=\s*(?:tl|torchlens)\.trace\(")
_IMPORT = re.compile(r"^\s*(?:import|from)\s+torchlens\b.*$", re.MULTILINE)


@dataclass(frozen=True)
class Violation:
    """One name on a first page that is not in the allowlist."""

    path: str
    kind: str
    name: str
    snippet: str


def load_allowlist(path: Path = ALLOWLIST) -> dict[str, Any]:
    """Read and sanity-check the allowlist data file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("tl", "trace_methods", "layer_attributes"):
        if not isinstance(data["allowed"].get(key), list):
            raise ValueError(f"allowed.{key} must be a list")
    for page in data["pages"]:
        if page["scope"] not in SCOPES:
            raise ValueError(f"{page['path']}: scope must be one of {SCOPES}")
        if not page["enforced"] and not page.get("reason", "").strip():
            raise ValueError(f"{page['path']}: an unenforced page must state its reason")
    return data


def first_screen(text: str, scope: str = "first_screen") -> str:
    """Return the part of a Markdown page the rule applies to."""

    if scope == "whole":
        return text
    lines = text.splitlines(keepends=True)
    in_fence = False
    for index, line in enumerate(lines):
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and _H2.match(line):
            return "".join(lines[:index])
    return text


def code_of(text: str) -> list[str]:
    """Return the code in fenced blocks and inline code spans, fences first."""

    fenced = [m.group(1) for m in _FENCE.finditer(text)]
    prose = _FENCE.sub("", text)
    return fenced + [m.group(1) for m in _INLINE.finditer(prose)]


def violations(text: str, path: str, allow: dict[str, Any]) -> list[Violation]:
    """List every non-core name in the code of ``text``."""

    allowed = allow["allowed"]
    found: list[Violation] = []
    chunks = [_IMPORT.sub("", chunk) for chunk in code_of(text)]
    trace_vars = set(allow["trace_variable_names"])
    for chunk in chunks:
        trace_vars.update(_TRACE_ASSIGN.findall(chunk))
    var_pattern = "|".join(sorted(map(re.escape, trace_vars)))
    method = re.compile(rf"(?<![\w.])(?:{var_pattern})\.([A-Za-z_]\w*)")
    indexed = re.compile(rf"(?<![\w.])(?:{var_pattern})\[[^\]\n]*\]\.([A-Za-z_]\w*)")
    for chunk in chunks:
        for line in chunk.splitlines():
            code = line.split("#", 1)[0]
            for kind, regex, names in (
                ("tl", _TL_NAME, allowed["tl"]),
                ("trace_method", method, allowed["trace_methods"]),
                ("layer_attribute", indexed, allowed["layer_attributes"]),
            ):
                for name in regex.findall(code):
                    if name not in names:
                        found.append(Violation(path, kind, name, line.strip()))
    return found


def page_violations(
    page: dict[str, Any], allow: dict[str, Any], root: Path = REPO_ROOT
) -> list[Violation]:
    """List the violations on one configured page (empty when the page is absent)."""

    file = root / page["path"]
    if not file.is_file():
        return []
    text = first_screen(file.read_text(encoding="utf-8"), page["scope"])
    return violations(text, page["path"], allow)


def main() -> int:
    """Print violations for every configured page; exit 1 when an enforced page fails."""

    allow = load_allowlist()
    status = 0
    for page in allow["pages"]:
        found = page_violations(page, allow)
        label = "enforced" if page["enforced"] else "not enforced"
        print(f"{page['path']} ({label}): {len(found)} violation(s)")
        for v in found:
            print(f"  {v.kind} {v.name!r}: {v.snippet}")
        if found and page["enforced"]:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
