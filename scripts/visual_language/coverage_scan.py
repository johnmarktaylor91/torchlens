"""Torch-free source scans of the visual language coverage check.

Reads the renderer source (through the shared test corpus when the suite has loaded it, so
a session parses each file once) and derives three universes: literal Graphviz tokens and
hex colours, label-template prefixes, and emission sites fingerprinted by the vocabulary
they emit. Nothing here imports torch.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "torchlens"

#: Graphviz attributes whose literal values are visual tokens.
TOKEN_KEYS = ("shape", "style", "arrowhead", "peripheries", "dir")
HEX_RE = re.compile(r"(?<![&\w])#([0-9A-Fa-f]{6})(?![0-9A-Fa-f])")
KV_RE = re.compile(r"\b(shape|style|arrowhead|peripheries|dir)\s*=\s*\"?([A-Za-z0-9_,]+)")
TAG_RE = re.compile(r"<[/!A-Za-z<][^>]*(?:>|$)")
SILENT_CALLS = frozenset(
    {"warn", "warning", "info", "debug", "error", "exception", "critical", "log", "_warn"}
)
EMIT_ATTRS = frozenset({"node", "edge", "attr", "subgraph"})
EMIT_NAMES = frozenset({"NodeSpec", "LegendRow"})

#: Label-template sources: file (relative to ``torchlens/visualization``) and the
#: functions whose strings count (``None`` means the whole file).
LABEL_SOURCES: Mapping[str, tuple[str, ...] | None] = {
    "_label_format.py": None,
    "_render_nodes.py": ("compute_default_node_lines",),
    "_render_edges.py": None,
    "_edge_multiplicity.py": None,
    "_render_leaf.py": ("cond", "branch", "label"),
}

# ---------------------------------------------------------------------------
# Universe items
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class DrawParam:
    """One draw-surface parameter and the entry point that accepts it."""

    entry: str
    name: str


@dataclass(frozen=True, order=True)
class VocabValue:
    """One value of a closed vocabulary."""

    vocab: str
    value: str
    source: str


@dataclass(frozen=True, order=True)
class LegendText:
    """One legend row text, its section title and where it was built."""

    section: str
    text: str
    source: str


@dataclass(frozen=True, order=True)
class SourceItem:
    """A token or template found in the source, with its first location."""

    key: str
    source: str


@dataclass(frozen=True, order=True)
class EmissionSite:
    """One function that emits Graphviz or legend vocabulary."""

    site: str
    fingerprint: str
    source: str


# ---------------------------------------------------------------------------
# Source access (shared test corpus when loaded)
# ---------------------------------------------------------------------------


def _corpus() -> Any:
    return sys.modules.get("_source_corpus")


@cache
def _local_files() -> tuple[Path, ...]:
    return tuple(sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts))


def _package_files() -> tuple[Path, ...]:
    corpus = _corpus()
    return corpus.package_files() if corpus is not None else _local_files()


@cache
def _local_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _tree(path: Path) -> ast.Module:
    corpus = _corpus()
    return corpus.module_ast(path) if corpus is not None else _local_ast(path)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def scanned_files() -> tuple[Path, ...]:
    """The renderer files the token and emission scans read."""

    def wanted(rel: str) -> bool:
        return (
            rel.startswith("torchlens/visualization/")
            or rel == "torchlens/_vocab/node_spec.py"
            or rel.startswith("torchlens/experimental/dagua/")
        )

    return tuple(path for path in _package_files() if wanted(_rel(path)))


def _module_name(path: Path) -> str:
    return _rel(path)[: -len(".py")].replace("/", ".")


# ---------------------------------------------------------------------------
# AST walking with context
# ---------------------------------------------------------------------------


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_silenced(node: ast.AST) -> bool:
    if isinstance(node, ast.Raise):
        return True
    if isinstance(node, ast.Call):
        name = _call_name(node)
        return name in SILENT_CALLS or name.endswith(("Error", "Exception", "Warning"))
    return False


RELEVANT = (ast.Constant, ast.Call, ast.Dict, ast.Assign, ast.JoinedStr, ast.Return)
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class _FileIndex:
    """One pass over a file: relevant nodes in source order, silenced ids, def spans."""

    nodes: tuple[ast.AST, ...]
    silenced: frozenset[int]
    fstring_parts: frozenset[int]
    defs: tuple[tuple[tuple[int, int], tuple[int, int], str], ...]


def _span(node: ast.AST) -> tuple[tuple[int, int], tuple[int, int]]:
    start = (node.lineno, node.col_offset)  # type: ignore[attr-defined]
    end = (node.end_lineno or 0, node.end_col_offset or 0)  # type: ignore[attr-defined]
    return start, end


@cache
def _index_tree(tree: ast.Module) -> _FileIndex:
    nodes: list[ast.AST] = []
    roots: list[ast.AST] = []
    defs = []
    skip: set[int] = set()
    parts: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts.update(id(value) for value in node.values)
        if isinstance(node, (ast.Module, *DEFS)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                skip.add(id(body[0].value))
            if not isinstance(node, ast.Module):
                defs.append((*_span(node), node.name))
        if isinstance(node, RELEVANT):
            nodes.append(node)
        if _is_silenced(node):
            roots.append(node)
    for root in roots:
        skip.update(id(sub) for sub in ast.walk(root))
    ordered = sorted(nodes, key=lambda n: (n.lineno, n.col_offset))  # type: ignore[attr-defined]
    return _FileIndex(tuple(ordered), frozenset(skip), frozenset(parts), tuple(sorted(defs)))


def _scope(index: _FileIndex, node: ast.AST) -> tuple[str, ...]:
    """Qualname parts of the defs enclosing ``node``, outermost first."""

    start, end = _span(node)
    return tuple(name for d_start, d_end, name in index.defs if d_start <= start and end <= d_end)


def _walk(tree: ast.Module) -> Iterator[tuple[ast.AST, _FileIndex]]:
    """Yield ``(node, index)`` for every relevant node outside docstrings and messages."""

    index = _index_tree(tree)
    for node in index.nodes:
        if id(node) not in index.silenced:
            yield node, index


# ---------------------------------------------------------------------------
# Universe 4: visual tokens
# ---------------------------------------------------------------------------


def _norm_token(key: str, value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip().lower().replace(" ", "")
    if not text or not re.fullmatch(r"[a-z0-9_,]+", text):
        return None
    if key == "style":
        text = ",".join(sorted(part for part in text.split(",") if part))
    return f"{key}={text}"


def _value_constants(value: ast.AST) -> list[object]:
    if isinstance(value, ast.Constant):
        return [value.value]
    if isinstance(value, ast.IfExp):
        return _value_constants(value.body) + _value_constants(value.orelse)
    if isinstance(value, ast.BoolOp):
        return [c for v in value.values for c in _value_constants(v)]
    return []


def _attr_pairs(node: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    if isinstance(node, ast.Call):
        for keyword in node.keywords:
            if keyword.arg is not None:
                yield keyword.arg, keyword.value
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                yield key.value, value
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                if isinstance(target.slice.value, str):
                    yield target.slice.value, node.value
            elif isinstance(target, ast.Name):
                yield target.id, node.value


def _string_tokens(text: str) -> Iterator[str]:
    for match in HEX_RE.finditer(text):
        yield "#" + match.group(1).lower()
    for match in KV_RE.finditer(text):
        token = _norm_token(match.group(1), match.group(2))
        if token is not None:
            yield token


def _attr_key(name: str) -> str | None:
    if name in TOKEN_KEYS:
        return name
    for key in ("shape", "style"):
        if name.endswith("_" + key):
            return key
    return None


def _render_common_constants(path: Path) -> Iterator[tuple[str, int]]:
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.isupper():
                if not target.id.startswith("_") and isinstance(
                    node.value, (ast.Constant, ast.List, ast.Tuple)
                ):
                    yield f"const:{target.id}", node.lineno


def _file_tokens(path: Path) -> Iterator[tuple[str, int]]:
    for node, _index in _walk(_tree(path)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for token in _string_tokens(node.value):
                yield token, line
        for name, value in _attr_pairs(node):
            key = _attr_key(name)
            if key is None:
                continue
            for constant in _value_constants(value):
                token = _norm_token(key, constant)
                if token is not None:
                    yield token, line
    if path.name == "_render_common.py":
        yield from _render_common_constants(path)


def visual_token_universe(files: Iterable[Path] | None = None) -> tuple[SourceItem, ...]:
    """Every literal Graphviz token and hex colour in the renderer, first location each."""

    first: dict[str, str] = {}
    for path in files if files is not None else scanned_files():
        for token, line in _file_tokens(path):
            first.setdefault(token, f"{_rel(path)}:{line}")
    return tuple(sorted(SourceItem(token, source) for token, source in first.items()))


# ---------------------------------------------------------------------------
# Universe 5: label templates
# ---------------------------------------------------------------------------


def _visible(text: str) -> str:
    return " ".join(TAG_RE.sub("", text).split())


def _fstring_prefix(node: ast.JoinedStr) -> str | None:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:
            break
    prefix = _visible("".join(parts))
    return prefix if len(prefix.replace(" ", "")) >= 3 else None


def _constant_template(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = _visible(value)
    return text if len(text.replace(" ", "")) >= 2 else None


def _scope_matches(scope: tuple[str, ...], wanted: tuple[str, ...] | None) -> bool:
    if wanted is None:
        return True
    return any(part in name for name in scope for part in wanted)


def _templates_in(path: Path, wanted: tuple[str, ...] | None) -> Iterator[tuple[str, int]]:
    tree = _tree(path)
    if wanted is None:
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
                node.value, ast.Constant
            ):
                template = _constant_template(node.value.value)
                if template is not None:
                    yield template, node.lineno
    for node, index in _walk(tree):
        if not isinstance(node, (ast.JoinedStr, ast.Return)):
            continue
        if isinstance(node, ast.JoinedStr):
            template = _fstring_prefix(node)
        elif isinstance(node.value, ast.Constant):
            template = _constant_template(node.value.value)
        else:
            continue
        if template is None:
            continue
        scope = _scope(index, node)
        if scope and _scope_matches(scope, wanted):
            yield template, node.lineno


def _collapse_token_templates(path: Path) -> Iterator[tuple[str, int]]:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == "CollapseTokens":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    yield f"CollapseTokens.{stmt.target.id}", stmt.lineno


def label_template_universe() -> tuple[SourceItem, ...]:
    """Label-template prefixes from the label and edge-label builders, first location each."""

    vis = PACKAGE_ROOT / "visualization"
    first: dict[str, str] = {}
    for name, wanted in LABEL_SOURCES.items():
        path = (vis / name).resolve()
        for template, line in _templates_in(path, wanted):
            first.setdefault(template, f"{_rel(path)}:{line}")
    themes_path = (vis / "themes.py").resolve()
    for template, line in _collapse_token_templates(themes_path):
        first.setdefault(template, f"{_rel(themes_path)}:{line}")
    return tuple(sorted(SourceItem(key, source) for key, source in first.items()))


# ---------------------------------------------------------------------------
# Universe 6: emission sites
# ---------------------------------------------------------------------------


def _is_emission(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in EMIT_NAMES
    return isinstance(func, ast.Attribute) and func.attr in EMIT_ATTRS


def _call_vocabulary(node: ast.Call) -> tuple[str, ...]:
    values: set[str] = set()
    for arg in (*node.args, *(k.value for k in node.keywords)):
        for sub in ast.walk(arg):
            if isinstance(sub, ast.JoinedStr):
                continue
            if isinstance(sub, ast.Constant) and isinstance(sub.value, (str, int, float)):
                values.add(repr(sub.value))
    keywords = sorted(k.arg or "**" for k in node.keywords)
    return (_call_name(node), ",".join(keywords), "|".join(sorted(values)))


def fingerprint(vocabulary: Iterable[tuple[str, ...]]) -> str:
    """Hash a site's emitted vocabulary (a set of call tuples) to 12 base32 characters."""

    payload = json.dumps(sorted(set(vocabulary)), separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return base64.b32encode(digest).decode("ascii")[:12].lower()


def emission_site_universe(files: Iterable[Path] | None = None) -> tuple[EmissionSite, ...]:
    """Every function that emits Graphviz nodes, edges, attributes, subgraphs or legend rows."""

    sites: dict[str, tuple[set[tuple[str, ...]], str]] = {}
    for path in files if files is not None else scanned_files():
        module = _module_name(path)
        index = _index_tree(_tree(path))
        for node in index.nodes:
            if not _is_emission(node):
                continue
            site = f"{module}:{'.'.join(_scope(index, node)) or '<module>'}"
            vocab, source = sites.setdefault(site, (set(), f"{_rel(path)}:{node.lineno}"))
            vocab.add(_call_vocabulary(node))  # type: ignore[arg-type]
    return tuple(
        sorted(EmissionSite(site, fingerprint(vocab), src) for site, (vocab, src) in sites.items())
    )


def encoding_legend_texts() -> list[LegendText]:
    """Encoding-legend line prefixes: ``NOTE_*`` constants and the legend-row builders."""

    path = (PACKAGE_ROOT / "visualization" / "_encoding.py").resolve()
    out: list[LegendText] = []
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id.startswith("NOTE_") and isinstance(node.value, ast.Constant):
                out.append(LegendText("TorchLens encoding", str(node.value.value), "_encoding"))
    wanted = ("_color_legend_rows", "_non_color_legend_rows")
    for node, index in _walk(_tree(path)):
        if not isinstance(node, (ast.JoinedStr, ast.Constant)) or id(node) in index.fstring_parts:
            continue
        scope = _scope(index, node)
        if not scope or scope[-1] not in wanted:
            continue
        if isinstance(node, ast.JoinedStr):
            text = _fstring_prefix(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and " " in node.value:
            text = _visible(node.value)
        else:
            continue
        if text:
            out.append(LegendText("TorchLens encoding", text.strip(), f"_encoding.{scope[-1]}"))
    return out
