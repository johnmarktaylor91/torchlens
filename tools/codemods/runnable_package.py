"""Runnable package move: the root runnable modules become the private ``torchlens/_runnable/`` package.

Usage: ``python tools/codemods/runnable_package.py [REPO] [--dry-run] [--check] [--roots A,B,...]
[--with-fast]``

The pass is re-runnable: a module already moved is skipped, every target spelling is one the
patterns do not match, and a second run changes nothing (``--check`` exits 1 if a run would
change anything). Merge another branch, then run it again over the merged tree.

Moves (``git mv``; the files' bytes do not change in the move):

- ``torchlens/_runnable.py``          -> ``torchlens/_runnable/__init__.py`` (the vocabulary)
- ``torchlens/_runnable_<name>.py``   -> ``torchlens/_runnable/<name>.py`` for the eighteen
  internals listed in ``SUBMODULES``
- with ``--with-fast`` also ``torchlens/_fast_<name>.py`` -> ``torchlens/_runnable/fast_<name>.py``

The dotted name ``torchlens._runnable`` keeps naming the vocabulary, so its importers need no edit.

Rewrites, per tracked file under the scanned roots (history files excepted):

- dotted ``torchlens.<old>`` and ``tl.<old>`` become ``torchlens._runnable.<new>``, word-bounded;
- path spellings ``torchlens/<old>.py`` and bare ``<old>.py`` become ``torchlens/_runnable/<new>.py``
  and ``_runnable/<new>.py`` (``_runnable.py`` itself becomes ``_runnable/__init__.py``);
- relative imports (read from the syntax tree, so multi-line blocks are covered): outside the
  package ``from .._runnable_<name> import`` becomes ``from .._runnable.<name> import`` and
  ``from . import _runnable_<name> as x`` becomes ``from ._runnable import <name> as x``; inside
  the package an import written for the old root location is re-pointed (``from ._runnable_<name>``
  becomes ``from .<name>``, ``from ._runnable import`` becomes ``from . import``, and an import of
  any other root module gains one leading dot). An import that already resolves under the new
  layout is left alone, which is what makes the pass idempotent; local names never change
  (``from . import _runnable_state`` becomes ``from . import state as _runnable_state``);
- ``from torchlens import _runnable_<name>`` becomes ``from torchlens._runnable import <name> as ...``;
- keyed tables: the torch-privates license and layer rows of ``torchlens/_architecture.py`` keyed
  by a moved module are deleted (the ``_runnable`` package row covers them), and the
  ``scripts/checks/arch_spine_layer_baseline.tsv`` rows are re-keyed to ``_runnable.<new>`` at the
  same counts and re-sorted;
- import order and format of the rewritten Python files (``ruff check --select I --fix`` and
  ``ruff format`` on those files only, when ``ruff`` is on the PATH).

Bare local names (``_runnable_execution.foo`` after ``from . import _runnable_execution``) and
trace field names that share a module's spelling (the ``_runnable_path_faithfulness`` field) are
never touched: only qualified spellings, import statements and path strings are rewritten.

No shim, no alias and no public name: the old dotted names stop existing (remove-and-rename).
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE = "_runnable"
SUBMODULES = (
    "attestation",
    "call_arguments",
    "call_outputs",
    "execution",
    "input_aliases",
    "input_metadata",
    "input_sites",
    "output_contracts",
    "path_faithfulness",
    "providers",
    "retention",
    "seam",
    "state",
    "state_context",
    "state_live",
    "transaction",
    "verification",
    "witness_contracts",
)
FAST_SUBMODULES = ("run", "live_steer", "live_plans", "live_until", "sparse_state")
ROOTS = (
    "torchlens",
    "tests",
    "docs",
    "examples",
    "notebooks",
    "tools",
    "scripts",
    "benchmarks",
    "README.md",
    "AGENTS.md",
    "LIMITATIONS.md",
    "pyproject.toml",
    "MANIFEST.in",
    ".pre-commit-config.yaml",
    ".github",
    ".project-context",
    "test-tiers.json",
)
EXCLUDES = (
    "tools/codemods/runnable_package.py",
    "tests/classics_corpus/models",
    "CHANGELOG.md",
    "MIGRATIONS.md",
    "torchlens/_io/_pre3_module_paths.py",
)
SUFFIXES = {
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".ipynb",
    ".txt",
    ".tsv",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".cfg",
    ".ini",
    ".in",
    ".html",
    ".csv",
}
#: Files that key a table by root module name (the layer map and the torch-privates license).
KEYED_FILES = ("torchlens/_architecture.py",)
#: Tab-separated baselines keyed by dotted module path at the start of a line (re-sorted).
KEYED_TSV = ("scripts/checks/arch_spine_layer_baseline.tsv",)


def build_mapping(with_fast: bool) -> dict[str, str]:
    """Old root module name -> submodule name inside the package."""
    mapping = {f"{PACKAGE}_{name}": name for name in SUBMODULES}
    if with_fast:
        mapping.update({f"_fast_{name}": f"fast_{name}" for name in FAST_SUBMODULES})
    return mapping


@dataclass
class Spellings:
    """The compiled patterns for one mapping."""

    mapping: dict[str, str]
    dotted: re.Pattern[str] = field(init=False)
    pathed: re.Pattern[str] = field(init=False)
    barepath: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        alt = "|".join(sorted(self.mapping, key=len, reverse=True))
        self.dotted = re.compile(rf"(?<![\w.])(torchlens|tl)\.({alt})\b")
        self.pathed = re.compile(rf"(?<![\w/])torchlens/({alt}|{PACKAGE})\.py\b")
        self.barepath = re.compile(rf"(?<![\w/.-])({alt}|{PACKAGE})\.py\b")

    def new_file(self, old: str) -> str:
        return "__init__" if old == PACKAGE else self.mapping[old]

    def rewrite_text(self, text: str) -> str:
        text = self.dotted.sub(lambda m: f"torchlens.{PACKAGE}.{self.mapping[m[2]]}", text)
        text = self.pathed.sub(lambda m: f"torchlens/{PACKAGE}/{self.new_file(m[1])}.py", text)
        return self.barepath.sub(lambda m: f"{PACKAGE}/{self.new_file(m[1])}.py", text)


# --- relative and absolute import statements (syntax tree) ---------------------------------


@dataclass(frozen=True)
class FileContext:
    """Where one Python file sits, under the new layout and the layout its imports assume."""

    repo: Path
    new_package: tuple[str, ...]
    old_package: tuple[str, ...]
    in_package: bool
    init_names: frozenset[str]


def _file_context(path: Path, repo: Path, init_names: frozenset[str]) -> FileContext:
    parts = path.relative_to(repo).with_suffix("").parts
    package = tuple(parts[:-1])
    in_package = package == ("torchlens", PACKAGE)
    # A file inside the package was a root module before the move: its relative imports were
    # written for the ``torchlens`` package until this pass re-points them.
    old_package = ("torchlens",) if in_package else package
    return FileContext(repo, package, old_package, in_package, init_names)


def _module_exists(repo: Path, dotted: tuple[str, ...]) -> bool:
    base = repo.joinpath(*dotted)
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def _resolves_in_new_layout(node: ast.ImportFrom, ctx: FileContext) -> bool:
    up = node.level - 1
    if up > len(ctx.new_package):
        return False
    base = ctx.new_package[: len(ctx.new_package) - up]
    if node.module is not None:
        return _module_exists(ctx.repo, base + tuple(node.module.split(".")))
    in_init = base == ("torchlens", PACKAGE)
    return all(
        _module_exists(ctx.repo, base + (alias.name,)) or (in_init and alias.name in ctx.init_names)
        for alias in node.names
    )


def _names_text(names: list[tuple[str, str | None]]) -> str:
    return ", ".join(name if asname is None else f"{name} as {asname}" for name, asname in names)


def _statement(indent: str, module: str, names: list[tuple[str, str | None]]) -> str:
    return f"{indent}from {module} import {_names_text(names)}"


def _split_names(
    node: ast.ImportFrom, mapping: dict[str, str]
) -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]]:
    moved = [(mapping[a.name], a.asname or a.name) for a in node.names if a.name in mapping]
    others = [(a.name, a.asname) for a in node.names if a.name not in mapping]
    return moved, others


def _plan_in_package(node: ast.ImportFrom, ctx: FileContext, mapping: dict[str, str]) -> list[str]:
    """New statement texts (without indent) for an import inside the package, or ``[]``."""
    if node.level != 1 or _resolves_in_new_layout(node, ctx):
        return []
    if node.module is None:
        moved, others = _split_names(node, mapping)
        out = [_statement("", ".", moved)] if moved else []
        return out + ([_statement("", "..", others)] if others else [])
    names = [(a.name, a.asname) for a in node.names]
    head, _, rest = node.module.partition(".")
    if head in mapping:
        target = "." + mapping[head] + (f".{rest}" if rest else "")
    elif node.module == PACKAGE:
        target = "."
    else:
        target = ".." + node.module
    return [_statement("", target, names)]


def _plan_outside(node: ast.ImportFrom, ctx: FileContext, mapping: dict[str, str]) -> list[str]:
    """New statement texts for an import outside the package that names a moved module."""
    if node.level == 0:
        if node.module != "torchlens":
            return []
        dots, base = "torchlens", ("torchlens",)
    else:
        up = node.level - 1
        if up >= len(ctx.old_package) or ctx.old_package[0] != "torchlens":
            return []
        dots, base = "." * node.level, ctx.old_package[: len(ctx.old_package) - up]
    if base != ("torchlens",):
        return []
    sep = "." if node.level == 0 else ""
    if node.module is None or node.module == "torchlens":
        moved, others = _split_names(node, mapping)
        if not moved:
            return []
        out = [_statement("", f"{dots}{sep}{PACKAGE}", moved)]
        return out + ([_statement("", dots, others)] if others else [])
    head, _, rest = node.module.partition(".")
    if head not in mapping:
        return []
    target = f"{dots}{PACKAGE}.{mapping[head]}" + (f".{rest}" if rest else "")
    return [_statement("", target, [(a.name, a.asname) for a in node.names])]


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


_HEAD = re.compile(r"from\s+([.\w]+)\s+import\b")


def _edit_for(node: ast.ImportFrom, text: str, start: int, end: int, statements: list[str]):
    """One replacement for ``node``: only the ``from X`` head when the names are unchanged."""
    original_names = [(a.name, a.asname) for a in node.names]
    if len(statements) == 1:
        module = statements[0].split()[1]
        if statements[0] == _statement("", module, original_names):
            head = _HEAD.match(text, start)
            if head is not None and head.end() <= end:
                return head.start(1), head.end(1), module, None
    warning = None
    if "#" in text[start:end]:
        warning = f"line {node.lineno}: comment inside a rewritten import was dropped"
    return start, end, f"\n{' ' * node.col_offset}".join(statements), warning


def rewrite_imports(text: str, ctx: FileContext, mapping: dict[str, str]) -> tuple[str, list[str]]:
    """Re-point the import statements of one Python file; return (text, warnings)."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return text, [f"unparsable, imports not rewritten: {exc}"]
    offsets = _line_offsets(text)
    edits: list[tuple[int, int, str]] = []
    warnings: list[str] = []
    plan = _plan_in_package if ctx.in_package else _plan_outside
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        statements = plan(node, ctx, mapping)
        if not statements:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset  # type: ignore[operator]
        lo, hi, new, warning = _edit_for(node, text, start, end, statements)
        if warning:
            warnings.append(warning)
        edits.append((lo, hi, new))
    for start, end, new in sorted(edits, reverse=True):
        text = text[:start] + new + text[end:]
    return text, warnings


# --- keyed tables ----------------------------------------------------------------------------


def rewrite_keyed(text: str, mapping: dict[str, str]) -> str:
    """Delete the layer-map and license rows keyed by a moved module (the package row covers them)."""
    alt = "|".join(sorted(mapping, key=len, reverse=True))
    row = re.compile(rf'^[ \t]*"({alt})"(?::[ \t]*"[^"\n]*")?,[ \t]*(?:#[^\n]*)?\n', re.M)
    return row.sub("", text)


def rewrite_tsv(text: str, mapping: dict[str, str]) -> str:
    """Re-key the leading module of each data row, keeping header lines in place; re-sort rows."""
    alt = "|".join(sorted(mapping, key=len, reverse=True))
    pattern = re.compile(rf"^({alt})(?=[.\t])", re.M)
    lines = text.splitlines()
    head = [ln for ln in lines if ln.startswith("#") or ln == "module\tcount"]
    rows = [ln for ln in lines if ln not in head]
    renamed = [pattern.sub(lambda m: f"{PACKAGE}.{mapping[m[1]]}", ln) for ln in rows]
    if renamed == rows:
        return text
    return "\n".join(head + sorted(renamed)) + ("\n" if text.endswith("\n") else "")


# --- driver ----------------------------------------------------------------------------------


def move_modules(repo: Path, spell: Spellings, dry: bool) -> list[str]:
    """``git mv`` each root module still at its old path; return the moves made."""
    pairs = [(f"{PACKAGE}.py", f"{PACKAGE}/__init__.py")]
    pairs += [(f"{old}.py", f"{PACKAGE}/{new}.py") for old, new in spell.mapping.items()]
    done = []
    for src_rel, dst_rel in pairs:
        src, dst = repo / "torchlens" / src_rel, repo / "torchlens" / dst_rel
        if src.exists() and not dst.exists():
            done.append(f"torchlens/{src_rel} -> torchlens/{dst_rel}")
            if not dry:
                dst.parent.mkdir(exist_ok=True)
                subprocess.run(["git", "-C", str(repo), "mv", str(src), str(dst)], check=True)
    return done


def iter_files(repo: Path, roots: tuple[str, ...]):
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--", *roots],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for rel in tracked:
        if any(rel == ex or rel.startswith(ex + "/") for ex in EXCLUDES):
            continue
        path = repo / rel
        if path.suffix in SUFFIXES and path.is_file():
            yield path


def _init_names(repo: Path) -> frozenset[str]:
    init = repo / "torchlens" / PACKAGE / "__init__.py"
    if not init.is_file():
        return frozenset()
    names: set[str] = set()
    for node in ast.parse(init.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return frozenset(names)


def rewrite_file(path: Path, repo: Path, spell: Spellings, init_names: frozenset[str]):
    """Return (new text, warnings) for one tracked file."""
    text = path.read_text(encoding="utf-8")
    warnings: list[str] = []
    rel = path.relative_to(repo)
    if path.suffix == ".py":
        text, warnings = rewrite_imports(text, _file_context(path, repo, init_names), spell.mapping)
    text = spell.rewrite_text(text)
    rel_posix = rel.as_posix()
    if rel_posix in KEYED_FILES:
        text = rewrite_keyed(text, spell.mapping)
    if rel_posix in KEYED_TSV:
        text = rewrite_tsv(text, spell.mapping)
    return text, warnings


def tidy(repo: Path, files: list[Path]) -> None:
    """Sort imports and format the rewritten Python files (those files only)."""
    ruff = shutil.which("ruff")
    names = [str(p) for p in files if p.suffix == ".py" and p.is_file()]
    if not names:
        return
    if ruff is None:
        print(
            "ruff not on PATH: run ruff check --select I --fix and ruff format on the rewritten files"
        )
        return
    subprocess.run(
        [ruff, "check", "--select", "I", "--fix", "--force-exclude", "-q", *names],
        cwd=repo,
        check=False,
    )
    subprocess.run([ruff, "format", "--force-exclude", "-q", *names], cwd=repo, check=False)


def _parse_args(argv: list[str]) -> tuple[Path, bool, bool, tuple[str, ...], bool]:
    positional = [a for a in argv[1:] if not a.startswith("--")]
    roots = ROOTS
    for arg in argv[1:]:
        if arg.startswith("--roots="):
            roots = tuple(r for r in arg.split("=", 1)[1].split(",") if r)
    check = "--check" in argv
    dry = "--dry-run" in argv or check
    repo = Path(positional[0] if positional else ".").resolve()
    return repo, check, dry, roots, "--with-fast" in argv


def main(argv: list[str]) -> int:
    repo, check, dry, roots, with_fast = _parse_args(argv)
    spell = Spellings(build_mapping(with_fast))
    moves = move_modules(repo, spell, dry)
    for line in moves:
        print("move", line)
    if dry and moves:
        print("dry run: the rewrite below assumes the moves above have been made")
    init_names = _init_names(repo)
    touched: list[Path] = []
    for path in iter_files(repo, roots):
        try:
            original = path.read_text(encoding="utf-8")
            new, warnings = rewrite_file(path, repo, spell, init_names)
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(repo).as_posix()
        for warning in warnings:
            print(f"warning {rel}: {warning}")
        if new != original:
            touched.append(path)
            if dry:
                print("would change", rel)
            else:
                path.write_text(new, encoding="utf-8")
    if not dry:
        tidy(repo, touched)
    print(f"{len(moves)} moves, {len(touched)} files {'to change' if dry else 'changed'}")
    return 1 if check and (moves or touched) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
