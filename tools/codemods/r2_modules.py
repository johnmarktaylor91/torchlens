"""Round-2 module moves: ``torchlens.fastlog`` becomes ``torchlens.recording`` and five packages go private.

Usage: ``python tools/codemods/r2_modules.py [REPO] [--dry-run] [--check]``

The pass is re-runnable: a package already moved is skipped, every target spelling either starts
with an underscore or is a new leaf the patterns do not match, and a second run changes nothing
(``--check`` exits 1 if a run would change anything). Merge another branch, then run it again.

Moves (``git mv``):

- ``torchlens/fastlog``       -> ``torchlens/recording``      (public: ``tl.recording``)
- ``torchlens/capture``       -> ``torchlens/_capture``
- ``torchlens/user_funcs.py`` -> ``torchlens/_user_funcs.py``
- ``torchlens/captured_run.py`` -> ``torchlens/_captured_run.py``
- ``torchlens/semantic``      -> ``torchlens/_semantic``
- ``torchlens/visualization`` -> ``torchlens/_visualization``

Rewrites, per tracked file under the scanned roots (history files excepted):

- dotted ``torchlens.<old>`` and alias ``tl.<old>``, word-bounded;
- path spellings ``torchlens/<old>`` and package-relative ``<old>/module.py`` in prose and strings;
- relative imports inside ``torchlens/`` that resolve to ``torchlens.<old>`` (``from ..old.x
  import y``; ``from .. import old`` stays bound to its local name);
- the root lazy table and ``__all__`` of ``torchlens/__init__.py`` (the ``fastlog`` row becomes the
  public ``recording`` row; the private packages lose their root rows) and the removed-name
  redirect messages that named a moved package;
- the 2.x module-path refusal set, which gains the old package names;
- import order and format of the rewritten Python files (``ruff check --select I --fix`` and
  ``ruff format`` on those files only, when ``ruff`` is on the PATH).

Names kept public from a now-private package already live at their decided homes (``tl.facets``,
``tl.mechinterp``, ``tl.viz``, ``tl.errors``, ``tl.types``, ``tl.summary``, ``tl.trace``): this pass
re-exports nothing and invents no alias.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

MAPPING = {
    "fastlog": "recording",
    "capture": "_capture",
    "user_funcs": "_user_funcs",
    "captured_run": "_captured_run",
    "semantic": "_semantic",
    "visualization": "_visualization",
}
PRIVATE_ROOT_ROWS = ("captured_run", "user_funcs", "visualization")
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
    "tools/codemods/r2_modules.py",
    "tests/test_name_lockstep_codemod.py",
    "tests/test_name_lockstep_inventory.py",
    "tests/classics_corpus/models",
    "CHANGELOG.md",
    "MIGRATIONS.md",
    "docs/migration",
    "docs/reference/deprecations.md",
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
ALT = "|".join(sorted(MAPPING, key=len, reverse=True))
DOTTED = re.compile(rf"(?<![\w.])(torchlens|tl)\.({ALT})\b")
PATHED = re.compile(rf"(?<![\w/])torchlens/({ALT})(?!\w)")
RELPATH = re.compile(rf"(?<=[\"'\s(])({ALT})/((?:\w+/)*(?:\w+\.py\b|_\w+))")
DOCPATH = re.compile(rf"(?<![\w/.-])({ALT})/((?:\w+/)*\w+\.py)\b")
REL_FROM = re.compile(rf"^(\s*from\s+)(\.+)({ALT})\b", re.M)
REL_BARE = re.compile(r"^(\s*from\s+)(\.+)(\s+import\s+)(\(?)([^\n#]*)", re.M)
ABS_BARE = re.compile(r"^(\s*from\s+)(torchlens)(\s+import\s+)(\(?)([^\n#]*)", re.M)

#: Exact-text edits to ``torchlens/__init__.py`` (each is skipped once applied).
ROOT_EDITS = (
    (
        '    "fastlog": ("torchlens.recording", None),\n',
        '    "recording": ("torchlens.recording", None),\n',
    ),
    ('    "fastlog",\n', '    "recording",\n'),
    (
        '    "draw_backward": "use torchlens._visualization.draw_backward",',
        '    "draw_backward": "use trace.draw_backward()",',
    ),
    (
        '    "draw_combined": "use torchlens._visualization.draw_combined",',
        '    "draw_combined": "use trace.draw_combined()",',
    ),
    (
        '"use torchlens._visualization.show_model_graph, or trace.draw()"',
        '"use trace.draw()"',
    ),
    (
        '    "show_model_graph": "use torchlens._visualization.show_model_graph",',
        '    "show_model_graph": "use trace.draw()",',
    ),
    (
        '"import torchlens._semantic, then use torchlens._semantic.logit_lens and its siblings'
        ' -- a submodule, not a root name"',
        '"use torchlens.facets for facet functions and torchlens.mechinterp for the logit lens'
        ' -- the semantic package is private"',
    ),
)
PRE_3_NAMES = ("capture", "captured_run", "fastlog", "semantic", "user_funcs", "visualization")


def _package_of(path: Path, repo: Path) -> list[str]:
    parts = list(path.relative_to(repo).with_suffix("").parts)
    return parts[:-1]


def _resolves_to_root(path: Path, repo: Path, dots: str) -> bool:
    if dots == "torchlens":
        return True
    package = _package_of(path, repo)
    up = len(dots) - 1
    base = package[: len(package) - up] if up else package
    return base == ["torchlens"]


def _rewrite_bare(match: re.Match[str], path: Path, repo: Path) -> str:
    head, dots, imp, paren, names = match.groups()
    if not _resolves_to_root(path, repo, dots):
        return match.group(0)
    out = []
    changed = False
    for chunk in names.split(","):
        bits = chunk.strip().split()
        if bits and bits[0] in MAPPING:
            new = MAPPING[bits[0]]
            local = bits[2] if len(bits) == 3 and bits[1] == "as" else bits[0]
            lead = chunk[: len(chunk) - len(chunk.lstrip())]
            trail = chunk[len(chunk.rstrip()) :]
            out.append(f"{lead}{new} as {local}{trail}")
            changed = True
        else:
            out.append(chunk)
    if not changed:
        return match.group(0)
    return f"{head}{dots}{imp}{paren}{','.join(out)}"


TL_PRIVATE = re.compile(r"\btl\.(_(?:capture|user_funcs|captured_run|semantic|visualization))\b")
TL_ALIAS_IMPORT = re.compile(r"^import torchlens as tl[ \t]*$", re.M)


def _private_attribute_access(text: str, path: Path) -> str:
    """Replace ``tl._pkg.x`` (a root attribute the private package no longer has) by an import.

    Python files get ``from torchlens import _pkg`` after ``import torchlens as tl`` and use
    ``_pkg.x``; prose and notebooks name the import path ``torchlens._pkg.x``.
    """
    if not TL_PRIVATE.search(text):
        return text
    if path.suffix == ".py":
        used = sorted(set(TL_PRIVATE.findall(text)))
        if TL_ALIAS_IMPORT.search(text) is None:
            return text
        text = TL_PRIVATE.sub(lambda m: m.group(1), text)
        match = TL_ALIAS_IMPORT.search(text)
        assert match is not None
        lines = "".join(f"\nfrom torchlens import {name}" for name in used)
        return text[: match.end()] + lines + text[match.end() :]
    return TL_PRIVATE.sub(lambda m: f"torchlens.{m.group(1)}", text)


def rewrite_text(text: str, path: Path, repo: Path) -> str:
    """Return ``text`` with every moved-package spelling respelled."""
    text = DOTTED.sub(lambda m: f"{m.group(1)}.{MAPPING[m.group(2)]}", text)
    text = _private_attribute_access(text, path)
    text = PATHED.sub(lambda m: f"torchlens/{MAPPING[m.group(1)]}", text)
    if path.suffix == ".py":
        text = RELPATH.sub(lambda m: f"{MAPPING[m.group(1)]}/{m.group(2)}", text)
    if path.suffix in {".md", ".rst", ".txt", ".ipynb", ".py"}:
        text = DOCPATH.sub(lambda m: f"{MAPPING[m.group(1)]}/{m.group(2)}", text)
    text = ABS_BARE.sub(lambda m: _rewrite_bare(m, path, repo), text)
    if path.suffix == ".py" and path.relative_to(repo).parts[0] == "torchlens":
        text = REL_FROM.sub(
            lambda m: (
                f"{m.group(1)}{m.group(2)}{MAPPING[m.group(3)]}"
                if _resolves_to_root(path, repo, m.group(2))
                else m.group(0)
            ),
            text,
        )
        text = REL_BARE.sub(lambda m: _rewrite_bare(m, path, repo), text)
    return text


def edit_root(text: str) -> str:
    """Apply the exact-text edits to the root ``__init__`` and drop the private packages' rows."""
    for old, new in ROOT_EDITS:
        text = text.replace(old, new)
    for name in PRIVATE_ROOT_ROWS:
        text = re.sub(rf'^    "{name}": \("torchlens\._\w+", None\),\n', "", text, flags=re.M)
    return text


def edit_pre3(text: str) -> str:
    """Add the old package names to the 2.x module-path refusal set (sorted, idempotent)."""
    match = re.search(r"_PRE_3_PACKAGES = frozenset\(\n    \{\n(.*?)    \}\n\)", text, re.S)
    if match is None:
        return text
    names = set(re.findall(r'"(\w+)"', match.group(1))) | set(PRE_3_NAMES)
    body = "".join(f'        "{name}",\n' for name in sorted(names))
    return text[: match.start(1)] + body + text[match.end(1) :]


def iter_files(repo: Path):
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--", *ROOTS],
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


def move_packages(repo: Path, dry: bool) -> list[str]:
    done = []
    for old, new in MAPPING.items():
        for src, dst in (
            (repo / "torchlens" / old, repo / "torchlens" / new),
            (repo / "torchlens" / f"{old}.py", repo / "torchlens" / f"{new}.py"),
        ):
            if src.exists() and not dst.exists():
                done.append(f"{src.relative_to(repo)} -> {dst.relative_to(repo)}")
                if not dry:
                    subprocess.run(["git", "-C", str(repo), "mv", str(src), str(dst)], check=True)
    return done


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
    subprocess.run([ruff, "check", "--select", "I", "--fix", "-q", *names], cwd=repo, check=False)
    subprocess.run([ruff, "format", "-q", *names], cwd=repo, check=False)


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    repo = Path(args[0] if args else ".").resolve()
    check = "--check" in argv
    dry = "--dry-run" in argv or check
    moves = move_packages(repo, dry)
    for line in moves:
        print("move", line)
    changed = 0
    touched: list[Path] = []
    for path in iter_files(repo):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = rewrite_text(text, path, repo)
        rel = path.relative_to(repo).as_posix()
        if rel == "torchlens/__init__.py":
            new = edit_root(new)
        if new != text:
            changed += 1
            touched.append(path)
            if not dry:
                path.write_text(new, encoding="utf-8")
    pre3 = repo / "torchlens/_io/_pre3_module_paths.py"
    if pre3.is_file():
        text = pre3.read_text(encoding="utf-8")
        new = edit_pre3(text)
        if new != text:
            changed += 1
            touched.append(pre3)
            if not dry:
                pre3.write_text(new, encoding="utf-8")
    if not dry:
        tidy(repo, touched)
    print(f"moved {len(moves)} packages, rewrote {changed} files")
    if check and (moves or changed):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
