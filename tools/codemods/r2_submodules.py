"""Surface ruling surface.6: the untaught public submodules go private.

Usage: ``python tools/codemods/r2_submodules.py [REPO] [--dry-run] [--check]``

A submodule whose name has no underscore and that its parent does not list in ``__all__`` read as
a public import path. The ruling keeps the taught paths public (``torchlens.backends.torch``,
``torchlens.backends.registry``, ``torchlens.intervention.errors``, ``torchlens.intervention.hooks``,
``torchlens.intervention.stochastic``, and the per-framework backend namespaces ``torchlens.backends.jax``,
``mlx``, ``paddle``, ``tf`` and ``tinygrad`` that the backend guides teach) and gives every other one a leading underscore. The parent's
``__getattr__`` refuses the old attribute spelling with the typed ``facade_redirect`` error
(``torchlens/_utils/_private_submodules.py``); this pass does not touch that table.

The pass is re-runnable: a module already moved is skipped, every target spelling starts with an
underscore the patterns do not match, and a second run changes nothing (``--check`` exits 1 if a
run would change anything). Merge another branch, then run it again.

Rewrites, per tracked file under the scanned roots (history files excepted):

- dotted ``torchlens.<parent>.<leaf>`` and alias ``tl.<parent>.<leaf>``, word-bounded (deeper
  paths under a moved package follow, since only the leaf component changes);
- path spellings ``torchlens/<parent>/<leaf>.py`` and ``torchlens/<parent>/<leaf>/`` and their
  package-relative forms ``<parent>/<leaf>.py`` in prose and strings;
- ``from torchlens.<parent> import <leaf>`` and, inside ``torchlens/``, relative imports that
  resolve to a moved module (``from .leaf import x``, ``from ..parent.leaf import x``,
  ``from . import leaf`` binds ``_leaf as leaf``);
- import order and format of the rewritten Python files (``ruff check --select I --fix`` and
  ``ruff format`` on those files only, when ``ruff`` is on the PATH).

Shadowed leaves (the parent exports a function or package of the same name, such as
``torchlens.recording.dry_run``) are rewritten only where the spelling is a module path: an
``import`` statement, a quoted full module path, a deeper dotted path, or a file path. ``tl.recording.dry_run(...)`` stays.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

#: ``parent.leaf`` (relative to ``torchlens``) for every submodule the ruling makes private.
MOVED = (
    "agent.cli",
    "attribution.graph",
    "attribution.onebackward",
    "backends.default_specs",
    "bridge.hf_attention",
    "errors.episode",
    "errors.runnable",
    "intervention.audit",
    "intervention.binding",
    "intervention.bundle",
    "intervention.edge_semantics",
    "intervention.edge_substitution",
    "intervention.handles",
    "intervention.helpers",
    "intervention.injection",
    "intervention.masked_edit",
    "intervention.model_door",
    "intervention.param_substitution",
    "intervention.predicates",
    "intervention.regions",
    "intervention.replay",
    "intervention.rerun",
    "intervention.runtime",
    "intervention.save",
    "intervention.site_keys",
    "intervention.spec",
    "intervention.spec_compat",
    "intervention.sweep",
    "intervention.types",
    "recording.cleanup",
    "recording.exceptions",
    "recording.options",
    "recording.storage_disk",
    "recording.storage_ram",
    "recording.types",
    "validation.backward",
    "validation.consolidated",
    "validation.core",
    "validation.diagnostics",
    "validation.exemptions",
    "validation.invariants",
    "validation.status",
    "viz.batch_summary",
    "viz.node_plots",
)
#: Moved modules whose parent also exports a same-named public object.
SHADOWED = (
    "intervention.compose",
    "intervention.population",
    "receptive_field.rules",
    "recording.dry_run",
    "recording.recover",
    "viz.feature_maps",
)
ALL_MOVED = MOVED + SHADOWED
#: Private leaf names that are not ``_<leaf>`` (``receptive_field/_rules.py`` already exists).
TARGET_LEAF = {"receptive_field.rules": "_builtin_rules"}
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
    "tools/codemods/r2_submodules.py",
    "tools/codemods/r2_modules.py",
    "tests/classics_corpus/models",
    "CHANGELOG.md",
    "MIGRATIONS.md",
    "docs/migration",
    "docs/reference/deprecations.md",
    "torchlens/_utils/_private_submodules.py",
    "torchlens/_io/_pre3_module_paths.py",
    # Demonstrates the old tl.viz.batch_summary spelling refusing typed.
    "notebooks/audit/11_visualization_in_workflow.ipynb",
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


def _split(entry: str) -> tuple[str, str]:
    parent, leaf = entry.split(".")
    return parent, leaf


def _alt(entries: tuple[str, ...]) -> str:
    return "|".join(re.escape(e) for e in sorted(entries, key=len, reverse=True))


def _private_leaf(entry: str) -> str:
    return TARGET_LEAF.get(entry, f"_{_split(entry)[1]}")


def _private(entry: str) -> str:
    return f"{_split(entry)[0]}.{_private_leaf(entry)}"


PLAIN = re.compile(rf"(?<![\w.])(torchlens|tl)\.({_alt(MOVED)})(?!\w)")
SHADOW_DEEP = re.compile(rf"(?<![\w.])(torchlens|tl)\.({_alt(SHADOWED)})(?=\.\w)")
SHADOW_QUOTED = re.compile(rf"(?<=[\"'])(torchlens)\.({_alt(SHADOWED)})(?=[\"'])")
# Raise-site keys (``torchlens.intervention.compose:_fn:Error::1``) name the module, never the export.
SHADOW_SITEKEY = re.compile(rf"(?<![\w.])(torchlens)\.({_alt(SHADOWED)})(?=:\w)")
SHADOW_IMPORT = re.compile(rf"^(\s*(?:from|import)\s+torchlens\.)({_alt(SHADOWED)})(?!\w)", re.M)
_SLASHED = _alt(tuple(e.replace(".", "/") for e in ALL_MOVED))
PATHED_FULL = re.compile(rf"(?<![\w.-])(torchlens/)({_SLASHED})(?=\.py\b|/)")
PATHED_REL = re.compile(rf"(?<![\w/.-])()({_SLASHED})(?=\.py\b|/)")
ABS_BARE = re.compile(r"^(\s*from\s+)torchlens\.(\w+)(\s+import\s+)(\(?)([^\n#]*)", re.M)
REL_FROM = re.compile(r"^(\s*from\s+)(\.+)([\w.]*)(\s+import\s+)(\(?)([^\n#]*)", re.M)


def _rewrite_names(parent: str, names: str) -> str:
    """Rewrite ``leaf`` (or ``leaf as x``) to ``_leaf as leaf`` for moved, unshadowed leaves."""
    out = []
    for chunk in names.split(","):
        bits = chunk.strip().split()
        if bits and f"{parent}.{bits[0]}" in MOVED:
            local = bits[2] if len(bits) == 3 and bits[1] == "as" else bits[0]
            lead = chunk[: len(chunk) - len(chunk.lstrip())]
            trail = chunk[len(chunk.rstrip()) :]
            out.append(f"{lead}{_private_leaf(f'{parent}.{bits[0]}')} as {local}{trail}")
        else:
            out.append(chunk)
    return ",".join(out)


def _abs_bare(match: re.Match[str]) -> str:
    head, parent, imp, paren, names = match.groups()
    return f"{head}torchlens.{parent}{imp}{paren}{_rewrite_names(parent, names)}"


def _remap_parts(parts: list[str]) -> list[str]:
    """Underscore the leaf component of a moved module named by absolute ``parts``."""
    if len(parts) >= 3 and parts[0] == "torchlens" and f"{parts[1]}.{parts[2]}" in ALL_MOVED:
        return [parts[0], parts[1], _private_leaf(f"{parts[1]}.{parts[2]}"), *parts[3:]]
    return parts


def _rel_from(match: re.Match[str], package: list[str]) -> str:
    head, dots, name, imp, paren, names = match.groups()
    up = len(dots) - 1
    if up > len(package):
        return match.group(0)
    base = package[: len(package) - up] if up else list(package)
    tail = name.split(".") if name else []
    old = base + tail
    new = _remap_parts(old)
    if new != old:
        changed = next(i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b)
        if changed < len(base):
            return match.group(0)
        tail = new[len(base) :]
        return f"{head}{dots}{'.'.join(tail)}{imp}{paren}{names}"
    if len(base) == 2 and base[0] == "torchlens" and not tail:
        return f"{head}{dots}{imp}{paren}{_rewrite_names(base[1], names)}"
    if len(old) == 2 and old[0] == "torchlens":
        return f"{head}{dots}{name}{imp}{paren}{_rewrite_names(old[1], names)}"
    return match.group(0)


def rewrite_text(text: str, path: Path, repo: Path) -> str:
    """Return ``text`` with every moved-submodule spelling respelled."""
    text = PLAIN.sub(lambda m: f"{m.group(1)}.{_private(m.group(2))}", text)
    text = SHADOW_DEEP.sub(lambda m: f"{m.group(1)}.{_private(m.group(2))}", text)
    text = SHADOW_QUOTED.sub(lambda m: f"{m.group(1)}.{_private(m.group(2))}", text)
    text = SHADOW_SITEKEY.sub(lambda m: f"{m.group(1)}.{_private(m.group(2))}", text)
    text = SHADOW_IMPORT.sub(lambda m: f"{m.group(1)}{_private(m.group(2))}", text)
    for pattern in (PATHED_FULL, PATHED_REL):
        text = pattern.sub(
            lambda m: f"{m.group(1)}{_private(m.group(2).replace('/', '.')).replace('.', '/')}",
            text,
        )
    if path.suffix == ".py":
        text = ABS_BARE.sub(_abs_bare, text)
        rel = path.relative_to(repo)
        if rel.parts[0] == "torchlens":
            package = list(rel.with_suffix("").parts)
            package = package[:-1]
            text = REL_FROM.sub(lambda m: _rel_from(m, package), text)
    return text


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


def move_modules(repo: Path, dry: bool) -> list[str]:
    done = []
    for entry in ALL_MOVED:
        parent, leaf = _split(entry)
        base = repo / "torchlens" / parent
        for src, dst in (
            (base / leaf, base / _private_leaf(entry)),
            (base / f"{leaf}.py", base / f"{_private_leaf(entry)}.py"),
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
        print("ruff not on PATH: run ruff check --select I --fix and ruff format on the rewrites")
        return
    subprocess.run(
        [ruff, "check", "--select", "I", "--fix", "-q", "--force-exclude", *names],
        cwd=repo,
        check=False,
    )
    subprocess.run([ruff, "format", "-q", "--force-exclude", *names], cwd=repo, check=False)


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    repo = Path(args[0] if args else ".").resolve()
    check = "--check" in argv
    dry = "--dry-run" in argv or check
    moves = move_modules(repo, dry)
    for line in moves:
        print("move", line)
    touched: list[Path] = []
    for path in iter_files(repo):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = rewrite_text(text, path, repo)
        if new != text:
            touched.append(path)
            if dry:
                print("would change", path.relative_to(repo).as_posix())
            else:
                path.write_text(new, encoding="utf-8")
    if not dry:
        tidy(repo, touched)
    print(f"moved {len(moves)} modules, rewrote {len(touched)} files")
    if check and (moves or touched):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
