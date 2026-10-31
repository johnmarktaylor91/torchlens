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
TOMLPATH = re.compile(rf'(?<=")({ALT})/(?=[\w*])')
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
        '    "draw_backward": "use torchlens.Trace.draw_backward (trace.draw_backward())",',
    ),
    (
        '    "draw_combined": "use torchlens._visualization.draw_combined",',
        '    "draw_combined": "use torchlens.Trace.draw_combined (trace.draw_combined())",',
    ),
    (
        '"use torchlens._visualization.show_model_graph, or trace.draw()"',
        '"use torchlens.Trace.draw (trace.draw())"',
    ),
    (
        '    "show_model_graph": "use torchlens._visualization.show_model_graph",',
        '    "show_model_graph": "use torchlens.Trace.draw (trace.draw())",',
    ),
    (
        '"import torchlens._semantic, then use torchlens._semantic.logit_lens and its siblings'
        ' -- a submodule, not a root name"',
        '"use torchlens.facets for facet functions and torchlens.mechinterp for the logit lens'
        ' -- the semantic package is private"',
    ),
)
#: Files that key a table by package name (layer map, ledgers, baselines): every quoted
#: ``"name"`` or ``"name.module"`` token of a moved package is renamed to the new package.
KEYED_FILES = (
    "torchlens/_architecture.py",
    "tests/test_arch_tail_vocab_lint.py",
    "tests/test_no_string_private_getattr.py",
)
#: Tab-separated baselines keyed by dotted module path at the start of a line (re-sorted).
KEYED_TSV = ("tests/test_arch_spine_layer_baseline.tsv",)
#: Exact-text edits to pins whose expected value or path part names a moved package.
PIN_EDITS: dict[str, tuple[tuple[str, str], ...]] = {
    "tests/test_w051_gate_facade_redirects.py": (
        (
            "mechinterp, semantic and transformer instead of root module rows)",
            "mechinterp and transformer instead of root module rows)",
        ),
        (
            "names.4: exactly mechinterp, semantic and transformer teach at the root.",
            "names.4: exactly mechinterp and transformer teach at the root.",
        ),
        (
            '_SUBPACKAGE_ROWS == ["mechinterp", "semantic", "transformer"]',
            '_SUBPACKAGE_ROWS == ["mechinterp", "transformer"]',
        ),
    ),
    "tests/test_train_mode/test_no_bare_detach.py": (
        ('REPO_ROOT / "torchlens" / "capture"', 'REPO_ROOT / "torchlens" / "_capture"'),
        ('REPO_ROOT / "torchlens" / "fastlog"', 'REPO_ROOT / "torchlens" / "recording"'),
    ),
    "tests/test_capture_outcome_matrix.py": (
        ('path.parent.name == "capture"', 'path.parent.name == "_capture"'),
    ),
    "tests/test_structure_only_capabilities.py": (
        ('path.parent.name == "capture"', 'path.parent.name == "_capture"'),
    ),
    "tests/test_capture_outcome_preview_halt.py": (
        ('TORCHLENS_DIR / "capture" / "outcome.py"', 'TORCHLENS_DIR / "_capture" / "outcome.py"'),
    ),
    "tests/test_observe_kit_labels.py": (
        (
            'TORCHLENS_DIR / "capture" / "_nonfinite_prefix.py"',
            'TORCHLENS_DIR / "_capture" / "_nonfinite_prefix.py"',
        ),
    ),
    "tests/test_r58_capture_cache_rce.py": (
        ('        "user_funcs.py",', '        "_user_funcs.py",'),
    ),
    "tests/test_trace_autoroute_kwarg_forwarding.py": (
        ('PACKAGE_ROOT / "user_funcs.py"', 'PACKAGE_ROOT / "_user_funcs.py"'),
        ("not found in user_funcs.py", "not found in _user_funcs.py"),
    ),
    "tests/test_s5_escaping_pack.py": (
        ('/ "torchlens" / "visualization"', '/ "torchlens" / "_visualization"'),
    ),
    "tests/test_no_legacy_tl_attrs.py": (
        (
            'if "visualization" not in path.relative_to(SOURCE_ROOT).parts:',
            'if "_visualization" not in path.relative_to(SOURCE_ROOT).parts:',
        ),
    ),
    "tests/test_observe_kit_peaks.py": (
        (
            '("report", "export", "visualization", "notebook", "bridge", "observe")',
            '("report", "export", "_visualization", "notebook", "bridge", "observe")',
        ),
    ),
    "tests/test_encoding_size.py": (
        (
            'scopes = ["visualization", "viz", "repgeom"]',
            'scopes = ["_visualization", "viz", "repgeom"]',
        ),
    ),
    "tests/test_api_surface.py": (('    "fastlog",\n', '    "recording",\n'),),
    "tests/test_capture_settlement.py": (
        (r"from \.\.\.capture\.outcome import", r"from \.\.\._capture\.outcome import"),
    ),
    "tests/test_arch_tail_seam_census.py": (
        (
            "from ..semantic.facets import FacetView",
            "from .._semantic.facets import FacetView",
        ),
        ("from ..semantic import FacetView", "from .._semantic import FacetView"),
    ),
    "tests/test_ecosystem_rt_migrate.py": (
        (
            '("from ..capture", "from ..backends", "from ..semantic", "import torch\\n")',
            '("from .._capture", "from ..backends", "from .._semantic", "import torch\\n")',
        ),
    ),
    "docs/migration/v2.0_api_changes.md": (
        (
            "# After\nfrom torchlens.visualization import show_model_graph\n\n"
            "show_model_graph(\n    model,\n    x,\n"
            '    visualization=tl.options.VisualizationOptions(container_path="graph", save_only=True),\n)\n',
            '# After\ntrace = tl.trace(model, x)\ntrace.draw(path="graph", vis_save_only=True)\n',
        ),
        (
            "# After\nfrom torchlens.visualization import draw_backward\n\n"
            "loss = trace[trace.output_layers[0]].out.sum()\ntrace.log_backward(loss)\n"
            'dot = draw_backward(\n    trace, visualization=tl.options.VisualizationOptions(container_path="backward")\n)\n',
            "# After\nloss = trace[trace.output_layers[0]].out.sum()\ntrace.log_backward(loss)\n"
            'dot = trace.draw_backward(path="backward")\n',
        ),
    ),
    "docs/migration/from_bertviz_ecco.md": (
        ("torchlens.semantic.logit_lens", "torchlens._semantic.logit_lens"),
    ),
    "tests/test_arch_spine_surface.py": (
        ("assert len(rows) == 114, (", "assert len(rows) == 111, ("),
        ("(frozen budget 114: ", "(frozen budget 111: "),
        (
            '"the table regenerated, the root holds 114 rows. "',
            '"the table regenerated, the root held 114 rows; round 2 then moved "\n'
            '        "the captured_run, user_funcs and visualization namespaces to private packages "\n'
            '        "(-3), so the root holds 111 rows. "',
        ),
        ("assert int(result.stdout.strip()) >= 112", "assert int(result.stdout.strip()) >= 111"),
    ),
    "tests/test_arch_tail_surface_audit.py": (
        (
            "#: only go down through a decided surface change.\nDEEP_WALK_FLOOR = 975",
            "#: only go down through a decided surface change. Round 2 (module moves) took it\n"
            "#: from 975 to 968: the private visualization (5 names) and semantic (2 names)\n"
            "#: packages no longer count.\nDEEP_WALK_FLOOR = 968",
        ),
    ),
    "tests/support/proofnet/realism_registry.tsv": (
        ("\nfastlog\ttoy\treferenced\tF36-freeze\n", "\nrecording\ttoy\treferenced\tF36-freeze\n"),
    ),
    "tests/producer_parity/test_p4_completeness.py": (
        (
            '("user_funcs.py", "_register_live_tensor_connection")',
            '("_user_funcs.py", "_register_live_tensor_connection")',
        ),
        ('        "user_funcs.py",\n', '        "_user_funcs.py",\n'),
    ),
    "scripts/visual_language/coverage_scan.py": (
        ('PACKAGE_ROOT / "visualization"', 'PACKAGE_ROOT / "_visualization"'),
    ),
    "tests/test_import_hygiene.py": (
        ('    ("fastlog", "DryRun"),\n', '    ("recording", "DryRun"),\n'),
        ('    ("user_funcs", "trace"),\n', ""),
        ('    if facade_name == "fastlog":', '    if facade_name == "recording":'),
        ('    "captured_run": [],\n', ""),
        (
            '    "fastlog": ["dry_run", "recover"], "intervention"',
            '    "recording": ["dry_run", "recover"], "intervention"',
        ),
        (
            '    "types": [], "user_funcs": [],\n    "validation": [], "visualization": [],\n',
            '    "types": [],\n    "validation": [],\n',
        ),
    ),
}
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
    if path.name == "pyproject.toml":
        text = TOMLPATH.sub(lambda m: f"{MAPPING[m.group(1)]}/", text)
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


_KEYED = re.compile(rf"([\"'])({ALT})([./][\w./]*)?\1")


def _rewrite_keyed(text: str) -> str:
    return _KEYED.sub(lambda m: f"{m[1]}{MAPPING[m[2]]}{m[3] or ''}{m[1]}", text)


def _rewrite_tsv(text: str) -> str:
    """Rename the leading module key of each data row, keeping header lines in place."""
    pattern = re.compile(rf"^({ALT})(?=[.\t])", re.M)
    lines = text.splitlines()
    head = [ln for ln in lines if ln.startswith("#") or ln == "module\tcount"]
    rows = [pattern.sub(lambda m: MAPPING[m[1]], ln) for ln in lines if ln not in head]
    return "\n".join(head + sorted(rows)) + ("\n" if text.endswith("\n") else "")


def apply_pins(repo: Path, dry: bool) -> list[Path]:
    """Rewrite the keyed pin tables and the exact-text pin edits; return the files changed."""
    changed: list[Path] = []
    jobs: list[tuple[str, object]] = [(rel, _rewrite_keyed) for rel in KEYED_FILES]
    jobs += [(rel, _rewrite_tsv) for rel in KEYED_TSV]
    for rel, edits in PIN_EDITS.items():
        jobs.append((rel, lambda text, edits=edits: _apply_exact(text, edits)))
    for rel, func in jobs:
        path = repo / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        new = func(text)  # type: ignore[operator]
        if new != text:
            changed.append(path)
            if not dry:
                path.write_text(new, encoding="utf-8")
    return changed


def _apply_exact(text: str, edits: tuple[tuple[str, str], ...]) -> str:
    for old, new in edits:
        text = text.replace(old, new)
    return text


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
            if dry:
                print("would change", rel)
            else:
                path.write_text(new, encoding="utf-8")
    for path in apply_pins(repo, dry):
        changed += 1
        touched.append(path)
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
