"""The nine generator-input censuses (oracles D9, build item 3).

Every census is a MACHINE-DERIVED denominator: a walk of the live tree (or a
schema table validated against the live tree) whose output is diffed against
a committed baseline by the gates. Nine roots feed the registry generator;
each root is itself gated and planted:

1. reachable-surface walk, module + class layer   (machine walk, _surface)
2. the six Options dataclasses (108 fields)       (machine walk)
3. signature reconciliation (tl.trace params)     (machine walk)
4. state classes / snapshot axes + postconditions (schema table)
5. qualifier lattice                              (schema table)
6. memoization sites                              (machine walk, AST)
7. rendered-artifact leaves                       (schema table, doors resolve)
8. doc + TEST numeral census                      (machine walk)
9. process histories (H-CLEAN/WARM/FAILED/BOTH)   (schema table)

Schema tables are Wave-0 PLUMBING (memo section 13): the table exists, is
validated, and later waves populate/enforce; a machine walk blocks on new
deltas from this package's first merge (D25).
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import inspect
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Numerals below this value are indices/shapes noise, not published
#: quantities; the counted set is disclosed with every published count (D8).
NUMERAL_FLOOR = 10

NUMERAL_COUNTED_SET = (
    "multi-digit numerals (value >= 10, comma/underscore groupings allowed) in "
    "published prose positions: markdown docs (README/CLAUDE/AGENTS/docs/**), "
    "test function/class NAMES and DOCSTRINGS (tests/**, vendored "
    "tests/classics_corpus/models excluded), package docstrings, and raise-statement message literals "
    "(torchlens/**); keyed (corpus, file, numeral), line-position-free"
)

_NUMERAL_RE = re.compile(r"\d[\d,._]*")


@dataclass(frozen=True)
class CensusResult:
    """One census run's output.

    Parameters
    ----------
    root_id:
        The census root id (``CR1``..``CR9``).
    counted_set:
        The counted-set identity published beside every count (D8).
    rows:
        Sorted, deduplicated row keys.
    """

    root_id: str
    counted_set: str
    rows: tuple[tuple[str, ...], ...] | tuple[str, ...]


@dataclass(frozen=True)
class CensusRoot:
    """Descriptor for one of the nine generator-input roots.

    Parameters
    ----------
    root_id:
        Stable id.
    title:
        One-line description.
    kind:
        ``machine_walk`` (live-diffed, new deltas block) or ``schema_table``
        (Wave-0 plumbing, validated + populated by later waves).
    baseline:
        The committed baseline/data filename under ``data/``.
    """

    root_id: str
    title: str
    kind: str
    baseline: str


CENSUS_ROOTS: tuple[CensusRoot, ...] = (
    CensusRoot(
        "CR1",
        "reachable surface (module + class layer)",
        "machine_walk",
        "surface_module_classification.tsv",
    ),
    CensusRoot("CR2", "options dataclass fields", "machine_walk", "options_fields.tsv"),
    CensusRoot("CR3", "entry-signature reconciliation", "machine_walk", "signature_params.tsv"),
    CensusRoot("CR4", "state snapshot axes + postconditions", "schema_table", "state_axes.tsv"),
    CensusRoot("CR5", "qualifier lattice", "schema_table", "qualifier_lattice.tsv"),
    CensusRoot("CR6", "memoization sites", "machine_walk", "memoization_sites.tsv"),
    CensusRoot("CR7", "rendered-artifact leaves", "schema_table", "artifact_leaves.tsv"),
    CensusRoot("CR8", "doc + test numeral census", "machine_walk", "numeral_baseline.tsv"),
    CensusRoot("CR9", "process histories", "schema_table", "history_cells.tsv"),
)

#: The six option groups (108 fields at wave-0 measurement).
OPTIONS_CLASS_NAMES = (
    "CaptureOptions",
    "InterventionOptions",
    "ReplayOptions",
    "SaveOptions",
    "StreamingOptions",
    "VisualizationOptions",
)


def census_options_fields() -> CensusResult:
    """Walk the six Options dataclasses (root CR2).

    Returns
    -------
    CensusResult
        Rows ``ClassName.field_name``.
    """

    import torchlens.options as options_module

    rows = []
    for class_name in OPTIONS_CLASS_NAMES:
        options_class = getattr(options_module, class_name)
        rows += [f"{class_name}.{field.name}" for field in dataclasses.fields(options_class)]
    return CensusResult(
        "CR2",
        "dataclasses.fields of the six torchlens.options groups",
        tuple(sorted(rows)),
    )


def census_signature_params() -> CensusResult:
    """Walk the ``tl.trace`` signature (root CR3).

    Returns
    -------
    CensusResult
        Rows ``trace.<param>=missing|explicit`` -- MISSING-sentinel defaults
        are flagged because the signature is a NON-AUTHORITY for those
        defaults (D20); the stored-convention constant lives on the Options
        dataclass.
    """

    import torchlens

    rows = []
    for name, param in inspect.signature(torchlens.trace).parameters.items():
        sentinel = "missing" if "MISSING" in repr(param.default) else "explicit"
        rows.append(f"trace.{name}={sentinel}")
    return CensusResult(
        "CR3",
        "inspect.signature(torchlens.trace) parameters with default class",
        tuple(sorted(rows)),
    )


def census_memoization_sites(package_root: Path | None = None) -> CensusResult:
    """AST-walk the package for memoization sites (root CR6).

    Covers ``functools.lru_cache`` / ``functools.cache`` decorations plus the
    named capture-cache key construction site. Cache staleness serving
    wrong-config results is a REAL family with two instances at HEAD (facts
    5-6), so a NEW memoization site must be registered consciously.

    Parameters
    ----------
    package_root:
        Package tree to scan (default: the shipped ``torchlens/``).

    Returns
    -------
    CensusResult
        Rows ``relpath::qualname`` (or ``relpath::<assign> name``).
    """

    root = package_root if package_root is not None else REPO_ROOT / "torchlens"
    rows: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(errors="replace")
        if not any(token in text for token in ("lru_cache", "functools.cache", "cache_config")):
            continue
        relative = path.relative_to(root.parent).as_posix()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    name = (
                        target.attr
                        if isinstance(target, ast.Attribute)
                        else (target.id if isinstance(target, ast.Name) else "")
                    )
                    if name in {"lru_cache", "cache"}:
                        rows.append(f"{relative}::{node.name}")
            elif isinstance(node, ast.Assign):
                for assign_target in node.targets:
                    if isinstance(assign_target, ast.Name) and assign_target.id == "cache_config":
                        rows.append(f"{relative}::<assign> cache_config")
    return CensusResult(
        "CR6",
        "functools.lru_cache/cache decorations plus cache_config assignment sites under torchlens/",
        tuple(sorted(set(rows))),
    )


def _markdown_numerals(repo_root: Path) -> set[tuple[str, str, str]]:
    """Collect (corpus, file, numeral) keys from public markdown."""

    keys: set[tuple[str, str, str]] = set()
    files = [
        repo_root / "README.md",
        repo_root / "CLAUDE.md",
        repo_root / "AGENTS.md",
        *sorted((repo_root / "docs").rglob("*.md")),
    ]
    for path in files:
        if not path.exists():
            continue
        relative = path.relative_to(repo_root).as_posix()
        for match in _NUMERAL_RE.finditer(path.read_text(errors="replace")):
            if _numeral_value(match.group(0)) >= NUMERAL_FLOOR:
                keys.add(("docs-md", relative, match.group(0)))
    return keys


def _numeral_value(token: str) -> int:
    """Parse a numeral token's integer magnitude (grouping-insensitive)."""

    digits = re.sub(r"[^\d]", "", token)
    return int(digits) if digits else 0


def _test_numerals(repo_root: Path) -> set[tuple[str, str, str]]:
    """Collect test-NAME and test-DOCSTRING numeral keys (D19)."""

    keys: set[tuple[str, str, str]] = set()
    for path in sorted((repo_root / "tests").rglob("*.py")):
        as_posix = path.relative_to(repo_root).as_posix()
        if as_posix.startswith("tests/classics_corpus/models/"):
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith(("test", "Test")):
                for match in _NUMERAL_RE.finditer(node.name):
                    if _numeral_value(match.group(0)) >= NUMERAL_FLOOR:
                        keys.add(("test-names", as_posix, match.group(0)))
            docstring = ast.get_docstring(node) or ""
            for match in _NUMERAL_RE.finditer(docstring):
                if _numeral_value(match.group(0)) >= NUMERAL_FLOOR:
                    keys.add(("test-docstrings", as_posix, match.group(0)))
    return keys


def _package_numerals(repo_root: Path) -> set[tuple[str, str, str]]:
    """Collect package docstring and raise-message numeral keys."""

    keys: set[tuple[str, str, str]] = set()
    for path in sorted((repo_root / "torchlens").rglob("*.py")):
        as_posix = path.relative_to(repo_root).as_posix()
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node) or ""
                for match in _NUMERAL_RE.finditer(docstring):
                    if _numeral_value(match.group(0)) >= NUMERAL_FLOOR:
                        keys.add(("package-docstrings", as_posix, match.group(0)))
            elif isinstance(node, ast.Raise) and node.exc is not None:
                for literal in ast.walk(node.exc):
                    if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                        for match in _NUMERAL_RE.finditer(literal.value):
                            if _numeral_value(match.group(0)) >= NUMERAL_FLOOR:
                                keys.add(("error-strings", as_posix, match.group(0)))
    return keys


def census_numerals(repo_root: Path | None = None) -> CensusResult:
    """Run the published-numeral census (root CR8, D19).

    A wrong number in a test is where the next engineer forms their model
    (the panel's fact 10: FOUR values for one quantity inside the very test
    that fixed the underlying lesson). Wave 0 lands the census + the dated
    legacy baseline; FactRef enforcement for NEW numerals is staged Wave 2
    (H5).

    Parameters
    ----------
    repo_root:
        Tree to scan (default: this checkout).

    Returns
    -------
    CensusResult
        Rows ``(corpus, file, numeral)``.
    """

    root = repo_root if repo_root is not None else REPO_ROOT
    keys = _markdown_numerals(root) | _test_numerals(root) | _package_numerals(root)
    return CensusResult("CR8", NUMERAL_COUNTED_SET, tuple(sorted(keys)))


def load_schema_table(filename: str, required_columns: tuple[str, ...]) -> tuple[dict, ...]:
    """Load and column-validate one schema table.

    Parameters
    ----------
    filename:
        File under ``data/``.
    required_columns:
        Columns every row must carry non-empty (unless named ``note``).

    Returns
    -------
    tuple[dict, ...]
        Parsed rows.

    Raises
    ------
    ValueError
        On a missing column or an empty required cell.
    """

    with (DATA_DIR / filename).open(newline="") as handle:
        reader = csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter="\t", restval=""
        )
        rows = tuple(reader)
    for row in rows:
        for column in required_columns:
            if column not in row:
                raise ValueError(f"{filename}: missing column {column!r}")
            if column != "note" and not (row[column] or "").strip():
                raise ValueError(f"{filename}: empty {column!r} in row {row!r}")
    return rows
