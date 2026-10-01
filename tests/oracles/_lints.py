"""The five wave-0 static lints (oracles build item 4; D8/D14/D19/D20/D7).

Each lint is static or sub-second, cannot flake, and returns FINDINGS as
data; the gates in ``test_oracle_w0_lints.py`` apply KNOWN-GAP monotone
mechanics (D25) and each ships a plant. The five:

1. cache-key set difference (D14): every option/entry parameter is keyed,
   declared dont-care (with an invariance-witness slot), or a dated
   KNOWN-GAP row. The repo's own 47-field hand-grown key -- written by an
   author who documented this exact failure mode -- still missed
   ``raise_on_nan``/``track_nonfinite`` (live silent-wrong instances at the
   panel's SHA): hand enumeration empirically does not converge.
2. default-truth 3-source (D20): a docstring's ``X (default)`` claim must
   match the signature default, or, where the signature holds a MISSING
   sentinel (a non-authority), the options-dataclass stored constant.
3. positive-control / unasserted-before (D7): a captured ``*_before``
   snapshot that is never read again is a dead measurement channel -- the
   vacuous-postcondition family (the panel's own pickle probe reported
   "False -> False" through a dead channel).
4. denominator 3-root (D8): the public-surface denominator agrees across
   independent roots (static AST, runtime, the declared-ledger constant,
   and the classification baseline) -- ``PUBLIC_SURFACE_SIZE`` green over a
   45%-short set is the purest instance of a defense on the wrong set.
5. numeral census (D19): implemented in ``_censuses.census_numerals``; the
   gate owns baseline mechanics.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from ._censuses import DATA_DIR, OPTIONS_CLASS_NAMES, REPO_ROOT

#: Entry-signature spellings whose cache-key coverage rides another name.
CACHE_KEY_ALIASES = {"save": "save_predicate"}

_DEFAULT_CLAIM_RE = re.compile(r"``([^`]+)``\s*\((?:the\s+)?default\)?")
_PARAM_HEADER_RE = re.compile(r"^(\w+)\s*:")
_BEFORE_NAME_RE = re.compile(r"(^before_|_before$|^before$)")


def cache_config_keys(user_funcs_path: Path | None = None) -> frozenset[str]:
    """Statically parse the capture-cache key's ``cache_config`` dict keys.

    Parameters
    ----------
    user_funcs_path:
        Source file holding the dict (default: shipped ``user_funcs.py``).

    Returns
    -------
    frozenset[str]
        Literal keys of the dict.

    Raises
    ------
    ValueError
        When the dict literal is not found -- the lint must fail loudly if
        the site moves, never report an empty key set as coverage.
    """

    path = user_funcs_path if user_funcs_path is not None else REPO_ROOT / "torchlens/user_funcs.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "cache_config":
                    return frozenset(
                        key.value
                        for key in node.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    )
    raise ValueError(f"cache_config dict literal not found in {path}")


def option_universe() -> frozenset[str]:
    """Return the option/entry-parameter universe the cache key must cover.

    Returns
    -------
    frozenset[str]
        ``Class.field`` rows for every PUBLIC field of the six Options
        groups plus ``trace.<param>`` rows for every ``tl.trace`` parameter.
        Private (underscore) fields are bookkeeping, not user-reachable
        configuration, and are excluded.
    """

    import dataclasses
    import inspect

    import torchlens
    import torchlens.options as options_module

    universe: set[str] = set()
    for class_name in OPTIONS_CLASS_NAMES:
        for field in dataclasses.fields(getattr(options_module, class_name)):
            if not field.name.startswith("_"):
                universe.add(f"{class_name}.{field.name}")
    for param in inspect.signature(torchlens.trace).parameters:
        universe.add(f"trace.{param}")
    return frozenset(universe)


def load_dont_care() -> dict[str, str]:
    """Load the declared dont-care list with its witness slots.

    Returns
    -------
    dict[str, str]
        ``field -> invariance_witness`` (possibly empty slot, funded by
        build item 6b).
    """

    import csv

    with (DATA_DIR / "cache_key_dont_care.tsv").open(newline="") as handle:
        rows = tuple(
            csv.DictReader(
                (line for line in handle if not line.startswith("#")), delimiter="\t", restval=""
            )
        )
    return {row["field"]: row["invariance_witness"] for row in rows}


def cache_key_uncovered() -> frozenset[str]:
    """Lint 1: the burden-inverted cache-key set difference (D14).

    Returns
    -------
    frozenset[str]
        Universe entries that are neither keyed, nor aliased to a key, nor
        declared dont-care. Every returned entry must hold a dated KNOWN-GAP
        row or the gate blocks.
    """

    keys = cache_config_keys()
    dont_care = set(load_dont_care())
    uncovered: set[str] = set()
    for entry in option_universe():
        leaf = entry.split(".", 1)[1]
        covered = (
            leaf in keys
            or CACHE_KEY_ALIASES.get(leaf) in keys
            or leaf in dont_care
            or entry in dont_care
        )
        if not covered:
            uncovered.add(entry)
    return frozenset(uncovered)


@dataclass(frozen=True)
class DefaultClaim:
    """One docstring default claim reconciled against its authority.

    Parameters
    ----------
    param:
        Parameter the claim sits under.
    claimed:
        The docstring's claimed default literal (as written).
    authority:
        The reconciled authority value's repr, or ``"<no-authority>"``.
    agrees:
        Whether claim and authority match.
    """

    param: str
    claimed: str
    authority: str
    agrees: bool


def _authority_default(param_name: str) -> tuple[bool, object]:
    """Resolve the stored-convention default for one trace parameter.

    Parameters
    ----------
    param_name:
        The ``tl.trace`` parameter name.

    Returns
    -------
    tuple[bool, object]
        ``(found, value)``. The signature default wins when it is not the
        MISSING sentinel; otherwise the same-named CaptureOptions field
        default is the stored-convention constant (D20).
    """

    import dataclasses
    import inspect

    import torchlens
    import torchlens.options as options_module

    parameters = inspect.signature(torchlens.trace).parameters
    if param_name in parameters and "MISSING" not in repr(parameters[param_name].default):
        return True, parameters[param_name].default
    for field in dataclasses.fields(options_module.CaptureOptions):
        if field.name == param_name and field.default is not dataclasses.MISSING:
            return True, field.default
    return False, None


def default_truth_findings() -> tuple[DefaultClaim, ...]:
    """Lint 2: reconcile docstring default claims across the three sources.

    Returns
    -------
    tuple[DefaultClaim, ...]
        Every parsed claim with its verdict; the gate red-flags rows where
        ``agrees`` is False (KNOWN-GAP mechanics apply).
    """

    import inspect

    import torchlens

    doc = inspect.getdoc(torchlens.trace) or ""
    parameters = frozenset(inspect.signature(torchlens.trace).parameters)
    claims: list[DefaultClaim] = []
    current_param = ""
    for line in doc.splitlines():
        header = _PARAM_HEADER_RE.match(line.strip())
        if header and header.group(1) in parameters and not line.startswith(" " * 8):
            current_param = header.group(1)
        for match in _DEFAULT_CLAIM_RE.finditer(line):
            if not current_param:
                continue
            claimed = match.group(1)
            found, authority = _authority_default(current_param)
            if not found:
                claims.append(DefaultClaim(current_param, claimed, "<no-authority>", False))
                continue
            try:
                claimed_value = ast.literal_eval(claimed)
            except (SyntaxError, ValueError):
                claimed_value = claimed
            agrees = claimed_value == authority or claimed == repr(authority)
            claims.append(DefaultClaim(current_param, claimed, repr(authority), agrees))
    return tuple(claims)


def unasserted_before_findings(paths: tuple[Path, ...]) -> tuple[str, ...]:
    """Lint 3: find ``*_before`` snapshots that are never read again (D7).

    Parameters
    ----------
    paths:
        Python files to scan (the oracle harness's own files in wave 0).

    Returns
    -------
    tuple[str, ...]
        ``relpath::function::name`` findings.
    """

    findings: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text())
        relative = path.name
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assigned: dict[str, int] = {}
            read: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and _BEFORE_NAME_RE.search(child.id):
                    if isinstance(child.ctx, ast.Store):
                        assigned.setdefault(child.id, child.lineno)
                    elif isinstance(child.ctx, ast.Load):
                        read.add(child.id)
            for name in sorted(set(assigned) - read):
                findings.append(f"{relative}::{node.name}::{name}")
    return tuple(findings)


def static_all_count(init_path: Path | None = None) -> int:
    """Lint 4 root A: count the ``__all__`` literal by static AST parse.

    Parameters
    ----------
    init_path:
        Package ``__init__.py`` (default: the shipped one).

    Returns
    -------
    int
        Number of string entries in the ``__all__`` list literal.

    Raises
    ------
    ValueError
        When the literal is not found.
    """

    path = init_path if init_path is not None else REPO_ROOT / "torchlens/__init__.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return len(node.value.elts)
    raise ValueError(f"__all__ literal not found in {path}")


def declared_ledger_constant() -> int:
    """Lint 4 root B: the declared-gate constant, read statically.

    Returns
    -------
    int
        ``PUBLIC_SURFACE_SIZE`` from tests/test_docs_lockstep_names.py --
        the declared lockstep gate this reachable gate CITES (never forks).

    Raises
    ------
    ValueError
        When the constant is not found.
    """

    text = (REPO_ROOT / "tests/test_docs_lockstep_names.py").read_text()
    match = re.search(r"^PUBLIC_SURFACE_SIZE = (\d+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("PUBLIC_SURFACE_SIZE not found in tests/test_docs_lockstep_names.py")
    return int(match.group(1))


def declared_ledger_length() -> int:
    """Lint 4 root C: the ``TARGET_ALL`` ledger length, read statically.

    Returns
    -------
    int
        Length of the hand ledger in tests/test_api_surface.py.

    Raises
    ------
    ValueError
        When the ledger is not found.
    """

    tree = ast.parse((REPO_ROOT / "tests/test_api_surface.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TARGET_ALL":
                    return len(node.value.elts)
    raise ValueError("TARGET_ALL not found in tests/test_api_surface.py")
