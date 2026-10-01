"""Static governance lint over the golden/oracle test infrastructure.

Four tripwires (b10 R78 round-3), each with a red-capable scanner unit test:

1. **Flag arming** — every read of a golden update/regen/record flag
   (``TORCHLENS_UPDATE_*``, ``TORCHLENS_REGEN_*``, ``TORCHLENS_ORACLE_RECORD_ENV``,
   ``TORCHLENS_ORACLE_ENFORCE``) must arm on the exact value ``"1"``: either an
   inline ``== "1"`` comparison or ``_oracle_env.flag_armed``. Truthy reads
   (``bool(environ.get(...))``) and presence reads (``NAME in os.environ``)
   armed regeneration on ``NAME=0``.
2. **Write-then-skip** — an update-flag branch that writes golden bytes must
   terminate in ``pytest.skip``/``pytest.fail``/``raise`` or ``return True``
   (the documented caller-skips pattern); returning the freshly written
   payload for comparison is the auto-green regen bug (the historical
   backend-parity path).
3. **xfail hygiene** — ``pyproject`` sets ``xfail_strict=true`` globally, so
   every ``pytest.mark.xfail`` must carry an explicit ``reason=`` string.
4. **Golden ledger** — every golden file under the tests tree must be either
   ENV-GOVERNED (resolved through ``tests/_oracle_env.py`` against committed
   ``ENV``/``ENV-<pkg>`` markers) or explicitly LEDGERED with a reason
   (environment-independent semantic golden, frozen input artifact, or a
   named documented residual). No golden family stays silently outside
   governance.
"""

from __future__ import annotations

import ast
import fnmatch
import functools
import re
from pathlib import Path

import pytest
from _oracle_env import GOLDEN_FLAG_PREFIXES, golden_flag_names_for_role

_TESTS_DIR = Path(__file__).resolve().parent

#: Env-var name shapes that arm golden mutation. DERIVED from the single
#: registry in tests/_oracle_env.py (b10 R78 round-4): this set and the root
#: conftest's CI session guard were two hand-maintained registries with
#: different memberships, and TORCHLENS_REFRESH_PRODUCER_LEDGER sat in
#: neither — its truthy-arming auto-green write was invisible to all four
#: scanners here.
GOLDEN_FLAG_NAMES = golden_flag_names_for_role("arming")


def _is_golden_flag(name: str) -> bool:
    """Return whether ``name`` is a golden update/regen/record flag."""

    return name.startswith(GOLDEN_FLAG_PREFIXES) or name in GOLDEN_FLAG_NAMES


# ---------------------------------------------------------------------------
# Shared AST plumbing
# ---------------------------------------------------------------------------


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Map ``id(child)`` -> parent for every node in ``tree``."""

    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _string_constants(tree: ast.AST) -> dict[str, str]:
    """Collect ``NAME = "literal"`` assignments (any scope) for key resolution.

    Scope-blind on purpose: a lint over test modules where the
    ``_UPDATE_ENV = "TORCHLENS_UPDATE_X"`` convention is universal.
    """

    constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                constants[target.id] = value.value
    return constants


def _resolve_key(node: ast.AST, constants: dict[str, str]) -> str | None:
    """Resolve an env-key expression to a string, or ``None``."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _mentions_environ(node: ast.AST) -> bool:
    """Heuristic: the expression reaches through an ``environ`` mapping."""

    return any(
        isinstance(inner, (ast.Name, ast.Attribute))
        and (getattr(inner, "id", None) == "environ" or getattr(inner, "attr", None) == "environ")
        for inner in ast.walk(node)
    )


def _env_read_key(node: ast.AST, constants: dict[str, str]) -> str | None:
    """Return the key of an env READ expression (``get``/``getenv``/subscript)."""

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and node.args:
            if func.attr == "get" and _mentions_environ(func.value):
                return _resolve_key(node.args[0], constants)
            if func.attr == "getenv":
                return _resolve_key(node.args[0], constants)
        if isinstance(func, ast.Name) and func.id == "getenv" and node.args:
            return _resolve_key(node.args[0], constants)
    if isinstance(node, ast.Subscript) and _mentions_environ(node.value):
        return _resolve_key(node.slice, constants)
    return None


def _is_eq_one_guard(parent: ast.AST | None, read: ast.AST) -> bool:
    """Return whether ``read``'s immediate parent compares it ``== "1"``."""

    if not isinstance(parent, ast.Compare) or len(parent.ops) != 1:
        return False
    if not isinstance(parent.ops[0], ast.Eq):
        return False
    others = [side for side in (parent.left, *parent.comparators) if side is not read]
    return any(isinstance(side, ast.Constant) and side.value == "1" for side in others)


# ---------------------------------------------------------------------------
# Scanner 1: flag arming semantics
# ---------------------------------------------------------------------------


def find_unguarded_flag_reads(
    source: str, filename: str = "<snippet>", *, tree: ast.AST | None = None
) -> list[str]:
    """Return descriptions of golden-flag env reads not guarded by ``== "1"``.

    Offending shapes: truthy reads (``bool(environ.get(FLAG))``, bare ``if``
    conditions, assignments), presence tests (``FLAG in os.environ``), and any
    other use whose immediate context is not an ``== "1"`` comparison.
    ``flag_armed(environ, FLAG)`` contains no keyed read and is clean by
    construction.
    """

    tree = ast.parse(source) if tree is None else tree
    constants = _string_constants(tree)
    parents = _parent_map(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and _mentions_environ(node.comparators[0])
        ):
            key = _resolve_key(node.left, constants)
            if key is not None and _is_golden_flag(key):
                violations.append(
                    f"{filename}:{node.lineno}: presence test of golden flag {key!r} "
                    '(arms on NAME=0); compare the read == "1" or use '
                    "_oracle_env.flag_armed"
                )
            continue
        key = _env_read_key(node, constants)
        if key is None or not _is_golden_flag(key):
            continue
        if not _is_eq_one_guard(parents.get(id(node)), node):
            violations.append(
                f"{filename}:{node.lineno}: read of golden flag {key!r} used without "
                'an == "1" comparison (truthy arming; NAME=0 would arm); use '
                '_oracle_env.flag_armed or an inline == "1"'
            )
    return violations


# ---------------------------------------------------------------------------
# Scanner 2: write-then-skip on update branches
# ---------------------------------------------------------------------------


def _flag_bool_names(tree: ast.AST, constants: dict[str, str]) -> dict[str, str]:
    """Map names assigned from armed-flag expressions to their flag.

    Covers ``X = flag_armed(environ, FLAG)`` and ``X = environ.get(FLAG) == "1"``
    at any scope (the ``_REGEN``/``regen`` conventions).
    """

    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        flag = _test_flag(node.value, constants, {})
        if flag is not None:
            names[target.id] = flag
    return names


def _test_flag(test: ast.AST, constants: dict[str, str], flag_bools: dict[str, str]) -> str | None:
    """Return the golden flag an ``if`` test arms on, if any."""

    # A bare truthy read (`if environ.get(FLAG):`) is ALSO an update branch:
    # the b10 R78 round-4 producer-ledger path armed exactly this way, and
    # scanner 2 previously recognized only flag_armed/comparison/bool-name
    # tests, so its auto-green write was invisible here.
    bare_key = _env_read_key(test, constants)
    if bare_key is not None and _is_golden_flag(bare_key):
        return bare_key
    if isinstance(test, ast.BoolOp):
        for value in test.values:
            flag = _test_flag(value, constants, flag_bools)
            if flag is not None:
                return flag
        return None
    if isinstance(test, ast.Name):
        return flag_bools.get(test.id)
    if isinstance(test, ast.Call):
        func = test.func
        func_name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if func_name == "flag_armed" and len(test.args) >= 2:
            key = _resolve_key(test.args[1], constants)
            if key is not None and _is_golden_flag(key):
                return key
        return None
    if isinstance(test, ast.Compare):
        for side in (test.left, *test.comparators):
            key = _env_read_key(side, constants)
            if key is not None and _is_golden_flag(key):
                return key
    return None


def _is_write_call(node: ast.AST) -> bool:
    """Return whether ``node`` writes file bytes (``write_text``/``write_bytes``)."""

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"write_text", "write_bytes"}
    )


def _is_update_terminator(node: ast.AST) -> bool:
    """Return whether ``node`` legally terminates a golden update branch."""

    if isinstance(node, ast.Raise):
        return True
    if isinstance(node, ast.Return):
        return isinstance(node.value, ast.Constant) and node.value.value is True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        func = node.func
        return func.attr in {"skip", "fail"} and getattr(func.value, "id", None) == "pytest"
    return False


def find_autogreen_update_branches(
    source: str, filename: str = "<snippet>", *, tree: ast.AST | None = None
) -> list[str]:
    """Return update-flag branches that write goldens without skip/fail/raise.

    A regen run must NEVER report a verifying green: after writing, the branch
    must ``pytest.skip``/``pytest.fail``/``raise``, or ``return True`` under
    the documented caller-skips pattern. Returning the written payload (the
    historical backend-parity shape) is flagged.
    """

    tree = ast.parse(source) if tree is None else tree
    constants = _string_constants(tree)
    flag_bools = _flag_bool_names(tree, constants)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        flag = _test_flag(node.test, constants, flag_bools)
        if flag is None:
            continue
        body_nodes = [inner for stmt in node.body for inner in ast.walk(stmt)]
        if not any(_is_write_call(inner) for inner in body_nodes):
            continue
        if not any(_is_update_terminator(inner) for inner in body_nodes):
            violations.append(
                f"{filename}:{node.lineno}: update branch for {flag!r} writes golden "
                "bytes but neither skips, fails, raises, nor returns True — a regen "
                "run could report a vacuous green (auto-green regen path)"
            )
    return violations


# ---------------------------------------------------------------------------
# Scanner 3: xfail markers must carry a reason
# ---------------------------------------------------------------------------


def find_bare_xfails(
    source: str, filename: str = "<snippet>", *, tree: ast.AST | None = None
) -> list[str]:
    """Return ``pytest.mark.xfail`` usages without a ``reason=`` string.

    ``xfail_strict=true`` is global (pyproject); a reasonless xfail is opaque
    about what gap it pins and when it should be retired.
    """

    tree = ast.parse(source) if tree is None else tree
    parents = _parent_map(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Attribute)
            and node.attr == "xfail"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
        ):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Call) and parent.func is node:
            if not any(kw.arg == "reason" for kw in parent.keywords):
                violations.append(f"{filename}:{node.lineno}: pytest.mark.xfail without reason=")
        else:
            violations.append(
                f"{filename}:{node.lineno}: bare pytest.mark.xfail (no call, no reason)"
            )
    return violations


# ---------------------------------------------------------------------------
# Golden governance ledger
# ---------------------------------------------------------------------------

#: Directories holding comparison goldens / frozen fixture artifacts. Sidecar
#: markers (ENV*, PROVENANCE) and recorded env-* baselines are excluded from
#: discovery.
_GOLDEN_ROOTS = (
    "golden",
    "surface_oracle/goldens",
    "godobject_oracle/goldens",
    "backend_parity/goldens",
    "capture_oracle/goldens",
    "fixtures/exports",
    # b10 R78-1 round 3: a 32-file corpus lived here OUTSIDE every governance
    # scanner and self-baselined unconditionally (write-then-skip, no flag).
    "snapshots",
    # b10 R78 round 4: the producer-parity ledger corpus lived outside the
    # roots while its refresh flag was registered nowhere.
    "producer_parity/ledger",
)

#: relpath-glob -> (category, reason). Categories:
#:  * ``env-governed``  — resolved through tests/_oracle_env.py (ENV markers,
#:    fail-closed off-canonical, PROVENANCE on update runs);
#:  * ``env-independent`` — pure-structure semantic golden, adjudicated in the
#:    owning test module's docstring;
#:  * ``frozen-input`` — committed input artifact, never regenerated: it IS
#:    the (legacy) format under test, not an oracle output.
GOLDEN_LEDGER: dict[str, tuple[str, str]] = {
    # --- env-governed families -------------------------------------------
    "surface_oracle/goldens/*.json": (
        "env-governed",
        "byte-identity public-surface dumps (TORCHLENS_UPDATE_SURFACE_ORACLE)",
    ),
    "godobject_oracle/goldens/viz_*.gv": (
        "env-governed",
        "raw DOT bytes; family fingerprint extends with the graphviz emitter "
        "version (TORCHLENS_UPDATE_GODOBJECT_VIZ_ORACLE)",
    ),
    "godobject_oracle/goldens/legacy_baseline_cnn_loaded.json": (
        "env-governed",
        "loaded-surface byte dump (TORCHLENS_UPDATE_LEGACY_ARTIFACT_ORACLE)",
    ),
    "godobject_oracle/goldens/state_keysets.json": (
        "env-governed",
        "state-keyset contract (TORCHLENS_UPDATE_STATE_KEYSET_ORACLE)",
    ),
    "golden/viz_render_identity_oracle.json": (
        "env-governed",
        "DOT bytes + pydot structural digests; family fingerprint extends with "
        "graphviz+pydot (TORCHLENS_UPDATE_VIZ_RENDER_ORACLE)",
    ),
    "golden/rank_render_ir_semantics.json": (
        "env-governed",
        "graphviz-emitted/pydot-parsed rank semantics; family fingerprint "
        "extends with graphviz+pydot (TORCHLENS_UPDATE_RANK_RENDER_IR)",
    ),
    # --- environment-independent semantic goldens -------------------------
    "golden/selector_semantics_matrix.json": (
        "env-independent",
        "selector BEHAVIOR matrix: pure-structure JSON over torchlens-owned "
        "label vocabulary; a torch-driven label change is a real behavior "
        "change this oracle must surface (see module docstring)",
    ),
    "backend_parity/goldens/*.json": (
        "env-independent",
        "metadata projections of torchlens-owned labels/edges/field orders; "
        "torch-version decomposition drift IS the parity break the gate exists "
        "for (see module docstring)",
    ),
    "capture_oracle/goldens/*.json": (
        "env-independent",
        "self-governed: subprocess generation plus a recorded torch_version "
        "gate that skips enforcement off the recording torch (b10 R78-7)",
    ),
    "fixtures/exports/*.json": (
        "env-independent",
        "normalized structural export contracts (viewer schemas, torchlens "
        "label vocabulary; see tests/test_exports.py docstring)",
    ),
    "snapshots/bundle_diff_clean_vs_zero_relu.svg": (
        "env-governed",
        "SVG rendered by the `dot` C BINARY (not the python wrapper); the "
        "emitting renderer version is recorded in snapshots/ENV-graphviz-dot "
        "and named in every divergence verdict (b10 R78-3/R78-4); pixel "
        "similarity exonerates benign drift, never converts a regression to "
        "a skip",
    ),
    "snapshots/module_containment/*.json": (
        "env-independent",
        "structural module-containment metadata over torchlens-owned labels; "
        "torch-spelling variance is pinned byte-exactly per variant via the "
        "committed TORCH_VARIANT_FIXTURES registry, and generation is "
        "flag-gated (TORCHLENS_UPDATE_MODULE_CONTAINMENT + reason + "
        "provenance; b10 R78-1 round 3)",
    ),
    "producer_parity/ledger/*.json": (
        "env-independent",
        "machine-independent producer/consumer inventories (repo-relative "
        "paths, line-free site counts, external callers collapsed); compared "
        "never written on normal runs, refresh is flag+reason gated "
        "(TORCHLENS_REFRESH_PRODUCER_LEDGER; b10 R78 round 4)",
    ),
    # --- frozen input artifacts -------------------------------------------
    "golden/io_v3_sample.tlspec/**": (
        "frozen-input",
        "committed v3 artifact exercising the load path; never regenerated",
    ),
    "godobject_oracle/goldens/legacy_baseline_cnn.tlspec/**": (
        "frozen-input",
        "pre-columnar baseline artifact (tree db2bc7a5); it IS the old format",
    ),
    "godobject_oracle/goldens/legacy_baseline_cnn_runnable.tlspec/**": (
        "frozen-input",
        "pre-columnar runnable baseline artifact; it IS the old format",
    ),
}


def _discover_golden_files() -> list[str]:
    """Return every discovered golden file as a tests/-relative posix path."""

    found: list[str] = []
    for root in _GOLDEN_ROOTS:
        base = _TESTS_DIR / root
        assert base.is_dir(), f"golden root vanished: {base}"
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(_TESTS_DIR).as_posix()
            parts = path.relative_to(base).as_posix().split("/")
            # Sidecar markers are excluded at ANY depth: nested golden dirs
            # (snapshots/module_containment) grow their own PROVENANCE files.
            if any(part.startswith(("ENV", "PROVENANCE", "env-")) for part in parts):
                continue
            if "__pycache__" in parts:
                continue
            found.append(rel)
    return found


def _ledger_entry(rel: str) -> tuple[str, tuple[str, str]] | None:
    """Return the (pattern, entry) covering ``rel``, or ``None``."""

    for pattern, entry in GOLDEN_LEDGER.items():
        if fnmatch.fnmatch(rel, pattern):
            return pattern, entry
        if pattern.endswith("/**") and (
            rel == pattern[: -len("/**")] or rel.startswith(pattern[: -len("/**")] + "/")
        ):
            return pattern, entry
    return None


@pytest.mark.smoke
def test_every_golden_file_is_governed_or_ledgered() -> None:
    """No golden family remains silently outside governance (b10 R78 round-3)."""

    unledgered = [rel for rel in _discover_golden_files() if _ledger_entry(rel) is None]
    assert not unledgered, (
        "golden files with no governance adjudication — route each through "
        "tests/_oracle_env.py (env-sensitive bytes) or add a ledger entry with "
        f"a reason (environment-independent / frozen input): {unledgered}"
    )


@pytest.mark.smoke
def test_ledger_has_no_dead_entries() -> None:
    """Every ledger pattern matches at least one committed file."""

    discovered = _discover_golden_files()
    dead = [
        pattern
        for pattern in GOLDEN_LEDGER
        if not any(
            _ledger_entry(rel) is not None and _ledger_entry(rel)[0] == pattern
            for rel in discovered
        )
    ]
    assert not dead, f"ledger entries matching no committed golden: {dead}"


#: Required marker FILES per env-governed directory (b10 R78-5 round 3: only
#: the base ENV was asserted, so deleting a load-bearing family marker like
#: ENV-pydot silently moved both viz byte families off-canonical — a CI skip
#: with no test noticing). A new env-governed family must declare its marker
#: set here; the reasons in GOLDEN_LEDGER name which packages key each family.
_ENV_GOVERNED_REQUIRED_MARKERS: dict[str, tuple[str, ...]] = {
    "golden": ("ENV", "ENV-graphviz", "ENV-pydot"),
    "godobject_oracle/goldens": ("ENV", "ENV-graphviz"),
    "surface_oracle/goldens": ("ENV",),
    # The bundle-diff SVG family is keyed on the dot C BINARY, which no
    # python-package marker can express; it carries no base ENV because it
    # does not resolve through resolve_env_golden (its two-layer byte+pixel
    # scheme lives in test_bundle_diff_renderer.py).
    "snapshots": ("ENV-graphviz-dot",),
}


@pytest.mark.smoke
def test_env_governed_ledger_dirs_carry_env_markers() -> None:
    """Every env-governed dir carries ALL of its declared marker files."""

    governed_dirs = {
        (_TESTS_DIR / pattern).parent.relative_to(_TESTS_DIR).as_posix()
        for pattern, (category, _) in GOLDEN_LEDGER.items()
        if category == "env-governed"
    }
    undeclared = sorted(governed_dirs - set(_ENV_GOVERNED_REQUIRED_MARKERS))
    assert not undeclared, (
        "env-governed goldens dirs with no declared marker set — add each to "
        f"_ENV_GOVERNED_REQUIRED_MARKERS with its family's markers: {undeclared}"
    )
    missing = [
        f"{directory}/{marker}"
        for directory, markers in sorted(_ENV_GOVERNED_REQUIRED_MARKERS.items())
        if directory in governed_dirs
        for marker in markers
        if not (_TESTS_DIR / directory / marker).exists()
    ]
    assert not missing, (
        "missing env-governed family markers (deleting one silently moves the "
        f"family off-canonical — b10 R78-5): {missing}"
    )


#: Emitter-package usage per env-keyed family (b10 R78-2 round 5): the
#: packages each family passes to ``resolve_env_golden`` / the update path.
#: The marker DECLARATIONS in ``_ENV_GOVERNED_REQUIRED_MARKERS`` and the
#: fingerprint EXTENSIONS the code actually keys on were previously untied:
#: adding one emitter to a family (or dropping a marker row) silently moved
#: the family off-canonical — CI skip, marker test still green. Each row is
#: (family test file, goldens dir as declared in the marker table, packages).
_EMITTER_PACKAGE_CENSUS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("test_viz_render_identity_oracle.py", "golden", ("graphviz", "pydot")),
    ("test_rank_render_ir_semantic_goldens.py", "golden", ("graphviz", "pydot")),
    ("godobject_oracle/test_viz_identity.py", "godobject_oracle/goldens", ("graphviz",)),
    ("godobject_oracle/test_legacy_artifact.py", "godobject_oracle/goldens", ()),
    ("test_state_keyset_contract.py", "godobject_oracle/goldens", ()),
    ("surface_oracle/test_surface_oracle.py", "surface_oracle/goldens", ()),
)

_EMITTER_TUPLE_RE = re.compile(r"^_EMITTER_PACKAGES\s*=\s*\(([^)]*)\)", re.MULTILINE)


def _declared_emitter_packages(source: str) -> tuple[str, ...]:
    """Parse a family file's ``_EMITTER_PACKAGES`` literal ("" if absent)."""

    match = _EMITTER_TUPLE_RE.search(source)
    if match is None:
        return ()
    return tuple(part.strip().strip("'\"") for part in match.group(1).split(",") if part.strip())


@pytest.mark.smoke
def test_emitter_packages_match_declared_env_markers() -> None:
    """Family emitter tuples and the marker table agree, both directions."""

    sources = dict(_test_texts())
    per_dir: dict[str, set[str]] = {}
    for relpath, directory, packages in _EMITTER_PACKAGE_CENSUS:
        source = sources.get(relpath)
        assert source is not None, f"emitter census row for missing file {relpath}"
        declared = _declared_emitter_packages(source)
        assert declared == packages, (
            f"{relpath} passes emitter packages {declared} but the census says "
            f"{packages} — update _EMITTER_PACKAGE_CENSUS AND the family's "
            "ENV-<pkg> markers together (b10 R78-2)"
        )
        per_dir.setdefault(directory, set()).update(packages)
    for directory, packages in per_dir.items():
        markers = _ENV_GOVERNED_REQUIRED_MARKERS.get(directory)
        assert markers is not None, f"emitter census names undeclared dir {directory}"
        marker_packages = {m.removeprefix("ENV-") for m in markers if m != "ENV"}
        assert marker_packages == packages, (
            f"{directory}: declared markers key on {sorted(marker_packages)} but "
            f"the code keys on {sorted(packages)} — a mismatch moves the family "
            "off-canonical with the marker test still green (b10 R78-2)"
        )


#: Golden GENERATION mode per update/regen flag (b10 R78-1 round 5, the
#: wrap-state guard census). "subprocess" families construct and capture in a
#: fresh ``_worker.py`` interpreter, so their bytes are pre-wrap by
#: construction; "in-process" families MUST call
#: ``guard_wrap_state_for_golden_update`` before generating, else a
#: mid-session regeneration freezes wrap-state-dependent bytes (SF-53).
#: ``test_module_containment_equality.py`` shipped without the guard while
#: every sibling carried it — this census makes the omission structural.
_GENERATION_WRAP_GUARD_CENSUS: dict[str, tuple[str, str]] = {
    "TORCHLENS_REGEN_EXPORT_GOLDENS": ("test_exports.py", "in-process"),
    "TORCHLENS_UPDATE_BACKEND_PARITY": (
        "backend_parity/test_torch_parity_gates.py",
        "in-process",
    ),
    "TORCHLENS_UPDATE_CAPTURE_ORACLE": (
        "capture_oracle/test_capture_oracle.py",
        "subprocess",
    ),
    "TORCHLENS_UPDATE_GODOBJECT_VIZ_ORACLE": (
        "godobject_oracle/test_viz_identity.py",
        "subprocess",
    ),
    "TORCHLENS_UPDATE_LEGACY_ARTIFACT_ORACLE": (
        "godobject_oracle/test_legacy_artifact.py",
        "in-process",
    ),
    "TORCHLENS_UPDATE_MODULE_CONTAINMENT": (
        "test_module_containment_equality.py",
        "in-process",
    ),
    "TORCHLENS_UPDATE_RANK_RENDER_IR": (
        "test_rank_render_ir_semantic_goldens.py",
        "in-process",
    ),
    "TORCHLENS_UPDATE_SELECTOR_MATRIX": ("test_selector_semantics_matrix.py", "in-process"),
    "TORCHLENS_UPDATE_STATE_KEYSET_ORACLE": ("test_state_keyset_contract.py", "in-process"),
    "TORCHLENS_UPDATE_SURFACE_ORACLE": ("surface_oracle/test_surface_oracle.py", "subprocess"),
    "TORCHLENS_UPDATE_VIZ_RENDER_ORACLE": ("test_viz_render_identity_oracle.py", "in-process"),
    "TORCHLENS_UPDATE_WRITER_CONTRACT": ("test_tlspec_envelope_contract.py", "in-process"),
}

#: Flag-shaped literals used only as planted fixtures by governance/guard
#: red-capability tests — never a real golden family. Additions here need the
#: same scrutiny as a census row: a REAL family hidden in this set would
#: escape the wrap-guard census entirely.
_PROBE_ONLY_FLAGS = frozenset(
    {
        "TORCHLENS_UPDATE_A",
        "TORCHLENS_UPDATE_B",
        "TORCHLENS_UPDATE_X",
        "TORCHLENS_UPDATE_WRAP_GUARD_PROBE_GREEN",
        "TORCHLENS_UPDATE_WRAP_GUARD_PROBE_RED",
    }
)

_FLAG_LITERAL_RE = re.compile(r"TORCHLENS_(?:UPDATE|REGEN)_[A-Z0-9_]+")


def _wrap_guard_census_violations(
    census: dict[str, tuple[str, str]], sources: dict[str, str]
) -> list[str]:
    """Return census violations given ``{relpath: source}`` family sources."""

    violations: list[str] = []
    for flag, (relpath, mode) in sorted(census.items()):
        source = sources.get(relpath)
        if source is None:
            violations.append(f"{flag}: censused file {relpath} does not exist")
            continue
        if mode == "in-process":
            if "guard_wrap_state_for_golden_update(" not in source:
                violations.append(
                    f"{flag}: {relpath} generates goldens in-process but never "
                    "calls guard_wrap_state_for_golden_update (SF-53)"
                )
        elif mode == "subprocess":
            if "_worker" not in source:
                violations.append(
                    f"{flag}: {relpath} is censused subprocess-isolated but no "
                    "longer references a _worker module"
                )
        else:  # pragma: no cover - census typo guard
            violations.append(f"{flag}: unknown generation mode {mode!r}")
    return violations


@pytest.mark.smoke
def test_golden_generators_carry_wrap_state_guard() -> None:
    """Every golden generator family is censused; in-process ones call the guard."""

    sources = dict(_test_texts())
    discovered: set[str] = set()
    for _relpath, source in sources.items():
        discovered.update(_FLAG_LITERAL_RE.findall(source))
    unclassified = sorted(discovered - set(_GENERATION_WRAP_GUARD_CENSUS) - _PROBE_ONLY_FLAGS)
    assert not unclassified, (
        "update/regen flags with no _GENERATION_WRAP_GUARD_CENSUS row — declare "
        "each new golden family's generation mode (in-process families must call "
        f"guard_wrap_state_for_golden_update): {unclassified}"
    )
    violations = _wrap_guard_census_violations(_GENERATION_WRAP_GUARD_CENSUS, sources)
    assert not violations, "\n".join(violations)


@pytest.mark.smoke
def test_wrap_guard_census_is_red_capable() -> None:
    """The census check fails on an in-process generator lacking the guard."""

    census = {"TORCHLENS_UPDATE_X": ("planted.py", "in-process")}
    unguarded = {"planted.py": "def regen():\n    write_golden()\n"}
    assert _wrap_guard_census_violations(census, unguarded)
    guarded = {"planted.py": "guard_wrap_state_for_golden_update(_UPDATE_ENV)\n"}
    assert not _wrap_guard_census_violations(census, guarded)
    missing_worker = {
        "TORCHLENS_UPDATE_X": ("planted.py", "subprocess"),
    }
    assert _wrap_guard_census_violations(missing_worker, unguarded)


# ---------------------------------------------------------------------------
# Red-capability unit tests for the scanners (planted defects must be caught)
# ---------------------------------------------------------------------------

_PLANTED_TRUTHY_FLAG = """
import os
_UPDATE_ENV = "TORCHLENS_UPDATE_SELECTOR_MATRIX"
_REGEN = bool(os.environ.get(_UPDATE_ENV))
"""

_PLANTED_PRESENCE_FLAG = """
import os
_UPDATE_ENV = "TORCHLENS_UPDATE_VIZ_RENDER_ORACLE"
if _UPDATE_ENV in __import__("os").environ:
    pass
"""

_PLANTED_BARE_IF_FLAG = """
import os
if os.environ.get("TORCHLENS_REGEN_EXPORT_GOLDENS"):
    pass
"""

_GUARDED_FLAG = """
import os
_UPDATE_ENV = "TORCHLENS_UPDATE_SELECTOR_MATRIX"
_REGEN = os.environ.get(_UPDATE_ENV) == "1"
if os.environ.get("TORCHLENS_REGEN_EXPORT_GOLDENS") == "1":
    pass
armed = flag_armed(os.environ, _UPDATE_ENV)
other = os.environ.get("TORCHLENS_CACHE_DIR")
"""


@pytest.mark.smoke
def test_flag_scanner_catches_pre_fix_selector_matrix_pattern() -> None:
    """The exact pre-fix truthy-armed pattern is a violation (red-capable)."""

    violations = find_unguarded_flag_reads(_PLANTED_TRUTHY_FLAG)
    assert len(violations) == 1
    assert "TORCHLENS_UPDATE_SELECTOR_MATRIX" in violations[0]


@pytest.mark.smoke
def test_flag_scanner_catches_presence_and_bare_if_reads() -> None:
    """Presence tests and bare truthy `if` reads are violations."""

    presence = find_unguarded_flag_reads(_PLANTED_PRESENCE_FLAG)
    assert len(presence) == 1 and "presence test" in presence[0]
    bare = find_unguarded_flag_reads(_PLANTED_BARE_IF_FLAG)
    assert len(bare) == 1 and "TORCHLENS_REGEN_EXPORT_GOLDENS" in bare[0]


@pytest.mark.smoke
def test_flag_scanner_accepts_guarded_reads() -> None:
    """== "1" comparisons, flag_armed, and non-flag reads are clean."""

    assert find_unguarded_flag_reads(_GUARDED_FLAG) == []


_PLANTED_AUTOGREEN = """
import json, os
_UPDATE_ENV = "TORCHLENS_UPDATE_BACKEND_PARITY"
def _read_or_update_golden(path, projection):
    payload = _golden_payload(projection)
    if os.environ.get(_UPDATE_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\\n", encoding="utf-8")
        return payload
    return json.loads(path.read_text(encoding="utf-8"))
"""

_FIXED_WRITE_THEN_SKIP = """
import json, os
import pytest
_UPDATE_ENV = "TORCHLENS_UPDATE_BACKEND_PARITY"
def _assert_golden(path, projection):
    if flag_armed(os.environ, _UPDATE_ENV):
        path.write_text(json.dumps(projection) + "\\n")
        return True
    if os.environ.get("TORCHLENS_UPDATE_X") == "1":
        path.write_text("x")
        pytest.skip("updated; re-run to verify")
    return False
"""


@pytest.mark.smoke
def test_autogreen_scanner_catches_pre_fix_backend_parity_pattern() -> None:
    """The historical write-then-compare-to-self regen path is caught."""

    violations = find_autogreen_update_branches(_PLANTED_AUTOGREEN)
    assert len(violations) == 1
    assert "TORCHLENS_UPDATE_BACKEND_PARITY" in violations[0]


@pytest.mark.smoke
def test_autogreen_scanner_accepts_write_then_skip_and_return_true() -> None:
    """pytest.skip and the caller-skips `return True` patterns are clean."""

    assert find_autogreen_update_branches(_FIXED_WRITE_THEN_SKIP) == []


_PLANTED_PRODUCER_LEDGER_REFRESH = """
import os
_REFRESH_ENV = "TORCHLENS_REFRESH_PRODUCER_LEDGER"
def assert_artifact_current(path, payload):
    if os.environ.get(_REFRESH_ENV):
        path.write_text(payload, encoding="utf-8")
        return
    assert path.read_text(encoding="utf-8") == payload
"""


@pytest.mark.smoke
def test_scanners_catch_pre_fix_producer_ledger_refresh_pattern() -> None:
    """The b10 R78 round-4 producer-ledger shape now trips BOTH scanners.

    The pre-fix ``tests/producer_parity/test_ledger.py`` refresh branch
    truthy-armed (``TORCHLENS_REFRESH_PRODUCER_LEDGER=0`` overwrote the
    tracked corpus) AND returned green having replaced its own evidence —
    defect classes 1 and 2 of this lint — yet was invisible because the flag
    was registered nowhere. With the flag in the derived registry, both
    scanners fire on the exact pre-fix shape, and the prefilter token now
    reaches the file.
    """

    truthy = find_unguarded_flag_reads(_PLANTED_PRODUCER_LEDGER_REFRESH)
    assert len(truthy) == 1 and "TORCHLENS_REFRESH_PRODUCER_LEDGER" in truthy[0]
    autogreen = find_autogreen_update_branches(_PLANTED_PRODUCER_LEDGER_REFRESH)
    assert len(autogreen) == 1 and "TORCHLENS_REFRESH_PRODUCER_LEDGER" in autogreen[0]
    assert any(token in _PLANTED_PRODUCER_LEDGER_REFRESH for token in _FLAG_TOKENS)


@pytest.mark.smoke
def test_flag_registry_roles_cover_both_governance_layers() -> None:
    """The single registry feeds both layers with the intended memberships.

    The arming lint must see the refresh/record/enforce flags; the CI guard
    must refuse every mutating flag (including the retired selector-matrix
    name) but never the enforce declaration CI's canonical row itself sets.
    """

    from _oracle_env import golden_mutation_flags_armed_under_ci

    assert "TORCHLENS_REFRESH_PRODUCER_LEDGER" in GOLDEN_FLAG_NAMES
    assert "TORCHLENS_ORACLE_ENFORCE" in GOLDEN_FLAG_NAMES
    ci_env = {
        "CI": "true",
        "TORCHLENS_REFRESH_PRODUCER_LEDGER": "0",
        "TL_SELECTOR_MATRIX_REGEN": "1",
        "TORCHLENS_ORACLE_ENFORCE": "1",
        "TORCHLENS_UPDATE_SURFACE_ORACLE": "1",
    }
    armed = golden_mutation_flags_armed_under_ci(ci_env)
    assert "TORCHLENS_REFRESH_PRODUCER_LEDGER" in armed, "presence is fail-closed under CI"
    assert "TL_SELECTOR_MATRIX_REGEN" in armed
    assert "TORCHLENS_UPDATE_SURFACE_ORACLE" in armed
    assert "TORCHLENS_ORACLE_ENFORCE" not in armed, "CI's canonical row sets ENFORCE itself"


_PLANTED_BARE_XFAIL = """
import pytest
@pytest.mark.xfail(strict=True)
def test_a():
    pass

@pytest.mark.xfail
def test_b():
    pass
"""

_REASONED_XFAIL = """
import pytest
@pytest.mark.xfail(strict=True, reason="documented residual: see module docstring")
def test_a():
    pass
"""


@pytest.mark.smoke
def test_xfail_scanner_catches_reasonless_markers() -> None:
    """Reasonless and bare xfail markers are violations (red-capable)."""

    violations = find_bare_xfails(_PLANTED_BARE_XFAIL)
    assert len(violations) == 2
    assert find_bare_xfails(_REASONED_XFAIL) == []


# ---------------------------------------------------------------------------
# Repo-wide enforcement
# ---------------------------------------------------------------------------


_GOVERNED_IDIOM_NAMES = frozenset(
    {"flag_armed", "require_env_golden", "require_update_reason", "write_provenance"}
)


def find_unflagged_baseline_writes(
    source: str, filename: str = "<snippet>", *, tree: ast.AST | None = None
) -> list[str]:
    """Return write-then-skip-on-missing-baseline sites outside the flag system.

    b10 R78-2 (round 3): the four repo-wide governance scanners are prefiltered
    on golden-flag tokens, so an UNFLAGGED self-baseliner — ``if not
    path.exists(): path.write_text(...); pytest.skip(...)`` — sat outside all
    four tripwires simultaneously (the tests/snapshots corpus). This scanner is
    flag-INDEPENDENT: it matches the exact silent-self-baseline shape (an
    ``if`` whose test consults ``.exists()`` and whose body both writes bytes
    and skips) and exempts only bodies that route through the governed idioms
    (``flag_armed`` / ``require_env_golden`` / ``require_update_reason`` /
    ``write_provenance``).
    """

    tree = ast.parse(source) if tree is None else tree
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_consults_exists = any(
            isinstance(sub, ast.Attribute) and sub.attr == "exists" for sub in ast.walk(node.test)
        )
        if not test_consults_exists:
            continue
        body_nodes = [sub for statement in node.body for sub in ast.walk(statement)]
        writes = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"write_text", "write_bytes"}
            for sub in body_nodes
        )
        skips = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "skip"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "pytest"
            for sub in body_nodes
        )
        governed = any(
            (isinstance(sub, ast.Name) and sub.id in _GOVERNED_IDIOM_NAMES)
            or (isinstance(sub, ast.Attribute) and sub.attr in _GOVERNED_IDIOM_NAMES)
            for sub in body_nodes
        )
        if writes and skips and not governed:
            violations.append(
                f"{filename}:{node.lineno}: unflagged write-then-skip self-baseline "
                "(missing-golden branch writes bytes and skips with no update flag, "
                "reason, or provenance — route it through _oracle_env)"
            )
    return violations


@functools.lru_cache(maxsize=1)
def _test_texts() -> tuple[tuple[str, str], ...]:
    """Return (relpath, source text) for every python file under tests/."""

    texts: list[tuple[str, str]] = []
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        texts.append((path.relative_to(_TESTS_DIR).as_posix(), path.read_text(encoding="utf-8")))
    return tuple(texts)


_PARSED: dict[str, ast.AST] = {}


def _tree_for(relpath: str, source: str) -> ast.AST:
    """Parse ``source`` once per file across the repo-wide scans."""

    tree = _PARSED.get(relpath)
    if tree is None:
        tree = _PARSED[relpath] = ast.parse(source)
    return tree


#: Cheap substring prefilters: a file can only violate a scan if the trigger
#: token appears in its text (the flag scanners require the flag literal in
#: the same file for key resolution; the xfail scan requires "xfail").
_FLAG_TOKENS = (*GOLDEN_FLAG_PREFIXES, *GOLDEN_FLAG_NAMES)


def _scan_repo(scanner, tokens: tuple[str, ...]) -> list[str]:
    """Run one scanner over every tests/ file whose text mentions a token."""

    violations: list[str] = []
    for relpath, source in _test_texts():
        if not any(token in source for token in tokens):
            continue
        violations.extend(scanner("", relpath, tree=_tree_for(relpath, source)))
    return violations


@pytest.mark.smoke
def test_no_unguarded_golden_flag_reads_in_tests() -> None:
    """Every golden mutation flag in tests/ arms on the exact value "1"."""

    violations = _scan_repo(find_unguarded_flag_reads, _FLAG_TOKENS)
    assert not violations, "\n".join(violations)


@pytest.mark.smoke
def test_no_autogreen_update_branches_in_tests() -> None:
    """Every golden update branch in tests/ is write-then-skip, never green."""

    violations = _scan_repo(find_autogreen_update_branches, _FLAG_TOKENS)
    assert not violations, "\n".join(violations)


@pytest.mark.smoke
def test_no_reasonless_xfails_in_tests() -> None:
    """Every xfail marker in tests/ names the gap it pins via reason=."""

    violations = _scan_repo(find_bare_xfails, ("xfail",))
    assert not violations, "\n".join(violations)


@pytest.mark.smoke
def test_no_unflagged_baseline_writes_in_tests() -> None:
    """No missing-golden branch may silently self-baseline (flag-independent).

    Prefiltered on the write tokens themselves, NOT the flag tokens — the
    whole point is catching sites the flag system never saw (b10 R78-2).
    """

    violations = _scan_repo(find_unflagged_baseline_writes, ("write_text", "write_bytes"))
    assert not violations, "\n".join(violations)


_PLANTED_UNFLAGGED_BASELINE = """
import json
import pytest

def test_snapshot(path, actual):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual))
        pytest.skip(f"baseline snapshot generated: {path}")
    assert json.loads(path.read_text()) == actual
"""

_GOVERNED_BASELINE = """
import json
import os
import pytest
from _oracle_env import flag_armed, require_update_reason, write_provenance

def test_snapshot(path, actual):
    if not path.exists():
        if flag_armed(os.environ, "TORCHLENS_UPDATE_X") and not os.environ.get("CI"):
            reason = require_update_reason("TORCHLENS_UPDATE_X")
            path.write_text(json.dumps(actual))
            write_provenance(path.parent, "test.py", "TORCHLENS_UPDATE_X", reason)
            pytest.skip("baseline generated")
        pytest.fail("missing golden")
    assert json.loads(path.read_text()) == actual
"""


@pytest.mark.smoke
def test_unflagged_baseline_scanner_is_red_capable() -> None:
    """The exact pre-fix tests/snapshots shape is caught; the governed shape passes."""

    violations = find_unflagged_baseline_writes(_PLANTED_UNFLAGGED_BASELINE)
    assert len(violations) == 1 and "self-baseline" in violations[0]
    assert find_unflagged_baseline_writes(_GOVERNED_BASELINE) == []


@pytest.mark.smoke
def test_repo_scan_prefilter_is_sound() -> None:
    """The substring prefilter can never hide a violation.

    Every scanner's violations require its trigger token to appear literally
    in the offending file's text: the flag scanners only fire on a RESOLVED
    flag name (a string literal in the same file), and the xfail scanner only
    fires on an ``xfail`` attribute. Planted offenders must therefore always
    contain their token — asserted here so a scanner change that breaks the
    assumption fails loudly.
    """

    for snippet, tokens in (
        (_PLANTED_TRUTHY_FLAG, _FLAG_TOKENS),
        (_PLANTED_PRESENCE_FLAG, _FLAG_TOKENS),
        (_PLANTED_BARE_IF_FLAG, _FLAG_TOKENS),
        (_PLANTED_AUTOGREEN, _FLAG_TOKENS),
        (_PLANTED_BARE_XFAIL, ("xfail",)),
        (_PLANTED_UNFLAGGED_BASELINE, ("write_text", "write_bytes")),
    ):
        assert any(token in snippet for token in tokens)


# ---------------------------------------------------------------------------
# Regression pin for the fixed backend-parity auto-green path (b10 R78 r3-2)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_backend_parity_update_run_never_reports_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixed fourth auto-green path: an update run writes, then SKIPS.

    Before the fix, ``_read_or_update_golden`` wrote the payload and returned
    it, so the caller compared the projection against itself and an update
    run reported green verification.
    """

    from backend_parity import test_torch_parity_gates as parity

    monkeypatch.setattr(parity, "_GOLDEN_DIR", tmp_path)
    monkeypatch.setenv(parity._UPDATE_ENV, "1")
    monkeypatch.setenv("TORCHLENS_GOLDEN_REASON", "governance-lint regression pin")
    regenerated = parity._assert_projection_matches_golden("probe", {"a": 1})
    assert regenerated is True, "update run must report regeneration, never verification"
    assert (tmp_path / "probe.json").exists()
    assert "reason: governance-lint regression pin" in (tmp_path / "PROVENANCE").read_text()
    with pytest.raises(pytest.skip.Exception):
        parity._skip_regenerated(regenerated)
    # The write did not green-wash: without the flag, a diverging projection
    # FAILS against the just-written golden, and the true one verifies.
    monkeypatch.delenv(parity._UPDATE_ENV)
    with pytest.raises(AssertionError):
        parity._assert_projection_matches_golden("probe", {"a": 2})
    assert parity._assert_projection_matches_golden("probe", {"a": 1}) is False


@pytest.mark.smoke
def test_backend_parity_update_flag_arms_on_exact_one_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TORCHLENS_UPDATE_BACKEND_PARITY=0 must NOT regenerate goldens."""

    from backend_parity import test_torch_parity_gates as parity

    monkeypatch.setattr(parity, "_GOLDEN_DIR", tmp_path)
    monkeypatch.setenv(parity._UPDATE_ENV, "0")
    (tmp_path / "probe.json").write_text(
        '{"sha256_chunks": [], "projection": {"a": 1}}\n', encoding="utf-8"
    )
    with pytest.raises(AssertionError):
        # Digest mismatch proves the disarmed flag COMPARED instead of writing.
        parity._assert_projection_matches_golden("probe", {"a": 1})
    assert '"a": 1' in (tmp_path / "probe.json").read_text(), "disarmed flag must not rewrite"


@pytest.mark.smoke
def test_session_warmup_carveout_matches_every_golden_flag_prefix() -> None:
    """r7 R77 (fable b2 MED): the warmup skip must honor ALL golden prefixes.

    The session warmup capture wraps torch before any test runs; SF-53
    wrap-state-guarded golden families therefore need the warmup skipped when
    ANY declared golden flag is armed. The fixwave-5 carve-out hardcoded
    ``TORCHLENS_UPDATE_`` and left the ``TORCHLENS_REGEN_`` families'
    documented regen recipes hard-failing at their own guard. Pin: the
    conftest predicate derives from ``GOLDEN_FLAG_PREFIXES``, never a
    hand-copied prefix literal.
    """

    conftest_text = (Path(__file__).resolve().parent / "conftest.py").read_text(encoding="utf-8")
    assert "key.startswith(GOLDEN_FLAG_PREFIXES)" in conftest_text, (
        "the session-warmup golden carve-out no longer derives from "
        "_oracle_env.GOLDEN_FLAG_PREFIXES"
    )
    assert 'key.startswith("TORCHLENS_UPDATE_")' not in conftest_text, (
        "a hand-copied single-prefix carve-out is back in conftest.py"
    )


@pytest.mark.smoke
def test_provenance_source_disclosure_ignores_the_familys_own_output() -> None:
    """r7 R78-1 (opus b10 MED): the dirty disclosure must be about the SOURCE.

    Golden bytes are rewritten BEFORE the provenance stamp, so counting the
    family's own fresh output in `git status --porcelain` made every real
    rebaseline read "dirty" -- an always-firing disclosure discloses nothing.
    The filter drops the stamping family's own paths and PROVENANCE sidecars;
    genuine source-tree dirt still reads dirty.
    """

    from _oracle_env import _foreign_porcelain_lines

    own = "tests/golden/family_a/"
    regen_only = (
        " M tests/golden/family_a/case1.dot\n"
        "?? tests/golden/family_a/case2.dot\n"
        " M tests/golden/family_a/PROVENANCE\n"
    )
    assert _foreign_porcelain_lines(regen_only, own) == []
    with_source_dirt = regen_only + " M torchlens/visualization/_render_dot.py\n"
    foreign = _foreign_porcelain_lines(with_source_dirt, own)
    assert foreign == [" M torchlens/visualization/_render_dot.py"]
    # A rename record is judged by its destination path.
    renamed = "R  tests/golden/family_a/old.dot -> tests/golden/family_a/new.dot\n"
    assert _foreign_porcelain_lines(renamed, own) == []
    # Sibling-family output still counts as dirt (conservative, disclosed).
    sibling = " M tests/golden/family_b/case.dot\n"
    assert _foreign_porcelain_lines(sibling, own) == [" M tests/golden/family_b/case.dot"]


@pytest.mark.smoke
def test_emitter_census_discovers_every_emitter_declaration() -> None:
    """r7 R78-2 (opus b10 MED): the emitter census is DERIVED-complete, both ways.

    The six census rows were hand-written with no discovery sweep, so a NEW
    file gaining an ``_EMITTER_PACKAGES`` tuple was simply absent and
    unchecked -- the declaration-vs-usage hole moved one level up (the
    wrap-state guard's regex-discovery is the in-file precedent). Sweep every
    test file: an ``_EMITTER_PACKAGES`` declaration outside the census is a
    refusal, and every ENV-governed marker directory must be reachable from
    the census.
    """

    censused = {relpath for relpath, _, _ in _EMITTER_PACKAGE_CENSUS}
    declaring = {relpath for relpath, source in _test_texts() if _EMITTER_TUPLE_RE.search(source)}
    undiscovered = declaring - censused
    assert not undiscovered, (
        f"file(s) declare _EMITTER_PACKAGES outside the census: {sorted(undiscovered)} "
        "-- add census rows (and the family's ENV-<pkg> markers) in the same change"
    )
    # Marker-table direction: every ENV-governed directory is reachable from
    # the census, so a dir added to the marker table cannot float unchecked.
    # Reason-bearing exemptions only (never bare names):
    exempt_dirs = {
        # Two-layer byte+pixel scheme in test_bundle_diff_renderer.py; keys
        # on the graphviz BINARY version (ENV-graphviz-dot), not on python
        # emitter packages, and does not resolve through resolve_env_golden.
        "snapshots",
    }
    census_dirs = {directory for _, directory, _ in _EMITTER_PACKAGE_CENSUS}
    unreachable = set(_ENV_GOVERNED_REQUIRED_MARKERS) - census_dirs - exempt_dirs
    assert not unreachable, (
        f"ENV-governed marker dir(s) unreachable from the emitter census: "
        f"{sorted(unreachable)} -- their emitter keying is unchecked"
    )
    stale_exempt = exempt_dirs - set(_ENV_GOVERNED_REQUIRED_MARKERS)
    assert not stale_exempt, f"stale emitter-census exemption(s): {sorted(stale_exempt)}"
