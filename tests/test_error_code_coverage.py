"""Provocation + reachability meta-gate over the frozen refusal vocabularies (R25-1).

Every existing lockstep gate checks SHAPE (doc row <-> declaration); none checked
REACHABILITY (a member with no raise site) or PROVOCATION (a code no test ever
asserts), which is exactly how dead members and unprovoked tripwires survive
forever — ``PathDivergenceError`` was provoked ~100x while a swap of any of its
12 divergence CODES would have failed zero tests.

Three enforcement layers, all static file scans (smoke-tier, no capture):

1. REACHABILITY — every ``RunnableErrorCode`` / ``MergedErrorCode`` member must
   be referenced in ``torchlens/`` beyond its declaration. A dead member goes
   red (the ``merged_export_unsupported`` precedent: declaration-only for a
   full release while refused surfaces raised different codes).
2. PROVOCATION RATCHET — every vocabulary code must be referenced in an
   ASSERTION CONTEXT of some test (r4 b6-sol R25: mere presence in
   executable text let inert fixture constants, unused tables, dead
   branches — and even ``def test_<code>`` FUNCTION NAMES, r4 b6-opus
   R25-2 — count as provocation). A code counts only inside a function
   whose body executes an assertion seam, or in a module-level provocation
   table such a function references (:func:`_provoked_codes_in_source`);
   docstring mentions, comments, and ``.value == "literal"`` self-identity
   spelling asserts never count (r3 b6-opus R25-2). Exceptions: the frozen
   ``UNPROVOKED_BASELINE`` (historical debt: may only SHRINK — delete
   entries as provocations land; adding is a conscious public decision)
   and the reasoned ``ENV_GATED_ALLOWLIST`` (codes whose provocation needs
   hardware/topology this suite cannot assume). Declared residual: the
   matrix's exactly-one-owner cardinality requirement is still unenforced
   (this is a static scan, not a runtime registry).
3. Anti-vacuity + red-capability — the scanners must find real universes and
   must be demonstrably able to fail.

A NEW code (enum member or contract row) therefore cannot ship unprovoked:
it is not in the baseline, so this gate refuses until a provoking test exists
or a reasoned allowlist/baseline entry is consciously added in the same change.
"""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

import pytest
from _source_corpus import package_ast, package_files

from torchlens.merged import MergedErrorCode
from torchlens.runnable import RunnableErrorCode

# Per-test smoke marks (the former module-level pytestmark was additive
# with the census's heavy re-tier and tripped the marker-combination
# lint; every non-census test stays in the smoke tier).

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "torchlens"
_TESTS_ROOT = _REPO_ROOT / "tests"
_CONTRACT_DOC = _REPO_ROOT / "docs" / "reference" / "error_refusal_contract.md"

_DOC_ROW_PATTERN = re.compile(r"^\| `([a-z0-9_]+)` \|")

# Codes whose provocation genuinely requires an environment this suite cannot
# assume (multi-rank process groups, CUDA devices, wall-clock nondeterminism).
# Every entry carries the reason; an entry whose reason stops holding must be
# moved to a provocation test, never silently kept.
ENV_GATED_ALLOWLIST: dict[str, str] = {
    "tlspec_format_markers_incoherent": (
        "TOCTOU defense-in-depth at the load dispatch (bundle.py): the r7 R73 root fix "
        "moved the spec.json-existence check INTO detect_tlspec_format, so this raise "
        "only fires if spec.json vanishes between detection and dispatch -- a race no "
        "deterministic test can provoke without fs mocking; kept as a belt."
    ),
}

# Historical provocation debt, frozen at FW2-ERRORS lane time. SHRINK-ONLY:
# when a provoking test lands for a code, DELETE its row here in the same
# change. Adding a row is a public-vocabulary governance decision, not a fix
# for a red gate.
UNPROVOKED_BASELINE: frozenset[str] = frozenset(
    {
        "annotation_backend_unsupported",
        "annotation_namespace_invalid",
        "annotation_tensor_not_portable",
        "backend_capability_conformance",
        "backend_error",
        "backward_capture_conflict",
        "backward_pass_filter_invalid",
        "batch_items_invalid",
        "batch_render_invalid",
        "bundle_diff_layout_invalid",
        "bundle_diff_members_invalid",
        "bundle_member_payload_missing",
        "bundle_member_unknown",
        "bundle_shape_mismatch",
        "bundle_stack_incomplete",
        "bundle_statistic_invalid",
        "code_panel_model_collected",
        "code_panel_side_invalid",
        "compiled_callable_unsupported",
        "container_leaf_not_saved",
        "container_not_reconstructable",
        "container_selector_requires_registry",
        "container_selector_unresolved",
        "container_value_source_invalid",
        "dagua_renderer_not_opted_in",
        "decoded_output_not_classification",
        "decoded_output_unavailable",
        "fsdp_capture_unsupported",
        "gradient_not_saved",
        "gradient_pass_ambiguous",
        "intervening_cluster_invalid",
        "intervention_action_type_invalid",
        "intervention_engine_invalid",
        # The two fire-results kinds enter governance in the same change that
        # hoisted their constants (r25 b6-fable carried MED: they were outside
        # every governance surface); provoking them needs a tensor that
        # accepts neither transient metadata nor a storage side table.
        "intervention_fire_results_cleanup_failed",
        "intervention_fire_results_unrecordable",
        "intervention_helper_unknown",
        "intervention_tensor_unsupported",
        "layer_pass_ambiguous",
        "layers_not_logged",
        "link_format_invalid",
        "max_pairs_invalid",
        "metric_name_invalid",
        "metric_shape_mismatch",
        "metric_tensor_type_invalid",
        "metric_type_invalid",
        "model_type_unsupported",
        "module_call_ambiguous",
        "module_focus_empty",
        "op_lookup_pass_out_of_range",
        "op_lookup_pass_required",
        "option_group_type_invalid",
        "output_device_invalid",
        # output_sink_conflict: row DELETED (shrink-only lock-in) -- the REM1
        # test_io_streaming migration to the typed ArgumentConflictError now
        # provokes it.
        "recording_events_not_retained",
        "recording_failed_not_convertible",
        "recording_halt_frontier_missing",
        "recording_multipass_not_convertible",
        "recording_option_duplicate",
        "renderer_capability_unsupported",
        # run_fast_divergence_policy_invalid, run_fast_requires_inputs,
        # run_legacy_options_conflict, run_source_model_collected: rows
        # DELETED (shrink-only lock-in) -- the L4 S1-seam merge's compat pins
        # now provoke them.
        "run_input_missing",
        "selector_function_pattern_type_invalid",
        "skip_fn_boundary_invalid",
        "stack_ordinals_duplicate",
        "stack_ordinals_unavailable",
        "stack_output_not_tensor",
        "storage_argument_conflict",
        "summary_fields_invalid",
        "summary_option_conflict",
        "sweep_intervention_conflict",
        "sweep_names_length_mismatch",
        "sweep_site_missing",
        "sweep_site_type_invalid",
        "sweep_values_empty",
        "sweep_values_missing",
        "tensor_connection_labels_missing",
        "trace_not_finished",
    }
)


def _iter_python_texts(root: Path) -> list[tuple[Path, str]]:
    """Read every Python file under ``root`` once."""

    return [(path, path.read_text()) for path in sorted(root.rglob("*.py"))]


def _documented_codes() -> set[str]:
    """Return the contract-doc code universe."""

    codes: set[str] = set()
    for line in _CONTRACT_DOC.read_text().splitlines():
        match = _DOC_ROW_PATTERN.match(line.strip())
        if match is not None:
            codes.add(match.group(1))
    return codes


def _vocabulary() -> dict[str, str]:
    """Return every governed code mapped to its vocabulary of origin."""

    universe: dict[str, str] = {}
    for member in RunnableErrorCode:
        universe[member.value] = "RunnableErrorCode"
    for member in MergedErrorCode:
        universe[member.value] = "MergedErrorCode"
    for code in _documented_codes():
        universe.setdefault(code, "error_refusal_contract")
    return universe


def _member_name_index() -> dict[str, set[str]]:
    """Map every frozen enum MEMBER name to the code value(s) it spells."""

    index: dict[str, set[str]] = {}
    for member in RunnableErrorCode:
        index.setdefault(member.name, set()).add(member.value)
    for member in MergedErrorCode:
        index.setdefault(member.name, set()).add(member.value)
    return index


def _terminal_name(expr: ast.expr) -> str:
    """Return the terminal identifier of a name/attribute chain."""

    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return ""


def _call_name(node: ast.Call) -> str:
    """Return the terminal callable name of one call node."""

    return _terminal_name(node.func)


def _is_self_identity_assert(node: ast.Assert) -> bool:
    """Return whether an assert only proves an enum member's SPELLING.

    A self-identity assert compares an ``*ErrorCode`` member's ``.value``
    against a string LITERAL (either direction). It proves the code's
    spelling, never that any surface RAISES it, so it must not count as
    provocation (r3 b6-opus R25-2). A genuine result assert
    (``exc.fields["code"] == RunnableErrorCode.X.value`` or
    ``mismatch.code.value == "literal"``) compares a RESULT and counts.
    """

    test = node.test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return False

    def _is_member_value(expr: ast.expr) -> bool:
        return (
            isinstance(expr, ast.Attribute)
            and expr.attr == "value"
            and isinstance(expr.value, ast.Attribute)
            and _terminal_name(expr.value.value).endswith("ErrorCode")
        )

    def _is_string_literal(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Constant) and isinstance(expr.value, str)

    left, right = test.left, test.comparators[0]
    return (_is_member_value(left) and _is_string_literal(right)) or (
        _is_string_literal(left) and _is_member_value(right)
    )


def _is_assertion_seam(node: ast.AST) -> bool:
    """Return whether one node is an executed assertion/refusal seam."""

    if isinstance(node, ast.Assert):
        return not _is_self_identity_assert(node)
    if isinstance(node, ast.Call):
        call_name = _call_name(node).lower()
        return call_name == "raises" or "assert" in call_name
    return False


def _codes_in_tree(
    node: ast.AST,
    universe: set[str],
    member_names: dict[str, set[str]],
) -> set[str]:
    """Collect governed codes referenced by executable expressions in a tree.

    Skips prose channels while descending: standalone string-expression
    statements (docstrings and free prose) and self-identity spelling
    asserts contribute nothing. Comments and ``def`` names never appear as
    AST expressions, so they can never count (r4 b6-opus R25 finding 2).
    """

    return _candidate_codes_in_tree(node, member_names) & universe


def _candidate_codes_in_tree(node: ast.AST, member_names: dict[str, set[str]]) -> set[str]:
    """Universe-independent core of :func:`_codes_in_tree`.

    Collects EVERY seam-context string constant and every spelled ErrorCode
    member value; callers intersect with their universe. The split makes the
    per-file result cacheable (the universe filter is a pure intersection,
    so filtering after collection is equivalent to filtering during it).
    """

    found: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if (
            isinstance(current, ast.Expr)
            and isinstance(current.value, ast.Constant)
            and isinstance(current.value.value, str)
        ):
            continue
        if isinstance(current, ast.Assert) and _is_self_identity_assert(current):
            continue
        if isinstance(current, ast.Constant) and isinstance(current.value, str):
            found.add(current.value)
        elif (
            isinstance(current, ast.Attribute)
            and current.attr in member_names
            and _terminal_name(current.value).endswith("ErrorCode")
        ):
            found.update(member_names[current.attr])
        stack.extend(ast.iter_child_nodes(current))
    return found


def _provoked_codes_in_source(
    text: str,
    universe: set[str],
    member_names: dict[str, set[str]],
) -> set[str]:
    """Return the codes one test file provokes, by ASSERTION CONTEXT.

    r4 b6-sol R25 (HIGH): the historical scan counted a code "provoked"
    whenever its literal/member name appeared anywhere in executable TEXT,
    so an inert fixture constant, unused table, or dead branch counted
    without any test executing a refusal. A code now counts only when it is
    referenced:

    - inside a FUNCTION whose body contains an executed assertion seam — a
      real ``assert`` (self-identity spelling asserts excluded), a
      ``pytest.raises(...)`` entry, or a call to an ``assert``-named helper
      — including that function's decorators, so inline ``parametrize``
      tables feeding an asserting test count; or
    - inside a MODULE-LEVEL assignment whose target name is referenced by a
      seam-bearing function (the ``RECORD_DOOR_CASES``-style provocation
      table pattern), transitively through table-of-tables assignments.

    Unreferenced module-level constants and assert-free functions never
    count. This is file-static (no runtime instrumentation), so a code
    asserted in one seam-bearing function while USED inertly elsewhere in
    the same function still counts — the granularity is the enclosing
    function/table, not the exact assert expression. The matrix's one-owner
    cardinality requirement remains unenforced (declared residual).
    """

    try:
        tree = ast.parse(text)
    except SyntaxError:  # unparseable file provokes nothing (fail-closed)
        return set()
    return _provoked_codes_in_tree(tree, universe, member_names)


def _provoked_codes_in_tree(
    tree: ast.Module,
    universe: set[str],
    member_names: dict[str, set[str]],
) -> set[str]:
    """Tree-based core of :func:`_provoked_codes_in_source` (cached parses)."""

    return _provoked_candidates_in_tree(tree, member_names) & universe


def _provoked_candidates_in_tree(
    tree: ast.Module,
    member_names: dict[str, set[str]],
) -> set[str]:
    """Universe-independent provocation candidates for one parsed file.

    Same seam-function + referenced-assignment fixpoint as the historical
    scan; only the universe intersection moves to the caller, which makes
    the per-file result cacheable across scanner calls.
    """

    provoked: set[str] = set()
    referenced_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        seam_statements = [sub for sub in ast.walk(node) if _is_assertion_seam(sub)]
        if not seam_statements:
            continue
        # Seam-anchored attribution (r5 b6 R25): the historical scan counted
        # EVERY string constant in a seam-bearing function, so a decorative
        # literal sharing a function with any unrelated assert counted as
        # "provoked". Candidates now come only from (a) the assertion seams
        # themselves, (b) the function's decorators (parametrize tables), and
        # (c) assignments / loop and with bindings whose names a seam
        # references, fixpointed locally and (below) at module level. A
        # constant with no dataflow into any seam never counts. Declared
        # residual: a literal inside a DIFFERENT exception's seam (e.g. an
        # unrelated ``match=`` regex) still counts — attribution is per seam
        # statement, not per asserted expression.
        seam_referenced: set[str] = set()
        for seam in seam_statements:
            provoked.update(_candidate_codes_in_tree(seam, member_names))
            seam_referenced.update(sub.id for sub in ast.walk(seam) if isinstance(sub, ast.Name))
        for decorator in node.decorator_list:
            provoked.update(_candidate_codes_in_tree(decorator, member_names))
            seam_referenced.update(
                sub.id for sub in ast.walk(decorator) if isinstance(sub, ast.Name)
            )
        local_binders: list[tuple[set[str], ast.AST]] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                names = {t.id for t in sub.targets if isinstance(t, ast.Name)}
                local_binders.append((names, sub))
            elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                local_binders.append(({sub.target.id}, sub))
            elif isinstance(sub, ast.For | ast.AsyncFor | ast.comprehension):
                names = {n.id for n in ast.walk(sub.target) if isinstance(n, ast.Name)}
                local_binders.append((names, sub.iter))
            elif isinstance(sub, ast.withitem) and sub.optional_vars is not None:
                names = {n.id for n in ast.walk(sub.optional_vars) if isinstance(n, ast.Name)}
                local_binders.append((names, sub.context_expr))
        counted_local: set[int] = set()
        local_changed = True
        while local_changed:
            local_changed = False
            for names, value in local_binders:
                if id(value) in counted_local or not names & seam_referenced:
                    continue
                counted_local.add(id(value))
                provoked.update(_candidate_codes_in_tree(value, member_names))
                seam_referenced.update(
                    sub.id for sub in ast.walk(value) if isinstance(sub, ast.Name)
                )
                local_changed = True
        referenced_names.update(seam_referenced)
    module_assignments = [
        statement for statement in tree.body if isinstance(statement, ast.Assign | ast.AnnAssign)
    ]
    counted: set[int] = set()
    changed = True
    while changed:
        changed = False
        for statement in module_assignments:
            if id(statement) in counted:
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if not names & referenced_names:
                continue
            counted.add(id(statement))
            provoked.update(_candidate_codes_in_tree(statement, member_names))
            referenced_names.update(
                sub.id for sub in ast.walk(statement) if isinstance(sub, ast.Name)
            )
            changed = True
    return provoked


# Governance gates whose own tables/asserts verify SPELLING and enrollment,
# never that any surface RAISES a code — their contents must not count as
# provocation (this file's baseline, the lockstep gate's constant-spelled
# enrollment table).
_GOVERNANCE_GATE_FILES = frozenset({Path(__file__).name, "test_error_contract_lockstep.py"})


@cache
def _parsed_test_tree(path_str: str) -> ast.Module | None:
    """Cached AST for one test file (``None`` = unparseable, fail-closed).

    The whole-tree parse is ~6s of genuine CPU across ~900 files; caching it
    keeps every scanner call after the first (and after the conftest warm
    hook below) walk-only, inside the smoke duration budget.
    """

    try:
        return ast.parse(Path(path_str).read_text())
    except SyntaxError:
        return None


@cache
def _file_candidate_codes(path_str: str) -> frozenset[str]:
    """Cached universe-independent provocation candidates for one test file."""

    tree = _parsed_test_tree(path_str)
    if tree is None:  # unparseable file provokes nothing (fail-closed)
        return frozenset()
    return frozenset(_provoked_candidates_in_tree(tree, _cached_member_name_index()))


@cache
def _cached_member_name_index() -> dict[str, set[str]]:
    """Process-stable member-name index (the frozen enums never change)."""

    return _member_name_index()


def warm_scan_caches() -> None:
    """Pre-fill the per-file AST + candidate caches OUTSIDE any charged window.

    Called from the root conftest's collection hook (uncharged time), same
    convention as ``test_skip_audit.warm_scan_caches``.
    """

    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        _file_candidate_codes(str(path))


def _test_referenced_codes(universe: set[str]) -> set[str]:
    """Return every code provoked by some test's assertion context."""

    referenced: set[str] = set()
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        if path.name in _GOVERNANCE_GATE_FILES:
            continue
        referenced.update(_file_candidate_codes(str(path)) & universe)
        if referenced >= universe:
            break
    return referenced & universe


@pytest.mark.smoke
def test_universe_is_nonempty() -> None:
    """Anti-vacuity: the scanners find real vocabularies."""

    universe = _vocabulary()
    assert len(universe) > 150, f"vocabulary scan collapsed: {len(universe)} codes"
    assert any(origin == "RunnableErrorCode" for origin in universe.values())
    assert any(origin == "MergedErrorCode" for origin in universe.values())
    assert any(origin == "error_refusal_contract" for origin in universe.values())


@pytest.mark.smoke
def test_every_enum_member_is_reachable_in_source() -> None:
    """No frozen enum member is declaration-only (dead vocabulary)."""

    texts = _iter_python_texts(_PACKAGE_ROOT)
    dead: list[str] = []
    for enum_cls, declaring in (
        (RunnableErrorCode, _PACKAGE_ROOT / "runnable.py"),
        (MergedErrorCode, _PACKAGE_ROOT / "merged" / "_enums.py"),
    ):
        for member in enum_cls:
            reachable = False
            for path, text in texts:
                if path == declaring:
                    continue
                if member.name in text or member.value in text:
                    reachable = True
                    break
            if not reachable:
                dead.append(f"{enum_cls.__name__}.{member.name}")
    assert dead == [], (
        "frozen error-code members with no reference outside their declaration "
        f"(dead vocabulary — raise or remove, with a doc change): {dead}"
    )


@pytest.mark.smoke
def test_every_code_is_provoked_or_consciously_ledgered() -> None:
    """The shrink-only provocation ratchet over the full vocabulary."""

    universe = set(_vocabulary())
    referenced = _test_referenced_codes(universe)
    unprovoked = universe - referenced - set(ENV_GATED_ALLOWLIST)

    newly_unprovoked = unprovoked - UNPROVOKED_BASELINE
    assert newly_unprovoked == set(), (
        "codes with no provoking test reference and no baseline/allowlist row "
        "(new vocabulary must ship provoked): "
        f"{sorted(newly_unprovoked)}"
    )

    stale_baseline = UNPROVOKED_BASELINE - unprovoked
    assert stale_baseline == set(), (
        "baseline rows whose codes are now provoked — delete these rows to "
        f"lock in the ratchet progress: {sorted(stale_baseline)}"
    )

    stale_allowlist = {
        code for code in ENV_GATED_ALLOWLIST if code not in universe or code in referenced
    }
    assert stale_allowlist == set(), (
        f"allowlist rows that are stale (unknown or now provoked): {sorted(stale_allowlist)}"
    )


@pytest.mark.smoke
def test_provocation_scanner_is_red_capable() -> None:
    """A code absent from every test is actually reported (gate can fail)."""

    probe = "zz_probe_code_that_no_test_references_zz"
    referenced = _test_referenced_codes({probe})
    assert referenced == set()


@pytest.mark.smoke
def test_prose_names_and_inert_tables_never_count_as_provocation() -> None:
    """Non-executed mention channels never count (r3+r4 b6 R25 lineage).

    r3 b6-opus R25-2: a ``.value == "literal"`` self-identity assert, a
    docstring mention, or a comment counted under the old text scan. r4
    b6-opus R25-2: a ``def test_<code>_...`` FUNCTION NAME line counted. r4
    b6-sol R25 (HIGH): a module-level fixture constant, unused table, or
    assert-free function counted by mere presence. All must be inert; only
    assertion-context references provoke.
    """

    universe = {
        "zz_doc_code_zz",
        "zz_comment_code_zz",
        "zz_spelled_code_zz",
        "zz_result_literal_zz",
        "zz_decorative_code_zz",
        "zz_inert_table_code_zz",
        "zz_looped_dead_code_zz",
        "zz_local_seam_code_zz",
        "zz_module_seam_code_zz",
        "zz_assert_free_code_zz",
        "zz_def_name_only_code_zz",
        "zz_param_code_zz",
        "zz_helper_code_zz",
        "zz_result_code_zz",
    }
    member_names = {
        "ZZ_SPELLED_CODE_ZZ": {"zz_spelled_code_zz"},
        "ZZ_RESULT_CODE_ZZ": {"zz_result_code_zz"},
    }
    sample = "\n".join(
        [
            '"""Docstring mentioning zz_doc_code_zz never provokes."""',
            "",
            'INERT_TABLE = [("zz_inert_table_code_zz", ValueError)]',
            'LOOPED_DEAD_CASES = [("zz_looped_dead_code_zz", ValueError)]',
            'MODULE_SEAM_CASES = [("zz_module_seam_code_zz", ValueError)]',
            "",
            "",
            "def _assert_free_helper():",
            '    return "zz_assert_free_code_zz"',
            "",
            "",
            "def test_zz_def_name_only_code_zz_is_typed():",
            '    """Mentions zz_doc_code_zz again."""',
            "",
            "    observed = None  # comment mentioning zz_comment_code_zz",
            "    return observed",
            "",
            "",
            "def test_spelling_only():",
            '    assert RunnableErrorCode.ZZ_SPELLED_CODE_ZZ.value == "zz_spelled_code_zz"',
            "",
            "",
            '@pytest.mark.parametrize("code", ["zz_param_code_zz"])',
            "def test_real_provocations(code):",
            '    decorative = [("zz_decorative_code_zz", ValueError)]',
            "    for dead_row in LOOPED_DEAD_CASES:",
            "        pass",
            '    local_cases = [("zz_local_seam_code_zz", ValueError)]',
            "    for local_row in local_cases:",
            '        assert exc.fields["code"] == local_row[0]',
            "    for module_row in MODULE_SEAM_CASES:",
            '        assert exc.fields["code"] == module_row[0]',
            '    assert exc.fields["code"] == RunnableErrorCode.ZZ_RESULT_CODE_ZZ.value',
            '    assert mismatch.code.value == "zz_result_literal_zz"',
            '    _assert_refuses("zz_helper_code_zz")',
        ]
    )
    provoked = _provoked_codes_in_source(sample, universe, member_names)
    # Inert channels: prose, comments, def names, assert-free functions,
    # spelling-only asserts, UNREFERENCED module tables, and — since the r5
    # seam-anchored attribution — decorative locals and tables consumed only
    # by assert-free loops inside an otherwise asserting test (the r5 b6
    # laundering channel: any string constant in an assert-bearing function
    # used to count).
    assert "zz_doc_code_zz" not in provoked
    assert "zz_comment_code_zz" not in provoked
    assert "zz_def_name_only_code_zz" not in provoked
    assert "zz_inert_table_code_zz" not in provoked
    assert "zz_assert_free_code_zz" not in provoked
    assert "zz_spelled_code_zz" not in provoked
    assert "zz_decorative_code_zz" not in provoked
    assert "zz_looped_dead_code_zz" not in provoked
    # Countable channels: result asserts (both spellings), assert-helper
    # calls, parametrize tables on an asserting test, and local or module
    # tables whose rows FLOW INTO a seam (loop/with/assignment dataflow).
    assert "zz_result_code_zz" in provoked
    assert "zz_result_literal_zz" in provoked
    assert "zz_helper_code_zz" in provoked
    assert "zz_param_code_zz" in provoked
    assert "zz_local_seam_code_zz" in provoked
    assert "zz_module_seam_code_zz" in provoked


# --------------------------------------------------------------------------- #
# R65/F9: codeless TorchLensIOError ratchet (shrink-only)                      #
# --------------------------------------------------------------------------- #

# Historical debt (b8-opus F9: 236/243 raise sites code-less). The number may
# only SHRINK: give a site fields["code"] (and ideally a remedy) and lower the
# ceiling in the same change. Raising it is a conscious public decision.
_MAX_CODELESS_IO_RAISES = 237
_IO_FAMILY_PATTERN = re.compile(r"^(TorchLensIOError|Artifact\w*Error)$")


def _codeless_io_raises() -> list[str]:
    """Return ``path:line`` for every IO-family raise without a ``code=``."""

    offenders: list[str] = []
    for path in package_files():
        tree = package_ast(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            func = node.exc.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else None)
            )
            if name is None or not _IO_FAMILY_PATTERN.fullmatch(name):
                continue
            # A **kwargs splat may carry the code; count it as coded rather
            # than producing a false red on builder helpers.
            has_code = any(kw.arg == "code" or kw.arg is None for kw in node.exc.keywords)
            if not has_code:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    return offenders


# Whole-package AST census: crossed the 5s smoke partition as the tree grew
# through the wave-0 merges (6.1s isolated at the L1 merge); heavy per the
# marker partition (5-20s), not a budget dodge.
@pytest.mark.heavy
def test_codeless_io_raise_count_only_shrinks() -> None:
    """The IO error family's codeless-raise debt is a shrink-only ratchet."""

    offenders = _codeless_io_raises()
    assert offenders, "scanner found no IO-family raises at all (vacuous scan)"
    assert len(offenders) <= _MAX_CODELESS_IO_RAISES, (
        f"{len(offenders)} codeless IO-family raises exceed the ratchet of "
        f"{_MAX_CODELESS_IO_RAISES}. New TorchLensIOError/Artifact*Error raises "
        "must carry fields['code'] (branch-on-fields doctrine); first offenders: "
        + ", ".join(offenders[:10])
    )
