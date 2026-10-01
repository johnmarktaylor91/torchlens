"""Tripwire mutation driver: prove the validation suite KILLS neutered checks.

Institutionalized from the b9 hunt's throwaway seeds (R74/75-5), with the
pristine-control protocol BAKED IN: an un-controlled adjudication on a tree
with baseline reds hallucinated two kills during the b9 hunt, so this driver
refuses to score mutants until the UNMUTATED suite is green in the same
sandbox.

Five mutant families, one per disarming direction or granularity:

* The METADATA-INVARIANT REGISTRY family is DERIVED at run time from
  ``torchlens.validation.invariants.METADATA_INVARIANT_CONTRACTS`` inside the
  sandbox (b9-opus R74-2: the hand-listed roster had enrolled 12 of 32
  contracts, so 20 tripwires were never margin-measured). One whole-function
  ``return None`` mutant per contract, id = the contract name; a contract
  landing in the registry is enrolled by construction.
* ``MUTANTS`` neuters the NON-REGISTRY checks (replay comparator,
  postprocess contract checker) with an unconditional ``return None``
  (the first statement after the docstring) -- the disarming direction for
  raise-on-violation invariants.
* ``BLOCK_MUTANTS`` plants a bare ``return None`` immediately BEFORE a named
  witness block inside a multi-check function (b9-opus R74-2: the
  whole-function operator is blind to SUB-CHECK deletion -- a bare ``return``
  before the edge-occurrence multiplicity witness survived the then-current
  suite 431/431 while every earlier sub-check still ran). Enroll one entry
  per comment-marked witness block appended to an existing check.
* ``EXEMPT_MUTANTS`` plants ``return True`` on the perturbation-exemption
  dispatcher and each ``_check_*_exempt`` gate (b9p3 R74p3-F2): for a
  predicate whose ``True`` means "skip the sensitivity check", ``return
  None`` is falsy and makes the tripwire STRICTER -- the dangerous direction
  is exempt-everything, so it needs its own operator.
* The PER-ARM family is DERIVED like the registry family (b9-opus R74r4-F1):
  every ``raise MetadataInvariantError`` statement inside a registered
  checker is one mutant (id ``<contract>#aNN``, lexical order) whose operator
  replaces exactly that raise with ``pass``. The whole-function operator is
  blind to single-arm disarms -- two arms (the op_log_fields functionless
  sentinel and the capture_edge_survival slot-rewire reconciliation) were
  PROVEN silent survivors over the 448-test arming suite -- and a
  hand-enrolled ``BLOCK_MUTANTS`` roster is the same drift shape the
  registry derivation eliminated. Arm enrollment is by construction: a new
  raise arm in any registered checker is margin-measured with no driver
  edit. A full arm campaign is ~161 suite runs; select subsets with explicit
  ids or ``--family arms``.

Kill attribution is per-test, not per-exit-code (b9 R74-2: a green control
still printed a "KILLED" off an unrelated flaky red): each run's FAILED node
ids are parsed and a mutant is KILLED only by ``killers = mutant_failures -
control_failures``, with the killer node ids named in the verdict. A mutant
run that reds without any parseable failed test (collection error, crash) is
an ERROR verdict, never a kill.

Usage (from the repo root)::

    python tests/support/mutation_driver.py --make-sandbox /tmp/tl-mut M04 X01
    python tests/support/mutation_driver.py --sandbox /tmp/tl-mut/repo  # all

The driver only ever writes inside the sandbox; running against the real
checkout is refused. It is a SCRIPT, deliberately not named ``test_*``: the
red-capability *tests* live in the suite itself; this measures their margin.
CI wiring (b9 round 5, two complementary legs): ``weekly.yml``'s
``mutation-margin`` job scores every bounded family
(registry/checks/corechecks/blocks/exempt/executor) each week in the
canonical pinned CPU env and fails on any survivor or scoring error;
``.github/workflows/mutation.yml`` automates the ~161-run per-arm campaign
as a weekly rotating shard (one of four arm shards per week, plus
``workflow_dispatch`` for any family), archives every verdict, and fails on
any SURVIVOR/ERROR/TIMEOUT (R74 b9-sol finding 1). A zero-survivor
arm-campaign log (driver sha + suite sha + survivor count) is a REQUIRED
convergence artifact before the R74 row may be declared converged.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: mutant id -> (relative file, function to neuter with ``return None``).
#: NON-REGISTRY checks only: every metadata-invariant contract is enrolled
#: automatically from the registry (``derive_registry_mutants``), so this
#: dict must never list one -- it would double-run under a drifting id.
MUTANTS: dict[str, tuple[str, str]] = {
    # The flagship per-op replay comparator (R74/75-2): neutering the
    # comparison result at the callsite must be killed by the corruption
    # battery, not by a single diagnostics test.
    "M13": ("torchlens/validation/core.py", "_deep_numeric_replay_matches_saved"),
    # The postprocess step-contract checker (b9-sol R74-1): a return-None
    # disarm previously survived because this suite listed only validation
    # files. Killed directly by tests/test_postprocess_contract_arming.py and
    # through real armed captures by the test_postprocess_dag enforcement
    # plants below.
    "M14": ("torchlens/postprocess/__init__.py", "_check_postprocess_contract"),
    # The b9-sol round-5 enrollment gap (R74-2): the direct roster covered
    # exactly one validation comparator and one postprocess checker while
    # validation/core.py carries five more raise-on-violation entry points
    # and postprocess/__init__.py two assert seams, all previously
    # margin-unmeasured. tests/test_mutation_driver_governance.py now
    # censuses the check-shaped functions in both files against this roster,
    # so the next entry point cannot ship unenrolled.
    "M15": ("torchlens/validation/core.py", "_check_layer_arguments_logged_correctly"),
    "M16": ("torchlens/validation/core.py", "_validate_layer_against_arg"),
    "M17": ("torchlens/validation/core.py", "_check_arglocs_correct_for_arg"),
    "M18": ("torchlens/validation/core.py", "_check_unattributed_arg_slots"),
    "M19": (
        "torchlens/validation/core.py",
        "_check_whether_func_on_saved_parents_yields_saved_tensor",
    ),
    "M20": ("torchlens/postprocess/__init__.py", "_assert_no_open_window"),
    "M21": ("torchlens/postprocess/__init__.py", "_assert_postprocess_contract"),
}

#: mutant id -> (relative file, core-validation checker). The checkers of
#: ``validation/core.py`` OUTSIDE the metadata-invariant registry (b9-sol
#: R74r5 finding 2: the non-registry roster held one comparator and one
#: postprocess checker while core.py carried five more verdict-steering
#: entry points with no mutant). Their dangerous neutral value is an
#: ALWAYS-VALIDATED result, planted via ``CORE_CHECK_NEUTER``. Enrollment
#: drift is refused at roster assembly: ``derive_core_check_roster`` scans
#: core.py for every ``_check_*`` / ``_validate_*`` def and demands each be
#: enrolled here or excluded with a reason in ``CORE_CHECK_EXCLUSIONS``.
CORE_CHECK_MUTANTS: dict[str, tuple[str, str]] = {
    "V01": ("torchlens/validation/core.py", "_check_layer_arguments_logged_correctly"),
    "V02": ("torchlens/validation/core.py", "_validate_layer_against_arg"),
    "V03": ("torchlens/validation/core.py", "_check_arglocs_correct_for_arg"),
    "V04": ("torchlens/validation/core.py", "_check_unattributed_arg_slots"),
    "V05": (
        "torchlens/validation/core.py",
        "_check_whether_func_on_saved_parents_yields_saved_tensor",
    ),
}

#: Planted return value for the corechecks family: the always-pass direction
#: for functions whose contract is a structured verdict.
CORE_CHECK_NEUTER = 'ValidationCheckResult.validated("R74-CORECHECK-MUTANT")'

#: Checker-shaped core.py functions deliberately NOT in CORE_CHECK_MUTANTS,
#: each with the reason (a name in neither table refuses the campaign).
CORE_CHECK_EXCLUSIONS: dict[str, str] = {
    "_check_perturbation_exemptions": (
        "enrolled as X01: its dangerous direction is exempt-everything "
        "(return True), not always-validated"
    ),
}


def derive_core_check_roster(sandbox: Path) -> None:
    """Refuse the campaign when a core.py checker is neither enrolled nor excluded.

    b9-sol R74r5 finding 2: exhaustive-coverage claims rested on the
    metadata-contract registry alone while core.py grew verdict-steering
    checkers with no mutant. This scan makes enrollment drift LOUD: every
    top-level ``_check_*`` / ``_validate_*`` def must appear in
    ``CORE_CHECK_MUTANTS`` or carry a reason in ``CORE_CHECK_EXCLUSIONS``.

    Parameters
    ----------
    sandbox:
        Repo root whose ``torchlens/validation/core.py`` is scanned.
    """

    core = sandbox / "torchlens" / "validation" / "core.py"
    tree = ast.parse(core.read_text(encoding="utf-8"))
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith(("_check_", "_validate_"))
    }
    enrolled = {func for _, func in CORE_CHECK_MUTANTS.values()}
    missing = sorted(names - enrolled - set(CORE_CHECK_EXCLUSIONS))
    if missing:
        raise SystemExit(
            f"unenrolled core.py checkers: {missing} -- add each to "
            "CORE_CHECK_MUTANTS or CORE_CHECK_EXCLUSIONS with a reason "
            "(b9-sol R74r5: silent enrollment drift is the defect)"
        )
    stale = sorted((enrolled | set(CORE_CHECK_EXCLUSIONS)) - names)
    if stale:
        raise SystemExit(
            f"CORE_CHECK ledger rows without a core.py def: {stale} -- "
            "the checker moved or was renamed; re-point the row"
        )


#: mutant id -> (relative file, function, comment marker). A bare ``return
#: None`` is planted immediately BEFORE the first comment line inside the
#: function that contains the marker: the sub-check deletion direction the
#: whole-function operator cannot see (b9-opus R74-2). Enroll one entry per
#: comment-marked witness block that was APPENDED to an existing check.
BLOCK_MUTANTS: dict[str, tuple[str, str, str]] = {
    "B01": (
        "torchlens/validation/_invariants_payloads.py",
        "_check_edge_use_parent_arg_invariants",
        "Edge-occurrence MULTIPLICITY witness (see docstring)",
    ),
}

#: mutant id -> (relative file, exemption predicate to disarm with ``return
#: True``). ``True`` means "this layer is exempt from the perturbation
#: sensitivity check", so the dangerous direction is exempt-everything --
#: killed by the deliberately-named ``*_is_not_exempt`` negative tests
#: (hand-run on the dispatcher during b9p3: 28 killers).
EXEMPT_MUTANTS: dict[str, tuple[str, str]] = {
    "X01": ("torchlens/validation/core.py", "_check_perturbation_exemptions"),
    "X02": ("torchlens/validation/exemptions.py", "_check_getitem_exempt"),
    "X03": ("torchlens/validation/exemptions.py", "_check_setitem_exempt"),
    "X04": ("torchlens/validation/exemptions.py", "_check_index_put_exempt"),
    "X05": ("torchlens/validation/exemptions.py", "_check_lstm_exempt"),
    "X06": ("torchlens/validation/exemptions.py", "_check_interpolate_exempt"),
    "X07": ("torchlens/validation/exemptions.py", "_check_scatter_exempt"),
    "X08": ("torchlens/validation/exemptions.py", "_check_one_arg_where_index_exempt"),
    "X09": ("torchlens/validation/exemptions.py", "_check_where_exempt"),
    "X10": ("torchlens/validation/exemptions.py", "_check_masked_fill_exempt"),
    "X11": ("torchlens/validation/exemptions.py", "_check_norm_running_stat_exempt"),
    "X12": ("torchlens/validation/exemptions.py", "_check_scatter_or_index_domain_exempt"),
    "X13": ("torchlens/validation/_invariants_backward_flow.py", "_is_func_call_id_exempt"),
    # Landed after the b9p3 inventory (R08 exemption-narrowing wave); swept in
    # so the newest gate is margin-measured like its siblings.
    "X14": ("torchlens/validation/exemptions.py", "_check_zipped_sibling_exempt"),
}

#: Bounded arming suite: the files whose job is to kill the mutants above.
SUITE = [
    "tests/test_validation.py",
    # Step-18 (streamed-bundle finalization) killers: the executor family's
    # first sample campaign proved _should_run_step_18 forced-False survived
    # every file below (r7 R74; no suite file streamed to disk). Removing
    # this file resurrects a PROVEN survivor.
    "tests/test_streaming_finalization_arming.py",
    "tests/test_replay_corruption_battery.py",
    "tests/test_internals.py",
    "tests/test_ancestry_closure_invariant.py",
    "tests/test_conditional_invariants.py",
    "tests/test_loop_synthesis_ground_truth.py",
    "tests/test_r29_capval_hardening.py",
    # B01 killers (b9-opus R74-2): the oracle-independence tamper battery is
    # the file arming the edge-occurrence multiplicity witness; without it a
    # bare return planted before that block survived the rest of this suite.
    # (Also the dedicated arming file for the newest oracle-independence
    # witnesses -- b9-opus R74r3-F2 part 1; this row was listed twice with
    # the two rationales, r7 dedupe merged them.)
    "tests/test_oracle_independence.py",
    # r7 R74 (opus, PROVEN suite-composition artifacts): 2 of 4 sampled
    # executor survivors were killed by tests OUTSIDE this suite --
    # _run_step_16_5 (graph_shape_hash) by six surface-oracle byte params,
    # _run_step_20 (release_param_refs) by the outcome-characterization
    # planted-propagation test. The weekly executor leg was structurally red
    # on non-defects; score the real killers instead of writing duplicates.
    "tests/surface_oracle/test_surface_oracle.py",
    "tests/test_capture_outcome_characterization.py",
    # M14 killers: the direct synthetic-input liveness file plus the two
    # armed-capture enforcement plants (undeclared read, in-place write
    # smuggle) that exercise the checker through a real postprocess run.
    "tests/test_postprocess_contract_arming.py",
    "tests/test_postprocess_dag.py::test_read_enforcement_trips_on_undeclared_read",
    "tests/test_postprocess_dag.py::test_executor_seam_patched_step_executes_and_audits",
    # W2 executor killers (lane L3 triage step 21, 2026-10-01): the bucket-name
    # golden pins every _vtimed-wrapped step's timing bucket on a default
    # capture, so a whole-function return-None disarm of _run_step_11_5,
    # _run_step_11_75, _run_step_12, or _run_step_16_5 drops its bucket from
    # the set and fails the exact-equality assert. The write-audit matrix
    # test independently proves the same steps (plus _run_step_19 /
    # _should_run_step_19) by their declared op-store columns never becoming
    # observed-effective on ANY capture axis when the step never runs. Step
    # 17.5 is deliberately unwrapped by _vtimed (writes=frozenset() too, so
    # the matrix test cannot see it either); its own direct killer is
    # test_step17_5_drops_capture_phase_workspaces.
    "tests/test_postprocess_dag.py::test_phase_timing_bucket_names_default_capture",
    "tests/test_postprocess_dag.py::test_step17_5_drops_capture_phase_workspaces",
    "tests/test_postprocess_enforcement.py::test_matrix_union_reports",
    # W3 exempt killers (X02, X07): direct negative-case calls already proved
    # the data parent/destination must stay strictly perturbed; they lived
    # outside this suite. X05/X06 killers are in test_bug_fixes_phase14.py.
    # Also hosts the new direct killers for X11/X12/X14 and the W4 registry
    # survivors (branching_invariants, layer_pass_layer_log_xrefs,
    # non_torch_backward_inert, non_torch_primitive_op_inert,
    # receptive_field_metadata, site_key_invariants); backend_neutral_graph_
    # topology and pass_count_consistency already had direct killers here too.
    "tests/validation_goldens/test_validation_exemption_hardening.py",
    "tests/test_bug_fixes_phase14.py",
]

#: Known baseline reds, deselected so a mutant verdict is never confounded.
#: KEEP THIS LIST SHORT AND DATED; every entry weakens the margin measurement
#: for whatever its tests would have killed. (The two ancestry-closure
#: deselects were removed 2026-08-15: both tests are green on tip and they
#: are M04's natural killers -- a stale entry here silently deleted M04's
#: margin. The two capture-r3 session-isolation deselects were removed later
#: the same day: green on tip for the second consecutive pass.)
#:
#: EXPIRY IS ENFORCED: tests/test_mutation_driver_governance.py runs every
#: entry and FAILS when a deselected node passes, so a stale entry can no
#: longer silently delete a mutant's margin (b9-opus R71r3-F1 part 2 -- the
#: second consecutive pass violated the keep-short-and-dated rule).
DESELECT: list[str] = []

#: Directories/patterns a sandbox never needs (b9p3 R74p3-F3: without these
#: --make-sandbox copied 6.0 GB -- 5.3 GB .venv + a 127 MB model corpus -- vs
#: ~450 MB with them; --python supplies the interpreter, so the sandbox
#: needs no venv).
SANDBOX_IGNORE = (
    ".git",
    "__pycache__",
    ".ruff_cache",
    "*.egg-info",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
    "*.tlspec",
)


#: Probe run INSIDE the sandbox to enumerate the metadata-invariant registry.
#: ``co_filename`` names the defining split module even for the rebind
#: wrappers ``invariants.py`` exports, so each contract mutates its real
#: implementation file.
_REGISTRY_PROBE = """
import json, os
from torchlens.validation.invariants import METADATA_INVARIANT_CONTRACTS
rows = {}
for contract in METADATA_INVARIANT_CONTRACTS:
    code = contract.check.__code__
    rows[contract.name] = [os.path.relpath(code.co_filename), contract.check.__name__]
print(json.dumps(rows))
"""


def derive_registry_mutants(python: str, sandbox: Path) -> dict[str, tuple[str, str]]:
    """Enumerate one whole-function mutant per metadata-invariant contract.

    Derivation runs in the SANDBOX with the scoring interpreter, so the
    enrolled roster always matches the code being mutated -- a contract added
    to ``METADATA_INVARIANT_CONTRACTS`` is margin-measured with no driver
    edit (b9-opus R74-2: the hand-listed roster had drifted to 12/32).

    Parameters
    ----------
    python:
        Python executable used for scoring runs.
    sandbox:
        Sandbox repo root to enumerate.

    Returns
    -------
    dict[str, tuple[str, str]]
        Contract name -> (relative file, check function name).
    """

    proc = subprocess.run(
        [python, "-c", _REGISTRY_PROBE],
        capture_output=True,
        text=True,
        cwd=sandbox,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )
    if proc.returncode != 0:
        raise SystemExit(f"registry derivation failed in {sandbox}:\n{proc.stderr}")
    rows = json.loads(proc.stdout)
    if not rows:
        raise SystemExit("registry derivation returned no contracts")
    return {name: (rel, func) for name, (rel, func) in sorted(rows.items())}


def _function_node(tree: ast.AST, func: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Return the first function node named ``func`` in ``tree``."""

    target = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func
        ),
        None,
    )
    if target is None:
        raise SystemExit(f"function {func} not found")
    return target


def _is_invariant_raise(node: ast.AST) -> bool:
    """Return whether ``node`` raises ``MetadataInvariantError``."""

    if not isinstance(node, ast.Raise) or node.exc is None:
        return False
    exc = node.exc
    callee = exc.func if isinstance(exc, ast.Call) else exc
    name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", None)
    return name == "MetadataInvariantError"


def enumerate_raise_arms(src: str, func: str) -> list[tuple[int, int, int]]:
    """Return every ``raise MetadataInvariantError`` arm of ``func``, in order.

    The per-arm operator's address space (b9-opus R74r4-F1): each arm is one
    mutant, so single-arm disarms are margin-measured instead of only
    whole-function neuters.

    Parameters
    ----------
    src:
        Module source text.
    func:
        Checker function name.

    Returns
    -------
    list[tuple[int, int, int]]
        ``(lineno, end_lineno, col_offset)`` per arm, lexical order.
    """

    target = _function_node(ast.parse(src), func)
    # ast.walk is breadth-first; sort into SOURCE order so arm indices are
    # stable, human-mappable addresses (id <contract>#aNN).
    return sorted(
        (node.lineno, node.end_lineno or node.lineno, node.col_offset)
        for node in ast.walk(target)
        if _is_invariant_raise(node)
    )


def while_exit_arm_keys(src: str, func: str) -> set[tuple[int, int, int]]:
    """Return the arm keys of ``func`` that sit inside a ``while`` body.

    A raise that is the sole exit of a ``while`` walk turns into an INFINITE
    LOOP under the ``pass`` operator (b9-opus R74r5-F2: the
    ``module_containment_logic`` cycle guard held 99.9% CPU for 28 minutes
    against a 2.5-minute suite). Those arms take the termination-preserving
    ``break`` operator instead — the raise is still disarmed (the violation
    goes undetected), but the mutant's margin is measurable at all.

    Parameters
    ----------
    src:
        Module source text.
    func:
        Checker function name.

    Returns
    -------
    set[tuple[int, int, int]]
        ``(lineno, end_lineno, col_offset)`` keys of while-body arms.
    """

    target = _function_node(ast.parse(src), func)
    keys: set[tuple[int, int, int]] = set()

    def _collect(node: ast.AST, in_while_body: bool) -> None:
        if in_while_body and _is_invariant_raise(node):
            raise_node = node
            keys.add(
                (
                    raise_node.lineno,
                    raise_node.end_lineno or raise_node.lineno,
                    raise_node.col_offset,
                )
            )
        if isinstance(node, ast.While):
            # ``break`` is legal in the body; the ``orelse`` block keeps the
            # enclosing status (a break there would be a syntax error).
            for stmt in node.body:
                _collect(stmt, True)
            for stmt in node.orelse:
                _collect(stmt, in_while_body)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if node is not target:
                return
        for child in ast.iter_child_nodes(node):
            _collect(child, in_while_body)

    _collect(target, False)
    return keys


def neuter_raise_arm(path: Path, func: str, index: int) -> str:
    """Disarm exactly one raise arm of ``func`` and return the original source.

    Every other arm and every other statement keeps running -- the surgical
    single-arm disarm the whole-function operator cannot model (b9-opus
    R74r4-F1: two such disarms survived the full arming suite silently).
    Arms inside a ``while`` body are replaced with ``break`` (termination
    preserved, R74r5-F2); all others with ``pass``.

    Parameters
    ----------
    path:
        File containing the checker.
    func:
        Checker function name.
    index:
        Lexical arm index from :func:`enumerate_raise_arms`.

    Returns
    -------
    str
        The file's original source, for restoration.
    """

    src = path.read_text(encoding="utf-8")
    arms = enumerate_raise_arms(src, func)
    if index >= len(arms):
        raise SystemExit(f"{func} in {path} has {len(arms)} arms; no index {index}")
    lineno, end_lineno, col = arms[index]
    keyword = arm_disarm_keyword(src, func, index)
    lines = src.splitlines(keepends=True)
    replacement = f"{' ' * col}{keyword}  # R74-ARM-MUTANT\n"
    lines[lineno - 1 : end_lineno] = [replacement]
    path.write_text("".join(lines), encoding="utf-8")
    return src


def arm_disarm_keyword(src: str, func: str, index: int) -> str:
    """Return the disarm keyword (``break``/``pass``) for one raise arm.

    Exposed separately so the archived verdict record can label the operator
    it ACTUALLY applied (r7 R74 F3: the record said "pass replacing raise
    arm 0" for the one while-exit arm where ``break`` was applied -- the very
    arm whose operator choice is load-bearing).
    """

    arms = enumerate_raise_arms(src, func)
    if index >= len(arms):
        raise SystemExit(f"{func} has {len(arms)} arms; no index {index}")
    return "break" if arms[index] in while_exit_arm_keys(src, func) else "pass"


def derive_executor_mutants(sandbox: Path) -> dict[str, tuple[str, str, str]]:
    """One skip mutant per postprocess-executor step and gate predicate.

    r7 R74 (sol b9 HIGH): the census never targeted
    ``postprocess/_executor.py`` -- 25 ``_run_step_*`` bodies plus the
    conditional ``_should_run_step_*`` predicates had NO mutation verdict,
    even though R74 explicitly scopes ``postprocess/``. DERIVED from the
    module's defs (like the registry family), so a new step enrolls itself:
    a ``_run_step_*`` neuters to ``return None`` (the step silently skips)
    and a ``_should_run_step_*`` to ``return False`` (the gate never fires).
    Import/collection errors stay ERROR verdicts, never kills.
    """

    rel = "torchlens/postprocess/_executor.py"
    tree = ast.parse((sandbox / rel).read_text(encoding="utf-8"))
    mutants: dict[str, tuple[str, str, str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_run_step_"):
            mutants[f"executor#{node.name}"] = (rel, node.name, "None")
        elif node.name.startswith("_should_run_step_"):
            mutants[f"executor#{node.name}"] = (rel, node.name, "False")
    if len(mutants) < 20:
        raise SystemExit(
            f"executor derivation found only {len(mutants)} step functions in "
            f"{rel} -- the module moved or the naming convention changed; "
            "re-point the derivation rather than scoring a hollow family"
        )
    return mutants


def derive_arm_mutants(
    sandbox: Path, registry: dict[str, tuple[str, str]]
) -> dict[str, tuple[str, str, int]]:
    """Enumerate one single-arm mutant per invariant raise in every contract.

    Derived from the same sandbox registry as the whole-function family, so a
    new arm is enrolled by construction (the hand-listed ``BLOCK_MUTANTS``
    shape drifts; b9-opus R74r4-F2).

    Parameters
    ----------
    sandbox:
        Sandbox repo root whose sources are enumerated.
    registry:
        Contract name -> (relative file, checker name) from
        :func:`derive_registry_mutants`.

    Returns
    -------
    dict[str, tuple[str, str, int]]
        Mutant id ``<contract>#aNN`` -> (relative file, checker, arm index).
    """

    arms: dict[str, tuple[str, str, int]] = {}
    for contract, (rel, func) in registry.items():
        src = (sandbox / rel).read_text(encoding="utf-8")
        for index in range(len(enumerate_raise_arms(src, func))):
            arms[f"{contract}#a{index:02d}"] = (rel, func, index)
    return arms


def neuter(path: Path, func: str, value: str) -> str:
    """Insert an early ``return <value>`` into ``func`` and return the original text.

    Parameters
    ----------
    path:
        File containing the function.
    func:
        Function name to neuter (first match wins).
    value:
        Source expression for the planted return value (the family's
        disarming direction: ``"None"`` for invariant checks, ``"True"``
        for exemption predicates).

    Returns
    -------
    str
        The file's original source, for restoration.
    """

    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    target = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func
        ),
        None,
    )
    if target is None:
        raise SystemExit(f"function {func} not found in {path}")
    body = target.body
    first = body[0]
    anchor = (
        body[1]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and len(body) > 1
        else first
    )
    lines = src.splitlines(keepends=True)
    lines.insert(anchor.lineno - 1, f"{' ' * anchor.col_offset}return {value}  # R74-MUTANT\n")
    path.write_text("".join(lines), encoding="utf-8")
    return src


def neuter_before_marker(path: Path, func: str, marker: str, value: str) -> str:
    """Insert an early ``return <value>`` before a marked block and return the original.

    The sub-check deletion operator (b9-opus R74-2): a bare return planted
    just before a comment-marked witness block leaves every earlier sub-check
    running, which the whole-function operator cannot model. Only COMMENT
    lines are matched, so a docstring restating the marker text never
    anchors the plant.

    Parameters
    ----------
    path:
        File containing the function.
    func:
        Function whose body holds the marked block.
    marker:
        Substring of the block's leading comment line.
    value:
        Source expression for the planted return value.

    Returns
    -------
    str
        The file's original source, for restoration.
    """

    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    target = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func
        ),
        None,
    )
    if target is None:
        raise SystemExit(f"function {func} not found in {path}")
    lines = src.splitlines(keepends=True)
    for lineno in range(target.body[0].lineno, (target.end_lineno or target.body[0].lineno) + 1):
        line = lines[lineno - 1]
        stripped = line.lstrip()
        if stripped.startswith("#") and marker in stripped:
            indent = len(line) - len(stripped)
            lines.insert(lineno - 1, f"{' ' * indent}return {value}  # R74-MUTANT\n")
            path.write_text("".join(lines), encoding="utf-8")
            return src
    raise SystemExit(f"marker {marker!r} not found as a comment inside {func} in {path}")


def parse_failures(stdout: str) -> frozenset[str]:
    """Extract failed/errored test node ids from a ``-rf -q`` pytest run.

    Parameters
    ----------
    stdout:
        Captured pytest stdout.

    Returns
    -------
    frozenset[str]
        Node ids reported ``FAILED`` or ``ERROR`` in the short summary.
    """

    failed = set()
    for line in stdout.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            node = line.split(" ", 2)[1]
            failed.add(node.split(" - ", 1)[0])
    return frozenset(failed)


class SuiteTimeout:
    """Sentinel result for a suite run killed at its wall-clock deadline.

    A TIMEOUT is its own verdict — never a kill, never a pass, same doctrine
    as ERROR (R74r5-F2: a non-terminating mutant held a sandbox for 28
    minutes because ``run_suite`` had no deadline and the driver waited
    forever).
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds


def run_suite(
    sandbox: Path, python: str, tag: str, timeout: float | None = None
) -> subprocess.CompletedProcess | SuiteTimeout:
    """Run the bounded arming suite inside the sandbox.

    No ``-x``: kill attribution needs the FULL failed set of every run, both
    to name each mutant's killers and to measure the margin (killer count),
    not just first-red-wins.

    Parameters
    ----------
    sandbox:
        Sandbox repo root.
    python:
        Python executable to run pytest with.
    tag:
        Unique tag for basetemp/cache isolation.
    timeout:
        Wall-clock deadline in seconds; ``None`` runs unbounded (the
        pristine control, whose wall time seeds the mutant deadline).

    Returns
    -------
    subprocess.CompletedProcess | SuiteTimeout
        The finished pytest process, or the timeout sentinel.
    """

    cache = sandbox / f".cache-{tag}"
    cache.mkdir(exist_ok=True)
    cmd = [python, "-m", "pytest", *SUITE, "-p", "no:randomly", "-q", "--tb=no", "-rf"]
    cmd += ["--basetemp", str(sandbox / f".bt-{tag}")]
    for node in DESELECT:
        cmd += ["--deselect", node]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="",
        TORCHLENS_CACHE_DIR=str(cache),
    )
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, env=env, cwd=sandbox, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return SuiteTimeout(float(timeout or 0.0))


def require_nonempty_selection(
    ids: list[str], *, family: str | None, arm_shard: str | None
) -> None:
    """Refuse an empty mutant selection instead of scoring vacuously green.

    r7 R79 (fable b10 MED, corroborated by the R82 lane): an empty ``ids``
    list skipped the campaign loop entirely and printed ``all mutants
    KILLED`` with exit 0 -- so a shard-slicing bug or a family-key rename
    would turn the scheduled leg permanently, silently green. Zero selected
    mutants is never a verdict; it is a selection failure.
    """

    if not ids:
        raise SystemExit(
            "EMPTY MUTANT SELECTION -- refusing to report a vacuous "
            f"'all mutants KILLED' (family={family!r}, arm_shard={arm_shard!r}); "
            "fix the family key or shard arithmetic"
        )


def main() -> None:
    """Parse arguments, enforce the pristine control, and score each mutant."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mutants", nargs="*", default=[], help="mutant ids (default: all)")
    parser.add_argument("--sandbox", type=Path, help="existing sandbox repo root")
    parser.add_argument(
        "--make-sandbox",
        type=Path,
        help="copy the current repo to DIR/repo and use it as the sandbox",
    )
    parser.add_argument("--python", default=sys.executable, help="python to run pytest with")
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="UNSAFE: skip the pristine control (only when just proven green)",
    )
    parser.add_argument(
        "--family",
        choices=("registry", "checks", "blocks", "exempt", "arms", "corechecks", "executor"),
        help="score only one mutant family (a full arm campaign is ~161 runs)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "per-mutant suite deadline in seconds (default: 4x the measured "
            "control wall time, min 300; 1800 with --skip-control)"
        ),
    )
    parser.add_argument(
        "--arm-shard",
        default=None,
        metavar="I/N",
        help="score only shard I of N (1-based) of the selected ids, for CI rotation",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    if args.make_sandbox:
        sandbox = args.make_sandbox / "repo"
        if not sandbox.exists():
            print(f"copying {repo} -> {sandbox} ...", flush=True)
            shutil.copytree(repo, sandbox, ignore=shutil.ignore_patterns(*SANDBOX_IGNORE))
    else:
        sandbox = args.sandbox
    if sandbox is None:
        raise SystemExit("need --sandbox DIR or --make-sandbox DIR")
    sandbox = sandbox.resolve()
    if sandbox == repo:
        raise SystemExit("refusing to mutate the real checkout; use --make-sandbox")

    # Assemble the roster: registry-derived contracts + derived arms + the
    # three hand lists. plan: mutant id -> (relative file, function,
    # marker-or-None, value, arm-index-or-None).
    registry = derive_registry_mutants(args.python, sandbox)
    overlap = sorted(mid for mid, target in MUTANTS.items() if target in set(registry.values()))
    if overlap:
        raise SystemExit(
            f"hand-listed MUTANTS duplicate registry contracts: {overlap} -- "
            "delete them; registry contracts enroll automatically"
        )
    arm_mutants = derive_arm_mutants(sandbox, registry)
    executor_mutants = derive_executor_mutants(sandbox)
    derive_core_check_roster(sandbox)
    plan: dict[str, tuple[str, str, str | None, str, int | None]] = {}
    families: dict[str, list[str]] = {}
    for mid, (rel, func) in registry.items():
        plan[mid] = (rel, func, None, "None", None)
        families.setdefault("registry", []).append(mid)
    for mid, (rel, func) in MUTANTS.items():
        plan[mid] = (rel, func, None, "None", None)
        families.setdefault("checks", []).append(mid)
    for mid, (rel, func) in CORE_CHECK_MUTANTS.items():
        plan[mid] = (rel, func, None, CORE_CHECK_NEUTER, None)
        families.setdefault("corechecks", []).append(mid)
    for mid, (rel, func, marker) in BLOCK_MUTANTS.items():
        plan[mid] = (rel, func, marker, "None", None)
        families.setdefault("blocks", []).append(mid)
    for mid, (rel, func) in EXEMPT_MUTANTS.items():
        plan[mid] = (rel, func, None, "True", None)
        families.setdefault("exempt", []).append(mid)
    for mid, (rel, func, arm_index) in arm_mutants.items():
        plan[mid] = (rel, func, None, "None", arm_index)
        families.setdefault("arms", []).append(mid)
    for mid, (rel, func, value) in executor_mutants.items():
        plan[mid] = (rel, func, None, value, None)
        families.setdefault("executor", []).append(mid)
    n_families = (
        len(registry)
        + len(MUTANTS)
        + len(CORE_CHECK_MUTANTS)
        + len(BLOCK_MUTANTS)
        + len(EXEMPT_MUTANTS)
        + len(arm_mutants)
        + len(executor_mutants)
    )
    if len(plan) != n_families:
        raise SystemExit("mutant id collision across families -- rename the clash")
    print(
        f"roster: {len(registry)} registry contracts + {len(MUTANTS)} checks + "
        f"{len(CORE_CHECK_MUTANTS)} core checkers + "
        f"{len(BLOCK_MUTANTS)} witness blocks + {len(EXEMPT_MUTANTS)} exemption gates + "
        f"{len(arm_mutants)} raise arms + {len(executor_mutants)} executor steps",
        flush=True,
    )

    if args.family:
        ids = args.mutants or sorted(families.get(args.family, []))
        outside = [mid for mid in ids if mid not in families.get(args.family, [])]
        if outside:
            raise SystemExit(f"ids outside --family {args.family}: {outside}")
    else:
        ids = args.mutants or sorted(plan)
    unknown = [mid for mid in ids if mid not in plan]
    if unknown:
        raise SystemExit(f"unknown mutant ids: {unknown}")
    if args.arm_shard:
        shard_index_text, _, shard_count_text = args.arm_shard.partition("/")
        shard_index, shard_count = int(shard_index_text), int(shard_count_text)
        if not (1 <= shard_index <= shard_count):
            raise SystemExit(f"bad --arm-shard {args.arm_shard!r}: need 1 <= I <= N")
        ids = [mid for pos, mid in enumerate(sorted(ids)) if pos % shard_count == shard_index - 1]
        print(f"shard {shard_index}/{shard_count}: {len(ids)} mutants", flush=True)
    require_nonempty_selection(ids, family=args.family, arm_shard=args.arm_shard)

    # Pristine control: verdicts are meaningless over a red baseline (the b9
    # hunt's un-controlled pass hallucinated 2 kills off pre-existing reds).
    control_failures: frozenset[str] = frozenset()
    timeout = args.timeout
    if not args.skip_control:
        control_started = time.monotonic()
        control = run_suite(sandbox, args.python, "control")
        control_wall = time.monotonic() - control_started
        assert not isinstance(control, SuiteTimeout)  # control runs unbounded
        control_failures = parse_failures(control.stdout)
        if control.returncode != 0:
            named = "\n".join(sorted(control_failures)) or "\n".join(
                control.stdout.strip().splitlines()[-8:]
            )
            raise SystemExit(
                "PRISTINE CONTROL RED -- fix or deselect the baseline before "
                f"scoring any mutant:\n{named}"
            )
        if timeout is None:
            # R74r5-F2: a non-terminating mutant must produce a TIMEOUT
            # verdict, never hold the sandbox forever.
            timeout = max(300.0, 4.0 * control_wall)
        print(f"control: GREEN ({control_wall:.0f}s; mutant deadline {timeout:.0f}s)", flush=True)
    elif timeout is None:
        timeout = 1800.0

    results: dict[str, dict[str, object]] = {}
    for mid in ids:
        rel, func, marker, value, arm_index = plan[mid]
        path = sandbox / rel
        if arm_index is not None:
            keyword = arm_disarm_keyword(path.read_text(encoding="utf-8"), func, arm_index)
            original = neuter_raise_arm(path, func, arm_index)
            operator = f"{keyword} replacing raise arm {arm_index}"
        elif marker is None:
            original = neuter(path, func, value)
            operator = f"return {value}"
        else:
            original = neuter_before_marker(path, func, marker, value)
            operator = f"return {value} before marker {marker!r}"
        try:
            proc = run_suite(sandbox, args.python, mid, timeout=timeout)
        finally:
            path.write_text(original, encoding="utf-8")
        if isinstance(proc, SuiteTimeout):
            # R74r5-F2: never a kill, never a pass -- same doctrine as ERROR.
            results[mid] = {
                "file": rel,
                "func": func,
                "operator": operator,
                "returncode": None,
                "verdict": "TIMEOUT",
                "killers": [],
                "n_killers": 0,
                "tail": [f"suite exceeded the {proc.seconds:.0f}s deadline"],
            }
            print(json.dumps({mid: results[mid]}), flush=True)
            continue
        failures = parse_failures(proc.stdout)
        killers = sorted(failures - control_failures)
        if proc.returncode != 0 and not failures:
            # Collection error / crash: the suite never scored the mutant.
            verdict = "ERROR"
        elif killers:
            verdict = "KILLED"
        else:
            verdict = "SURVIVOR"
        results[mid] = {
            "file": rel,
            "func": func,
            "operator": operator,
            "returncode": proc.returncode,
            "verdict": verdict,
            "killers": killers,
            "n_killers": len(killers),
            "tail": proc.stdout.strip().splitlines()[-4:],
        }
        print(json.dumps({mid: results[mid]}), flush=True)

    survivors = sorted(mid for mid, row in results.items() if row["verdict"] == "SURVIVOR")
    errors = sorted(mid for mid, row in results.items() if row["verdict"] == "ERROR")
    timeouts = sorted(mid for mid, row in results.items() if row["verdict"] == "TIMEOUT")
    print("RESULTS " + json.dumps(results))
    if errors:
        print(f"ERRORS: {errors} -- suite crashed before scoring; not a kill, not a pass")
    if timeouts:
        print(f"TIMEOUTS: {timeouts} -- suite never terminated; not a kill, not a pass")
    if survivors:
        print(f"SURVIVORS: {survivors} -- each needs a new planted-corruption test")
    if errors or survivors or timeouts:
        raise SystemExit(1)
    print("all mutants KILLED")


if __name__ == "__main__":
    main()
