"""Preview-backend validation mutants: prove each preview tripwire is armed.

The eager torch validation path has five mutant families in
``mutation_driver.py``; until this module no mutant had ever been planted in
the preview backends' validation paths (JAX, tinygrad, MLX), so the claim that
preview validation catches capture bugs rested on checks nobody had broken on
purpose. This module hand-rosters up to eight mutants per preview backend on
its replay / perturbation / coverage path and scores them with the driver's
protocol:

* the UNMUTATED suite runs first (pristine control); its reds are recorded
  and excluded from attribution (``--strict-control`` refuses them outright,
  the driver's CI posture), so a baseline red never reads as a kill;
* ``killers = mutant_failures - control_failures``; a red run with no
  parseable failed test id is ERROR, never a kill; a run past the deadline is
  TIMEOUT;
* mutants are applied to a SANDBOX copy and reverted after every run; the
  real checkout is refused.

Each backend's arming suite is the CI leg's own selection
(``.github/workflows/pr.yml`` "Backend protocol" job): every test module that
carries the backend marker, run with ``-m "<marker> and not slow"``. The file
list is derived by scanning ``tests/`` at run time, so a new marker-bearing
module enrolls itself.

Operators: ``return`` plants an early ``return <value>`` (the driver's
``neuter``); ``replace`` substitutes one exact source snippet inside one
function (exactly one match, else the campaign refuses). The replace operator
expresses the brief's mutant kinds (wrong replay input, skipped comparison,
dropped dtype check, swallowed exception, tolerance misuse, dropped metadata
field) as single-line disarms the whole-function operator cannot model.

Usage (from the repo root, inside a venv with the backend installed)::

    python tests/support/mutation_preview.py --backend jax --make-sandbox /tmp/tl-pv
    DEV=PYTHON python tests/support/mutation_preview.py --backend tinygrad --sandbox /tmp/tl-pv/repo
    MLX_DISABLE_COMPILE=1 python tests/support/mutation_preview.py --backend mlx ...

``--results PATH`` writes a JSON record (control, per-mutant verdict, killers,
wall times); the lane folder keeps the measured results beside the campaign
logs, like the generic-operator campaign's seed ledger.
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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_driver import SANDBOX_IGNORE, SuiteTimeout, neuter, parse_failures  # noqa: E402

_JAX_BACKEND = "torchlens/backends/jax/backend.py"
_JAX_HELPERS = "torchlens/backends/jax/_trace_graph_helpers.py"
_TG_BACKEND = "torchlens/backends/tinygrad/backend.py"
_MLX_VALIDATION = "torchlens/backends/mlx/validation.py"
_MLX_RECORDS = "torchlens/backends/mlx/_capture_records.py"


@dataclass(frozen=True)
class Mutant:
    """One planted disarm: where, what and why it is dangerous."""

    rel: str
    func: str
    kind: str  # "return" or "replace"
    old: str  # return: planted value; replace: exact snippet to replace
    new: str  # replace: replacement snippet ("" for return)
    note: str


def _ret(rel: str, func: str, value: str, note: str) -> Mutant:
    return Mutant(rel, func, "return", value, "", note)


def _rep(rel: str, func: str, old: str, new: str, note: str) -> Mutant:
    return Mutant(rel, func, "replace", old, new, note)


#: backend -> (pytest marker, extra environment for the suite run).
BACKENDS: dict[str, tuple[str, dict[str, str]]] = {
    "jax": ("backend_jax", {}),
    "tinygrad": ("backend_tinygrad", {"DEV": "PYTHON"}),
    "mlx": (
        "backend_mlx",
        {"MLX_DISABLE_COMPILE": "1", "TORCHLENS_RUN_UNSTABLE_MLX_TESTS": "1"},
    ),
}

JAX_MUTANTS: dict[str, Mutant] = {
    "J01": _rep(
        _JAX_HELPERS,
        "_values_close",
        "rtol, atol = float_replay_tolerances(jnp.finfo(left_array.dtype))",
        "rtol, atol = 1e-1, 1e-1",
        "tolerance misuse: the eps-derived band replaced by a 10% band",
    ),
    "J02": _rep(
        _JAX_HELPERS,
        "_values_close",
        "if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:",
        "if left_array.shape != right_array.shape:",
        "dropped dtype check: a wrong-dtype replay compares by value only",
    ),
    "J03": _rep(
        _JAX_BACKEND,
        "_validate_one_jax_equation",
        "        return False\n    if not jax.tree.all(",
        "        return True\n    if not jax.tree.all(",
        "swallowed exception: a replay that raises passes the equation",
    ),
    "J04": _rep(
        _JAX_BACKEND,
        "_validate_one_jax_equation",
        "if not jax.tree.all(jax.tree.map(_values_close, replayed, saved_outputs)):",
        "if False:",
        "skipped comparison: replayed outputs are never compared to saved",
    ),
    "J05": _rep(
        _JAX_BACKEND,
        "_validate_one_jax_equation",
        "if not _parent_perturbations_change_output(capture, op, inputs, reference):",
        "if False:",
        "skipped perturbation: parent sensitivity is never checked",
    ),
    "J06": _ret(
        _JAX_BACKEND,
        "_jax_captures_consistent",
        "True",
        "dropped metadata check: capture/op count and kind multiset never compared",
    ),
    "J07": _ret(
        _JAX_BACKEND,
        "_validate_jax_regions",
        "True",
        "skipped op: region boundary and projection seams never replayed",
    ),
    "J08": _rep(
        _JAX_BACKEND,
        "validation_status",
        "                run_metadata_invariants(trace)\n",
        "                pass\n",
        "skipped metadata invariants on the JAX validation path",
    ),
}

TINYGRAD_MUTANTS: dict[str, Mutant] = {
    "T01": _rep(
        _TG_BACKEND,
        "_payload_values_close",
        "rtol, atol = float_replay_tolerances_for_dtype_name(dtype_name)",
        "rtol, atol = 1e-1, 1e-1",
        "tolerance misuse: the eps-derived band replaced by a 10% band",
    ),
    "T02": _rep(
        _TG_BACKEND,
        "_payloads_close",
        "if tuple(left.shape) != tuple(right.shape) or str(left.dtype) != str(right.dtype):",
        "if tuple(left.shape) != tuple(right.shape):",
        "dropped dtype check: a wrong-dtype replay compares by value only",
    ),
    "T03": _rep(
        _TG_BACKEND,
        "_uop_capture_failure",
        "if not _payloads_close(replayed, saved_output):",
        "if False:",
        "skipped comparison: replayed UOp output never compared to saved",
    ),
    "T04": _rep(
        _TG_BACKEND,
        "_parent_perturbations_change_output",
        "return not attempted",
        "return True",
        "vacuous pass: an insensitive child passes when every candidate matched",
    ),
    "T05": _ret(
        _TG_BACKEND,
        "_parent_perturbations_change_output",
        "True",
        "skipped perturbation: parent sensitivity is never checked",
    ),
    "T06": _rep(
        _TG_BACKEND,
        "_uop_capture_failure",
        "        except (KeyError, ValueError) as error:\n"
        '            return ReplayVerdict.fail(label, "replay_mismatch", str(error), index)',
        "        except (KeyError, ValueError):\n            return None",
        "swallowed exception: a replay that raises passes the capture",
    ),
    "T07": _rep(
        _TG_BACKEND,
        "_replay_uop_from_trace_graph",
        "if tuple(sorted(graph_positions.items())) != tuple(sorted(capture.parent_arg_positions)):",
        "if False:",
        "dropped metadata check: parent_arg_positions drift after capture ignored",
    ),
    "T08": _rep(
        _TG_BACKEND,
        "_replay_uop_from_trace_graph",
        "                src[position] = parent_value.uop\n",
        "                pass\n",
        "wrong replay input: replay keeps the frozen source instead of the saved parent",
    ),
}

MLX_MUTANTS: dict[str, Mutant] = {
    "X01": _rep(
        _MLX_RECORDS,
        "_payloads_close",
        "rtol, atol = float_replay_tolerances(np.finfo(a_np.dtype))",
        "rtol, atol = 1e-1, 1e-1",
        "tolerance misuse: the eps-derived band replaced by a 10% band",
    ),
    "X02": _rep(
        _MLX_RECORDS,
        "_payloads_close",
        "if a_np.shape != b_np.shape or a_np.dtype != b_np.dtype:",
        "if a_np.shape != b_np.shape:",
        "dropped dtype check: a wrong-dtype replay compares by value only",
    ),
    "X03": _ret(
        _MLX_VALIDATION,
        "_coverage_failure_count",
        "0",
        "skipped coverage: inventory never compared to the surviving captures",
    ),
    "X04": _rep(
        _MLX_VALIDATION,
        "_saved_payload_mismatch",
        "if not _payloads_close(f_out, e_out)",
        "if False",
        "skipped comparison: replayed outputs never compared to saved payloads",
    ),
    "X05": _rep(
        _MLX_VALIDATION,
        "_saved_payload_mismatch",
        "if len(final) != len(expected):",
        "if False:",
        "dropped arity check: output count drift compares the common prefix",
    ),
    "X06": _rep(
        _MLX_VALIDATION,
        "_replay_one",
        "if not _declared_parents_consistent(trace, ops_by_label, capture):",
        "if False:",
        "dropped metadata check: declared parents never cross-checked with leaves",
    ),
    "X07": _rep(
        _MLX_VALIDATION,
        "_perturbation_decision",
        "if evidence == PERTURBATION_UNPROVED:",
        "if False:",
        "vacuous pass: an unproved perturbation records a passed decision",
    ),
    "X08": _rep(
        _MLX_VALIDATION,
        "_coverage_failure_count",
        "        if isinstance(label, str) and label not in inventoried_labels:\n"
        "            failures += 1",
        "        if isinstance(label, str) and label not in inventoried_labels:\n            pass",
        "dropped coverage arm: an uninventoried materialized op is not a failure",
    ),
}

ROSTERS: dict[str, dict[str, Mutant]] = {
    "jax": JAX_MUTANTS,
    "tinygrad": TINYGRAD_MUTANTS,
    "mlx": MLX_MUTANTS,
}


def function_span(src: str, func: str) -> tuple[int, int]:
    """Return the 1-based inclusive line span of ``func`` (first match wins)."""

    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            return node.lineno, node.end_lineno or node.lineno
    raise SystemExit(f"function {func} not found")


def replace_in_function(path: Path, func: str, old: str, new: str) -> str:
    """Replace exactly one occurrence of ``old`` inside ``func``; return the original."""

    src = path.read_text(encoding="utf-8")
    start, end = function_span(src, func)
    lines = src.splitlines(keepends=True)
    head, body, tail = "".join(lines[: start - 1]), "".join(lines[start - 1 : end]), ""
    tail = "".join(lines[end:])
    count = body.count(old)
    if count != 1:
        raise SystemExit(f"{path}:{func}: snippet {old!r} matched {count} times, need exactly 1")
    path.write_text(head + body.replace(old, new) + tail, encoding="utf-8")
    return src


def apply_mutant(sandbox: Path, mutant: Mutant) -> str:
    """Plant ``mutant`` in the sandbox and return the file's original source."""

    path = sandbox / mutant.rel
    if mutant.kind == "return":
        return neuter(path, mutant.func, mutant.old)
    return replace_in_function(path, mutant.func, mutant.old, mutant.new)


def derive_suite(sandbox: Path, marker: str) -> list[str]:
    """Every ``tests/**/test_*.py`` module naming ``marker`` (the CI leg's selection)."""

    suite = sorted(
        str(path.relative_to(sandbox))
        for path in (sandbox / "tests").rglob("test_*.py")
        if marker in path.read_text(encoding="utf-8", errors="replace")
    )
    if not suite:
        raise SystemExit(f"no test module carries marker {marker!r} under {sandbox / 'tests'}")
    return suite


def run_suite(
    sandbox: Path,
    python: str,
    tag: str,
    suite: list[str],
    marker: str,
    extra_env: dict[str, str],
    timeout: float | None,
) -> subprocess.CompletedProcess | SuiteTimeout:
    """Run the backend's arming suite inside the sandbox (no ``-x``: full failed set)."""

    cache = sandbox / f".cache-{tag}"
    cache.mkdir(exist_ok=True)
    cmd = [python, "-m", "pytest", *suite, "-m", f"{marker} and not slow"]
    cmd += ["-p", "no:randomly", "-q", "--tb=no", "-rfE", "--basetemp", str(sandbox / f".bt-{tag}")]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="",
        TORCHLENS_CACHE_DIR=str(cache),
        **extra_env,
    )
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, env=env, cwd=sandbox, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return SuiteTimeout(float(timeout or 0.0))


def _verdict(
    run: subprocess.CompletedProcess | SuiteTimeout, control_failures: frozenset[str]
) -> tuple[str, list[str]]:
    if isinstance(run, SuiteTimeout):
        return "TIMEOUT", []
    failures = parse_failures(run.stdout)
    killers = sorted(failures - control_failures)
    if killers:
        return "KILLED", killers
    if run.returncode == 0 and not control_failures:
        return "SURVIVED", []
    # A tolerated red control makes every run exit 1; the mutant survived when
    # the run reproduced exactly the control's reds and nothing else.
    if run.returncode == 1 and failures == control_failures and failures:
        return "SURVIVED", []
    return "ERROR", []


def main() -> None:
    """Parse arguments, run the pristine control, then score each mutant."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mutants", nargs="*", default=[], help="mutant ids (default: all)")
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--sandbox", type=Path, help="existing sandbox repo root")
    parser.add_argument("--make-sandbox", type=Path, help="copy the repo to DIR/repo and use it")
    parser.add_argument("--python", default=sys.executable, help="python to run pytest with")
    parser.add_argument("--results", type=Path, help="write the JSON results record here")
    parser.add_argument("--timeout", type=float, default=None, help="per-mutant deadline (s)")
    parser.add_argument(
        "--strict-control",
        action="store_true",
        help="refuse to score when the unmutated suite is red (CI mode); by default "
        "control reds are recorded, excluded from kill attribution, and reported",
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

    marker, extra_env = BACKENDS[args.backend]
    roster = ROSTERS[args.backend]
    ids = args.mutants or sorted(roster)
    unknown = [mid for mid in ids if mid not in roster]
    if unknown:
        raise SystemExit(f"unknown mutant ids for {args.backend}: {unknown}")
    if not ids:
        raise SystemExit("EMPTY MUTANT SELECTION -- refusing a vacuous 'all KILLED'")
    # Every target must exist before any suite runs (a stale snippet is a
    # roster bug, not a survivor).
    for mid in ids:
        original = apply_mutant(sandbox, roster[mid])
        (sandbox / roster[mid].rel).write_text(original, encoding="utf-8")
    suite = derive_suite(sandbox, marker)
    print(f"{args.backend}: {len(ids)} mutants; suite {len(suite)} modules: {suite}", flush=True)

    started = time.monotonic()
    control = run_suite(sandbox, args.python, "control", suite, marker, extra_env, None)
    control_wall = time.monotonic() - started
    assert not isinstance(control, SuiteTimeout)
    control_failures = parse_failures(control.stdout)
    summary = control.stdout.strip().splitlines()[-1:] if control.stdout.strip() else []
    print(f"control: rc={control.returncode} {control_wall:.0f}s {summary}", flush=True)
    if control.returncode != 0:
        named = "\n".join(sorted(control_failures)) or control.stdout[-2000:]
        if args.strict_control or not control_failures:
            raise SystemExit(f"PRISTINE CONTROL RED -- fix the baseline before scoring:\n{named}")
        # Host or train reds on the unmutated tree are a finding in their own
        # right (reported in the record); they never attribute a kill, and
        # every mutant they would have killed loses that margin here.
        print(f"CONTROL REDS (excluded from attribution):\n{named}", flush=True)
    timeout = args.timeout if args.timeout else max(300.0, 4.0 * control_wall)

    record: dict[str, object] = {
        "backend": args.backend,
        "marker": marker,
        "python": args.python,
        "suite": suite,
        "control": {
            "wall_s": round(control_wall, 1),
            "summary": summary,
            "reds": sorted(control_failures),
        },
        "mutants": {},
    }
    survivors: list[str] = []
    for mid in ids:
        mutant = roster[mid]
        original = apply_mutant(sandbox, mutant)
        run_started = time.monotonic()
        try:
            run = run_suite(sandbox, args.python, mid, suite, marker, extra_env, timeout)
        finally:
            (sandbox / mutant.rel).write_text(original, encoding="utf-8")
        wall = time.monotonic() - run_started
        verdict, killers = _verdict(run, control_failures)
        if verdict != "KILLED":
            survivors.append(mid)
        record["mutants"][mid] = {  # type: ignore[index]
            "verdict": verdict,
            "killers": killers,
            "wall_s": round(wall, 1),
            "target": f"{mutant.rel}::{mutant.func}",
            "note": mutant.note,
        }
        print(f"{mid} {verdict} ({wall:.0f}s) killers={len(killers)} :: {mutant.note}", flush=True)
        for killer in killers[:6]:
            print(f"    {killer}", flush=True)
        if args.results:
            args.results.write_text(json.dumps(record, indent=1), encoding="utf-8")
    if survivors:
        raise SystemExit(f"{len(survivors)} mutant(s) not killed: {survivors}")
    print(f"all {len(ids)} {args.backend} mutants KILLED", flush=True)


if __name__ == "__main__":
    main()
