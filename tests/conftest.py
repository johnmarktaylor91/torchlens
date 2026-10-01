import gc
import os
import random
import sys
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from os.path import join as opj
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from torchlens import _state

# Output directories are assigned under pytest's private basetemp in
# ``pytest_configure`` before test modules import these constants.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_OUTPUTS_DIR = ""
REPORTS_DIR = ""
VIS_OUTPUT_DIR = ""

_MISSING = object()
_WARN_ONCE_SENTINELS: tuple[tuple[str, str, object], ...] = (
    # NOTE: this sentinel is a weakref.WeakSet in the package (type-keyed cache
    # eviction, F3a); resetting it to a plain set() would strong-pin model
    # classes — the reset must preserve the weak container type.
    ("torchlens._capture_state_helpers", "_VALIDATION_DEEPCOPY_WARNING_TYPES", weakref.WeakSet()),
    ("torchlens._capture_state_helpers", "_COMPILED_MODEL_UNWRAP_WARNED", False),
    ("torchlens._capture_state_helpers", "_COMPILED_FORCED_EAGER_WARNED", False),
    ("torchlens._io", "_LEGACY_THREAD_WARNING_EMITTED", {"flag": False}),
    ("torchlens._io.bundle", "_NONPERSISTENT_DISCLOSURE_WARNED", False),
    ("torchlens._io.bundle", "_UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED", False),
    ("torchlens._state", "_functorch_warning_emitted", False),
    ("torchlens._state", "_dynamo_warning_emitted", False),
    ("torchlens.backends.tf._tf_compat", "_warned_missing_capabilities", set()),
    ("torchlens.backends.torch.ops", "_UNSUPPORTED_OUTPUT_CONTAINER_WARNED", set()),
    ("torchlens.data_classes.op", "_WARNED_REFERENCE_SAVE_MODE", False),
    ("torchlens.distributed._lifecycle", "_AUTO_ARM_DEGRADATION", None),
    ("torchlens.fastlog._storage_resolver", "_WARNED_REFERENCE_SAVE_MODE", False),
    ("torchlens.postprocess.ast_branches", "_source_drift_warned", set()),
    # F17 quickstart nongold-value gate: once-per-trace raw-read warning keyed
    # by id(trace) (plain int keys, so a plain set() reset pins nothing).
    ("torchlens.quickstart._gate", "_WARNED_TRACE_IDS", set()),
    # F04 one-backward reads: the once-per-process frozen-default disclosure
    # (read_frozen_default_linearization) resets per test.
    ("torchlens.attribution.onebackward._frozen", "_DISCLOSURE_WARNED", set()),
    ("torchlens.utils._torch_compat", "_warned_missing_capabilities", set()),
    # Capture-scoped CUDA RNG retry latch (grind p5, B2P3-16): re-armed per
    # capture in the package, and reset per test here so a test that trips it
    # via a direct log_current_rng_states() call cannot degrade later tests.
    ("torchlens.utils.rng", "_cuda_rng_unusable", False),
    ("torchlens.user_funcs", "_BATCHNORM_TRAIN_STATS_WARNED", False),
    ("torchlens.utils.introspection", "_col_offset_cache_warned", False),
    ("torchlens.validation._stock_layer_grads", "_PASS_INDEX_PARSE_WARNED", False),
    ("torchlens.visualization._render_common", "_SIBLING_ORDER_WARNING_EMITTED", False),
    # The functional (global-mutated) copy moved to _render_ordering with the
    # renderer-thinning extraction; _render_dot keeps only an inert
    # star-import binding.
    ("torchlens.visualization._render_ordering", "_SIBLING_ORDER_WARNING_EMITTED", False),
    ("torchlens.visualization.auto_collapse", "_COUNT_MISMATCH_WARNING_EMITTED", False),
    # WeakSet of Traces that already got the collapse-ceiling decline warning;
    # a plain set() reset would strong-pin those Traces for the session.
    ("torchlens.visualization.collapse_optimizer", "_CEILING_WARNED_TRACES", weakref.WeakSet()),
    ("torchlens.visualization._collapse_disclosures", "_BUDGET_WARNED_TRACES", weakref.WeakSet()),
)


# ---------------------------------------------------------------------------
# Coverage: auto-generate text report when pytest --cov is used
# ---------------------------------------------------------------------------


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Initialize session-private outputs with shipped usage stats disabled.

    Parameters
    ----------
    config:
        Active pytest configuration.
    """

    global TEST_OUTPUTS_DIR, REPORTS_DIR, VIS_OUTPUT_DIR

    # A CI run must never mutate goldens: any armed update/regen/record flag
    # would silently rebaseline instead of verifying (b7 R53-3 / b10 R78-8).
    from _oracle_env import GOLDEN_FLAG_PREFIXES, golden_mutation_flags_armed_under_ci

    armed = golden_mutation_flags_armed_under_ci(os.environ)
    if armed:
        raise pytest.UsageError(
            "golden update/regen flags are forbidden under CI: " + ", ".join(armed)
        )

    output_root = config._tmp_path_factory.getbasetemp() / "torchlens-generated"
    TEST_OUTPUTS_DIR = str(output_root)
    REPORTS_DIR = str(output_root / "reports")
    VIS_OUTPUT_DIR = str(output_root / "visualizations")
    config._tl_prior_test_outputs_dir = os.environ.get("TORCHLENS_TEST_OUTPUTS_DIR")
    os.environ["TORCHLENS_TEST_OUTPUTS_DIR"] = TEST_OUTPUTS_DIR
    # Arm the collapse/sibling-order verification tripwire for OUR suite.
    # The library no longer keys strictness on the ambient PYTEST_CURRENT_TEST
    # marker (r3 b7-opus R47-A: a downstream project's pytest rendering a
    # TorchLens graph must never inherit our hard-assert mode), so the
    # TorchLens suite opts in through the torchlens-owned knob explicitly.
    config._tl_prior_collapse_strict = os.environ.get("TORCHLENS_COLLAPSE_STRICT")
    os.environ.setdefault("TORCHLENS_COLLAPSE_STRICT", "1")
    config._tl_warn_once_sentinel_specs = _WARN_ONCE_SENTINELS
    # Pay PyTorch's one-time RNG and deterministic-mode initialization during
    # session setup, not against whichever smoke test happens to run first.
    torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)
    # Pay TorchLens's one-time capture-machinery cost (lazy wrap_torch install,
    # dispatcher/completeness tables) at session setup too: under randomized
    # ordering, whichever test captured FIRST was charged ~5s of one-time CPU
    # and sporadically tripped its duration budget — the exact noise the old
    # 15s budget crutch existed to absorb. Semantically equivalent to "some
    # early test captured" (wrappers stay installed until explicit unwrap),
    # which every full-suite run already implies. Skipped for collect-only
    # sessions, which never run a capture.
    # ...and skipped when a golden-update flag is armed: in-process golden
    # families guard that generation starts on UNWRAPPED torch (SF-53), and
    # this warmup capture would trip that guard before any test ran, making
    # the documented single-family regen recipe impossible to execute.
    # r7 R77 (fable b2 MED): match EVERY declared golden-flag prefix, not one
    # hardcoded spelling -- the TORCHLENS_REGEN_ families (export goldens) are
    # SF-53 wrap-state-guarded too, and the UPDATE_-only carve-out left their
    # documented regen recipe hard-failing at its own guard.
    golden_update_armed = any(
        key.startswith(GOLDEN_FLAG_PREFIXES) and value == "1" for key, value in os.environ.items()
    )
    # A06 interim (packaging_requests.tsv row filed 2026-08-26): the
    # batchnorm_train_stats_mutated disclosure is once-per-process in the
    # package but the autouse sentinel reset re-arms it per test, and
    # train-mode BatchNorm fixtures are a commonly exercised idiom -- like the
    # save_mode='reference' caveat it is visible-not-fatal and asserted
    # locally (tests/test_capopts_truth_batchnorm_warn.py resets the flag
    # first). DROP this line when the pyproject filterwarnings row lands
    # (later entries win, so this appended default overrides the broad
    # error::UserWarning promotion).
    config.addinivalue_line(
        "filterwarnings",
        "default:tracing runs the model's REAL forward.*:UserWarning:torchlens(\\.|$)",
    )
    if not config.option.collectonly and not golden_update_armed:
        import warnings as _warnings

        import torchlens as _tl

        # This warmup capture is a pure performance convenience (pay the
        # one-time wrapper-install cost here, not against whichever test
        # captures first); it carries no correctness contract of its own.
        # A raise here used to escape `pytest_configure` -- a hook pytest
        # runs outside any test item's exception handling -- and crash the
        # whole session with INTERNALERROR (exit 3) instead of a normal
        # test failure. That made a capture-breaking bug here UNKILLABLE by
        # the mutation-margin suite: the session never got far enough to
        # run a single real test, so no FAILED/ERROR node id was ever
        # produced for the driver to attribute a kill to (mutation_driver.py
        # W2's `_run_step_8`/`_run_step_17` survivors). Swallow and report
        # instead: every other SUITE test that captures will still raise
        # and fail normally, which is the real, attributable kill.
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                _tl.trace(torch.nn.Linear(2, 2), torch.zeros(1, 2)).cleanup()
        except Exception as warmup_exc:  # noqa: BLE001 - see note above
            print(
                f"tests/conftest.py: warmup capture failed ({warmup_exc!r}); "
                "continuing unwarmed, the first real test pays the one-time cost",
            )
    _state._collect_usage_stats = False
    _state._function_call_counts.clear()
    _state._function_call_models.clear()


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore the caller's test-output environment after pytest exits.

    Parameters
    ----------
    config:
        Active pytest configuration.
    """

    prior = getattr(config, "_tl_prior_test_outputs_dir", None)
    if prior is None:
        os.environ.pop("TORCHLENS_TEST_OUTPUTS_DIR", None)
    else:
        os.environ["TORCHLENS_TEST_OUTPUTS_DIR"] = prior
    prior_strict = getattr(config, "_tl_prior_collapse_strict", None)
    if prior_strict is None:
        os.environ.pop("TORCHLENS_COLLAPSE_STRICT", None)
    else:
        os.environ["TORCHLENS_COLLAPSE_STRICT"] = prior_strict


# Tier duration budgets (see tests/test_marker_lint.py). The r3 re-tier
# landed, so the ENFORCEMENT budget now equals the documented PARTITION
# boundary: smoke and unmarked tests must fit 5s, heavy 20s (each load-scaled
# below; the pre-r3 15s crutch let a 59s test stay smoke under sprint load —
# R41 b2 opus+sol). `slow` is unbounded, `rare` only runs on request, and
# `serial` is NOT exempt: it resolves its heavy/smoke/unmarked budget
# normally (r7 R41 sol LOW: this comment used to claim the opposite of
# `_duration_budget_tier` and tests/AGENTS.md — the code is the contract).
#
# CHARGED TIME (round-4, the load-flake fix): a test is charged
# min(wall seconds, CPU seconds incl. subprocess children). Wall alone
# false-fails under parallel orchestrator load (r3settle: the same tests read
# 2.9-14.2s quiet but 15.6-34.9s loaded); CPU alone false-fails multithreaded
# torch ops (intra-op threads make CPU exceed wall several-fold on a quiet
# box). Requiring BOTH measures to exceed the budget is robust to each: a
# load-inflated test keeps its true CPU cost, a multithreaded test keeps its
# true wall cost, and a genuinely over-budget test exceeds both. Accepted
# residual: a test that mostly SLEEPS (low CPU, high wall) is no longer
# catchable — the partition boundary is about compute cost, and a sleeping
# test's wall time carries no load-independent meaning.
SMOKE_DURATION_BUDGET_SECONDS = 5.0
HEAVY_DURATION_BUDGET_SECONDS = 20.0
#: Absolute enforcement grace added on top of the load-scaled budget. The
#: charged window unavoidably absorbs BOUNDARY NOISE that belongs to no test:
#: deferred GC of earlier tests' traces and prior-module fixture teardown both
#: run inside whatever protocol window they happen to land in (measured: a
#: pure-AST lint test read 5.6s in one shuffled composition and 0.4s alone).
#: A small absolute grace kills that flap while a genuinely mis-tiered test
#: (the 59s smoke incident) still trips by an order of magnitude. This is an
#: enforcement tolerance on the partition boundary, not a new boundary — and
#: never the pre-r3 15s crutch (3x the budget); it is documented in the
#: budget sentence the docs-lockstep gate parses.
DURATION_BUDGET_GRACE_SECONDS = 2.0
#: Per-parametrize-cell allowance for a smoke family's aggregate budget: a
#: family's cost legitimately scales with its cell count (278 selector cells
#: at ~57ms/cell), so the aggregate bar is max(2x the per-test budget,
#: this allowance x n_cells) — genuine per-cell ballooning still trips it.
SMOKE_FAMILY_PER_CELL_SECONDS = 0.1


#: Session-frozen load factor (r7 R41, fixwave-7): the factor was recomputed
#: PER ITEM from instantaneous loadavg, so the same test passed or failed on
#: momentary run-queue pressure and could raise its own budget by finishing
#: during a load spike. One reading at first use now holds for the whole
#: session — deterministic within a run, still load-aware across runs.
_SESSION_LOAD_FACTOR: float | None = None


def _smoke_budget_load_factor() -> float:
    """Scale the wall-clock budget by CPU oversubscription, frozen per session.

    Wall-clock durations inflate roughly with run-queue pressure; a fixed
    budget false-trips whenever an orchestrator runs sibling lanes on the same
    box (measured 2026-08-13: the same four tests read 2.9-14.2s quiet but
    15.6-34.9s at loadavg ~5x nproc). Capped so a pathological load reading
    can never disarm the lint entirely, and frozen at its first reading so
    per-item load wobble cannot flip verdicts mid-session.
    """

    global _SESSION_LOAD_FACTOR
    if _SESSION_LOAD_FACTOR is not None:
        return _SESSION_LOAD_FACTOR
    try:
        load_per_cpu = os.getloadavg()[0] / max(os.cpu_count() or 1, 1)
    except OSError:  # pragma: no cover - getloadavg unsupported on the platform.
        load_per_cpu = 1.0
    _SESSION_LOAD_FACTOR = min(max(load_per_cpu, 1.0), 4.0)
    return _SESSION_LOAD_FACTOR


def _duration_budget_tier(item: pytest.Item) -> tuple[str, float] | None:
    """Return the duration-budget tier for one collected item.

    The budget is TWO-directional (b2 3-lab: 217-391 unmarked files escaped
    it entirely): an UNMARKED test runs in the mid/phase backstops, so it is
    held to the same 5s partition boundary as smoke — if it needs longer it
    needs a `heavy` or `slow` marker, chosen consciously.

    Parameters
    ----------
    item:
        Collected test item.

    Returns
    -------
    tuple[str, float] | None
        ``(tier_name, base_budget_seconds)``, or ``None`` for exempt tiers
        (``slow`` unbounded, ``rare`` request-only).

    ``serial`` is deliberately NOT exempt (b2 R41 round 5): it means
    "load-sensitive, run away from parallel worker load", never "unbudgeted"
    — the former blanket exemption let any unmarked test dodge the 5s
    partition boundary by adding one ``@pytest.mark.serial``, with the
    marker's isolation claim enforced by nothing. A serial item resolves its
    heavy/smoke/unmarked budget normally; the load-factor scaling can only
    help it.
    """

    for exempt in ("slow", "rare"):
        if item.get_closest_marker(exempt) is not None:
            return None
    if item.get_closest_marker("heavy") is not None:
        return ("heavy", HEAVY_DURATION_BUDGET_SECONDS)
    if item.get_closest_marker("smoke") is not None:
        return ("smoke", SMOKE_DURATION_BUDGET_SECONDS)
    return ("unmarked", SMOKE_DURATION_BUDGET_SECONDS)


def _process_cpu_seconds() -> float:
    """Return cumulative CPU seconds of this process AND its waited children.

    ``os.times()`` sums user+system for the process (all threads) plus the
    user+system of terminated, waited-for children, so subprocess-heavy tests
    are charged their real compute cost too.
    """

    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Iterator[pytest.TestReport]:
    """Accumulate per-phase wall durations for the tier duration budget."""

    report = yield
    durations = getattr(item, "_tl_phase_durations", None)
    if durations is None:
        durations = {}
        item._tl_phase_durations = durations
    durations[report.when] = report.duration
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[object]:
    """Record tests whose CHARGED time exceeds their tier budget.

    A static lint cannot know runtimes, so a slow test landing in a bounded
    tier is only catchable at runtime. Offenders are stashed on the session
    and asserted empty by ``test_marker_lint.py`` (ordered last), which names
    each offender, its tier, and both measured durations.

    Charged time is ``min(wall, cpu)`` — see the budget-constant comment
    block above for why either measure alone false-fails (wall under
    orchestrator load, CPU under torch intra-op threading).
    """

    cpu_before = _process_cpu_seconds()
    result = yield
    # Coverage-instrumented sessions are provably outside the budget
    # contract: instrumentation slows every test by design, and the
    # partition boundary is about UNINSTRUMENTED compute cost (r7 R41/R72:
    # the nightly coverage job used to --deselect ONE budget test by a name
    # that had since been renamed, so the real assertions ran instrumented
    # anyway and the always-on sessionfinish tripwire could not be
    # deselected at all). Every uninstrumented leg still records + enforces.
    if bool(getattr(item.config.option, "cov_source", None)):
        return result
    cpu_seconds = _process_cpu_seconds() - cpu_before
    wall_seconds = sum(getattr(item, "_tl_phase_durations", {}).values())
    charged = min(wall_seconds, cpu_seconds)
    load_factor = _smoke_budget_load_factor()
    tier_budget = _duration_budget_tier(item)
    if tier_budget is not None and tier_budget[0] in {"smoke", "unmarked"}:
        # Aggregate family budgets cover every 5s-bounded tier (smoke AND
        # unmarked — R41-4: untiered families previously had no aggregate
        # bound at all), charged on the same min(wall, cpu) measure.
        family = getattr(item, "originalname", None) or item.name.split("[")[0]
        family_stats = getattr(item.session, "_tl_smoke_family_stats", None)
        if family_stats is None:
            family_stats = {}
            item.session._tl_smoke_family_stats = family_stats
        family_key = f"{item.path}::{family}"
        family_total, family_count = family_stats.get(family_key, (0.0, 0))
        family_total += charged
        family_count += 1
        family_stats[family_key] = (family_total, family_count)
        family_budget = (
            load_factor
            * max(
                2.0 * SMOKE_DURATION_BUDGET_SECONDS,
                SMOKE_FAMILY_PER_CELL_SECONDS * family_count,
            )
            + DURATION_BUDGET_GRACE_SECONDS
        )
        family_budgets = getattr(item.session, "_tl_smoke_family_budgets", None)
        if family_budgets is None:
            family_budgets = {}
            item.session._tl_smoke_family_budgets = family_budgets
        family_budgets[family_key] = family_budget
    if tier_budget is None:
        return result
    tier, base_budget = tier_budget
    budget = base_budget * load_factor + DURATION_BUDGET_GRACE_SECONDS
    if charged > budget:
        offenders = getattr(item.session, "_tl_duration_budget_offenders", None)
        if offenders is None:
            offenders = []
            item.session._tl_duration_budget_offenders = offenders
        offenders.append((item.nodeid, tier, wall_seconds, cpu_seconds, budget))
    return result


def _duration_budget_failure_lines(session: pytest.Session) -> list[str]:
    """Render every recorded duration-budget violation for one session.

    Covers both the per-item ledger and the aggregate 5s-tier family stats,
    mirroring exactly what ``tests/test_marker_lint.py`` asserts.
    """

    lines = [
        f"{nodeid} [{tier}]: wall {wall:.1f}s / cpu {cpu:.1f}s "
        f"(budget {budget:.0f}s on min(wall, cpu))"
        for nodeid, tier, wall, cpu, budget in getattr(session, "_tl_duration_budget_offenders", [])
    ]
    family_budgets = getattr(session, "_tl_smoke_family_budgets", {})
    for family, (total, count) in getattr(session, "_tl_smoke_family_stats", {}).items():
        budget = family_budgets.get(family, float("inf"))
        if total > budget:
            lines.append(
                f"{family}: {total:.1f}s charged over {count} cells (family budget {budget:.1f}s)"
            )
    return lines


def _enforce_duration_budget_at_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Always-on duration-budget tripwire (r7 R41, sol b2 HIGH).

    Enforcement used to live ONLY in
    ``test_marker_lint.py::test_bounded_tier_tests_stay_within_duration_budget``,
    so any targeted invocation that did not collect that file exited GREEN
    over budget — contradicting the tests/AGENTS.md sentence "checked at the
    end of every session" and removing the tripwire from the documented
    per-step targeted workflow. The marker-lint test remains the rich
    reporting surface inside gate runs; this helper (called from the module's
    single ``pytest_sessionfinish`` hook) makes the session exit non-zero
    even when that file was never collected.
    """

    if exitstatus != 0:
        return  # already failing (incl. the marker-lint assertion itself)
    lines = _duration_budget_failure_lines(session)
    if not lines:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    header = (
        "duration-budget tripwire (tests/conftest.py::pytest_sessionfinish): "
        "tests exceeded their tier budget this session"
    )
    if reporter is not None:
        reporter.write_line(header, red=True)
        for line in lines:
            reporter.write_line("  " + line, red=True)
    else:  # pragma: no cover - headless embedding without a terminal reporter
        print(header + "\n  " + "\n  ".join(lines))
    session.exitstatus = 1


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Iterator[pytest.CollectReport]:
    """Record test-module import/collection wall AND CPU time for enforcement.

    Parameters
    ----------
    collector:
        Collector whose work is about to run.
    """

    started = time.perf_counter()
    cpu_before = _process_cpu_seconds()
    report = yield
    if isinstance(collector, pytest.Module):
        durations = getattr(collector.session, "_tl_module_collection_durations", None)
        if durations is None:
            durations = {}
            collector.session._tl_module_collection_durations = durations
        durations[str(collector.path)] = (
            time.perf_counter() - started,
            _process_cpu_seconds() - cpu_before,
        )
    return report


def _is_full_usage_stats_run(config: pytest.Config) -> bool:
    """Return whether pytest selected the complete default non-rare suite.

    Parameters
    ----------
    config:
        Active pytest configuration.

    Returns
    -------
    bool
        ``True`` only for an unfiltered invocation rooted at ``tests/``.
    """

    if config.option.keyword:
        return False
    mark_expression = (config.option.markexpr or "").strip().replace("(", "").replace(")", "")
    # The coverage audit asserts called-functions ⊆ ArgSpec entries, so any
    # broad subset is sound (a smaller run can only check less, never lie).
    # Arm it on the nightly fast tier too: with only {"", "not rare"} accepted
    # no CI invocation ever collected stats and the gate skipped in 100% of CI
    # runs (b10 R79 / opus-R79-1). Clause-set comparison, not literal strings
    # (b2 R41/B2R5-15): the docs teach `-m "not rare and not slow"` and the
    # former string set accepted only the other word order, so both DOCUMENTED
    # backstop spellings silently disarmed the gate.
    clauses = frozenset(
        clause.strip() for clause in mark_expression.split(" and ") if clause.strip()
    )
    if not clauses <= {"not rare", "not slow", "not heavy"}:
        return False
    requested_paths = [Path(str(arg).split("::", maxsplit=1)[0]).resolve() for arg in config.args]
    return requested_paths == [Path(TESTS_DIR).resolve()]


def pytest_collection_finish(session: pytest.Session) -> None:
    """Freeze the import-time heap so per-test gc cost is O(session-created objects).

    Collection imports every selected test module up front, so a full session
    starts with millions of live tracked objects (measured 2026-08-18: 5.17M at
    test #1 of a 1,000-test smoke slice vs 445k isolated). Every full-heap gc
    pass -- the explicit ``gc.collect()`` calls in weakref/leak tests AND
    CPython's automatic threshold gen-2 collections -- scans that heap at
    O(live objects), charging 4-9s of pure CPU to whichever test happens to be
    running (69.9s of gen-2 gc in a 210s slice; the duration tripwire's
    rotating "flakes" were exactly these). Freezing after one settling collect
    moves the permanent import-time heap into gc's frozen generation, which
    collections never scan, making per-test gc cost proportional to what the
    session has created since -- ~200k objects per 1,000 tests, milliseconds.

    Isolation is unchanged: freezing happens BEFORE the first test, so every
    test-created object stays in the scanned generations and remains fully
    collectable; frozen objects are import-time state that was never eligible
    for collection while the session lives. Modules imported lazily mid-session
    land post-freeze and are simply scanned as normal.

    The shared package-source corpus (``tests/_source_corpus.py``) is
    prewarmed here, before the freeze, iff a consumer test module was
    imported during collection: built lazily mid-session its ~270 MB of AST
    nodes would be scanned by every later gen-2 collection — exactly the
    per-test gc drag this freeze exists to kill.
    """

    del session
    corpus = sys.modules.get("_source_corpus")
    if corpus is not None:
        corpus.prewarm()
    gc.collect()
    gc.freeze()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Order the ArgSpec coverage test last; skip assertion-dependent tests under -O.

    ``python -O`` strips every ``assert``, which disables the postprocess contract
    audit outright -- arming it then raises rather than silently reporting a clean
    audit that verified nothing. Tests that arm the audit therefore cannot run
    under ``-O`` and must SKIP, not fail, so the ``-O`` leg stays a meaningful
    verdict-identity check on everything else.
    """

    if not __debug__:
        skip_no_assertions = pytest.mark.skip(
            reason="requires assertions; the postprocess audit cannot run under python -O"
        )
        for item in items:
            if item.get_closest_marker("requires_assertions") is not None:
                item.add_marker(skip_no_assertions)

    collect_usage_stats = _is_full_usage_stats_run(config)
    _state._collect_usage_stats = collect_usage_stats
    if collect_usage_stats:
        _state._function_call_counts.clear()
        _state._function_call_models.clear()

    skip_partial_usage_stats = pytest.mark.skip(
        reason="ArgSpec usage coverage requires an unfiltered full tests/ run"
    )
    coverage_tests = []
    lint_tests = []
    other_tests = []
    for item in items:
        if "test_arg_positions" in item.nodeid:
            if not collect_usage_stats:
                item.add_marker(skip_partial_usage_stats)
            coverage_tests.append(item)
        elif "test_marker_lint" in item.nodeid:
            # The duration-budget lint reads offenders recorded during the run,
            # so it must execute after every other test in the session.
            lint_tests.append(item)
        else:
            other_tests.append(item)
    items[:] = other_tests + coverage_tests + lint_tests
    # Pre-fill whole-tree scan caches during collection (uncharged time): a
    # module may expose `warm_scan_caches()` when its scanners' one-time parse
    # cost (~5-8s of genuine CPU) would otherwise land in whichever of its
    # tests runs first and sit on the duration-budget boundary. UNCONDITIONAL:
    # the budget tripwire enforces at sessionfinish in EVERY session, so a
    # targeted run of a scanner module (no marker-lint collected) must warm
    # too or its first test eats the parse cost and trips the always-on gate.
    # Only sessions that collected a warm-capable module pay the cost, and
    # they would pay it inside a charged test window otherwise.
    warmed: set[int] = set()
    for item in items:
        module = getattr(item, "module", None)
        warm = getattr(module, "warm_scan_caches", None)
        if warm is not None and id(module) not in warmed:
            warmed.add(id(module))
            warm()


def _coverage_requested(config: pytest.Config) -> bool:
    """Return whether this session explicitly requested pytest-cov collection.

    Parameters
    ----------
    config:
        Active pytest configuration.

    Returns
    -------
    bool
        ``True`` when pytest-cov is active for this session.
    """

    cov_source = getattr(config.option, "cov_source", None)
    return bool(cov_source)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Enforce the duration budget, then write coverage artifacts.

    Parameters
    ----------
    session:
        Active pytest session.
    exitstatus:
        Final pytest exit status.
    """

    _enforce_duration_budget_at_sessionfinish(session, exitstatus)
    del exitstatus
    _state._collect_usage_stats = False
    _state._function_call_counts.clear()
    _state._function_call_models.clear()
    config = session.config
    if config.option.collectonly or not _coverage_requested(config):
        return
    try:
        from coverage import Coverage
        from coverage.exceptions import NoDataError
    except ImportError:
        return

    try:
        cov = Coverage()
        cov.load()
        report_path = opj(REPORTS_DIR, "coverage_report.txt")
        with open(report_path, "w") as f:
            cov.report(file=f, show_missing=True, skip_empty=True)
        html_dir = opj(REPORTS_DIR, "coverage_html")
        cov.html_report(directory=html_dir, skip_empty=True)
    except (FileNotFoundError, NoDataError):
        return


# Fixtures


@pytest.fixture(autouse=True, scope="session")
def _isolated_torchlens_cache(tmp_path_factory: pytest.TempPathFactory):
    """Point ``TORCHLENS_CACHE_DIR`` at a fresh per-session directory.

    ``tl.trace(..., cache=True)`` without an explicit ``cache_dir=`` falls back to
    ``~/.cache/torchlens`` (read lazily from ``TORCHLENS_CACHE_DIR`` on every call).
    A shared on-disk cache makes cache-hit assertions order- and history-dependent:
    a stale entry left by an earlier session or another worktree turns a
    first-capture cache-miss assertion into a phantom failure. Every test session
    gets its own empty cache root instead; the prior environment is restored on
    teardown so the suite never leaks state into the invoking shell.
    """

    prior = os.environ.get("TORCHLENS_CACHE_DIR")
    os.environ["TORCHLENS_CACHE_DIR"] = str(tmp_path_factory.mktemp("torchlens_cache"))
    yield
    if prior is None:
        os.environ.pop("TORCHLENS_CACHE_DIR", None)
    else:
        os.environ["TORCHLENS_CACHE_DIR"] = prior


def _copy_sentinel_value(value: object) -> object:
    """Return an independent snapshot of a supported warn-once value.

    Parameters
    ----------
    value:
        Boolean, set, or dictionary sentinel value.

    Returns
    -------
    object
        Independent mutable copy, or the original immutable value.
    """

    if isinstance(value, (set, dict, weakref.WeakSet)):
        return value.copy()
    return value


def _set_sentinel_default(module: ModuleType, name: str, default: object) -> None:
    """Reset one imported warn-once sentinel to its cold-process value.

    Parameters
    ----------
    module:
        Imported module that owns the sentinel.
    name:
        Module attribute name.
    default:
        Cold-process sentinel value.
    """

    _assign_sentinel(module, name, _copy_sentinel_value(default))


def _assign_sentinel(module: ModuleType, name: str, value: object) -> None:
    """Write a sentinel value, mutating container sentinels IN PLACE.

    Rebinding a set/dict/WeakSet sentinel booby-traps every ``from module
    import NAME`` alias in a test module: the alias keeps the OLD object, so
    its reads and ``.clear()`` calls silently target dead state after the
    first fixture cycle (r7 R76/R77 b2 remainder — the
    ``_WARNED_DEPRECATIONS`` from-import in test_conditional_lifecycle was
    exactly this). Same-type containers are cleared and refilled instead;
    immutable sentinels (bools, None) still rebind.
    """

    current = getattr(module, name, _MISSING)
    if type(current) is type(value) and isinstance(current, (set, dict, weakref.WeakSet)):
        current.clear()
        current.update(value)
    else:
        setattr(module, name, value)


@pytest.fixture(autouse=True)
def _reset_warn_once_sentinels() -> Iterator[None]:
    """Give every test fresh warn-once state and restore its incoming state."""

    snapshots: dict[tuple[str, str], object] = {}
    for module_name, name, default in _WARN_ONCE_SENTINELS:
        module = sys.modules.get(module_name)
        if module is None:
            snapshots[(module_name, name)] = _MISSING
            continue
        snapshots[(module_name, name)] = _copy_sentinel_value(getattr(module, name, _MISSING))
        _set_sentinel_default(module, name, default)

    try:
        yield
    finally:
        for module_name, name, default in _WARN_ONCE_SENTINELS:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            prior = snapshots[(module_name, name)]
            if prior is _MISSING:
                _set_sentinel_default(module, name, default)
            else:
                _assign_sentinel(module, name, prior)


#: Public content registries that tests mutate through PUBLIC registration
#: APIs with no unregister spelling (r7 R76, sol b2 MED): a registered
#: container class or custom op rule was a permanent process-global, so
#: full-suite and targeted runs saw different registry state depending on
#: which tests had run first. Snapshot/restore per test, same lazy
#: sys.modules discipline as the warn-once sentinels. The third column
#: names the registry's own lock attribute (or None): the container
#: registry has a writer/iterator race lock that the restore must honor.
#:
#: r7 R76/R77 adjudication (fixwave-7): two same-day parallel lanes each
#: landed a twin of this fixture — one lazy but lockless, one locked but
#: force-importing ``capture.flops`` suite-wide. This is the MERGED single
#: fixture: lazy module lookup (flops stays unimported until a test needs
#: it) AND lock-disciplined container access.
_CONTENT_REGISTRIES: tuple[tuple[str, str, str | None], ...] = (
    ("torchlens.ir.container", "_CONTAINER_REGISTRY", "_CONTAINER_REGISTRY_LOCK"),
    ("torchlens.capture.flops", "_CUSTOM_OP_RULES", None),
)


@contextmanager
def _registry_lock(module: object, lock_name: str | None) -> Iterator[None]:
    """Hold the registry's own lock when it declares one."""

    if lock_name is None:
        yield
    else:
        with getattr(module, lock_name):
            yield


@pytest.fixture(autouse=True)
def _restore_content_registries() -> Iterator[None]:
    """Restore registered-container and custom-op-rule state after every test."""

    snapshots: dict[tuple[str, str], object] = {}
    for module_name, name, lock_name in _CONTENT_REGISTRIES:
        module = sys.modules.get(module_name)
        if module is None:
            snapshots[(module_name, name)] = _MISSING
            continue
        with _registry_lock(module, lock_name):
            snapshots[(module_name, name)] = dict(getattr(module, name))
    try:
        yield
    finally:
        for module_name, name, lock_name in _CONTENT_REGISTRIES:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            prior = snapshots[(module_name, name)]
            if prior is _MISSING:
                # Module imported DURING the test: whatever it registered at
                # import time is legitimate baseline; drop only test-added
                # rows is impossible to distinguish, so leave as-is.
                continue
            with _registry_lock(module, lock_name):
                registry = getattr(module, name)
                registry.clear()
                registry.update(prior)


_CAPABILITY_DEPENDENT_CACHES: tuple[tuple[str, str], ...] = (
    # Second-layer lru_caches whose cached value DERIVES from a lazy HAS_*
    # capability probe (grind p5 §3.9: the b7fe953e class one layer down).
    # Restoring the _torch_compat latches alone leaves a value computed under
    # a stubbed runtime frozen in these caches for the whole process, so the
    # probe restore must clear them too. The census in
    # tests/test_marker_lint.py::test_capability_dependent_caches_are_cleared
    # keeps this list complete.
    ("torchlens._runnable_state_context", "_fake_tensor_mode_class"),
    ("torchlens._runnable_state_context", "_count_bounded_fake_tensor_mode_class"),
)


def _clear_capability_dependent_caches() -> None:
    """Clear every registered probe-derived lru_cache (loaded modules only)."""

    for module_name, attr in _CAPABILITY_DEPENDENT_CACHES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        getattr(module, attr).cache_clear()


@pytest.fixture(autouse=True)
def _restore_lazy_capability_probes() -> Iterator[None]:
    """Restore lazy ``HAS_*`` capability latches to their pre-test state.

    The lazy ``_torch_compat`` probes latch on first use, so a test that stubs
    ``sys.modules`` (or otherwise shims the runtime) while a probe fires
    poisons the flag for the whole process -- the recorded ``b7fe953e``
    incident class, previously patched per-test rather than systemically.
    Snapshotting before and restoring after every test bounds any mis-latch to
    the test that caused it; an un-latched probe simply re-probes on its next
    use, which is cheap and hits the real runtime. Second-layer caches built
    FROM a probed capability (``_CAPABILITY_DEPENDENT_CACHES``) are cleared in
    the same breath -- restoring the flag while a derived cache keeps the
    poisoned value would just move the incident one layer down.
    """

    from torchlens.utils import _torch_compat

    snapshot = _torch_compat.capability_probe_snapshot()
    try:
        yield
    finally:
        _torch_compat.restore_capability_probes(snapshot)
        _clear_capability_dependent_caches()


@pytest.fixture(autouse=True)
def _reset_rng_state() -> Iterator[None]:
    """Seed each test deterministically and restore all incoming RNG settings."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_initialized = torch.cuda.is_initialized()
    cuda_states = torch.cuda.get_rng_state_all() if cuda_initialized else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if cuda_initialized:
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)


@pytest.fixture
def default_input1():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input2():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input3():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input4():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def zeros_input():
    return torch.zeros(6, 3, 224, 224)


@pytest.fixture
def ones_input():
    return torch.ones(6, 3, 224, 224)


@pytest.fixture
def vector_input():
    return torch.rand(5)


@pytest.fixture
def input_2d():
    return torch.rand(5, 5)


@pytest.fixture
def input_complex():
    return (torch.complex(torch.rand(3, 3), torch.rand(3, 3)),)


@pytest.fixture
def small_input():
    return torch.rand(2, 3, 32, 32)


@pytest.fixture
def seq_input():
    """(seq_len, batch, embed_dim) for transformer models."""
    return torch.rand(10, 2, 16)


@pytest.fixture
def token_input():
    """Integer tokens for embedding models."""
    return torch.randint(0, 100, (2, 10))


@pytest.fixture
def input_3d():
    """Volumetric input for Conv3d models."""
    return torch.rand(1, 1, 4, 4, 4)


@pytest.fixture
def input_1d_seq():
    """1D sequence input for Conv1d models."""
    return torch.rand(2, 3, 16)
