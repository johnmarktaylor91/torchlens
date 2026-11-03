"""Root pytest configuration for the TorchLens suite.

Session setup (output directories, thread pinning, warm-ups), collection
ordering and the usage-stats run live here. Everything else is a plugin
module beside this file, registered through ``pytest_plugins``:

- ``_tiers_plugin``: tier markers, the smoke/heavy/slow/serial/rare partition
  check at collection and the CI parity note;
- ``_collection_warmup``: scan-cache and source-corpus warm-up, then ``gc.freeze``;
- ``_state_isolation``: autouse per-test state resets;
- ``_leak_guards``: capture-slot, thread and draw-outpath tripwires;
- ``_input_fixtures``: plain tensor inputs.

Duration budgets are a repository check over ``tests/tiers/durations.csv``
(``scripts/tier_durations.py check``), not a runtime tripwire.
"""

from __future__ import annotations

import os
import re
from os.path import join as opj
from pathlib import Path

import _state_isolation
import _xdist_groups
import pytest
import torch

# Snapshot the R1 offline-venue signature before any test module can set the flags.
from support import r1_venue  # noqa: F401

from torchlens import _state

pytest_plugins = [
    "_tiers_plugin",
    "_collection_warmup",
    "_state_isolation",
    "_leak_guards",
    "_input_fixtures",
]

# Output directories are assigned under pytest's private basetemp in
# ``pytest_configure`` before test modules import these constants.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_OUTPUTS_DIR = ""
REPORTS_DIR = ""
VIS_OUTPUT_DIR = ""


#: Size ceiling of the smoke tier in collected tests; ``tests/tiers/collection.json``
#: records the measured count and ``tests/test_tiers_artifact.py`` compares the two.
SMOKE_TIER_SIZE_CEILING = 400


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Initialize session-private outputs with shipped usage stats disabled.

    Parameters
    ----------
    config:
        Active pytest configuration.
    """

    global TEST_OUTPUTS_DIR, REPORTS_DIR, VIS_OUTPUT_DIR

    # A CI run must never mutate goldens: an armed update/regen/record flag
    # would silently rebaseline instead of verifying.
    from _oracle_env import GOLDEN_FLAG_PREFIXES, golden_mutation_flags_armed_under_ci

    armed = golden_mutation_flags_armed_under_ci(os.environ)
    if armed:
        raise pytest.UsageError(
            "golden update/regen flags are forbidden under CI: " + ", ".join(armed)
        )

    # The first float ``torch.tanh`` of a process can compute one intra-op
    # thread's chunk by a different kernel path (later calls never do), so a
    # test whose capture or plain forward is its worker's first tanh can see a
    # few 1e-5 of drift. One warm-up over every intra-op thread keeps that
    # first call out of every test; each xdist worker runs this hook.
    torch.tanh(torch.zeros(max(torch.get_num_threads(), 1), 32768))

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
    config._tl_warn_once_sentinel_specs = _state_isolation._WARN_ONCE_SENTINELS
    _pin_test_threads(config)
    # pytest 9.2 + xdist 3.8 drop ``xdist_group`` under ``--dist loadgroup``
    # (tests/_xdist_groups.py); restamp the group on the id pytest reads.
    _xdist_groups.register(config)
    # Pay PyTorch's one-time RNG and deterministic-mode initialization during
    # session setup, not against whichever smoke test happens to run first.
    torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)
    # Pay TorchLens's one-time capture-machinery cost (lazy wrap_torch install,
    # dispatcher/completeness tables) at session setup too, so no test is
    # charged for it; wrappers stay installed until an explicit unwrap, which
    # every full-suite run already implies. Skipped for collect-only sessions
    # and whenever any golden-update flag is armed: in-process golden families
    # require generation to start on UNWRAPPED torch (SF-53).
    golden_update_armed = any(
        key.startswith(GOLDEN_FLAG_PREFIXES) and value == "1" for key, value in os.environ.items()
    )
    # The batchnorm_train_stats_mutated disclosure is once-per-process in the
    # package but the autouse sentinel reset re-arms it per test; keep it
    # visible-not-fatal (asserted in tests/test_capopts_truth_batchnorm_warn.py).
    # Later filterwarnings entries win, so this overrides error::UserWarning.
    config.addinivalue_line(
        "filterwarnings",
        "default:tracing runs the model's REAL forward.*:UserWarning:torchlens(\\.|$)",
    )
    if not config.option.collectonly and not golden_update_armed:
        import warnings as _warnings

        import torchlens as _tl

        # The warm-up is a performance convenience with no correctness
        # contract. A raise here escapes ``pytest_configure`` and ends the
        # session with INTERNALERROR before any test runs, so swallow and
        # report: every suite test that captures still fails attributably.
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
    _restore_test_threads(config)
    prior_strict = getattr(config, "_tl_prior_collapse_strict", None)
    if prior_strict is None:
        os.environ.pop("TORCHLENS_COLLAPSE_STRICT", None)
    else:
        os.environ["TORCHLENS_COLLAPSE_STRICT"] = prior_strict


TEST_THREADS_ENV = "TORCHLENS_TEST_THREADS"
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _test_thread_count() -> int:
    """Return the requested per-process torch thread count (default 1)."""

    raw = os.environ.get(TEST_THREADS_ENV, "1")
    try:
        threads = int(raw)
    except ValueError as exc:
        raise pytest.UsageError(
            f"{TEST_THREADS_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if threads < 1:
        raise pytest.UsageError(f"{TEST_THREADS_ENV} must be a positive integer, got {raw!r}")
    return threads


def _pin_test_threads(config: pytest.Config) -> None:
    """Set torch and the BLAS/OpenMP environment to the test thread count.

    The environment covers subprocess-spawning tests; the prior values are
    restored at unconfigure.
    """

    threads = _test_thread_count()
    config._tl_prior_thread_env = {name: os.environ.get(name) for name in _THREAD_ENV_VARS}
    config._tl_prior_torch_threads = torch.get_num_threads()
    for name in _THREAD_ENV_VARS:
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)


def _restore_test_threads(config: pytest.Config) -> None:
    """Undo ``_pin_test_threads``."""

    for name, prior in getattr(config, "_tl_prior_thread_env", {}).items():
        if prior is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior
    prior_threads = getattr(config, "_tl_prior_torch_threads", None)
    if prior_threads is not None:
        torch.set_num_threads(prior_threads)


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
    # The coverage audit asserts called-functions are a subset of the ArgSpec
    # entries, so any broad tier is sound (a smaller run checks less, never
    # lies). Compare the clause SET, not the literal string, so either word
    # order of the documented backstop spellings arms the gate.
    clauses = frozenset(
        clause.strip() for clause in mark_expression.split(" and ") if clause.strip()
    )
    # ``not serial`` too: CI runs the serial tests in a separate non-xdist step,
    # and excluding them from the xdist session must not disarm the gate.
    if not clauses <= {"not rare", "not slow", "not heavy", "not serial"}:
        return False
    requested_paths = [Path(str(arg).split("::", maxsplit=1)[0]).resolve() for arg in config.args]
    return requested_paths == [Path(TESTS_DIR).resolve()]


#: A test id embedding an object address (the default repr of a lambda or a
#: plain object used as a parametrize value) differs between processes, so
#: pytest-xdist workers collect "different tests" and the whole run errors.
_UNSTABLE_NODEID_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _reject_process_unstable_nodeids(items: list[pytest.Item]) -> None:
    """Fail collection when any test id carries a per-process object address.

    Parameters
    ----------
    items:
        Collected test items.

    Raises
    ------
    pytest.UsageError
        Naming every unstable id; give the parametrization explicit ``ids=``.
    """

    unstable = [item.nodeid for item in items if _UNSTABLE_NODEID_RE.search(item.nodeid)]
    if unstable:
        raise pytest.UsageError(
            "test ids embed a per-process object address (breaks pytest-xdist and "
            "-k selection); give these parametrizations explicit ids=: " + ", ".join(unstable)
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Order the ArgSpec coverage test last; skip assertion-dependent tests under -O.

    ``python -O`` strips every ``assert``, which disables the postprocess contract
    audit outright -- arming it then raises rather than silently reporting a clean
    audit that verified nothing. Tests that arm the audit therefore cannot run
    under ``-O`` and must SKIP, not fail, so the ``-O`` leg stays a meaningful
    verdict-identity check on everything else.
    """

    _reject_process_unstable_nodeids(items)

    from known_failures import apply_xfail_marks

    apply_xfail_marks(items)

    # ``cuda`` capability marker: the GPU leg selects these with ``-m cuda``;
    # without a device they skip here even if a site forgot its own skipif.
    if not torch.cuda.is_available():
        skip_no_cuda = pytest.mark.skip(reason="needs a CUDA device (cuda marker)")
        for item in items:
            if item.get_closest_marker("cuda") is not None:
                item.add_marker(skip_no_cuda)

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
    """Reset usage stats, then write coverage artifacts when pytest-cov ran.

    Parameters
    ----------
    session:
        Active pytest session.
    exitstatus:
        Final pytest exit status.
    """

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
