"""Order-isolation infrastructure guards (R76, reopened 2026-08-15).

The suite's order-isolation story leans on ``pytest-randomly``: randomized
default ordering surfaces order-coupling, and every ``-p no:randomly`` in the
drivers/regen paths deliberately opts back into deterministic order. All of
that is REAL only while the plugin is actually present — the reopened R76
finding was that the plugin silently vanished from the test extra, turning
every ``-p no:randomly`` into a no-op and the "randomized" default into plain
definition order. These guards make that failure mode loud:

* the declaration guard pins the extra (a re-removal goes red immediately);
* the registration guard proves the plugin is live in environments that have
  it (an installed-but-not-loaded plugin would silently un-randomize);
* the disable-flag guard proves ``-p no:randomly`` genuinely stabilizes
  ordering while the default genuinely shuffles (red-capable in both
  directions: a no-op flag or a no-op shuffle fails it).

Environments without the plugin (partial installs) skip the live guards but
never the declaration guard; the skip-audit ledger tracks the plugin as a
test-extra-tier dependency.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from test_skip_audit import FULL_TEST_EXTRA_SENTINEL

_REPO_ROOT = Path(__file__).resolve().parents[1]

_HAS_RANDOMLY = importlib.util.find_spec("pytest_randomly") is not None

#: Shares the skip audit's FULL_TEST_EXTRA_SENTINEL (never a second, driftable
#: literal here -- a canary leg that installs "timm" directly without the
#: rest of [test] previously made the old hardcoded "timm" check here falsely
#: claim full coverage, same as the skip audit's own sentinel, round-2 CI
#: triage 2026-10-01): when the environment CLAIMS the full test extra, a
#: missing pytest-randomly is INSTALL BREAKAGE, not a legitimate partial
#: environment -- the live guards must then FAIL, never skip (3.16 reopened
#: row: the declaration landed but every guard skipped everywhere, so the
#: gate layer stayed unarmed even on full installs).
_CLAIMS_FULL_TEST_EXTRA = importlib.util.find_spec(FULL_TEST_EXTRA_SENTINEL) is not None


def _require_randomly_or_skip() -> None:
    """Skip on genuine partial environments; FAIL on full-extra installs."""

    if _HAS_RANDOMLY:
        return
    if _CLAIMS_FULL_TEST_EXTRA:
        pytest.fail(
            "this environment carries the full [test] extra (sentinel import "
            "succeeded) but pytest-randomly is missing: the order-isolation "
            "gate layer is silently unarmed on an environment that promised "
            "it (R76 reopened, 3.16 #4). Reinstall the [test] extra."
        )
    else:
        # Branch shape keeps this a CONDITIONAL skip for the skip audit's
        # unconditional-skip scanner (it deliberately ignores early returns).
        pytest.skip("pytest-randomly not installed (partial environment); declaration guard ran")


def _test_extra_deps(pyproject_text: str) -> list[str]:
    """Extract the [test] extra's dependency strings from pyproject source.

    Line-based on purpose: ``tomllib`` only exists on 3.11+ and the suite
    floor is lower, while the extra's shape (one quoted dependency per line
    inside ``test = [ ... ]``) is pinned by the file's own style.

    Parameters
    ----------
    pyproject_text:
        Raw pyproject.toml contents.

    Returns
    -------
    list[str]
        Dependency requirement strings declared in the test extra.
    """

    deps: list[str] = []
    in_extra = False
    for line in pyproject_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("test = ["):
            in_extra = True
            continue
        if in_extra:
            if stripped.startswith("]"):
                break
            if stripped.startswith('"'):
                deps.append(stripped.strip('",'))
    return deps


def test_pytest_randomly_declared_in_test_extra() -> None:
    """The [test] extra must declare pytest-randomly (R76 regression pin)."""

    test_extra = _test_extra_deps((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert test_extra, "failed to locate the [test] extra in pyproject.toml"
    assert any(
        dep.split("[")[0].split(">=")[0].split("==")[0].strip() == "pytest-randomly"
        for dep in test_extra
    ), (
        "pytest-randomly is missing from the [test] extra again: every "
        "`-p no:randomly` in the suite and the mutation driver is a silent "
        "no-op without it, and default runs stop randomizing (R76, reopened "
        "2026-08-15 after 9810a3d7)"
    )


def test_randomly_plugin_registration_matches_invocation(request: pytest.FixtureRequest) -> None:
    """When installed, the plugin must be live unless explicitly disabled."""

    _require_randomly_or_skip()
    disabled = any(
        arg == "no:randomly" for arg in request.config.invocation_params.args
    ) or "no:randomly" in getattr(request.config.option, "plugins", [])
    registered = request.config.pluginmanager.hasplugin("randomly")
    if disabled:
        assert not registered, "-p no:randomly was given but the plugin still registered"
    else:
        assert registered, (
            "pytest-randomly is importable but not registered: the default run "
            "is silently un-randomized (broken entry point or stray disable)"
        )


_MINI_SUITE = "".join(f"def test_case_{index}():\n    pass\n\n\n" for index in range(12))


def _collect_order(suite_dir: Path, extra_args: list[str]) -> list[str]:
    """Return the collected node-id order of a planted 12-test mini-suite.

    The probe deliberately collects a MINI-SUITE outside the repo rather than
    a real test module: collecting tests/ loads the root conftest (a full
    torch import) and its collection-time scan-cache warmers — measured at
    ~24s PER SUBPROCESS, 72s for the three probe runs. The property under
    test is pytest-randomly's flag behavior, which is module-agnostic.

    Parameters
    ----------
    suite_dir:
        Directory holding the planted mini-suite (a ``test_probe.py``).
    extra_args:
        Ordering-relevant pytest flags for this collection run.

    Returns
    -------
    list[str]
        Collected node ids in session order.
    """

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_probe.py",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=suite_dir,
    )
    order = [line for line in proc.stdout.splitlines() if "::" in line]
    assert order, f"collection produced no items:\n{proc.stdout}\n{proc.stderr}"
    return order


@pytest.mark.heavy
def test_no_randomly_flag_actually_disables_shuffling(tmp_path: Path) -> None:
    """``-p no:randomly`` must stabilize order; the default must shuffle.

    Red-capable in both directions: if the flag is a silent no-op (the
    reopened R76 state) the two disabled runs diverge under different seeds;
    if shuffling itself is broken the seeded run matches definition order.
    """

    _require_randomly_or_skip()
    suite_dir = tmp_path
    (suite_dir / "test_probe.py").write_text(_MINI_SUITE, encoding="utf-8")
    # A blocked plugin contributes no CLI options, so the disabled runs carry
    # no seed flag: were the flag a no-op (plugin still live), each run would
    # draw a fresh time-based seed and the two orders would diverge.
    disabled_a = _collect_order(suite_dir, ["-p", "no:randomly"])
    disabled_b = _collect_order(suite_dir, ["-p", "no:randomly"])
    assert disabled_a == disabled_b, (
        "-p no:randomly did not produce a stable order across runs — the "
        "disable flag is not actually disabling the plugin"
    )
    shuffled = _collect_order(suite_dir, ["--randomly-seed=1"])
    assert sorted(shuffled) == sorted(disabled_a)
    assert shuffled != disabled_a, (
        "the seeded default run did not shuffle the mini-suite — randomized "
        "ordering is silently inert"
    )
    assert shuffled != disabled_a, (
        "a seeded default run produced definition order — shuffling is not "
        "actually happening (order-isolation coverage is fictional)"
    )


def test_content_registries_are_restored_between_tests(
    request: pytest.FixtureRequest,
) -> None:
    """r7 R76 (sol b2 MED): public-registry mutations must not outlive a test.

    ``register_op_rule`` and container registration are plain global
    assignments with no unregister spelling, so a registered test class was a
    permanent process-global -- full-suite and targeted runs diverged on
    registry state. The conftest autouse fixture snapshots and restores both
    content registries; this drives the fixture generator directly and
    proves the restore (red-capable: delete the fixture and this fails).
    """

    conftest_plugin = next(
        (
            plugin
            for plugin in request.config.pluginmanager.get_plugins()
            if hasattr(plugin, "_restore_content_registries")
        ),
        None,
    )
    assert conftest_plugin is not None, "conftest lost _restore_content_registries"
    fixture_fn = conftest_plugin._restore_content_registries
    raw = getattr(fixture_fn, "__wrapped__", None) or fixture_fn.__pytest_wrapped__.obj

    from torchlens.capture.flops import _CUSTOM_OP_RULES

    sentinel = "r76_isolation_probe_op"
    assert sentinel not in _CUSTOM_OP_RULES
    generator = raw()
    next(generator)
    _CUSTOM_OP_RULES[sentinel] = (lambda *_a: 1, None)
    assert sentinel in _CUSTOM_OP_RULES
    generator.close()  # fixture teardown restores the snapshot
    assert sentinel not in _CUSTOM_OP_RULES, (
        "the content-registry fixture failed to restore _CUSTOM_OP_RULES"
    )
