"""Every public function and class carries a runnable NumPy ``Examples`` section.

The docstring standard (docs/agent-reference/conventions.md, "Docstrings") puts one
runnable example on every public function and class. ``tools/docstring_examples.py``
lists those symbols from the lockstep inventory, reads each ``Examples`` section as
doctest examples and runs them in the ambient namespace the docs pages also assume
(``tl``, ``torch``, ``nn``, ``model``, ``x``). This gate fails when a public symbol has
no example and when an example raises or prints something other than what it shows.

The only escape hatch is :data:`PENDING`, an exact ledger of symbols whose example is
not written yet, each with a reason. It is checked both ways: an unledgered symbol
without an example fails, and a ledger entry whose symbol now carries an example (or no
longer exists) ALSO fails, so the ledger cannot rot into a permanent exemption. An
example that needs an optional dependency starts with ``>>> # requires: <module>`` and
is skipped, not failed, when that module is absent.

Each example captures a small model, so the family is ``heavy`` (worker and backstop
tiers, never smoke); one cell per package (the packages the committed census lists, read
without importing torchlens) keeps every cell inside the heavy budget. A stale census
fails here too, so the cell list cannot drift from the surface.
"""

from __future__ import annotations

import functools
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import docstring_examples  # noqa: E402

pytestmark = pytest.mark.heavy

#: Public symbols whose example is not written yet: ``public path -> reason``.
#: EMPTY is the intended steady state. Adding a row needs a reason and an owner; a row
#: whose symbol gains an example must leave in the same change (the staleness test).
PENDING: dict[str, str] = {}


@functools.lru_cache(maxsize=1)
def _symbols() -> tuple[docstring_examples.PublicSymbol, ...]:
    """Build the public symbol list once per process."""

    return tuple(docstring_examples.public_symbols())


def test_every_public_function_and_class_has_an_example() -> None:
    """Fail on every unledgered public function or class without a ``>>>`` example."""

    missing = [
        f"{symbol.path} ({symbol.module})"
        for symbol in _symbols()
        if not docstring_examples.example_of(symbol.obj) and symbol.path not in PENDING
    ]
    assert not missing, (
        f"{len(missing)} public symbols lack a runnable Examples section "
        "(docs/agent-reference/conventions.md, Docstrings):\n  " + "\n  ".join(missing)
    )


def test_pending_ledger_has_no_stale_entries() -> None:
    """Fail on a PENDING row whose symbol now has an example or no longer exists."""

    by_path = {symbol.path: symbol for symbol in _symbols()}
    stale = []
    for path in PENDING:
        symbol = by_path.get(path)
        if symbol is None:
            stale.append(f"{path}: no such public symbol")
        elif docstring_examples.example_of(symbol.obj):
            stale.append(f"{path}: has an example; remove the row")
    assert not stale, "stale PENDING rows:\n  " + "\n  ".join(stale)


def test_census_is_current() -> None:
    """The committed census matches a fresh render, so its package list stays true."""

    rendered = docstring_examples.render_census(
        list(_symbols()), docstring_examples.public_members()
    )
    committed = docstring_examples.CENSUS_PATH.read_text()
    assert committed == rendered, (
        "docs/maintainers/docstring_examples_census.md is stale: run "
        "`python tools/docstring_examples.py census` and commit the result"
    )


_MEMBERS_AROUND_A_ROUND_TRIP = textwrap.dedent(
    """
    import json
    import pathlib
    import sys

    import torch
    from torch import nn

    sys.path.insert(0, sys.argv[1])
    import docstring_examples

    import torchlens as tl


    def member_ids():
        rows = docstring_examples.public_members()
        return sorted(row["id"] for row in rows) + sorted(
            f"reached:{owner}.{row['name']}" for row in rows for owner in row["owner_names"]
        )


    fresh = member_ids()
    trace = tl.trace(nn.Linear(3, 2), torch.ones(1, 3))
    out_dir = pathlib.Path(sys.argv[2])
    trace.save(out_dir / "round_trip.tlspec")
    tl.load(out_dir / "round_trip.tlspec")
    tl.io.read_safetensors(tl.export.safetensors(trace, out_dir / "round_trip.safetensors"))
    print(json.dumps({"fresh": fresh, "after_round_trip": member_ids()}))
    """
)


def test_public_members_do_not_depend_on_a_prior_save_or_load(tmp_path: Path) -> None:
    """A fresh interpreter and one that ran a save/load round trip list the same members.

    Saving and loading (the bundle and the safetensors trace export, whose
    writer contract imports the kernel-telemetry module) import private modules
    a fresh ``import torchlens`` does not; a public member that appeared only
    after such an import made ``test_census_is_current`` red or green by test
    order. The subprocess keeps this test independent of whatever this worker
    imported before it.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _MEMBERS_AROUND_A_ROUND_TRIP,
            str(REPO_ROOT / "tools"),
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    members = json.loads(completed.stdout.strip().splitlines()[-1])
    fresh, after = set(members["fresh"]), set(members["after_round_trip"])
    assert fresh == after, (
        f"members only after a save/load: {sorted(after - fresh)}; "
        f"members only before: {sorted(fresh - after)}"
    )
    assert {"reached:Op.gpu_kernels", "reached:AtenOp.gpu_kernels"} <= fresh


#: Package cells whose examples build real small models for most symbols and so run past
#: the ``heavy`` per-test budget: a per-cell ``slow`` refinement of this heavy family
#: (``tests/_tiers_plugin.py``). ``mechinterp`` measured 40 s on a two-core worker.
SLOW_CELLS = frozenset({"mechinterp"})


def _cells() -> list[object]:
    """One parametrize value per census package, the slow ones marked as such."""

    return [
        pytest.param(package, marks=pytest.mark.slow) if package in SLOW_CELLS else package
        for package in docstring_examples.census_packages()
    ]


@pytest.mark.parametrize("package", _cells())
def test_every_example_runs(package: str) -> None:
    """Execute every example filed under ``package`` and report each failure."""

    symbols = [symbol for symbol in _symbols() if symbol.package == package]
    assert symbols, f"the census lists {package} but no public symbol is filed there"
    namespace = docstring_examples.ambient_namespace()
    failures = []
    for symbol in symbols:
        if symbol.path in PENDING:
            continue
        result = docstring_examples.run_example(symbol, namespace)
        if result.status == "failed":
            failures.append(f"{result.path}:\n{result.detail}")
    assert not failures, f"{len(failures)} examples failed:\n" + "\n".join(failures)


def _absent_optional() -> None:
    """Stand-in whose example needs a module that is never installed.

    Examples
    --------
    >>> # requires: torchlens_absent_optional_module
    >>> 1 / 0
    """


def _present_optional() -> None:
    """Stand-in whose example needs a module that always imports.

    Examples
    --------
    >>> # requires: json
    >>> json_text = "[1]"
    >>> json_text
    '[1]'
    """


def test_requires_marker_skips_when_module_is_absent() -> None:
    """A comment-only ``>>> # requires: X`` first line skips the example where X is absent."""

    symbol = docstring_examples.PublicSymbol(
        "tl.absent_optional", __name__, "function", _absent_optional
    )
    result = docstring_examples.run_example(symbol, {})
    assert (result.status, result.detail) == ("skipped", "torchlens_absent_optional_module")


def test_requires_marker_runs_when_module_is_present() -> None:
    """The same marker runs the example where X imports."""

    symbol = docstring_examples.PublicSymbol(
        "tl.present_optional", __name__, "function", _present_optional
    )
    assert docstring_examples.run_example(symbol, {}).status == "ok"


#: Optional modules the enforcing CI environment installs (``scripts/smoke_ci_parity.py``):
#: their ``# requires:`` examples run in the package cells above.
GATE_ENV_EXAMPLE_MODULES: frozenset[str] = frozenset(
    {
        "PIL",
        "graphviz",
        "lovely_tensors",
        "pandas",
        "pyarrow",
        "safetensors",
        "torchshow",
        "torchvision",
        "transformers",
    }
)

#: Optional modules the enforcing environment lacks. Their examples skip in the package
#: cells, so :func:`test_optional_examples_run` runs them, one cell per module, in the
#: legs that install the module (nightly ``bridges-real`` and ``lit-bridge``); a leg's
#: PASSED floor counts the cells it installs, so a skip there fails the leg.
OPTIONAL_EXAMPLE_MODULES: tuple[str, ...] = (
    "brainscore_core",
    "brainscore_vision",
    "captum",
    "dagua",
    "depyf",
    "dialz",
    "inseq",
    "lightning",
    "lit_nlp",
    "mcp",
    "pytorch_grad_cam",
    "repeng",
    "rsatoolbox",
    "sae_lens",
    "sentence_transformers",
    "steering_vectors",
    "tensorboard",
    "timm",
    "torchextractor",
    "treescope",
    "wandb",
    "xarray",
)


def _required_by(symbol: docstring_examples.PublicSymbol) -> str | None:
    """Return the optional module ``symbol``'s example declares, if any."""

    return docstring_examples._required_module(inspect.getdoc(symbol.obj) or "")


def test_optional_example_modules_are_listed() -> None:
    """Every ``# requires:`` module is listed once, as gate-installed or as optional."""

    declared = {module for symbol in _symbols() if (module := _required_by(symbol))}
    listed = GATE_ENV_EXAMPLE_MODULES | set(OPTIONAL_EXAMPLE_MODULES)
    assert not GATE_ENV_EXAMPLE_MODULES & set(OPTIONAL_EXAMPLE_MODULES)
    assert declared == listed, (
        f"unlisted: {sorted(declared - listed)}; listed but unused: {sorted(listed - declared)}"
    )


@pytest.mark.optional
@pytest.mark.parametrize("module", OPTIONAL_EXAMPLE_MODULES)
def test_optional_examples_run(module: str) -> None:
    """Run every example that requires ``module``; skip only where ``module`` is absent."""

    pytest.importorskip(module)
    symbols = [symbol for symbol in _symbols() if _required_by(symbol) == module]
    assert symbols, f"no example declares `# requires: {module}`"
    namespace = docstring_examples.ambient_namespace()
    failures = []
    for symbol in symbols:
        result = docstring_examples.run_example(symbol, namespace)
        if result.status != "ok":
            failures.append(f"{result.path} ({result.status}):\n{result.detail}")
    assert not failures, f"{len(failures)} examples did not run:\n" + "\n".join(failures)
