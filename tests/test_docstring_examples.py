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
is skipped, not failed, when that module is absent. An example that needs a torch surface
the running build lacks (one-backward reads without GradientEdge) stops, as ``refused``, at
that capability's named refusal (``tests/support/capability_routes.py``); where the surface
exists the same refusal fails like any other exception.

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

# ``support`` (tests/ on sys.path), not ``tests.support``: the brainscore leg's
# brainscore_vision 2.4.4 installs a top-level ``tests`` package that shadows
# this repo's ``tests`` namespace package.
from support.capability_routes import accepts_capability_refusal  # noqa: E402

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


_CENSUS_WITH_PEERS_BLOCKED = textwrap.dedent(
    """
    import pathlib
    import sys

    sys.path.insert(0, sys.argv[1])
    import docstring_examples

    for peer in sorted(set(docstring_examples.OPTIONAL_PEER_MODULES.values())):
        sys.modules[peer] = None
    sys.modules["pytorch_lightning"] = None
    text = docstring_examples.render_census(
        docstring_examples.public_symbols(), docstring_examples.public_members()
    )
    pathlib.Path(sys.argv[2]).write_text(text)
    """
)


def test_census_is_the_same_without_the_optional_peers(tmp_path: Path) -> None:
    """With every optional peer blocked, the census still matches the committed file.

    The Lightning callbacks subclass Lightning's ``Callback`` on first access, so where
    ``lightning`` is absent they are read from source; this keeps one committed census
    true both in an environment with the ``test`` extra and in one without it. The
    subprocess blocks the peers (a ``None`` entry in ``sys.modules``) before torchlens
    is imported.
    """

    out = tmp_path / "census.md"
    completed = subprocess.run(
        [sys.executable, "-c", _CENSUS_WITH_PEERS_BLOCKED, str(REPO_ROOT / "tools"), str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert out.read_text() == docstring_examples.CENSUS_PATH.read_text()


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
    # GPU kernel recording is not public in this release: its record member stays off the list.
    assert not {"reached:Op.gpu_kernels", "reached:AtenOp.gpu_kernels"} & (fresh | after)


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
        result = docstring_examples.run_example(
            symbol, namespace, accept=accepts_capability_refusal
        )
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


def _refusing_read() -> None:
    """Stand-in whose example stops at the typed one-backward refusal.

    Examples
    --------
    >>> from torchlens.attribution._onebackward import ReadError
    >>> raise ReadError("no GradientEdge here", code="onebackward_torch_unsupported")
    >>> name_only_the_refused_read_would_bind
    1
    """


def _fails_before_refusing() -> None:
    """Stand-in whose example prints the wrong value before the refusal.

    Examples
    --------
    >>> 1 + 1
    3
    >>> from torchlens.attribution._onebackward import ReadError
    >>> raise ReadError("no GradientEdge here", code="onebackward_torch_unsupported")
    """


def _accept_refusal(exc: BaseException) -> bool:
    """Accept the typed one-backward refusal regardless of the running torch."""

    from torchlens.attribution._onebackward import ReadError

    return isinstance(exc, ReadError) and exc.fields.get("code") == "onebackward_torch_unsupported"


def test_accepted_exception_stops_the_example_as_refused() -> None:
    """An exception the caller accepts ends the example: later lines do not run or fail."""

    symbol = docstring_examples.PublicSymbol("tl.refusing", __name__, "function", _refusing_read)
    result = docstring_examples.run_example(symbol, {}, accept=_accept_refusal)
    assert result.status == "refused"
    assert "onebackward_torch_unsupported" in result.detail


def test_unaccepted_exception_still_fails() -> None:
    """Without an accepting predicate (or with one that declines) the refusal fails."""

    symbol = docstring_examples.PublicSymbol("tl.refusing", __name__, "function", _refusing_read)
    assert docstring_examples.run_example(symbol, {}).status == "failed"
    declined = docstring_examples.run_example(symbol, {}, accept=lambda _exc: False)
    assert declined.status == "failed"


def test_failure_before_an_accepted_refusal_still_fails() -> None:
    """A wrong output earlier in the example is never excused by a later refusal."""

    symbol = docstring_examples.PublicSymbol(
        "tl.fails_first", __name__, "function", _fails_before_refusing
    )
    result = docstring_examples.run_example(symbol, {}, accept=_accept_refusal)
    assert result.status == "failed"


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
        result = docstring_examples.run_example(
            symbol, namespace, accept=accepts_capability_refusal
        )
        if result.status not in {"ok", "refused"}:
            failures.append(f"{result.path} ({result.status}):\n{result.detail}")
    assert not failures, f"{len(failures)} examples did not run:\n" + "\n".join(failures)


def test_unvetted_collective_build_route_accepts_only_the_arm_refusal() -> None:
    """Graceful floor test for the ``is_vetted_build`` route the distributed examples use.

    On a torch whose collective census is not vetted, ``arm()`` refuses with the typed
    layer-1 ``UncapturedCollectiveOpError`` and the route accepts exactly that; on a
    vetted torch the route accepts nothing, so an example there must run to the end.
    """

    from support.capability_routes import ROUTES

    from torchlens import distributed

    route = ROUTES["is_vetted_build"]
    assert not route.accepts(RuntimeError("unrelated"))
    if distributed.is_vetted_build():
        assert not route.accepts(_unvetted_refusal_stand_in())
        return
    with pytest.raises(RuntimeError) as excinfo:
        distributed.arm()
    assert route.accepts(excinfo.value)


def _unvetted_refusal_stand_in() -> BaseException:
    """Build the layer-1 arm refusal without needing an unvetted torch."""

    from torchlens._distributed._recognizer import (
        UNCAPTURED_COLLECTIVE_OP,
        UncapturedCollectiveOpError,
    )

    return UncapturedCollectiveOpError(
        "stand-in", kind=UNCAPTURED_COLLECTIVE_OP, layer=1, mismatches={}
    )
