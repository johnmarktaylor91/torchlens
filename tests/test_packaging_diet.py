"""Packaging-diet tests for lazy pandas and IPython imports."""

import importlib.util
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

import torchlens as tl


class _TinyModel(nn.Module):
    """Small model for packaging smoke tests."""

    def __init__(self) -> None:
        """Initialize the tiny model layers."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single linear layer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return self.linear(x)


def _make_log() -> tl.Trace:
    """Create a completed model log for packaging tests.

    Returns
    -------
    tl.Trace
        Completed log for a tiny model.
    """

    return tl.trace(_TinyModel(), torch.randn(1, 3))


def test_core_import_capture_and_show_without_optional_tabular_or_notebook_use() -> None:
    """Core import, capture, and show path succeed in the current dev environment."""

    log = _make_log()

    assert len(log.layer_list) > 0
    assert log.show(vis_mode="none") is None


def test_plain_capture_never_imports_dynamo_or_fsdp() -> None:
    """A fresh-process plain trace/record must not import torch._dynamo or FSDP.

    W21 cold-start guarantee: the compiled-wrapper and FSDP guards are lazy
    sys.modules probes, so a plain eager capture never pays those imports
    (~2s process time, ~874 modules, ~128MB RSS cold). Once FSDP genuinely is
    imported, the same probe must still detect and reject an FSDP wrapper.
    """

    script = """
import sys
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
trace = tl.trace(model, torch.randn(2, 3))
assert len(trace.ops) > 0
recording = tl.record(model, torch.randn(2, 3), save=tl.func("relu"))
offenders = [
    name
    for name in sys.modules
    if name == "torch._dynamo"
    or name.startswith("torch._dynamo.")
    or name == "torch.distributed.fsdp"
    or name.startswith("torch.distributed.fsdp.")
]
assert not offenders, f"plain capture imported: {offenders}"

try:
    from torch.distributed.fsdp import FullyShardedDataParallel
except ImportError:
    FullyShardedDataParallel = None
if FullyShardedDataParallel is not None:
    wrapped = FullyShardedDataParallel.__new__(FullyShardedDataParallel)
    nn.Module.__init__(wrapped)
    wrapped.module = nn.Linear(3, 3)
    try:
        tl.trace(wrapped, torch.randn(2, 3))
    except RuntimeError as exc:
        assert "FullyShardedDataParallel" in str(exc)
    else:
        raise AssertionError("FSDP wrapper was not rejected after fsdp import")
print("COLD_IMPORT_OK")
"""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "COLD_IMPORT_OK" in result.stdout


def test_to_pandas_succeeds_when_tabular_extra_is_available() -> None:
    """Trace.to_pandas succeeds when pandas is installed."""

    log = _make_log()
    frame = log.to_pandas()

    assert not frame.empty
    assert "layer_label" in frame.columns


def test_to_pandas_missing_pandas_mentions_tabular_extra() -> None:
    """Trace.to_pandas raises a helpful extra-install hint when pandas is missing."""

    log = _make_log()

    with patch.dict("sys.modules", {"pandas": None}):
        try:
            log.to_pandas()
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected ImportError when pandas is unavailable.")

    assert "pandas is required for this feature" in message
    assert "pip install torchlens[tabular]" in message


def test_repr_html_succeeds_when_notebook_extra_is_available() -> None:
    """Trace._repr_html_ returns the CardTree card when IPython is installed."""
    pytest.importorskip("IPython")

    log = _make_log()
    html = log._repr_html_()

    assert html.startswith("<style>")
    assert "TorchLens Trace" in html
    assert "NaN/Inf" in html


def test_repr_html_without_ipython_still_renders_the_card() -> None:
    """Card generation is stdlib-only: no IPython gate (treescope memo B1).

    The historical gate returned a 94-byte plain repr whenever IPython was
    not importable, which also let naive card tests pass against the
    fallback -- card tests assert CONTENT.
    """

    log = _make_log()

    with patch.dict("sys.modules", {"IPython": None, "IPython.display": None}):
        html = log._repr_html_()

    assert "TorchLens Trace" in html
    assert "NaN/Inf" in html
    assert html != repr(log)


def test_packaging_metadata_uses_resolvable_gradcam_and_guarded_tinygrad_extra() -> None:
    """Packaging metadata keeps the corrected grad-cam name and tinygrad marker."""

    pyproject_text = Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text()

    assert 'gradcam = ["grad-cam~=1.5"]' in pyproject_text
    assert '"grad-cam~=1.5",' in pyproject_text
    assert "tinygrad = [\"tinygrad>=0.13,<0.14; python_version >= '3.11'\"]" in pyproject_text


def test_precommit_hook_entrypoints_use_python3_safe_predicates() -> None:
    """The hook entrypoints stay on the hardened conditions."""

    project_root = Path(__file__).resolve().parent.parent
    precommit_text = project_root.joinpath(".pre-commit-config.yaml").read_text()

    assert "entry: python scripts/check_no_breaking_markers.py" not in precommit_text
    assert "entry: scripts/check_no_breaking_markers.py --commit-msg" in precommit_text
    assert "entry: scripts/check_no_breaking_markers.py --pre-push" in precommit_text


def test_ci_workflows_pin_torch_and_scope_lint_to_owned_paths() -> None:
    """Packaging CI keeps torch pins honest and scopes lint to the owned paths."""

    project_root = Path(__file__).resolve().parent.parent
    nightly_text = project_root.joinpath(".github", "workflows", "nightly.yml").read_text()
    weekly_text = project_root.joinpath(".github", "workflows", "weekly.yml").read_text()
    lint_text = project_root.joinpath(".github", "workflows", "lint.yml").read_text()

    for workflow_text in (nightly_text, weekly_text):
        assert "torch==2.7.*" in workflow_text
        assert 'uv pip install --system -c "${{ runner.temp }}/torch-2.7-constraints.txt"' in (
            workflow_text
        )
        assert "uv pip check" in workflow_text
        assert 'assert torch.__version__.startswith("2.7.")' in workflow_text

    # Pin the FULL widened scope (grind r3, R70 / OL#41), not a prefix of it: a
    # prefix assertion still passes when the contributor-facing trees are dropped
    # back off the gate, which is exactly the regression worth catching.
    lint_scope = "torchlens tests scripts tools benchmarks examples notebooks"
    assert f"ruff format --check {lint_scope}" in lint_text
    assert f"ruff check {lint_scope}" in lint_text

    # The excluded set moved from lint.yml CLI flags into pyproject's
    # `[tool.ruff] extend-exclude` so that pre-commit -- which passes explicit
    # staged filenames and therefore ignores CLI --exclude -- reaches the same
    # verdict as this gate. Assert the boundary at its single authority, and that
    # it has NOT drifted back into duplicate CLI flags.
    pyproject_text = project_root.joinpath("pyproject.toml").read_text()
    extend_exclude = re.search(
        r"^\s*extend-exclude\s*=\s*\[(.*?)\]", pyproject_text, re.DOTALL | re.MULTILINE
    )
    assert extend_exclude is not None, "pyproject [tool.ruff] must declare extend-exclude"
    excluded = set(re.findall(r'"([^"]+)"', extend_exclude.group(1)))
    # The two generated artifacts are excluded because their generator is the
    # authority and tests/test_schema_lockstep.py compares them byte-for-byte, so
    # a ruff rewrite would make that gate permanently red. That half is pinned to
    # the self-declaring generated set by
    # test_schema_lockstep.py::test_ruff_excludes_every_generated_artifact.
    # tests/release_goldens is the harvested-corpus provenance dir (P05 request,
    # landed by the A12 packaging batch) and tests/classics_corpus/models holds
    # the vendored classics sources, sha256-pinned in the corpus manifest: the
    # provenance is the authority, not ruff.
    assert excluded == {
        "tests/classics_corpus/models",
        "tests/release_goldens",
        "torchlens/data_classes/_schema_bindings.py",
        "torchlens/ir/op_record_manifest.py",
    }
    assert "--exclude" not in lint_text


def test_ruff_pin_is_identical_across_declaration_sites() -> None:
    """The ruff that WRITES the code and the ruff that JUDGES it must be one version.

    Three files independently name a ruff version: pyproject's dev extra, the Lint
    workflow's install step, and the ruff-pre-commit ``rev``. When they drift, the
    pre-commit formatter rewrites code to a style CI then rejects -- which is exactly
    how the repo accumulated 147 format-stale files under a v0.9.7 hook while CI
    judged with 0.15.4.
    """

    project_root = Path(__file__).resolve().parent.parent
    pyproject_text = project_root.joinpath("pyproject.toml").read_text()
    lint_text = project_root.joinpath(".github", "workflows", "lint.yml").read_text()
    precommit_text = project_root.joinpath(".pre-commit-config.yaml").read_text()

    dev_pins = set(re.findall(r'"ruff==([0-9]+\.[0-9]+\.[0-9]+)"', pyproject_text))
    ci_pins = set(re.findall(r"ruff==([0-9]+\.[0-9]+\.[0-9]+)", lint_text))
    # The rev is a full commit SHA (R61: the one --fix hook must not ride a
    # mutable tag); the version lockstep reads the `# vX.Y.Z` provenance
    # trailer, the same pattern the other SHA-pinned hook repos use.
    hook_revs = set(
        re.findall(
            r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n"
            r"(?:\s*#.*\n)*"
            r"\s*rev:\s*[0-9a-f]{40}\s*#\s*v([0-9]+\.[0-9]+\.[0-9]+)",
            precommit_text,
        )
    )

    assert len(dev_pins) == 1, f"expected exactly one ruff dev pin, got {dev_pins}"
    assert len(ci_pins) == 1, f"expected exactly one ruff CI pin, got {ci_pins}"
    assert len(hook_revs) == 1, f"expected exactly one ruff-pre-commit rev, got {hook_revs}"
    assert dev_pins == ci_pins == hook_revs, (
        "ruff version drift: pyproject dev extra "
        f"{dev_pins}, lint.yml {ci_pins}, .pre-commit-config.yaml {hook_revs}. "
        "The formatter and the gate must be the same ruff."
    )


def test_third_party_actions_are_sha_pinned() -> None:
    """Every third-party Action ref is SHA-pinned.

    A mutable tag like ``@v4`` means whoever controls that tag controls what runs
    in CI, including in the job that publishes to PyPI.
    """

    workflow_dir = Path(__file__).resolve().parent.parent / ".github" / "workflows"

    unpinned: list[str] = []
    pinned_count = 0
    for workflow in sorted(workflow_dir.glob("*.yml")):
        for ref in re.findall(r"uses:\s*(\S+)", workflow.read_text()):
            if ref.startswith("./"):  # local reusable workflow, not third-party
                continue
            if re.search(r"@[0-9a-f]{40}$", ref):
                pinned_count += 1
                continue
            unpinned.append(f"{workflow.name}: {ref}")

    assert not unpinned, (
        "third-party Action refs must be pinned to a full 40-char commit SHA "
        f"with no exceptions: {unpinned}"
    )
    assert pinned_count > 0, "expected to find SHA-pinned action refs"


def test_release_job_python_stack_is_hash_locked() -> None:
    """The release job installs its Python deps hash-verified, wheels-only.

    r4 b2-sol R61 (HIGH): the job exact-pinned python-semantic-release but
    resolved its TRANSITIVES unpinned and unhashed, then ran the stack with
    the repo-write App token — a compromised or dependency-confused
    transitive could push with that token. The job must install from the
    committed lock with ``--require-hashes`` (every requirement carries
    ``--hash``) and ``--only-binary :all:`` (no sdist code execution at
    install time).
    """

    repo_root = Path(__file__).resolve().parent.parent
    release_yml = (repo_root / ".github" / "workflows" / "release.yml").read_text()

    assert "--require-hashes" in release_yml, (
        "release.yml no longer installs with --require-hashes; the "
        "semantic-release stack runs with the repo-write token and must be "
        "byte-audited"
    )
    assert "--only-binary :all:" in release_yml
    assert "release-requirements.txt" in release_yml
    assert 'pip install "python-semantic-release' not in release_yml, (
        "release.yml regained a bare unhashed pip install"
    )

    lock_path = repo_root / ".github" / "workflows" / "release-requirements.txt"
    lock_text = lock_path.read_text()
    requirement_lines = [
        line for line in lock_text.splitlines() if re.match(r"^[A-Za-z0-9_.-]+==", line)
    ]
    assert requirement_lines, "release lock lost its pinned requirements"
    assert any(line.startswith("python-semantic-release==") for line in requirement_lines)

    # Every pinned requirement must carry at least one hash: pip refuses
    # mixed hashed/unhashed input under --require-hashes, but the lock is
    # ALSO the review surface, so enforce it directly.
    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+==)", lock_text)
    unhashed = [
        block.splitlines()[0]
        for block in blocks
        if re.match(r"^[A-Za-z0-9_.-]+==", block) and "--hash=sha256:" not in block
    ]
    assert not unhashed, f"release lock entries without hashes: {unhashed}"

    # The lock is only airtight if NOTHING ELSE installs inside the token
    # scope: build_command formerly ran its own unhashed
    # `pip install 'build==1.5.0'` after checkout, escaping the lock (grind
    # r5, P9/R61). The builder must come from the lock (asserted present
    # here) and build_command must never regain an install.
    pyproject_text = (repo_root / "pyproject.toml").read_text()
    match = re.search(r'^build_command = "(.*)"$', pyproject_text, flags=re.MULTILINE)
    assert match, "no build_command found in pyproject.toml"
    assert "pip install" not in match.group(1), (
        "build_command runs its own pip install inside the release token "
        "scope, escaping the hash lock; add the package to "
        "release-requirements.txt instead"
    )
    assert any(line.startswith("build==") for line in requirement_lines), (
        "the release lock no longer pins the `build` builder"
    )

    # r7 R86 (fable HIGH + opus MED-HIGH, MEASURED): `python -m build`
    # WITHOUT --no-isolation creates an isolated env and pip-installs the
    # build backend (setuptools) from the LIVE index at build time -- inside
    # the token scope, with the repo-write App token persisted in
    # .git/config. That is arbitrary unpinned code with push access, escaping
    # the hash lock this test guards. The builder must run --no-isolation
    # against the hash-locked env, which therefore must pin the backend.
    assert "--no-isolation" in match.group(1), (
        "build_command runs `python -m build` without --no-isolation: the "
        "isolated build env pip-installs an UNPINNED setuptools from the "
        "live index inside the release token scope"
    )
    # r7 R84: the normalizer decides the published bytes of BOTH artifacts;
    # dropping either argument shipped machine-dependent bytes with the
    # first signal a post-release nightly red.
    assert "python scripts/normalize_sdist.py dist/*.tar.gz dist/*.whl" in match.group(1), (
        "build_command no longer normalizes both artifacts "
        "(scripts/normalize_sdist.py dist/*.tar.gz dist/*.whl)"
    )
    setuptools_pins = [line for line in requirement_lines if line.startswith("setuptools==")]
    assert setuptools_pins, (
        "release lock does not pin the setuptools build backend; "
        "--no-isolation builds resolve it from this lock"
    )
    backend_floor = re.search(
        r'^requires = \["setuptools>=(\d+)"\]', pyproject_text, flags=re.MULTILINE
    )
    assert backend_floor, "pyproject [build-system] requires lost its setuptools floor"
    pinned_version = setuptools_pins[0].split("==")[1].split()[0].strip("\\").strip()
    assert int(pinned_version.split(".")[0]) >= int(backend_floor.group(1)), (
        f"locked setuptools {pinned_version} is below the [build-system] "
        f"floor >={backend_floor.group(1)} (CVE-2026-59890 sdist-governance fix)"
    )

    # The nightly double-build gate must attest the SAME no-isolation builder
    # the release uses, or its byte-identity proof is about a different
    # (index-resolved) backend than the one that ships.
    nightly_text = (repo_root / ".github" / "workflows" / "nightly.yml").read_text()
    nightly_builds = [
        line for line in nightly_text.splitlines() if re.search(r"python -m build\b", line)
    ]
    assert nightly_builds, "nightly.yml lost its double-build gate invocations"
    isolated_nightly = [line for line in nightly_builds if "--no-isolation" not in line]
    assert not isolated_nightly, (
        f"nightly build invocation(s) without --no-isolation: {isolated_nightly}"
    )


@pytest.mark.slow
def test_built_wheel_manifest_is_diet(tmp_path: Path) -> None:
    """Assert the built wheel's manifest: schemas in, py.typed in, nothing else top-level.

    Nothing used to test the wheel manifest, and it had drifted three ways at
    once: a non-library reference corpus was in the distributed package set
    (2886 of 3316 members, ~13.5 MB, plus a second top-level import name),
    ``torchlens/py.typed`` was missing so downstream mypy ignored every
    annotation in the package (PEP 561), and only the schema files were checked.
    """

    project_root = Path(__file__).resolve().parent.parent
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    # Probe build.__main__, not "build": a setuptools `build/` directory in the
    # repo root is importable as a package named `build` WITHOUT a __main__, so
    # find_spec("build") can succeed while `python -m build` then dies with
    # "'build' is a package and cannot be directly executed".
    #
    # find_spec imports the PARENT first and so RAISES ModuleNotFoundError when
    # `build` is absent entirely, rather than returning None -- which is exactly
    # the CI case (build is a declared dev dependency now, but this probe must
    # still degrade to the pip fallback instead of erroring).
    try:
        has_build_main = importlib.util.find_spec("build.__main__") is not None
    except ModuleNotFoundError:
        has_build_main = False

    if has_build_main:
        command = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_dir),
            str(project_root),
        ]
    elif importlib.util.find_spec("pip") is not None:
        command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(project_root),
            "--no-deps",
            "-w",
            str(wheel_dir),
        ]
    else:
        pytest.skip("neither build nor pip is importable for wheel construction")

    # Neutral cwd + explicit srcdir, for the same shadowing reason: building with
    # cwd=project_root puts the repo (and any ./build/) on sys.path[0]. A stale
    # build/lib/ ALSO makes this test report agent docs the current config
    # correctly excludes -- a false positive that reads exactly like a real
    # packaging regression (2026-08-19).
    subprocess.run(command, cwd=tmp_path, check=True)
    wheels = sorted(wheel_dir.glob("torchlens-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel_zip:
        members = wheel_zip.namelist()
        top_level_members = [m for m in members if m.endswith("top_level.txt")]
        assert len(top_level_members) == 1
        top_level = wheel_zip.read(top_level_members[0]).decode().split()

    schema_members = [
        m for m in members if m.startswith("torchlens/schemas/") and m.endswith(".json")
    ]
    assert schema_members, "expected at least one torchlens/schemas/*.json wheel member"

    # PEP 561: without this marker file downstream type checkers treat the
    # package as untyped and skip every annotation it ships.
    assert "torchlens/py.typed" in members, "wheel must ship the PEP 561 py.typed marker"

    # Only the library and its dist-info ship; repo corpora never do.
    stray_members = [m for m in members if not m.startswith(("torchlens/", "torchlens-"))]
    assert not stray_members, (
        f"wheel ships {len(stray_members)} non-library member(s), e.g. {stray_members[:3]}; "
        "only the torchlens package is distributed (see [tool.setuptools.packages.find])"
    )
    assert top_level == ["torchlens"], (
        f"wheel installs top-level name(s) {top_level}; torchlens must be the only one"
    )

    # The agent docs (CLAUDE.md / AGENTS.md under torchlens/) are internal
    # working notes on a PUBLIC repo; both pyproject's exclude-package-data
    # block and MANIFEST.in claim THIS test enforces their absence, and until
    # r7 R84-1 neither claim was true — the one packaging regression that has
    # already shipped once had no tripwire.
    agent_doc_members = [m for m in members if m.rsplit("/", 1)[-1] in ("CLAUDE.md", "AGENTS.md")]
    assert not agent_doc_members, (
        f"wheel ships internal agent docs: {agent_doc_members} — "
        "[tool.setuptools.exclude-package-data] or MANIFEST.in regressed"
    )


def test_nightly_gate_installs_release_locked_builder() -> None:
    """The nightly double-build gate installs the release's exact builder (R61).

    SINGLE PIN AUTHORITY: build_command installs NOTHING inside the token
    scope (asserted by test_release_job_python_stack_is_hash_locked above); the
    builder comes from the hash-locked release-requirements.txt the release
    job installs. The nightly gate must install hash-verified from that SAME
    lock -- a name/version-only install (the old ``uv pip install
    "$BUILD_PIN"``) verified no artifact bytes, and a second builder lock
    would let the gate attest a different builder than the release uses.
    """

    repo_root = Path(__file__).resolve().parent.parent
    nightly_yml = (repo_root / ".github" / "workflows" / "nightly.yml").read_text()

    # r7 R84: scope the assertions to the wheel job's BUILD STEP -- the old
    # whole-file substring passed if the flag appeared anywhere in the
    # 600-line workflow, not necessarily in the step that installs the
    # builder the gate then attests.
    build_step = re.search(
        r"name: Build wheel and sdist with the release's exact builder.*?(?=\n\s*- name:)",
        nightly_yml,
        flags=re.DOTALL,
    )
    assert build_step is not None, "nightly.yml lost its exact-builder build step"
    step_text = build_step.group(0)
    assert "--require-hashes" in step_text, (
        "the nightly double-build gate no longer installs the builder hash-verified"
    )
    assert "--only-binary :all:" in step_text
    assert ".github/workflows/release-requirements.txt" in step_text, (
        "the nightly double-build gate no longer installs the release's exact builder"
    )
    install_pos = step_text.find("--require-hashes")
    build_pos = step_text.find("python -m build")
    assert 0 <= install_pos < build_pos, (
        "the hash-verified install must precede `python -m build` inside the build step"
    )
    assert "build-requirements.txt" not in nightly_yml, (
        "a second builder lock reappeared; release-requirements.txt is the single pin authority"
    )
    workflows_dir = repo_root / ".github" / "workflows"
    assert not (workflows_dir / "build-requirements.txt").exists(), (
        "the retired build-requirements.txt lock is back; the builder pin "
        "lives in release-requirements.txt (single pin authority)"
    )


def test_precommit_pin_is_single_valued_and_inside_the_contributor_band() -> None:
    """r7 R87-2 (opus LOW): the tool that RUNS the parity gate gets a parity gate.

    CI judges hooks with an exact ``pre-commit==`` while contributors resolve
    the dev extra's ``>=4,<5`` band; nothing tied the two the way ruff,
    pydot, graphviz, and pip-audit are tied. Every workflow pin must be ONE
    version and it must satisfy the contributor band, so a CI-only verdict a
    contributor cannot reproduce needs a conscious band edit first.
    """

    project_root = Path(__file__).resolve().parent.parent
    workflow_text = "".join(
        path.read_text() for path in sorted((project_root / ".github" / "workflows").glob("*.yml"))
    )
    ci_pins = set(re.findall(r"pre-commit==([0-9]+\.[0-9]+\.[0-9]+)", workflow_text))
    assert len(ci_pins) == 1, f"expected ONE pre-commit CI pin across workflows: {ci_pins}"
    pyproject_text = (project_root / "pyproject.toml").read_text()
    band = re.search(r'"pre-commit>=([0-9]+),<([0-9]+)"', pyproject_text)
    assert band is not None, "dev extra lost its pre-commit band"
    major = int(next(iter(ci_pins)).split(".")[0])
    assert int(band.group(1)) <= major < int(band.group(2)), (
        f"CI pre-commit pin {ci_pins} escaped the contributor band "
        f">={band.group(1)},<{band.group(2)}"
    )


# ---------------------------------------------------------------------------
# Pin-lockstep gates (r7 R87-1): these five assertions previously lived ONLY
# as inline scripts in lint.yml's actionlint job — a job the repo documents
# as advisory, so none was PR-blocking and none was locally runnable
# (`pytest tests/` could not reach them; the "one version authority"
# doctrine was enforceable only by pushing). They are the same checks,
# ported verbatim; lint.yml now points here instead of duplicating them.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"


def _all_workflow_text() -> str:
    """Concatenate every workflow file (the scan corpus for inline pins)."""

    return "".join(path.read_text() for path in sorted(_WORKFLOWS_DIR.glob("*.yml")))


@pytest.mark.smoke
def test_newest_admitted_torch_literal_lockstep() -> None:
    """quality.yml's inline torch pins equal tests.yml's newest matrix row.

    The newest-admitted torch version is inlined in quality.yml (mypy +
    dep-audit envs) and repeatedly in tests.yml's matrix with nothing keeping
    them equal — a matrix bump that skips quality.yml silently type-checks
    and audits an older torch. (The nightly capture-byte-oracle pin
    deliberately tracks the golden ENV marker instead and is lockstepped by
    test_ci_packaging_gates.)
    """

    torch_re = re.compile(r"torch(?:==|: \")(\d+\.\d+\.\d+)\+cpu")
    tests_versions = torch_re.findall((_WORKFLOWS_DIR / "tests.yml").read_text())
    quality_versions = torch_re.findall((_WORKFLOWS_DIR / "quality.yml").read_text())
    assert tests_versions and quality_versions, (
        "expected torch==X.Y.Z+cpu literals in tests.yml and quality.yml"
    )
    newest = max(tests_versions, key=lambda v: tuple(map(int, v.split("."))))
    stale = sorted(set(quality_versions) - {newest})
    assert not stale, (
        f"quality.yml pins torch {stale} but tests.yml's newest-admitted row is {newest}: "
        "bump them together so type-check/audit run on the newest admitted torch"
    )


@pytest.mark.smoke
def test_pydot_inline_pins_match_the_test_extra_authority() -> None:
    """Every inline workflow pydot pin equals the [test] extra's pin.

    pydot is a DOT-render golden fingerprint KEY: the [test] extra is the
    pin authority and the lean .[dev,tabular] legs repeat it inline, so a
    bump that misses a site silently renders goldens with a different pydot.
    The scan covers EVERY workflow file — a hand-enumerated list let
    mutation.yml's third pin drift unguarded from birth.
    """

    pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
    authority = set(re.findall(r'"pydot==([0-9][^"]*)"', pyproject))
    assert len(authority) == 1, f"expected ONE pydot pin in pyproject: {authority}"
    inline = set(re.findall(r"pydot==([0-9][^\"'\s]*)", _all_workflow_text()))
    assert inline == authority, (
        f"inline pydot pins {inline} drifted from the [test] extra authority "
        f"{authority}: bump every site together or the DOT goldens' recording "
        "environment forks between tiers"
    )


@pytest.mark.smoke
def test_pip_audit_inline_pin_matches_the_dev_extra_authority() -> None:
    """quality.yml's inline pip-audit pin equals the dev extra's exact pin."""

    pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
    authority = set(re.findall(r'"pip-audit==([0-9][^"]*)"', pyproject))
    assert len(authority) == 1, f"expected ONE pip-audit pin in pyproject: {authority}"
    inline = set(
        re.findall(r"pip-audit==([0-9][^\"'\s]*)", (_WORKFLOWS_DIR / "quality.yml").read_text())
    )
    assert inline == authority, (
        f"quality.yml pip-audit pin {inline} drifted from the dev extra "
        f"authority {authority}: the inline copy is the one that actually "
        "audits releases"
    )


@pytest.mark.smoke
def test_graphviz_inline_pins_match_the_committed_env_markers() -> None:
    """Every inline graphviz pin equals the committed ENV-graphviz markers.

    graphviz (the python DOT emitter) is a golden fingerprint KEY but a CORE
    runtime dep that must stay a floor for users — so its authority is the
    committed marker itself, and the CI inline pins must equal it. A free
    resolution moves the byte families off-canonical on the enforcing row.
    """

    markers = {
        rel: (_REPO_ROOT / rel).read_text().strip()
        for rel in (
            "tests/golden/ENV-graphviz",
            "tests/godobject_oracle/goldens/ENV-graphviz",
        )
    }
    marker_versions = set(markers.values())
    assert len(marker_versions) == 1, f"ENV-graphviz markers disagree: {markers}"
    inline = set(re.findall(r"graphviz==([0-9][^\"'\s]*)", _all_workflow_text()))
    assert inline == marker_versions, (
        f"inline graphviz pins {inline} drifted from the committed "
        f"ENV-graphviz markers {marker_versions}: the python DOT emitter is a "
        "golden fingerprint KEY — a free resolution moves the byte families "
        "off-canonical on the enforcing row"
    )
