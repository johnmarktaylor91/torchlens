"""Environment-fingerprinted golden resolution for the god-lane oracles (M13).

The byte-identity oracles (surface-v1, viz identity, legacy artifact, state
keysets) freeze REGRESSION baselines, not cross-environment identities: a
different torch/python build can legitimately change qualnames, float
formatting inside reprs, or source-line content without any TorchLens
behavior change. Cluster portability therefore keys goldens per environment:

* The checked-in canonical goldens carry the ``ENV`` marker file naming the
  fingerprint they were recorded under. On a matching environment they are
  enforced exactly as before.
* On any OTHER environment, goldens live under ``env-<fingerprint>/`` inside
  the same goldens directory and are enforced byte-exactly when COMMITTED.
* A MISSING off-canonical golden is FAIL-CLOSED (b10 R78-4): the historical
  record-and-skip behavior silently self-baselined every ephemeral CI leg
  forever (no cache, no committed ``env-*`` dir, so "every later run
  enforces" never happened) and auto-rebaselined fresh dev boxes. Recording
  a first-run baseline now requires the EXPLICIT ``TORCHLENS_ORACLE_RECORD_ENV=1``
  opt-in (deliberate provisioning of a new long-lived box); ephemeral CI
  environments (``CI`` set) skip with a visible reason and never write.

``env_fingerprint`` deliberately stays NARROW (py-major.minor + torch version
sans build tag). It is known-blind to build variant, CPU ISA, thread count,
and BLAS (SF-51): do NOT try to close cross-machine float drift by widening
this key — the fail-closed missing-golden policy above is the guard, and a
structural/subprocess oracle is the fix direction of record.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest

#: Explicit opt-in for recording a first-run baseline on a new (off-canonical,
#: non-ephemeral) environment. One deliberate provisioning run, then commit
#: the recorded ``env-*`` directory if the environment is meant to enforce.
RECORD_ENV_VAR = "TORCHLENS_ORACLE_RECORD_ENV"

#: Declares the running leg the ENFORCING leg for the byte-oracle goldens.
#: On an enforcing leg every missing-golden outcome is a hard FAILURE — the
#: CI skip and the record opt-in are both refused — so the one leg whose
#: environment is supposed to match the committed ENV marker can never drift
#: off-canonical (e.g. a matrix torch bump without a golden rebaseline) and
#: silently skip every golden case while staying green (grind-p3 T13.1).
ENFORCE_ENV_VAR = "TORCHLENS_ORACLE_ENFORCE"

#: Required WHY sidecar for every golden update/record run (b10 R78-8a
#: follow-up): the PROVENANCE record documents how AND why a golden changed.
REASON_ENV_VAR = "TORCHLENS_GOLDEN_REASON"

#: Reviewed refresh flag for the producer-parity ledger corpus
#: (``tests/producer_parity/ledger/*.json``). Registered here so the flag
#: participates in BOTH governance layers (see the registry below).
PRODUCER_LEDGER_REFRESH_ENV = "TORCHLENS_REFRESH_PRODUCER_LEDGER"

#: Name PREFIXES that always denote golden mutation flags, in every layer.
GOLDEN_FLAG_PREFIXES = ("TORCHLENS_UPDATE_", "TORCHLENS_REGEN_")

#: THE single named-flag registry (b10 R78 round-4). Two hand-maintained
#: registries drifted apart — ``TL_SELECTOR_MATRIX_REGEN`` was in the CI
#: guard but not the arming lint, and ``TORCHLENS_REFRESH_PRODUCER_LEDGER``
#: was in NEITHER, so a ledger refresh flag truthy-armed and auto-greened
#: invisibly. Every non-prefix golden/enforcement flag is declared ONCE here
#: with its roles; the governance lint's arming scanner and the root
#: conftest's CI session guard both DERIVE their sets from this table.
#:
#: Roles:
#:  * ``"arming"`` — reads of the flag in tests/ must arm on the exact value
#:    ``"1"`` (``flag_armed`` or inline ``== "1"``), enforced by
#:    ``tests/test_golden_governance_lint.py``.
#:  * ``"ci-forbidden"`` — the flag mutates committed artifacts, so a CI run
#:    with it set (ANY value; presence is fail-closed there) hard-errors in
#:    the root conftest instead of rebaselining.
#: ``ENFORCE_ENV_VAR`` deliberately carries no ``"ci-forbidden"`` role (CI's
#: canonical row SETS it), and the retired ``TL_SELECTOR_MATRIX_REGEN`` no
#: ``"arming"`` role (its sole read is an any-value hard error on the old
#: name).
GOLDEN_FLAG_REGISTRY: dict[str, frozenset[str]] = {
    RECORD_ENV_VAR: frozenset({"arming", "ci-forbidden"}),
    ENFORCE_ENV_VAR: frozenset({"arming"}),
    "TL_SELECTOR_MATRIX_REGEN": frozenset({"ci-forbidden"}),
    PRODUCER_LEDGER_REFRESH_ENV: frozenset({"arming", "ci-forbidden"}),
}


def golden_flag_names_for_role(role: str) -> frozenset[str]:
    """Return every registered named flag carrying ``role``.

    Parameters
    ----------
    role:
        ``"arming"`` or ``"ci-forbidden"`` (see ``GOLDEN_FLAG_REGISTRY``).

    Returns
    -------
    frozenset[str]
        Registered flag names with that role.
    """

    return frozenset(name for name, roles in GOLDEN_FLAG_REGISTRY.items() if role in roles)


def flag_armed(environ: Mapping[str, str], name: str) -> bool:
    """Return whether a golden update/regen/record flag is ARMED.

    A flag arms on the exact value ``"1"`` ONLY (b10 R78 round-3): truthy
    interpretation (``bool(environ.get(name))``) armed regeneration on
    ``NAME=0``, ``NAME=false``, and every other non-empty spelling a user
    types to DISARM it. Every golden mutation flag read must route through
    this predicate (or an inline ``== "1"`` comparison, enforced by
    ``tests/test_golden_governance_lint.py``).

    Parameters
    ----------
    environ:
        Environment mapping to inspect (normally ``os.environ``).
    name:
        Flag variable name.

    Returns
    -------
    bool
        True exactly when ``environ[name] == "1"``.
    """

    return environ.get(name) == "1"


def _package_version(package: str) -> str:
    """Return the installed distribution version of ``package``.

    Parameters
    ----------
    package:
        Distribution name (e.g. ``"graphviz"``).

    Returns
    -------
    str
        The installed version string, or ``"absent"`` when the distribution
        is not installed — an honest fingerprint component either way.
    """

    import importlib.metadata

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def env_fingerprint(extra_packages: tuple[str, ...] = ()) -> str:
    """Return the golden-environment fingerprint for this interpreter.

    Parameters
    ----------
    extra_packages:
        Family-scoped fingerprint extension (b10 R78 round-3, SF-51-compatible):
        packages whose version DIRECTLY generates the family's golden bytes
        (e.g. the ``graphviz`` python package emits the DOT the viz-identity
        goldens freeze). Extending per-family keeps the GLOBAL key narrow —
        do not add packages here to chase float drift; this is only for
        direct byte-generators of the calling family.

    Returns
    -------
    str
        ``py<maj>.<min>-torch<ver>`` plus one ``-<pkg><ver>`` segment per
        extra package.
    """

    import torch

    torch_version = torch.__version__.split("+", 1)[0]
    base = f"py{sys.version_info.major}.{sys.version_info.minor}-torch{torch_version}"
    extras = "".join(f"-{package}{_package_version(package)}" for package in extra_packages)
    return base + extras


def resolve_env_golden(
    golden_dir: Path, name: str, extra_packages: tuple[str, ...] = ()
) -> tuple[Path, bool]:
    """Resolve one golden file for the current environment.

    Parameters
    ----------
    golden_dir:
        Directory holding the canonical goldens, the ``ENV`` marker, and (for
        families with ``extra_packages``) one ``ENV-<pkg>`` marker per direct
        byte-generator package naming the version the canonical goldens were
        recorded under.
    name:
        Golden file name.
    extra_packages:
        Family-scoped fingerprint extension; see :func:`env_fingerprint`.
        The environment is canonical only when the base ``ENV`` marker AND
        every ``ENV-<pkg>`` marker match the running versions — a missing or
        mismatched emitter marker moves the family onto the fail-closed
        env-keyed path instead of silently comparing bytes emitted by a
        different generator version.

    Returns
    -------
    tuple[Path, bool]
        The golden path to use and whether this environment is off-canonical
        (True exactly when the path is env-keyed). Callers enforcing a golden
        should prefer :func:`require_env_golden`, which owns the fail-closed
        missing-file policy.
    """

    marker = golden_dir / "ENV"
    if not marker.exists():
        # Fail CLOSED (b10 R78 round 5): a family routed through the
        # env-keyed resolver without a committed base ENV marker previously
        # counted as "canonical", so its bytes were enforced on EVERY
        # environment — the opposite default of a layer whose whole thesis
        # is fail-closed env keying. A missing marker is a setup bug in the
        # family, never a blessing.
        raise RuntimeError(
            f"{golden_dir} has no committed ENV marker but resolves through "
            "resolve_env_golden; commit the canonical fingerprint (see "
            "_ENV_GOVERNED_REQUIRED_MARKERS in test_golden_governance_lint.py) "
            "or route the family off the env-keyed resolver explicitly"
        )
    canonical_env = marker.read_text().strip()
    current_env = env_fingerprint()
    canonical = current_env == canonical_env
    if canonical:
        for package in extra_packages:
            extras_marker = golden_dir / f"ENV-{package}"
            recorded = extras_marker.read_text().strip() if extras_marker.exists() else None
            if recorded != _package_version(package):
                canonical = False
                break
    if canonical:
        return golden_dir / name, False
    return golden_dir / f"env-{env_fingerprint(extra_packages)}" / name, True


def require_env_golden(
    golden_dir: Path, name: str, update_env: str, extra_packages: tuple[str, ...] = ()
) -> Path:
    """Return the enforceable golden path for this environment, fail-closed.

    Policy for a MISSING golden (b10 R78-4):

    * canonical environment — hard failure (unchanged historical behavior);
    * ``TORCHLENS_ORACLE_ENFORCE=1`` (the declared enforcing leg) — hard
      failure, taking precedence over the record opt-in and the CI skip: an
      enforcing leg that finds itself off-canonical has drifted from the
      committed ENV marker and must go red, never quietly skip (T13.1);
    * off-canonical with ``TORCHLENS_ORACLE_RECORD_ENV=1`` and no ``CI`` —
      the caller may record a first-run baseline: the path is returned with
      its parent created, and the caller writes it then SKIPS;
    * off-canonical under ``CI`` — skip with a visible reason, never write
      (an ephemeral checkout can never satisfy "every later run enforces");
    * off-canonical otherwise — FAIL with recording instructions, so a fresh
      box never silently self-baselines while a root cause is open.

    Parameters
    ----------
    golden_dir:
        Directory holding the canonical goldens and the ``ENV`` marker.
    name:
        Golden file name.
    update_env:
        The owning family's update flag, named in failure messages.
    extra_packages:
        Family-scoped fingerprint extension; see :func:`resolve_env_golden`.

    Returns
    -------
    Path
        Path of an EXISTING golden to enforce, or (record opt-in only) the
        path to record.
    """

    golden_path, off_canonical = resolve_env_golden(golden_dir, name, extra_packages)
    if golden_path.exists():
        return golden_path
    if not off_canonical:
        pytest.fail(
            f"missing canonical golden {golden_path}; generate deliberately with "
            f"{update_env}=1 (the update run reports SKIP, then re-run to verify)"
        )
    if flag_armed(os.environ, ENFORCE_ENV_VAR):
        pytest.fail(
            f"this leg declares {ENFORCE_ENV_VAR}=1 (byte-oracle enforcement) but "
            f"runs off-canonical environment {env_fingerprint()!r} with no committed "
            f"golden ({golden_path} missing). The enforcing leg has drifted from the "
            "committed ENV marker — rebaseline the goldens deliberately (one "
            f"{RECORD_ENV_VAR}=1 run on the new environment, reviewed and committed) "
            "or restore the leg's environment; an enforcing leg never skips"
        )
    if flag_armed(os.environ, RECORD_ENV_VAR) and not os.environ.get("CI"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        return golden_path
    if os.environ.get("CI"):
        pytest.skip(
            f"no committed golden for environment {env_fingerprint()!r} "
            f"({golden_path} missing); byte enforcement runs on environments "
            "with committed baselines only"
        )
    pytest.fail(
        f"no golden for environment {env_fingerprint()!r} ({golden_path} missing). "
        f"Refusing to self-baseline: record ONE deliberate first-run baseline with "
        f"{RECORD_ENV_VAR}=1, review it, and commit the env-* directory if this "
        "environment should enforce byte identity"
    )


def golden_mutation_flags_armed_under_ci(environ: Mapping[str, str]) -> list[str]:
    """Return golden update/regen/record flags armed in a CI environment.

    Consumed by the root conftest's session guard (b7 R53-3): a CI run with
    any of these armed would rebaseline instead of verifying.

    Parameters
    ----------
    environ:
        Environment mapping to inspect.

    Returns
    -------
    list[str]
        Sorted offending variable names; empty outside CI or when none armed.
    """

    if not environ.get("CI"):
        return []
    forbidden = golden_flag_names_for_role("ci-forbidden")
    return sorted(
        name for name in environ if name.startswith(GOLDEN_FLAG_PREFIXES) or name in forbidden
    )


def require_update_reason(update_env: str) -> str:
    """Return the mandatory WHY for a golden update/record run, fail-closed.

    ``write_provenance`` historically recorded only HOW a golden was
    regenerated (generator, flag, env, date) — never WHY. Every update run
    must now carry ``TORCHLENS_GOLDEN_REASON`` (b10 R78 round-3): an empty
    or missing reason fails the update run BEFORE bytes are written, so a
    rebaseline can never land without a git-greppable justification.

    Parameters
    ----------
    update_env:
        The owning family's update flag, named in the failure message.

    Returns
    -------
    str
        The non-empty reason string.
    """

    reason = os.environ.get(REASON_ENV_VAR, "").strip()
    if not reason:
        pytest.fail(
            f"{update_env}=1 is a deliberate golden rebaseline and requires "
            f"{REASON_ENV_VAR} to record WHY (e.g. {REASON_ENV_VAR}='r21 collapse "
            "schedule fix, enumerated in the sprint report'). Refusing to write "
            "goldens without a reason."
        )
    return reason


#: Update flags whose wrap-state guard already verified a clean start in this
#: process. Intra-family wrapping DURING generation is inherent to in-process
#: multi-capture families and deterministic in a fresh single-family run; the
#: guard's job is to refuse generation on a torch some EARLIER test already
#: wrapped (SF-53), so it checks once per family per process.
_WRAP_GUARD_CLEARED: set[str] = set()


def guard_wrap_state_for_golden_update(update_env: str) -> None:
    """Refuse a golden update run whose generation starts on wrapped torch.

    In-process golden generation constructs models and captures on whatever
    torch state earlier tests left behind: torchlens torch-function wrappers
    install lazily on the first capture and STAY installed, so bytes
    generated mid-session can silently freeze wrap-state artifacts (SF-53,
    the b10 R78-1 generator-design gap the surface oracle fixed with a
    subprocess worker). Families that still generate in-process call this at
    the START of an update run's generation: if torch is already wrapped by
    earlier activity in this pytest process, the update run FAILS with
    instructions — a guard and a clear failure, never a silent unwrap.

    Parameters
    ----------
    update_env:
        The owning family's update flag, named in the failure message.
    """

    if update_env in _WRAP_GUARD_CLEARED:
        return
    state = sys.modules.get("torchlens._state")
    if state is not None and getattr(state, "_is_decorated", False):
        pytest.fail(
            f"{update_env}=1 golden generation must start on UNWRAPPED torch, but "
            "torchlens torch-function wrappers are already installed from earlier "
            "captures in this pytest process (SF-53: generated bytes may depend on "
            "wrap state). Regenerate in a fresh interpreter running ONLY this "
            f"family, e.g.: {update_env}=1 {REASON_ENV_VAR}='<why>' pytest <this "
            "family's test file>"
        )
    _WRAP_GUARD_CLEARED.add(update_env)


def _source_identity(golden_dir: Path | None = None) -> str:
    """Return ``<HEAD sha> (clean|dirty)`` for the generating checkout.

    A reviewed rebaseline must be mechanically tied to the exact source that
    emitted it, and a dirty-tree generation must be DISCLOSED, not silently
    recorded as if it came from a commit (b10 R78 round 5). Best-effort: a
    non-git checkout records ``unknown`` rather than failing the update run.

    ``golden_dir`` names the family's OWN output directory: the golden bytes
    are rewritten BEFORE the provenance stamp, so counting them made the
    disclosure structurally always-"dirty" and therefore inert (r7 R78-1 --
    the line exists to disclose a dirty SOURCE TREE, not the regeneration's
    own output). PROVENANCE sidecars are excluded for the same reason. A
    multi-family regen session still reads dirty from the OTHER families'
    fresh outputs -- conservative, disclosed here.
    """

    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown (git unavailable)"
    own_prefix = None
    if golden_dir is not None:
        try:
            own_prefix = golden_dir.resolve().relative_to(repo_root).as_posix() + "/"
        except ValueError:
            own_prefix = None
    lines = _foreign_porcelain_lines(status, own_prefix)
    return f"{head} ({'dirty' if lines else 'clean'})"


def _foreign_porcelain_lines(status: str, own_prefix: str | None) -> list[str]:
    """Porcelain lines OUTSIDE the stamping family's own output (pure, testable)."""

    lines = []
    for line in status.splitlines():
        path_part = line[3:].split(" -> ")[-1].strip().strip('"')
        if own_prefix is not None and (
            path_part.startswith(own_prefix) or path_part == own_prefix.rstrip("/")
        ):
            continue
        if path_part.endswith("PROVENANCE"):
            continue
        lines.append(line)
    return lines


def write_provenance(golden_dir: Path, generator: str, update_env: str, reason: str) -> None:
    """APPEND how and why the goldens in ``golden_dir`` were (re)generated.

    Written by update/record runs only — a sidecar, never compared, so it
    documents regeneration without perturbing golden bytes (b10 R78-8a).
    Records are APPENDED with full history (b10 R78 round-3): the historical
    single overwritten file lied whenever one directory hosted multiple flag
    families or a partial update — the last writer erased every earlier
    family's record.

    Parameters
    ----------
    golden_dir:
        Goldens directory receiving the ``PROVENANCE`` sidecar.
    generator:
        Human-readable generator identity (test file / family).
    update_env:
        The flag that armed this write.
    reason:
        WHY the goldens changed; thread from :func:`require_update_reason`.
    """

    import datetime

    import torch

    record = (
        f"generator: {generator}\n"
        f"flag: {update_env}=1\n"
        f"reason: {reason}\n"
        f"env: {env_fingerprint()}\n"
        f"torch: {torch.__version__}\n"
        f"source: {_source_identity(golden_dir)}\n"
        f"recorded: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
    )
    path = golden_dir / "PROVENANCE"
    existing = path.read_text() if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    separator = "---\n" if existing else ""
    path.write_text(existing + separator + record)


@contextmanager
def expect_bundle_minor_version_mismatch() -> Iterator[None]:
    """Narrowly filter the advisory Bundle-torch-minor-mismatch warning.

    R6 (Lead ruling via lane-L3-ci-triage.md section 5 step 18, 2026-10-01):
    several committed cross-env goldens (``godobject_oracle/goldens/``, the
    legacy fixture in ``test_grouping_stamp.py``, and siblings) were recorded
    on a torch 2.13 CUDA build. Loading them under a different torch MINOR
    fires ``torchlens._io.manifest``'s advisory ``TorchLensWarning`` ("Bundle
    torch_version=... differs from runtime torch_version=... (minor version
    mismatch)."), which is correct by design -- a same-major, different-minor
    bundle loads fine -- not a product bug. The repo's pytest
    ``filterwarnings`` promotes every TorchLens-originated ``UserWarning`` to
    an error by default (deliberate: TorchLens warnings should be assertable),
    so a golden-loading test must narrowly expect/filter THIS exact advisory
    rather than the loader weakening or the global promotion rule loosening.

    On a runtime whose torch minor happens to match the golden's recording
    (e.g. a 2.13 CI row), the warning never fires and this filter is simply
    unused -- safe on every row, not just the mismatched ones. The warning's
    own firing behavior is pinned independently in
    ``tests/test_tlspec_envelope_ceiling.py::test_manifest_policy_warns_not_raises_on_minor_mismatch``.
    """

    import warnings

    from torchlens.errors import TorchLensWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Bundle torch_version=.* differs from runtime torch_version=.* "
            r"\(minor version mismatch\)\.",
            category=TorchLensWarning,
        )
        yield
