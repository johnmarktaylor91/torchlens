"""The artifact compatibility LEDGER (ecosystem MEMO 3.1/3.2, build item B1).

One package-owned, dependency-light ledger drives runtime policy, generated
documentation, typed refusals and their remedies, migration planning, and CI.
Nothing loads, refuses, or claims by assertion: every row is backed by the
harvested golden corpus (``tests/release_goldens/``, sha256-pinned) or by the
live writer contract, and every remedy release the ledger names is verified
by the remedy-actually-loads CI test (``tests/test_ecosystem_rt_ledger.py``).

Two digests, never one (MEMO 3.1): ``writer_contract_digest`` is the
NORMATIVE digest over the canonicalized persistence contract
(``torchlens._io.writer_contract``); the golden corpus digest is the
immutable digest of the real harvested bytes. One observed artifact cannot
define a grammar, and a schema stamp is not a schema identity -- released
v2.33.0 and v2.34.1 both stamp tlspec 6 with different persisted field sets.

Producer PAIR-CONSISTENCY (gate G5): a ``(writer, tlspec_version)`` pair
refuses only when no governed ledger window says that writer emitted that
stamp. This replaces the deleted hand-typed ``< "2.33"`` inequality that
orphaned lawful released v2.31.0/v2.32.4 artifacts (the first tlspec-6
writer was v2.31.0; measured on genuine wheels, ecosystem panel r3).

Migrated artifacts stay distinguishable from native ones: ``tl.migrate``
never forges writer identity. Instead it writes the append-only migration
witness sidecar (``tl_migration_provenance.json``), and the pair check
accepts a (writer, current-stamp) pair exactly when a VALID witness chains
the artifact from a governed origin pair through adjacent upgrade steps to
the observed stamp. An invalid witness refuses typed -- never a silent gap.

The public promise window number is FORK F1 (unadjudicated):
``PROMISE_WINDOW_MONTHS`` ships ``None`` and ``promised_until`` renders the
honest pending state. The mechanism is complete under both fork branches.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from ._json import read_bounded
from .format_contract import MIN_TLSPEC_VERSION, TLSPEC_VERSION, below_floor_error
from .format_errors import TorchLensIOError

__tl_layer__ = "L1"

#: FORK F1 (ecosystem MEMO section 9): the public analysis-load window in
#: months from exact-writer-contract retirement. ``None`` means the number is
#: pending adjudication; the ledger mechanism is identical under both
#: branches (60 for LONG, 24 for MID) and only the rendered prose changes.
PROMISE_WINDOW_MONTHS: int | None = None

#: Immutable digest of the harvested golden corpus (MEMO 3.7, build gate G1;
#: pinned independently by ``tests/test_tlspec_envelope_goldens.py``).
# The corpus digest is public provenance data, not a credential.
GOLDEN_CORPUS_SHA256 = (
    "3429a84fbd406d374afde5b50f6a8be0867ada77b728d6e88309713c6eadcdb3"  # pragma: allowlist secret
)

#: Migration witness sidecar filename and its closed schema identifier.
MIGRATION_WITNESS_FILENAME = "tl_migration_provenance.json"
MIGRATION_WITNESS_SCHEMA = "tl_migration_provenance_v1"


class SupportClass(str, Enum):
    """Closed support classification for one ledger row (MEMO 3.1)."""

    STABLE_PROMISED = "stable-promised"
    INTERNAL_DEV = "internal-dev"
    LEGACY_EXCEPTION = "legacy-exception"
    SECURITY_QUARANTINED = "security-quarantined"
    REFUSED = "refused"


@dataclass(frozen=True)
class LedgerRow:
    """One governed artifact-compatibility row (MEMO 3.1 row schema).

    Parameters
    ----------
    artifact_kind:
        Product family the row governs (``"trace"``, ``"intervention_spec"``,
        ``"modellog"``).
    writer_release:
        The exact writer release the row describes (dev builds self-report
        their base release; see ``build_commit``/``build_dirty_state``).
    writer_date:
        ISO date of the writer release tag (verified from repository tags).
    exact_release:
        ``True`` when ``writer_release`` is a released wheel; ``False`` for
        dev-tree writers whose self-report is not a release identity.
    build_commit:
        Build commit for dev writers when known, else ``None``.
    build_dirty_state:
        ``"release"`` for released wheels, ``"unknown"`` for harvested dev
        builds (the dev-build identity ruling, MEMO 3.1).
    tlspec_version:
        The manifest schema stamp this writer emitted, or ``None`` for
        pre-tlspec formats (genuine v2.16 ``io_format_version`` bundles).
    writer_contract_digest:
        Normative contract digest when captured; ``None`` for historical
        writers that predate the digest regime (honest absence, never
        backfilled).
    save_levels:
        Save levels the row's goldens cover.
    support_class:
        Closed :class:`SupportClass` value.
    first_reader:
        Earliest release verified to read the row's goldens.
    retired_on:
        ISO date the writer contract retired, or ``None`` while active. The
        promise window starts here (MEMO 3.2).
    runtime_limitations:
        Non-calendar runtime caveats (the save-level asymmetry lives here
        and in runtime policy, never in a second window number).
    golden_ids:
        Harvested golden artifact directory names backing the row.
    last_reader:
        Latest release verified to read the goldens, or ``None`` when the
        current runtime is expected to (and the remedy-actually-loads test
        proves it live).
    bridge_reader:
        The release a refusal remedy may NAME as verified to READ this row's
        artifacts (MEMO 3.1: "verified to READ", never "verified to
        migrate"), or ``None`` when no verified reader exists.
    bridge_reader_evidence:
        Provenance of the bridge-reader verification (executed probe or the
        live CI test), so no remedy is ever named by assertion.
    compatibility_class:
        Free-form class label rendered in reports.
    """

    artifact_kind: str
    writer_release: str
    writer_date: str
    exact_release: bool
    build_commit: str | None
    build_dirty_state: str
    tlspec_version: int | None
    writer_contract_digest: str | None
    save_levels: tuple[str, ...]
    support_class: SupportClass
    first_reader: str
    retired_on: str | None
    runtime_limitations: str
    golden_ids: tuple[str, ...]
    last_reader: str | None
    bridge_reader: str | None
    bridge_reader_evidence: str
    compatibility_class: str

    def promised_until(self) -> str:
        """Render the row's dated promise horizon (MEMO 3.2).

        Returns
        -------
        str
            An ISO date once the writer contract is retired and FORK F1 is
            adjudicated; otherwise the honest pending disclosure. Promises
            lengthen safely and never shorten, so rendering the pending
            state is strictly conservative.
        """

        if self.support_class is not SupportClass.STABLE_PROMISED:
            return f"not-promised ({self.support_class.value})"
        if self.retired_on is None:
            return "active (window starts at exact-writer-contract retirement)"
        if PROMISE_WINDOW_MONTHS is None:
            return f"retired {self.retired_on}; window length pending FORK F1"
        year, month, day = (int(part) for part in self.retired_on.split("-"))
        total = (year * 12 + (month - 1)) + PROMISE_WINDOW_MONTHS
        return f"{total // 12:04d}-{total % 12 + 1:02d}-{day:02d}"


def _released_row(**overrides: Any) -> LedgerRow:
    """Build a released-wheel trace row with the shared column defaults."""

    base: dict[str, Any] = {
        "artifact_kind": "trace",
        "exact_release": True,
        "build_commit": None,
        "build_dirty_state": "release",
        "writer_contract_digest": None,
        "save_levels": ("portable",),
        "support_class": SupportClass.STABLE_PROMISED,
        "retired_on": None,
        "runtime_limitations": (
            "analysis load; replay/execution stay runtime- and capability-attested"
        ),
        "last_reader": None,
        "bridge_reader": "current",
        "bridge_reader_evidence": (
            "live remedy-actually-loads CI (tests/test_ecosystem_rt_ledger.py) "
            "loads the row's goldens under the current runtime on every run"
        ),
        "compatibility_class": "governed-stable",
    }
    base.update(overrides)
    return LedgerRow(**base)


#: The curated ledger rows for every harvested writer (MEMO 3.7: goldens are
#: harvested bytes with provenance; rows are never invented for writers the
#: corpus does not witness). The CURRENT runtime's row is generated live by
#: :func:`runtime_row` so dev-build identity is stamped, never curated.
LEDGER_ROWS: tuple[LedgerRow, ...] = (
    LedgerRow(
        artifact_kind="modellog",
        writer_release="2.16.0",
        writer_date="2026-04-30",
        exact_release=True,
        build_commit=None,
        build_dirty_state="release",
        tlspec_version=None,
        writer_contract_digest=None,
        save_levels=("portable",),
        support_class=SupportClass.LEGACY_EXCEPTION,
        first_reader="2.16.0",
        retired_on="2026-05-01",
        runtime_limitations=(
            "genuine io_format_version=2 ModelLog bundle; readable only by the "
            "verified bridge reader; tl.migrate v1 does not support it"
        ),
        golden_ids=("art_v2.16.0_portable",),
        last_reader="2.17.0",
        bridge_reader="2.17.0",
        bridge_reader_evidence=(
            "executed panel probe (tests/release_goldens/generators/load216.py, "
            "2026-08-26): v2.17.0 loads the genuine bundle; every tested release "
            "v2.22.0 through v2.34.1 fails untyped; the break is bounded to "
            "(2.17.0, 2.22.0]"
        ),
        compatibility_class="legacy-exception-verified-reader",
    ),
    LedgerRow(
        artifact_kind="intervention_spec",
        writer_release="2.16.0",
        writer_date="2026-04-30",
        exact_release=True,
        build_commit=None,
        build_dirty_state="release",
        tlspec_version=None,
        writer_contract_digest=None,
        save_levels=("audit", "portable", "executable_with_callables"),
        support_class=SupportClass.STABLE_PROMISED,
        first_reader="2.16.0",
        retired_on=None,
        runtime_limitations=(
            "spec grammar only; custom-callable execution stays behind the trusted-callable opt-in"
        ),
        golden_ids=(),
        last_reader=None,
        bridge_reader="current",
        bridge_reader_evidence=(
            "v2.16 intervention specs load against faithful fixtures in the "
            "standing suite (tests/test_tlspec_backcompat.py)"
        ),
        compatibility_class="governed-stable",
    ),
    _released_row(
        writer_release="2.31.0",
        writer_date="2026-07-10",
        tlspec_version=6,
        save_levels=("portable", "audit"),
        first_reader="2.31.0",
        golden_ids=("art_v2.31.0_portable", "art_v2.31.0_audit"),
    ),
    _released_row(
        writer_release="2.32.4",
        writer_date="2026-07-27",
        tlspec_version=6,
        first_reader="2.32.4",
        golden_ids=("art_v2.32.4_portable",),
    ),
    _released_row(
        writer_release="2.33.0",
        writer_date="2026-07-27",
        tlspec_version=6,
        first_reader="2.33.0",
        golden_ids=("art_v2.33.0_portable",),
        compatibility_class="governed-stable (same-stamp drift pair with 2.34.1)",
    ),
    _released_row(
        writer_release="2.34.1",
        writer_date="2026-08-10",
        tlspec_version=6,
        first_reader="2.34.1",
        golden_ids=("art_v2.34.1_portable",),
        compatibility_class="governed-stable (same-stamp drift pair with 2.33.0)",
    ),
    LedgerRow(
        artifact_kind="trace",
        writer_release="2.34.1",
        writer_date="2026-08-26",
        exact_release=False,
        build_commit=None,
        build_dirty_state="unknown",
        tlspec_version=8,
        writer_contract_digest=None,
        save_levels=("portable",),
        support_class=SupportClass.INTERNAL_DEV,
        first_reader="unreleased main",
        retired_on=None,
        runtime_limitations=(
            "unreleased dev writer self-reporting 2.34.1; internal transition, "
            "never invented stable history (tlspec 7/8 exist only in main)"
        ),
        golden_ids=("art_main_portable",),
        last_reader=None,
        bridge_reader="current",
        bridge_reader_evidence=(
            "live remedy-actually-loads CI loads art_main_portable under the "
            "current runtime on every run"
        ),
        compatibility_class="internal-dev-transition",
    ),
)


def runtime_row() -> LedgerRow:
    """Generate the current runtime's writer row with dev-build identity.

    The row is generated, never curated: a dev tree self-reports its base
    release while writing a newer schema, so ``exact_release``/dirty state
    are stamped honestly at call time and the live
    ``writer_contract_digest`` is computed from the tree that will write.

    Returns
    -------
    LedgerRow
        The current writer's row (support class ``internal-dev`` until the
        release process curates a released row).
    """

    from .. import __version__
    from .writer_contract import writer_contract_digest

    return LedgerRow(
        artifact_kind="trace",
        writer_release=__version__,
        writer_date="unreleased",
        exact_release=False,
        build_commit=None,
        build_dirty_state="unknown",
        tlspec_version=TLSPEC_VERSION,
        writer_contract_digest=writer_contract_digest(),
        save_levels=("audit", "portable", "executable_with_callables", "runnable"),
        support_class=SupportClass.INTERNAL_DEV,
        first_reader=__version__,
        retired_on=None,
        runtime_limitations=("current writer; becomes a stable-promised row at its release"),
        golden_ids=(),
        last_reader=None,
        bridge_reader="current",
        bridge_reader_evidence="the running interpreter loads its own writes",
        compatibility_class="current-writer",
    )


#: Governed producer windows for pair-consistency (gate G5). Each entry is
#: ``(min_writer_inclusive, max_writer_exclusive_or_None, governed stamps)``.
#: tlspec 6 was first written by released v2.31.0 (measured on genuine
#: wheels; the "first shipped in 2.33" comment was false). tlspec 7/8/9 have
#: only ever been written by dev builds of main self-reporting >= 2.34.1.
_GOVERNED_PRODUCER_WINDOWS: tuple[tuple[str, str | None, frozenset[int]], ...] = (
    ("2.31.0", None, frozenset({6})),
    ("2.34.1", None, frozenset({7, 8, 9})),
)


def governed_stamps_for_writer(writer_release: str) -> frozenset[int]:
    """Return every tlspec stamp the ledger governs for one writer release.

    Parameters
    ----------
    writer_release:
        PEP 440 writer version text from the manifest.

    Returns
    -------
    frozenset[int]
        Governed stamps; empty when the writer is unknown to the ledger or
        the version text does not parse.
    """

    try:
        writer = Version(writer_release)
    except InvalidVersion:
        return frozenset()
    governed: set[int] = set()
    for low, high, stamps in _GOVERNED_PRODUCER_WINDOWS:
        if writer >= Version(low) and (high is None or writer < Version(high)):
            governed.update(stamps)
    return frozenset(governed)


def pair_is_governed(writer_release: str, tlspec_version: int) -> bool:
    """Producer pair-consistency predicate (gate G5).

    Parameters
    ----------
    writer_release:
        Manifest ``torchlens_version`` text.
    tlspec_version:
        Manifest ``tlspec_version`` stamp.

    Returns
    -------
    bool
        ``True`` when a governed window says this writer emitted this stamp.
    """

    return tlspec_version in governed_stamps_for_writer(writer_release)


def _sha256_of_path(path: Path) -> str:
    """SHA-256 hex digest of one file's bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_migration_witness(bundle_path: Path) -> dict[str, Any] | None:
    """Read and structurally validate the migration witness sidecar.

    Parameters
    ----------
    bundle_path:
        Artifact directory that may carry ``tl_migration_provenance.json``.

    Returns
    -------
    dict[str, Any] | None
        The parsed witness when present and valid, else ``None`` when the
        sidecar is absent.

    Raises
    ------
    TorchLensIOError
        With ``code="migration_witness_invalid"`` when a sidecar is present
        but unparseable, mis-schemed, or structurally incoherent. A broken
        witness is never a silent gap: the artifact claims a migrated
        identity it cannot prove, which is exactly the forged-pair case the
        ledger exists to catch.
    """

    witness_path = bundle_path / MIGRATION_WITNESS_FILENAME
    if not witness_path.exists() or witness_path.is_symlink():
        return None
    try:
        witness = read_bounded(witness_path)
    except Exception as exc:  # json/OS errors carry no artifact context
        raise TorchLensIOError(
            f"Migration witness at {witness_path} failed to parse: {exc}. "
            "Remedy: re-run tl.migrate on the original (backup) artifact, or "
            "delete the sidecar if the artifact was never migrated.",
            code="migration_witness_invalid",
            remedy="re-run tl.migrate on the original artifact",
            path=str(witness_path),
        ) from exc
    problems = _witness_structure_problems(witness)
    if problems:
        raise TorchLensIOError(
            f"Migration witness at {witness_path} is structurally invalid: "
            f"{'; '.join(problems)}. Remedy: re-run tl.migrate on the original "
            "(backup) artifact; witnesses are append-only tool output, never "
            "hand-edited.",
            code="migration_witness_invalid",
            remedy="re-run tl.migrate on the original artifact",
            path=str(witness_path),
        )
    return witness


def _witness_step_problems(steps: list[Any], previous: int | None) -> tuple[list[str], int | None]:
    """Validate the witness step chain; return (problems, chain tail).

    Parameters
    ----------
    steps:
        The parsed ``steps`` list.
    previous:
        The origin ``tlspec_version`` the first step must chain from
        (``None`` when the origin block was itself invalid).

    Returns
    -------
    tuple[list[str], int | None]
        Problems found, and the final chained stamp (``None`` on a
        malformed chain).
    """

    problems: list[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return [*problems, f"step {index} is not an object"], None
        source = step.get("source_tlspec")
        target = step.get("target_tlspec")
        if not isinstance(source, int) or not isinstance(target, int):
            return [*problems, f"step {index} missing integer source/target stamps"], None
        if target != source + 1:
            problems.append(f"step {index} is not adjacent ({source} -> {target})")
        if previous is not None and source != previous:
            problems.append(f"step {index} source {source} does not chain from {previous}")
        previous = target
    return problems, previous


def _witness_structure_problems(witness: Any) -> list[str]:
    """Collect structural problems with one parsed migration witness."""

    if not isinstance(witness, dict):
        return ["witness is not a JSON object"]
    problems: list[str] = []
    if witness.get("witness_schema") != MIGRATION_WITNESS_SCHEMA:
        problems.append(
            f"unknown witness_schema {witness.get('witness_schema')!r} "
            f"(expected {MIGRATION_WITNESS_SCHEMA!r})"
        )
    origin = witness.get("origin")
    steps = witness.get("steps")
    if not isinstance(origin, dict) or not isinstance(origin.get("tlspec_version"), int):
        problems.append("origin block missing or missing integer tlspec_version")
    if not isinstance(steps, list) or not steps:
        problems.append("steps must be a non-empty list")
        return problems
    origin_version = origin.get("tlspec_version") if isinstance(origin, dict) else None
    step_problems, previous = _witness_step_problems(steps, origin_version)
    problems.extend(step_problems)
    if step_problems and previous is None:
        return problems
    final = witness.get("final_tlspec_version")
    if not isinstance(final, int) or final != previous:
        problems.append(
            f"final_tlspec_version {final!r} does not match the step chain tail {previous!r}"
        )
    return problems


def migrated_pair_is_governed(
    writer_release: str,
    tlspec_version: int,
    witness: dict[str, Any],
    *,
    bundle_path: Path | None = None,
) -> bool:
    """Pair-consistency for a MIGRATED artifact via its witness chain.

    A migrated artifact keeps its original writer identity (migration never
    forges a writer), so its ``(writer, stamp)`` pair is governed exactly
    when the witness chains from a governed ORIGIN pair through adjacent
    steps to the observed stamp, and -- when the artifact directory is
    available -- the witness's final manifest digest matches the manifest
    actually on disk (a witness copied beside a different manifest proves
    nothing).

    Parameters
    ----------
    writer_release:
        Manifest ``torchlens_version`` (the ORIGINAL writer's identity).
    tlspec_version:
        The observed (post-migration) manifest stamp.
    witness:
        Parsed witness from :func:`read_migration_witness`.
    bundle_path:
        Artifact directory for the manifest-digest cross-check, when the
        caller has it in scope.

    Returns
    -------
    bool
        ``True`` when the witness proves a governed lineage for the pair.
    """

    origin = witness.get("origin", {})
    origin_writer = origin.get("torchlens_version")
    origin_stamp = origin.get("tlspec_version")
    if origin_writer != writer_release or not isinstance(origin_stamp, int):
        return False
    if not pair_is_governed(origin_writer, origin_stamp):
        return False
    if witness.get("final_tlspec_version") != tlspec_version:
        return False
    if bundle_path is not None:
        recorded = witness.get("final_manifest_sha256")
        manifest_path = bundle_path / "manifest.json"
        if not isinstance(recorded, str) or not manifest_path.exists():
            return False
        if _sha256_of_path(manifest_path) != recorded:
            return False
    return True


def ungoverned_pair_error(
    writer_release: str,
    tlspec_version: int,
    *,
    subject: str = "Bundle",
    code: str = "artifact_producer_pair_ungoverned",
) -> TorchLensIOError:
    """Build the typed producer pair-consistency refusal (gate G5).

    Parameters
    ----------
    writer_release:
        Manifest writer version text.
    tlspec_version:
        Manifest schema stamp.
    subject:
        Human-readable subject for the message.
    code:
        Always ``"artifact_producer_pair_ungoverned"``; raise sites pass it
        explicitly so the S-17 census sees the code where the raise happens,
        not buried in this factory (registry-kernel precedent).

    Returns
    -------
    TorchLensIOError
        Typed refusal with ``code="artifact_producer_pair_ungoverned"`` and
        a ledger-derived remedy.
    """

    governed = sorted(governed_stamps_for_writer(writer_release))
    if governed:
        detail = f"that writer's governed stamps are {governed}"
    else:
        detail = "no governed ledger window covers that writer"
    remedy = (
        "verify the artifact's provenance; a lawful pair appears in "
        "torchlens.ecosystem.compat_window(), and a migrated artifact must "
        "carry its tl_migration_provenance.json witness"
    )
    return TorchLensIOError(
        f"{subject} claims torchlens_version={writer_release} with "
        f"tlspec_version={tlspec_version}, but {detail}. No governed ledger row "
        f"says that writer emitted that stamp; the pair is inconsistent "
        f"(forged, corrupted, or an unwitnessed migration). Remedy: {remedy}.",
        code=code,
        remedy=remedy,
        writer_release=writer_release,
        observed_tlspec_version=tlspec_version,
        governed_tlspec_versions=governed,
    )


def raise_if_modellog_portable(tlspec_format: str, subject: str, path: str) -> None:
    """Refuse a genuine v2.16 portable ModelLog artifact with the G6 remedy.

    The ONE construction of the ledger-derived v2.16 refusal, shared by the
    ``tl.load`` dispatch and the schema-validation door (gates G6/G7): the
    shipped refusal told genuine v2.16 ModelLog holders to use ">= 2.33" --
    releases the panel proved unable to read their artifacts. The remedy
    names the ledger's verified bridge reader (v2.17.0) and states that
    tl.migrate v1 does not support this artifact. Any other format value
    returns without effect, so callers keep their dispatch shape.

    Parameters
    ----------
    tlspec_format:
        The ``detect_tlspec_format`` result for the artifact.
    subject:
        Human-readable subject for the refusal message.
    path:
        Artifact path for the structured fields.

    Raises
    ------
    ArtifactVersionBelowFloorError
        When ``tlspec_format`` is the genuine v2.16 portable ModelLog format.
    """

    if tlspec_format != "v2.16_modellog_portable":
        return
    raise below_floor_error(
        observed="the TorchLens 2.16 portable ModelLog format (pre-tlspec)",
        subject=subject,
        path=path,
        remedy=bridge_reader_remedy("modellog", writer_release="2.16.0"),
        code="artifact_version_below_floor",
    )


def bridge_reader_remedy(artifact_kind: str, *, writer_release: str | None = None) -> str:
    """Derive a refusal remedy from the ledger's bridge_reader column (G6).

    Remedies are DERIVED, never hand-written in a constructor: main shipped a
    typed refusal telling genuine v2.16 ModelLog holders to use ">= 2.33",
    releases proven unable to read their artifacts. A remedy may name a
    release only when a ledger row records verified read evidence for it.

    Parameters
    ----------
    artifact_kind:
        Ledger artifact kind (``"modellog"``, ``"trace"``,
        ``"intervention_spec"``).
    writer_release:
        Writer version text when known, to select among rows.

    Returns
    -------
    str
        The remedy sentence. When no verified bridge reader exists the
        remedy honestly says so instead of naming an unverified release.
    """

    for row in LEDGER_ROWS:
        if row.artifact_kind != artifact_kind:
            continue
        if writer_release is not None and row.writer_release != writer_release:
            continue
        if row.bridge_reader is None:
            break
        if row.bridge_reader == "current":
            return (
                "load the artifact with the current torchlens release (verified "
                "by the remedy-actually-loads CI test on the row's goldens)"
            )
        return (
            f"inspect the artifact with torchlens=={row.bridge_reader} (the "
            f"verified reader: {row.bridge_reader_evidence}); tl.migrate v1 "
            "does not support this artifact"
        )
    return (
        "no verified bridge reader is recorded in the compatibility ledger for "
        "this format era; re-save with the release that wrote it. See "
        "torchlens.ecosystem.compat_window() for the governed rows"
    )


def ledger_rows() -> tuple[LedgerRow, ...]:
    """Return every governed row plus the generated current-runtime row.

    Returns
    -------
    tuple[LedgerRow, ...]
        Curated harvested-writer rows followed by :func:`runtime_row`.
    """

    return (*LEDGER_ROWS, runtime_row())


def floor_metadata() -> dict[str, Any]:
    """The rehydration floor facts the ledger publishes.

    Returns
    -------
    dict[str, Any]
        Floor/ceiling stamps and the producer floor writer (2.31.0, the
        first tlspec-6 writer, measured on genuine wheels -- gate G5).
    """

    return {
        "floor_tlspec_version": MIN_TLSPEC_VERSION,
        "ceiling_tlspec_version": TLSPEC_VERSION,
        "producer_floor_writer": "2.31.0",
        "golden_corpus_sha256": GOLDEN_CORPUS_SHA256,
        "promise_window_months": PROMISE_WINDOW_MONTHS,
        "promise_window_status": (
            "pending FORK F1 adjudication" if PROMISE_WINDOW_MONTHS is None else "adjudicated"
        ),
    }
