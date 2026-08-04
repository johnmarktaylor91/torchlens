"""Honest disposition for a model routed to an environment that cannot serve it.

Routing is decided at intake from the immutable roster row (zoo and era); the
distribution a model actually needs is decided later, by the author, in the
declarative recipe. When the two disagree the model lands in an intent whose
package set does not contain its library at all, and
``recipe.resolve_environment_artifact_digest`` correctly refuses to invent a
digest for a distribution the routed environment does not install.

That refusal is right and stays exactly as it is. What was wrong is the RECORD
it produced. A DGL model routed to ``core`` -- because its roster zoo carried no
graph marker -- terminalized as ``failed:runner`` / ``protocol-violation``, which
asserts our pipeline broke on the model. Nothing broke. The catalog deliberately
keeps ``core`` free of DGL (conda-forge's ``dgl`` hard-pins torch to 2.3.1 and
drags in TensorFlow, so admitting it would downgrade torch and numpy for every
core model), and there is a ``graph`` intent that carries DGL precisely so those
models have a home. The truthful statement is "this needs the graph
environment", and it names the thing that would cover the model later.

This module derives that statement:

* :func:`assess_environment_coverage` compares the recipe's OWN declared
  distribution against the routed intent's exact resolved-export inventory --
  the same inventory, through the same namespace bridge, that the refusal reads
  -- so the verdict can never disagree with the refusal it explains.
* When the routed intent cannot serve it, the registry is searched for an intent
  that can, and the row records which intent and on what BASIS: a locked
  inventory that provably contains it, a declared dependency that asks for it,
  or the routing table that already maps that package to an intent.
* The result is a durable, append-only row carrying the declared distribution,
  the routed intent, the covering intent, and a recheck hint -- so a reviewer
  can act on it without re-running anything.

Three properties are deliberate, and they mirror
:mod:`menagerie.crawler.host_capacity` on purpose: these are the same kind of
fact ("this campaign cannot run this model here, and here is what would") and
they should read the same way.

**It is a refusal, not a relaxation.** Nothing here weakens a validation check.
A model whose library genuinely is not installed still never runs, still gets no
artifact digest, and still asserts no rung. It is simply withheld and recorded
instead of being blamed.

**It is not a canonical terminal.** Every canonical ``deferred:*`` model status
is proved either by a checker-adjudicated DEFER gate or by an accepted BLOCKED
gate; there is no third proof path. This disposition is machine-derived from an
inventory, has no adjudication behind it, and therefore does not claim one.
See :data:`COVERAGE_DEFERRAL_DISPOSITION`.

**It is fail-open toward attempting.** A model is withheld ONLY when the routed
environment publishes an inventory AND that inventory provably lacks the
declared distribution. No inventory, no declarative recipe, or an unreadable
recipe all mean "not assessable", which admits the model unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import fcntl
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    canonical_json_bytes,
    fsync_directory,
    stable_hash,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.package_namespace import (
    canonical_distribution_name,
    inventory_row_provides_distribution,
)

COVERAGE_DEFERRAL_SCHEMA_VERSION = "menagerie.crawler.environment-coverage-deferral.v1"

COVERAGE_DEFERRAL_DISPOSITION = "deferred:needs-environment-coverage"
"""Greppable disposition for a model routed to an intent that cannot serve it.

It follows the established ``deferred:needs-<capability>`` shape
(``needs-cuda``, ``needs-x86``, ``needs-source-access``, ``needs-more-memory``):
a real, located, correctly authored model that this campaign cannot execute AS
ROUTED, naming the capability that would recover it. A different environment
intent is such a capability, and the row names the exact one.

It is deliberately NOT ``failed:*``. A ``failed:`` code asserts that our pipeline
broke on a model. Nothing broke: the author declared a real distribution, the
routed environment honestly does not install it, and the refusal that says so is
working. The defect is in the ROUTING, which is a campaign-level fact and is
recoverable by re-running the model against the intent that carries its library.

It is also deliberately not minted as a canonical model terminal, for exactly the
reason ``host_capacity.CAPACITY_DEFERRAL_DISPOSITION`` is not: a canonical
``deferred:*`` record must carry a checker-adjudicated terminal gate over frozen
sources (see ``authority._derive_deferral`` and
``reducer._derive_blocked_capability_deferral_proof``), and this disposition is
machine-derived from a package inventory with no such adjudication. Claiming
that terminal would mean fabricating a proof. The model instead stays
UNCOMPLETED for this campaign and is recorded here, which is the truthful
statement: not attempted, not failed, recoverable in a named environment.
"""

COVERAGE_DEFERRAL_RELATIVE_PATH = (
    Path("intake-extensions") / "environment-coverage-deferrals.jsonl"
)

_SPEC_NAME = re.compile(r"[A-Za-z0-9._-]+")


class CoverageVerdict(StrEnum):
    """Closed routed-environment coverage outcomes."""

    SERVED = "served"
    """The routed environment installs the declared distribution."""

    UNSERVABLE = "unservable"
    """The routed environment publishes an inventory that provably lacks it."""

    NOT_ASSESSABLE = "not-assessable"
    """No inventory or no declarative distribution; the model is admitted."""


class CoverageBasis(StrEnum):
    """How a covering intent was established, strongest evidence first."""

    LOCKED_INVENTORY = "locked-inventory"
    """The candidate intent's own locked resolved export contains the package."""

    DECLARED_DEPENDENCY = "declared-dependency"
    """The candidate intent's environment spec asks for the package by name."""

    ROUTING_TABLE = "routing-table"
    """``routing._PACKAGE_INTENTS`` already maps this package to the intent."""


class CoverageDeferralError(ValueError):
    """Raised when an environment-coverage deferral row is malformed or unpersistable."""


@dataclass(frozen=True)
class DeclaredLibrary:
    """The distribution and version one declarative recipe pins."""

    distribution: str
    version: str

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this pin."""

        return {"distribution": self.distribution, "version": self.version}


@dataclass(frozen=True)
class CoveringIntent:
    """One environment intent that would carry the declared distribution."""

    intent: str
    basis: CoverageBasis
    evidence: str

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this coverage claim."""

        return {"intent": self.intent, "basis": self.basis.value, "evidence": self.evidence}


@dataclass(frozen=True)
class CoverageAssessment:
    """One complete routed-environment coverage decision with its justification."""

    verdict: CoverageVerdict
    routed_intent: Optional[str]
    library: Optional[DeclaredLibrary]
    covering: tuple[CoveringIntent, ...]
    explanation: str

    @property
    def deferred(self) -> bool:
        """Return whether this assessment withholds the model from its route."""

        return self.verdict is CoverageVerdict.UNSERVABLE

    @property
    def preferred_intent(self) -> Optional[str]:
        """Return the strongest-evidence covering intent, when one exists."""

        return self.covering[0].intent if self.covering else None

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this decision."""

        return {
            "verdict": self.verdict.value,
            "routed_intent": self.routed_intent,
            "library": self.library.to_json() if self.library is not None else None,
            "covering_intents": [item.to_json() for item in self.covering],
            "explanation": self.explanation,
        }


def declared_library(implementation: Mapping[str, Any]) -> Optional[DeclaredLibrary]:
    """Return the distribution pin a declarative-library recipe declares.

    The guard conditions mirror ``recipe.bind_library_artifact_digest`` exactly,
    so this can only ever speak about recipes that binding actually resolves.

    Parameters
    ----------
    implementation:
        Proposal implementation block.

    Returns
    -------
    DeclaredLibrary | None
        The pinned distribution and version, or ``None`` when this recipe is not
        a declarative-library pin at all.
    """

    if implementation.get("recipe_type") != "declarative-library":
        return None
    recipe = implementation.get("library_recipe")
    if not isinstance(recipe, Mapping):
        return None
    distribution = recipe.get("distribution")
    version = recipe.get("version")
    if not isinstance(distribution, str) or not isinstance(version, str):
        return None
    if not distribution.strip():
        return None
    return DeclaredLibrary(distribution=distribution, version=version)


def _inventory_provides(
    packages: Sequence[Mapping[str, Any]], distribution: str
) -> Optional[str]:
    """Return the inventory row name providing ``distribution``, if any."""

    for row in packages:
        if not isinstance(row, Mapping):
            continue
        name = row.get("name")
        if isinstance(name, str) and inventory_row_provides_distribution(name, distribution):
            return name
    return None


def dependency_spec_name(spec: str) -> str:
    """Return the bare package name a conda match spec asks for.

    Specs in an intent's ``environment.yml`` carry channels and version bounds
    (``conda-forge::pytorch>=2.3``); the coverage question is only about the
    NAME, so everything after it is discarded.

    Parameters
    ----------
    spec:
        One declared dependency match spec.

    Returns
    -------
    str
        The bare package name, or an empty string for an unparseable spec.
    """

    candidate = spec.strip()
    if "::" in candidate:
        candidate = candidate.rsplit("::", 1)[1]
    match = _SPEC_NAME.match(candidate)
    return match.group(0) if match is not None else ""


def _routing_table_intent(distribution: str) -> Optional[str]:
    """Return the intent ``routing._PACKAGE_INTENTS`` already maps this package to."""

    # Imported here rather than at module scope: `routing` is a peer of the
    # driver's import graph and this module is also loaded by the CLI, which has
    # no reason to pull the routing tables in.
    from menagerie.crawler.routing import _PACKAGE_INTENTS

    wanted = canonical_distribution_name(distribution)
    for intent, markers in _PACKAGE_INTENTS:
        if any(canonical_distribution_name(marker) == wanted for marker in markers):
            return intent
    return None


def find_covering_intents(
    distribution: str,
    registry: Any,
    *,
    exclude: Optional[str] = None,
) -> tuple[CoveringIntent, ...]:
    """Find every intent that would carry ``distribution``, strongest basis first.

    Three independent signals are consulted, in descending order of how directly
    each one proves the claim. The BASIS travels with the answer, so a reader can
    tell a solved lock ("this intent demonstrably installs it") from a declared
    dependency ("this intent asks for it") from the routing table ("this package
    is already meant to route there"), rather than seeing one undifferentiated
    recommendation.

    Parameters
    ----------
    distribution:
        Declared Python distribution the model needs.
    registry:
        Loaded :class:`menagerie.crawler.envs.EnvironmentRegistry`, or ``None``.
    exclude:
        Routed intent to omit; it is by construction not a candidate.

    Returns
    -------
    tuple[CoveringIntent, ...]
        One entry per covering intent, ordered by basis strength then name. Empty
        when nothing in this campaign declares the distribution.
    """

    found: dict[str, CoveringIntent] = {}
    intents = getattr(registry, "intents", None)
    if isinstance(intents, Mapping):
        for name in sorted(intents):
            if name == exclude:
                continue
            intent = intents[name]
            packages = _resolved_export_packages(intent)
            row_name = _inventory_provides(packages, distribution)
            if row_name is not None:
                found[name] = CoveringIntent(
                    intent=name,
                    basis=CoverageBasis.LOCKED_INVENTORY,
                    evidence=(
                        f"the locked resolved export for {name!r} installs package "
                        f"{row_name!r}, which provides {distribution!r}"
                    ),
                )
                continue
            for spec in getattr(intent, "dependencies", ()):  # declared, not solved
                spec_name = dependency_spec_name(str(spec))
                if spec_name and inventory_row_provides_distribution(spec_name, distribution):
                    found[name] = CoveringIntent(
                        intent=name,
                        basis=CoverageBasis.DECLARED_DEPENDENCY,
                        evidence=(
                            f"the {name!r} environment spec declares dependency "
                            f"{str(spec).strip()!r}"
                        ),
                    )
                    break
    routed = _routing_table_intent(distribution)
    if routed is not None and routed != exclude and routed not in found:
        found[routed] = CoveringIntent(
            intent=routed,
            basis=CoverageBasis.ROUTING_TABLE,
            evidence=(
                f"routing._PACKAGE_INTENTS maps package {distribution!r} to intent {routed!r}"
            ),
        )
    order = {
        CoverageBasis.LOCKED_INVENTORY: 0,
        CoverageBasis.DECLARED_DEPENDENCY: 1,
        CoverageBasis.ROUTING_TABLE: 2,
    }
    return tuple(sorted(found.values(), key=lambda item: (order[item.basis], item.intent)))


def _resolved_export_packages(intent: Any) -> tuple[Mapping[str, Any], ...]:
    """Return one intent's exact resolved-export package rows, or nothing."""

    lock = getattr(intent, "lock", None)
    export_bytes = getattr(lock, "export_bytes", None)
    if not isinstance(export_bytes, (bytes, bytearray)):
        return ()
    # Imported lazily for the same reason as the routing table: a ledger reader
    # should not need the environment-exactness machinery to print rows.
    from menagerie.crawler.env_lifecycle import EnvironmentExactnessError, parse_resolved_export

    try:
        value = json.loads(parse_resolved_export(bytes(export_bytes)))
    except (EnvironmentExactnessError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    packages = value.get("packages") if isinstance(value, Mapping) else None
    if not isinstance(packages, list):
        return ()
    return tuple(row for row in packages if isinstance(row, Mapping))


def assess_environment_coverage(
    implementation: Mapping[str, Any],
    *,
    routed_intent: Optional[str],
    routed_packages: Sequence[Mapping[str, Any]],
    registry: Any = None,
) -> CoverageAssessment:
    """Decide whether the routed environment can serve one authored recipe.

    Parameters
    ----------
    implementation:
        Proposal implementation block carrying the declarative recipe.
    routed_intent:
        Environment intent this model was routed to at intake.
    routed_packages:
        Exact package rows from that intent's resolved export. Empty means the
        intent is unlocked, which is "unknown", never "absent".
    registry:
        Loaded environment registry used to name a covering intent.

    Returns
    -------
    CoverageAssessment
        Verdict, the declared pin, any covering intents, and a plain-language
        reason.
    """

    library = declared_library(implementation)
    if library is None:
        return CoverageAssessment(
            verdict=CoverageVerdict.NOT_ASSESSABLE,
            routed_intent=routed_intent,
            library=None,
            covering=(),
            explanation=(
                "the proposal pins no declarative library distribution, so routed "
                "coverage says nothing about it and the model is admitted"
            ),
        )
    if not routed_packages:
        return CoverageAssessment(
            verdict=CoverageVerdict.NOT_ASSESSABLE,
            routed_intent=routed_intent,
            library=library,
            covering=(),
            explanation=(
                f"the routed intent {routed_intent!r} publishes no package inventory, so "
                "whether it installs "
                f"{library.distribution!r} is unknown and the model is admitted"
            ),
        )
    row_name = _inventory_provides(routed_packages, library.distribution)
    if row_name is not None:
        return CoverageAssessment(
            verdict=CoverageVerdict.SERVED,
            routed_intent=routed_intent,
            library=library,
            covering=(),
            explanation=(
                f"the routed intent {routed_intent!r} installs package {row_name!r}, which "
                f"provides {library.distribution!r}"
            ),
        )
    covering = find_covering_intents(library.distribution, registry, exclude=routed_intent)
    if covering:
        names = ", ".join(repr(item.intent) for item in covering)
        remedy = f"the {names} environment carries it"
    else:
        remedy = "no environment intent in this campaign declares it"
    return CoverageAssessment(
        verdict=CoverageVerdict.UNSERVABLE,
        routed_intent=routed_intent,
        library=library,
        covering=covering,
        explanation=(
            f"the model pins distribution {library.distribution!r}, and the intent it "
            f"routes to ({routed_intent!r}) installs no package providing it; {remedy}"
        ),
    )


def coverage_deferral_path(records_root: Path) -> Path:
    """Return the canonical environment-coverage deferral ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only environment-coverage deferral JSONL path.
    """

    return records_root / COVERAGE_DEFERRAL_RELATIVE_PATH


def build_coverage_deferral_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    assessment: CoverageAssessment,
) -> JsonObject:
    """Assemble one complete, self-describing environment-coverage deferral row.

    The row states the declared pin, the intent that could not serve it, and
    every intent that could -- with the evidence for each -- so a later reviewer
    can act on it without the code that made the call.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the active scheduled work generation.
    campaign_id, run_id, machine_id:
        Campaign and host provenance for the deferral.
    created_at:
        UTC timestamp of the decision.
    assessment:
        The withholding coverage decision.

    Returns
    -------
    dict[str, Any]
        Content-addressed deferral row.

    Raises
    ------
    CoverageDeferralError
        If the assessment does not actually withhold the model.
    """

    if not assessment.deferred:
        raise CoverageDeferralError(
            "only an unservable coverage assessment may be recorded as a deferral"
        )
    assert assessment.library is not None  # guaranteed by CoverageVerdict.UNSERVABLE
    distribution = assessment.library.distribution
    if assessment.preferred_intent is not None:
        recheck_hint = (
            f"re-run this model routed to the {assessment.preferred_intent!r} environment "
            f"intent, which carries distribution {distribution!r}; the intent it routed to "
            f"({assessment.routed_intent!r}) deliberately does not"
        )
    else:
        recheck_hint = (
            f"declare distribution {distribution!r} in an environment intent (or add an "
            "intent that carries it) and re-run this model; no intent in this campaign "
            "declares it today"
        )
    row: JsonObject = {
        "schema_version": COVERAGE_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": COVERAGE_DEFERRAL_DISPOSITION,
        "coverage": assessment.to_json(),
        "recheck_hint": recheck_hint,
    }
    row["row_sha256"] = stable_hash(row)
    return row


def validate_coverage_deferral_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one environment-coverage deferral row and return its canonical copy.

    Parameters
    ----------
    row:
        Candidate deferral row.

    Returns
    -------
    dict[str, Any]
        Validated row.

    Raises
    ------
    CoverageDeferralError
        If a required field is missing, mistyped, or the digest disagrees.
    """

    required = (
        "schema_version",
        "stable_id",
        "work_id",
        "name",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
        "disposition",
        "coverage",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise CoverageDeferralError(
            f"environment-coverage deferral row is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != COVERAGE_DEFERRAL_SCHEMA_VERSION:
        raise CoverageDeferralError(
            f"environment-coverage deferral row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] != COVERAGE_DEFERRAL_DISPOSITION:
        raise CoverageDeferralError(
            "environment-coverage deferral row declares unknown disposition "
            f"{row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise CoverageDeferralError(
                f"environment-coverage deferral row has an empty {field}"
            )
    coverage = row["coverage"]
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("verdict") != CoverageVerdict.UNSERVABLE.value
    ):
        raise CoverageDeferralError(
            "environment-coverage deferral row does not record an unservable routing"
        )
    library = coverage.get("library")
    if not isinstance(library, Mapping) or not isinstance(library.get("distribution"), str):
        raise CoverageDeferralError(
            "environment-coverage deferral row must name the distribution it could not serve"
        )
    if not isinstance(coverage.get("covering_intents"), list):
        raise CoverageDeferralError(
            "environment-coverage deferral row must state its covering intents, even when empty"
        )
    if not isinstance(row["recheck_hint"], str) or not row["recheck_hint"].strip():
        raise CoverageDeferralError(
            "environment-coverage deferral row must name what would cover the model later"
        )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    digest = validated.pop("row_sha256")
    if stable_hash(validated) != digest:
        raise CoverageDeferralError(
            "environment-coverage deferral row digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def append_coverage_deferral_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one environment-coverage deferral durably and idempotently.

    Parameters
    ----------
    path:
        Destination environment-coverage deferral JSONL.
    row:
        Complete typed deferral row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    CoverageDeferralError
        If the row is malformed.

    Notes
    -----
    The FIRST deferral recorded for a work generation is the record, and a later
    run that re-derives it is a no-op -- the same rule
    ``host_capacity.append_capacity_deferral_row`` follows, and for the same
    reason: ``created_at`` and ``run_id`` necessarily differ on every resume, so
    treating a re-derived deferral as a conflict would abort the driver on the
    second pass over the same model. A model that has stopped being unservable
    writes nothing here, so a stale row can only ever be superseded by the model
    completing normally.
    """

    validated = validate_coverage_deferral_row(row)
    key = (validated["stable_id"], validated["work_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = _decode_coverage_lines(handle.read(), path)
        matching = [item for item in existing if (item["stable_id"], item["work_id"]) == key]
        if matching:
            return matching[0]
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def _decode_coverage_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one deferral ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CoverageDeferralError(
                f"environment-coverage deferral ledger {path} has an unreadable row "
                f"at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise CoverageDeferralError(
                f"environment-coverage deferral ledger {path} has a non-object row "
                f"at line {index}"
            )
        rows.append(validate_coverage_deferral_row(decoded))
    return tuple(rows)


def load_coverage_deferral_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load environment-coverage deferrals from one or more campaign roots.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Rows ordered by stable ID then work ID.
    """

    by_key: dict[tuple[str, str], JsonObject] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in _decode_coverage_lines(path.read_bytes(), path):
            by_key[(row["stable_id"], row["work_id"])] = row
    return tuple(by_key[key] for key in sorted(by_key))
