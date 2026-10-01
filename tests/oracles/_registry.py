"""The obligation registry: Table A + Table B (oracles D9, build item 3).

Every correctness contract is defined ONCE in Table A (obligation
definitions: what a quantity/behavior means, its unit and convention, its
state contract, its required evidence grade) and bound to every generated
place it is accepted or emitted in Table B (exposure bindings: one row per
door, machine-walked, so no door can hide). The A/B split makes the parity
sweep a one-line query over bindings sharing an obligation id -- and parity
NEVER populates an oracle cell (D4): internal agreement between our own
doors is not evidence.

Evidence grades (I0/I1/I2, SOL's independence ladder): I0 = same-root /
parity (never an oracle), I1 = independent re-derivation from raw state,
I2 = closed form or external implementation at a pinned version.

Wave 0 ships the schema, the validation gates, the nine census-root
descriptors (``_censuses.CENSUS_ROOTS``), and a seed population; Waves 1-2
populate the full binding matrix from the censuses.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

EVIDENCE_GRADES = ("I0", "I1", "I2")
PURITY_CONTRACTS = ("PURE_OBSERVER", "FORWARD_EQUIVALENT", "DECLARED_MUTATOR", "UNSET")
PURITY_MECHANISMS = ("untouched", "restored", "n/a", "unset")
DISCLOSURE_KINDS = ("convention", "limitation", "n/a")
HISTORY_CELLS = ("H-CLEAN", "H-WARM", "H-FAILED", "H-BOTH")


@dataclass(frozen=True)
class ObligationDef:
    """Table A row: one correctness contract, defined once.

    Parameters
    ----------
    obligation_id:
        Stable id (``OBL-...``).
    title:
        One-line contract statement.
    meaning:
        What the quantity/behavior means: unit, convention, counted set,
        or state contract -- typed here ONCE so cross-door divergence can
        never live inside the registry (D9).
    evidence_grade:
        Required independence grade (``I1``/``I2``; ``I0`` is parity and can
        gate but never satisfy an obligation).
    state_contract:
        One of ``PURITY_CONTRACTS`` (``UNSET`` until FORK-A rules).
    owner:
        Lane/owner accountable for the obligation.
    """

    obligation_id: str
    title: str
    meaning: str
    evidence_grade: str
    state_contract: str
    owner: str


@dataclass(frozen=True)
class ExposureBinding:
    """Table B row: one door bound to one obligation.

    Parameters
    ----------
    binding_id:
        Stable id (``BND-...``).
    obligation_id:
        The Table A row this door is bound to; must resolve exactly.
    door:
        Dotted path of the public place the value comes out of.
    census_root:
        The generator census that produced this binding (``CR1``..``CR9``).
    classification:
        Surface classification of the door (``declared``/``submodule``/
        ``deprecated``/``undeclared``).
    purity_mechanism:
        Metadata selecting required tests (D10): ``untouched`` rows need no
        restore plants; ``restored`` rows OWE the mid-call exception plant.
    scale_sensitive:
        ``yes``/``no``: whether the predicate re-runs at a scale profile.
    scale_profile_id:
        Named profile when ``scale_sensitive`` is ``yes`` (D23), else empty.
    disclosure_kind:
        ``convention`` (permanent, two-sided tested) vs ``limitation``
        (dated, expiring, counted) vs ``n/a`` (D12).
    positive_control:
        Named check proving the measurement channel is ALIVE (D7); required
        non-empty for every binding with a fixture channel.
    invariance_witness:
        Registered witness id for declared dont-cares (D14), else empty.
    history_cells:
        Comma-joined subset of ``HISTORY_CELLS`` this binding must run under
        (D16), else empty.
    """

    binding_id: str
    obligation_id: str
    door: str
    census_root: str
    classification: str
    purity_mechanism: str
    scale_sensitive: str
    scale_profile_id: str
    disclosure_kind: str
    positive_control: str
    invariance_witness: str
    history_cells: str


@dataclass(frozen=True)
class Registry:
    """The loaded two-table registry.

    Parameters
    ----------
    obligations:
        Table A rows.
    bindings:
        Table B rows.
    """

    obligations: tuple[ObligationDef, ...]
    bindings: tuple[ExposureBinding, ...]


def _load_rows(filename: str, row_type: type) -> tuple:
    """Load one TSV into typed rows.

    Parameters
    ----------
    filename:
        File under ``data/``.
    row_type:
        Frozen dataclass to construct per row.

    Returns
    -------
    tuple
        Typed rows.

    Raises
    ------
    ValueError
        When columns do not exactly match the dataclass fields.
    """

    expected = tuple(field.name for field in fields(row_type))
    with (DATA_DIR / filename).open(newline="") as handle:
        reader = csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter="\t", restval=""
        )
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"{filename}: columns {reader.fieldnames} != schema {expected} -- the "
                "registry schema is a contract; widen it consciously, never by drift"
            )
        return tuple(row_type(**record) for record in reader)


def load_registry(
    obligations_file: str = "obligations.tsv",
    bindings_file: str = "bindings.tsv",
) -> Registry:
    """Load both tables.

    Parameters
    ----------
    obligations_file:
        Table A filename under ``data/``.
    bindings_file:
        Table B filename under ``data/``.

    Returns
    -------
    Registry
        The loaded registry (validate separately with :func:`validate`).
    """

    return Registry(
        obligations=_load_rows(obligations_file, ObligationDef),
        bindings=_load_rows(bindings_file, ExposureBinding),
    )


def validate(registry: Registry) -> tuple[str, ...]:
    """Validate the registry's closure and closed vocabularies.

    The Table A/B join is EXACT (H0): every binding's obligation resolves,
    every obligation is bound at least once, ids are unique, and every
    enumerated column holds a token from its closed vocabulary.

    Parameters
    ----------
    registry:
        The loaded registry.

    Returns
    -------
    tuple[str, ...]
        Findings; empty means valid. Returned as data so the corruption
        plants can assert on exact findings.
    """

    findings: list[str] = []
    obligation_ids = [row.obligation_id for row in registry.obligations]
    if len(set(obligation_ids)) != len(obligation_ids):
        findings.append("duplicate obligation ids")
    binding_ids = [row.binding_id for row in registry.bindings]
    if len(set(binding_ids)) != len(binding_ids):
        findings.append("duplicate binding ids")
    known_obligations = set(obligation_ids)
    known_roots = {root.root_id for root in _census_roots()}
    bound: set[str] = set()
    for row in registry.obligations:
        if row.evidence_grade not in ("I1", "I2"):
            findings.append(
                f"{row.obligation_id}: evidence grade {row.evidence_grade!r} cannot "
                "satisfy an obligation (I0/parity never populates an oracle cell)"
            )
        if row.state_contract not in PURITY_CONTRACTS:
            findings.append(f"{row.obligation_id}: unknown state contract {row.state_contract!r}")
    for row in registry.bindings:
        bound.add(row.obligation_id)
        if row.obligation_id not in known_obligations:
            findings.append(
                f"{row.binding_id}: binds unknown obligation {row.obligation_id!r} "
                "(the A/B join is exact; a dangling binding is corruption)"
            )
        if row.census_root not in known_roots:
            findings.append(f"{row.binding_id}: unknown census root {row.census_root!r}")
        if row.purity_mechanism not in PURITY_MECHANISMS:
            findings.append(f"{row.binding_id}: unknown purity mechanism")
        if row.scale_sensitive not in ("yes", "no"):
            findings.append(f"{row.binding_id}: scale_sensitive must be yes/no")
        if row.scale_sensitive == "yes" and not row.scale_profile_id:
            findings.append(f"{row.binding_id}: scale-sensitive without a named profile (D23)")
        if row.disclosure_kind not in DISCLOSURE_KINDS:
            findings.append(f"{row.binding_id}: unknown disclosure kind")
        for cell in filter(None, row.history_cells.split(",")):
            if cell not in HISTORY_CELLS:
                findings.append(f"{row.binding_id}: unknown history cell {cell!r}")
    for obligation_id in sorted(known_obligations - bound):
        findings.append(f"{obligation_id}: obligation has NO binding (defined but doorless)")
    return tuple(findings)


def _census_roots():
    """Return the nine census-root descriptors (import-cycle-free)."""

    from ._censuses import CENSUS_ROOTS

    return CENSUS_ROOTS
