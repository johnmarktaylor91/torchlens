"""Serialized HealthFacts + the three-state nonfinite verdict (C02; sumfam 8).

D9: capture-basis per-op verdicts normalize into an immutable,
serializable record (basis, pass-qualified identities, coverage counts,
capture revision) persisted at ``trace.annotations["health_facts"]`` --
two labs independently measured the DROP-on-save policy that made a
payload-stripped all-NaN artifact answer ``()`` to ``nonfinite_ops``.

D5: every health claim renders one of THREE states -- ``found`` /
``checked_and_clean`` / ``not_checked``; silence is not a state, and
``nonfinite_verdict`` is the ONLY thing a surface may branch on (the
no-truthiness lint in tests/test_factcore_import_lint.py bans builder
reads of ``nonfinite_ops`` truthiness).

D19: alias rows (input/output mirrors, buffer state rows) are EXCLUDED
from the identity partition of the health counts and disclosed as a
separately named alias view -- checked=6/nonfinite=4 on a 3-op defect was
the panel's central claim in miniature.

Normalization runs at first derivation on a live finished trace and via
the explicit :func:`normalize_health_facts` seam; the capture-finalization
call site belongs to the capture-core fence (A05->A06->F20) and is filed
as an owner amendment -- until it lands, a capture saved with NO health
read serves the saved-payload basis on load (never a false clean).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Annotations key carrying the persisted record (field_intent.tsv row C02).
HEALTH_FACTS_ANNOTATIONS_KEY = "health_facts"
#: HealthFacts payload schema version.
HEALTH_FACTS_SCHEMA_VERSION = 1
#: The three-state verdict vocabulary (D5). Closed; silence is not a state.
HEALTH_VERDICTS: tuple[str, ...] = ("found", "checked_and_clean", "not_checked")


@dataclass(frozen=True)
class HealthFacts:
    """Immutable normalized health observations for one capture.

    An observation store, never a judgment producer (D14): severities and
    follow-ups belong to audit's FindingSet. ``basis`` is ``capture``
    (per-op capture-time checks), ``saved_payloads`` (post-hoc scan over
    retained payloads), or ``persisted`` (loaded from the artifact record,
    with the ORIGINAL basis in ``source_basis``).
    """

    schema_version: int
    basis: str
    source_basis: str | None
    checked: int
    nonfinite_labels: tuple[str, ...]
    alias_nonfinite_labels: tuple[str, ...]
    unchecked: int
    unexamined: int
    unmapped: int
    capture_revision: str

    @property
    def nonfinite_count(self) -> int:
        """Nonfinite ops in the identity partition (alias rows excluded)."""

        return len(self.nonfinite_labels)

    @property
    def verdict(self) -> str:
        """The three-state verdict (D5) -- the ONE branchable health fact.

        ``found`` on any nonfinite evidence (alias-view hits count: a hit
        is a hit, wherever disclosed); ``checked_and_clean`` ONLY when
        every examinable op was checked and none hit; ``not_checked``
        otherwise -- partial clean coverage never upgrades to clean.
        """

        if self.nonfinite_labels or self.alias_nonfinite_labels:
            return "found"
        if self.checked > 0 and self.unexamined == 0 and self.unchecked == 0:
            return "checked_and_clean"
        return "not_checked"

    def to_payload(self) -> dict[str, Any]:
        """JSON-primitive payload for the annotations channel."""

        return {
            "schema_version": self.schema_version,
            "basis": self.basis,
            "checked": self.checked,
            "nonfinite_labels": list(self.nonfinite_labels),
            "alias_nonfinite_labels": list(self.alias_nonfinite_labels),
            "unchecked": self.unchecked,
            "unexamined": self.unexamined,
            "unmapped": self.unmapped,
            "capture_revision": self.capture_revision,
        }


def _not_checked(basis: str, revision: str, *, unexamined: int = 0) -> HealthFacts:
    """The NOT-CHECKED shape (an artifact without a basis, never clean)."""

    return HealthFacts(
        schema_version=HEALTH_FACTS_SCHEMA_VERSION,
        basis=basis,
        source_basis=None,
        checked=0,
        nonfinite_labels=(),
        alias_nonfinite_labels=(),
        unchecked=0,
        unexamined=unexamined,
        unmapped=0,
        capture_revision=revision,
    )


def _alias_labels(trace: Any) -> set[str]:
    """Pass-qualified labels of alias rows (D19 exclusion set)."""

    labels: set[str] = set()
    for op in getattr(trace, "layer_list", ()) or ():
        if (
            getattr(op, "is_input", False)
            or getattr(op, "is_output", False)
            or getattr(op, "is_buffer", False)
        ):
            label = getattr(op, "label", None)
            if label is not None:
                labels.add(str(label))
    return labels


def _parse_persisted(payload: Any, revision: str) -> HealthFacts:
    """Validate a persisted payload FAIL-CLOSED (never a false clean)."""

    if not isinstance(payload, dict):
        return _not_checked("persisted_invalid", revision)
    try:
        basis = str(payload["basis"])
        record = HealthFacts(
            schema_version=int(payload["schema_version"]),
            basis="persisted",
            source_basis=basis,
            checked=int(payload["checked"]),
            nonfinite_labels=tuple(str(label) for label in payload["nonfinite_labels"]),
            alias_nonfinite_labels=tuple(str(label) for label in payload["alias_nonfinite_labels"]),
            unchecked=int(payload["unchecked"]),
            unexamined=int(payload["unexamined"]),
            unmapped=int(payload["unmapped"]),
            capture_revision=str(payload["capture_revision"]),
        )
    except (KeyError, TypeError, ValueError):
        return _not_checked("persisted_invalid", revision)
    if record.checked < 0 or record.unexamined < 0 or record.unchecked < 0:
        return _not_checked("persisted_invalid", revision)
    return record


def _derive(trace: Any) -> HealthFacts:
    """Derive HealthFacts from the strongest basis in hand."""

    from ..data_classes._nonfinite import nonfinite_coverage, nonfinite_op_labels
    from ._factcore import capture_fingerprint

    revision = capture_fingerprint(trace)
    coverage = nonfinite_coverage(trace)
    labels = nonfinite_op_labels(trace)
    aliases = _alias_labels(trace)
    partition_labels = tuple(label for label in labels if label not in aliases)
    alias_hits = tuple(label for label in labels if label in aliases)
    return HealthFacts(
        schema_version=HEALTH_FACTS_SCHEMA_VERSION,
        basis=coverage.basis,
        source_basis=None,
        checked=coverage.checked,
        nonfinite_labels=partition_labels,
        alias_nonfinite_labels=alias_hits,
        unchecked=coverage.unchecked,
        unexamined=coverage.unexamined,
        unmapped=coverage.unmapped,
        capture_revision=revision,
    )


def health_facts(trace: Any, *, allow_scan: bool = True) -> HealthFacts:
    """Return the normalized HealthFacts for one finished trace.

    Order of authority: the persisted artifact record (validated
    fail-closed), then a live derivation over the strongest available
    basis -- which is also ATTACHED to the annotations channel so a later
    save carries it (D9).

    Parameters
    ----------
    trace:
        Finished trace-like object.
    allow_scan:
        ``True`` (the explicit door) may pay for a first saved-payload
        scan. ``False`` is the D4 render-surface contract: serve only the
        basis already in hand (capture-time record, prior scan memo, or
        the persisted artifact record) and otherwise return the honest
        NOT-CHECKED shape (basis ``"unscanned"``) -- a render never
        implicitly triggers a payload scan.
    """

    annotations = getattr(trace, "annotations", None)
    if isinstance(annotations, dict) and HEALTH_FACTS_ANNOTATIONS_KEY in annotations:
        from ._factcore import capture_fingerprint

        revision = capture_fingerprint(trace)
        persisted = _parse_persisted(annotations[HEALTH_FACTS_ANNOTATIONS_KEY], revision)
        # The persisted record is the authority while it matches THIS
        # capture revision (that is exactly what survives a payload strip);
        # a revision mismatch (mutated/forked graph) re-derives, and an
        # invalid payload falls through fail-closed -- never a false clean.
        if persisted.basis != "persisted_invalid" and persisted.capture_revision == revision:
            return persisted
    if not allow_scan:
        from ..data_classes._nonfinite import has_scan_evidence

        if not has_scan_evidence(trace):
            from ._factcore import capture_fingerprint

            unexamined = sum(
                1
                for op in getattr(trace, "layer_list", ()) or ()
                if getattr(op, "has_saved_activation", False)
            )
            return _not_checked("unscanned", capture_fingerprint(trace), unexamined=unexamined)
    return normalize_health_facts(trace)


def normalize_health_facts(trace: Any) -> HealthFacts:
    """Derive AND persist HealthFacts on the annotations channel (D9).

    The capture-finalization seam: idempotent, safe on any finished trace.
    Returns the derived record; attaches its payload so ``save`` carries
    it. Never raises through a render path -- an underivable basis lands
    the NOT-CHECKED shape.
    """

    try:
        record = _derive(trace)
    except Exception:  # noqa: BLE001 -- health must degrade, never break render
        from ._factcore import capture_fingerprint

        try:
            revision = capture_fingerprint(trace)
        except Exception:  # noqa: BLE001
            revision = "fc1-unavailable"
        record = _not_checked("underivable", revision)
    annotations = getattr(trace, "annotations", None)
    if isinstance(annotations, dict) and record.basis not in (
        "underivable",
        "persisted_invalid",
    ):
        annotations[HEALTH_FACTS_ANNOTATIONS_KEY] = record.to_payload()
    return record


def nonfinite_verdict(trace: Any) -> str:
    """The three-state health verdict (D5): the ONE branchable spelling.

    ``found`` / ``checked_and_clean`` / ``not_checked``. Builders branch on
    THIS, never on ``nonfinite_ops`` truthiness -- a payload-stripped
    all-NaN artifact returns an empty tuple there, and the false negative
    happens inside a user's ``if`` before any renderer runs.
    """

    return health_facts(trace).verdict
