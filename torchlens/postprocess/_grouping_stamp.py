"""The grouping-policy stamp (``grouping_policy_v1``): writer, loader
validation (coherence rules C1-C8), and the canonical degraded settlement.

``trace.grouping_policy`` is a site-granular, load-validated record of HOW a
trace was grouped AND how its product rows were site-joined -- two distinct
axes: ``policy`` records only the step-7 grouping that actually ran;
``site_join`` records the product-layer join (always ``"none"`` from the
wave-0 writers; episode products are L2's).

S3 DISCIPLINE: the field is ``FieldPolicy.DROP`` under tlspec v7 and
prerelease-registered; portability gates run under the test-only switch --
never an active v7 write.

S2 STATUS: ALL vocabulary and refusal spellings here are PROVISIONAL
pending the S2 amendment (memo 4.3; the amendment also updates the
marker-combination table and the P1 totality test in the same change).
C8's capture-kind predicate is L2's spelling wired at that amendment; until
then any ``site_join != "none"`` is fail-closed incoherent (no wave-0
writer can legitimately produce one).

Adopt-or-degrade at load: parse failure or incoherence warns ONCE and
settles the stamp to the canonical DEGRADED SETTLEMENT -- a payload legal
under the exact writer schema, so verdicts only worsen across persistence
and the settled payload round-trips byte-stable (monotonic).
"""

from __future__ import annotations

import warnings
from typing import Any

from ..data_classes._site_key import SITE_KEY_PREFIX
from ..errors._base import TorchLensWarning

GROUPING_POLICY_SCHEMA = "grouping_policy_v1"

#: Closed step-7 policy vocabulary (S2-provisional).
POLICY_VALUES = frozenset({"structural", "strict_shapes", "fold_sites", "params_only", "unknown"})

#: Closed requested-knob vocabulary ("unknown" is the degraded value).
REQUESTED_VALUES = frozenset({"structural", "strict_shapes", "fold_sites", "unknown"})

#: Closed detector vocabulary (versioned algorithm names).
DETECTOR_VALUES = frozenset({"grouper_v1", "unknown"})

#: The exact writer key set: loads refuse any other shape (tamper check).
_STAMP_KEYS = frozenset(
    {
        "schema",
        "policy",
        "requested",
        "folded_sites",
        "site_join",
        "detector",
        "effective",
        "settlement_note",
    }
)

#: The `grouping=` knob values that are legal VOCABULARY at trace entry.
GROUPING_KNOB_VALUES = ("structural", "strict_shapes", "fold_sites")


def validate_grouping_knob(value: Any) -> None:
    """Validate the ``grouping=`` trace kwarg (closed vocabulary + entry
    legality). Only ``"structural"`` proceeds in wave 0.

    Raises
    ------
    InvalidArgumentError
        ``grouping_invalid`` for values outside the closed vocabulary;
        ``grouping_policy_unavailable`` for legal vocabulary that is not
        entry-legal yet: ``"strict_shapes"`` is refused until its own
        reviewed design + churn census land, and ``"fold_sites"`` is
        refused on plain captures until an affirmative D1 ruling (episode
        products fold by default via the product-layer site join instead).
    """

    from .._errors import InvalidArgumentError

    if value not in GROUPING_KNOB_VALUES:
        raise InvalidArgumentError(
            f"grouping={value!r} is not a recognized grouping policy.",
            code="grouping_invalid",
            remedy=f"choose one of {', '.join(GROUPING_KNOB_VALUES)}",
        )
    if value == "strict_shapes":
        raise InvalidArgumentError(
            "grouping='strict_shapes' is reserved vocabulary: its semantics "
            "are deliberately undesigned this wave (no silent half-design).",
            code="grouping_policy_unavailable",
            remedy="use the default grouping='structural'",
        )
    if value == "fold_sites":
        raise InvalidArgumentError(
            "grouping='fold_sites' is not entry-legal on plain captures "
            "until an affirmative D1 ruling activates within-capture "
            "folding (the flip PR).",
            code="grouping_policy_unavailable",
            remedy="use the default grouping='structural'",
        )


def build_grouping_policy_stamp(
    *,
    ran_recurrence_grouping: bool,
    requested: str,
) -> dict[str, Any]:
    """Return the healthy wave-0 stamp for one live capture.

    ``policy`` records what step 7 actually ran: ``"structural"`` when the
    neutral grouper ran, ``"params_only"`` (the vocabulary spelling of the
    ``recurrence_detection=False`` path) otherwise. Every live writer
    writes ``settlement_note=None`` so the loader's writer-derived exact
    key set is ONE set for healthy and degraded stamps alike.
    """

    return {
        "schema": GROUPING_POLICY_SCHEMA,
        "policy": "structural" if ran_recurrence_grouping else "params_only",
        "requested": requested,
        "folded_sites": "none",
        "site_join": "none",
        "detector": "grouper_v1" if ran_recurrence_grouping else "unknown",
        "effective": True,
        "settlement_note": None,
    }


def degraded_grouping_policy_stamp(reason: str) -> dict[str, Any]:
    """Return THE canonical degraded settlement (one representation).

    Legal under the exact writer schema; round-trips byte-stable and stays
    degraded (verdicts only worsen across persistence).
    """

    return {
        "schema": GROUPING_POLICY_SCHEMA,
        "policy": "unknown",
        "requested": "unknown",
        "folded_sites": "none",
        "site_join": "none",
        "detector": "unknown",
        "effective": False,
        "settlement_note": f"grouping_stamp_{reason}",
    }


def _site_list_valid(value: Any) -> bool:
    """Validate a folded_sites/site_join entry list (C4): parseable ``s1|``
    site keys, sorted, unique."""

    if not isinstance(value, list) or not value:
        return False
    previous: str | None = None
    for entry in value:
        if not isinstance(entry, str) or not entry.startswith(SITE_KEY_PREFIX + "|"):
            return False
        if previous is not None and not (previous < entry):
            return False
        previous = entry
    return True


def _stamp_shape_violation(payload: Any) -> str | None:
    """Shape/schema layer of stamp validation: malformed / unknown_key."""

    if not isinstance(payload, dict):
        return "malformed"
    if set(payload.keys()) != _STAMP_KEYS:
        return "unknown_key" if set(payload.keys()) - _STAMP_KEYS else "malformed"
    if payload["schema"] != GROUPING_POLICY_SCHEMA:
        return "malformed"
    return None


def _stamp_vocabulary_violation(payload: dict[str, Any]) -> str | None:
    """Closed-vocabulary layer of stamp validation."""

    policy = payload["policy"]
    requested = payload["requested"]
    detector = payload["detector"]
    effective = payload["effective"]
    settlement_note = payload["settlement_note"]
    if policy not in POLICY_VALUES or requested not in REQUESTED_VALUES:
        return "vocabulary"
    if detector not in DETECTOR_VALUES or not isinstance(effective, bool):
        return "vocabulary"
    if not (settlement_note is None or isinstance(settlement_note, str)):
        return "vocabulary"
    if isinstance(settlement_note, str) and not settlement_note.startswith("grouping_stamp_"):
        return "vocabulary"
    return None


def validate_grouping_policy_stamp(
    payload: Any,
    *,
    recurrence_detection: Any = None,
    grouping: Any = None,
) -> str | None:
    """Validate one loaded stamp payload; return the violated rule's NAME.

    ``None`` means coherent (adopt verbatim). Rule names: ``malformed``
    (shape/schema), ``unknown_key`` (exact-key tamper check),
    ``vocabulary`` (closed-vocab violation), and the cross-field coherence
    rules ``C1``-``C8`` (user-facing on violation).
    """

    shape_violation = _stamp_shape_violation(payload)
    if shape_violation is not None:
        return shape_violation
    policy = payload["policy"]
    requested = payload["requested"]
    effective = payload["effective"]
    settlement_note = payload["settlement_note"]
    vocabulary_violation = _stamp_vocabulary_violation(payload)
    if vocabulary_violation is not None:
        return vocabulary_violation
    folded_sites = payload["folded_sites"]
    site_join = payload["site_join"]
    for axis_value in (folded_sites, site_join):
        if axis_value in ("none", "all"):
            continue
        if not _site_list_valid(axis_value):
            return "C4"
    coherence_rules: tuple[tuple[str, bool], ...] = (
        # C1/C2: fold state must match the policy that claims it.
        ("C1", policy == "fold_sites" and folded_sites == "none"),
        ("C2", policy != "fold_sites" and folded_sites != "none"),
        # C3: a degraded/declined grouping never claims a policy it did not run.
        ("C3", effective is False and policy != "unknown"),
        # C5: stamp/recurrence_detection coherence (params_only <-> False).
        (
            "C5",
            recurrence_detection is not None
            and policy in ("structural", "params_only")
            and (policy == "params_only") != (recurrence_detection is False),
        ),
        # C6: mirror coherence with the requested-knob field.
        ("C6", grouping is not None and requested != "unknown" and requested != grouping),
        # C7: a healthy stamp never carries a settlement; a settled stamp never
        # claims health.
        ("C7", settlement_note is not None and not (policy == "unknown" and effective is False)),
        # C8: a product-layer join is only legal on episode products. The exact
        # capture_kind predicate is L2's spelling (wired at the S2 amendment);
        # until then no artifact can legitimately carry one -- fail closed.
        ("C8", site_join != "none"),
    )
    for rule_name, violated in coherence_rules:
        if violated:
            return rule_name
    return None


def settle_loaded_grouping_policy(state: dict[str, Any]) -> dict[str, Any]:
    """Resolve the stamp for one loaded trace state (adopt-or-degrade).

    Absent stamp (every pre-stamp v7 artifact) settles SILENTLY to the
    canonical legacy settlement; a present-but-invalid stamp warns once and
    settles with the violated rule's name in the settlement token.
    """

    payload = state.get("grouping_policy")
    if payload is None:
        return degraded_grouping_policy_stamp("legacy")
    violated = validate_grouping_policy_stamp(
        payload,
        recurrence_detection=state.get("recurrence_detection"),
        grouping=state.get("grouping"),
    )
    if violated is None:
        return dict(payload)
    warnings.warn(
        f"Loaded grouping_policy stamp is invalid (rule {violated}): settling "
        "to the degraded representation; stamp-consuming operations will "
        "refuse typed.",
        TorchLensWarning,
        stacklevel=2,
    )
    return degraded_grouping_policy_stamp(violated.lower())
