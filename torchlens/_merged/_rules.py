"""Shared identity, boundary and provenance rules over rank-core evidence.

Two sides of the merge read the same ``collective_boundary_v1`` journal. The
evidence validator (``_evidence`` running the ordered parse groups of
``_boundary_rules`` and ``_boundary_membership``) refuses a malformed core
typed; the derivation (``_engine``, ``_derive_steps``, ``_relation_table`` and
``_witness``) repeats the belts that keep directly constructed
``RankEvidence`` honest. Every predicate both sides apply is defined ONCE
here, so the evidence accepted and the merge derived can never drift apart.
The callers keep their own refusal messages and finding texts; this module
raises nothing and never imports either side.

Three families:

identity
    What names a boundary and its communicator: the rank-local correlation
    key and its group-uid and channel prefixes, the recorded members, the
    membership-digest binding, presence against membership, and a group
    rank's place in its membership.
boundary
    What a boundary of a given kind carries: the tensor-role vocabulary and
    selectors, and the witness digest lists.
provenance
    What a record could honestly have observed: install epochs, completion
    bindings, witness policies and not-present reasons, disclosure tokens,
    and the per-core latch shape of the backend group-sequence probe.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final, TypeGuard

from .._distributed._ledger import membership_digest_for_ranks
from ._enums import TENSORLESS_KINDS

__all__ = (
    "ALL_ROLES",
    "ARMED_BEFORE_ANY_GROUP",
    "COLLECTIVE_CHANNEL",
    "COMPLETION_BINDINGS",
    "CONTRIBUTION_ROLES",
    "CORRELATION_FIELDS",
    "DESTINATION_ROLES",
    "DISCLOSURE_TOKENS",
    "INFLIGHT_READ_TOKEN",
    "INSTALL_EPOCHS",
    "NOT_PRESENT_REASONS",
    "ROLE_NAMES",
    "SEQ_READ_FAILED_TOKEN",
    "UNOBSERVED_COMPLETION",
    "WITNESS_DIGEST_FIELDS",
    "WITNESS_POLICIES",
    "ChannelKey",
    "CorrelationKey",
    "GroupRankPresence",
    "GroupSeqLatch",
    "GroupUid",
    "armed_before_any_group",
    "channel_key",
    "correlation_key",
    "destination_unobserved",
    "empty_for_tensor_kind",
    "expected_not_present_reason",
    "group_rank_in_range",
    "group_uid",
    "is_digest_sequence",
    "is_strict_int",
    "membership_digest_bound",
    "ranks_outside_membership",
    "recorded_members",
    "role_shapes",
    "roles_of",
)

# ---------------------------------------------------------------------------
# Identity rules
# ---------------------------------------------------------------------------

CorrelationKey = tuple[str, int, str, int]
"""``(membership_digest, lifetime_ordinal, channel, seq)``: one rank-local issue."""

GroupUid = tuple[str, int]
"""``(membership_digest, lifetime_ordinal)``: one communicator lifetime."""

ChannelKey = tuple[str, int, str]
"""``(membership_digest, lifetime_ordinal, channel)``: one join channel."""

CORRELATION_FIELDS: Final = frozenset({"membership_digest", "lifetime_ordinal", "channel", "seq"})
"""The exact field set of a boundary's correlation record."""

COLLECTIVE_CHANNEL: Final = "coll"
"""The symmetric-collective correlation channel; every other channel is p2p scope."""


def is_strict_int(value: Any) -> TypeGuard[int]:
    """Return whether ``value`` is an ``int`` that is not a ``bool``."""

    return isinstance(value, int) and not isinstance(value, bool)


def correlation_key(correlation: Mapping[str, Any]) -> CorrelationKey:
    """Return a boundary's rank-local correlation key, normalised to ``str``/``int``.

    Parameters
    ----------
    correlation:
        The boundary's shape-checked correlation mapping.
    """

    return (
        str(correlation["membership_digest"]),
        int(correlation["lifetime_ordinal"]),
        str(correlation["channel"]),
        int(correlation["seq"]),
    )


def group_uid(correlation: Mapping[str, Any]) -> GroupUid:
    """Return the communicator-lifetime prefix of :func:`correlation_key`."""

    digest, ordinal, _channel, _seq = correlation_key(correlation)
    return digest, ordinal


def channel_key(correlation: Mapping[str, Any]) -> ChannelKey:
    """Return the join-channel prefix of :func:`correlation_key`."""

    digest, ordinal, channel, _seq = correlation_key(correlation)
    return digest, ordinal, channel


def recorded_members(entry: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the boundary's recorded group membership as integers, in c10d order."""

    return tuple(int(member) for member in entry["group"]["global_ranks"])


def membership_digest_bound(correlation: Mapping[str, Any], members: Iterable[Any]) -> bool:
    """Return whether the correlation's membership digest names ``members``.

    The digest is definitionally ``sha256(sorted(global_ranks))`` and freely
    recomputable; a digest bound to a different membership rebinds the
    boundary's joins, lifetime ordinals and audit row to another communicator.
    """

    return bool(correlation["membership_digest"] == membership_digest_for_ranks(members))


def ranks_outside_membership(ranks: Iterable[Any], members: Iterable[Any]) -> tuple[Any, ...]:
    """Return the ``ranks`` that are not in ``members``, in the order given.

    Presence must be a subset of the recorded membership: a rank presenting a
    boundary for a group it is not part of, at parse or at join time, is never
    honest evidence.
    """

    member_set = set(members)
    return tuple(rank for rank in ranks if rank not in member_set)


def group_rank_in_range(value: Any, group_size: int) -> bool:
    """Return whether ``value`` is a strict integer position inside a group of ``group_size``."""

    return is_strict_int(value) and 0 <= value < group_size


class GroupRankPresence:
    """Track whether ``my_group_rank`` is uniformly present or absent per group within one core.

    The writer mints ``None`` only when ``dist.get_group_rank`` raises, a
    group-level fact, so mixed presence within one core and group is tamper.
    """

    def __init__(self) -> None:
        self._absent: dict[GroupUid, bool] = {}

    def mixed(self, uid: GroupUid, is_absent: bool) -> bool:
        """Record one boundary's presence state; return whether it contradicts the group's."""

        seen = self._absent.setdefault(uid, is_absent)
        return seen != is_absent


# ---------------------------------------------------------------------------
# Boundary rules
# ---------------------------------------------------------------------------

ROLE_NAMES: Final = frozenset({"contribution", "destination", "contribution_destination"})
"""The closed vocabulary of tensor-role names."""

CONTRIBUTION_ROLES: Final = ("contribution", "contribution_destination")
"""The role names under which a rank contributes bytes to a collective."""

DESTINATION_ROLES: Final = ("destination", "contribution_destination")
"""The role names under which a rank receives bytes from a collective."""

ALL_ROLES: Final = ("contribution", "destination", "contribution_destination")
"""Every role name, in the relation table's order."""

WITNESS_DIGEST_FIELDS: Final = ("contribution_digests", "destination_digests")
"""The witness record's digest lists, in check order."""


def roles_of(entry: Mapping[str, Any], role_names: Iterable[str]) -> list[dict[str, Any]]:
    """Return the ordered role entries whose name is in ``role_names``."""

    wanted = set(role_names)
    return [role for role in entry.get("roles", ()) if role.get("role") in wanted]


def role_shapes(entry: Mapping[str, Any], role_names: Iterable[str]) -> tuple[tuple[int, ...], ...]:
    """Return the shapes of ``entry``'s roles matching ``role_names``, in order."""

    return tuple(tuple(role["shape"]) for role in roles_of(entry, role_names))


def empty_for_tensor_kind(kind: Any, items: Any) -> bool:
    """Return whether an empty record is tamper for ``kind``.

    Every tensor-carrying kind records at least one tensor role, and under
    witness policy ``"digest"`` at least one digest, on a successful call;
    only the tensorless kinds legitimately record empty lists.
    """

    return kind not in TENSORLESS_KINDS and not items


def is_digest_sequence(value: Any) -> TypeGuard[list[Any] | tuple[Any, ...]]:
    """Return whether a non-null digest field is a sequence of digests, not a bare string.

    A bare string would char-split through ``tuple(...)`` and its characters
    would then be compared as digests, so two cores carrying the same garbage
    string fabricated an attested verdict.
    """

    return isinstance(value, (list, tuple)) and not isinstance(value, str)


# ---------------------------------------------------------------------------
# Provenance rules
# ---------------------------------------------------------------------------

ARMED_BEFORE_ANY_GROUP: Final = "armed_before_any_group"
"""The install epoch of a rank armed before any process group existed."""

INSTALL_EPOCHS: Final = frozenset({ARMED_BEFORE_ANY_GROUP, "seeded"})
"""The closed vocabulary of arming install epochs."""

UNOBSERVED_COMPLETION: Final = "unobserved"
"""The completion binding of an async collective whose completion was never awaited."""

COMPLETION_BINDINGS: Final = frozenset({"issue_sync", UNOBSERVED_COMPLETION})
"""The closed vocabulary of completion bindings."""

WITNESS_POLICIES: Final = frozenset({"none", "digest"})
"""The closed vocabulary of resolved witness policies."""

NOT_PRESENT_REASONS: Final = frozenset({"async_completion_unobserved"})
"""Closed vocabulary for ``witness.not_present_reason`` (null is the other value)."""

INFLIGHT_READ_TOKEN: Final = "read_of_inflight_destination"
"""Disclosed on every unobserved completion: the destination was read in flight."""

SEQ_READ_FAILED_TOKEN: Final = "c10d_group_seq_read_failed"
"""Disclosed exactly once per core, where the group-sequence probe latched off."""

DISCLOSURE_TOKENS: Final = frozenset({INFLIGHT_READ_TOKEN, SEQ_READ_FAILED_TOKEN})
"""Closed vocabulary of boundary disclosure tokens the recorder can emit."""


def armed_before_any_group(install_epoch: Any) -> bool:
    """Return whether a rank's absolute issue counters are comparable across ranks.

    Ranks armed before any group tick their per-channel counters on every
    issue from group creation on, so the same collective carries the same
    absolute seq on every such rank; a seeded rank's base is rank-local.
    """

    return bool(install_epoch == ARMED_BEFORE_ANY_GROUP)


def destination_unobserved(completion_binding: Any) -> bool:
    """Return whether the destination bytes were never observed at capture.

    An unobserved completion is exactly the async case: the recorder sets
    ``async_op``, discloses the in-flight destination read, and can present
    no destination digests.
    """

    return bool(completion_binding == UNOBSERVED_COMPLETION)


def expected_not_present_reason(policy_resolved: Any, completion_binding: Any) -> str | None:
    """Return the one ``not_present_reason`` the policy and completion imply, or ``None``."""

    if policy_resolved == "digest" and destination_unobserved(completion_binding):
        return "async_completion_unobserved"
    return None


class GroupSeqLatch:
    """The recorder's process-global latch shape for ``c10d_group_seq`` within one core.

    A prefix of values, at most one disclosing read-failure boundary, then an
    all-null suffix. Selective nulling of a value (the cheap tamper that
    deleted a ``correlation_delta_mismatch`` finding) is a defect; uniformly
    absent probes stay finding-free (absence of the probe never demotes
    anything), and an all-null rewrite is the documented coherent-reauthoring
    boundary, not an open residual.
    """

    VALUE_AFTER_DROP = "value_after_drop"
    UNDISCLOSED_DROP = "undisclosed_drop"
    REPEATED_DISCLOSURE = "repeated_disclosure"

    def __init__(self) -> None:
        self._seen_value = False
        self._dropped = False

    def observe(self, value: Any, disclosed: bool) -> str | None:
        """Record one boundary's probe value; return the defect it proves, or ``None``.

        Parameters
        ----------
        value:
            The boundary's ``c10d_group_seq`` (an integer or ``None``).
        disclosed:
            Whether the boundary carries the failed-read disclosure.
        """

        if value is not None:
            if self._dropped:
                return self.VALUE_AFTER_DROP
            self._seen_value = True
            return None
        if not self._dropped:
            self._dropped = True
            if self._seen_value and not disclosed:
                return self.UNDISCLOSED_DROP
            return None
        if disclosed:
            return self.REPEATED_DISCLOSURE
        return None
