"""Per-boundary parse rules for ``collective_boundary_v1`` rank evidence.

``_evidence._validate_boundary`` runs these rule groups in a fixed order over
one journal entry; each group raises the typed ``merged_schema_invalid``
refusal for the first defect it finds. The order is part of the contract:
when an entry carries several defects, the earliest group's refusal is the
one reported (pinned by ``tests/test_merged_boundary_rules.py``). The closed
vocabularies the rules read live here too, so the rules have no import back
into ``_evidence``; that module re-exports them.
"""

from __future__ import annotations

import re
from typing import Any, TypeGuard

from ..distributed._ledger import membership_digest_for_ranks
from ._enums import REDUCE_OP_KINDS, TENSORLESS_KINDS, MergedErrorCode
from ._errors import MergeInputError

# Shared rule parameters, documented once: ``entry`` is one boundary journal
# entry (a mapping), ``kind`` its validated kind, ``where`` the diagnostic
# prefix of each message (``"boundary <i> of <source>"``) and ``source`` the
# merge input origin carried on the refusal payload. Every rule raises
# ``MergeInputError`` (``merged_schema_invalid``) for the first defect found.

__all__ = (
    "BOUNDARY_SCHEMA",
    "KNOWN_KINDS",
    "P2P_KINDS",
    "check_boundary_identity",
    "check_completion_events",
    "check_correlation_key",
    "check_disclosures",
    "check_group_coherence",
    "check_group_membership",
    "check_kind_class_records",
    "check_lifetime_evidence",
    "check_op_label_backrefs",
    "check_roles",
    "check_scalar_records",
    "check_witness_digests",
    "check_witness_policy",
)

BOUNDARY_SCHEMA = "collective_boundary_v1"

KNOWN_KINDS = frozenset(
    {
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "broadcast",
        "reduce",
        "all_to_all",
        "all_to_all_single",
        "gather",
        "scatter",
        "send",
        "recv",
        "barrier",
        "all_gather_object",
        "broadcast_object_list",
        "gather_object",
        "scatter_object_list",
    }
)
"""The closed C0 boundary-kind vocabulary; an unknown kind refuses typed."""

P2P_KINDS = frozenset({"send", "recv"})
"""Point-to-point kinds: out of C1 scope (pipeline pairing is rung C3)."""

_COMPLETION_BINDINGS = frozenset({"issue_sync", "unobserved"})
_WITNESS_POLICIES = frozenset({"none", "digest"})
_INSTALL_EPOCHS = frozenset({"armed_before_any_group", "seeded"})
_ROLE_NAMES = frozenset({"contribution", "destination", "contribution_destination"})
_NOT_PRESENT_REASONS = frozenset({"async_completion_unobserved"})
"""Closed vocabulary for ``witness.not_present_reason`` (null is the other value)."""
_DISCLOSURE_TOKENS = frozenset({"read_of_inflight_destination", "c10d_group_seq_read_failed"})
"""Closed vocabulary of boundary disclosure tokens the recorder can emit."""
_VALUE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
"""Byte-exact witness digests are SHA-256 hex, same shape as membership digests."""


def _refuse(
    detail: str,
    *,
    code: MergedErrorCode = MergedErrorCode.MERGED_SCHEMA_INVALID,
    **payload: Any,
) -> MergeInputError:
    """Build the typed parse refusal for malformed rank evidence.

    Carries a ``fields["remedy"]`` like every other merged refusal (R65-12:
    this shared constructor was the one remedy-less family in the package).
    ``code`` defaults to the schema refusal; a raise site may spell it
    explicitly so the S-17 census reads the site as coded.
    """

    return MergeInputError(
        f"Rank-core evidence is not a valid collective_boundary_v1 journal: {detail}",
        code=code,
        remedy=(
            "treat the rank core as tampered or corrupt; re-capture the rank "
            "under the distributed opt-in rather than merging this evidence"
        ),
        **payload,
    )


def _is_strict_int(value: Any) -> TypeGuard[int]:
    """Return whether ``value`` is an ``int`` that is not a ``bool``."""

    return isinstance(value, int) and not isinstance(value, bool)


def check_boundary_identity(entry: dict[str, Any], where: str, source: str) -> Any:
    """Check the entry's schema tag and boundary kind.

    Returns
    -------
    Any
        The entry's kind, a member of :data:`KNOWN_KINDS`.

    Raises
    ------
    MergeInputError
        On a foreign schema tag or a kind outside the closed vocabulary.
    """

    if entry.get("schema") != BOUNDARY_SCHEMA:
        raise _refuse(f"{where} has schema {entry.get('schema')!r}", source=source)
    kind = entry.get("kind")
    if kind not in KNOWN_KINDS:
        raise _refuse(f"{where} has unknown kind {kind!r}", source=source)
    return kind


def check_correlation_key(entry: dict[str, Any], where: str, source: str) -> dict[str, Any]:
    """Check the rank-local correlation key's exact field set and field types.

    Returns
    -------
    dict[str, Any]
        The correlation mapping: string digest and channel, non-negative
        integer lifetime ordinal and seq.

    Raises
    ------
    MergeInputError
        On a missing or extra key, or a field of the wrong type or sign.
    """

    correlation = entry.get("correlation")
    if not isinstance(correlation, dict) or set(correlation) != {
        "membership_digest",
        "lifetime_ordinal",
        "channel",
        "seq",
    }:
        raise _refuse(f"{where} has a malformed correlation key", source=source)
    if not isinstance(correlation["membership_digest"], str):
        raise _refuse(f"{where} membership_digest is not a string", source=source)
    if not isinstance(correlation["channel"], str):
        raise _refuse(f"{where} correlation channel is not a string", source=source)
    if not _is_strict_int(correlation["lifetime_ordinal"]) or correlation["lifetime_ordinal"] < 0:
        raise _refuse(f"{where} lifetime_ordinal is not a non-negative integer", source=source)
    if not _is_strict_int(correlation["seq"]) or correlation["seq"] < 0:
        raise _refuse(f"{where} seq is not a non-negative integer", source=source)
    return correlation


def check_group_membership(entry: dict[str, Any], where: str, source: str) -> list[Any]:
    """Check the group record's rank list and this rank's place in it.

    Returns
    -------
    list[Any]
        The recorded ``global_ranks``: distinct integers containing
        ``my_global_rank``.

    Raises
    ------
    MergeInputError
        On a missing group record or rank list, a missing ``my_global_rank``,
        a non-integer or duplicate member, or a rank outside its own group.
    """

    group = entry.get("group")
    if not isinstance(group, dict) or not isinstance(group.get("global_ranks"), list):
        raise _refuse(f"{where} has no group membership record", source=source)
    if not isinstance(group.get("my_global_rank"), int):
        raise _refuse(f"{where} has no my_global_rank", source=source)
    global_ranks: list[Any] = group["global_ranks"]
    if any(not _is_strict_int(rank) for rank in global_ranks):
        raise _refuse(f"{where} group membership contains a non-integer rank", source=source)
    if len(set(global_ranks)) != len(global_ranks):
        raise _refuse(f"{where} group membership contains duplicate ranks", source=source)
    if group["my_global_rank"] not in global_ranks:
        raise _refuse(
            f"{where} claims rank {group['my_global_rank']} outside its group membership",
            source=source,
        )
    return global_ranks


def check_group_coherence(
    group: dict[str, Any],
    global_ranks: list[Any],
    correlation: dict[str, Any],
    where: str,
    source: str,
) -> None:
    """Check that the group record's redundant fields agree with its rank list.

    ``my_group_rank``, ``size`` and the correlation key's membership digest
    are each derivable from ``global_ranks``; a value that disagrees is a
    second, contradictory membership claim.

    Parameters
    ----------
    group:
        The entry's group record, already shape-checked by
        :func:`check_group_membership`.
    global_ranks:
        The recorded member ranks.
    correlation:
        The entry's correlation key, already shape-checked.

    Raises
    ------
    MergeInputError
        On a non-integer or mispositioned ``my_group_rank``, a ``size`` other
        than the member count, a non-string backend, or a membership digest
        bound to a different rank list.
    """

    my_group_rank = group.get("my_group_rank")
    if my_group_rank is not None and not _is_strict_int(my_group_rank):
        raise _refuse(f"{where} my_group_rank is not an integer or null", source=source)
    # The group rank is definitionally the position of my_global_rank in the
    # recorded (c10d-ordered) rank list -- the writer reads it from
    # ``dist.get_group_rank`` over the same group whose
    # ``get_process_group_ranks`` list it records. A permuted or out-of-range
    # value rebinds this rank's slice pairings in the gather/scatter/all_to_all
    # witness derivation (and used to reach the engine's list indexing as a raw
    # IndexError), so incoherence is tamper, never honest evidence (R18-2).
    if my_group_rank is not None and my_group_rank != global_ranks.index(group["my_global_rank"]):
        raise _refuse(
            f"{where} my_group_rank {my_group_rank} does not equal the position "
            f"of my_global_rank {group['my_global_rank']} in its own recorded "
            f"group membership {global_ranks}",
            source=source,
        )
    # ``size`` is written by the recorder as len(global_ranks) and was never
    # validated: a forged size is a second, contradictory membership claim.
    size = group.get("size")
    if not _is_strict_int(size) or size != len(global_ranks):
        raise _refuse(
            f"{where} group size {size!r} does not equal its recorded "
            f"membership of {len(global_ranks)} rank(s)",
            source=source,
        )
    backend = group.get("backend")
    if backend is not None and not isinstance(backend, str):
        raise _refuse(f"{where} group backend is not a string or null", source=source)
    _check_membership_digest_binding(correlation, global_ranks, where, source)


def _check_membership_digest_binding(
    correlation: dict[str, Any], global_ranks: list[Any], where: str, source: str
) -> None:
    """Check that the correlation key's membership digest names this rank list.

    Parameters
    ----------
    correlation:
        The entry's shape-checked correlation key.
    global_ranks:
        The recorded member ranks.

    Raises
    ------
    MergeInputError
        When the digest is not ``membership_digest_for_ranks(global_ranks)``.
    """

    # The membership digest is definitionally sha256(sorted(global_ranks)) and
    # freely recomputable. A digest bound to a DIFFERENT membership rebinds this
    # boundary's correlation joins, lifetime ordinals, and pre-join audit row to
    # another communicator while the presence/relation checks keep reading the
    # rank list -- the two views are attacker-separable unless tied here.
    if correlation["membership_digest"] != membership_digest_for_ranks(global_ranks):
        raise _refuse(
            f"{where} membership_digest does not equal the digest of its own "
            f"recorded group membership {sorted(int(r) for r in global_ranks)}",
            source=source,
        )


def check_roles(entry: dict[str, Any], kind: Any, where: str, source: str) -> None:
    """Check the tensor-role list: cardinality against the kind, then each role.

    Raises
    ------
    MergeInputError
        On a missing roles list, an empty list on a tensor-carrying kind, or
        the first malformed role entry (see :func:`_check_role_entry`).
    """

    roles = entry.get("roles")
    if not isinstance(roles, list):
        raise _refuse(f"{where} has no roles list", source=source)
    # Role cardinality vs boundary kind (b6-opus-R18-1): deleting the roles
    # record from EVERY member of a join used to vacuously satisfy the
    # set-of-shapes agreement checks (asymmetric deletion was caught; uniform
    # corruption -- the merge threat model -- was the escape). Every
    # tensor-carrying kind records at least one role on a successful call, so
    # an empty record refuses here, at the one chokepoint merge time and load
    # rederivation share.
    if not roles and kind not in TENSORLESS_KINDS:
        raise _refuse(
            f"{where} is a tensor-carrying {kind} boundary with zero tensor "
            "roles; a successful collective of this kind always records at "
            "least one role, so an empty or deleted roles record is not "
            "honest evidence",
            source=source,
        )
    seen_role_indexes: dict[str, set[int]] = {}
    for role_index, role in enumerate(roles):
        _check_role_entry(role, role_index, seen_role_indexes, where, source)


def _check_role_entry(
    role: Any,
    role_index: int,
    seen_role_indexes: dict[str, set[int]],
    where: str,
    source: str,
) -> None:
    """Check one role entry's name, shape and per-role-unique tensor position.

    Parameters
    ----------
    role:
        One element of the entry's roles list.
    role_index:
        Its position in that list (diagnostic).
    seen_role_indexes:
        Tensor positions already seen per exact role name; updated in place.

    Raises
    ------
    MergeInputError
        On a non-mapping role, a role name outside the closed vocabulary, a
        shape that is not a list of non-negative integers, a missing or
        negative index, or an index repeated within one role name.
    """

    if not isinstance(role, dict):
        raise _refuse(f"{where} role entry {role_index} is not a mapping", source=source)
    if role.get("role") not in _ROLE_NAMES:
        raise _refuse(
            f"{where} role entry {role_index} has role {role.get('role')!r} "
            "outside the closed vocabulary",
            source=source,
        )
    shape = role.get("shape")
    if not isinstance(shape, list) or any(not _is_strict_int(dim) or dim < 0 for dim in shape):
        raise _refuse(
            f"{where} role entry {role_index} has no well-formed shape "
            "(a list of non-negative integers)",
            source=source,
        )
    # ``index`` is the recorder's tensor POSITION (input position for
    # contribution/contribution_destination, output position for
    # destination). Positions are unique per exact role name; they are NOT
    # required dense 0..n-1 because a tensor serving as both contribution
    # and destination keeps its input position and leaves a hole in the
    # destination positions -- an honest recorder shape (R18-1 ii).
    index_value = role.get("index")
    if not _is_strict_int(index_value) or index_value < 0:
        raise _refuse(
            f"{where} role entry {role_index} has no non-negative integer index",
            source=source,
        )
    used = seen_role_indexes.setdefault(str(role.get("role")), set())
    if index_value in used:
        raise _refuse(
            f"{where} role entry {role_index} duplicates index {index_value} "
            f"within role {role.get('role')!r}; the recorder emits each "
            "tensor position at most once per role",
            source=source,
        )
    used.add(index_value)


def check_scalar_records(entry: dict[str, Any], kind: Any, where: str, source: str) -> Any:
    """Check the reduce-op record its kind requires and the group-sequence probe.

    Returns
    -------
    Any
        The recorded group-sequence probe value: an integer or ``None``.

    Raises
    ------
    MergeInputError
        On a reduce kind without a non-empty ``reduce_op`` string, a deleted
        group-sequence key, or a non-integer group-sequence value.
    """

    # Reduce-op cardinality vs kind (sibling of the roles-deletion escape):
    # uniform deletion of ``reduce_op`` from every rank core vacuously
    # satisfied the reduce-op agreement check the same way.
    if kind in REDUCE_OP_KINDS:
        reduce_op = entry.get("reduce_op")
        if not isinstance(reduce_op, str) or not reduce_op:
            raise _refuse(
                f"{where} is a {kind} boundary without its reduce_op record; "
                "a successful call of this kind always records one",
                source=source,
            )
    # A tampered non-integer seq crashed the engine's delta arithmetic with a
    # raw TypeError instead of the promised typed refusal (the load-side
    # descriptor check in _artifact already enforced this; parse now matches).
    # The KEY itself is required: the recorder always writes it (null when the
    # probe is absent), so a deleted key is tamper, not honest absence (R18-3).
    if "c10d_group_seq" not in entry:
        raise _refuse(f"{where} lacks its c10d_group_seq record", source=source)
    c10d_group_seq = entry.get("c10d_group_seq")
    if c10d_group_seq is not None and not _is_strict_int(c10d_group_seq):
        raise _refuse(f"{where} c10d_group_seq is not an integer or null", source=source)
    return c10d_group_seq


def check_completion_events(entry: dict[str, Any], where: str, source: str) -> Any:
    """Check the events record: completion binding and its async flag agree.

    Returns
    -------
    Any
        The completion binding, a member of the closed binding vocabulary.

    Raises
    ------
    MergeInputError
        On a malformed events record, a non-boolean ``async_op``, or an
        ``async_op`` that contradicts the completion binding.
    """

    events = entry.get("events")
    if not isinstance(events, dict) or events.get("completion_binding") not in _COMPLETION_BINDINGS:
        raise _refuse(f"{where} has a malformed events record", source=source)
    completion_binding = events["completion_binding"]
    async_op = events.get("async_op")
    if not isinstance(async_op, bool):
        raise _refuse(f"{where} events async_op is not a boolean", source=source)
    if async_op != (completion_binding == "unobserved"):
        raise _refuse(
            f"{where} events record is incoherent: async_op={async_op} with "
            f"completion_binding={completion_binding!r} (an unobserved completion "
            "is exactly the async case)",
            source=source,
        )
    return completion_binding


def check_disclosures(
    entry: dict[str, Any],
    completion_binding: Any,
    group_seq: Any,
    where: str,
    source: str,
) -> None:
    """Check disclosure tokens against the vocabulary and the records they disclose.

    Parameters
    ----------
    completion_binding:
        The entry's validated completion binding.
    group_seq:
        The entry's validated group-sequence probe value (integer or ``None``).

    Raises
    ------
    MergeInputError
        On a non-list or non-string token list, an unknown token, an
        in-flight-read disclosure that disagrees with the completion binding,
        or a failed-seq-read disclosure alongside a seq value.
    """

    disclosures = entry.get("disclosures")
    if not isinstance(disclosures, list) or any(
        not isinstance(token, str) for token in disclosures
    ):
        raise _refuse(f"{where} disclosures is not a list of strings", source=source)
    unknown_tokens = set(disclosures) - _DISCLOSURE_TOKENS
    if unknown_tokens:
        raise _refuse(
            f"{where} disclosures contain tokens {sorted(unknown_tokens)} outside "
            "the closed vocabulary",
            source=source,
        )
    inflight_read = "read_of_inflight_destination" in disclosures
    if (completion_binding == "unobserved") != inflight_read:
        raise _refuse(
            f"{where} disclosure record is incoherent: completion_binding "
            f"{completion_binding!r} with read_of_inflight_destination "
            f"{'present' if inflight_read else 'absent'} "
            "(every unobserved completion records the in-flight-destination read, "
            "and no observed completion does)",
            source=source,
        )
    if "c10d_group_seq_read_failed" in disclosures and group_seq is not None:
        raise _refuse(
            f"{where} discloses a failed c10d_group_seq read yet carries a c10d_group_seq value",
            source=source,
        )


def check_kind_class_records(entry: dict[str, Any], kind: Any, where: str, source: str) -> None:
    """Check the records whose presence follows from the kind's class.

    Tensorless kinds are journal-only (no op node); point-to-point kinds carry
    a peer record and collective kinds never do.

    Raises
    ------
    MergeInputError
        On an ``op_node`` flag that is not the kind's tensor-carrying class,
        a p2p kind without a peer mapping, or a collective with a peer record.
    """

    tensorless = kind in TENSORLESS_KINDS
    op_node = entry.get("op_node")
    if not isinstance(op_node, bool) or op_node != (not tensorless):
        raise _refuse(
            f"{where} op_node does not match its kind's tensorless class "
            f"({kind!r} boundaries are journal-{'only' if tensorless else 'plus-op'})",
            source=source,
        )
    peer = entry.get("peer")
    if kind in P2P_KINDS:
        if not isinstance(peer, dict):
            raise _refuse(f"{where} is a p2p boundary without its peer record", source=source)
    elif peer is not None:
        raise _refuse(f"{where} is a collective boundary carrying a peer record", source=source)


def check_witness_policy(
    entry: dict[str, Any], completion_binding: Any, where: str, source: str
) -> dict[str, Any]:
    """Check the witness policy and that its not-present reason fits the completion.

    Parameters
    ----------
    completion_binding:
        The entry's validated completion binding.

    Returns
    -------
    dict[str, Any]
        The witness record, with ``policy_resolved`` in the closed vocabulary.

    Raises
    ------
    MergeInputError
        On a missing record or unknown policy, an unknown not-present reason,
        or a reason other than the one the policy and completion imply.
    """

    witness = entry.get("witness")
    if not isinstance(witness, dict) or witness.get("policy_resolved") not in _WITNESS_POLICIES:
        raise _refuse(
            f"{where} has witness policy {witness.get('policy_resolved') if isinstance(witness, dict) else witness!r} "
            "outside the closed vocabulary",
            source=source,
        )
    policy_resolved = witness["policy_resolved"]
    not_present_reason = witness.get("not_present_reason")
    if not_present_reason is not None and not_present_reason not in _NOT_PRESENT_REASONS:
        raise _refuse(
            f"{where} witness not_present_reason {not_present_reason!r} outside "
            "the closed vocabulary",
            source=source,
        )
    expected_reason = (
        "async_completion_unobserved"
        if policy_resolved == "digest" and completion_binding == "unobserved"
        else None
    )
    if not_present_reason != expected_reason:
        raise _refuse(
            f"{where} witness not_present_reason {not_present_reason!r} is "
            f"incoherent with policy {policy_resolved!r} and completion_binding "
            f"{completion_binding!r} (expected {expected_reason!r})",
            source=source,
        )
    return witness


def check_witness_digests(
    witness: dict[str, Any],
    completion_binding: Any,
    kind: Any,
    where: str,
    source: str,
) -> None:
    """Check each witness digest list against the policy, completion and kind.

    Digest fields must be null or a LIST of lowercase hex digests. A bare
    string here used to char-split through ``tuple(...)`` in the engine and
    compare single characters as digests -- two cores carrying the same
    garbage string rendered a fabricated ATTESTED verdict. Tensorless kinds
    legitimately record EMPTY digest lists under witness policy "digest"
    (zero tensors to digest); for every tensor-carrying kind an empty list
    is tamper, same as the roles rule.

    Parameters
    ----------
    witness:
        The entry's witness record, policy already validated.
    completion_binding:
        The entry's validated completion binding.

    Raises
    ------
    MergeInputError
        On digests under policy ``"none"``, destination digests under an
        unobserved completion, a non-list or wrongly empty list, or a value
        that is not a lowercase hex digest; contribution digests are
        checked before destination digests.
    """

    for digest_field in ("contribution_digests", "destination_digests"):
        digests = witness.get(digest_field)
        if digests is None:
            continue
        _check_digest_provenance(
            digest_field, witness["policy_resolved"], completion_binding, where, source
        )
        _check_digest_list(digest_field, digests, kind, where, source)


def _check_digest_provenance(
    digest_field: str, policy_resolved: Any, completion_binding: Any, where: str, source: str
) -> None:
    """Check that a present digest list could have been computed at capture.

    Parameters
    ----------
    digest_field:
        ``"contribution_digests"`` or ``"destination_digests"``.
    policy_resolved:
        The witness policy, already in the closed vocabulary.
    completion_binding:
        The entry's validated completion binding.

    Raises
    ------
    MergeInputError
        On any digests under policy ``"none"``, or destination digests under
        an unobserved completion.
    """

    if policy_resolved == "none":
        raise _refuse(
            f"{where} witness carries {digest_field} under policy_resolved "
            "'none'; a rank that never computed witnesses cannot present "
            "digests, so these are forged, not evidence",
            source=source,
        )
    if digest_field == "destination_digests" and completion_binding == "unobserved":
        raise _refuse(
            f"{where} witness carries destination_digests under an "
            "unobserved completion; the destination bytes were never "
            "observed at capture, so these digests are forged, not evidence",
            source=source,
        )


def _check_digest_list(digest_field: str, digests: Any, kind: Any, where: str, source: str) -> None:
    """Check that a present digest field is a list of lowercase hex digests.

    Parameters
    ----------
    digest_field:
        ``"contribution_digests"`` or ``"destination_digests"``.
    digests:
        The field's non-null value.

    Raises
    ------
    MergeInputError
        On a non-list, an empty list on a tensor-carrying kind, or a value
        that is not a lowercase hex digest.
    """

    if not isinstance(digests, list) or (not digests and kind not in TENSORLESS_KINDS):
        raise _refuse(
            f"{where} witness {digest_field} is not null or a non-empty list",
            source=source,
        )
    if any(not isinstance(item, str) or not _VALUE_DIGEST_RE.match(item) for item in digests):
        raise _refuse(
            f"{where} witness {digest_field} contains a value that is not a "
            "lowercase hex SHA-256 digest",
            source=source,
        )


def check_op_label_backrefs(entry: dict[str, Any], kind: Any, where: str, source: str) -> None:
    """Check the op-label back-references: types, cardinality vs kind, uniqueness.

    Raises
    ------
    MergeInputError
        On a missing or non-string label list, labels on a tensorless kind, no
        labels on an op-bearing kind, or a duplicated label.
    """

    op_labels_raw = entry.get("op_labels_raw")
    if not isinstance(op_labels_raw, list):
        raise _refuse(f"{where} lacks op_labels_raw back-references", source=source)
    if any(not isinstance(label, str) for label in op_labels_raw):
        raise _refuse(f"{where} op_labels_raw contains a non-string label", source=source)
    # Back-reference cardinality vs kind (R18-1 i): the recorder emits an op
    # node (>= 1 labeled output tensor) for EVERY op-bearing kind and none for
    # tensorless kinds, and each label names a DISTINCT logged output tensor.
    # An empty list on an op-bearing boundary, labels on a journal-only
    # boundary, or duplicate labels are all shapes no honest recorder writes;
    # they used to parse silently and derive top verdicts over lying
    # back-references. A FABRICATED but well-formed label list still parses
    # here (parse never dereferences the member trace); it is caught typed at
    # ``MergedTrace.join_ops()``, which fail-closed resolves every recorded
    # label through the core's own raw-to-final seam.
    if kind in TENSORLESS_KINDS:
        if op_labels_raw:
            raise _refuse(
                f"{where} is a journal-only {kind} boundary carrying op-label "
                "back-references; tensorless kinds never emit an op node",
                source=source,
            )
    elif not op_labels_raw:
        raise _refuse(
            f"{where} is an op-bearing {kind} boundary with zero op-label "
            "back-references; a successful call of this kind always logs at "
            "least one boundary output tensor",
            source=source,
        )
    if len(set(op_labels_raw)) != len(op_labels_raw):
        raise _refuse(
            f"{where} op_labels_raw contains duplicate labels; each "
            "back-reference names a distinct logged output tensor",
            source=source,
        )


def check_lifetime_evidence(entry: dict[str, Any], where: str, source: str) -> None:
    """Check the lifetime-evidence record's install epoch vocabulary.

    Raises
    ------
    MergeInputError
        On a missing record or an install epoch outside the closed vocabulary.
    """

    lifetime = entry.get("lifetime_evidence")
    if not isinstance(lifetime, dict) or lifetime.get("install_epoch") not in _INSTALL_EPOCHS:
        raise _refuse(
            f"{where} lifetime_evidence install_epoch "
            f"{lifetime.get('install_epoch') if isinstance(lifetime, dict) else lifetime!r} "
            "outside the closed vocabulary",
            source=source,
        )
