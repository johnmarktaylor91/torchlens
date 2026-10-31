"""Rank-core evidence extraction for the merge engine.

One extraction path serves live traces, loaded traces, and rank-core bundle
paths, so merge time and load rederivation consume IDENTICAL evidence
(design-merge-ranks-c v5, 4.3: one derivation function called twice). The
extractor validates the ``collective_boundary_v1`` payloads against their
closed vocabularies at parse time; a malformed core refuses typed
(``merged_schema_invalid``) instead of degrading silently.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..distributed._ledger import GroupLifecycleLedger, membership_digest_for_ranks
from ._enums import REDUCE_OP_KINDS, TENSORLESS_KINDS, MergedErrorCode
from ._errors import MergeInputError

__all__ = [
    "BOUNDARY_SCHEMA",
    "KNOWN_KINDS",
    "P2P_KINDS",
    "REDUCE_OP_KINDS",
    "TENSORLESS_KINDS",
    "RankEvidence",
    "extract_rank_evidence",
    "resolve_rank_inputs",
]

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


@dataclass(frozen=True)
class RankEvidence:
    """Everything the merge engine reads from one rank core.

    Parameters
    ----------
    rank:
        The core's global rank, proven by its own boundary records.
    boundaries:
        The ordered boundary entries from the trace-level distributed
        journal (issue order; each carries the ``collective_boundary_v1``
        payload plus ``op_labels_raw`` back-references).
    ledger:
        The rank's group-lifecycle ledger, rebuilt from its portable payload.
    install_epoch:
        The rank's arming install-epoch record.
    source:
        Where the evidence came from (diagnostic): ``"live"``, ``"loaded"``,
        or the bundle path string.
    shard_local:
        Whether the member trace carries the shard-local capture marker
        (``distributed_scope == "rank_local_shard"``; L8 census plan 3.2c(2)).
        REQUIRED with no default -- every constructor, direct construction
        included, must consciously supply it, so the ``_guard_scope`` marker
        key can never be silently omitted. Populated here in
        :func:`extract_rank_evidence`, the single common ancestor of all
        ``derive_merge`` entry paths (public input resolution, save-time
        reverify, load rederivation).
    """

    rank: int
    boundaries: tuple[dict[str, Any], ...]
    ledger: GroupLifecycleLedger
    install_epoch: str
    source: str
    shard_local: bool


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


def _validate_boundary(entry: dict[str, Any], index: int, source: str) -> None:
    """Validate one journal entry against the closed C0 vocabularies."""

    where = f"boundary {index} of {source}"
    if entry.get("schema") != BOUNDARY_SCHEMA:
        raise _refuse(f"{where} has schema {entry.get('schema')!r}", source=source)
    kind = entry.get("kind")
    if kind not in KNOWN_KINDS:
        raise _refuse(f"{where} has unknown kind {kind!r}", source=source)
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
    if (
        not isinstance(correlation["lifetime_ordinal"], int)
        or isinstance(correlation["lifetime_ordinal"], bool)
        or correlation["lifetime_ordinal"] < 0
    ):
        raise _refuse(f"{where} lifetime_ordinal is not a non-negative integer", source=source)
    if (
        not isinstance(correlation["seq"], int)
        or isinstance(correlation["seq"], bool)
        or correlation["seq"] < 0
    ):
        raise _refuse(f"{where} seq is not a non-negative integer", source=source)
    group = entry.get("group")
    if not isinstance(group, dict) or not isinstance(group.get("global_ranks"), list):
        raise _refuse(f"{where} has no group membership record", source=source)
    if not isinstance(group.get("my_global_rank"), int):
        raise _refuse(f"{where} has no my_global_rank", source=source)
    global_ranks = group["global_ranks"]
    if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in global_ranks):
        raise _refuse(f"{where} group membership contains a non-integer rank", source=source)
    if len(set(global_ranks)) != len(global_ranks):
        raise _refuse(f"{where} group membership contains duplicate ranks", source=source)
    if group["my_global_rank"] not in global_ranks:
        raise _refuse(
            f"{where} claims rank {group['my_global_rank']} outside its group membership",
            source=source,
        )
    my_group_rank = group.get("my_group_rank")
    if my_group_rank is not None and (
        not isinstance(my_group_rank, int) or isinstance(my_group_rank, bool)
    ):
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
    if isinstance(size, bool) or not isinstance(size, int) or size != len(global_ranks):
        raise _refuse(
            f"{where} group size {size!r} does not equal its recorded "
            f"membership of {len(global_ranks)} rank(s)",
            source=source,
        )
    backend = group.get("backend")
    if backend is not None and not isinstance(backend, str):
        raise _refuse(f"{where} group backend is not a string or null", source=source)
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
        if not isinstance(role, dict):
            raise _refuse(f"{where} role entry {role_index} is not a mapping", source=source)
        if role.get("role") not in _ROLE_NAMES:
            raise _refuse(
                f"{where} role entry {role_index} has role {role.get('role')!r} "
                "outside the closed vocabulary",
                source=source,
            )
        shape = role.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
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
        if isinstance(index_value, bool) or not isinstance(index_value, int) or index_value < 0:
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
    if c10d_group_seq is not None and (
        isinstance(c10d_group_seq, bool) or not isinstance(c10d_group_seq, int)
    ):
        raise _refuse(f"{where} c10d_group_seq is not an integer or null", source=source)
    events = entry.get("events")
    if not isinstance(events, dict) or events.get("completion_binding") not in _COMPLETION_BINDINGS:
        raise _refuse(f"{where} has a malformed events record", source=source)
    # Event/disclosure/witness COHERENCE (R18 fixwave-6): every field below is
    # redundant with the others on an honest record, and each redundancy is a
    # forgery axis when left unchecked. A tampered core that flips ONE of them
    # (forged destination digests on an async boundary, digests under witness
    # policy "none", a stripped in-flight-read disclosure) used to sail
    # through parse and could only be caught -- or worse, ATTESTED -- by the
    # cross-rank derivation. The only forgery that survives these checks is a
    # fully coherent reauthoring of every field on every core, which is the
    # documented out-of-scope boundary (contract section 11 analog), not an
    # open residual.
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
    if (completion_binding == "unobserved") != ("read_of_inflight_destination" in disclosures):
        raise _refuse(
            f"{where} disclosure record is incoherent: completion_binding "
            f"{completion_binding!r} with read_of_inflight_destination "
            f"{'present' if 'read_of_inflight_destination' in disclosures else 'absent'} "
            "(every unobserved completion records the in-flight-destination read, "
            "and no observed completion does)",
            source=source,
        )
    if "c10d_group_seq_read_failed" in disclosures and c10d_group_seq is not None:
        raise _refuse(
            f"{where} discloses a failed c10d_group_seq read yet carries a c10d_group_seq value",
            source=source,
        )
    op_node = entry.get("op_node")
    if not isinstance(op_node, bool) or op_node != (kind not in TENSORLESS_KINDS):
        raise _refuse(
            f"{where} op_node does not match its kind's tensorless class "
            f"({kind!r} boundaries are journal-{'only' if kind in TENSORLESS_KINDS else 'plus-op'})",
            source=source,
        )
    peer = entry.get("peer")
    if kind in P2P_KINDS:
        if not isinstance(peer, dict):
            raise _refuse(f"{where} is a p2p boundary without its peer record", source=source)
    elif peer is not None:
        raise _refuse(f"{where} is a collective boundary carrying a peer record", source=source)
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
    # Digest fields must be null or a LIST of SHA-256 hex strings. A bare
    # string here used to char-split through ``tuple(...)`` in the engine and
    # compare single characters as digests -- two cores carrying the same
    # garbage string rendered a fabricated ATTESTED verdict. Tensorless kinds
    # legitimately record EMPTY digest lists under witness policy "digest"
    # (zero tensors to digest); for every tensor-carrying kind an empty list
    # is tamper, same as the roles rule above.
    for digest_field in ("contribution_digests", "destination_digests"):
        digests = witness.get(digest_field)
        if digests is None:
            continue
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
    lifetime = entry.get("lifetime_evidence")
    if not isinstance(lifetime, dict) or lifetime.get("install_epoch") not in _INSTALL_EPOCHS:
        raise _refuse(
            f"{where} lifetime_evidence install_epoch "
            f"{lifetime.get('install_epoch') if isinstance(lifetime, dict) else lifetime!r} "
            "outside the closed vocabulary",
            source=source,
        )


def extract_rank_evidence(trace: Any, source: str) -> RankEvidence:
    """Extract and validate one rank core's merge evidence.

    Parameters
    ----------
    trace:
        A finished (live or loaded) ``Trace`` captured under the distributed
        opt-in.
    source:
        Diagnostic origin string recorded on the evidence.

    Returns
    -------
    RankEvidence
        Validated evidence for the engine.

    Raises
    ------
    MergeInputError
        If the trace carries no distributed record, its journal is malformed,
        or its boundary records disagree about the rank's own identity.
    """

    record = getattr(trace, "annotations", {}).get("distributed")
    if not isinstance(record, dict) or not record.get("boundaries"):
        raise MergeInputError(
            f"Merge input {source} carries no distributed boundary evidence "
            "(trace.annotations['distributed'] is absent or empty). Only rank "
            "captures taken under the distributed opt-in "
            "(torchlens.distributed.arm() or SPMD lazy arming) can be merged.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="not_a_rank_capture",
            remedy="capture each rank under torchlens.distributed.arm() and merge those",
            source=source,
        )
    boundaries = record["boundaries"]
    if not isinstance(boundaries, (list, tuple)):
        # W051-CAPT3 (IO remainder): the per-rank journal the merge JOINS is
        # validated here at merge entry, live and loaded cores alike. A
        # non-sequence ``boundaries`` (an int, a bool, a mapping, a string)
        # previously escaped as a bare TypeError from the row walk or was
        # mis-read row-by-row; it refuses typed with the container named.
        raise _refuse(
            f"boundaries of {source} is not a list of boundary records "
            f"(got {type(boundaries).__name__})",
            code=MergedErrorCode.MERGED_SCHEMA_INVALID,
            source=source,
        )
    install_epoch = record.get("install_epoch")
    if install_epoch not in _INSTALL_EPOCHS:
        raise _refuse(
            f"install_epoch {install_epoch!r} outside the closed vocabulary",
            source=source,
        )
    ranks: set[int] = set()
    seen_correlation_keys: set[tuple[str, int, str, int]] = set()
    group_rank_absence: dict[tuple[str, int], bool] = {}
    seq_seen_value = False
    seq_dropped = False
    for index, entry in enumerate(boundaries):
        if not isinstance(entry, dict):
            raise _refuse(f"boundary {index} of {source} is not a mapping", source=source)
        if entry.get("schema") == "functional_collective_boundary_v0":
            # Merge-ranks C2 recording (fail-closed): funcol boundaries carry
            # the documented-unstable v0 payload, which the frozen C1
            # derivation cannot join. Refusing the CORE typed is strictly
            # honest -- the shipped alternative was merging with the funcol
            # traffic invisibly absent. The merged-side funcol join is C2
            # merged-side work behind its own ruling (L8 plan 3.2c).
            raise MergeInputError(
                f"Merge input {source} records a functional-collective "
                f"(funcol) boundary (index {index}, kind "
                f"{entry.get('kind')!r}). C1 joins the frozen "
                "collective_boundary_v1 payload only; funcol boundary joining "
                "is rung-C2 merged-side scope and stays refused until its "
                "capture-fidelity census and authorizing ruling land.",
                code=MergedErrorCode.MERGE_SCOPE_UNSUPPORTED,
                reason="functional_collective_boundary_unsupported",
                source=source,
                boundary_index=index,
            )
        _validate_boundary(entry, index, source)
        # my_group_rank None-ness is UNIFORM per group within one rank core:
        # the writer mints None only when ``dist.get_group_rank`` raises -- a
        # group-level fact -- so mixed presence within one core+group is
        # tamper. Without this, stripping my_group_rank from one boundary of a
        # slice-witnessed join silently deleted its value_divergence finding
        # (R18-2; findings must never disappear by record deletion).
        uid = (
            entry["correlation"]["membership_digest"],
            entry["correlation"]["lifetime_ordinal"],
        )
        group_rank_is_none = entry["group"].get("my_group_rank") is None
        if uid in group_rank_absence and group_rank_absence[uid] != group_rank_is_none:
            raise _refuse(
                f"boundary {index} of {source} mixes my_group_rank presence and "
                f"absence within one group {uid}; group-rank readability is a "
                "group-level fact, so selective absence is tamper",
                source=source,
            )
        group_rank_absence.setdefault(uid, group_rank_is_none)
        # c10d_group_seq follows the recorder's process-global LATCH shape: a
        # prefix of values, at most ONE disclosing read-failure boundary, then
        # an all-null suffix. Selective nulling of a value (the cheap tamper
        # that deleted a correlation_delta_mismatch finding) refuses here;
        # uniformly-absent cores stay finding-free per the contract ("absence
        # of the probe never demotes anything") -- an all-null rewrite is the
        # documented coherent-reauthoring boundary, not an open residual.
        seq_value = entry["c10d_group_seq"]
        seq_disclosed = "c10d_group_seq_read_failed" in entry["disclosures"]
        if seq_value is not None:
            if seq_dropped:
                raise _refuse(
                    f"boundary {index} of {source} carries a c10d_group_seq value "
                    "after an earlier boundary of this core recorded none; the "
                    "probe never recovers within one capture",
                    source=source,
                )
            seq_seen_value = True
        elif not seq_dropped:
            if seq_seen_value and not seq_disclosed:
                raise _refuse(
                    f"boundary {index} of {source} drops c10d_group_seq without "
                    "the c10d_group_seq_read_failed disclosure while earlier "
                    "boundaries of this core carry values; an undisclosed drop "
                    "is tamper, not honest probe absence",
                    source=source,
                )
            seq_dropped = True
        elif seq_disclosed:
            raise _refuse(
                f"boundary {index} of {source} repeats the "
                "c10d_group_seq_read_failed disclosure after the probe already "
                "latched off; the recorder discloses the failed read exactly once",
                source=source,
            )
        # A rank has exactly ONE install epoch; a boundary claiming a
        # different one is a forged record trying to promote (or demote) its
        # own lifetime completeness independently of the rank's arming record.
        if entry["lifetime_evidence"]["install_epoch"] != install_epoch:
            raise _refuse(
                f"boundary {index} of {source} claims install_epoch "
                f"{entry['lifetime_evidence']['install_epoch']!r} but the rank "
                f"record's install_epoch is {install_epoch!r}",
                source=source,
            )
        ranks.add(int(entry["group"]["my_global_rank"]))
        correlation = entry["correlation"]
        correlation_key = (
            correlation["membership_digest"],
            correlation["lifetime_ordinal"],
            correlation["channel"],
            correlation["seq"],
        )
        if correlation_key in seen_correlation_keys:
            raise _refuse(
                f"boundary {index} of {source} duplicates rank-local correlation key "
                f"{correlation_key}",
                source=source,
            )
        seen_correlation_keys.add(correlation_key)
    if len(ranks) != 1:
        raise MergeInputError(
            f"Merge input {source} claims multiple global ranks {sorted(ranks)}; "
            "a rank core is a single-rank capture.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="multiple_ranks_in_one_core",
            remedy="re-capture the rank; one core must come from exactly one rank",
            source=source,
        )
    ledger_payload = record.get("group_lifecycle_ledger")
    if not isinstance(ledger_payload, list) or not ledger_payload:
        raise _refuse("group_lifecycle_ledger is absent or empty", source=source)
    try:
        ledger = GroupLifecycleLedger.from_payload(ledger_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise _refuse(f"group_lifecycle_ledger does not parse ({exc})", source=source) from exc
    # ``lineage_vectors()`` stamps each vector's install epoch from the EVENTS, while
    # the audit's completeness reasoning also reads the record-level epoch. A rank
    # has exactly one epoch, so a disagreement is a forged/corrupt sidecar trying to
    # promote a ``seeded`` rank to a complete witness; refuse rather than let the two
    # readings diverge.
    event_epochs = {event.install_epoch for event in ledger.events}
    if event_epochs != {install_epoch}:
        raise _refuse(
            f"group_lifecycle_ledger event install_epochs {sorted(event_epochs)} disagree "
            f"with the record install_epoch {install_epoch!r}",
            source=source,
        )
    return RankEvidence(
        rank=ranks.pop(),
        boundaries=tuple(boundaries),
        ledger=ledger,
        install_epoch=str(install_epoch),
        source=source,
        # The marker key of _guard_scope (L8 3.2c(2)): the geometry key alone
        # does not fire on TP/FSDP2 boundaries (funcol/c10d traffic on
        # to_local()-ed plain tensors carries no role geometry), so the
        # trace-level marker travels on the evidence itself. Hostile inputs:
        # anything other than the exact marker string reads False -- absence
        # of the marker never blocks, presence of ANY other value never
        # blocks; only the one documented value engages the scope guard.
        shard_local=(getattr(trace, "distributed_scope", None) == "rank_local_shard"),
    )


def _require_mergeable_outcome(trace: Any, source: str) -> None:
    """Refuse typed when a member core's settled capture outcome cannot merge.

    R06c: ``merge_ranks`` never consulted member capture outcomes, so a FAILED
    or UNKNOWN member core merged into ``aligned``/``attested_complete`` with
    zero findings. FAILED / ABORTED_NONFINITE / UNKNOWN members refuse here at
    the one input chokepoint (live traces and path-loaded bundles alike);
    HALTED and legacy UNATTESTED members merge and are DISCLOSED through
    ``MergedTrace.member_outcomes`` and ``summary()`` (demote-only: the
    disclosure is presenter-side and never edits the derivation). A trace
    object carrying no settled outcome sidecar (hand-built evidence carriers)
    makes no outcome claim and is not refused here -- the boundary-journal
    parse remains its gate.
    """

    from ..capture.outcome import CaptureStatus, outcome_for

    outcome = outcome_for(trace)
    if outcome is None:
        return
    if outcome.status in (
        CaptureStatus.FAILED,
        CaptureStatus.ABORTED_NONFINITE,
        CaptureStatus.UNKNOWN,
    ):
        raise MergeInputError(
            f"Merge input {source} carries a settled capture outcome "
            f"{outcome.status.value!r}; a failed, aborted, or unprovable rank "
            "capture cannot join a cross-rank merge.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="member_outcome_not_mergeable",
            member_status=outcome.status.value,
            remedy=(
                "re-capture the rank to a settled complete (or halted) outcome and merge that core"
            ),
            source=source,
        )


def resolve_rank_inputs(inputs: Sequence[Any]) -> dict[int, tuple[RankEvidence, Any]]:
    """Resolve merge inputs (traces or bundle paths) into per-rank evidence.

    Parameters
    ----------
    inputs:
        Live/loaded ``Trace`` objects or ``.tlspec`` rank-core paths, in any
        mix and order. Paths are loaded for analysis.

    Returns
    -------
    dict[int, tuple[RankEvidence, Any]]
        Mapping from global rank to ``(evidence, trace)``, ordered by rank.

    Raises
    ------
    MergeInputError
        On empty input, unloadable paths, duplicate ranks, or invalid cores.
    """

    if not inputs:
        raise MergeInputError(
            "merge_ranks requires at least one rank capture or rank-core path.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="empty_inputs",
            remedy="pass at least one rank capture or rank-core path",
        )
    resolved: dict[int, tuple[RankEvidence, Any]] = {}
    for position, item in enumerate(inputs):
        if isinstance(item, (str, Path)):
            from .._io.bundle import load as load_bundle

            source = str(item)
            try:
                trace = load_bundle(source)
            except Exception as exc:
                raise MergeInputError(
                    f"Merge input {source} failed to load as a rank core: {exc}",
                    code=MergedErrorCode.MERGE_INPUT_INVALID,
                    reason="member_load_failed",
                    remedy="inspect the chained cause; pass a loadable rank-core bundle",
                    source=source,
                ) from exc
        else:
            trace = item
            source = f"live[{position}]"
        evidence = extract_rank_evidence(trace, source)
        _require_mergeable_outcome(trace, source)
        if evidence.rank in resolved:
            raise MergeInputError(
                f"Merge inputs contain global rank {evidence.rank} twice "
                f"({resolved[evidence.rank][0].source} and {source}); every rank "
                "core must come from a distinct rank of one run.",
                code=MergedErrorCode.MERGE_INPUT_INVALID,
                reason="duplicate_rank",
                remedy="pass one core per rank; drop the duplicate input",
            )
        resolved[evidence.rank] = (evidence, trace)
    return dict(sorted(resolved.items()))
