"""Characterization matrix for the per-boundary parse rules of merge evidence.

``torchlens._merged._evidence._validate_boundary`` is the one chokepoint that
validates a ``collective_boundary_v1`` journal entry before the merge engine
reads it. This module pins its observable behaviour: a set of recorder-coherent
entries it ACCEPTS, and one mutation per raise site, each recorded with the
exception class, the ``code`` field, the full message and the payload keys.
The expected rows were recorded against the single-function implementation;
any split of the rules into helpers must reproduce them exactly, including
which refusal wins when one entry carries two defects (``order_*`` rows).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from torchlens._merged._errors import MergeInputError
from torchlens._merged._evidence import _validate_boundary, extract_rank_evidence
from torchlens.distributed._ledger import membership_digest_for_ranks

INDEX = 3
SOURCE = "live[0]"
WORLD_RANKS = [0, 1]
WORLD = membership_digest_for_ranks(WORLD_RANKS)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
TENSORLESS_SAMPLE = ("barrier", "all_gather_object")


def _entry(kind: str = "all_reduce", **overrides: Any) -> dict[str, Any]:
    """Return a recorder-coherent boundary entry of ``kind``.

    Parameters
    ----------
    kind:
        Boundary kind; tensorless kinds get no roles, no op node and no
        back-references, p2p kinds a peer record, reduce kinds a reduce op.
    **overrides:
        Top-level keys replaced after the coherent defaults are built.

    Returns
    -------
    dict[str, Any]
        A fresh entry that ``_validate_boundary`` accepts.
    """

    tensorless = kind in TENSORLESS_SAMPLE
    entry: dict[str, Any] = {
        "schema": "collective_boundary_v1",
        "kind": kind,
        "func": f"torch.distributed.{kind}",
        "correlation": {
            "membership_digest": WORLD,
            "lifetime_ordinal": 0,
            "channel": "coll",
            "seq": 4,
        },
        "group": {
            "global_ranks": list(WORLD_RANKS),
            "size": len(WORLD_RANKS),
            "backend": "gloo",
            "my_global_rank": 1,
            "my_group_rank": 1,
            "coord_provenance": "test",
        },
        "reduce_op": "RedOpType.SUM" if kind in ("all_reduce", "reduce_scatter") else None,
        "peer": {"raw": {"tag": 0}, "canonical": {"src": 0, "dst": 1}}
        if kind in ("send", "recv")
        else None,
        "events": {"async_op": False, "completion_binding": "issue_sync"},
        "roles": []
        if tensorless
        else [{"role": "contribution_destination", "index": 0, "shape": [2, 4]}],
        "witness": {
            "policy_resolved": "none",
            "contribution_digests": None,
            "destination_digests": None,
            "not_present_reason": None,
        },
        "lifetime_evidence": {"install_epoch": "seeded"},
        "c10d_group_seq": 7,
        "disclosures": [],
        "op_node": not tensorless,
        "op_labels_raw": [] if tensorless else [f"{kind}_1_1:1"],
    }
    entry.update(overrides)
    return entry


def _async_digest_entry() -> dict[str, Any]:
    """Return a coherent async (unobserved-completion) digest-witness entry."""

    entry = _entry()
    entry["events"] = {"async_op": True, "completion_binding": "unobserved"}
    entry["disclosures"] = ["read_of_inflight_destination"]
    entry["witness"] = {
        "policy_resolved": "digest",
        "contribution_digests": [DIGEST_A],
        "destination_digests": None,
        "not_present_reason": "async_completion_unobserved",
    }
    return entry


def _sync_digest_entry() -> dict[str, Any]:
    """Return a coherent synchronous digest-witness entry."""

    entry = _entry()
    entry["witness"] = {
        "policy_resolved": "digest",
        "contribution_digests": [DIGEST_A],
        "destination_digests": [DIGEST_B],
        "not_present_reason": None,
    }
    return entry


def _with(base: Callable[[], dict[str, Any]], *edits: tuple[str, Any]) -> dict[str, Any]:
    """Apply dotted-path edits to a fresh entry; ``...`` deletes the key.

    Parameters
    ----------
    base:
        Builder of the entry to mutate.
    *edits:
        ``(dotted_path, value)`` pairs; integer path parts index lists.

    Returns
    -------
    dict[str, Any]
        The mutated entry.
    """

    entry = copy.deepcopy(base())
    for path, value in edits:
        *parents, leaf = path.split(".")
        target: Any = entry
        for part in parents:
            target = target[int(part)] if isinstance(target, list) else target[part]
        key: Any = int(leaf) if isinstance(target, list) else leaf
        if value is ...:
            del target[key]
        else:
            target[key] = value
    return entry


ACCEPTED: dict[str, Callable[[], dict[str, Any]]] = {
    "all_reduce": _entry,
    "barrier_tensorless": lambda: _entry("barrier"),
    "all_gather_object_digest_empty_lists": lambda: _with(
        lambda: _entry("all_gather_object"),
        ("witness.policy_resolved", "digest"),
        ("witness.contribution_digests", []),
        ("witness.destination_digests", []),
    ),
    "broadcast_without_reduce_op_key": lambda: _with(
        lambda: _entry("broadcast"), ("reduce_op", ...)
    ),
    "send_with_peer": lambda: _entry("send"),
    "recv_with_peer": lambda: _entry("recv"),
    "async_digest": _async_digest_entry,
    "sync_digest": _sync_digest_entry,
    "my_group_rank_null": lambda: _with(_entry, ("group.my_group_rank", None)),
    "my_group_rank_absent": lambda: _with(_entry, ("group.my_group_rank", ...)),
    "backend_null": lambda: _with(_entry, ("group.backend", None)),
    "c10d_group_seq_null": lambda: _with(_entry, ("c10d_group_seq", None)),
    "c10d_group_seq_read_failed": lambda: _with(
        _entry,
        ("c10d_group_seq", None),
        ("disclosures", ["c10d_group_seq_read_failed"]),
    ),
    "destination_position_hole": lambda: _with(
        _entry,
        (
            "roles",
            [
                {"role": "contribution", "index": 0, "shape": [3]},
                {"role": "destination", "index": 1, "shape": [0, 2]},
                {"role": "contribution_destination", "index": 0, "shape": []},
            ],
        ),
    ),
    "armed_install_epoch": lambda: _with(
        _entry, ("lifetime_evidence.install_epoch", "armed_before_any_group")
    ),
    "single_rank_group": lambda: _with(
        _entry,
        ("group.global_ranks", [5]),
        ("group.size", 1),
        ("group.my_global_rank", 5),
        ("group.my_group_rank", 0),
        ("correlation.membership_digest", membership_digest_for_ranks([5])),
    ),
    "unsorted_membership": lambda: _with(
        _entry,
        ("group.global_ranks", [3, 1, 2]),
        ("group.size", 3),
        ("group.my_global_rank", 1),
        ("group.my_group_rank", 1),
        ("correlation.membership_digest", membership_digest_for_ranks([1, 2, 3])),
    ),
}
"""Recorder-coherent entries the parse must accept."""


REFUSED: dict[str, Callable[[], dict[str, Any]]] = {
    # identity: schema and kind
    "schema_wrong": lambda: _with(_entry, ("schema", "collective_boundary_v0")),
    "schema_missing": lambda: _with(_entry, ("schema", ...)),
    "kind_unknown": lambda: _with(_entry, ("kind", "all_reduce_v2")),
    "kind_missing": lambda: _with(_entry, ("kind", ...)),
    # correlation key
    "correlation_not_mapping": lambda: _with(_entry, ("correlation", [])),
    "correlation_extra_key": lambda: _with(_entry, ("correlation.extra", 1)),
    "correlation_missing_key": lambda: _with(_entry, ("correlation.seq", ...)),
    "membership_digest_not_str": lambda: _with(_entry, ("correlation.membership_digest", 5)),
    "channel_not_str": lambda: _with(_entry, ("correlation.channel", None)),
    "lifetime_ordinal_str": lambda: _with(_entry, ("correlation.lifetime_ordinal", "0")),
    "lifetime_ordinal_bool": lambda: _with(_entry, ("correlation.lifetime_ordinal", False)),
    "lifetime_ordinal_negative": lambda: _with(_entry, ("correlation.lifetime_ordinal", -1)),
    "seq_float": lambda: _with(_entry, ("correlation.seq", 1.0)),
    "seq_bool": lambda: _with(_entry, ("correlation.seq", True)),
    "seq_negative": lambda: _with(_entry, ("correlation.seq", -3)),
    # group membership
    "group_not_mapping": lambda: _with(_entry, ("group", None)),
    "group_ranks_not_list": lambda: _with(_entry, ("group.global_ranks", (0, 1))),
    "my_global_rank_missing": lambda: _with(_entry, ("group.my_global_rank", ...)),
    "my_global_rank_str": lambda: _with(_entry, ("group.my_global_rank", "1")),
    "group_rank_bool": lambda: _with(_entry, ("group.global_ranks", [0, True])),
    "group_rank_str": lambda: _with(_entry, ("group.global_ranks", [0, "1"])),
    "group_ranks_duplicate": lambda: _with(_entry, ("group.global_ranks", [1, 1])),
    "my_global_rank_outside": lambda: _with(_entry, ("group.my_global_rank", 4)),
    "my_global_rank_bool_outside": lambda: _with(
        _entry, ("group.global_ranks", [0, 2]), ("group.my_global_rank", True)
    ),
    "my_group_rank_bool": lambda: _with(_entry, ("group.my_group_rank", True)),
    "my_group_rank_str": lambda: _with(_entry, ("group.my_group_rank", "1")),
    "my_group_rank_permuted": lambda: _with(_entry, ("group.my_group_rank", 0)),
    "my_group_rank_out_of_range": lambda: _with(_entry, ("group.my_group_rank", 9)),
    "group_size_wrong": lambda: _with(_entry, ("group.size", 3)),
    "group_size_bool": lambda: _with(
        _entry,
        ("group.global_ranks", [1]),
        ("group.my_group_rank", 0),
        ("group.size", True),
    ),
    "group_size_missing": lambda: _with(_entry, ("group.size", ...)),
    "group_size_str": lambda: _with(_entry, ("group.size", "2")),
    "group_backend_not_str": lambda: _with(_entry, ("group.backend", 3)),
    "membership_digest_foreign": lambda: _with(
        _entry, ("correlation.membership_digest", membership_digest_for_ranks([0, 1, 2]))
    ),
    # tensor roles
    "roles_not_list": lambda: _with(_entry, ("roles", None)),
    "roles_empty_tensor_kind": lambda: _with(_entry, ("roles", [])),
    "role_not_mapping": lambda: _with(_entry, ("roles", ["contribution"])),
    "role_name_unknown": lambda: _with(_entry, ("roles.0.role", "source")),
    "role_name_missing": lambda: _with(_entry, ("roles.0.role", ...)),
    "role_shape_not_list": lambda: _with(_entry, ("roles.0.shape", (2, 4))),
    "role_shape_negative": lambda: _with(_entry, ("roles.0.shape", [2, -1])),
    "role_shape_bool": lambda: _with(_entry, ("roles.0.shape", [True])),
    "role_index_bool": lambda: _with(_entry, ("roles.0.index", False)),
    "role_index_negative": lambda: _with(_entry, ("roles.0.index", -1)),
    "role_index_missing": lambda: _with(_entry, ("roles.0.index", ...)),
    "role_index_duplicate": lambda: _with(
        _entry,
        (
            "roles",
            [
                {"role": "destination", "index": 2, "shape": [1]},
                {"role": "contribution", "index": 2, "shape": [1]},
                {"role": "destination", "index": 2, "shape": [1]},
            ],
        ),
    ),
    # reduce op
    "reduce_op_null": lambda: _with(_entry, ("reduce_op", None)),
    "reduce_op_empty": lambda: _with(_entry, ("reduce_op", "")),
    "reduce_op_missing_reduce_scatter": lambda: _with(
        lambda: _entry("reduce_scatter"), ("reduce_op", ...)
    ),
    # c10d group seq
    "c10d_group_seq_missing": lambda: _with(_entry, ("c10d_group_seq", ...)),
    "c10d_group_seq_bool": lambda: _with(_entry, ("c10d_group_seq", False)),
    "c10d_group_seq_str": lambda: _with(_entry, ("c10d_group_seq", "7")),
    # events
    "events_not_mapping": lambda: _with(_entry, ("events", "sync")),
    "events_binding_unknown": lambda: _with(_entry, ("events.completion_binding", "waited")),
    "events_async_not_bool": lambda: _with(_entry, ("events.async_op", 0)),
    "events_async_missing": lambda: _with(_entry, ("events.async_op", ...)),
    "events_async_incoherent": lambda: _with(_entry, ("events.async_op", True)),
    "events_unobserved_not_async": lambda: _with(_async_digest_entry, ("events.async_op", False)),
    # disclosures
    "disclosures_missing": lambda: _with(_entry, ("disclosures", ...)),
    "disclosures_not_list": lambda: _with(_entry, ("disclosures", "none")),
    "disclosures_non_str": lambda: _with(_entry, ("disclosures", [1])),
    "disclosures_unknown_tokens": lambda: _with(
        _entry, ("disclosures", ["zeta", "alpha", "c10d_group_seq_read_failed"])
    ),
    "disclosure_inflight_stripped": lambda: _with(_async_digest_entry, ("disclosures", [])),
    "disclosure_inflight_on_sync": lambda: _with(
        _entry, ("disclosures", ["read_of_inflight_destination"])
    ),
    "disclosure_seq_failed_with_value": lambda: _with(
        _entry, ("disclosures", ["c10d_group_seq_read_failed"])
    ),
    # op node
    "op_node_not_bool": lambda: _with(_entry, ("op_node", 1)),
    "op_node_false_on_tensor_kind": lambda: _with(_entry, ("op_node", False)),
    "op_node_true_on_tensorless": lambda: _with(lambda: _entry("barrier"), ("op_node", True)),
    # peer
    "p2p_without_peer": lambda: _with(lambda: _entry("send"), ("peer", None)),
    "p2p_peer_not_mapping": lambda: _with(lambda: _entry("recv"), ("peer", [0, 1])),
    "collective_with_peer": lambda: _with(_entry, ("peer", {"src": 0})),
    # witness
    "witness_not_mapping": lambda: _with(_entry, ("witness", "none")),
    "witness_missing": lambda: _with(_entry, ("witness", ...)),
    "witness_policy_unknown": lambda: _with(_entry, ("witness.policy_resolved", "payload")),
    "witness_reason_unknown": lambda: _with(_entry, ("witness.not_present_reason", "lost")),
    "witness_reason_missing_async": lambda: _with(
        _async_digest_entry, ("witness.not_present_reason", None)
    ),
    "witness_reason_on_sync": lambda: _with(
        _sync_digest_entry, ("witness.not_present_reason", "async_completion_unobserved")
    ),
    "witness_reason_policy_none": lambda: _with(
        _entry, ("witness.not_present_reason", "async_completion_unobserved")
    ),
    "witness_contribution_under_none": lambda: _with(
        _entry, ("witness.contribution_digests", [DIGEST_A])
    ),
    "witness_destination_under_none": lambda: _with(
        _entry, ("witness.destination_digests", [DIGEST_B])
    ),
    "witness_destination_unobserved": lambda: _with(
        _async_digest_entry, ("witness.destination_digests", [DIGEST_B])
    ),
    "witness_contribution_bare_string": lambda: _with(
        _sync_digest_entry, ("witness.contribution_digests", DIGEST_A)
    ),
    "witness_destination_empty_tensor_kind": lambda: _with(
        _sync_digest_entry, ("witness.destination_digests", [])
    ),
    "witness_contribution_uppercase": lambda: _with(
        _sync_digest_entry, ("witness.contribution_digests", ["A" * 64])
    ),
    "witness_destination_short": lambda: _with(
        _sync_digest_entry, ("witness.destination_digests", ["b" * 63])
    ),
    "witness_destination_non_str": lambda: _with(
        _sync_digest_entry, ("witness.destination_digests", [7])
    ),
    "witness_tensorless_bad_digest": lambda: _with(
        lambda: _entry("barrier"),
        ("witness.policy_resolved", "digest"),
        ("witness.contribution_digests", ["x"]),
    ),
    # op-label back-references
    "op_labels_missing": lambda: _with(_entry, ("op_labels_raw", ...)),
    "op_labels_tuple": lambda: _with(_entry, ("op_labels_raw", ("a",))),
    "op_labels_non_str": lambda: _with(_entry, ("op_labels_raw", ["a", 2])),
    "op_labels_on_tensorless": lambda: _with(
        lambda: _entry("all_gather_object"), ("op_labels_raw", ["x_1_1:1"])
    ),
    "op_labels_empty_op_bearing": lambda: _with(_entry, ("op_labels_raw", [])),
    "op_labels_duplicate": lambda: _with(_entry, ("op_labels_raw", ["a", "b", "a"])),
    # lifetime evidence
    "lifetime_not_mapping": lambda: _with(_entry, ("lifetime_evidence", "seeded")),
    "lifetime_missing": lambda: _with(_entry, ("lifetime_evidence", ...)),
    "lifetime_epoch_unknown": lambda: _with(_entry, ("lifetime_evidence.install_epoch", "late")),
    # two defects at once: the earlier rule group must win
    "order_schema_before_kind": lambda: _with(_entry, ("schema", None), ("kind", "nope")),
    "order_kind_before_correlation": lambda: _with(_entry, ("kind", "nope"), ("correlation", None)),
    "order_correlation_before_group": lambda: _with(
        _entry, ("correlation.seq", -1), ("group", None)
    ),
    "order_size_before_backend": lambda: _with(_entry, ("group.size", 9), ("group.backend", 1)),
    "order_backend_before_digest": lambda: _with(
        _entry,
        ("group.backend", 1),
        ("correlation.membership_digest", DIGEST_A),
    ),
    "order_digest_before_roles": lambda: _with(
        _entry, ("correlation.membership_digest", DIGEST_A), ("roles", None)
    ),
    "order_first_bad_role_wins": lambda: _with(
        _entry,
        (
            "roles",
            [
                {"role": "contribution", "index": 0, "shape": [-1]},
                {"role": "nope", "index": 0, "shape": [1]},
            ],
        ),
    ),
    "order_role_before_reduce_op": lambda: _with(
        _entry, ("roles.0.index", -1), ("reduce_op", None)
    ),
    "order_reduce_op_before_seq_key": lambda: _with(
        _entry, ("reduce_op", None), ("c10d_group_seq", ...)
    ),
    "order_seq_before_events": lambda: _with(_entry, ("c10d_group_seq", "x"), ("events", None)),
    "order_events_before_disclosures": lambda: _with(
        _entry, ("events.async_op", None), ("disclosures", None)
    ),
    "order_disclosures_before_op_node": lambda: _with(
        _entry, ("disclosures", ["zeta"]), ("op_node", None)
    ),
    "order_op_node_before_peer": lambda: _with(_entry, ("op_node", None), ("peer", {})),
    "order_peer_before_witness": lambda: _with(_entry, ("peer", {}), ("witness", None)),
    "order_contribution_before_destination": lambda: _with(
        _sync_digest_entry,
        ("witness.contribution_digests", ["bad"]),
        ("witness.destination_digests", "bad"),
    ),
    "order_witness_before_labels": lambda: _with(
        _entry, ("witness.policy_resolved", None), ("op_labels_raw", None)
    ),
    "order_labels_before_lifetime": lambda: _with(
        _entry, ("op_labels_raw", ["a", "a"]), ("lifetime_evidence", None)
    ),
}
"""One mutation per raise site (several sites take more than one), plus order rows."""


_MESSAGE_PREFIX = (
    "Rank-core evidence is not a valid collective_boundary_v1 journal: boundary 3 of live[0] "
)
_REMEDY = (
    "treat the rank core as tampered or corrupt; re-capture the rank under the "
    "distributed opt-in rather than merging this evidence"
)

EXPECTED_CLASS = "MergeInputError"
EXPECTED_CODE = "merged_schema_invalid"
"""The recording run saw exactly this class and code on every refusal row."""

EXPECTED_DETAILS: dict[str, str] = {
    "schema_wrong": "has schema 'collective_boundary_v0'",
    "schema_missing": "has schema None",
    "kind_unknown": "has unknown kind 'all_reduce_v2'",
    "kind_missing": "has unknown kind None",
    "correlation_not_mapping": "has a malformed correlation key",
    "correlation_extra_key": "has a malformed correlation key",
    "correlation_missing_key": "has a malformed correlation key",
    "membership_digest_not_str": "membership_digest is not a string",
    "channel_not_str": "correlation channel is not a string",
    "lifetime_ordinal_str": "lifetime_ordinal is not a non-negative integer",
    "lifetime_ordinal_bool": "lifetime_ordinal is not a non-negative integer",
    "lifetime_ordinal_negative": "lifetime_ordinal is not a non-negative integer",
    "seq_float": "seq is not a non-negative integer",
    "seq_bool": "seq is not a non-negative integer",
    "seq_negative": "seq is not a non-negative integer",
    "group_not_mapping": "has no group membership record",
    "group_ranks_not_list": "has no group membership record",
    "my_global_rank_missing": "has no my_global_rank",
    "my_global_rank_str": "has no my_global_rank",
    "group_rank_bool": "group membership contains a non-integer rank",
    "group_rank_str": "group membership contains a non-integer rank",
    "group_ranks_duplicate": "group membership contains duplicate ranks",
    "my_global_rank_outside": "claims rank 4 outside its group membership",
    "my_global_rank_bool_outside": "claims rank True outside its group membership",
    "my_group_rank_bool": "my_group_rank is not an integer or null",
    "my_group_rank_str": "my_group_rank is not an integer or null",
    "my_group_rank_permuted": "my_group_rank 0 does not equal the position of my_global_rank 1 in its own recorded group membership [0, 1]",
    "my_group_rank_out_of_range": "my_group_rank 9 does not equal the position of my_global_rank 1 in its own recorded group membership [0, 1]",
    "group_size_wrong": "group size 3 does not equal its recorded membership of 2 rank(s)",
    "group_size_bool": "group size True does not equal its recorded membership of 1 rank(s)",
    "group_size_missing": "group size None does not equal its recorded membership of 2 rank(s)",
    "group_size_str": "group size '2' does not equal its recorded membership of 2 rank(s)",
    "group_backend_not_str": "group backend is not a string or null",
    "membership_digest_foreign": "membership_digest does not equal the digest of its own recorded group membership [0, 1]",
    "roles_not_list": "has no roles list",
    "roles_empty_tensor_kind": "is a tensor-carrying all_reduce boundary with zero tensor roles; a successful collective of this kind always records at least one role, so an empty or deleted roles record is not honest evidence",
    "role_not_mapping": "role entry 0 is not a mapping",
    "role_name_unknown": "role entry 0 has role 'source' outside the closed vocabulary",
    "role_name_missing": "role entry 0 has role None outside the closed vocabulary",
    "role_shape_not_list": "role entry 0 has no well-formed shape (a list of non-negative integers)",
    "role_shape_negative": "role entry 0 has no well-formed shape (a list of non-negative integers)",
    "role_shape_bool": "role entry 0 has no well-formed shape (a list of non-negative integers)",
    "role_index_bool": "role entry 0 has no non-negative integer index",
    "role_index_negative": "role entry 0 has no non-negative integer index",
    "role_index_missing": "role entry 0 has no non-negative integer index",
    "role_index_duplicate": "role entry 2 duplicates index 2 within role 'destination'; the recorder emits each tensor position at most once per role",
    "reduce_op_null": "is a all_reduce boundary without its reduce_op record; a successful call of this kind always records one",
    "reduce_op_empty": "is a all_reduce boundary without its reduce_op record; a successful call of this kind always records one",
    "reduce_op_missing_reduce_scatter": "is a reduce_scatter boundary without its reduce_op record; a successful call of this kind always records one",
    "c10d_group_seq_missing": "lacks its c10d_group_seq record",
    "c10d_group_seq_bool": "c10d_group_seq is not an integer or null",
    "c10d_group_seq_str": "c10d_group_seq is not an integer or null",
    "events_not_mapping": "has a malformed events record",
    "events_binding_unknown": "has a malformed events record",
    "events_async_not_bool": "events async_op is not a boolean",
    "events_async_missing": "events async_op is not a boolean",
    "events_async_incoherent": "events record is incoherent: async_op=True with completion_binding='issue_sync' (an unobserved completion is exactly the async case)",
    "events_unobserved_not_async": "events record is incoherent: async_op=False with completion_binding='unobserved' (an unobserved completion is exactly the async case)",
    "disclosures_missing": "disclosures is not a list of strings",
    "disclosures_not_list": "disclosures is not a list of strings",
    "disclosures_non_str": "disclosures is not a list of strings",
    "disclosures_unknown_tokens": "disclosures contain tokens ['alpha', 'zeta'] outside the closed vocabulary",
    "disclosure_inflight_stripped": "disclosure record is incoherent: completion_binding 'unobserved' with read_of_inflight_destination absent (every unobserved completion records the in-flight-destination read, and no observed completion does)",
    "disclosure_inflight_on_sync": "disclosure record is incoherent: completion_binding 'issue_sync' with read_of_inflight_destination present (every unobserved completion records the in-flight-destination read, and no observed completion does)",
    "disclosure_seq_failed_with_value": "discloses a failed c10d_group_seq read yet carries a c10d_group_seq value",
    "op_node_not_bool": "op_node does not match its kind's tensorless class ('all_reduce' boundaries are journal-plus-op)",
    "op_node_false_on_tensor_kind": "op_node does not match its kind's tensorless class ('all_reduce' boundaries are journal-plus-op)",
    "op_node_true_on_tensorless": "op_node does not match its kind's tensorless class ('barrier' boundaries are journal-only)",
    "p2p_without_peer": "is a p2p boundary without its peer record",
    "p2p_peer_not_mapping": "is a p2p boundary without its peer record",
    "collective_with_peer": "is a collective boundary carrying a peer record",
    "witness_not_mapping": "has witness policy 'none' outside the closed vocabulary",
    "witness_missing": "has witness policy None outside the closed vocabulary",
    "witness_policy_unknown": "has witness policy 'payload' outside the closed vocabulary",
    "witness_reason_unknown": "witness not_present_reason 'lost' outside the closed vocabulary",
    "witness_reason_missing_async": "witness not_present_reason None is incoherent with policy 'digest' and completion_binding 'unobserved' (expected 'async_completion_unobserved')",
    "witness_reason_on_sync": "witness not_present_reason 'async_completion_unobserved' is incoherent with policy 'digest' and completion_binding 'issue_sync' (expected None)",
    "witness_reason_policy_none": "witness not_present_reason 'async_completion_unobserved' is incoherent with policy 'none' and completion_binding 'issue_sync' (expected None)",
    "witness_contribution_under_none": "witness carries contribution_digests under policy_resolved 'none'; a rank that never computed witnesses cannot present digests, so these are forged, not evidence",
    "witness_destination_under_none": "witness carries destination_digests under policy_resolved 'none'; a rank that never computed witnesses cannot present digests, so these are forged, not evidence",
    "witness_destination_unobserved": "witness carries destination_digests under an unobserved completion; the destination bytes were never observed at capture, so these digests are forged, not evidence",
    "witness_contribution_bare_string": "witness contribution_digests is not null or a non-empty list",
    "witness_destination_empty_tensor_kind": "witness destination_digests is not null or a non-empty list",
    "witness_contribution_uppercase": "witness contribution_digests contains a value that is not a lowercase hex SHA-256 digest",
    "witness_destination_short": "witness destination_digests contains a value that is not a lowercase hex SHA-256 digest",
    "witness_destination_non_str": "witness destination_digests contains a value that is not a lowercase hex SHA-256 digest",
    "witness_tensorless_bad_digest": "witness contribution_digests contains a value that is not a lowercase hex SHA-256 digest",
    "op_labels_missing": "lacks op_labels_raw back-references",
    "op_labels_tuple": "lacks op_labels_raw back-references",
    "op_labels_non_str": "op_labels_raw contains a non-string label",
    "op_labels_on_tensorless": "is a journal-only all_gather_object boundary carrying op-label back-references; tensorless kinds never emit an op node",
    "op_labels_empty_op_bearing": "is an op-bearing all_reduce boundary with zero op-label back-references; a successful call of this kind always logs at least one boundary output tensor",
    "op_labels_duplicate": "op_labels_raw contains duplicate labels; each back-reference names a distinct logged output tensor",
    "lifetime_not_mapping": "lifetime_evidence install_epoch 'seeded' outside the closed vocabulary",
    "lifetime_missing": "lifetime_evidence install_epoch None outside the closed vocabulary",
    "lifetime_epoch_unknown": "lifetime_evidence install_epoch 'late' outside the closed vocabulary",
    "order_schema_before_kind": "has schema None",
    "order_kind_before_correlation": "has unknown kind 'nope'",
    "order_correlation_before_group": "seq is not a non-negative integer",
    "order_size_before_backend": "group size 9 does not equal its recorded membership of 2 rank(s)",
    "order_backend_before_digest": "group backend is not a string or null",
    "order_digest_before_roles": "membership_digest does not equal the digest of its own recorded group membership [0, 1]",
    "order_first_bad_role_wins": "role entry 0 has no well-formed shape (a list of non-negative integers)",
    "order_role_before_reduce_op": "role entry 0 has no non-negative integer index",
    "order_reduce_op_before_seq_key": "is a all_reduce boundary without its reduce_op record; a successful call of this kind always records one",
    "order_seq_before_events": "c10d_group_seq is not an integer or null",
    "order_events_before_disclosures": "events async_op is not a boolean",
    "order_disclosures_before_op_node": "disclosures contain tokens ['zeta'] outside the closed vocabulary",
    "order_op_node_before_peer": "op_node does not match its kind's tensorless class ('all_reduce' boundaries are journal-plus-op)",
    "order_peer_before_witness": "is a collective boundary carrying a peer record",
    "order_contribution_before_destination": "witness contribution_digests contains a value that is not a lowercase hex SHA-256 digest",
    "order_witness_before_labels": "has witness policy None outside the closed vocabulary",
    "order_labels_before_lifetime": "op_labels_raw contains duplicate labels; each back-reference names a distinct logged output tensor",
}
"""Recorded on the single-function implementation: message text after the prefix."""


def test_matrix_rows_line_up() -> None:
    """Every refusal row has a recorded expectation and no expectation is stale."""

    assert set(EXPECTED_DETAILS) == set(REFUSED)
    assert not set(ACCEPTED) & set(REFUSED)


@pytest.mark.parametrize("name", sorted(ACCEPTED))
def test_coherent_entry_is_accepted(name: str) -> None:
    """A recorder-coherent entry parses without refusal and is not mutated."""

    entry = ACCEPTED[name]()
    snapshot = copy.deepcopy(entry)
    assert _validate_boundary(entry, INDEX, SOURCE) is None
    assert entry == snapshot


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_mutated_entry_refuses_exactly_as_recorded(name: str) -> None:
    """One mutation refuses with the recorded class, code, message and payload."""

    with pytest.raises(MergeInputError) as excinfo:
        _validate_boundary(REFUSED[name](), INDEX, SOURCE)
    error = excinfo.value
    assert type(error).__name__ == EXPECTED_CLASS
    assert str(error) == _MESSAGE_PREFIX + EXPECTED_DETAILS[name]
    assert dict(error.fields) == {
        "code": EXPECTED_CODE,
        "findings": (),
        "remedy": _REMEDY,
        "source": SOURCE,
    }


@pytest.mark.parametrize("name", ["schema_wrong", "role_index_duplicate", "lifetime_missing"])
def test_extract_rank_evidence_reaches_the_same_refusal(name: str) -> None:
    """The public extraction path surfaces the per-boundary refusal unchanged."""

    trace = type(
        "_Carrier",
        (),
        {
            "annotations": {
                "distributed": {
                    "boundaries": [
                        *(_with(_entry, ("correlation.seq", seq)) for seq in range(INDEX)),
                        REFUSED[name](),
                    ],
                    "install_epoch": "seeded",
                }
            }
        },
    )()
    with pytest.raises(MergeInputError) as excinfo:
        extract_rank_evidence(trace, SOURCE)
    assert str(excinfo.value) == _MESSAGE_PREFIX + EXPECTED_DETAILS[name]
