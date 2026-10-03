"""Unit matrix for the C1 merge derivation over synthetic rank evidence.

Pure-function tests of ``torchlens.merged._engine.derive_merge`` (no process
groups, no torch.distributed init): the PRE-JOIN audit matrix (merge-side
rows of design-merge-ranks-c v5, 1.3), seq-delta alignment, relation and
cross-check conflicts, the totalized witness derivation, expected-ranks
widening, determinism, and the contract-doc lockstep gates.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from torchlens.distributed._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    membership_digest_for_ranks,
)
from torchlens.merged import (
    MERGE_FINDING_KINDS,
    BoundaryConsistency,
    MergeAlignment,
    MergedErrorCode,
    MergeValueStatus,
    derive_merge,
)
from torchlens.merged._errors import MergeInputError
from torchlens.merged._evidence import (
    P2P_KINDS,
    TENSORLESS_KINDS,
    RankEvidence,
    extract_rank_evidence,
)

pytestmark = pytest.mark.smoke

WORLD = membership_digest_for_ranks([0, 1])


def seeded_ledger(digest: str = WORLD) -> GroupLifecycleLedger:
    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent(0, "seed", digest, 0, "seeded", "seeded", 0))
    return ledger


def armed_ledger(digest: str = WORLD, generations: int = 1) -> GroupLifecycleLedger:
    ledger = GroupLifecycleLedger()
    index = 0
    for ordinal in range(generations):
        ledger.append(
            GroupLifecycleEvent(
                index, "create", digest, ordinal, "wrapped", "armed_before_any_group", ordinal
            )
        )
        index += 1
        if ordinal < generations - 1:
            ledger.append(
                GroupLifecycleEvent(
                    index, "destroy", digest, ordinal, "wrapped", "armed_before_any_group"
                )
            )
            index += 1
    return ledger


def boundary(
    rank: int,
    seq: int,
    *,
    kind: str = "all_reduce",
    digest: str = WORLD,
    ordinal: int = 0,
    channel: str = "coll",
    members: tuple[int, ...] = (0, 1),
    backend: str | None = "gloo",
    roles: list[dict] | None = None,
    witness_policy: str = "none",
    contribution_digests: list[str] | None = None,
    destination_digests: list[str] | None = None,
    reduce_op: str | None = "RedOpType.SUM",
    c10d_group_seq: int | None = None,
    async_op: bool = False,
) -> dict:
    if roles is None:
        roles = [
            {
                "role": "contribution_destination",
                "index": 0,
                "shape": [2, 4],
                "logical_shape": None,
                "placements": None,
            }
        ]
    return {
        "schema": "collective_boundary_v1",
        "kind": kind,
        "func": f"torch.distributed.{kind}",
        "correlation": {
            "membership_digest": digest,
            "lifetime_ordinal": ordinal,
            "channel": channel,
            "seq": seq,
        },
        "group": {
            "global_ranks": list(members),
            "size": len(members),
            "backend": backend,
            "my_global_rank": rank,
            "my_group_rank": members.index(rank) if rank in members else None,
            "coord_provenance": "test",
        },
        "reduce_op": reduce_op,
        "peer": (
            {"raw": {"tag": 0}, "canonical": {"src": 0, "dst": 1}} if kind in P2P_KINDS else None
        ),
        "events": {
            "async_op": async_op,
            "completion_binding": "unobserved" if async_op else "issue_sync",
        },
        "roles": roles,
        "witness": {
            "policy_resolved": witness_policy,
            "contribution_digests": contribution_digests,
            "destination_digests": destination_digests,
            "not_present_reason": (
                "async_completion_unobserved" if async_op and witness_policy == "digest" else None
            ),
        },
        "lifetime_evidence": {
            "ordinal_source": "seeded",
            "install_epoch": "seeded",
            "arming_source": "explicit",
        },
        "c10d_group_seq": c10d_group_seq,
        # Recorder-coherent derived fields: the parse chokepoint refuses
        # records whose disclosure/op_node surface contradicts the rest
        # (tensorless kinds emit no op node, hence no back-references).
        "disclosures": ["read_of_inflight_destination"] if async_op else [],
        "op_labels_raw": [] if kind in TENSORLESS_KINDS else [f"{kind}_{seq}_raw_r{rank}"],
        "op_node": kind not in TENSORLESS_KINDS,
    }


def evidence(rank: int, boundaries: list[dict], ledger=None, epoch: str = "seeded") -> RankEvidence:
    return RankEvidence(
        shard_local=False,
        rank=rank,
        boundaries=tuple(boundaries),
        ledger=ledger if ledger is not None else seeded_ledger(),
        install_epoch=epoch,
        source=f"synthetic[{rank}]",
    )


def digest_kwargs(dest: str = "aa") -> dict:
    return {
        "witness_policy": "digest",
        "contribution_digests": ["cc"],
        "destination_digests": [dest],
    }


def trace_for_boundaries(boundaries: list[dict], ledger: GroupLifecycleLedger) -> SimpleNamespace:
    """Build the minimal trace surface consumed by rank-evidence extraction.

    Parameters
    ----------
    boundaries:
        Synthetic collective-boundary journal.
    ledger:
        Matching group-lifecycle ledger.

    Returns
    -------
    SimpleNamespace
        Trace-shaped object carrying distributed annotations.
    """

    return SimpleNamespace(
        annotations={
            "distributed": {
                "boundaries": boundaries,
                "group_lifecycle_ledger": ledger.to_payload(),
                "install_epoch": "seeded",
            }
        }
    )


class TestDeltaAlignment:
    def test_differing_absolute_bases_join_by_delta(self):
        # Rank 0 armed earlier and ticked warmups: absolute seqs 5,6 vs 0,1.
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 5), boundary(0, 6)]),
                1: evidence(1, [boundary(1, 0), boundary(1, 1)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED
        assert [j.key[3] for j in d.joins] == [0, 1]
        assert all(j.presence == (0, 1) for j in d.joins)
        assert {r.seq_abs for r in d.joins[0].per_rank.values()} == {5, 0}

    def test_three_rank_join(self):
        digest3 = membership_digest_for_ranks([0, 1, 2])

        def led():
            return seeded_ledger(digest3)

        cores = {
            rank: evidence(
                rank,
                [boundary(rank, 0, digest=digest3, members=(0, 1, 2))],
                ledger=led(),
            )
            for rank in (0, 1, 2)
        }
        d = derive_merge(cores)
        assert d.stored_alignment is MergeAlignment.ALIGNED
        (join,) = d.joins
        assert join.presence == (0, 1, 2)

    def test_missing_key_is_presence_gap_and_partial(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0), boundary(0, 1)]),
                1: evidence(1, [boundary(1, 0)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.PARTIAL
        gaps = d.gap_findings
        assert len(gaps) == 1 and gaps[0].ranks == (1,)

    def test_member_rank_without_core_is_gap_on_every_join(self):
        # Membership records [0, 1] but only rank 0's core is presented.
        d = derive_merge({0: evidence(0, [boundary(0, 0)])})
        assert d.stored_alignment is MergeAlignment.PARTIAL
        (gap,) = d.gap_findings
        assert gap.ranks == (1,)


class TestAuditMatrix:
    """Merge-side rows of the v5 1.3 PRE-JOIN audit matrix."""

    def test_asymmetric_arming_conflicts_with_zero_gaps_and_zero_joins(self):
        # The review's repro shape: rank 0 evidences {0: seeded (destroyed), 1: wrapped}
        # and captured the recreated group as uid (digest, 1); late-arming
        # rank 1 evidences {0: seeded} and captured it as (digest, 0). The
        # uids never join -- the audit must refuse BEFORE presence-gap
        # derivation, so the outcome is structural with ZERO gaps.
        led0 = GroupLifecycleLedger()
        led0.append(GroupLifecycleEvent(0, "seed", WORLD, 0, "seeded", "seeded", 0))
        led0.append(GroupLifecycleEvent(1, "destroy", WORLD, 0, "seeded", "seeded"))
        led0.append(GroupLifecycleEvent(2, "create", WORLD, 1, "wrapped", "seeded", 1))
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, ordinal=1)], ledger=led0),
                1: evidence(1, [boundary(1, 0, ordinal=0)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert [f.kind for f in d.findings] == ["group_lifetime_evidence_conflict"]
        assert d.joins == ()
        assert d.gap_findings == ()
        assert WORLD in d.conflicted_memberships

    def test_seed_discharge_positive_mixed_armed_and_seeded(self):
        # Rank 0 armed before any group (complete witness, ONE generation);
        # rank 1 seeded ordinal 0. The seed discharges: clean join, no
        # over-refusal of the legitimate mixed MPMD case.
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(1, [boundary(1, 0)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED
        assert len(d.joins) == 1 and d.joins[0].presence == (0, 1)

    def test_seed_discharge_negative_two_generation_witness(self):
        # The complete witness evidences TWO generations: a seeded ordinal-0
        # entry from another rank is unprovable and the membership refuses.
        led0 = armed_ledger(generations=2)
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0, ordinal=1)],
                    ledger=led0,
                    epoch="armed_before_any_group",
                ),
                1: evidence(1, [boundary(1, 0, ordinal=0)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert d.joins == () and d.gap_findings == ()

    def test_no_witness_identical_vectors_join(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0)], ledger=seeded_ledger()),
                1: evidence(1, [boundary(1, 0)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED

    def test_complete_witness_disagreement_is_evidence_corruption(self):
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0)],
                    ledger=armed_ledger(generations=1),
                    epoch="armed_before_any_group",
                ),
                1: evidence(
                    1,
                    [boundary(1, 0, ordinal=1)],
                    ledger=armed_ledger(generations=2),
                    epoch="armed_before_any_group",
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert d.findings[0].kind == "group_lifetime_evidence_conflict"
        assert "complete witnesses disagree" in d.findings[0].detail


class TestScopeAndInputRefusals:
    def test_p2p_kind_refuses_typed(self):
        core = evidence(0, [boundary(0, 0, kind="send", channel="p2p/0->1", reduce_op=None)])
        with pytest.raises(MergeInputError) as excinfo:
            derive_merge({0: core})
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGE_SCOPE_UNSUPPORTED.value

    def test_dtensor_dual_geometry_refuses_typed(self):
        roles = [
            {
                "role": "contribution_destination",
                "index": 0,
                "shape": [2, 4],
                "logical_shape": [4, 4],
                "placements": ["Shard(dim=0)"],
            }
        ]
        core = evidence(0, [boundary(0, 0, roles=roles)])
        with pytest.raises(MergeInputError) as excinfo:
            derive_merge({0: core})
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGE_SCOPE_UNSUPPORTED.value

    def test_parse_refuses_rank_outside_recorded_membership(self) -> None:
        """A rank core cannot claim evidence for a group it does not belong to."""

        forged = boundary(5, 0, members=(0, 1), **digest_kwargs())
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries([forged], seeded_ledger()),
                "forged-rank-5",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_join_refuses_presence_outside_recorded_membership(self) -> None:
        """Direct engine callers receive the same presence-subset refusal."""

        with pytest.raises(MergeInputError) as excinfo:
            derive_merge(
                {
                    0: evidence(0, [boundary(0, 0, **digest_kwargs())]),
                    5: evidence(5, [boundary(5, 0, **digest_kwargs())]),
                }
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_parse_refuses_duplicate_rank_local_sequence(self) -> None:
        """Duplicate absolute sequence keys cannot overwrite a boundary silently."""

        duplicated = [boundary(0, 7), boundary(0, 7)]
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries(duplicated, seeded_ledger()),
                "duplicate-seq-rank-0",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_join_refuses_duplicate_rank_local_sequence(self) -> None:
        """Direct engine evidence cannot exploit duplicate-sequence overwrite."""

        with pytest.raises(MergeInputError) as excinfo:
            derive_merge(
                {
                    0: evidence(0, [boundary(0, 7), boundary(0, 7)]),
                    1: evidence(1, [boundary(1, 3)]),
                }
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


class TestBoundaryParseValidation:
    """Deep-hunt F2: roles and witness digest fields are validated typed at parse.

    Fail-before: a role entry without ``shape`` passed extraction and escaped
    ``derive_merge`` as a raw ``KeyError('shape')`` from both ``merge_ranks``
    and load rederivation; a bare-STRING digest field char-split through
    ``tuple(...)`` and two cores carrying the same garbage string rendered a
    fabricated ``attested_complete``.
    """

    def _extract(self, entry: dict) -> None:
        extract_rank_evidence(
            trace_for_boundaries([entry], seeded_ledger()),
            "forged-boundary",
        )

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            self._extract(entry)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_role_entry_missing_shape_refuses_typed(self):
        self._assert_refuses(
            boundary(0, 0, roles=[{"role": "contribution_destination", "index": 0}])
        )

    def test_role_entry_not_a_mapping_refuses_typed(self):
        self._assert_refuses(boundary(0, 0, roles=["contribution"]))

    def test_roles_not_a_list_refuses_typed(self):
        entry = boundary(0, 0)
        entry["roles"] = {"role": "contribution"}
        self._assert_refuses(entry)

    def test_role_name_outside_vocabulary_refuses_typed(self):
        self._assert_refuses(
            boundary(0, 0, roles=[{"role": "spectator", "index": 0, "shape": [2]}])
        )

    def test_role_shape_with_non_integer_dim_refuses_typed(self):
        self._assert_refuses(
            boundary(0, 0, roles=[{"role": "contribution", "index": 0, "shape": [2, "x"]}])
        )

    def test_string_digest_field_refuses_typed(self):
        entry = boundary(0, 0, witness_policy="digest")
        entry["witness"]["contribution_digests"] = "ccdd"
        entry["witness"]["destination_digests"] = "aabb"
        self._assert_refuses(entry)

    def test_non_hex_digest_element_refuses_typed(self):
        entry = boundary(0, 0, witness_policy="digest")
        entry["witness"]["destination_digests"] = ["not-a-digest"]
        self._assert_refuses(entry)

    def test_non_string_op_label_refuses_typed(self):
        entry = boundary(0, 0)
        entry["op_labels_raw"] = ["fine", 7]
        self._assert_refuses(entry)

    def test_non_integer_my_group_rank_refuses_typed(self):
        entry = boundary(0, 0)
        entry["group"]["my_group_rank"] = "0"
        self._assert_refuses(entry)

    def test_non_string_backend_refuses_typed(self):
        entry = boundary(0, 0)
        entry["group"]["backend"] = 7
        self._assert_refuses(entry)

    def test_non_string_channel_refuses_typed(self):
        entry = boundary(0, 0)
        entry["correlation"]["channel"] = 0
        self._assert_refuses(entry)

    def test_unhashable_list_channel_refuses_typed(self):
        # A LIST channel used to escape the parse as a raw
        # ``TypeError: unhashable type: 'list'`` from the rank-local
        # correlation-key dedup set, not the promised typed refusal.
        entry = boundary(0, 0)
        entry["correlation"]["channel"] = ["coll"]
        self._assert_refuses(entry)

    def test_negative_seq_refuses_typed(self):
        self._assert_refuses(boundary(0, -1))

    def test_valid_sha256_digest_lists_still_parse(self):
        entry = boundary(
            0,
            0,
            witness_policy="digest",
            contribution_digests=["c" * 64],
            destination_digests=["a" * 64],
        )
        self._extract(entry)  # must not raise

    def test_parse_refuses_membership_digest_ranks_incoherence(self):
        """Deep-hunt F3: the digest must equal sha256(sorted(global_ranks)).

        Fail-before: two cores presenting the digest of a DIFFERENT membership
        ([5, 6, 7]) over global_ranks [0, 1] merged ALIGNED, rebinding one
        communicator's boundaries to another membership's digest, ordinal
        lineage, and audit row.
        """

        fake = membership_digest_for_ranks([5, 6, 7])
        self._assert_refuses(boundary(0, 0, digest=fake))

    def test_engine_refuses_membership_digest_ranks_incoherence(self):
        """Direct-engine evidence receives the same digest-coherence refusal."""

        fake = membership_digest_for_ranks([5, 6, 7])
        with pytest.raises(MergeInputError) as excinfo:
            derive_merge(
                {
                    0: evidence(0, [boundary(0, 0, digest=fake)], ledger=seeded_ledger(fake)),
                    1: evidence(1, [boundary(1, 0, digest=fake)], ledger=seeded_ledger(fake)),
                }
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_engine_belt_refuses_string_digests_typed(self):
        """Direct-engine evidence cannot fabricate ATTESTED via char-split.

        Fail-before: consistency rendered ``attested`` and the merge presented
        ``attested_complete`` from two identical garbage strings.
        """

        def forged(rank: int) -> dict:
            entry = boundary(rank, 0, witness_policy="digest")
            entry["witness"]["contribution_digests"] = "ccdd"
            entry["witness"]["destination_digests"] = "aabb"
            return entry

        with pytest.raises(MergeInputError) as excinfo:
            derive_merge({0: evidence(0, [forged(0)]), 1: evidence(1, [forged(1)])})
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


class TestRolesDeletionVacuousTruth:
    """b6-opus-R18-1: uniform roles deletion must refuse, never render TOP.

    Fail-before: deleting the ``roles`` record from EVERY member of a join
    vacuously satisfied the set-of-shapes agreement (all ranks presented the
    same EMPTY shape set) and the merge rendered the top verdict --
    aligned / attested_complete with zero findings. Asymmetric deletion was
    caught; uniform corruption, the merge threat model, was the escape.
    """

    def _extract(self, entry: dict) -> None:
        extract_rank_evidence(
            trace_for_boundaries([entry], seeded_ledger()),
            "roles-tamper",
        )

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            self._extract(entry)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_roles_key_deleted_refuses_typed(self):
        entry = boundary(0, 0)
        del entry["roles"]
        self._assert_refuses(entry)

    def test_tensor_kind_with_empty_roles_refuses_typed(self):
        self._assert_refuses(boundary(0, 0, roles=[]))

    def test_tensorless_kind_with_empty_roles_still_parses(self):
        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[])
        self._extract(entry)

    def test_uniform_roles_deletion_refuses_on_every_rank_core(self):
        """The headline escape: BOTH rank cores tampered identically."""

        for rank in (0, 1):
            entry = boundary(rank, 0)
            del entry["roles"]
            with pytest.raises(MergeInputError) as excinfo:
                extract_rank_evidence(
                    trace_for_boundaries([entry], seeded_ledger()),
                    f"uniform-tamper[{rank}]",
                )
            assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_engine_belt_flags_zero_role_entries_per_rank(self):
        """Direct ``derive_merge`` callers bypass evidence parse; the relation
        table still names every rank presenting zero roles for a
        tensor-carrying kind instead of agreeing on the empty shape set."""

        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, roles=[])]),
                1: evidence(1, [boundary(1, 0, roles=[])]),
            }
        )
        zero_role = [
            f
            for f in d.findings
            if f.kind == "relation_violation" and "zero tensor roles" in f.detail
        ]
        assert {f.detail.split("rank ")[1][0] for f in zero_role} == {"0", "1"}

    def test_tensorless_kinds_mirror_capture_side_tensorless_flags(self):
        """The evidence vocabulary tracks ``CollectiveSite.tensorless`` exactly."""

        from torchlens.backends.torch.collectives import COLLECTIVE_SITES
        from torchlens.merged._evidence import TENSORLESS_KINDS

        assert {site.kind for site in COLLECTIVE_SITES if site.tensorless} == TENSORLESS_KINDS


class TestSweepFieldValidation:
    """p5 sibling sweep: the reduce-op and seq cross-check fields parse typed.

    Fail-before: uniform ``reduce_op`` deletion from every rank core vacuously
    satisfied the reduce-op agreement check (same escape class as roles
    deletion), and a tampered non-integer ``c10d_group_seq`` crashed the
    engine's delta arithmetic with a raw ``TypeError`` instead of the promised
    typed refusal.
    """

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries([entry], seeded_ledger()),
                "sweep-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_reduce_op_deleted_on_reduce_kind_refuses_typed(self):
        entry = boundary(0, 0)  # all_reduce
        del entry["reduce_op"]
        self._assert_refuses(entry)

    def test_reduce_op_null_on_reduce_kind_refuses_typed(self):
        self._assert_refuses(boundary(0, 0, reduce_op=None))

    def test_reduce_op_null_on_non_reduce_kind_still_parses(self):
        extract_rank_evidence(
            trace_for_boundaries(
                [boundary(0, 0, kind="broadcast", reduce_op=None)], seeded_ledger()
            ),
            "clean-broadcast",
        )

    def test_non_integer_c10d_group_seq_refuses_typed(self):
        entry = boundary(0, 0)
        entry["c10d_group_seq"] = "5"
        self._assert_refuses(entry)

    def test_bool_c10d_group_seq_refuses_typed(self):
        entry = boundary(0, 0)
        entry["c10d_group_seq"] = True
        self._assert_refuses(entry)

    def test_reduce_op_kinds_mirror_capture_side_has_reduce_op_flags(self):
        """The evidence vocabulary tracks ``CollectiveSite.has_reduce_op`` exactly."""

        from torchlens.backends.torch.collectives import COLLECTIVE_SITES
        from torchlens.merged._evidence import REDUCE_OP_KINDS

        assert {site.kind for site in COLLECTIVE_SITES if site.has_reduce_op} == REDUCE_OP_KINDS


class TestWireVocabularyLockstep:
    """R49: the distributed->merged wire vocabulary cannot drift silently.

    The collective_boundary_v1 payload is WRITTEN by
    ``backends/torch/collectives.py`` (+ the lifecycle ledger) and READ by
    ``merged/_evidence.py``; both sides used to re-spell the closed
    vocabularies independently with zero drift gate, so a writer-side rename
    silently turned every future artifact unparseable (or, worse, unvalidated
    on the renamed axis). Declared residual: a NEW writer-side token is only
    caught at parse time; hoisting the writer's literals into one shared
    constant home is relayed to the capture lane.
    """

    def _writer_string_literals(self) -> set[str]:
        import inspect

        from torchlens.backends.torch import collectives

        tree = ast.parse(inspect.getsource(collectives))
        return {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

    def test_boundary_schema_matches_writer(self):
        from torchlens.backends.torch import collectives
        from torchlens.merged import _evidence

        assert collectives.BOUNDARY_SCHEMA == _evidence.BOUNDARY_SCHEMA

    def test_install_epoch_vocabulary_matches_ledger_literal(self):
        from typing import get_args

        from torchlens.distributed._ledger import InstallEpoch
        from torchlens.merged._evidence import _INSTALL_EPOCHS

        assert set(get_args(InstallEpoch)) == set(_INSTALL_EPOCHS)

    def test_writer_spells_every_reader_vocabulary_token(self):
        from torchlens.merged import _evidence

        writer_literals = self._writer_string_literals()
        for vocab_name in (
            "_COMPLETION_BINDINGS",
            "_WITNESS_POLICIES",
            "_DISCLOSURE_TOKENS",
            "_NOT_PRESENT_REASONS",
        ):
            vocab = getattr(_evidence, vocab_name)
            missing = set(vocab) - writer_literals
            assert not missing, (
                f"reader vocabulary {vocab_name} member(s) {sorted(missing)} never "
                "appear in the writer module -- a writer-side rename drifted the wire"
            )

    def test_reader_vocabularies_are_pinned(self):
        from torchlens.merged import _evidence

        assert set(_evidence._COMPLETION_BINDINGS) == {"issue_sync", "unobserved"}
        assert set(_evidence._WITNESS_POLICIES) == {"none", "digest"}
        assert set(_evidence._DISCLOSURE_TOKENS) == {
            "read_of_inflight_destination",
            "c10d_group_seq_read_failed",
        }
        assert set(_evidence._NOT_PRESENT_REASONS) == {"async_completion_unobserved"}


class TestReleaseContract:
    """Contract section 8: release() refuses typed, never lies about members.

    Fail-before (opus R18 [W]): release() landed contradicting the contract
    doc ("no separate cleanup surface"), post-release member access raised
    bare ``KeyError`` from the emptied handle dict, and ``merged.ranks``
    presented a released presenter as a ZERO-MEMBER merge (empty mapping,
    ``len() == 0``) while ``rank_ids`` still listed the ranks -- a silent
    presence lie.
    """

    def _merged(self):
        from torchlens.merged import merge_ranks

        return merge_ranks(
            [
                trace_for_boundaries([boundary(0, 0)], seeded_ledger()),
                trace_for_boundaries([boundary(1, 0)], seeded_ledger()),
            ]
        )

    def _assert_released_refusal(self, call) -> None:
        from torchlens.merged._errors import MergedSurfaceUnsupportedError

        with pytest.raises(MergedSurfaceUnsupportedError) as excinfo:
            call()
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_MEMBER_RELEASED.value

    def test_member_surfaces_refuse_typed_after_release(self, tmp_path):
        merged = self._merged()
        merged.release()
        self._assert_released_refusal(lambda: merged.ranks)
        self._assert_released_refusal(lambda: merged["anything"])
        self._assert_released_refusal(lambda: merged.super_op("anything"))
        self._assert_released_refusal(lambda: merged.save(tmp_path / "released"))

    def test_join_ops_refuses_typed_after_release(self):
        merged = self._merged()
        (join,) = merged.joins
        merged.release()
        self._assert_released_refusal(lambda: merged.join_ops(join))

    def test_release_never_presents_zero_members(self):
        """A released presenter must not read as an empty merge."""

        merged = self._merged()
        merged.release()
        with pytest.raises(Exception) as excinfo:
            len(merged.ranks)
        assert getattr(excinfo.value, "fields", {}).get("code") == (
            MergedErrorCode.MERGED_MEMBER_RELEASED.value
        )

    def test_verdicts_stay_readable_after_release(self):
        merged = self._merged()
        before = (merged.alignment, merged.value_status, merged.rank_ids)
        merged.release()
        assert (merged.alignment, merged.value_status, merged.rank_ids) == before
        assert merged.report.alignment is before[0]
        assert merged.joins and merged.gaps == merged._derivation.gap_findings
        assert isinstance(merged.findings, tuple)
        assert "MergedTrace" in repr(merged)
        assert "alignment" in merged.summary()

    def test_release_is_idempotent(self):
        merged = self._merged()
        merged.release()
        merged.release()
        self._assert_released_refusal(lambda: merged.ranks)

    def test_pre_release_member_access_unchanged(self):
        merged = self._merged()
        assert set(merged.ranks) == {0, 1}
        assert merged.ranks[0] is not None


class TestWitnessCompletionCoherence:
    """R18 fixwave-6: forged witness/completion/disclosure records refuse at parse.

    Fail-before (sol HIGH, 4th round): an async boundary
    (``completion_binding="unobserved"``) carrying FORGED ``destination_digests``
    -- bytes the recorder definitionally never observed -- rendered
    ``attested``/``attested_complete`` when the forgery matched across cores;
    ``policy_resolved="none"`` cores presenting digests attested the same way
    (opus+sol); and 15/17 disclosure-tamper arms (``async_op`` flips, stripped
    ``read_of_inflight_destination``, spurious tokens, ``op_node``/``peer``
    rewrites, per-boundary install-epoch promotion) passed parse untouched.
    Every axis now refuses typed at the one chokepoint merge time and load
    rederivation share.
    """

    def _extract(self, entry: dict) -> None:
        extract_rank_evidence(
            trace_for_boundaries([entry], seeded_ledger()),
            "coherence-tamper",
        )

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            self._extract(entry)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    # --- the headline forgery: async destination digests -------------------

    def test_forged_destination_digests_on_unobserved_completion_refuse(self):
        entry = boundary(
            0,
            0,
            async_op=True,
            witness_policy="digest",
            contribution_digests=["c" * 64],
            destination_digests=["a" * 64],
        )
        self._assert_refuses(entry)

    def test_honest_async_digest_record_still_parses(self):
        entry = boundary(
            0,
            0,
            async_op=True,
            witness_policy="digest",
            contribution_digests=["c" * 64],
            destination_digests=None,
        )
        self._extract(entry)  # must not raise

    # --- digests under witness policy "none" -------------------------------

    def test_contribution_digests_under_policy_none_refuse(self):
        entry = boundary(0, 0)
        entry["witness"]["contribution_digests"] = ["c" * 64]
        self._assert_refuses(entry)

    def test_destination_digests_under_policy_none_refuse(self):
        entry = boundary(0, 0)
        entry["witness"]["destination_digests"] = ["a" * 64]
        self._assert_refuses(entry)

    # --- events coherence ---------------------------------------------------

    def test_async_op_flag_contradicting_completion_binding_refuses(self):
        entry = boundary(0, 0, async_op=True)
        entry["events"]["async_op"] = False
        self._assert_refuses(entry)

    def test_sync_record_claiming_unobserved_binding_refuses(self):
        entry = boundary(0, 0)
        entry["events"]["completion_binding"] = "unobserved"
        self._assert_refuses(entry)

    def test_non_boolean_async_op_refuses(self):
        entry = boundary(0, 0)
        entry["events"]["async_op"] = "no"
        self._assert_refuses(entry)

    # --- disclosure coherence -----------------------------------------------

    def test_stripped_inflight_read_disclosure_refuses(self):
        entry = boundary(0, 0, async_op=True)
        entry["disclosures"] = []
        self._assert_refuses(entry)

    def test_spurious_inflight_read_disclosure_refuses(self):
        entry = boundary(0, 0)
        entry["disclosures"] = ["read_of_inflight_destination"]
        self._assert_refuses(entry)

    def test_unknown_disclosure_token_refuses(self):
        entry = boundary(0, 0)
        entry["disclosures"] = ["totally_fine_trust_me"]
        self._assert_refuses(entry)

    def test_group_seq_value_with_read_failed_disclosure_refuses(self):
        entry = boundary(0, 0, c10d_group_seq=7)
        entry["disclosures"] = ["c10d_group_seq_read_failed"]
        self._assert_refuses(entry)

    # --- not_present_reason coherence ----------------------------------------

    def test_not_present_reason_outside_vocabulary_refuses(self):
        entry = boundary(0, 0)
        entry["witness"]["not_present_reason"] = "because"
        self._assert_refuses(entry)

    def test_spurious_async_reason_on_sync_digest_record_refuses(self):
        entry = boundary(
            0,
            0,
            witness_policy="digest",
            contribution_digests=["c" * 64],
            destination_digests=["a" * 64],
        )
        entry["witness"]["not_present_reason"] = "async_completion_unobserved"
        self._assert_refuses(entry)

    def test_missing_async_reason_on_async_digest_record_refuses(self):
        entry = boundary(
            0,
            0,
            async_op=True,
            witness_policy="digest",
            contribution_digests=["c" * 64],
        )
        entry["witness"]["not_present_reason"] = None
        self._assert_refuses(entry)

    # --- op_node / peer / lifetime coherence ---------------------------------

    def test_op_node_false_on_tensor_kind_refuses(self):
        entry = boundary(0, 0)
        entry["op_node"] = False
        self._assert_refuses(entry)

    def test_op_node_true_on_tensorless_kind_refuses(self):
        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[])
        entry["op_node"] = True
        self._assert_refuses(entry)

    def test_peer_record_on_collective_kind_refuses(self):
        entry = boundary(0, 0)
        entry["peer"] = {"canonical": {"src": 0, "dst": 1}}
        self._assert_refuses(entry)

    def test_missing_peer_record_on_p2p_kind_refuses(self):
        entry = boundary(0, 0, kind="send", channel="p2p/0->1", reduce_op=None)
        entry["peer"] = None
        self._assert_refuses(entry)

    def test_lifetime_epoch_outside_vocabulary_refuses(self):
        entry = boundary(0, 0)
        entry["lifetime_evidence"]["install_epoch"] = "definitely_complete"
        self._assert_refuses(entry)

    def test_boundary_epoch_promotion_against_record_epoch_refuses(self):
        entry = boundary(0, 0)
        entry["lifetime_evidence"]["install_epoch"] = "armed_before_any_group"
        self._assert_refuses(entry)

    # --- tensorless empty digest lists (honest recorder shape) ---------------

    def test_tensorless_empty_digest_lists_still_parse(self):
        """The recorder emits [] digest lists for barrier under policy digest."""

        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[], witness_policy="digest")
        entry["witness"]["contribution_digests"] = []
        entry["witness"]["destination_digests"] = []
        self._extract(entry)  # must not raise

    def test_tensor_kind_empty_digest_list_still_refuses(self):
        entry = boundary(0, 0, witness_policy="digest")
        entry["witness"]["contribution_digests"] = []
        self._assert_refuses(entry)


class TestRelationsAndCrossChecks:
    def test_kind_disagreement_at_joined_key_conflicts(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, kind="all_reduce")]),
                1: evidence(1, [boundary(1, 0, kind="broadcast", reduce_op=None)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(f.kind == "relation_violation" for f in d.findings)

    def test_reduce_op_disagreement_conflicts(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, reduce_op="RedOpType.SUM")]),
                1: evidence(1, [boundary(1, 0, reduce_op="RedOpType.MAX")]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED

    def test_all_gather_wrong_destination_count_conflicts(self):
        roles = [
            {
                "role": "contribution",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
            {
                "role": "destination",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]  # group of 2 but only ONE destination
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, kind="all_gather", roles=roles, reduce_op=None)]),
                1: evidence(1, [boundary(1, 0, kind="all_gather", roles=roles, reduce_op=None)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED

    def test_backend_disagreement_conflicts_and_demotes_witness(self):
        """Deep-hunt F4: cross-rank backend disagreement is never silent.

        Fail-before: ``group_backend.setdefault`` was first-writer-wins in
        rank order -- rank 0 claiming "gloo" flipped the group into the
        witness-verdict backends and the join rendered ATTESTED under gloo
        contract semantics while rank 1 recorded an unknown backend.
        """

        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, backend="gloo", **digest_kwargs())]),
                1: evidence(1, [boundary(1, 0, backend="mystery_backend", **digest_kwargs())]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(f.kind == "relation_violation" and "backend" in f.detail for f in d.findings)
        # The disputed backend is demoted: never verdict-grade.
        assert d.joins[0].backend is None
        assert d.joins[0].consistency is BoundaryConsistency.NOT_APPLICABLE

    def test_backend_agreement_reports_no_finding(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, **digest_kwargs())]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs())]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED
        assert d.joins[0].backend == "gloo"

    def test_c10d_group_seq_delta_disagreement_conflicts(self):
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0, c10d_group_seq=10), boundary(0, 1, c10d_group_seq=11)],
                ),
                1: evidence(
                    1,
                    [boundary(1, 0, c10d_group_seq=20), boundary(1, 1, c10d_group_seq=25)],
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(f.kind == "correlation_delta_mismatch" for f in d.findings)

    def test_armed_rank_base_misalignment_is_a_correlation_conflict(self):
        """Deep-hunt F5: differing capture windows cannot fabricate a join.

        Rank 0 recorded absolute seqs {0, 1}; rank 1 recorded {1} only. Delta
        alignment paired rank 0's seq 0 with rank 1's seq 1 -- two DIFFERENT
        collectives presented as one honest correspondence (invisible under
        witness "none", and the c10d cross-check cancels constant offsets).
        Both ranks are armed before any group, so their counters tick on every
        issue and equal-seq is provable: the disagreement must conflict.
        """

        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0), boundary(0, 1)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(
                    1,
                    [boundary(1, 1)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(
            f.kind == "correlation_delta_mismatch" and "absolute issue sequences" in f.detail
            for f in d.findings
        )

    def test_armed_ranks_with_equal_absolute_seqs_stay_aligned(self):
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 3), boundary(0, 4)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(
                    1,
                    [boundary(1, 3), boundary(1, 4)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED

    def test_seeded_rank_base_offsets_never_compared(self):
        # Mixed epochs: the seeded rank's absolute base is a rank-local fact
        # (arm-time histories differ); only armed-before-any-group ranks are
        # held to equal absolute seqs, so this stays an honest delta join.
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(1, [boundary(1, 7)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED

    def test_c10d_group_seq_absent_never_demotes(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, c10d_group_seq=None)]),
                1: evidence(1, [boundary(1, 0, c10d_group_seq=7)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED

    def test_interleaved_group_orders_are_an_order_contradiction(self):
        # Two generations of the same membership, both wrapped by complete
        # witnesses (identical lineage vectors -> audit-compatible), but the
        # ranks issue them in OPPOSITE local orders: the join graph has a
        # cycle and the merge is structurally conflicted.
        def led():
            ledger = GroupLifecycleLedger()
            ledger.append(
                GroupLifecycleEvent(0, "create", WORLD, 0, "wrapped", "armed_before_any_group", 0)
            )
            ledger.append(
                GroupLifecycleEvent(1, "create", WORLD, 1, "wrapped", "armed_before_any_group", 1)
            )
            return ledger

        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0, ordinal=0), boundary(0, 0, ordinal=1)],
                    ledger=led(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(
                    1,
                    [boundary(1, 0, ordinal=1), boundary(1, 0, ordinal=0)],
                    ledger=led(),
                    epoch="armed_before_any_group",
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(f.kind == "order_contradiction" for f in d.findings)


class TestWitnessDerivation:
    def test_matching_digests_attest(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, **digest_kwargs())]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs())]),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.ATTESTED
        assert d.stored_value_status is MergeValueStatus.ATTESTED_COMPLETE

    def test_mismatch_demotes_value_status_only(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, **digest_kwargs("aa"))]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs("bb"))]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED  # never structural
        assert d.joins[0].consistency is BoundaryConsistency.MISMATCHED
        assert d.stored_value_status is MergeValueStatus.DIVERGENT
        assert any(f.kind == "value_divergence" for f in d.findings)

    def test_witness_level_none_is_not_present_and_unwitnessed(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0)]),
                1: evidence(1, [boundary(1, 0)]),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.NOT_PRESENT
        assert d.stored_value_status is MergeValueStatus.UNWITNESSED

    def test_async_unobserved_completion_is_not_present_never_mismatched(self):
        # Async all_reduce: contribution digests exist (pre-reduce bytes,
        # rank-distinct), destination digests absent. Falling back to the
        # contribution digests would FABRICATE a mismatch; the verdict must
        # be not_present.
        kwargs = {
            "witness_policy": "digest",
            "contribution_digests": None,
            "destination_digests": None,
            "async_op": True,
        }
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [
                        boundary(
                            0,
                            0,
                            contribution_digests=["r0-pre"],
                            **{k: v for k, v in kwargs.items() if k != "contribution_digests"},
                        )
                    ],
                ),
                1: evidence(
                    1,
                    [
                        boundary(
                            1,
                            0,
                            contribution_digests=["r1-pre"],
                            **{k: v for k, v in kwargs.items() if k != "contribution_digests"},
                        )
                    ],
                ),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.NOT_PRESENT
        assert d.stored_value_status is MergeValueStatus.UNWITNESSED

    def test_reduce_is_not_applicable_at_every_level(self):
        roles_root = [
            {
                "role": "contribution_destination",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]
        roles_leaf = [
            {
                "role": "contribution",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0, kind="reduce", roles=roles_root, **digest_kwargs())],
                ),
                1: evidence(
                    1,
                    [
                        boundary(
                            1,
                            0,
                            kind="reduce",
                            roles=roles_leaf,
                            witness_policy="digest",
                            contribution_digests=["x"],
                        )
                    ],
                ),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.NOT_APPLICABLE
        assert d.stored_value_status is MergeValueStatus.UNWITNESSED

    def test_unknown_backend_is_not_applicable(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, backend="fancy_tpu", **digest_kwargs())]),
                1: evidence(1, [boundary(1, 0, backend="fancy_tpu", **digest_kwargs())]),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.NOT_APPLICABLE

    def test_broadcast_root_contribution_witnesses_destinations(self):
        root_roles = [
            {
                "role": "contribution",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]
        leaf_roles = [
            {
                "role": "destination",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [
                        boundary(
                            0,
                            0,
                            kind="broadcast",
                            roles=root_roles,
                            reduce_op=None,
                            witness_policy="digest",
                            contribution_digests=["same"],
                        )
                    ],
                ),
                1: evidence(
                    1,
                    [
                        boundary(
                            1,
                            0,
                            kind="broadcast",
                            roles=leaf_roles,
                            reduce_op=None,
                            witness_policy="digest",
                            destination_digests=["same"],
                        )
                    ],
                ),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.ATTESTED

    def test_partial_attestation_is_attested_partial(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, **digest_kwargs()), boundary(0, 1)]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs()), boundary(1, 1)]),
            }
        )
        assert d.stored_value_status is MergeValueStatus.ATTESTED_PARTIAL


class TestPresenterLookupNarrowing:
    """Deep-hunt F7: ``__getitem__``'s rank scan must not swallow core defects.

    Fail-before: ``except Exception`` read a rank core whose lookup raised
    ``RuntimeError`` as a MISS, so a defective core silently vanished and
    another rank's hit presented as an unambiguous single-rank result --
    ``super_op`` directly below was already narrowed (b5 R45-2) for exactly
    this reason.
    """

    class _BrokenTrace:
        def __getitem__(self, item):
            raise RuntimeError("corrupt core: internal invariant violated")

    class _GoodTrace:
        def __getitem__(self, item):
            return f"op<{item}>"

    class _MissTrace:
        def __getitem__(self, item):
            raise KeyError(item)

    def _merged(self, trace0, trace1):
        from torchlens.merged._presenter import MergedTrace, _RankHandle

        derivation = derive_merge(
            {0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [boundary(1, 0)])}
        )
        return MergedTrace(
            derivation,
            {0: _RankHandle(0, trace=trace0), 1: _RankHandle(1, trace=trace1)},
        )

    def test_rank_core_defect_surfaces_from_getitem(self):
        merged = self._merged(self._BrokenTrace(), self._GoodTrace())
        with pytest.raises(RuntimeError, match="corrupt core"):
            merged["relu_1_2"]

    def test_lookup_miss_still_reads_as_a_miss(self):
        merged = self._merged(self._MissTrace(), self._GoodTrace())
        assert merged["relu_1_2"] == "op<relu_1_2>"


class TestExpectedRanksWidenOnly:
    def test_declared_ranks_without_cores_are_gaps(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0)]),
                1: evidence(1, [boundary(1, 0)]),
            },
            expected_ranks=[0, 1, 2, 3],
        )
        assert d.stored_alignment is MergeAlignment.PARTIAL
        (gap,) = d.gap_findings
        assert gap.ranks == (2, 3)

    def test_narrow_declaration_never_removes_gaps(self):
        # Membership records [0, 1]; declaring only [0] must not erase the
        # gap for missing rank 1 (rank cores are gap-derivation authority).
        d = derive_merge({0: evidence(0, [boundary(0, 0)])}, expected_ranks=[0])
        assert d.stored_alignment is MergeAlignment.PARTIAL
        assert d.gap_findings[0].ranks == (1,)


class TestDeterminism:
    def test_identical_evidence_yields_byte_identical_payloads(self):
        def build():
            return {
                0: evidence(0, [boundary(0, 3, **digest_kwargs()), boundary(0, 4)]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs()), boundary(1, 1)]),
            }

        left = json.dumps(derive_merge(build()).to_payload(), sort_keys=True)
        right = json.dumps(derive_merge(build()).to_payload(), sort_keys=True)
        assert left == right


class TestContractLockstep:
    """The contract document IS the spec: frozen tables track the code exactly."""

    DOC = Path(__file__).resolve().parents[1] / "docs" / "reference" / "merged_trace_contract.md"

    def test_error_code_table_matches_enum_exactly(self):
        doc = self.DOC.read_text()
        match = re.search(
            r"The exact `MergedErrorCode` values are:\n\n```text\n(.*?)```", doc, re.S
        )
        assert match is not None
        doc_codes = [line.strip() for line in match.group(1).strip().split("\n")]
        assert doc_codes == [member.value for member in MergedErrorCode]

    def test_finding_kind_table_matches_tuple_exactly(self):
        doc = self.DOC.read_text()
        match = re.search(
            r"The exact `MERGE_FINDING_KINDS` values are:\n\n```text\n(.*?)```", doc, re.S
        )
        assert match is not None
        doc_kinds = [line.strip() for line in match.group(1).strip().split("\n")]
        assert doc_kinds == list(MERGE_FINDING_KINDS)

    def test_tree_hash_framing_matches_frozen_contract(self) -> None:
        """The tree-hash prose must name the implementation's unambiguous framing."""

        doc = self.DOC.read_text()
        assert "8-byte big-endian path length" in doc
        assert re.search(r"8-byte big-endian\s+file\s+size", doc)
        assert "32 raw SHA-256 bytes" in doc
