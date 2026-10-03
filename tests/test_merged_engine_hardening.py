"""Adversarial hardening matrix for the C1 merge engine.

Forgery, tamper, and outcome-gate tests split from ``test_merged_engine.py``
(same synthetic-rank-evidence substrate, imported from there): op-label
back-reference forgery, role-index validation, group-record forgery and
rank tamper, the c10d group-seq latch, uniform role-shape rewrites, the
member-outcome gate, halted-member artifact round-trips, and the
module-level-save typed refusal.
"""

from __future__ import annotations

import pytest
from test_merged_engine import (
    boundary,
    evidence,
    seeded_ledger,
    trace_for_boundaries,
)

from torchlens.merged import (
    BoundaryConsistency,
    MergeAlignment,
    MergedErrorCode,
    MergeValueStatus,
    derive_merge,
)
from torchlens.merged._errors import MergeInputError
from torchlens.merged._evidence import extract_rank_evidence


def _hex64(char: str) -> str:
    """A syntactically valid lowercase SHA-256 hex digest for fixtures."""

    return char * 64


class TestOpLabelBackReferenceForgery:
    """R18-1(i): op-label back-references parse against the recorder's shape.

    Fail-before (b6 R18 merged-evidence forging family): ``op_labels_raw``
    was validated only as list-of-strings, so fabricated, EMPTY, or DUPLICATE
    label lists derived ``aligned``/``attested_complete`` with zero findings
    while the join table's op back-references lied. The recorder emits one
    label per logged boundary output tensor (>= 1 for every op-bearing kind,
    exactly 0 for tensorless kinds), so those shapes refuse at the one parse
    chokepoint merge time and load rederivation share.

    Declared residual (documented, not open): a FABRICATED but well-formed
    label list still parses -- the parse chokepoint never dereferences the
    member trace (synthetic evidence carriers and payload-free loads have no
    resolvable op table). It cannot improve any verdict (no verdict reads the
    labels) and every consumer that RESOLVES back-references
    (``MergedTrace.join_ops``) refuses typed on an unresolvable label; a
    forger who also reauthors the member's op table to match is the
    documented coherent-reauthoring boundary.
    """

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries([entry], seeded_ledger()),
                "op-label-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_empty_op_labels_on_op_bearing_kind_refuses(self):
        entry = boundary(0, 0)
        entry["op_labels_raw"] = []
        self._assert_refuses(entry)

    def test_duplicate_op_labels_refuse(self):
        entry = boundary(0, 0)
        entry["op_labels_raw"] = ["allreduce_1_raw", "allreduce_1_raw"]
        self._assert_refuses(entry)

    def test_labels_on_tensorless_kind_refuse(self):
        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[])
        entry["op_labels_raw"] = ["barrier_1_raw"]
        self._assert_refuses(entry)

    def test_tensorless_empty_labels_still_parse(self):
        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[])
        extract_rank_evidence(
            trace_for_boundaries([entry], seeded_ledger()),
            "honest-barrier",
        )

    @pytest.mark.smoke
    def test_fabricated_label_refuses_typed_at_join_ops_access(self):
        """The access-time half of the residual: resolution is fail-closed."""

        from torchlens.merged._presenter import MergedTrace, _RankHandle

        class _UnresolvingCore:
            _raw_to_final_op_labels: dict = {}

            def __getitem__(self, item):
                raise KeyError(item)

        derivation = derive_merge(
            {0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [boundary(1, 0)])}
        )
        merged = MergedTrace(
            derivation,
            {
                0: _RankHandle(0, trace=_UnresolvingCore()),
                1: _RankHandle(1, trace=_UnresolvingCore()),
            },
        )
        (join,) = merged.joins
        with pytest.raises(MergeInputError) as excinfo:
            merged.join_ops(join)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


class TestRoleIndexValidation:
    """R18-1(ii): ``roles[].index`` parses typed (was unvalidated and unread)."""

    def _role(self, name: str, index) -> dict:
        return {
            "role": name,
            "index": index,
            "shape": [2],
            "logical_shape": None,
            "placements": None,
        }

    def _assert_refuses(self, roles: list[dict]) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries([boundary(0, 0, roles=roles)], seeded_ledger()),
                "role-index-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_missing_index_refuses(self):
        role = self._role("contribution", 0)
        del role["index"]
        self._assert_refuses([role])

    def test_negative_index_refuses(self):
        self._assert_refuses([self._role("contribution", -1)])

    def test_boolean_index_refuses(self):
        self._assert_refuses([self._role("contribution", True)])

    def test_non_integer_index_refuses(self):
        self._assert_refuses([self._role("contribution", "0")])

    def test_duplicate_index_within_one_role_name_refuses(self):
        self._assert_refuses([self._role("contribution", 0), self._role("contribution", 0)])

    def test_positional_holes_across_role_names_still_parse(self):
        # NOT dense 0..n-1 by design: the recorder keys contribution roles by
        # INPUT position and destination roles by OUTPUT position, and a
        # tensor serving as both keeps its input position -- leaving an
        # honest hole in the destination positions.
        extract_rank_evidence(
            trace_for_boundaries(
                [
                    boundary(
                        0,
                        0,
                        roles=[
                            self._role("contribution_destination", 0),
                            self._role("destination", 1),
                        ],
                    )
                ],
                seeded_ledger(),
            ),
            "honest-role-holes",
        )


class TestGroupRecordForgery:
    """R18-2: the group record's size / group-rank redundancies are closed.

    Fail-before: ``group["size"]`` was written by the recorder but never
    validated; ``my_group_rank`` was never tied to the membership position it
    definitionally equals, so a permuted value silently rebound slice-witness
    pairings and an out-of-range value crashed the engine's group-rank list
    indexing as a raw IndexError; and selectively STRIPPING ``my_group_rank``
    from a slice-witnessed join deleted its ``value_divergence`` finding.
    """

    def _assert_refuses(self, boundaries: list[dict]) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries(boundaries, seeded_ledger()),
                "group-record-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_size_deleted_refuses(self):
        entry = boundary(0, 0)
        del entry["group"]["size"]
        self._assert_refuses([entry])

    def test_size_mismatch_refuses(self):
        entry = boundary(0, 0)
        entry["group"]["size"] = 3
        self._assert_refuses([entry])

    def test_boolean_size_refuses(self):
        entry = boundary(0, 0)
        entry["group"]["size"] = True
        self._assert_refuses([entry])

    def test_permuted_my_group_rank_refuses(self):
        entry = boundary(0, 0)  # rank 0 of members (0, 1): position is 0
        entry["group"]["my_group_rank"] = 1
        self._assert_refuses([entry])

    def test_out_of_range_my_group_rank_refuses(self):
        entry = boundary(0, 0)
        entry["group"]["my_group_rank"] = 5
        self._assert_refuses([entry])

    def test_selective_my_group_rank_strip_refuses(self):
        # The writer mints None only when dist.get_group_rank RAISES -- a
        # group-level fact -- so mixed presence within one core+group is
        # tamper. This is exactly the strip that used to delete a
        # value_divergence finding from a slice-witnessed join.
        stripped = boundary(0, 0)
        stripped["group"]["my_group_rank"] = None
        self._assert_refuses([stripped, boundary(0, 1)])

    def test_uniform_absence_still_parses(self):
        # get_group_rank genuinely raising on this rank's runtime is honest:
        # every boundary of the group records None uniformly.
        cores = []
        for seq in (0, 1):
            entry = boundary(0, seq)
            entry["group"]["my_group_rank"] = None
            cores.append(entry)
        extract_rank_evidence(
            trace_for_boundaries(cores, seeded_ledger()),
            "honest-uniform-absence",
        )

    def test_engine_belt_refuses_out_of_range_group_rank_typed(self):
        """Direct engine evidence gets a typed refusal, never an IndexError."""

        def role(name: str, index: int) -> dict:
            return {
                "role": name,
                "index": index,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            }

        root = boundary(
            0,
            0,
            kind="gather",
            reduce_op=None,
            roles=[role("contribution", 0), role("destination", 0), role("destination", 1)],
            witness_policy="digest",
            contribution_digests=[_hex64("a")],
            destination_digests=[_hex64("a"), _hex64("b")],
        )
        leaf = boundary(
            1,
            0,
            kind="gather",
            reduce_op=None,
            roles=[role("contribution", 0)],
            witness_policy="digest",
            contribution_digests=[_hex64("b")],
        )
        leaf["group"]["my_group_rank"] = 7  # bypasses parse: direct evidence
        with pytest.raises(MergeInputError) as excinfo:
            derive_merge({0: evidence(0, [root]), 1: evidence(1, [leaf])})
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


class TestGatherDivergenceSurvivesGroupRankTamper:
    """Findings-never-deleted for the verified my_group_rank strip escape."""

    def _role(self, name: str, index: int) -> dict:
        return {
            "role": name,
            "index": index,
            "shape": [2],
            "logical_shape": None,
            "placements": None,
        }

    def _root(self) -> dict:
        return boundary(
            0,
            0,
            kind="gather",
            reduce_op=None,
            roles=[
                self._role("contribution", 0),
                self._role("destination", 0),
                self._role("destination", 1),
            ],
            witness_policy="digest",
            contribution_digests=[_hex64("a")],
            # Slice 1 disagrees with the leaf's contribution: value_divergence.
            destination_digests=[_hex64("a"), _hex64("f")],
        )

    def _leaf(self) -> dict:
        return boundary(
            1,
            0,
            kind="gather",
            reduce_op=None,
            roles=[self._role("contribution", 0)],
            witness_policy="digest",
            contribution_digests=[_hex64("b")],
        )

    def _extract(self, rank: int, boundaries: list[dict]):
        return extract_rank_evidence(
            trace_for_boundaries(boundaries, seeded_ledger()),
            f"gather-tamper[{rank}]",
        )

    def test_honest_mismatch_yields_value_divergence(self):
        d = derive_merge({0: self._extract(0, [self._root()]), 1: self._extract(1, [self._leaf()])})
        assert d.joins[0].consistency is BoundaryConsistency.MISMATCHED
        assert any(f.kind == "value_divergence" for f in d.findings)

    def test_selective_strip_refuses_at_parse(self):
        # A leaf core holding another boundary of the SAME group with a
        # recorded group rank: stripping only the gather join's my_group_rank
        # is mixed presence and refuses -- the finding can no longer be
        # deleted by the one-field strip that used to work.
        stripped = self._leaf()
        stripped["group"]["my_group_rank"] = None
        with pytest.raises(MergeInputError) as excinfo:
            self._extract(1, [stripped, boundary(1, 1)])
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_group_wide_strip_retreats_to_no_claim_never_attested(self):
        # Stripping the WHOLE group's my_group_rank on a single-boundary core
        # is byte-identical to an honest get_group_rank failure, so it parses
        # (uniform absence). The witness derivation then retreats to
        # NOT_PRESENT/unwitnessed -- a retreat to NO claim, consistent with
        # the demote-only witness model (identical to stripping the digests
        # themselves) -- and can never IMPROVE to any attested flavor.
        stripped = self._leaf()
        stripped["group"]["my_group_rank"] = None
        d = derive_merge({0: self._extract(0, [self._root()]), 1: self._extract(1, [stripped])})
        assert d.joins[0].consistency is BoundaryConsistency.NOT_PRESENT
        assert d.stored_value_status is MergeValueStatus.UNWITNESSED


class TestC10dGroupSeqLatch:
    """R18-3: per-core c10d_group_seq presence follows the recorder's latch.

    Fail-before: only value+disclosure COEXISTENCE was refused, so nulling a
    single recorded value (or deleting the key) silently removed a
    ``correlation_delta_mismatch`` finding -- the only in-band detector in
    the base-misalignment neighborhood. Legal per-core shapes are exactly the
    recorder's: all values; all null (capability absent); or a prefix of
    values, ONE boundary disclosing ``c10d_group_seq_read_failed``, then an
    all-null suffix.
    """

    def _assert_refuses(self, boundaries: list[dict]) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries(boundaries, seeded_ledger()),
                "seq-latch-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_deleted_key_refuses(self):
        entry = boundary(0, 0)
        del entry["c10d_group_seq"]
        self._assert_refuses([entry])

    def test_undisclosed_drop_after_values_refuses(self):
        self._assert_refuses(
            [boundary(0, 0, c10d_group_seq=10), boundary(0, 1, c10d_group_seq=None)]
        )

    def test_value_after_null_refuses(self):
        self._assert_refuses(
            [boundary(0, 0, c10d_group_seq=None), boundary(0, 1, c10d_group_seq=11)]
        )

    def test_value_after_disclosed_drop_refuses(self):
        dropped = boundary(0, 1, c10d_group_seq=None)
        dropped["disclosures"] = ["c10d_group_seq_read_failed"]
        self._assert_refuses(
            [boundary(0, 0, c10d_group_seq=10), dropped, boundary(0, 2, c10d_group_seq=12)]
        )

    def test_repeated_disclosure_after_latch_refuses(self):
        first = boundary(0, 0, c10d_group_seq=None)
        first["disclosures"] = ["c10d_group_seq_read_failed"]
        second = boundary(0, 1, c10d_group_seq=None)
        second["disclosures"] = ["c10d_group_seq_read_failed"]
        self._assert_refuses([first, second])

    def test_honest_latch_shape_parses(self):
        dropped = boundary(0, 1, c10d_group_seq=None)
        dropped["disclosures"] = ["c10d_group_seq_read_failed"]
        extract_rank_evidence(
            trace_for_boundaries(
                [
                    boundary(0, 0, c10d_group_seq=10),
                    dropped,
                    boundary(0, 2, c10d_group_seq=None),
                ],
                seeded_ledger(),
            ),
            "honest-latch",
        )

    def test_selective_null_cannot_delete_the_delta_mismatch_finding(self):
        # The verified escape, cheap-tamper arm: the delta-disagreement
        # fixture (10,11 vs 20,25) conflicts; nulling ONE contradicting value
        # now refuses at parse instead of silently un-finding the conflict.
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries(
                    [
                        boundary(1, 0, c10d_group_seq=20),
                        boundary(1, 1, c10d_group_seq=None),
                    ],
                    seeded_ledger(),
                ),
                "selective-null",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    @pytest.mark.smoke
    def test_all_null_rewrite_is_the_documented_reauthoring_residual(self):
        # Nulling EVERY value on EVERY core is byte-identical to an honest
        # capability-absent capture (the probe returns null with no
        # disclosure from the first boundary on), so it parses and the
        # cross-check honestly reports nothing: "absence of the probe never
        # demotes anything" (contract section 4), and
        # test_c10d_group_seq_absent_never_demotes pins the honest side.
        # This is the coherent-reauthoring boundary -- a weaker program's
        # honest capture -- documented here as a scope statement, not an
        # open residual; the latch rules above force a forger all the way to
        # it instead of the one-field null that used to suffice.
        def cores(seqs0, seqs1):
            return {
                0: extract_rank_evidence(
                    trace_for_boundaries(
                        [boundary(0, i, c10d_group_seq=s) for i, s in enumerate(seqs0)],
                        seeded_ledger(),
                    ),
                    "all-null[0]",
                ),
                1: extract_rank_evidence(
                    trace_for_boundaries(
                        [boundary(1, i, c10d_group_seq=s) for i, s in enumerate(seqs1)],
                        seeded_ledger(),
                    ),
                    "all-null[1]",
                ),
            }

        honest = derive_merge(cores([10, 11], [20, 25]))
        assert honest.stored_alignment is MergeAlignment.CONFLICTED
        rewritten = derive_merge(cores([None, None], [None, None]))
        assert rewritten.stored_alignment is MergeAlignment.ALIGNED
        assert not any(f.kind == "correlation_delta_mismatch" for f in rewritten.findings)


class TestUniformRoleShapeRewrite:
    """R18-1(iii) fallback: asymmetric shape tamper is caught for every
    symmetric kind; the uniform rewrite is the documented residual.

    The full per-core cross-check (role shapes against the member's own op
    records) is not implementable at the parse chokepoint: parse never
    dereferences the member trace (synthetic evidence carriers, payload-free
    and analysis-only loads have no uniformly readable op table). The engine
    belt below extends shape agreement to the remaining symmetric kinds as
    distinct-shape-SET agreement (per-rank role COUNTS legitimately differ:
    root list vs leaf tensor), so an asymmetric rewrite conflicts; a rewrite
    applied identically on EVERY core remains coherent reauthoring -- the
    documented out-of-scope boundary.
    """

    def _all_gather(self, rank: int, shape: list[int]) -> dict:
        def role(name: str, index: int) -> dict:
            return {
                "role": name,
                "index": index,
                "shape": list(shape),
                "logical_shape": None,
                "placements": None,
            }

        return boundary(
            rank,
            0,
            kind="all_gather",
            reduce_op=None,
            roles=[role("contribution", 0), role("destination", 0), role("destination", 1)],
        )

    @pytest.mark.smoke
    def test_asymmetric_shape_rewrite_conflicts(self):
        d = derive_merge(
            {
                0: evidence(0, [self._all_gather(0, [2])]),
                1: evidence(1, [self._all_gather(1, [9, 9])]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(
            f.kind == "relation_violation" and "distinct role shapes" in f.detail
            for f in d.findings
        )

    def test_uniform_shape_rewrite_is_the_documented_residual(self):
        d = derive_merge(
            {
                0: evidence(0, [self._all_gather(0, [9, 9])]),
                1: evidence(1, [self._all_gather(1, [9, 9])]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED


class TestMemberOutcomeGate:
    """R06c: member capture outcomes gate the merge at input resolution.

    Fail-before: ``merge_ranks`` never consulted member outcomes, so a HALTED
    (or synthetic FAILED) member core merged into aligned/attested_complete
    with zero findings and no disclosure anywhere on the merged surface.
    """

    def _core(self, rank: int, status=None, boundaries=None):
        from torchlens.capture.outcome import CaptureOutcome

        trace = trace_for_boundaries(
            boundaries if boundaries is not None else [boundary(rank, 0)],
            seeded_ledger(),
        )
        if status is not None:
            trace._capture_outcome = CaptureOutcome(status=status)
        return trace

    def _assert_refuses(self, cores) -> None:
        from torchlens.merged import merge_ranks

        with pytest.raises(MergeInputError) as excinfo:
            merge_ranks(cores)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGE_INPUT_INVALID.value
        assert excinfo.value.fields["reason"] == "member_outcome_not_mergeable"

    def test_failed_member_refuses_typed(self):
        from torchlens.capture.outcome import CaptureStatus

        self._assert_refuses([self._core(0, CaptureStatus.FAILED), self._core(1)])

    def test_aborted_nonfinite_member_refuses_typed(self):
        from torchlens.capture.outcome import CaptureStatus

        self._assert_refuses([self._core(0), self._core(1, CaptureStatus.ABORTED_NONFINITE)])

    def test_unknown_member_refuses_typed(self):
        from torchlens.capture.outcome import CaptureStatus

        self._assert_refuses([self._core(0, CaptureStatus.UNKNOWN), self._core(1)])

    def test_merge_report_shares_the_gate(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_report

        with pytest.raises(MergeInputError) as excinfo:
            merge_report([self._core(0, CaptureStatus.FAILED), self._core(1)])
        assert excinfo.value.fields["reason"] == "member_outcome_not_mergeable"

    def _attested_pair(self, status0, status1):
        def core(rank: int, status):
            return self._core(
                rank,
                status,
                [
                    boundary(
                        rank,
                        0,
                        witness_policy="digest",
                        contribution_digests=[_hex64("c")],
                        destination_digests=[_hex64("a")],
                    )
                ],
            )

        return [core(0, status0), core(1, status1)]

    def test_halted_member_merges_with_disclosure_never_silent_attested_complete(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        merged = merge_ranks(self._attested_pair(CaptureStatus.HALTED, CaptureStatus.COMPLETE))
        assert merged.value_status is MergeValueStatus.ATTESTED_COMPLETE
        assert merged.member_outcomes == {0: "halted", 1: "complete"}
        summary = merged.summary()
        assert "halted" in summary and "attested_complete" in summary
        # The disclosure precedes the witness-coverage claim.
        assert summary.index("halted") < summary.index("attested_complete")

    def test_unattested_member_merges_with_disclosure(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        merged = merge_ranks(self._attested_pair(CaptureStatus.COMPLETE, CaptureStatus.UNATTESTED))
        assert merged.member_outcomes[1] == "unattested"
        assert "unattested" in merged.summary()

    def test_disclosure_survives_release(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        merged = merge_ranks(self._attested_pair(CaptureStatus.HALTED, CaptureStatus.COMPLETE))
        merged.release()
        assert merged.member_outcomes == {0: "halted", 1: "complete"}
        assert "halted" in merged.summary()

    def test_members_without_outcome_sidecar_make_no_claim(self):
        from torchlens.merged import merge_ranks

        merged = merge_ranks([self._core(0), self._core(1)])
        assert merged.member_outcomes == {}
        assert "member capture outcomes" not in merged.summary()

    def test_complete_members_carry_no_disclosure_line(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        merged = merge_ranks(self._attested_pair(CaptureStatus.COMPLETE, CaptureStatus.COMPLETE))
        assert merged.member_outcomes == {0: "complete", 1: "complete"}
        assert "member capture outcomes" not in merged.summary()

    def test_path_supplied_member_gets_the_same_gate(self, monkeypatch, tmp_path):
        """Path inputs load through resolve_rank_inputs' one chokepoint."""

        import torchlens._io.bundle as bundle_module
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        failed = self._core(0, CaptureStatus.FAILED)
        monkeypatch.setattr(bundle_module, "load", lambda path: failed)
        fake = tmp_path / "rank0.tlspec"
        with pytest.raises(MergeInputError) as excinfo:
            merge_ranks([fake])
        assert excinfo.value.fields["reason"] == "member_outcome_not_mergeable"
        assert excinfo.value.fields["source"] == str(fake)

    @pytest.mark.smoke
    def test_path_supplied_halted_member_is_disclosed(self, monkeypatch, tmp_path):
        import torchlens._io.bundle as bundle_module
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        cores = {
            str(tmp_path / "rank0.tlspec"): self._attested_pair(
                CaptureStatus.HALTED, CaptureStatus.COMPLETE
            )[0]
        }
        monkeypatch.setattr(bundle_module, "load", lambda path: cores[str(path)])
        merged = merge_ranks(
            [tmp_path / "rank0.tlspec", self._attested_pair(None, CaptureStatus.COMPLETE)[1]]
        )
        assert merged.member_outcomes[0] == "halted"
        assert "halted" in merged.summary()


class TestHaltedMemberArtifactRoundTrip:
    """A merged artifact over a genuinely-persisted HALTED member core loads
    clean and keeps the disclosure -- the same derivation reruns verbatim at
    load, so the outcome gate/disclosure never perturbs descriptor equality."""

    def _real_core(self, rank: int, halted: bool):
        import torch
        from torch import nn

        import torchlens as tl

        torch.manual_seed(0)
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        if halted:
            from torchlens.capture.outcome import CaptureOutcome, CaptureStatus

            # Persist a coherent HALTED attestation: the structural halted
            # marker plus the settled stamp, exactly what a halt= capture
            # leaves behind (analysis-level member saves accept halted cores).
            log.halted = True
            log._capture_outcome = CaptureOutcome(
                status=CaptureStatus.HALTED, reason="synthetic halt for merge disclosure"
            )
        entry = boundary(
            rank,
            0,
            witness_policy="digest",
            contribution_digests=[_hex64("c")],
            destination_digests=[_hex64("a")],
        )
        entry["op_labels_raw"] = ["allreduce_1_raw"]
        log.annotations["distributed"] = {
            "boundaries": [entry],
            "group_lifecycle_ledger": seeded_ledger().to_payload(),
            "install_epoch": "seeded",
        }
        return log

    def test_halted_member_round_trip_keeps_disclosure(self, tmp_path):
        from torchlens.merged import merge_ranks
        from torchlens.merged._artifact import load_merged

        merged = merge_ranks([self._real_core(0, halted=True), self._real_core(1, halted=False)])
        assert merged.member_outcomes == {0: "halted", 1: "complete"}
        art = tmp_path / "halted_member.tlspec"
        merged.save(art)
        loaded = load_merged(art)
        assert loaded.load_degradations == ()
        assert loaded.member_outcomes == {0: "halted", 1: "complete"}
        assert "halted" in loaded.summary()
        assert loaded.value_status is merged.value_status


class TestModuleLevelSaveRefusesTyped:
    """tl.save(merged, ...) is a contract-promised typed refusal, not an AttributeError."""

    def test_tl_save_merged_trace_refuses_typed(self, tmp_path):
        # The module-level bundle save used to reach the runnable poison gate
        # and die as a bare AttributeError ('MergedTrace' has no '_runnable');
        # the contract (2.5) promises every merged export surface refuses typed.
        import torchlens as tl
        from torchlens.merged._errors import MergedSurfaceUnsupportedError
        from torchlens.merged._presenter import MergedTrace, _RankHandle

        derivation = derive_merge(
            {0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [boundary(1, 0)])}
        )
        merged = MergedTrace(
            derivation, {0: _RankHandle(0, trace=None), 1: _RankHandle(1, trace=None)}
        )
        with pytest.raises(MergedSurfaceUnsupportedError) as excinfo:
            tl.save(merged, tmp_path / "merged_refused.tlspec")
        assert excinfo.value.fields["code"] == "merged_surface_unsupported"
        assert not (tmp_path / "merged_refused.tlspec").exists()
