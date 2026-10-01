"""One-backward reads: the by= door and composition rows (F04 item 5, D9).

"Select the 50 most important sites" is one line: the three rank producers
accept a single-target ReadTable, rank at the table's declared grain, and
carry full Selection provenance. The closed loop (read -> select ->
intervene -> re-read) and the algebra-composition rows from M(reads)
section 6 ride here too.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.attribution import onebackward as ob
from torchlens.selection import SelectionError

pytestmark = pytest.mark.smoke


def _trace(**kwargs) -> tl.Trace:
    """Trace the toy MLP."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    return tl.trace(model, torch.randn(2, 4), **kwargs)


def _site_table(trace: tl.Trace) -> ob.ReadTable:
    """A site-grain magnitude table over the implicit population."""

    return ob.read(
        trace,
        target=ob.seed("output_1", index=(0, 0)),
        method="grad",
        reduce="sum_of_abs",
    )


class TestSiteGrainRanking:
    """Site-grain: k counts SITES; a win selects the whole existing mask."""

    def test_top_k_selects_highest_scored_sites(self) -> None:
        trace = _trace()
        table = _site_table(trace)
        resolved = tl.top_k(k=2, by=table).resolve(trace)
        selected = {entry.site_key for entry in resolved if entry.selected_count}
        # Contract tie-break: stable sort by score, ties resolve by CANONICAL
        # SITE ORDER (alias twins share a score; the earlier site wins).
        rows = list(table.rows())
        order = sorted(range(len(rows)), key=lambda i: -rows[i].score)
        expected = {rows[i].address for i in order[:2]}
        assert selected == expected
        for entry in resolved:
            assert entry.selected_count == math.prod(entry.shape), (
                "a site-grain win must select the complete site mask"
            )

    def test_top_fraction_and_largest_false(self) -> None:
        trace = _trace()
        table = _site_table(trace)
        half = tl.top_fraction(fraction=0.5, by=table).resolve(trace)
        assert len([entry for entry in half if entry.selected_count]) == 2
        bottom = tl.top_k(k=1, by=table, largest=False).resolve(trace)
        lowest = min((row.score, row.address) for row in table.rows())
        (winner,) = [entry for entry in bottom if entry.selected_count]
        assert winner.site_key == lowest[1]

    def test_threshold_site_grain(self) -> None:
        trace = _trace()
        table = _site_table(trace)
        cut = sorted(row.score for row in table.rows())[1]
        resolved = tl.threshold(above=cut, by=table).resolve(trace)
        assert {entry.site_key for entry in resolved} == {
            row.address for row in table.rows() if row.score > cut
        }

    def test_k_over_population_refuses(self) -> None:
        trace = _trace()
        table = _site_table(trace)
        with pytest.raises(SelectionError) as excinfo:
            tl.top_k(k=99, by=table).resolve(trace)
        assert excinfo.value.fields["reason"] == "population_too_small"

    def test_provenance_carries_metric_and_counts(self) -> None:
        trace = _trace()
        table = _site_table(trace)
        resolved = tl.top_k(k=1, by=table).resolve(trace)
        source = next(iter(resolved)).provenance.source
        for fragment in ("read_table_v1", "method='grad'", "reduction='sum_of_abs'", "frozen="):
            assert fragment in source, source


class TestElementGrainRanking:
    """Element-grain: dense values rank globally; k counts elements."""

    def test_top_k_elements(self) -> None:
        trace = _trace()
        table = ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce=None)
        resolved = tl.top_k(k=3, by=table).resolve(trace)
        assert sum(entry.selected_count for entry in resolved) == 3

    def test_element_threshold(self) -> None:
        trace = _trace()
        table = ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce=None)
        resolved = tl.threshold(above=0.0, by=table).resolve(trace)
        expected = sum(int((row.value > 0).sum()) for row in table.rows())
        assert sum(entry.selected_count for entry in resolved) == expected


class TestByDoorContract:
    """D9 at the door: foreign/stale/multi-target/coverage refusals."""

    def test_foreign_table_refuses(self) -> None:
        trace_a, trace_b = _trace(), _trace()
        table = _site_table(trace_a)
        with pytest.raises(SelectionError) as excinfo:
            tl.top_k(k=1, by=table).resolve(trace_b)
        assert excinfo.value.fields["code"] == "by_score_invalid"
        assert excinfo.value.fields["reason"] == "by_table_foreign"

    def test_stale_table_refuses_after_cleanup(self) -> None:
        trace = _trace()
        table = _site_table(trace)
        selection = tl.top_k(k=1, by=table)
        trace.cleanup()
        with pytest.raises(SelectionError) as excinfo:
            selection.resolve(trace)
        assert excinfo.value.fields["reason"] in ("by_table_foreign", "by_table_stale")

    def test_multi_target_requires_explicit_fold(self) -> None:
        trace = _trace()
        table = ob.read(
            trace,
            target=[
                ob.seed("output_1", index=(0, 0)),
                ob.seed("output_1", index=(0, 1)),
            ],
            method="grad",
            reduce="sum_of_abs",
        )
        with pytest.raises(SelectionError) as excinfo:
            tl.top_k(k=1, by=table).resolve(trace)
        assert excinfo.value.fields["reason"] == "by_table_multi_target"
        narrowed = tl.top_k(k=1, by=table.for_target("t0")).resolve(trace)
        assert any(entry.selected_count for entry in narrowed)
        folded_table = table.aggregate_targets(lambda scores: max(scores), name="max")
        folded = tl.top_k(k=1, by=folded_table).resolve(trace)
        assert any(entry.selected_count for entry in folded)

    def test_explicit_population_not_covered(self) -> None:
        trace = _trace()
        table = ob.read(
            trace,
            target=ob.seed("gelu_1_2", index=(0, 0)),
            method="grad",
            reduce="sum",
            within=tl.units("linear_1_1", [(0, 0)]) | tl.units("linear_2_3", [(0, 0)]),
        )
        # linear_2_3 is not upstream of the target: its row is 'unreachable',
        # so an explicit by-population naming it refuses.
        with pytest.raises(SelectionError) as excinfo:
            tl.top_k(k=1, by=table, within="linear_2_3").resolve(trace)
        assert excinfo.value.fields["code"] == "population_not_covered"

    def test_string_by_still_works(self) -> None:
        """The existing by='value'/'abs' door is untouched."""

        trace = _trace()
        resolved = tl.top_k(k=2, by="abs").resolve(trace)
        assert sum(entry.selected_count for entry in resolved) == 2
        with pytest.raises(ValueError):
            tl.top_k(k=1, by="magnitude")


class TestCompositionRows:
    """M(reads) section 6 composition rows at toy scale."""

    def test_table_selection_composes_with_algebra(self) -> None:
        trace = _trace()
        table = _site_table(trace)
        combined = tl.top_k(k=2, by=table) | tl.units("linear_1_1", [(0, 0)])
        resolved = combined.resolve(trace)
        assert len(list(resolved)) >= 2
        narrowed = tl.top_k(k=2, by=table) - tl.units("gelu_1_2", [(0, 0)])
        assert narrowed.resolve(trace) is not None

    def test_closed_loop_read_select_intervene_reread(self) -> None:
        """read -> top-k -> zero_ablate -> re-score: the R16 flagship loop."""

        trace = _trace(capture=tl.options.CaptureOptions(intervention_ready=True))
        table = _site_table(trace)
        # The table binds to the BASE trace, so the circuit resolves there;
        # align_to is the one explicit door onto the fork (L6 stage 4a).
        circuit = tl.top_k(k=1, by=table).resolve(trace)
        fork = trace.fork()
        fork.do(circuit.align_to(fork), tl.zero_ablate())
        rescored = ob.read(
            fork,
            target=ob.seed("output_1", index=(0, 0)),
            method="grad",
            reduce="sum_of_abs",
        )
        assert len(rescored) > 0
        assert rescored.trace is fork

    def test_within_and_frozen_on_one_call(self) -> None:
        """Composition row: within= x frozen= on one read."""

        trace = _trace()
        table = ob.read(
            trace,
            target=ob.seed("output_1", index=(0, 0)),
            method="grad",
            reduce="sum",
            within="linear_1_1",
            frozen="gelu_1_2",
        )
        (row,) = table.rows()
        assert row.address == ("linear_1_1", 1)
        assert table.provenance.frozen_policy == "explicit"

    def test_multipass_addresses_rank_distinctly(self) -> None:
        """Recurrence row: pass-qualified addresses are distinct table rows."""

        torch.manual_seed(0)

        class Reuser(nn.Module):
            """Applies one linear twice (multi-pass layer)."""

            def __init__(self) -> None:
                super().__init__()
                self.shared = nn.Linear(4, 4)

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.shared(torch.tanh(self.shared(value)))

        trace = tl.trace(Reuser(), torch.randn(2, 4))
        index = ob.read_edge_index(trace)
        multi_pass = [edge.label for edge in index.edges.values() if edge.label.endswith(":2")]
        assert multi_pass, "the reused linear must produce a second pass"
        table = ob.read(
            trace,
            target=ob.seed("output_1", index=(0, 0)),
            method="grad",
            reduce="sum_of_abs",
        )
        addresses = {row.address for row in table.rows()}
        passes = {address[1] for address in addresses}
        assert {1, 2} <= passes, "pass-qualified addresses must stay distinct rows"
